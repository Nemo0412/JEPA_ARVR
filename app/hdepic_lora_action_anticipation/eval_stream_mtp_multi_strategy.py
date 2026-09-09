#!/usr/bin/env python3
"""[ATTN-CORNER-SINK] B18 · Multi-strategy streaming-MTP eval that runs the (frozen, dominant)
ENCODER exactly ONCE per batch and evaluates every prune strategy on the shared encoder output.

The encoder is frozen + deterministic, and every KV-cache prune strategy operates on the same
``x_full`` -> sharing it is bit-for-bit identical to the per-strategy jobs, at ~3-5x less compute
(the encoder is the O(N^2) 64-slot bottleneck; pruning is post-encoder). No disk cache.

Per batch: encode once (optionally with the ``TokenPruner`` last-block patch active so the
``attention`` strategy's importance is captured in that same forward), then for each strategy
derive the kept indices, run the predictor pass-2 on the (pruned) context, classify, and
accumulate native Action Top-5 @2/4/6 s per strategy. One JSON per strategy (same schema as
``eval_stream_mtp_kvcache_prune``).
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (
    FpsSubsampledStreamMTPDataset, enlarge_predictor_budget, RecentTokenPruner,
    OfflineCalibPruner, PredictorScorePrunedModel, WindowOffsetPruner, _LimitedLoader,
)


def result_provenance(args):
    """Capture result identity at startup, before a long evaluation can outlive code edits."""
    prefixes = {"EGTEA_": "egtea", "HD_EPIC_": "hdepic"}
    inferred = {label for prefix, label in prefixes.items()
                if args.train_csv.name.startswith(prefix) and args.val_csv.name.startswith(prefix)}
    dataset = args.dataset or (next(iter(inferred)) if inferred else None)
    if dataset is None:
        raise SystemExit("Cannot identify dataset from both CSV names; specify --dataset")
    if not dataset or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in dataset):
        raise SystemExit("--dataset must be a lowercase filename-safe identifier")
    if inferred and dataset not in inferred:
        raise SystemExit("--dataset conflicts with the train/val CSV dataset prefixes")
    stats_path = args.val_csv.parent / "split_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    split = args.split or stats.get("protocol", "unspecified (see val_csv)")
    code_root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    source_paths = [Path(__file__).resolve(), Path(T.__file__).resolve(),
                    Path(__file__).with_name("eval_stream_mtp_kvcache_prune.py"),
                    Path(__file__).with_name("analyze_encoder_head_attn_corners.py")]
    provenance = {
        "dataset": dataset, "split": split, "metric_scope": "native",
        "eval_path": f"frozen {dataset} stream-MTP val; split={split}; "
                     "shared-encoder multi-strategy; native Action Top-5; "
                     "valid labels in training vocabulary; horizons=" + args.horizons_sec,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "run_tag": args.tag,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code": {"root": str(code_root), "git_commit": revision,
                 "source_sha256": {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in source_paths}},
    }
    return provenance


def predict_from_encoded(core, x_full, anticipation_times):
    """Predictor pass on an already-encoded (and possibly pruned) context -> x_accumulate.
    Mirrors ``PrunedAnticipativeModel.forward`` from just after the encoder/prune step."""
    B, N, D_full = x_full.size()
    embed_dim = core.encoder.embed_dim
    use_hier = D_full > embed_dim
    x = x_full[:, :, -embed_dim:] if use_hier else x_full
    x_acc = x.clone()
    ctxt_positions = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
    anticipation_steps = (anticipation_times * core.frames_per_second / core.tubelet_size).to(torch.int64)
    skip_positions = N + int(core.grid_size ** 2) * anticipation_steps
    N_pred = int(core.grid_size ** 2 * (core.num_output_frames // core.tubelet_size))
    tgt_positions = torch.arange(N_pred, device=x.device).unsqueeze(0).repeat(B, 1) + skip_positions.unsqueeze(1)
    x_pred_input = x_full
    for _ in range(core.num_steps):
        pred_out = core.predictor(x_pred_input, masks_x=ctxt_positions, masks_y=tgt_positions)
        x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        x_pred = x_pred_full[:, :, -embed_dim:] if x_pred_full.size(-1) != embed_dim else x_pred_full
        x_acc = torch.cat([x_acc, x_pred], dim=1)
        x_pred_for_input = x_pred_full if x_pred_full.size(-1) == x_pred_input.size(-1) else x_pred
        x_pred_input = torch.cat([x_pred_input[:, N_pred:, :], x_pred_for_input], dim=1)
    return x_acc


class StrategySelector:
    """Given the shared full encoder output, return the kept-context tokens for one strategy."""

    def __init__(self, name, base, keep_count, gp, *, calib_path=None, score_block=0,
                 token_pruner=None):
        self.name = name
        self.base = base
        self.gp = gp
        self.keep_count = keep_count
        self.token_pruner = token_pruner        # shared TokenPruner (attention) -- importance set per batch
        if name in ("pred_offline_high", "pred_offline_low"):
            self.pruner = OfflineCalibPruner(calib_path, keep_count, gp,
                                             "high" if name.endswith("high") else "low")
        elif name.startswith("window@"):
            self.pruner = WindowOffsetPruner(int(name.split("@")[1]), keep_count, gp)
        elif name == "recent":
            self.pruner = RecentTokenPruner(keep_count, gp)
        elif name in ("pred_attention_high", "pred_attention_low"):
            self.scorer = PredictorScorePrunedModel(base, keep_count, gp,
                                                    mode="high" if name.endswith("high") else "low",
                                                    score_block=score_block)
        elif name not in ("none", "attention"):
            raise SystemExit(f"unknown strategy {name!r}")
        self.has_pruner = hasattr(self, "pruner")

    def gather(self, x_full, anticipation_times):
        if self.name == "none":
            return x_full
        if self.name == "attention":
            feats, _ = self.token_pruner.prune(x_full)   # uses importance captured this batch
            return feats
        if self.has_pruner:   # recent, pred_offline_*, window@*
            feats, _ = self.pruner.prune(x_full)
            return feats
        # online predictor-score
        idx = self.scorer._score_select(self.base, x_full, anticipation_times)
        return x_full.gather(1, idx.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--init-from-ckpt", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--strategies", type=str, required=True, help="comma/space list")
    ap.add_argument("--pred-calib-path", type=str, default=None)
    ap.add_argument("--pred-score-block", type=int, default=0)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--primary-horizon-sec", type=float, default=2.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--only-context-sec", type=float, default=16.0)
    ap.add_argument("--max-val-batches", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="multi")
    ap.add_argument("--dataset", default=None, help="result label; inferred for EGTEA/HD_EPIC CSV pairs")
    ap.add_argument("--split", default=None, help="result label; defaults to split_stats.json protocol")
    args = ap.parse_args()
    provenance = result_provenance(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategies = [s for s in args.strategies.replace(",", " ").split() if s]
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    primary_h = float(args.primary_horizon_sec)
    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)

    val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size,
                                           src_fps=args.src_fps, fps=args.fps)
    if args.only_context_sec > 0:
        want = float(args.only_context_sec)
        val_ds.rows = [r for r in val_ds.rows if abs(float(r["context_sec"]) - want) < 1e-6]
        print(f"[filter] context_sec={want}: {len(val_ds.rows)} rows", flush=True)
    sampler = T.ContextBucketBatchSampler(val_ds, args.batch_size, shuffle=False, seed=0)
    lk = dict(num_workers=args.num_workers, collate_fn=T.collate_stream, pin_memory=False)
    if args.num_workers > 0:
        lk["prefetch_factor"] = 2
    loader = DataLoader(val_ds, batch_sampler=sampler, **lk)
    if args.max_val_batches > 0:
        loader = _LimitedLoader(loader, args.max_val_batches)

    base = T.build_model(device, args.max_frames, args.fps, args.img_size, str(args.checkpoint))
    for p in base.encoder.parameters():
        p.requires_grad = False
    T.load_lora_sidecars(base, str(args.encoder_lora) if args.encoder_lora else None,
                         str(args.predictor_lora) if args.predictor_lora else None)
    gp = int(base.grid_size ** 2)
    enlarge_predictor_budget(base, (args.max_frames // int(base.tubelet_size)) * gp, gp)

    classifier = T.AttentiveClassifier(verb_classes=verb_map, noun_classes=noun_map,
                                       action_classes=action_map, embed_dim=int(base.encoder.embed_dim),
                                       num_heads=16, depth=4, use_activation_checkpointing=True).to(device)
    mtp_clf = T.CommunicatingMLPMTPClassifier(classifier, horizons_sec=horizons, comm_layers=2,
                                              comm_heads=4).to(device)
    ck = torch.load(args.init_from_ckpt, map_location="cpu", weights_only=False)
    # model weights live under encoder/predictor of base wrapped by PrunedAnticipativeModel("model")
    wrap = T.PrunedAnticipativeModel(base, None, prune_threshold=10 ** 9)
    wrap.load_state_dict(ck["model"], strict=False)
    mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
    print(f"[load] parent best={ck.get('best')}", flush=True)
    del ck
    for m in (base, mtp_clf):
        for p in m.parameters():
            p.requires_grad = False
    base.eval(); mtp_clf.eval()

    # Shared TokenPruner patch on the encoder last block IF the attention strategy is requested;
    # its importance is populated during the single shared encoder forward each batch.
    shared_tp = None
    if "attention" in strategies:
        shared_tp = T.TokenPruner(base.encoder, keep_count=args.keep_count, gp=gp)
    selectors = [StrategySelector(s, base, args.keep_count, gp, calib_path=args.pred_calib_path,
                                  score_block=args.pred_score_block, token_pruner=shared_tp)
                 for s in strategies]

    totals = {s: defaultdict(float) for s in strategies}
    counts = {s: defaultdict(int) for s in strategies}
    t0 = time.time()
    for it, batch in enumerate(loader):
        clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
        clips = clips.sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
        mtp_verbs = batch["mtp_verbs"].to(device); mtp_nouns = batch["mtp_nouns"].to(device)
        mtp_mask = batch["mtp_mask"].to(device)
        B = clips.size(0)
        ant = torch.full((B,), float(args.anticipation_sec), device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            x_full = base.encoder(clips)                       # <-- the ONE shared encoder forward
            for sel in selectors:
                x_ctx = sel.gather(x_full, ant)
                tokens = predict_from_encoded(base, x_ctx, ant)
                outputs = mtp_clf(tokens)
                for hi, h in enumerate(horizons):
                    valid = mtp_mask[:, hi] > 0.5
                    if not bool(valid.any()):
                        continue
                    v_lab, n_lab, a_lab, keep = T.map_labels(mtp_verbs[valid, hi], mtp_nouns[valid, hi],
                                                             verb_map, noun_map, action_map, device)
                    if not keep:
                        continue
                    vp = valid.nonzero(as_tuple=False).view(-1)[keep]
                    o = outputs[float(h)]
                    key = f"action_top5@{h:g}s"
                    totals[sel.name][key] += T.topk_acc(o["action"][vp].float(), a_lab, k=5) * len(keep)
                    counts[sel.name][key] += len(keep)
        if it % 20 == 0:
            k0 = f"action_top5@{primary_h:g}s"
            msg = " | ".join(f"{s}:{totals[s][k0]/max(1,counts[s][k0]):.3f}" for s in strategies)
            print(f"itr={it}/{len(loader)} ctx={float(batch['context_sec'][0]):.0f}s  {msg}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    secs = time.time() - t0
    for s in strategies:
        rep = {"prune_strategy": s, "shared_encoder": True, "max_frames": args.max_frames,
               "keep_count": args.keep_count, "only_context_sec": args.only_context_sec,
               **provenance,
               "action_top5": {f"{h:g}s": (totals[s][f"action_top5@{h:g}s"] /
                                           max(1, counts[s][f"action_top5@{h:g}s"])) for h in horizons},
               "n": {f"{h:g}s": counts[s][f"action_top5@{h:g}s"] for h in horizons},
               "seconds_total_shared": round(secs, 1)}
        (args.out_dir / f"{provenance['dataset']}-{s}-{args.tag}.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2), flush=True)
    print(f"[multi] {len(strategies)} strategies, one encoder pass, {secs:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
