#!/usr/bin/env python3
"""B13 route-3 GATING PROBE: is per-token task-saliency predictable from LOW-layer
encoder features?

Target (LOCKED): per-token grad x activation task-loss saliency
    s_i = | <dL_task/dt_i , t_i> |
where t_i = post-encoder token i over the FULL (un-pruned) history, L_task = the
streaming-MTP head loss (sum_horizon CE(verb)+CE(noun)+CE(action)). One backward per
sample. Secondary reference target (no extra backward): the final-block attention
importance (current champion selector) -- predicting IT from low layers asks whether
attention-pruning can be moved earlier.

Predictors: per-token output of encoder ``blocks[k]`` for k in --layers (+ final as an
upper reference). Encoder preserves token count, so block-k token i aligns to the final
token i and to saliency i by index.

Fit a closed-form ridge linear probe per (layer, target) on TRAIN samples (accumulate
A=Sum f f^T, b=Sum f s with per-sample z-scored s); evaluate on held-out TEST samples
by per-sample Spearman(pred,true) and top-K overlap. Baselines: recency (token
position), random. Linear = conservative lower bound for route-3 viability.

No finetuning; frozen base + mtp_clf loaded exactly like the eval harness. batch=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (
    FpsSubsampledStreamMTPDataset,
    enlarge_predictor_budget,
)


# ── post-encoder forward (mirror of PrunedAnticipativeModel.forward, no encoder) ──
def predict_from_tokens(core, x_full, anticipation_times):
    """Predictor rollout starting from provided post-encoder tokens (no prune)."""
    B, N, D_full = x_full.size()
    embed_dim = core.encoder.embed_dim
    use_hierarchical = D_full > embed_dim
    x = x_full[:, :, -embed_dim:] if use_hierarchical else x_full
    x_accumulate = x
    ctxt_positions = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
    anticipation_steps = (anticipation_times * core.frames_per_second / core.tubelet_size).to(torch.int64)
    skip_positions = N + int(core.grid_size**2) * anticipation_steps
    N_pred = int(core.grid_size**2 * (core.num_output_frames // core.tubelet_size))
    tgt_positions = torch.arange(N_pred, device=x.device).unsqueeze(0).repeat(B, 1)
    tgt_positions = tgt_positions + skip_positions.unsqueeze(1)
    x_pred_input = x_full
    for _ in range(core.num_steps):
        pred_out = core.predictor(x_pred_input, masks_x=ctxt_positions, masks_y=tgt_positions)
        x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        x_pred = x_pred_full[:, :, -embed_dim:] if x_pred_full.size(-1) != embed_dim else x_pred_full
        x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)
        x_pred_for_input = x_pred_full if x_pred_full.size(-1) == x_pred_input.size(-1) else x_pred
        x_pred_input = torch.cat([x_pred_input[:, N_pred:, :], x_pred_for_input], dim=1)
    return x_accumulate


def head_loss_from_tokens(tokens, mtp_clf, batch, horizons, verb_map, noun_map, action_map, device):
    """Replicate run_epoch's head_loss for a single sample (B=1)."""
    crit = nn.CrossEntropyLoss()
    outputs = mtp_clf(tokens)
    mtp_verbs = batch["mtp_verbs"].to(device)
    mtp_nouns = batch["mtp_nouns"].to(device)
    mtp_mask = batch["mtp_mask"].to(device)
    loss = tokens.new_zeros(())
    n_valid = 0
    for hi, h in enumerate(horizons):
        valid = mtp_mask[:, hi] > 0.5
        if not bool(valid.any()):
            continue
        v_lab, n_lab, a_lab, keep = T.map_labels(
            mtp_verbs[valid, hi], mtp_nouns[valid, hi], verb_map, noun_map, action_map, device
        )
        if not keep:
            continue
        valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
        o = outputs[float(h)]
        loss = loss + (
            crit(o["verb"][valid_pos], v_lab)
            + crit(o["noun"][valid_pos], n_lab)
            + crit(o["action"][valid_pos], a_lab)
        )
        n_valid += 1
    return loss, n_valid


# ── metrics ──────────────────────────────────────────────────────────────────
def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x))
    return r


def spearman(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 3:
        return float("nan")
    rp, rt = _ranks(pred), _ranks(true)
    rp -= rp.mean(); rt -= rt.mean()
    denom = np.sqrt((rp * rp).sum() * (rt * rt).sum())
    return float((rp * rt).sum() / denom) if denom > 0 else float("nan")


def topk_overlap(pred: np.ndarray, true: np.ndarray, k: int) -> float:
    k = min(k, len(pred))
    if k <= 0:
        return float("nan")
    pk = set(np.argpartition(-pred, k - 1)[:k].tolist())
    tk = set(np.argpartition(-true, k - 1)[:k].tolist())
    return len(pk & tk) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--init-from-ckpt", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--layers", type=str, default="2,4,8,16",
                    help="encoder block indices used as low-layer predictors")
    ap.add_argument("--keep-count", type=int, default=4096, help="top-K budget for overlap metric")
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=240, help="total samples used (train+test)")
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--val-subset-seed", type=int, default=0)
    ap.add_argument("--ridge-lambda-frac", type=float, default=1e-2,
                    help="ridge lambda = frac * mean(diag(A_feature))")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]

    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    val_ds = FpsSubsampledStreamMTPDataset(
        args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps
    )

    # Deterministic sample subset (stable hash of video_id+tick_frame), same keying as eval.
    salt = str(args.val_subset_seed)
    def _rowkey(r):
        return hashlib.md5(f"{salt}|{r['video_id']}|{r['tick_frame']}".encode()).hexdigest()
    order = sorted(range(len(val_ds.rows)), key=lambda i: _rowkey(val_ds.rows[i]))
    picked = order[: min(args.n_samples, len(order))]
    n_train = int(round(len(picked) * args.train_frac))
    train_ids = set(picked[:n_train])
    test_ids = set(picked[n_train:])
    print(f"[probe] n_samples={len(picked)} train={len(train_ids)} test={len(test_ids)} "
          f"layers={layers}", flush=True)

    base = T.build_model(device, args.max_frames, args.fps, args.img_size, str(args.checkpoint))
    for p in base.encoder.parameters():
        p.requires_grad = False
    T.load_lora_sidecars(
        base,
        str(args.encoder_lora) if args.encoder_lora else None,
        str(args.predictor_lora) if args.predictor_lora else None,
    )
    gp = int(base.grid_size**2)
    embed_dim = int(base.encoder.embed_dim)
    # Full-context (no-prune) predictor rollout overflows the pretrained num_patches;
    # raise the position budget (free, no new params) exactly as the eval harness does.
    num_tokens_full = (int(args.max_frames) // int(base.tubelet_size)) * gp
    enlarge_predictor_budget(base, num_tokens_full, gp)

    mtp_clf = T.CommunicatingMLPMTPClassifier(
        T.AttentiveClassifier(
            verb_classes=verb_map, noun_classes=noun_map, action_classes=action_map,
            embed_dim=embed_dim, num_heads=16, depth=4, use_activation_checkpointing=True,
        ),
        horizons_sec=horizons, comm_layers=2, comm_heads=4,
    ).to(device)

    # Load parent best.pt (model + mtp_classifier). model keys target base via PrunedAnticipativeModel
    # naming ("base.*"); load onto a matching wrapper then keep base for our manual forward.
    wrapper = T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).to(device)
    ck = torch.load(args.init_from_ckpt, map_location="cpu", weights_only=False)
    m_miss, m_unexp = wrapper.load_state_dict(ck["model"], strict=False)
    h_miss, h_unexp = mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
    print(f"[load] parent best={ck.get('best')} model(missing={len(m_miss)} unexpected={len(m_unexp)}) "
          f"mtp(missing={len(h_miss)} unexpected={len(h_unexp)})", flush=True)
    del ck
    for p in wrapper.parameters():
        p.requires_grad = False
    for p in mtp_clf.parameters():
        p.requires_grad = False
    core = base
    core.eval(); mtp_clf.eval()

    # block-k feature hooks
    feat_store = {}
    hooks = []
    def _mk_hook(k):
        def _h(_m, _i, out):
            feat_store[k] = (out[0] if isinstance(out, tuple) else out).detach()
        return _h
    for k in layers:
        hooks.append(core.encoder.blocks[k].register_forward_hook(_mk_hook(k)))

    # final-block attention importance (secondary target) via the existing TokenPruner patch
    attn_pruner = T.TokenPruner(core.encoder, keep_count=args.keep_count, gp=gp)

    all_layers = layers + ["final"]
    targets = ["task", "attn"]
    Dp = embed_dim + 1  # + bias column
    A = {k: torch.zeros(Dp, Dp, dtype=torch.float64) for k in all_layers}
    b = {(k, t): torch.zeros(Dp, dtype=torch.float64) for k in all_layers for t in targets}
    n_tokens_train = 0
    test_cache = []  # list of dicts: {feat:{k:np}, tgt:{t:np}, pos:np}

    ant = torch.full((1,), float(args.anticipation_sec), device=device)
    n_used = 0
    for idx in picked:
        batch = T.collate_stream([val_ds[idx]])
        clips = batch["clip"].to(device).float().div_(255.0)
        clips = clips.sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))

        feat_store.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            x_full = core.encoder(clips)
        N = x_full.shape[1]
        attn_imp = attn_pruner._importance[0, :N].detach().float().cpu().numpy()

        # task saliency: grad x activation wrt post-encoder tokens
        t = x_full.detach().float().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            tokens = predict_from_tokens(core, t, ant)
            loss, n_valid = head_loss_from_tokens(
                tokens, mtp_clf, batch, horizons, verb_map, noun_map, action_map, device
            )
        if n_valid == 0 or not torch.isfinite(loss):
            continue
        grad = torch.autograd.grad(loss, t)[0]  # [1,N,D]
        sal = (grad.float() * t.float()).sum(-1).abs().squeeze(0).detach().cpu().numpy()  # [N]

        # per-token low-layer features (+ final)
        feats = {}
        for k in layers:
            feats[k] = feat_store[k][0, :, -embed_dim:].detach().float().cpu().numpy()  # [N,D]
        feats["final"] = x_full[0, :, -embed_dim:].detach().float().cpu().numpy()

        tgt_vals = {"task": sal, "attn": attn_imp}
        pos = np.arange(N, dtype=np.float64)  # recency: larger = more recent
        n_used += 1

        if idx in train_ids:
            for k in all_layers:
                F = torch.from_numpy(feats[k]).double()  # [N,D]
                F1 = torch.cat([F, torch.ones(F.shape[0], 1, dtype=torch.float64)], dim=1)  # [N,D+1]
                A[k] += F1.t() @ F1
                for tname in targets:
                    s = tgt_vals[tname].astype(np.float64)
                    s = (s - s.mean()) / (s.std() + 1e-8)  # per-sample z-score
                    b[(k, tname)] += F1.t() @ torch.from_numpy(s).double()
            n_tokens_train += N
        else:
            test_cache.append({
                "feat": {k: feats[k].astype(np.float16) for k in all_layers},
                "tgt": {t_: tgt_vals[t_].astype(np.float32) for t_ in targets},
                "pos": pos,
            })
        if n_used % 20 == 0:
            print(f"[probe] processed {n_used}/{len(picked)} (N={N})", flush=True)

    for h in hooks:
        h.remove()
    attn_pruner.remove()

    # fit ridge per (layer, target)
    weights = {}
    for k in all_layers:
        Ak = A[k].clone()
        diag_feat = Ak.diagonal()[:-1].mean().item()
        lam = args.ridge_lambda_frac * max(diag_feat, 1e-6)
        reg = torch.eye(Dp, dtype=torch.float64) * lam
        reg[-1, -1] = 0.0  # no penalty on bias
        for tname in targets:
            try:
                w = torch.linalg.solve(Ak + reg, b[(k, tname)])
            except RuntimeError:
                w = torch.linalg.lstsq(Ak + reg, b[(k, tname)].unsqueeze(1)).solution.squeeze(1)
            weights[(k, tname)] = w.numpy()

    # evaluate on test
    K = args.keep_count
    report = {"config": {"layers": layers, "keep_count": K, "n_train_samples": len(train_ids),
                         "n_test_samples": len(test_cache), "n_tokens_train": n_tokens_train,
                         "ridge_lambda_frac": args.ridge_lambda_frac,
                         "anticipation_sec": args.anticipation_sec, "horizons": horizons},
              "results": {}}
    for tname in targets:
        report["results"][tname] = {}
        # probes per layer
        for k in all_layers:
            w = weights[(k, tname)]
            sp, ov = [], []
            for c in test_cache:
                F = c["feat"][k].astype(np.float64)
                pred = F @ w[:-1] + w[-1]
                true = c["tgt"][tname].astype(np.float64)
                sp.append(spearman(pred, true))
                ov.append(topk_overlap(pred, true, K))
            report["results"][tname][f"block{k}" if k != "final" else "final"] = {
                "spearman_mean": float(np.nanmean(sp)),
                "spearman_std": float(np.nanstd(sp)),
                "topk_overlap_mean": float(np.nanmean(ov)),
            }
        # baselines
        sp_rec, ov_rec, sp_rnd, ov_rnd = [], [], [], []
        rng = np.random.default_rng(0)
        for c in test_cache:
            true = c["tgt"][tname].astype(np.float64)
            sp_rec.append(spearman(c["pos"], true))
            ov_rec.append(topk_overlap(c["pos"], true, K))
            rnd = rng.standard_normal(len(true))
            sp_rnd.append(spearman(rnd, true))
            ov_rnd.append(topk_overlap(rnd, true, K))
        report["results"][tname]["baseline_recency"] = {
            "spearman_mean": float(np.nanmean(sp_rec)), "topk_overlap_mean": float(np.nanmean(ov_rec))}
        report["results"][tname]["baseline_random"] = {
            "spearman_mean": float(np.nanmean(sp_rnd)), "topk_overlap_mean": float(np.nanmean(ov_rnd))}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
