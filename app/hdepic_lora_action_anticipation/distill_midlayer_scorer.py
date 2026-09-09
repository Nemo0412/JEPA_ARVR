#!/usr/bin/env python3
# [B17-LEARNED-EARLY-PRUNER] Stage 1: distill a MidLayerTokenScorer to per-token
# TASK saliency (and, as a control arm, to received-attention importance) from
# block-L* features. Saves scorer_{task,attn}.pt for Stage 2 co-adaptation.
"""Stage 1 distillation trainer for candidate #1.

Per train sample: encoder forward (hook block-L* output) + one backward for
grad×act post-encoder task saliency (reuses the gating-probe machinery). Train a
small MLP to predict per-sample z-scored saliency (MSE); evaluate held-out by
per-sample Spearman + top-4096 overlap. Same block-L* OUTPUT features the Stage-2
LearnedMidLayerEncoderPruner scores at inference -> the distilled scorer transfers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (
    FpsSubsampledStreamMTPDataset,
    enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.gating_probe_saliency import (
    predict_from_tokens,
    head_loss_from_tokens,
    spearman,
    topk_overlap,
)
from app.hdepic_lora_action_anticipation.midlayer_scorer import MidLayerTokenScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--init-from-ckpt", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--prune-layer", type=int, default=8)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--targets", type=str, default="task,attn",
                    help="which scorer arms to distill (task=grad×act saliency, attn=received attn)")
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-subset-seed", type=int, default=0)
    ap.add_argument("--mlp-epochs", type=int, default=30)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-batch-tokens", type=int, default=65536)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    targets = [t for t in args.targets.split(",") if t.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    val_ds = FpsSubsampledStreamMTPDataset(
        args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps
    )
    salt = str(args.val_subset_seed)
    def _rowkey(r):
        return hashlib.md5(f"{salt}|{r['video_id']}|{r['tick_frame']}".encode()).hexdigest()
    order = sorted(range(len(val_ds.rows)), key=lambda i: _rowkey(val_ds.rows[i]))
    picked = order[: min(args.n_samples, len(order))]
    n_train = int(round(len(picked) * args.train_frac))
    train_ids = set(picked[:n_train])
    print(f"[distill] n={len(picked)} train={len(train_ids)} test={len(picked)-len(train_ids)} "
          f"L*={args.prune_layer} targets={targets}", flush=True)

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
    num_tokens_full = (int(args.max_frames) // int(base.tubelet_size)) * gp
    enlarge_predictor_budget(base, num_tokens_full, gp)

    mtp_clf = T.CommunicatingMLPMTPClassifier(
        T.AttentiveClassifier(
            verb_classes=verb_map, noun_classes=noun_map, action_classes=action_map,
            embed_dim=embed_dim, num_heads=16, depth=4, use_activation_checkpointing=True,
        ),
        horizons_sec=horizons, comm_layers=2, comm_heads=4,
    ).to(device)
    wrapper = T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).to(device)
    ck = torch.load(args.init_from_ckpt, map_location="cpu", weights_only=False)
    wrapper.load_state_dict(ck["model"], strict=False)
    mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
    del ck
    for p in wrapper.parameters():
        p.requires_grad = False
    for p in mtp_clf.parameters():
        p.requires_grad = False
    core = base
    core.eval(); mtp_clf.eval()

    feat_store = {}
    h = core.encoder.blocks[args.prune_layer].register_forward_hook(
        lambda _m, _i, out: feat_store.__setitem__("f", (out[0] if isinstance(out, tuple) else out).detach())
    )
    attn_pruner = T.TokenPruner(core.encoder, keep_count=args.keep_count, gp=gp)

    # collect
    train_feats, train_tgt = [], {t: [] for t in targets}  # pooled tokens
    test_samples = []  # per-sample dicts for held-out metrics
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
        feat = feat_store["f"][0, :, -embed_dim:].detach().float().cpu()  # [N,D]
        attn_imp = attn_pruner._importance[0, :N].detach().float().cpu().numpy()

        t = x_full.detach().float().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            tokens = predict_from_tokens(core, t, ant)
            loss, n_valid = head_loss_from_tokens(
                tokens, mtp_clf, batch, horizons, verb_map, noun_map, action_map, device
            )
        if n_valid == 0 or not torch.isfinite(loss):
            continue
        grad = torch.autograd.grad(loss, t)[0]
        sal = (grad.float() * t.float()).sum(-1).abs().squeeze(0).detach().cpu().numpy()  # [N]
        tvals = {"task": sal, "attn": attn_imp}
        n_used += 1

        if idx in train_ids:
            train_feats.append(feat.half())
            for tn in targets:
                s = tvals[tn].astype(np.float32)
                s = (s - s.mean()) / (s.std() + 1e-8)  # per-sample z-score
                train_tgt[tn].append(torch.from_numpy(s))
        else:
            test_samples.append({"feat": feat.half(),
                                 "tgt": {tn: tvals[tn].astype(np.float32) for tn in targets}})
        if n_used % 25 == 0:
            print(f"[distill] collected {n_used}/{len(picked)} (N={N})", flush=True)

    h.remove(); attn_pruner.remove()
    Xtr = torch.cat(train_feats, dim=0)  # [Ttot, D] fp16 cpu
    print(f"[distill] train tokens={Xtr.shape[0]} dim={Xtr.shape[1]}", flush=True)

    report = {"config": {"prune_layer": args.prune_layer, "keep_count": args.keep_count,
                         "n_train_samples": len(train_ids), "n_test_samples": len(test_samples),
                         "n_train_tokens": int(Xtr.shape[0]), "hidden": args.hidden,
                         "mlp_epochs": args.mlp_epochs, "targets": targets},
              "results": {}}

    for tn in targets:
        ytr = torch.cat(train_tgt[tn], dim=0).float()  # [Ttot]
        scorer = MidLayerTokenScorer(embed_dim, hidden=args.hidden).to(device)
        opt = torch.optim.Adam(scorer.parameters(), lr=args.mlp_lr)
        Ttot = Xtr.shape[0]
        bs = args.mlp_batch_tokens
        scorer.train()
        for ep in range(args.mlp_epochs):
            perm = torch.randperm(Ttot)
            tot = 0.0
            for i in range(0, Ttot, bs):
                sel = perm[i:i + bs]
                xb = Xtr[sel].to(device).float()
                yb = ytr[sel].to(device)
                pred = scorer(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tot += float(loss) * len(sel)
            if ep % 5 == 0 or ep == args.mlp_epochs - 1:
                print(f"[distill:{tn}] epoch {ep} mse={tot/Ttot:.4f}", flush=True)

        # held-out metrics
        scorer.eval()
        sp, ov = [], []
        with torch.no_grad():
            for c in test_samples:
                xb = c["feat"].to(device).float()
                pred = scorer(xb).cpu().numpy()
                true = c["tgt"][tn]
                sp.append(spearman(pred, true))
                ov.append(topk_overlap(pred, true, args.keep_count))
        report["results"][tn] = {"spearman_mean": float(np.nanmean(sp)),
                                 "spearman_std": float(np.nanstd(sp)),
                                 "topk_overlap_mean": float(np.nanmean(ov))}
        print(f"[distill:{tn}] HELD-OUT sp={np.nanmean(sp):.3f} ov={np.nanmean(ov):.3f}", flush=True)
        ckpt = {"state_dict": scorer.state_dict(), "embed_dim": embed_dim, "hidden": args.hidden,
                "prune_layer": args.prune_layer, "target": tn, "keep_count": args.keep_count}
        torch.save(ckpt, args.out_dir / f"scorer_{tn}.pt")
        print(f"[distill:{tn}] saved {args.out_dir / f'scorer_{tn}.pt'}", flush=True)

    (args.out_dir / "distill_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
