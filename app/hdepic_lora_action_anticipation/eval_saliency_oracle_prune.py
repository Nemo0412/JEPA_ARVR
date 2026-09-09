#!/usr/bin/env python3
# [B13-ORACLE] Prune by a per-sample ORACLE importance signal and report native
# Action Top-5 -- the UPPER BOUND of a learned scorer for that signal (perfect scorer).
"""Oracle-saliency prune eval (B13).

Answers "is the task-saliency TARGET even a good oracle for the keep decision?" by
pruning to top-K with the TRUE per-sample importance (not a learned scorer), then
measuring native Action Top-5 @ +2/+4/+6 s on the frozen reader. If the oracle keep-set
underperforms attention, the target is the problem (no scorer/co-adapt can fix it).

Signals (--saliency):
  * ``task``      : grad×act of the streaming-MTP head loss (labels; our Stage-1 target)
  * ``attention`` : final-block received attention (sanity -- should reproduce the
                    attention baseline 39.84, validating the two-pass harness)
  * ``jepa``      : grad×act of the online self-supervised JEPA loss (predictor's
                    prediction of the held-out most-recent chunk vs its GT encoder
                    latent). Held-out chunk force-kept (recency); history ranked by it.

Two-pass per sample: pass-1 backward for the saliency, pass-2 prune top-K + forward for
the metric (frozen). batch=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (
    FpsSubsampledStreamMTPDataset,
    enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.gating_probe_saliency import (
    predict_from_tokens,
    head_loss_from_tokens,
)


def jepa_saliency(core, x_full_leaf, *, held_out_tokens: int, embed_dim: int):
    """Online self-supervised JEPA loss: predict the held-out most-recent chunk from the
    preceding history and compare to its GT encoder latent. Returns (loss, hist_len).
    Gradient flows only to the history tokens (GT chunk detached)."""
    B, N, D = x_full_leaf.shape
    H = min(held_out_tokens, N - core.grid_size**2)  # keep >=1 history slot
    n_hist = N - H
    hist = x_full_leaf[:, :n_hist, :]
    gt = x_full_leaf[:, n_hist:, -embed_dim:].detach()
    ctxt_positions = torch.arange(n_hist, device=x_full_leaf.device).unsqueeze(0).repeat(B, 1)
    tgt_positions = torch.arange(n_hist, N, device=x_full_leaf.device).unsqueeze(0).repeat(B, 1)
    pred_out = core.predictor(hist, masks_x=ctxt_positions, masks_y=tgt_positions)
    pred = pred_out[0] if isinstance(pred_out, tuple) else pred_out
    pred = pred[:, :, -embed_dim:] if pred.size(-1) != embed_dim else pred
    pred = pred[:, :H, :]
    loss = F.mse_loss(pred, gt)
    return loss, n_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--init-from-ckpt", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--saliency", choices=["task", "attention", "jepa"], required=True)
    ap.add_argument("--saliency-agg", choices=["innerprod", "meanabs"], default="innerprod",
                    help="per-dim aggregation of grad*z: 'innerprod'=|sum_D(grad*z)| (Taylor, "
                         "allows cancellation); 'meanabs'=mean_D|grad*z| (loss_aware's exact "
                         "aggregation, no cancellation). Controls the aggregation variable when "
                         "comparing to loss_aware.")
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--jepa-held-out-sec", type=float, default=2.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--primary-horizon-sec", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--val-subset-n", type=int, default=0)
    ap.add_argument("--val-subset-seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    val_ds = FpsSubsampledStreamMTPDataset(
        args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps
    )
    if args.val_subset_n > 0 and args.val_subset_n < len(val_ds.rows):
        salt = str(args.val_subset_seed)
        def _rk(r):
            return hashlib.md5(f"{salt}|{r['video_id']}|{r['tick_frame']}".encode()).hexdigest()
        order = sorted(range(len(val_ds.rows)), key=lambda i: _rk(val_ds.rows[i]))
        keep = set(order[: args.val_subset_n])
        val_ds.rows = [r for i, r in enumerate(val_ds.rows) if i in keep]
    print(f"[oracle] saliency={args.saliency} keep={args.keep_count} n={len(val_ds.rows)}", flush=True)

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
    for p in list(wrapper.parameters()) + list(mtp_clf.parameters()):
        p.requires_grad = False
    core = base
    core.eval(); mtp_clf.eval()

    attn_pruner = T.TokenPruner(core.encoder, keep_count=args.keep_count, gp=gp) if args.saliency == "attention" else None
    held_out_tokens = int(round(args.jepa_held_out_sec * args.fps / int(base.tubelet_size))) * gp

    totals, counts = defaultdict(float), defaultdict(int)
    K = max(gp, (args.keep_count // gp) * gp)
    ant = torch.full((1,), float(args.anticipation_sec), device=device)
    n_used = 0
    for idx in range(len(val_ds.rows)):
        batch = T.collate_stream([val_ds[idx]])
        clips = batch["clip"].to(device).float().div_(255.0)
        clips = clips.sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            x_full = core.encoder(clips)
        N = x_full.shape[1]
        if N <= K:  # no pruning needed
            sal = torch.arange(N, device=device).float()  # keep all -> order irrelevant
        elif args.saliency == "attention":
            sal = attn_pruner._importance[0, :N].detach().float()
        else:
            t = x_full.detach().float().clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                if args.saliency == "task":
                    loss, nv = head_loss_from_tokens(
                        predict_from_tokens(core, t, ant), mtp_clf, batch,
                        horizons, verb_map, noun_map, action_map, device)
                else:  # jepa
                    loss, n_hist = jepa_saliency(core, t, held_out_tokens=held_out_tokens, embed_dim=embed_dim)
                    nv = 1
            if nv == 0 or not torch.isfinite(loss):
                continue
            grad = torch.autograd.grad(loss, t)[0]
            gz = grad.float() * t.float()
            if args.saliency_agg == "meanabs":  # loss_aware's exact aggregation
                sal = gz.abs().mean(-1).squeeze(0)
            else:  # innerprod (Taylor)
                sal = gz.sum(-1).abs().squeeze(0)  # [N]
            if args.saliency == "jepa":  # force-keep the held-out recent chunk
                sal[N - held_out_tokens:] = sal.max() + 1.0

        # prune top-K (token-level, chronological order -- mirrors attention TokenPruner)
        if N > K:
            keep_idx = sal.topk(K, dim=0).indices.sort().values
            pruned = x_full[:, keep_idx, :]
        else:
            pruned = x_full

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            tokens = predict_from_tokens(core, pruned, ant)
            outputs = mtp_clf(tokens)
        mtp_verbs = batch["mtp_verbs"].to(device); mtp_nouns = batch["mtp_nouns"].to(device)
        mtp_mask = batch["mtp_mask"].to(device)
        for hi, h in enumerate(horizons):
            if not bool(mtp_mask[:, hi].gt(0.5).any()):
                continue
            valid = mtp_mask[:, hi] > 0.5
            v_lab, n_lab, a_lab, keep = T.map_labels(mtp_verbs[valid, hi], mtp_nouns[valid, hi],
                                                     verb_map, noun_map, action_map, device)
            if not keep:
                continue
            vp = valid.nonzero(as_tuple=False).view(-1)[keep]
            acc = T.topk_acc(outputs[float(h)]["action"][vp].float(), a_lab, k=5) * len(keep)
            totals[f"action_top5@{h:g}s"] += acc
            counts[f"action_top5@{h:g}s"] += len(keep)
        n_used += 1
        if n_used % 50 == 0:
            print(f"[oracle] {n_used}/{len(val_ds.rows)} "
                  f"@2s={100*totals['action_top5@2s']/max(1,counts['action_top5@2s']):.2f}", flush=True)

    report = {"saliency": args.saliency, "keep_count": args.keep_count, "n": dict(counts),
              "jepa_held_out_sec": args.jepa_held_out_sec if args.saliency == "jepa" else None,
              "metric_scope": "native oracle prune; frozen EGTEA split1 stream-MTP val",
              "action_top5": {f"{h:g}s": (100 * totals[f"action_top5@{h:g}s"] /
                              max(1, counts[f"action_top5@{h:g}s"])) for h in horizons}}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
