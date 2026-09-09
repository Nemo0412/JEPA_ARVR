#!/usr/bin/env python3
# [B13-JEPA] Is the ONLINE self-supervised JEPA-loss saliency a discriminative,
# task-aligned, deployable keep signal?
"""JEPA-saliency analysis (B13, candidate = online JEPA-loss pruning signal).

Per sample computes THREE per-token importances over the full context:
  * task  : grad×act of the streaming-MTP head loss  (labels; offline-only)
  * jepa  : grad×act of the online JEPA loss = predictor's prediction of the held-out
            most-recent chunk vs its GT encoder latent  (self-supervised; online-available)
  * attn  : final-block received attention  (the frozen champion selector)

Reports, averaged over samples:
  * DISCRIMINATIVENESS  -- coefficient of variation (std/mean) of each signal; ~0 = uniform
    (the B12 "predictor attention is uniform / OOD" risk -> signal is dead).
  * TASK-ALIGNMENT      -- per-sample Spearman(jepa, task): high => jepa is a LABEL-FREE
    proxy for the (offline-only) task saliency; the whole pitch.
  * CHAMPION-ALIGNMENT  -- Spearman(jepa, attn), Spearman(task, attn).
  * RECENCY PROFILE     -- Spearman(signal, token position): does each favour recent tokens
    (recency being the known good prior; loss_aware ~0.72 recent)?

Decisive: if jepa is discriminative AND task-aligned, we have an online, label-free,
deployable signal that matches the task target -> strong candidate + strong story. If
uniform or task-anti-aligned, the online JEPA signal is not usable as-is.
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
    FpsSubsampledStreamMTPDataset, enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.gating_probe_saliency import (
    predict_from_tokens, head_loss_from_tokens, spearman,
)
from app.hdepic_lora_action_anticipation.eval_saliency_oracle_prune import jepa_saliency


def _saliency_from_loss(core, x_full, loss_fn):
    t = x_full.detach().float().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(x_full.device.type == "cuda")):
        loss, ok = loss_fn(t)
    if not ok or not torch.isfinite(loss):
        return None
    grad = torch.autograd.grad(loss, t)[0]
    return (grad.float() * t.float()).sum(-1).abs().squeeze(0).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--init-from-ckpt", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--jepa-held-out-sec", type=float, default=2.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--val-subset-seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    val_ds = FpsSubsampledStreamMTPDataset(
        args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps
    )
    salt = str(args.val_subset_seed)
    def _rk(r):
        return hashlib.md5(f"{salt}|{r['video_id']}|{r['tick_frame']}".encode()).hexdigest()
    order = sorted(range(len(val_ds.rows)), key=lambda i: _rk(val_ds.rows[i]))
    picked = order[: min(args.n_samples, len(order))]

    base = T.build_model(device, args.max_frames, args.fps, args.img_size, str(args.checkpoint))
    for p in base.encoder.parameters():
        p.requires_grad = False
    T.load_lora_sidecars(base, str(args.encoder_lora) if args.encoder_lora else None,
                         str(args.predictor_lora) if args.predictor_lora else None)
    gp = int(base.grid_size**2); embed_dim = int(base.encoder.embed_dim)
    enlarge_predictor_budget(base, (int(args.max_frames)//int(base.tubelet_size))*gp, gp)

    mtp_clf = T.CommunicatingMLPMTPClassifier(
        T.AttentiveClassifier(verb_classes=verb_map, noun_classes=noun_map, action_classes=action_map,
                              embed_dim=embed_dim, num_heads=16, depth=4, use_activation_checkpointing=True),
        horizons_sec=horizons, comm_layers=2, comm_heads=4).to(device)
    wrapper = T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).to(device)
    ck = torch.load(args.init_from_ckpt, map_location="cpu", weights_only=False)
    wrapper.load_state_dict(ck["model"], strict=False); mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
    del ck
    for p in list(wrapper.parameters()) + list(mtp_clf.parameters()):
        p.requires_grad = False
    core = base; core.eval(); mtp_clf.eval()

    attn_pruner = T.TokenPruner(core.encoder, keep_count=args.keep_count, gp=gp)
    held = int(round(args.jepa_held_out_sec * args.fps / int(base.tubelet_size))) * gp
    ant = torch.full((1,), float(args.anticipation_sec), device=device)

    def _cv(a):
        m = float(np.mean(a)); s = float(np.std(a))
        return s / (abs(m) + 1e-12)

    acc = {k: [] for k in ["cv_task", "cv_jepa", "cv_attn", "sp_jepa_task", "sp_jepa_attn",
                           "sp_task_attn", "sp_task_pos", "sp_jepa_pos", "sp_attn_pos"]}
    n_used = 0
    for idx in picked:
        batch = T.collate_stream([val_ds[idx]])
        clips = batch["clip"].to(device).float().div_(255.0)
        clips = clips.sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            x_full = core.encoder(clips)
        N = x_full.shape[1]
        if N <= args.keep_count + gp:  # need real pruning regime + a held-out chunk
            continue
        attn = attn_pruner._importance[0, :N].detach().float().cpu().numpy()
        task = _saliency_from_loss(core, x_full, lambda t: head_loss_from_tokens(
            predict_from_tokens(core, t, ant), mtp_clf, batch, horizons, verb_map, noun_map, action_map, device))
        jep = _saliency_from_loss(core, x_full, lambda t: (jepa_saliency(core, t, held_out_tokens=held, embed_dim=embed_dim)[0], True))
        if task is None or jep is None:
            continue
        # jepa saliency is defined over history tokens only; restrict all comparisons to
        # the history region [0, N-held) so the held-out (force-kept) chunk doesn't skew it
        hist = N - held
        task_h, jep_h, attn_h = task[:hist], jep[:hist], attn[:hist]
        pos = np.arange(hist, dtype=np.float64)
        acc["cv_task"].append(_cv(task_h)); acc["cv_jepa"].append(_cv(jep_h)); acc["cv_attn"].append(_cv(attn_h))
        acc["sp_jepa_task"].append(spearman(jep_h, task_h))
        acc["sp_jepa_attn"].append(spearman(jep_h, attn_h))
        acc["sp_task_attn"].append(spearman(task_h, attn_h))
        acc["sp_task_pos"].append(spearman(task_h, pos))
        acc["sp_jepa_pos"].append(spearman(jep_h, pos))
        acc["sp_attn_pos"].append(spearman(attn_h, pos))
        n_used += 1
        if n_used % 25 == 0:
            print(f"[jepa-analysis] {n_used}/{len(picked)} "
                  f"cv_jepa={np.nanmean(acc['cv_jepa']):.3f} sp_jepa_task={np.nanmean(acc['sp_jepa_task']):.3f}", flush=True)

    report = {"n_samples": n_used, "keep_count": args.keep_count, "jepa_held_out_sec": args.jepa_held_out_sec,
              "results": {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))} for k, v in acc.items()}}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
