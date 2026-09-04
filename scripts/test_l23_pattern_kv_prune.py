#!/usr/bin/env python3
"""A/B test: baseline vs L23 pattern KV prune (video-only V-JEPA)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

from app.hdepic_lora_action_anticipation.l23_pattern_kv_prune import (  # noqa: E402
    CONTENT_HEADS,
    L23PatternKVPruner,
    L23PruneConfig,
    STABLE_HEADS,
    build_head_keep_mask,
)

_spec = importlib.util.spec_from_file_location(
    "analyze_fastgen_heads", PROJECT_ROOT / "scripts" / "analyze_fastgen_heads.py"
)
_fastgen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fastgen)

logger = logging.getLogger("l23_prune_test")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def column_sum_importance(q, k, scale, chunk=128) -> torch.Tensor:
    _, h, n, _ = q.shape
    imp = torch.zeros(h, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)
    for ci in range(0, n, chunk):
        logits = torch.matmul(q[:, :, ci : ci + chunk].float(), k_t) * scale
        imp += logits.softmax(dim=-1).sum(dim=2).mean(dim=0)
    return imp


def capture_l23(encoder, clip, chunk=128):
    attn = encoder.blocks[-1].attn
    orig = attn.forward
    holder = {}

    def wrapped(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        q, k, v = _fastgen._rope_qk(attn, x, mask, T, H_patches, W_patches)
        holder["imp"] = column_sum_importance(q, k, attn.scale, chunk=chunk)
        with torch.backends.cuda.sdp_kernel():
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=attn.is_causal, attn_mask=attn_mask
            )
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2])
        y = attn.proj(y)
        y = attn.proj_drop(y)
        return y

    attn.forward = wrapped
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = encoder(clip)
    attn.forward = orig
    return out, holder["imp"]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(F.cosine_similarity(a[None], b[None]).item())


def try_gaze(clip, meta, args, grid: int):
    try:
        from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate
        from app.hdepic_lora_action_anticipation.pose_map_builder import GazePoseInputMapBuilder

        cfg = _fastgen.gaze_cfg(args)
        gate = GazeTokenGate({**cfg, "mode": "token_gate"})
        builder = GazePoseInputMapBuilder(cfg, gate=gate)
        aux = builder.build(clip, [meta])
        gaze_ch = aux[:, 0]
        return _fastgen.gaze_token_mask(gaze_ch[0], 2, grid)
    except Exception as exc:  # noqa: BLE001
        logger.info("gaze unavailable (%s); content heads use center∪recent", exc)
        return None


def mean_key(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("TRI_MODAL_FRAME_CACHE", "/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1")
    samples = _fastgen.load_val_clips(args)
    model = _fastgen.load_video_model(args, device)
    encoder = model.encoder
    grid = int(getattr(model, "grid_size", 16))
    cfg = L23PruneConfig(
        grid=grid, border=args.border, recent_frac=args.recent_frac, center_frac=args.center_frac
    )

    recs = []
    for si, (clip, meta) in enumerate(samples):
        vid = str(meta.get("video_id", f"s{si}"))
        clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
        ant = torch.full((1,), float(args.anticipation), device=device)
        gaze = try_gaze(clip_b, meta, args, grid)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            enc0, imp = capture_l23(encoder, clip_b, chunk=args.chunk)
            pred0 = model(clip_b, ant)

        n_tok = int(imp.shape[1])
        n_heads = int(imp.shape[0])
        keep = build_head_keep_mask(n_tok, n_heads, cfg, gaze, device=imp.device)
        mass = imp.clamp_min(0)
        tot = mass.sum(dim=1).clamp_min(1e-12)
        recov = (mass * keep.float()).sum(dim=1) / tot

        pruner = L23PatternKVPruner(encoder, cfg)
        pruner.set_gaze(gaze.to(device) if gaze is not None else None)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            enc1 = encoder(clip_b)
            pred1 = model(clip_b, ant)
        stats = pruner.keep_stats()
        pruner.remove()

        row = {
            "video_id": vid,
            "used_gaze": bool(gaze is not None and bool(gaze.any())),
            "keep": stats,
            "mass_recovery_mean": float(recov.mean()),
            "mass_recovery_stable": float(recov[list(STABLE_HEADS)].mean()),
            "mass_recovery_content": float(recov[list(CONTENT_HEADS)].mean()),
            "mass_recovery_per_head": [float(x) for x in recov.cpu()],
            "encoder_cosine": cosine(enc0, enc1),
            "predictor_cosine": cosine(pred0, pred1),
        }
        recs.append(row)
        logger.info(
            "[%d/%d] %s keep=%.3f recov=%.3f enc_cos=%.4f pred_cos=%.4f gaze=%s",
            si + 1,
            len(samples),
            vid,
            stats.get("all_keep", 0.0),
            row["mass_recovery_mean"],
            row["encoder_cosine"],
            row["predictor_cosine"],
            row["used_gaze"],
        )

    summary = {
        "n_clips": len(recs),
        "policy": {
            "stable_heads": list(STABLE_HEADS),
            "content_heads": list(CONTENT_HEADS),
            "stable": "keep border∪recent",
            "content": "keep gaze∪center∪recent",
            "other": "full KV",
            "border": cfg.border,
            "recent_frac": cfg.recent_frac,
            "center_frac": cfg.center_frac,
        },
        "mean_keep": float(np.mean([r["keep"]["all_keep"] for r in recs])),
        "mean_keep_stable": float(np.mean([r["keep"]["stable_keep"] for r in recs])),
        "mean_keep_content": float(np.mean([r["keep"]["content_keep"] for r in recs])),
        "mean_mass_recovery": mean_key(recs, "mass_recovery_mean"),
        "mean_mass_recovery_stable": mean_key(recs, "mass_recovery_stable"),
        "mean_mass_recovery_content": mean_key(recs, "mass_recovery_content"),
        "mean_encoder_cosine": mean_key(recs, "encoder_cosine"),
        "mean_predictor_cosine": mean_key(recs, "predictor_cosine"),
        "gaze_clips": int(sum(r["used_gaze"] for r in recs)),
        "per_clip": recs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    logger.info("=== summary ===")
    for k, v in summary.items():
        if k in ("policy", "per_clip"):
            continue
        logger.info("  %s = %s", k, v)
    logger.info("wrote %s", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sample", type=int, default=12)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--anticipation", type=float, default=1.0)
    p.add_argument("--border", type=int, default=2)
    p.add_argument("--recent-frac", type=float, default=0.30)
    p.add_argument("--center-frac", type=float, default=0.50)
    p.add_argument("--val-csv", default="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv")
    p.add_argument("--video-root", default="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos")
    p.add_argument("--vjepa-ckpt", default="/scratch/ll5914/models/vjepa2/vitl.pt")
    p.add_argument(
        "--video-enc-lora",
        default="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/encoder_lora_best.pt",
    )
    p.add_argument(
        "--video-pred-lora",
        default="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt",
    )
    p.add_argument("--gaze-root", default="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze")
    p.add_argument("--gaze-extract", default="/scratch/ll5914/datasets/HD-EPIC/_gaze_extract")
    p.add_argument("--gaze-sync", default="/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos")
    p.add_argument("--pose-slam", default="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi")
    p.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "experiments/fastgen_head_heatmaps/encoder_L23_multiclip/l23_kv_prune_test.json"),
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
