#!/usr/bin/env python3
"""Generate 3 clear explainer figures for Encoder L23 multiclip findings."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("mc", PROJECT_ROOT / "scripts/analyze_encoder_last_layer_multiclip.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

STABLE = [0, 1, 3, 4, 6, 9, 11, 12]
CONTENT = [8, 10, 14]
EXAMPLE_VIDS = 3


def collect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = mc.load_diverse_clips(args)
    model = mc._fastgen.load_video_model(args, device)
    encoder = model.encoder
    grid = int(getattr(model, "grid_size", 16))
    last = len(encoder.blocks) - 1
    rows = []
    for si, (clip, meta) in enumerate(samples):
        cap = mc.LastLayerCapture(encoder, last, args.chunk)
        clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
        ant = torch.full((1,), float(args.anticipation), device=device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model(clip_b, ant)
        cap.restore()
        if out is None or cap.imp is None:
            continue
        per_head = {}
        for h in range(cap.imp.shape[0]):
            f = mc.head_features(cap.imp[h], grid)
            per_head[h] = f
        rows.append({"video_id": meta["video_id"], "heads": per_head})
    return rows, grid, last


def plot_stable_vs_content(rows, grid, last, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    # Row 0: stable head h3 on 3 videos + mean
    h = 3
    for col in range(EXAMPLE_VIDS):
        sp = rows[col]["heads"][h]["spatial"]
        mx = sp.max() or 1
        axes[0, col].imshow(sp / mx, cmap="viridis", vmin=0, vmax=1)
        axes[0, col].set_title(f"stable h3\n{rows[col]['video_id'][-6:]}", fontsize=8)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    mean_sp = np.mean([r["heads"][h]["spatial"] for r in rows], axis=0)
    axes[0, 3].imshow(mean_sp / (mean_sp.max() or 1), cmap="viridis", vmin=0, vmax=1)
    axes[0, 3].set_title("stable h3\n20-video mean", fontsize=8)
    axes[0, 3].set_xticks([]); axes[0, 3].set_yticks([])
    fig.text(0.02, 0.72, "Stable heads (corner/border anchor)\nh0,h1,h3,h4,h6,h9,h11,h12", fontsize=10, va="center")

    # Row 1: content head h8 on 3 videos + mean
    h = 8
    for col in range(EXAMPLE_VIDS):
        sp = rows[col]["heads"][h]["spatial"]
        mx = sp.max() or 1
        axes[1, col].imshow(sp / mx, cmap="viridis", vmin=0, vmax=1)
        axes[1, col].set_title(f"content h8\n{rows[col]['video_id'][-6:]}", fontsize=8)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
    mean_sp = np.mean([r["heads"][h]["spatial"] for r in rows], axis=0)
    axes[1, 3].imshow(mean_sp / (mean_sp.max() or 1), cmap="viridis", vmin=0, vmax=1)
    axes[1, 3].set_title("content h8\n20-video mean", fontsize=8)
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])
    fig.text(0.02, 0.28, "Content heads (center salient, varies by video)\nh8,h10,h14", fontsize=10, va="center")
    fig.suptitle(f"Encoder L{last}: stable vs content heads (same head, different videos)", fontsize=12)
    fig.tight_layout(rect=[0.08, 0, 1, 0.95])
    fig.savefig(out_dir / "explain_1_stable_vs_content.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recency(rows, last, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    slots = np.arange(16)
    # all heads mean temporal
    all_temp = np.stack([np.mean([r["heads"][h]["temporal"] for r in rows], axis=0) for h in range(16)])
    mean_t = all_temp.mean(axis=0)
    std_t = all_temp.std(axis=0)
    axes[0].fill_between(slots, mean_t - std_t, mean_t + std_t, alpha=0.25, color="#4CAF50")
    axes[0].plot(slots, mean_t, "o-", color="#2E7D32", lw=2)
    axes[0].axvline(10.5, color="red", ls="--", lw=1, label="recent cutoff (slot≥11)")
    axes[0].set_xlabel("Temporal slot (0=earliest, 15=latest)")
    axes[0].set_ylabel("Key attention mass (normalized)")
    axes[0].set_title("All 16 heads: where keys are attended (avg over 20 videos)")
    axes[0].legend(fontsize=8)

    recent_mass = [np.mean([r["heads"][h]["recent_mass"] for r in rows]) for h in range(16)]
    colors = ["#2E7D32" if h in STABLE else "#F44336" if h in CONTENT else "#FF9800" for h in range(16)]
    axes[1].bar(range(16), recent_mass, color=colors)
    axes[1].axhline(np.mean(recent_mass), color="gray", ls="--", label=f"mean={np.mean(recent_mass):.2f}")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Mass in latest 30% slots (11–15)")
    axes[1].set_title("Recency bias per head")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Encoder L{last}: temporal recency bias", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "explain_2_recency_bias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_edge_center(rows, last, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(16)
    edge = [np.mean([r["heads"][h]["edge_mass"] for r in rows]) for h in range(16)]
    center = [np.mean([r["heads"][h]["center_mass"] for r in rows]) for h in range(16)]
    w = 0.35
    ax.bar(x - w / 2, edge, w, label="border/corner mass", color="#1565C0")
    ax.bar(x + w / 2, center, w, label="center mass", color="#E65100")
    for h in STABLE:
        ax.axvspan(h - 0.5, h + 0.5, alpha=0.08, color="green")
    for h in CONTENT:
        ax.axvspan(h - 0.5, h + 0.5, alpha=0.08, color="red")
    ax.set_xlabel("Head (green band=stable, red band=content)")
    ax.set_ylabel("Fraction of spatial attention mass")
    ax.set_title(f"Encoder L{last}: border vs center (20 videos avg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "explain_3_border_vs_center.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-sample", type=int, default=20)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--anticipation", type=float, default=1.0)
    p.add_argument("--val-csv", default="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv")
    p.add_argument("--video-root", default="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos")
    p.add_argument("--vjepa-ckpt", default="/scratch/ll5914/models/vjepa2/vitl.pt")
    p.add_argument("--video-enc-lora", default="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/encoder_lora_best.pt")
    p.add_argument("--video-pred-lora", default="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt")
    p.add_argument("--out", default=str(PROJECT_ROOT / "experiments/fastgen_head_heatmaps/encoder_L23_multiclip"))
    args = p.parse_args()
    os.environ.setdefault("TRI_MODAL_FRAME_CACHE", "/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1")

    out_dir = Path(args.out)
    rows, grid, last = collect(args)
    plot_stable_vs_content(rows, grid, last, out_dir)
    plot_recency(rows, last, out_dir)
    plot_edge_center(rows, last, out_dir)
    print("wrote explain_*.png to", out_dir)


if __name__ == "__main__":
    main()
