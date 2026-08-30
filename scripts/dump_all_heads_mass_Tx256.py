#!/usr/bin/env python3
"""L23 all heads: attn mass as [T_slots x 256 spatial], linear, one row per head.

Default: 128 frames @ 8fps = 16s -> 64 temporal slots (tubelet=2).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/home/ll5914/Jepa_yifan/JEPA_ARVR")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
# separate cache from 32-frame dumps
os.environ.setdefault(
    "TRI_MODAL_FRAME_CACHE",
    "/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f128_at1",
)

import src.models.utils.modules as modules  # noqa: E402
from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mc", PROJECT_ROOT / "scripts/analyze_encoder_last_layer_multiclip.py"
)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

TARGET_VID = "P01_20240202-110250"
FRAMES = 128  # 16s @ 8fps
FPS = 8
TUBELET = 2
T_SLOTS = FRAMES // TUBELET  # 64
GRID = 16
GP = GRID * GRID
CHUNK = 64  # smaller: N=16384
CORNERS = {"TL": 0, "TR": 15, "BL": 240, "BR": 255}
OUT = Path(
    "/home/ll5914/Jepa_yifan/encoder_L23_hdepic_sink_example/"
    f"t_spatial_{T_SLOTS}x256_all_heads_{FRAMES}f"
)


def apply_rope(attn, q, k, T=None, Hp=None, Wp=None):
    m = attn
    _B, _H, N, _D = q.shape
    gd = int(N // (m.grid_size * m.grid_size))
    if T is None or Hp is None or Wp is None:
        mask_p = torch.arange(int(gd * m.grid_size * m.grid_size), device=q.device)
    else:
        mask_p = torch.arange(int(T * Hp * Wp), device=q.device)
    d_mask, h_mask, w_mask = m.separate_positions(mask_p, Hp, Wp)
    s = 0
    qd = rotate_queries_or_keys(q[..., s : s + m.d_dim], pos=d_mask)
    kd = rotate_queries_or_keys(k[..., s : s + m.d_dim], pos=d_mask)
    s += m.d_dim
    qh = rotate_queries_or_keys(q[..., s : s + m.h_dim], pos=h_mask)
    kh = rotate_queries_or_keys(k[..., s : s + m.h_dim], pos=h_mask)
    s += m.h_dim
    qw = rotate_queries_or_keys(q[..., s : s + m.w_dim], pos=w_mask)
    kw = rotate_queries_or_keys(k[..., s : s + m.w_dim], pos=w_mask)
    s += m.w_dim
    if s < m.head_dim:
        return (
            torch.cat([qd, qh, qw, q[..., s:]], dim=-1),
            torch.cat([kd, kh, kw, k[..., s:]], dim=-1),
        )
    return torch.cat([qd, qh, qw], dim=-1), torch.cat([kd, kh, kw], dim=-1)


def attn_mass_from_qk(q, k, scale, chunk=CHUNK):
    h, n, _ = q.shape
    out = torch.zeros(h, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)
    for ci in range(0, n, chunk):
        logits = torch.bmm(q[:, ci : ci + chunk].float(), k_t) * scale
        out += logits.softmax(dim=-1).sum(dim=1)
        if ci % (chunk * 32) == 0:
            print(f"  mass progress {ci}/{n}", flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRI_MODAL_FRAME_CACHE"]).mkdir(parents=True, exist_ok=True)

    class A:
        pass

    args = A()
    args.n_sample = 1
    args.chunk = CHUNK
    args.frames = FRAMES
    args.fps = FPS
    args.img_size = 256
    args.anticipation = 1.0
    args.clips_dir = ""
    args.val_csv = (
        "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/"
        "clip_split/HD_EPIC_val_vjepa.csv"
    )
    args.video_root = "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
    args.vjepa_ckpt = "/scratch/ll5914/models/vjepa2/vitl.pt"
    args.video_enc_lora = (
        "/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/"
        "action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/"
        "encoder_lora_best.pt"
    )
    args.video_pred_lora = (
        "/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/"
        "action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/"
        "predictor_lora_best.pt"
    )

    df = pd.read_csv(args.val_csv)
    rows = df[df["video_id"].astype(str) == TARGET_VID]
    if rows.empty:
        raise RuntimeError(f"{TARGET_VID} not in csv")
    tmp = Path(tempfile.gettempdir()) / f"sink_16s_{os.getpid()}.csv"
    rows.head(1).to_csv(tmp, index=False)
    args.val_csv = str(tmp)

    device = torch.device("cuda")
    print(f"loading {FRAMES} frames @ {FPS}fps ≈ {FRAMES/FPS:.1f}s -> {T_SLOTS} slots", flush=True)
    samples = mc.load_diverse_clips(args)
    tmp.unlink(missing_ok=True)
    clip, meta = samples[0]
    vid = meta["video_id"]
    assert clip.shape[1] == FRAMES, clip.shape
    print("clip", tuple(clip.shape), "vid", vid, "indices span", meta.get("frame_indices", [])[:3], "...", flush=True)

    model = mc._fastgen.load_video_model(args, device)
    core = model.base_model if hasattr(model, "base_model") else model
    encoder = core.encoder
    last = len(encoder.blocks) - 1
    block = encoder.blocks[last]
    attn = block.attn
    bag = {}

    orig = block.forward

    def wrapped(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        # Capture L23 Q/K mass only. Skip real block + predictor: long clips
        # (N=16384) can trip RoPE/index asserts after we already have mass.
        with torch.no_grad():
            qkv = attn.qkv(block.norm1(x)).unflatten(-1, (3, attn.num_heads, -1)).permute(
                2, 0, 3, 1, 4
            )
            q, k = apply_rope(attn, qkv[0], qkv[1], T, H_patches, W_patches)
            print(f"L{last} q shape {tuple(q.shape)} computing mass...", flush=True)
            mass_t = attn_mass_from_qk(q[0], k[0], float(attn.scale))
            bag["mass"] = mass_t.float().cpu()
            bag["N"] = int(x.shape[1])
            bag["H"] = int(attn.num_heads)
            print("mass captured on CPU; skipping L23 attn/mlp + predictor", flush=True)
        raise RuntimeError("MASS_CAPTURE_DONE")

    block.forward = wrapped
    clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
    ant = torch.full((1,), 1.0, device=device)
    print("forward...", flush=True)
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = model(clip_b, ant)
    except RuntimeError as exc:
        if "MASS_CAPTURE_DONE" not in str(exc):
            block.forward = orig
            raise
    finally:
        block.forward = orig

    if "mass" not in bag:
        raise RuntimeError("failed to capture L23 attention mass")
    mass = bag["mass"].numpy()
    H, N = mass.shape
    assert N == T_SLOTS * GP, (N, T_SLOTS * GP)
    M = mass.reshape(H, T_SLOTS, GP)
    np.save(OUT / f"all_heads_mass_Hx{T_SLOTS}x256.npy", M)

    lines = [
        f"video={vid} L{last} frames={FRAMES} fps={FPS} dur≈{FRAMES/FPS:.1f}s "
        f"slots={T_SLOTS} N={N} | I_j [T x 256] LINEAR",
        "",
        f"{'head':>4s} {'TL0':>8s} {'TR15':>8s} {'BL240':>8s} {'BR255':>8s} {'other':>8s} {'c/o':>7s}",
    ]
    for h in range(H):
        mh = M[h]
        cvals = [float(mh[:, CORNERS[n]].mean()) for n in ("TL", "TR", "BL", "BR")]
        mask = np.ones(GP, dtype=bool)
        for idx in CORNERS.values():
            mask[idx] = False
        other = float(mh[:, mask].mean())
        ratio = float(np.mean(cvals) / (other + 1e-12))
        lines.append(
            f"{h:4d} {cvals[0]:8.3f} {cvals[1]:8.3f} {cvals[2]:8.3f} {cvals[3]:8.3f} "
            f"{other:8.3f} {ratio:7.2f}"
        )
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # one row per head, wide + tall (T=64 is 4x previous)
    fig_w = 20
    fig_h = 2.2 * H + 1.0
    fig, axes = plt.subplots(H, 1, figsize=(fig_w, fig_h), sharex=True)
    if H == 1:
        axes = [axes]
    for h in range(H):
        ax = axes[h]
        mh = M[h]
        vmax = float(np.percentile(mh, 99))
        im = ax.imshow(mh, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
        for name, idx in CORNERS.items():
            ax.axvline(idx, color="cyan", lw=0.6, alpha=0.8)
            if h == 0:
                ax.text(idx, -1.5, name, color="cyan", ha="center", fontsize=8, fontweight="bold")
        ax.set_yticks([0, 16, 32, 48, 63])
        ax.set_ylabel(f"h{h}\nt", fontsize=9, rotation=0, labelpad=22, va="center")
        cbar = plt.colorbar(im, ax=ax, fraction=0.01, pad=0.006)
        cbar.ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("spatial index 0..255  (TL=0, TR=15, BL=240, BR=255)", fontsize=11)
    fig.suptitle(
        f"{vid}  L{last}  {FRAMES}f@{FPS}fps≈{FRAMES/FPS:.0f}s  "
        f"attn mass I_j [{T_SLOTS} time × 256 spatial] LINEAR\n"
        "one row = one head; cyan = corners",
        fontsize=13,
        y=0.997,
    )
    fig.tight_layout(rect=[0.02, 0.01, 1, 0.98])
    out_png = OUT / f"all_heads_mass_{T_SLOTS}x256_linear.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print("wrote", out_png, flush=True)

    per = OUT / "per_head"
    per.mkdir(exist_ok=True)
    for h in range(H):
        mh = M[h]
        vmax = float(np.percentile(mh, 99))
        fig, ax = plt.subplots(figsize=(18, 4.5))
        im = ax.imshow(mh, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
        for name, idx in CORNERS.items():
            ax.axvline(idx, color="cyan", lw=0.7, alpha=0.85)
            ax.text(idx, -1.2, name, color="cyan", ha="center", fontsize=8)
        ax.set_yticks([0, 16, 32, 48, 63])
        ax.set_ylabel("time slot t")
        ax.set_xlabel("spatial index 0..255")
        ax.set_title(f"L{last} h{h} mass [{T_SLOTS}×256] linear  {FRAMES}f≈{FRAMES/FPS:.0f}s")
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="I_j")
        fig.tight_layout()
        fig.savefig(per / f"h{h:02d}_mass_{T_SLOTS}x256_linear.png", dpi=130, bbox_inches="tight")
        plt.close()
        np.save(per / f"h{h:02d}_mass_{T_SLOTS}x256.npy", mh)

    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
