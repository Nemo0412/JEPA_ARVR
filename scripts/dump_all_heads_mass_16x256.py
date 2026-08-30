#!/usr/bin/env python3
"""L23 all heads: attn mass I_j as [16 time x 256 spatial], linear-scale plots only."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path("/home/ll5914/Jepa_yifan/JEPA_ARVR")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
os.environ.setdefault(
    "TRI_MODAL_FRAME_CACHE",
    "/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1",
)

import src.models.utils.modules as modules  # noqa: E402
from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mc", PROJECT_ROOT / "scripts/analyze_encoder_last_layer_multiclip.py"
)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

OUT = Path("/home/ll5914/Jepa_yifan/encoder_L23_hdepic_sink_example/t_spatial_16x256_all_heads")
TARGET_VID = "P01_20240202-110250"
GRID = 16
GP = GRID * GRID
T_SLOTS = 16
CHUNK = 128
CORNERS = {"TL": 0, "TR": 15, "BL": 240, "BR": 255}


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
    """q,k [H,N,D] -> [H,N] mass."""
    h, n, _ = q.shape
    out = torch.zeros(h, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)
    for ci in range(0, n, chunk):
        logits = torch.bmm(q[:, ci : ci + chunk].float(), k_t) * scale
        out += logits.softmax(dim=-1).sum(dim=1)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    class A:
        pass

    args = A()
    args.n_sample = 1
    args.chunk = 128
    args.frames = 32
    args.fps = 8
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
    # force same clip as prior dumps
    import pandas as pd
    import tempfile

    df = pd.read_csv(args.val_csv)
    rows = df[df["video_id"].astype(str) == TARGET_VID]
    if rows.empty:
        raise RuntimeError(f"{TARGET_VID} not in {args.val_csv}")
    tmp = Path(tempfile.gettempdir()) / f"sink_oneclip_{os.getpid()}.csv"
    rows.head(1).to_csv(tmp, index=False)
    args.val_csv = str(tmp)
    args.n_sample = 1

    device = torch.device("cuda")
    samples = mc.load_diverse_clips(args)
    tmp.unlink(missing_ok=True)
    clip, meta = samples[0]
    vid = meta["video_id"]
    assert vid == TARGET_VID, vid
    model = mc._fastgen.load_video_model(args, device)
    core = model.base_model if hasattr(model, "base_model") else model
    encoder = core.encoder
    last = len(encoder.blocks) - 1
    block = encoder.blocks[last]
    attn = block.attn
    bag = {}

    orig = block.forward

    def wrapped(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        with torch.no_grad():
            qkv = attn.qkv(block.norm1(x)).unflatten(-1, (3, attn.num_heads, -1)).permute(
                2, 0, 3, 1, 4
            )
            q, k = apply_rope(attn, qkv[0], qkv[1], T, H_patches, W_patches)
            bag["mass"] = attn_mass_from_qk(q[0], k[0], float(attn.scale))
            bag["N"] = int(x.shape[1])
            bag["H"] = int(attn.num_heads)
        return orig(
            x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches
        )

    block.forward = wrapped
    clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
    ant = torch.full((1,), 1.0, device=device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _ = model(clip_b, ant)
    block.forward = orig

    mass = bag["mass"].float().cpu().numpy()  # [H, N]
    H, N = mass.shape
    assert N == T_SLOTS * GP, (N, T_SLOTS * GP)
    M = mass.reshape(H, T_SLOTS, GP)
    np.save(OUT / "all_heads_mass_Hx16x256.npy", M)

    lines = [
        f"video={vid} L{last} all {H} heads | I_j mass [16 t x 256 spat] LINEAR only",
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
    print("\n".join(lines))

    ncols = 4
    nrows = (H + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows), squeeze=False)
    for h in range(nrows * ncols):
        ax = axes[h // ncols][h % ncols]
        if h >= H:
            ax.axis("off")
            continue
        mh = M[h]
        vmax = float(np.percentile(mh, 99))
        im = ax.imshow(mh, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
        for idx in CORNERS.values():
            ax.axvline(idx, color="cyan", lw=0.5, alpha=0.75)
        ax.set_title(f"h{h}", fontsize=10)
        ax.set_yticks([0, 5, 10, 15])
        if h % ncols == 0:
            ax.set_ylabel("t")
        if h // ncols == nrows - 1:
            ax.set_xlabel("spatial 0..255")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.suptitle(
        f"{vid}  L{last} attn mass I_j  [16×256] LINEAR (per-head p99 vmax)\n"
        "cyan lines: TL=0 TR=15 BL=240 BR=255",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "all_heads_mass_16x256_linear.png", dpi=150, bbox_inches="tight")
    plt.close()

    per = OUT / "per_head"
    per.mkdir(exist_ok=True)
    for h in range(H):
        mh = M[h]
        vmax = float(np.percentile(mh, 99))
        fig, ax = plt.subplots(figsize=(12, 3.0))
        im = ax.imshow(mh, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
        for name, idx in CORNERS.items():
            ax.axvline(idx, color="cyan", lw=0.7, alpha=0.85)
            ax.text(idx, -0.7, name, color="cyan", ha="center", fontsize=8)
        ax.set_yticks(range(T_SLOTS))
        ax.set_ylabel("time slot t")
        ax.set_xlabel("spatial index 0..255")
        ax.set_title(f"L{last} h{h} mass [16×256] linear  vmax=p99={vmax:.2f}")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="I_j")
        fig.tight_layout()
        fig.savefig(per / f"h{h:02d}_mass_16x256_linear.png", dpi=140, bbox_inches="tight")
        plt.close()
        np.save(per / f"h{h:02d}_mass_16x256.npy", mh)

    print("wrote", OUT)


if __name__ == "__main__":
    main()
