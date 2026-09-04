#!/usr/bin/env python3
"""Reconcile RoPE vs corner-sink: three conditions on same HD-EPIC clip.

A) full RoPE (normal forward)
B) ablate RoPE only at L23  (previous 'pre-RoPE' — may still show corners)
C) ablate RoPE in ALL encoder blocks (identity rotate) — closer to colleague

Print corner/other mass ratio + spatial maps for h3.
"""
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
from src.models.utils.modules import rotate_queries_or_keys as ROTATE_ORIG  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mc", PROJECT_ROOT / "scripts/analyze_encoder_last_layer_multiclip.py"
)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

# also patch analyze_fastgen if it imported rotate by name
import scripts  # noqa: F401

OUT = Path("/home/ll5914/Jepa_yifan/encoder_L23_hdepic_sink_example/rope_ablation")
GRID = 16
GP = GRID * GRID
HEAD = 3
CHUNK = 128


def identity_rotate(x, pos):
    return x


def corner_idx(n):
    t = n // GP
    out = []
    for ti in range(t):
        b = ti * GP
        for y, x in ((0, 0), (0, 15), (15, 0), (15, 15)):
            out.append(b + y * GRID + x)
    return np.array(out)


def attn_mass_from_qk(q, k, scale, chunk=CHUNK):
    """q,k [H,N,D] -> [H,N] mass."""
    h, n, _ = q.shape
    out = torch.zeros(h, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)
    for ci in range(0, n, chunk):
        logits = torch.bmm(q[:, ci : ci + chunk].float(), k_t) * scale
        out += logits.softmax(dim=-1).sum(dim=1)
    return out


def apply_rope_mod(attn, q, k, T, Hp, Wp, use_rope: bool):
    if not use_rope:
        return q, k
    # reuse RoPEAttention path via modules.rotate
    m = attn
    _B, _H, N, _D = q.shape
    gd = int(N // (m.grid_size * m.grid_size))
    if T is None or Hp is None or Wp is None:
        mask_p = torch.arange(int(gd * m.grid_size * m.grid_size), device=q.device)
    else:
        mask_p = torch.arange(int(T * Hp * Wp), device=q.device)
    d_mask, h_mask, w_mask = m.separate_positions(mask_p, Hp, Wp)
    rot = modules.rotate_queries_or_keys
    s = 0
    qd = rot(q[..., s : s + m.d_dim], pos=d_mask)
    kd = rot(k[..., s : s + m.d_dim], pos=d_mask)
    s += m.d_dim
    qh = rot(q[..., s : s + m.h_dim], pos=h_mask)
    kh = rot(k[..., s : s + m.h_dim], pos=h_mask)
    s += m.h_dim
    qw = rot(q[..., s : s + m.w_dim], pos=w_mask)
    kw = rot(k[..., s : s + m.w_dim], pos=w_mask)
    s += m.w_dim
    if s < m.head_dim:
        return (
            torch.cat([qd, qh, qw, q[..., s:]], dim=-1),
            torch.cat([kd, kh, kw, k[..., s:]], dim=-1),
        )
    return torch.cat([qd, qh, qw], dim=-1), torch.cat([kd, kh, kw], dim=-1)


def run_condition(model, encoder, clip_b, ant, condition: str):
    """condition: full | l23_only_norope | all_norope"""
    last = len(encoder.blocks) - 1
    block = encoder.blocks[last]
    attn = block.attn
    bag = {}

    # set global rotate
    if condition == "all_norope":
        modules.rotate_queries_or_keys = identity_rotate
    else:
        modules.rotate_queries_or_keys = ROTATE_ORIG

    # Also patch any already-bound references in RoPEAttention forward by
    # temporarily replacing the function used inside modules module —
    # RoPEAttention.forward looks up rotate_queries_or_keys from modules globals.
    # Good.

    orig = block.forward

    def wrapped(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        with torch.no_grad():
            qkv = attn.qkv(block.norm1(x)).unflatten(-1, (3, attn.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q0, k0 = qkv[0], qkv[1]
            use_l23_rope = condition == "full"
            q, k = apply_rope_mod(attn, q0, k0, T, H_patches, W_patches, use_rope=use_l23_rope)
            mass = attn_mass_from_qk(q[0], k[0], float(attn.scale))
            bag["mass"] = mass
            bag["N"] = int(x.shape[1])
            # For full / all_norope we still need the real block forward under the
            # corresponding global rotate setting.
        if condition == "l23_only_norope":
            # L23 attn without rope, but keep residual path consistent:
            # run attn manually without rope then continue mlp via orig? Simpler:
            # call orig but with identity rotate only during this block.
            prev = modules.rotate_queries_or_keys
            modules.rotate_queries_or_keys = identity_rotate
            try:
                return orig(x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
            finally:
                modules.rotate_queries_or_keys = prev
        return orig(x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)

    block.forward = wrapped
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _ = model(clip_b, ant)
    block.forward = orig
    modules.rotate_queries_or_keys = ROTATE_ORIG

    mass = bag["mass"][HEAD].float().cpu().numpy()
    n = bag["N"]
    c = corner_idx(n)
    other = np.delete(mass, c)
    sp = mass.reshape(-1, GRID, GRID).mean(0)
    stats = {
        "corner_mean": float(mass[c].mean()),
        "other_mean": float(other.mean()),
        "ratio": float(mass[c].mean() / (other.mean() + 1e-12)),
        "delta": float(mass[c].mean() - other.mean()),
        "map": sp,
        "mass": mass,
        "top8": [idx_to_tyx(int(i), mass[i]) for i in np.argsort(-mass)[:8]],
    }
    return stats


def idx_to_tyx(idx, val):
    t = idx // GP
    y, x = divmod(idx % GP, GRID)
    corner = y in (0, 15) and x in (0, 15)
    return {"idx": idx, "t": t, "y": y, "x": x, "val": float(val), "corner": corner}


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

    device = torch.device("cuda")
    samples = mc.load_diverse_clips(args)
    clip, meta = samples[0]
    vid = meta["video_id"]
    model = mc._fastgen.load_video_model(args, device)
    core = model.base_model if hasattr(model, "base_model") else model
    encoder = core.encoder
    clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
    ant = torch.full((1,), 1.0, device=device)

    results = {}
    for cond in ("full", "l23_only_norope", "all_norope"):
        print("running", cond)
        results[cond] = run_condition(model, encoder, clip_b, ant, cond)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = {
        "full": "A) full RoPE",
        "l23_only_norope": "B) no RoPE only at L23\n(my previous pre-RoPE)",
        "all_norope": "C) no RoPE all encoder layers\n(colleague-like)",
    }
    for ax, cond in zip(axes, ("full", "l23_only_norope", "all_norope")):
        st = results[cond]
        im = ax.imshow(st["map"], cmap="viridis")
        ax.set_title(
            f"{titles[cond]}\nratio={st['ratio']:.2f} Δ={st['delta']:+.2f}",
            fontsize=9,
        )
        for y, x in ((0, 0), (0, 15), (15, 0), (15, 15)):
            ax.text(x, y, f"{st['map'][y,x]:.2f}", color="w", ha="center", va="center", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"RoPE ablation — L23 h{HEAD} attn mass map — {vid}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "rope_ablation_h3_maps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    lines = [f"video={vid} head=h{HEAD}", ""]
    for cond in ("full", "l23_only_norope", "all_norope"):
        st = results[cond]
        lines.append(
            f"{cond:20s} corner={st['corner_mean']:.4f} other={st['other_mean']:.4f} "
            f"ratio={st['ratio']:.3f} delta={st['delta']:+.4f}"
        )
        lines.append("  top8: " + ", ".join(
            f"(t={t['t']},y={t['y']},x={t['x']}){t['val']:.3f}{'*' if t['corner'] else ''}"
            for t in st["top8"]
        ))
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  If B≈A but C loses corners → previous conclusion was an unfair ablation;")
    lines.append("  corners need RoPE across depth (pos identity), not the last rotate alone.")
    lines.append("  If C still has corners → colleague protocol differs (e.g. train w/o RoPE).")
    text = "\n".join(lines) + "\n"
    (OUT / "rope_ablation_verdict.txt").write_text(text)
    print(text)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
