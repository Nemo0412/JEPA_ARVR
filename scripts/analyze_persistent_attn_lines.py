#!/usr/bin/env python3
"""Find persistent bright columns in [T x 256] mass maps — what are they, do they track video?

1) Detect columns that stay bright across many time slots (the 'vertical lines').
2) Map flat index -> (y,x) and overlay those patches on RGB frames at several t.
3) Compare two videos (+ optional roll) to see if the same spatial IDs stay bright.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path("/home/ll5914/Jepa_yifan/JEPA_ARVR")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
os.environ.setdefault(
    "TRI_MODAL_FRAME_CACHE",
    "/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1",
)

from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mc", PROJECT_ROOT / "scripts/analyze_encoder_last_layer_multiclip.py"
)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

OUT = Path("/home/ll5914/Jepa_yifan/encoder_L23_hdepic_sink_example/persistent_lines")
MASS64 = Path(
    "/home/ll5914/Jepa_yifan/encoder_L23_hdepic_sink_example/"
    "t_spatial_64x256_all_heads_128f/all_heads_mass_Hx64x256.npy"
)
VID_A = "P01_20240202-110250"
VID_B = "P01_20240202-161354"  # different clip, same participant
GRID = 16
GP = 256
PATCH = 16
IMG = 256
CHUNK = 128


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
    return out


def capture_mass(model, encoder, clip_b, ant):
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
            bag["mass"] = attn_mass_from_qk(q[0], k[0], float(attn.scale)).float().cpu()
        raise RuntimeError("MASS_CAPTURE_DONE")

    block.forward = wrapped
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = model(clip_b, ant)
    except RuntimeError as e:
        if "MASS_CAPTURE_DONE" not in str(e):
            block.forward = orig
            raise
    finally:
        block.forward = orig
    return bag["mass"].numpy()


def detect_persistent_cols(M_tx256: np.ndarray, top_k=12, bright_q=0.90):
    """M: [T,256]. Score = mean * (#slots above per-row quantile)."""
    T, S = M_tx256.shape
    thr = np.quantile(M_tx256, bright_q, axis=1, keepdims=True)  # [T,1]
    above = (M_tx256 >= thr).astype(np.float64)
    frac = above.mean(axis=0)  # how often this col is among brightest
    mean = M_tx256.mean(axis=0)
    # combine: must be frequently bright AND high mean
    score = mean * (0.25 + frac)
    order = np.argsort(-score)
    cols = []
    for j in order[:top_k]:
        y, x = divmod(int(j), GRID)
        cols.append(
            {
                "spat": int(j),
                "y": int(y),
                "x": int(x),
                "mean_I": float(mean[j]),
                "frac_bright": float(frac[j]),
                "score": float(score[j]),
                "edge": bool(y in (0, GRID - 1) or x in (0, GRID - 1)),
                "corner": bool(y in (0, GRID - 1) and x in (0, GRID - 1)),
            }
        )
    return cols, score


def denorm_frame(clip_cthw: torch.Tensor, t_frame: int) -> np.ndarray:
    """Approximate RGB uint8 from normalized clip [C,T,H,W]."""
    # VideoTransform typically uses ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = clip_cthw[:, t_frame].float().cpu().numpy()
    x = np.clip(x * std + mean, 0, 1)
    return (x.transpose(1, 2, 0) * 255).astype(np.uint8)


def overlay_patches(rgb: np.ndarray, cols, color=(0, 255, 255), lw=2):
    out = rgb.copy()
    for c in cols:
        y0, x0 = c["y"] * PATCH, c["x"] * PATCH
        y1, x1 = y0 + PATCH - 1, x0 + PATCH - 1
        out[y0 : y0 + lw, x0 : x1 + 1] = color
        out[y1 - lw + 1 : y1 + 1, x0 : x1 + 1] = color
        out[y0 : y1 + 1, x0 : x0 + lw] = color
        out[y0 : y1 + 1, x1 - lw + 1 : x1 + 1] = color
    return out


def load_one_clip(video_id: str, frames=32, fps=8):
    class A:
        pass

    args = A()
    args.n_sample = 1
    args.chunk = CHUNK
    args.frames = frames
    args.fps = fps
    args.img_size = IMG
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
    rows = df[df["video_id"].astype(str) == video_id].head(1)
    if rows.empty:
        raise RuntimeError(video_id)
    tmp = Path(tempfile.gettempdir()) / f"pline_{os.getpid()}_{video_id}.csv"
    rows.to_csv(tmp, index=False)
    args.val_csv = str(tmp)
    samples = mc.load_diverse_clips(args)
    tmp.unlink(missing_ok=True)
    return samples[0], args


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M_all = np.load(MASS64)  # [H,64,256]
    H, T, S = M_all.shape
    assert S == 256

    # ---- 1) detect lines on existing 16s dump ----
    lines_doc = [f"Persistent bright columns from 64x256 mass (video A={VID_A}, 128f/16s)", ""]
    head_cols = {}
    for h in range(H):
        cols, score = detect_persistent_cols(M_all[h], top_k=10)
        head_cols[h] = cols
        # keep those with frac_bright >= 0.25 as 'lines'
        lines = [c for c in cols if c["frac_bright"] >= 0.25]
        lines_doc.append(
            f"h{h:02d} persistent(frac>=0.25): "
            + ", ".join(
                f"id={c['spat']} yx=({c['y']},{c['x']}) mean={c['mean_I']:.2f} "
                f"frac={c['frac_bright']:.2f}{' CORNER' if c['corner'] else (' EDGE' if c['edge'] else '')}"
                for c in lines[:8]
            )
        )
        np.save(OUT / f"h{h:02d}_col_score.npy", score)

    # consensus: columns that are in top-10 persistent for many heads
    vote = np.zeros(256, dtype=np.int32)
    for h, cols in head_cols.items():
        for c in cols[:8]:
            if c["frac_bright"] >= 0.20:
                vote[c["spat"]] += 1
    consensus = [
        {
            "spat": int(j),
            "y": int(j // GRID),
            "x": int(j % GRID),
            "n_heads": int(vote[j]),
            "edge": bool((j // GRID) in (0, 15) or (j % GRID) in (0, 15)),
            "corner": bool((j // GRID) in (0, 15) and (j % GRID) in (0, 15)),
        }
        for j in np.argsort(-vote)[:20]
        if vote[j] >= 3
    ]
    lines_doc.append("")
    lines_doc.append("Consensus spatial IDs (in top persistent of >=3 heads):")
    for c in consensus:
        tag = "CORNER" if c["corner"] else ("EDGE" if c["edge"] else "INTERIOR")
        lines_doc.append(
            f"  spat={c['spat']:3d} yx=({c['y']:2d},{c['x']:2d}) heads={c['n_heads']:2d}  {tag}"
        )

    # ---- 2) overlay on frames of video A (reuse 32f load for RGB; also show mid of 16s if we load 128) ----
    device = torch.device("cuda")
    (clip_a, meta_a), args = load_one_clip(VID_A, frames=32, fps=8)
    model = mc._fastgen.load_video_model(args, device)
    core = model.base_model if hasattr(model, "base_model") else model
    encoder = core.encoder
    ant = torch.full((1,), 1.0, device=device)

    # use h3 lines from 64-slot dump for overlay
    h3_lines = [c for c in head_cols[3] if c["frac_bright"] >= 0.25][:8]
    cons_for_overlay = [
        {"y": c["y"], "x": c["x"], "spat": c["spat"]} for c in consensus[:8]
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, tf in zip(axes, [0, 8, 16, 31]):
        rgb = denorm_frame(clip_a, tf)
        ov = overlay_patches(rgb, cons_for_overlay, color=(0, 255, 255))
        ax.imshow(ov)
        ax.set_title(f"A frame≈{tf} (of 32f preview)")
        ax.axis("off")
    fig.suptitle(
        "Cyan boxes = consensus persistent spatial IDs from 16s/64-slot mass\n"
        "(same token IDs overlaid on preview frames — if boxes sit on different objects over time, ID≠content)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "overlay_consensus_on_videoA.png", dpi=140, bbox_inches="tight")
    plt.close()

    # also plot h3 64x256 with top line columns marked
    fig, ax = plt.subplots(figsize=(16, 4.2))
    Mh = M_all[3]
    vmax = np.percentile(Mh, 99)
    im = ax.imshow(Mh, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    for c in h3_lines[:6]:
        ax.axvline(c["spat"], color="cyan", lw=0.9, alpha=0.9)
        ax.text(c["spat"], -1.2, f"({c['y']},{c['x']})", color="cyan", ha="center", fontsize=7, rotation=90)
    ax.set_ylabel("t (0..63)")
    ax.set_xlabel("spatial 0..255")
    ax.set_title("h3 64×256 with detected persistent columns (cyan)")
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    fig.tight_layout()
    fig.savefig(OUT / "h3_64x256_lines_marked.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ---- 3) second video + rolled A: do the SAME spatial IDs light up? ----
    print("mass video A 32f...", flush=True)
    mass_a = capture_mass(
        model, encoder, clip_a.unsqueeze(0).to(device=device, dtype=torch.float32), ant
    )
    Ma = mass_a.reshape(mass_a.shape[0], -1, GP)  # [H,T,256]

    print("mass video B...", flush=True)
    (clip_b, meta_b), _ = load_one_clip(VID_B, frames=32, fps=8)
    mass_b = capture_mass(
        model, encoder, clip_b.unsqueeze(0).to(device=device, dtype=torch.float32), ant
    )
    Mb = mass_b.reshape(mass_b.shape[0], -1, GP)

    print("mass video A rolled half-grid...", flush=True)
    clip_r = torch.roll(clip_a, shifts=(8 * PATCH, 8 * PATCH), dims=(-2, -1))
    mass_r = capture_mass(
        model, encoder, clip_r.unsqueeze(0).to(device=device, dtype=torch.float32), ant
    )
    Mr = mass_r.reshape(mass_r.shape[0], -1, GP)

    def top_set(M_htS, h, k=10):
        cols, _ = detect_persistent_cols(M_htS[h], top_k=k)
        return {c["spat"] for c in cols if c["frac_bright"] >= 0.20}

    # compare across heads
    lines_doc.append("")
    lines_doc.append("=== Do persistent line IDs track video content? (32f / 16 slots) ===")
    lines_doc.append(
        "Jaccard of top persistent spatial IDs: high A↔B => ID/geometry; "
        "low A↔roll with content-follow would need tracking (here: if A↔roll still high => ID stays)."
    )
    jac_ab, jac_ar = [], []
    for h in range(H):
        sa, sb, sr = top_set(Ma, h), top_set(Mb, h), top_set(Mr, h)
        jab, jar = jaccard(sa, sb), jaccard(sa, sr)
        jac_ab.append(jab)
        jac_ar.append(jar)
        if h in (1, 3, 8, 12):
            lines_doc.append(
                f"  h{h}: A={sorted(sa)[:8]} B={sorted(sb)[:8]} roll={sorted(sr)[:8]} "
                f"| J(A,B)={jab:.2f} J(A,roll)={jar:.2f}"
            )

    lines_doc.append("")
    lines_doc.append(
        f"Mean Jaccard over 16 heads: A vs different video B = {np.mean(jac_ab):.3f}; "
        f"A vs spatially-rolled A = {np.mean(jac_ar):.3f}"
    )
    # interpretation
    m_ab, m_ar = float(np.mean(jac_ab)), float(np.mean(jac_ar))
    lines_doc.append("")
    if m_ab >= 0.35 and m_ar >= 0.35:
        lines_doc.append(
            "VERDICT: bright vertical lines are mostly FIXED SPATIAL TOKEN IDs / grid positions "
            "(similar IDs light up on different video AND after rolling content away). "
            "They are NOT locked to a particular object in the video."
        )
    elif m_ab < 0.25 and m_ar < 0.25:
        lines_doc.append(
            "VERDICT: lines CHANGE with video / roll — more CONTENT-driven which tokens stay bright."
        )
    else:
        lines_doc.append(
            f"VERDICT: MIXED (A↔B={m_ab:.2f}, A↔roll={m_ar:.2f}). "
            "Some heads stick to geometry IDs; content still modulates which IDs win."
        )

    # plot side-by-side 16x256 for h3 A / B / roll
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    for ax, M, title in zip(
        axes,
        [Ma[3], Mb[3], Mr[3]],
        [f"A {VID_A}", f"B {VID_B}", "A rolled +8,+8 patches"],
    ):
        vmax = np.percentile(M, 99)
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        cols, _ = detect_persistent_cols(M, top_k=6)
        for c in cols:
            if c["frac_bright"] >= 0.2:
                ax.axvline(c["spat"], color="cyan", lw=0.7, alpha=0.85)
        ax.set_ylabel(title + "\nt")
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    axes[0].set_title("h3 persistent columns (cyan) — same IDs across videos? after roll?")
    axes[-1].set_xlabel("spatial index 0..255")
    fig.tight_layout()
    fig.savefig(OUT / "h3_lines_A_vs_B_vs_roll.png", dpi=140, bbox_inches="tight")
    plt.close()

    # overlay B vs A consensus
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, clip, title in zip(
        axes, [clip_a, clip_b], [f"A {VID_A}", f"B {VID_B}"]
    ):
        rgb = denorm_frame(clip, 16)
        # use that video's own h3 persistent cols
        M = Ma[3] if title.startswith("A") else Mb[3]
        cols, _ = detect_persistent_cols(M, top_k=8)
        cols = [c for c in cols if c["frac_bright"] >= 0.2][:8]
        ov = overlay_patches(rgb, cols)
        ax.imshow(ov)
        ax.set_title(title + "\n" + ", ".join(f"({c['y']},{c['x']})" for c in cols))
        ax.axis("off")
    fig.suptitle("Persistent line patches on mid-frame (cyan) — compare A vs B", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "overlay_A_vs_B_persistent.png", dpi=140, bbox_inches="tight")
    plt.close()

    # save json summary
    summary = {
        "video_A_16s_consensus": consensus,
        "h3_persistent_64slot": h3_lines,
        "mean_jaccard_A_vs_B": m_ab,
        "mean_jaccard_A_vs_roll": m_ar,
        "per_head_jaccard_AB": jac_ab,
        "per_head_jaccard_Aroll": jac_ar,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    text = "\n".join(lines_doc) + "\n"
    (OUT / "verdict.txt").write_text(text)
    print(text, flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
