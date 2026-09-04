#!/usr/bin/env python3
"""t=0, h=3: dump 256x256 attention among spatial tokens + explain alignment with numbers."""
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

OUT = Path("/home/ll5914/Jepa_yifan/encoder_L23_hdepic_sink_example/t0_h3_attn256")
GRID = 16
GP = GRID * GRID
HEAD = 3
T_SLOT = 0


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


def yx(i):
    return divmod(i, GRID)


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
            qkv = attn.qkv(block.norm1(x)).unflatten(-1, (3, attn.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q0, k0, v0 = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(attn, q0, k0, T, H_patches, W_patches)
            bag["q"] = q[0, HEAD].float().cpu()  # [N,D]
            bag["k"] = k[0, HEAD].float().cpu()
            bag["v"] = v0[0, HEAD].float().cpu()
            bag["q_pre"] = q0[0, HEAD].float().cpu()
            bag["k_pre"] = k0[0, HEAD].float().cpu()
            bag["scale"] = float(attn.scale)
            bag["N"] = int(x.shape[1])
            bag["D"] = int(q.shape[-1])
        return orig(x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)

    block.forward = wrapped
    clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
    ant = torch.full((1,), 1.0, device=device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _ = model(clip_b, ant)
    block.forward = orig

    q = bag["q"]
    k = bag["k"]
    v = bag["v"]
    scale = bag["scale"]
    n = bag["N"]
    d = bag["D"]
    print("N", n, "D", d, "scale", scale, "video", meta["video_id"])

    # indices for temporal slot t=0
    sl = slice(T_SLOT * GP, (T_SLOT + 1) * GP)
    q0 = q[sl]  # [256,D]
    k0 = k[sl]
    v0 = v[sl]
    k_pre0 = bag["k_pre"][sl]
    q_pre0 = bag["q_pre"][sl]

    # ---- 256 x 256 attention (queries in t=0, keys in t=0) ----
    logits_256 = (q0 @ k0.T) * scale  # [256,256]
    attn_256 = torch.softmax(logits_256, dim=-1)
    np.save(OUT / "t0_h3_logits_256x256.npy", logits_256.numpy())
    np.save(OUT / "t0_h3_attn_256x256.npy", attn_256.numpy())

    # Also FULL attention mass on t=0 keys from ALL queries (true sink score)
    # I_j = sum_i softmax(q_i k^T)_ij for j in t=0
    # compute in chunks on CPU float32
    I_full = torch.zeros(GP)
    for ci in range(0, n, 128):
        logits = (q[ci : ci + 128] @ k[sl].T) * scale  # [c,256]
        I_full += torch.softmax(logits, dim=-1).sum(dim=0)
    np.save(OUT / "t0_keys_mass_from_all_queries.npy", I_full.numpy())

    # Restricted: mass from only t=0 queries onto t=0 keys
    I_t0 = attn_256.sum(dim=0)  # [256]
    np.save(OUT / "t0_keys_mass_from_t0_queries.npy", I_t0.numpy())

    corners = {"TL": 0, "TR": 15, "BL": 15 * 16, "BR": 15 * 16 + 15}
    center = 8 * 16 + 8

    lines = []
    lines.append(f"video={meta['video_id']}  L{last} h{HEAD}  t={T_SLOT}")
    lines.append(f"Within t=0 there are 256 tokens (16x16). Full video has N={n}.")
    lines.append("")
    lines.append("WHAT 'ALIGNMENT' MEANS (concrete):")
    lines.append("  attention logit_ij = (q_i · k_j) / sqrt(d)")
    lines.append("  If q_i and k_j point in similar directions, dot product is LARGE")
    lines.append("  -> after softmax, A_ij is LARGE. That is 'aligned'.")
    lines.append("  It is NOT about ||k_j|| being big (center can have bigger ||k|| but low dot).")
    lines.append("")

    # Pick a few query positions and print attention to corners
    query_ids = {
        "Q_center": center,
        "Q_TL": corners["TL"],
        "Q_BR": corners["BR"],
        "Q_mid_edge": 8 * 16 + 0,  # left mid
    }
    lines.append("Example: for specific queries, print logits & attn to 4 corners + center key:")
    for qname, qi in query_ids.items():
        lines.append(f"  -- {qname} at yx={yx(qi)} --")
        for kname, kj in list(corners.items()) + [("CTR", center)]:
            logit = float(logits_256[qi, kj])
            a = float(attn_256[qi, kj])
            # also cos
            cos = float(torch.nn.functional.cosine_similarity(q0[qi:qi+1], k0[kj:kj+1]).item())
            lines.append(
                f"    key {kname:3s} yx={yx(kj)}  q·k/√d={logit:+8.3f}  cos(q,k)={cos:+.3f}  A={a:.4f}"
            )
        # where does this query put most mass among ALL 256 keys?
        top = int(attn_256[qi].argmax())
        lines.append(
            f"    -> argmax key among 256 = yx={yx(top)} A={float(attn_256[qi,top]):.4f}"
            + (" CORNER" if top in corners.values() else "")
        )
        lines.append("")

    lines.append("Column sum (how much total attention each KEY receives from 256 t=0 queries):")
    for kname, kj in list(corners.items()) + [("CTR", center)]:
        lines.append(
            f"  key {kname}  I_from_t0Q={float(I_t0[kj]):.3f}  "
            f"I_from_allQ={float(I_full[kj]):.3f}  ||k||={float(k0[kj].norm()):.3f}"
        )
    other = [i for i in range(GP) if i not in corners.values()]
    lines.append(
        f"  other_mean I_from_t0Q={float(I_t0[other].mean()):.3f}  "
        f"BR/other={float(I_t0[corners['BR']]/(I_t0[other].mean()+1e-12)):.2f}x"
    )
    lines.append("")
    lines.append("WHY BR wins at t=0:")
    lines.append("  Look at almost any query: logit to BR is higher than to TL/TR/BL/center")
    lines.append("  because q is more cosine-aligned with k_BR than with other keys.")
    lines.append("  Softmax then puts most of that query's mass on BR -> BR column lights up.")

    (OUT / "alignment_explained.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- Figures ----
    # 1) 256x256 attention heatmap (log scale for visibility)
    A = attn_256.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    im0 = axes[0].imshow(A, cmap="viridis", aspect="auto")
    axes[0].set_title("256×256 attention A[i,j]\nrows=query yx, cols=key yx (t=0 only)")
    axes[0].set_xlabel("key index 0..255 (row-major yx)")
    axes[0].set_ylabel("query index 0..255")
    # mark corner key columns
    for name, kj in corners.items():
        axes[0].axvline(kj, color="r", lw=0.8, alpha=0.7)
        axes[0].text(kj, -8, name, color="r", ha="center", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.log10(A + 1e-8), cmap="viridis", aspect="auto")
    axes[1].set_title("same matrix, log10(A) (see BR column)")
    for name, kj in corners.items():
        axes[1].axvline(kj, color="r", lw=0.8, alpha=0.7)
        axes[1].text(kj, -8, name, color="r", ha="center", fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(f"t=0 h3: full attention among 256 spatial tokens — {meta['video_id']}")
    fig.tight_layout()
    fig.savefig(OUT / "t0_h3_attn_256x256.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2) Key maps: ||k||, mass from t0 Q, mass from all Q; and a mean-query alignment map
    q_mean = q0.mean(dim=0, keepdim=True)  # [1,D]
    align = ((q_mean @ k0.T).squeeze(0) * scale).numpy().reshape(GRID, GRID)  # mean-q logit to each key
    kn = k0.norm(dim=-1).numpy().reshape(GRID, GRID)
    mass_t0 = I_t0.numpy().reshape(GRID, GRID)
    mass_all = I_full.numpy().reshape(GRID, GRID)
    # value norm too (user asked key value)
    vn = v0.norm(dim=-1).numpy().reshape(GRID, GRID)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    panels = [
        (axes[0, 0], kn, "||k|| map (t=0 keys)", False),
        (axes[0, 1], vn, "||v|| map (t=0 values)", False),
        (axes[0, 2], align, "alignment: mean(q)·k /√d\n(THIS predicts sink)", False),
        (axes[1, 0], mass_t0, "key mass I_j from t=0 queries\nΣ_i A_ij (256 queries)", False),
        (axes[1, 1], mass_all, "key mass I_j from ALL queries\n(true sink score)", False),
        (axes[1, 2], A.mean(axis=0).reshape(GRID, GRID), "column-mean of 256×256 A", False),
    ]
    for ax, m, title, signed in panels:
        im = ax.imshow(m, cmap="viridis")
        for name, (y, x) in {"TL": (0, 0), "TR": (0, 15), "BL": (15, 0), "BR": (15, 15)}.items():
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec="r", lw=1.5))
            ax.text(x, y, f"{name}\n{m[y,x]:.2f}", color="w", ha="center", va="center", fontsize=7, fontweight="bold")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Keys/values vs ALIGNMENT vs attention mass — why BR wins", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "t0_h3_key_value_align_mass.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3) One query walkthrough: center query attention map over 256 keys
    qi = center
    row = attn_256[qi].numpy().reshape(GRID, GRID)
    row_logit = logits_256[qi].numpy().reshape(GRID, GRID)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, m, title in [
        (axes[0], row_logit, f"ONE query at center (8,8): logits to all 256 keys"),
        (axes[1], row, f"same query: softmax attention A[center, :]"),
    ]:
        im = ax.imshow(m, cmap="viridis")
        for name, (y, x) in {"TL": (0, 0), "TR": (0, 15), "BL": (15, 0), "BR": (15, 15)}.items():
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec="r", lw=1.5))
            ax.text(x, y, f"{m[y,x]:.3f}", color="w", ha="center", va="center", fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Alignment in one step: center query prefers BR key (high logit -> high A)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "t0_h3_one_query_center_to_keys.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # save small CSV of attn from center query
    np.savetxt(OUT / "centerQ_attn_to_256keys_map16.csv", row, fmt="%.6f", delimiter=",")
    np.savetxt(OUT / "centerQ_logits_to_256keys_map16.csv", row_logit, fmt="%.6f", delimiter=",")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
