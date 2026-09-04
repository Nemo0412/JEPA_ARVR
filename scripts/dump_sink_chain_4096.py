#!/usr/bin/env python3
"""Dump full N=4096 vectors at each attention-chain stage → maps + printed numbers.

For L23 head h3 (and h8): save .npy of length-4096 scores, time-mean 16x16 maps,
print top-k (t,y,x) and the 4 corners at every stage.
Also dump one center-query's full attention row A[i,:] pre/post RoPE.
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

from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mc", PROJECT_ROOT / "scripts/analyze_encoder_last_layer_multiclip.py"
)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

OUT = Path("/home/ll5914/Jepa_yifan/encoder_L23_hdepic_sink_example/chain_4096")
GRID = 16
GP = GRID * GRID
HEADS = (3, 8)
CHUNK = 128


def apply_rope(module, q, k, T=None, H_patches=None, W_patches=None):
    m = module
    _B, _H, N, _D = q.shape
    grid_depth = int(N // (m.grid_size * m.grid_size))
    if T is None or H_patches is None or W_patches is None:
        mask_p = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=q.device)
    else:
        mask_p = torch.arange(int(T * H_patches * W_patches), device=q.device)
    d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)
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
        q2 = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
        k2 = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
    else:
        q2 = torch.cat([qd, qh, qw], dim=-1)
        k2 = torch.cat([kd, kh, kw], dim=-1)
    return q2, k2


def mean_key_logits(q, k, scale, chunk=CHUNK):
    h, n, _ = q.shape
    out = torch.zeros(h, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)
    for ci in range(0, n, chunk):
        logits = torch.bmm(q[:, ci : ci + chunk].float(), k_t) * scale
        out += logits.sum(dim=1)
    return out / float(n)


def attention_mass(q, k, scale, chunk=CHUNK):
    h, n, _ = q.shape
    out = torch.zeros(h, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)
    for ci in range(0, n, chunk):
        logits = torch.bmm(q[:, ci : ci + chunk].float(), k_t) * scale
        out += logits.softmax(dim=-1).sum(dim=1)
    return out


def one_query_attn(q_row, k, scale):
    """q_row [D], k [N,D] -> attn [N] and logits [N]."""
    logits = (q_row.float() @ k.float().T) * scale
    attn = logits.softmax(dim=-1)
    return logits, attn


def idx_to_tyx(idx: int):
    t = idx // GP
    rem = idx % GP
    y, x = divmod(rem, GRID)
    return t, y, x


def corner_indices(n: int):
    tslots = n // GP
    idxs = []
    for t in range(tslots):
        base = t * GP
        for y, x in ((0, 0), (0, GRID - 1), (GRID - 1, 0), (GRID - 1, GRID - 1)):
            idxs.append(base + y * GRID + x)
    return idxs


def spatial_mean(v: np.ndarray) -> np.ndarray:
    return v.reshape(-1, GRID, GRID).mean(axis=0)


def print_and_save_stage(name: str, vec: np.ndarray, head: int, logf, topk: int = 16):
    """vec shape [N]."""
    n = vec.size
    path = OUT / f"h{head}_{name}_N{n}.npy"
    np.save(path, vec.astype(np.float32))
    sp = spatial_mean(vec)
    np.save(OUT / f"h{head}_{name}_map16.npy", sp.astype(np.float32))

    cidx = corner_indices(n)
    # last-slot corners (most recent) + mean over time of each corner type
    tslots = n // GP
    last_base = (tslots - 1) * GP
    corners_last = {
        "TL": vec[last_base + 0],
        "TR": vec[last_base + (GRID - 1)],
        "BL": vec[last_base + (GRID - 1) * GRID],
        "BR": vec[last_base + (GRID - 1) * GRID + (GRID - 1)],
    }
    # mean over all temporal copies of the 4 corners
    corner_all = vec[cidx]
    other = np.delete(vec, cidx)

    logf.write(f"\n=== h{head} | {name} | N={n} ===\n")
    logf.write(
        f"  corner_mean={corner_all.mean():.6f}  other_mean={other.mean():.6f}  "
        f"Δ={corner_all.mean()-other.mean():+.6f}  "
        f"ratio={corner_all.mean()/(other.mean()+1e-12):.4f}\n"
    )
    logf.write(
        f"  last-slot corners: TL={corners_last['TL']:.6f} TR={corners_last['TR']:.6f} "
        f"BL={corners_last['BL']:.6f} BR={corners_last['BR']:.6f}\n"
    )
    logf.write(f"  time-mean map corners: TL={sp[0,0]:.6f} TR={sp[0,-1]:.6f} "
               f"BL={sp[-1,0]:.6f} BR={sp[-1,-1]:.6f}\n")
    top = np.argsort(-vec)[:topk]
    logf.write(f"  top{topk} (idx, t,y,x, val):\n")
    for i, idx in enumerate(top):
        t, y, x = idx_to_tyx(int(idx))
        tag = " CORNER" if (y in (0, GRID - 1) and x in (0, GRID - 1)) else ""
        logf.write(f"    #{i+1:02d} idx={int(idx):4d} (t={t:2d},y={y:2d},x={x:2d}) {vec[idx]:.6f}{tag}\n")
    return sp


def plot_stage_maps(head: int, stages: list[tuple[str, np.ndarray]], out_png: Path):
    n = len(stages)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, sp) in zip(axes.ravel(), stages):
        ax.axis("on")
        if "delta" in name or name.startswith("d_"):
            lim = float(np.max(np.abs(sp)) + 1e-12)
            im = ax.imshow(sp, cmap="coolwarm", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(sp, cmap="viridis")
        # annotate 4 corners with numbers
        for (y, x), lab in [((0, 0), "TL"), ((0, 15), "TR"), ((15, 0), "BL"), ((15, 15), "BR")]:
            ax.text(
                x, y, f"{sp[y, x]:.3f}",
                color="white", ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, lw=0),
            )
            ax.plot(x, y, "r.", markersize=4)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"L23 h{head}: each stage is time-mean of full N=4096 scores → 16×16", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_attn_row_maps(head, name, attn_row, out_png):
    """One query's A[i,:] length 4096 → 16 time panels or time-mean."""
    sp = spatial_mean(attn_row)
    tmap = attn_row.reshape(-1, GRID, GRID)  # [T,16,16]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    im0 = axes[0, 0].imshow(sp, cmap="viridis")
    axes[0, 0].set_title(f"h{head} {name}\ntime-mean of A[i,:]")
    for y, x in ((0, 0), (0, 15), (15, 0), (15, 15)):
        axes[0, 0].text(x, y, f"{sp[y,x]:.4f}", color="w", ha="center", va="center", fontsize=7)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # last / mid / first temporal slot
    for ax, ti, title in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        [tmap.shape[0] - 1, tmap.shape[0] // 2, 0],
        ["last t", "mid t", "first t"],
    ):
        im = ax.imshow(tmap[ti], cmap="viridis")
        ax.set_title(f"{title} slot A[i,:]  max={tmap[ti].max():.4f}")
        for y, x in ((0, 0), (0, 15), (15, 0), (15, 15)):
            ax.text(x, y, f"{tmap[ti][y,x]:.3f}", color="w", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Single query attention row (full 4096 → maps)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    print("video", vid)

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
            h_in = x[0].float()
            h_hat = block.norm1(x)[0].float()
            qkv = attn.qkv(block.norm1(x)).unflatten(-1, (3, attn.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q0, k0 = qkv[0][0], qkv[1][0]  # [H,N,D]
            q_r, k_r = apply_rope(attn, q0.unsqueeze(0), k0.unsqueeze(0), T, H_patches, W_patches)
            q_r, k_r = q_r[0], k_r[0]
            scale = float(attn.scale)
            bag["h_l2"] = h_in.norm(dim=-1)
            bag["hn_l2"] = h_hat.norm(dim=-1)
            bag["q_l2"] = q0.float().norm(dim=-1)
            bag["k_l2"] = k0.float().norm(dim=-1)
            bag["logit_pre"] = mean_key_logits(q0, k0, scale)
            bag["logit_post"] = mean_key_logits(q_r, k_r, scale)
            bag["mass_pre"] = attention_mass(q0, k0, scale)
            bag["mass_post"] = attention_mass(q_r, k_r, scale)
            bag["q0"] = q0.float()
            bag["k0"] = k0.float()
            bag["qr"] = q_r.float()
            bag["kr"] = k_r.float()
            bag["scale"] = scale
            bag["N"] = int(x.shape[1])
        return orig(x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)

    block.forward = wrapped
    clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
    ant = torch.full((1,), 1.0, device=device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _ = model(clip_b, ant)
    block.forward = orig

    n = bag["N"]
    log_path = OUT / "printed_4096_stages.txt"
    logf = open(log_path, "w")
    logf.write(f"video={vid} layer={last} N={n} grid={GRID} T={n//GP}\n")
    logf.write("Each stage dumps full length-N vector; map = mean over T → 16x16.\n")
    logf.write("Attention mass I_j = sum_i softmax(q_i·k)_ij  (this is the sink score).\n")

    for head in HEADS:
        stages_maps = []
        seq = [
            ("01_input_h_l2", bag["h_l2"].cpu().numpy()),
            ("02_after_norm_l2", bag["hn_l2"].cpu().numpy()),
            ("03_q_l2", bag["q_l2"][head].cpu().numpy()),
            ("04_k_l2", bag["k_l2"][head].cpu().numpy()),
            ("05_mean_logit_pre_RoPE", bag["logit_pre"][head].cpu().numpy()),
            ("06_mean_logit_post_RoPE", bag["logit_post"][head].cpu().numpy()),
            ("07_delta_logit_RoPE", (bag["logit_post"][head] - bag["logit_pre"][head]).cpu().numpy()),
            ("08_attn_mass_pre_RoPE", bag["mass_pre"][head].cpu().numpy()),
            ("09_attn_mass_post_RoPE", bag["mass_post"][head].cpu().numpy()),
            ("10_delta_mass_RoPE", (bag["mass_post"][head] - bag["mass_pre"][head]).cpu().numpy()),
        ]
        for name, vec in seq:
            sp = print_and_save_stage(name, vec, head, logf)
            stages_maps.append((name, sp))
        plot_stage_maps(head, stages_maps, OUT / f"h{head}_all_stages_maps.png")

        # One center query (mid time, center spatial) full attention row
        t_mid = (n // GP) // 2
        q_idx = t_mid * GP + (GRID // 2) * GRID + (GRID // 2)
        scale = bag["scale"]
        logits_pre, attn_pre = one_query_attn(bag["q0"][head, q_idx], bag["k0"][head], scale)
        logits_post, attn_post = one_query_attn(bag["qr"][head, q_idx], bag["kr"][head], scale)
        for tag, logits, attn in [
            ("centerQ_preRoPE", logits_pre, attn_pre),
            ("centerQ_postRoPE", logits_post, attn_post),
        ]:
            arr = attn.detach().cpu().numpy()
            np.save(OUT / f"h{head}_{tag}_attn_row4096.npy", arr)
            np.save(OUT / f"h{head}_{tag}_logits_row4096.npy", logits.detach().cpu().numpy())
            print_and_save_stage(f"{tag}_attn", arr, head, logf, topk=20)
            plot_attn_row_maps(head, tag, arr, OUT / f"h{head}_{tag}_attn_maps.png")

            # print corner vs sum for this single query
            cidx = corner_indices(n)
            logf.write(
                f"  single-query corner_mass_sum={arr[cidx].sum():.6f}  "
                f"(4*T={4*(n//GP)} corner keys)  "
                f"max_corner={arr[cidx].max():.6f}  max_all={arr.max():.6f}\n"
            )

        # Compare jumps numerically
        m_pre = bag["mass_pre"][head].cpu().numpy()
        m_post = bag["mass_post"][head].cpu().numpy()
        cidx = corner_indices(n)
        logf.write(
            f"\n*** h{head} JUMP SUMMARY ***\n"
            f"  mass corner_mean: pre={m_pre[cidx].mean():.4f} post={m_post[cidx].mean():.4f}\n"
            f"  mass other_mean:  pre={np.delete(m_pre,cidx).mean():.4f} post={np.delete(m_post,cidx).mean():.4f}\n"
            f"  If pre already >> other, Softmax(q·k) before RoPE created the sink.\n"
            f"  If only post jumps, RoPE created it.\n"
        )

    logf.close()
    print("wrote", OUT)
    print(log_path.read_text()[-4000:])


if __name__ == "__main__":
    main()
