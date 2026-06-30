#!/usr/bin/env python
"""Predictor attention visualization — 4s in-distribution and 1min block-11 variants.

Motivation: The block-0 / 1min experiment (2026-06-28, job 11941438) found center/edge=1.002
(uniform) because temporal positions 0..239 are 15× OOD for the predictor's RoPE.
Two follow-up experiments to see whether predictor attention CAN provide useful signal:

  MODE=4s_block0    — 4s clips: 32f/8fps/2-tubelet = 16 temporal slots, positions 0..15
                       → FULLY in-distribution for predictor's RoPE (trained on 0..15).
                       Key question: does predictor block-0 focus center when positions are ok?

  MODE=1min_block11 — 1min clips: same OOD positions but predictor block-11 (last layer).
                       12 layers of MLP+attention may compensate for OOD positional distortion.
                       Key question: is last-layer attention more discriminative than block-0?

  MODE=4s_prune     — For 4s clips, compare "further prune 4096→K using predictor block-0 signal"
                       vs "keep top-K by encoder col-sum" (uniform 4096, so col-sum = flat baseline).
                       Concretely: what spatial distribution does predictor pick when reducing 4s tokens?

Cache layouts:
  4s:   nf32_fps8.0_px256_keep4096_idx  — all 4096 tokens (32f/2=16 slots × 16×16=256 spatial)
  1min: nf480_fps8.0_px256_keep61440_idx — all 61440 tokens (240 slots × 256 spatial)

Spatial grid:
  Both: GRID=16, patches 16×16 per slot (ViT-L/256 with patch_size=16).
  At 4s positions 0..15 are in-distribution; at 1min positions 0..239 are OOD.

Position decode (same for both):
  idx → slot = idx // 256, h = (idx%256)//16, w = idx%16
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/path/to/VJEPA2-EXP")
CODE_ROOT = os.environ.get("PROJECT_ROOT", SHARED)
sys.path.insert(0, os.path.join(CODE_ROOT, "vjepa2"))
sys.path.insert(0, CODE_ROOT)

MODE = os.environ.get("MODE", "4s_block0")   # 4s_block0 | 1min_block11 | 4s_prune
N_SAMPLE = int(os.environ.get("N_SAMPLE", "100"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "256"))
CKPT = os.path.join(SHARED, "checkpoints/vitl.pt")

# ── mode-specific settings ────────────────────────────────────────────────────

if MODE in ("4s_block0", "4s_prune"):
    CACHE = os.path.join(SHARED, "data/preproc_cache_vjepa/nf32_fps8.0_px256_keep4096")
    NTOK = 4096
    SLOTS = 16       # 32f / tubelet=2 = 16 temporal slots; positions 0..15 IN DISTRIBUTION
    PRED_BLOCK = 0
    KEEP_K_PRUNE = int(os.environ.get("KEEP_K", "256"))   # 4s → 256 further pruning
    OUT = os.environ.get("OUT_DIR", os.path.join(SHARED, "outputs/pred_attn_4s_block0"))
    LABEL = "4s block-0 (in-distribution positions 0..15)"
    LIFT_NPATCHES = False   # not needed; 4s stays within predictor's trained range
elif MODE == "1min_block11":
    CACHE = os.path.join(SHARED, "data/preproc_cache_vjepa/nf480_fps8.0_px256_keep61440_idx")
    NTOK = 61440
    SLOTS = 240
    PRED_BLOCK = 11
    KEEP_K_PRUNE = int(os.environ.get("KEEP_K", "4096"))
    OUT = os.environ.get("OUT_DIR", os.path.join(SHARED, "outputs/pred_attn_1min_block11"))
    LABEL = "1min block-11 (OOD positions 0..239)"
    LIFT_NPATCHES = True    # same lift as the original 1min block-0 experiment
else:
    raise ValueError(f"Unknown MODE={MODE!r}. Use: 4s_block0 | 1min_block11 | 4s_prune")

GRID = 16
GP = GRID * GRID   # 256 patches per slot
FPS_EFF = 8 / 2   # 4 slots/s (8fps / tubelet 2)

os.makedirs(OUT, exist_ok=True)

print(f"[config] MODE={MODE}  PRED_BLOCK={PRED_BLOCK}  NTOK={NTOK}  SLOTS={SLOTS}  KEEP_K_PRUNE={KEEP_K_PRUNE}", flush=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def decode(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    slot = idx // GP
    within = idx % GP
    return slot, within // GRID, within % GRID


def accum_hard(idx: np.ndarray, spatial: np.ndarray, temporal: np.ndarray) -> None:
    _, h, w = decode(idx)
    np.add.at(spatial, (h, w), 1)
    slot, _, _ = decode(idx)
    np.add.at(temporal, slot, 1)


def accum_soft(imp: np.ndarray, spatial: np.ndarray, temporal: np.ndarray, norm_k: int) -> None:
    """Weighted accumulation scaled to norm_k total."""
    w = imp / imp.sum() * norm_k
    slot_all, h_all, ww = decode(np.arange(len(imp)))
    np.add.at(spatial, (h_all, ww), w)
    np.add.at(temporal, slot_all, w)


def center_edge(spatial: np.ndarray, lo: int = 4, hi: int = 12) -> Tuple[float, float]:
    """Center (inner 8×8) vs edge mean for 16×16 grid."""
    c = spatial[lo:hi, lo:hi].mean()
    n_edge = GP - (hi - lo) ** 2
    e = (spatial.sum() - spatial[lo:hi, lo:hi].sum()) / n_edge
    return float(c), float(e)


# ── predictor attention hook ──────────────────────────────────────────────────

def make_block_hook(blk_attn, chunk_size: int):
    """Patch blk_attn.forward (RoPEAttention) to capture chunked col-sum importance."""
    from src.models.utils.modules import rotate_queries_or_keys  # noqa

    m = blk_attn
    orig = m.forward
    imp_store = [None]

    def _fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()

        # Position decode (same as RoPEAttention.forward)
        if mask is not None:
            mask_p = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
        else:
            grid_depth = N // (m.grid_size * m.grid_size)
            mask_p = torch.arange(grid_depth * m.grid_size * m.grid_size, device=x.device)
        d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)

        qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        s = 0
        qd = rotate_queries_or_keys(q[..., s:s + m.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s:s + m.d_dim], pos=d_mask)
        s += m.d_dim
        qh = rotate_queries_or_keys(q[..., s:s + m.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s:s + m.h_dim], pos=h_mask)
        s += m.h_dim
        qw = rotate_queries_or_keys(q[..., s:s + m.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s:s + m.w_dim], pos=w_mask)
        s += m.w_dim
        q = torch.cat([qd, qh, qw] + ([q[..., s:]] if s < m.head_dim else []), dim=-1)
        k = torch.cat([kd, kh, kw] + ([k[..., s:]] if s < m.head_dim else []), dim=-1)

        with torch.no_grad():
            k_t = k.transpose(-2, -1)
            imp = torch.zeros(B, N, device=x.device, dtype=torch.float32)
            for ci in range(0, N, chunk_size):
                q_c = q[:, :, ci:ci + chunk_size, :]
                logits = torch.matmul(q_c, k_t) * m.scale
                imp += logits.float().softmax(dim=-1).sum(dim=2).sum(dim=1)
                del logits
        imp_store[0] = imp.cpu()

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = m.proj(out)
        out = m.proj_drop(out)
        return out

    m.forward = _fwd
    return orig, imp_store


# ── build predictor ──────────────────────────────────────────────────────────

def build_predictor(device: torch.device):
    import app.hdepic_lora_action_anticipation.train_vjepa_prune_anticipation as M
    model = M.build_model(device, frames_per_clip=480, fps=8, img_size=256, checkpoint=CKPT)
    if LIFT_NPATCHES:
        n_full = NTOK
        n_pred = GRID * GRID * round(1.0 * 8 / 2)  # 1s×fps/tubelet = 4 slots × 256
        new_cap = ((n_full + n_pred) // GP + 8) * GP
        model.predictor.num_patches = new_cap
        print(f"  predictor.num_patches lifted to {new_cap}", flush=True)
    return model.predictor


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}", flush=True)

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
    if not files:
        raise RuntimeError(f"No .pt files found in {CACHE}")
    random.seed(42)
    samp = random.sample(files, min(N_SAMPLE, len(files)))
    print(f"[data] {len(samp)} / {len(files)} samples", flush=True)

    print("[model] loading predictor...", flush=True)
    t0 = time.time()
    predictor = build_predictor(dev)
    predictor.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  hooking predictor_blocks[{PRED_BLOCK}]", flush=True)

    # For 1min_block11 we need to run through blocks 0..10 to get the input to block 11.
    # Hook block PRED_BLOCK's attn module.
    blk_attn = predictor.predictor_blocks[PRED_BLOCK].attn
    orig_fwd, imp_store = make_block_hook(blk_attn, CHUNK_SIZE)

    # Accumulators
    spatial_pred = np.zeros((GRID, GRID), dtype=np.float64)
    temporal_pred = np.zeros(SLOTS, dtype=np.float64)
    spatial_topk = np.zeros((GRID, GRID), dtype=np.float64)
    temporal_topk = np.zeros(SLOTS, dtype=np.float64)

    ctx_pos = torch.arange(NTOK, device=dev).unsqueeze(0)   # [1, NTOK]

    n_done = 0
    t_start = time.time()

    for fpath in samp:
        t_s = time.time()
        try:
            o = torch.load(fpath, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  [skip] {os.path.basename(fpath)}: {e}", flush=True)
            continue

        tok = o["tok"] if isinstance(o, dict) else o   # [NTOK, D]
        if tok.shape[0] != NTOK:
            # Short clip (e.g. 4s cache of a clip with fewer frames) — skip
            print(f"  [skip] {os.path.basename(fpath)}: tok.shape={tok.shape}", flush=True)
            continue
        tok = tok.unsqueeze(0).to(device=dev, dtype=torch.bfloat16)   # [1, NTOK, D]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = predictor.predictor_embed(tok)   # [1, NTOK, 384]

            if PRED_BLOCK == 0:
                # Directly run block 0 — hook fires here
                predictor.predictor_blocks[0](x, mask=ctx_pos, attn_mask=None)
            else:
                # Run blocks 0..PRED_BLOCK-1 normally, then block PRED_BLOCK triggers hook
                for i in range(PRED_BLOCK):
                    x = predictor.predictor_blocks[i](x, mask=ctx_pos, attn_mask=None)
                # Hook fires inside block PRED_BLOCK
                predictor.predictor_blocks[PRED_BLOCK](x, mask=ctx_pos, attn_mask=None)

        if imp_store[0] is None:
            print(f"  [warn] no importance for {os.path.basename(fpath)}", flush=True)
            continue

        imp_np = imp_store[0][0].numpy()   # [NTOK]
        imp_store[0] = None

        accum_soft(imp_np, spatial_pred, temporal_pred, norm_k=KEEP_K_PRUNE)

        topk_idx = np.argsort(imp_np)[-KEEP_K_PRUNE:]
        accum_hard(topk_idx, spatial_topk, temporal_topk)

        del tok, x
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done}/{len(samp)}  {time.time()-t_s:.1f}s/sample  "
                  f"total {time.time()-t_start:.0f}s", flush=True)

    blk_attn.forward = orig_fwd

    if n_done == 0:
        print("[error] no valid samples", flush=True)
        return

    spatial_pred /= n_done
    temporal_pred /= n_done
    spatial_topk /= n_done
    temporal_topk /= n_done

    np.save(os.path.join(OUT, "spatial_soft.npy"), spatial_pred)
    np.save(os.path.join(OUT, "temporal_soft.npy"), temporal_pred)
    np.save(os.path.join(OUT, "spatial_topk.npy"), spatial_topk)
    np.save(os.path.join(OUT, "temporal_topk.npy"), temporal_topk)

    unif_s = KEEP_K_PRUNE / GP
    unif_t = KEEP_K_PRUNE / SLOTS
    secs = np.arange(SLOTS) / FPS_EFF   # 0..3.75 for 4s, 0..59.75 for 1min
    last10 = secs >= max(0, secs.max() - 10)
    first10 = secs <= 10

    c_soft, e_soft = center_edge(spatial_pred)
    c_topk, e_topk = center_edge(spatial_topk)

    summary = {
        "mode": MODE, "pred_block": PRED_BLOCK,
        "n_samples": n_done, "ntok": NTOK, "keep_k_prune": KEEP_K_PRUNE,
        "uniform_spatial": round(unif_s, 2),
        "uniform_temporal": round(unif_t, 2),
        "label": LABEL,
        f"pred_block{PRED_BLOCK}_soft": {
            "spatial_center8x8": round(c_soft, 3),
            "spatial_edge_mean": round(e_soft, 3),
            "center_vs_edge_ratio": round(c_soft / e_soft, 3),
            "frac_last10s": round(float(temporal_pred[last10].sum() / temporal_pred.sum()), 4),
            "frac_first10s": round(float(temporal_pred[first10].sum() / temporal_pred.sum()), 4),
        },
        f"pred_block{PRED_BLOCK}_top{KEEP_K_PRUNE}": {
            "spatial_center8x8": round(c_topk, 3),
            "spatial_edge_mean": round(e_topk, 3),
            "center_vs_edge_ratio": round(c_topk / e_topk, 3),
        },
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vmax = max(spatial_pred.max(), spatial_topk.max())

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, data, c, e, tag in [
            (axes[0], spatial_pred, c_soft, e_soft,
             f"soft (weighted)\ncenter/edge={c_soft:.2f}/{e_soft:.2f}  ratio={c_soft/e_soft:.3f}"),
            (axes[1], spatial_topk, c_topk, e_topk,
             f"top-{KEEP_K_PRUNE} (hard)\ncenter/edge={c_topk:.2f}/{e_topk:.2f}  ratio={c_topk/e_topk:.3f}"),
        ]:
            im = ax.imshow(data, cmap="magma", vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, label=f"avg importance (uniform≈{unif_s:.1f})")
            ax.set_title(f"predictor block-{PRED_BLOCK} {tag}", fontsize=9)
            ax.set_xlabel("w"); ax.set_ylabel("h")
        fig.suptitle(f"Predictor block-{PRED_BLOCK} spatial — {LABEL}\n(n={n_done})", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "spatial.png"), dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        bw = (secs[1] - secs[0]) * 0.45 if len(secs) > 1 else 0.1
        ax.bar(secs - bw, temporal_pred, width=bw, color="#c44", alpha=0.8,
               label=f"block-{PRED_BLOCK} soft")
        ax.bar(secs, temporal_topk / temporal_topk.sum() * temporal_pred.sum(),
               width=bw, color="#37b", alpha=0.7, label=f"block-{PRED_BLOCK} top-{KEEP_K_PRUNE}")
        ax.axhline(unif_t, color="k", ls="--", lw=1, label=f"uniform ({unif_t:.1f})")
        ax.set_xlabel("time in window (s)")
        ax.set_ylabel("avg importance / slot")
        ax.set_title(f"Temporal — predictor block-{PRED_BLOCK} ({LABEL}, n={n_done})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "temporal.png"), dpi=130)
        plt.close(fig)

        # Spatial slices
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, label, sv, sk in [
            (axes[0], "center column (w=7-8)",
             spatial_pred[:, 7:9].mean(1), spatial_topk[:, 7:9].mean(1)),
            (axes[1], "edge column (w=0 or 15)",
             (spatial_pred[:, 0] + spatial_pred[:, 15]) / 2,
             (spatial_topk[:, 0] + spatial_topk[:, 15]) / 2),
        ]:
            ax.plot(np.arange(GRID), sv, "o-", color="#c44", label="soft")
            ax.plot(np.arange(GRID), sk, "s-", color="#37b", label=f"top-{KEEP_K_PRUNE}")
            ax.axhline(unif_s, color="k", ls="--", lw=1, label=f"uniform ({unif_s:.1f})")
            ax.set_xlabel("h"); ax.set_ylabel("avg importance")
            ax.set_title(label); ax.legend(fontsize=8)
        fig.suptitle(f"Predictor block-{PRED_BLOCK} spatial slices — {LABEL}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "spatial_slice.png"), dpi=130)
        plt.close(fig)

        print(f"[plots] saved to {OUT}", flush=True)
    except Exception as e:
        print(f"[plots] failed ({e}); arrays saved", flush=True)

    print(f"[done] {n_done} samples in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
