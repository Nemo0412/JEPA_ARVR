#!/usr/bin/env python
"""Visualize predictor block-0 attention column-sum and compare to encoder block-24.

Direction A gate (2026-06-28): does predictor block-0, run on all 61440 context tokens
(true positions 0..61439), focus attention on image center (foreground) or show the same
periphery-sink bias as the encoder's last-block column-sum?

Encoder comparison baseline comes from the keep4096_idx cache (which 4096 tokens were
selected by encoder column-sum). Predictor block-0 importance is computed fresh from the
full-ctx cache (61440 tokens, 126 MB/sample) via chunked attention (no full N^2 matrix).

Index layout: flat idx = slot*256 + h*16 + w
  slot in 0..239  (temporal; slot s ~ s*0.25s, 239=most recent)
  h, w in 0..15   (spatial; 16x16 grid over 256px input)

Outputs -> OUT_DIR (default outputs/pred_block0_attn/):
  spatial_enc.npy / spatial_pred.npy          avg kept count per 16x16 patch
  temporal_enc.npy / temporal_pred.npy        avg per-slot kept count
  importance_samples.npy                      [N_SAMPLE, 61440] float32 importance scores
  topk4096_pred_spatial.npy                   pred importance top-4096 spatial pattern
  spatial_comparison.png
  temporal_comparison.png
  topk_comparison.png                         encoder top-4096 vs pred top-4096 spatial
  summary.json
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/path/to/VJEPA2-EXP")
CODE_ROOT = os.environ.get("PROJECT_ROOT", SHARED)
sys.path.insert(0, os.path.join(CODE_ROOT, "vjepa2"))
sys.path.insert(0, CODE_ROOT)

FULLCTX_CACHE = os.path.join(SHARED, "data/preproc_cache_vjepa/nf480_fps8.0_px256_keep61440_idx")
ENC_CACHE = os.path.join(SHARED, "data/preproc_cache_vjepa/nf480_fps8.0_px256_keep4096_idx")
CKPT = os.path.join(SHARED, "checkpoints/vitl.pt")
OUT = os.environ.get("OUT_DIR", os.path.join(SHARED, "outputs/pred_block0_attn"))

N_SAMPLE = int(os.environ.get("N_SAMPLE", "100"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "256"))  # tokens per attention chunk; 256 -> ~180 MB fp16
KEEP_K = 4096

GRID = 16    # spatial patches per frame side
SLOTS = 240  # temporal slots (480f / tubelet_size=2)
GP = GRID * GRID  # 256 patches per slot
NTOK = SLOTS * GP  # 61440 total tokens
FPS_EFF = 8 / 2    # effective slot rate (fps / tubelet) = 4 slots/s

os.makedirs(OUT, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def decode_positions(idx_flat: np.ndarray):
    """idx_flat [K] → (slot [K], h [K], w [K]) — for true-position layout (slot*256+h*16+w)."""
    slot = idx_flat // GP
    within = idx_flat % GP
    return slot, within // GRID, within % GRID


def spatial_accum(idx_flat: np.ndarray, spatial: np.ndarray, n_samples: int):
    """Accumulate per-patch kept-count into spatial [16,16]."""
    _, h, w = decode_positions(idx_flat)
    np.add.at(spatial, (h, w), 1)


def temporal_accum(idx_flat: np.ndarray, temporal: np.ndarray):
    """Accumulate per-slot kept-count into temporal [240]."""
    slot, _, _ = decode_positions(idx_flat)
    np.add.at(temporal, slot, 1)


def load_enc_idx(path: str) -> np.ndarray:
    """Load encoder-selected top-4096 indices from keep4096_idx cache."""
    o = torch.load(path, map_location="cpu", weights_only=True)
    idx = (o["idx"] if isinstance(o, dict) else o).numpy().astype(np.int64)
    return idx[(idx >= 0) & (idx < NTOK)]


def load_fullctx_tok(path: str, device: torch.device) -> torch.Tensor:
    """Load all 61440 encoder tokens from full-ctx cache -> [1, 61440, 1024] bf16."""
    o = torch.load(path, map_location="cpu", weights_only=True)
    tok = o["tok"] if isinstance(o, dict) else o  # [61440, 1024] fp16
    return tok.unsqueeze(0).to(device=device, dtype=torch.bfloat16)


# ── attention importance hook for predictor block-0 ──────────────────────────

def make_block0_hook(attn_module, chunk_size: int):
    """Monkey-patch attn_module.forward to compute chunked column-sum importance.

    Works with RoPEAttention (the Block class in modules.py at line 505 uses RoPEAttention
    when use_rope=True). Mirrors the TokenPruner approach from train_vjepa_prune_anticipation.py
    but adapted for the predictor's RoPEAttention.forward signature:
      (self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None)

    Column-sum importance: importance[j] = sum_heads sum_queries softmax(q_i * k_j^T / scale)[j]
    This is the same metric as the encoder's pruner but measured from predictor block-0.
    """
    from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

    m = attn_module
    _orig_fwd = m.forward
    _imp_store = [None]  # mutable container for the captured importance

    def _patched_fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()

        # Compute positions (same as RoPEAttention.forward)
        if mask is not None:
            mask_p = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
            d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)
        else:
            grid_depth = N // (m.grid_size * m.grid_size)
            mask_p = torch.arange(grid_depth * m.grid_size * m.grid_size, device=x.device)
            d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)

        # QKV projection
        qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        # Apply RoPE
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
        if s < m.head_dim:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        # Chunked importance: column-sum of attention weights, no full N^2 materialisation
        with torch.no_grad():
            k_t = k.transpose(-2, -1)  # [B, heads, head_dim, N]
            imp = torch.zeros(B, N, device=x.device, dtype=torch.float32)
            for ci in range(0, N, chunk_size):
                q_c = q[:, :, ci:ci + chunk_size, :]  # [B, heads, chunk, head_dim]
                logits = torch.matmul(q_c, k_t) * m.scale  # [B, heads, chunk, N] bf16
                imp += logits.float().softmax(dim=-1).sum(dim=2).sum(dim=1)
                del logits
        _imp_store[0] = imp.cpu()

        # Real SDPA forward (returns block output normally)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = m.proj(out)
        out = m.proj_drop(out)
        return out

    m.forward = _patched_fwd
    return _orig_fwd, _imp_store


# ── build predictor (no encoder needed) ──────────────────────────────────────

def build_predictor(device: torch.device):
    """Load the predictor (and encoder) from checkpoint; we use only the predictor."""
    import app.hdepic_lora_action_anticipation.train_vjepa_prune_anticipation as M
    model = M.build_model(device, frames_per_clip=480, fps=8, img_size=256, checkpoint=CKPT)
    # Lift num_patches so true positions up to 61440+1024 don't OOB
    n_full = NTOK
    n_pred = GRID * GRID * round(1.0 * 8 / 2)  # 1s * fps / tubelet = 4 slots * 256 = 1024
    new_cap = ((n_full + n_pred) // GP + 8) * GP
    model.predictor.num_patches = new_cap
    print(f"  predictor.num_patches lifted to {new_cap}", flush=True)
    return model.predictor


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}", flush=True)

    # Find matching sample IDs in both caches
    fullctx_files = {os.path.basename(p): p for p in glob.glob(os.path.join(FULLCTX_CACHE, "*.pt"))}
    enc_files = {os.path.basename(p): p for p in glob.glob(os.path.join(ENC_CACHE, "*.pt"))}
    common = sorted(set(fullctx_files) & set(enc_files))
    if not common:
        raise RuntimeError(f"No common files between {FULLCTX_CACHE} and {ENC_CACHE}")
    random.seed(42)
    samp = random.sample(common, min(N_SAMPLE, len(common)))
    print(f"[data] {len(samp)} samples (from {len(common)} common)", flush=True)

    # Load predictor
    print("[model] loading predictor...", flush=True)
    t0 = time.time()
    predictor = build_predictor(dev)
    predictor.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Install hook on block-0 attention
    blk0_attn = predictor.predictor_blocks[0].attn
    orig_fwd, imp_store = make_block0_hook(blk0_attn, chunk_size=CHUNK_SIZE)

    # Accumulators for both patterns
    spatial_enc = np.zeros((GRID, GRID), dtype=np.float64)
    temporal_enc = np.zeros(SLOTS, dtype=np.float64)
    spatial_pred = np.zeros((GRID, GRID), dtype=np.float64)
    temporal_pred = np.zeros(SLOTS, dtype=np.float64)
    spatial_topk_pred = np.zeros((GRID, GRID), dtype=np.float64)  # top-4096 by pred importance
    temporal_topk_pred = np.zeros(SLOTS, dtype=np.float64)
    all_imp = []  # list of [61440] float32 arrays for optional inspection

    ctx_pos = torch.arange(NTOK, device=dev).unsqueeze(0)  # [1, 61440] true positions

    n_done = 0
    t_start = time.time()
    for fname in samp:
        t_s = time.time()

        # ---- encoder baseline: load the 4096 selected indices ----
        enc_idx = load_enc_idx(enc_files[fname])
        spatial_accum(enc_idx, spatial_enc, n_done)
        temporal_accum(enc_idx, temporal_enc)

        # ---- predictor block-0: load full-ctx tokens, run block-0, capture importance ----
        tok = load_fullctx_tok(fullctx_files[fname], dev)  # [1, 61440, 1024]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = predictor.predictor_embed(tok)  # [1, 61440, 384]
            # Run through block-0 only (includes norm1 internally via Block.forward)
            predictor.predictor_blocks[0](x, mask=ctx_pos, attn_mask=None)

        imp_np = imp_store[0].float().numpy()  # [1, 61440]
        imp_flat = imp_np[0]  # [61440]
        all_imp.append(imp_flat)

        # Accumulate spatial/temporal for raw predictor importance
        # (treat importance as a soft weight map, normalise to sum to KEEP_K for fair comparison)
        imp_scaled = (imp_flat / imp_flat.sum() * KEEP_K)
        _, h_all, w_all = decode_positions(np.arange(NTOK))
        slot_all, _, _ = decode_positions(np.arange(NTOK))
        np.add.at(spatial_pred, (h_all, w_all), imp_scaled)
        np.add.at(temporal_pred, slot_all, imp_scaled)

        # Top-K by predictor importance (hard selection)
        topk_idx = np.argsort(imp_flat)[-KEEP_K:]  # top-4096 indices
        topk_idx = topk_idx[(topk_idx >= 0) & (topk_idx < NTOK)]
        spatial_accum(topk_idx, spatial_topk_pred, n_done)
        temporal_accum(topk_idx, temporal_topk_pred)

        del tok, x
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done}/{len(samp)}  {time.time()-t_s:.1f}s/sample  "
                  f"total {time.time()-t_start:.0f}s", flush=True)

    # Restore original forward
    blk0_attn.forward = orig_fwd

    # Normalise by n_done
    spatial_enc /= n_done
    temporal_enc /= n_done
    spatial_pred /= n_done
    temporal_pred /= n_done
    spatial_topk_pred /= n_done
    temporal_topk_pred /= n_done

    # Save arrays
    np.save(os.path.join(OUT, "spatial_enc.npy"), spatial_enc)
    np.save(os.path.join(OUT, "temporal_enc.npy"), temporal_enc)
    np.save(os.path.join(OUT, "spatial_pred.npy"), spatial_pred)
    np.save(os.path.join(OUT, "temporal_pred.npy"), temporal_pred)
    np.save(os.path.join(OUT, "spatial_topk_pred.npy"), spatial_topk_pred)
    np.save(os.path.join(OUT, "temporal_topk_pred.npy"), temporal_topk_pred)
    np.save(os.path.join(OUT, "importance_samples.npy"), np.stack(all_imp))

    # Summary statistics
    unif_s = KEEP_K / GP          # 16.0 per patch (uniform)
    unif_t = KEEP_K / SLOTS       # 17.07 per slot (uniform)
    secs = np.arange(SLOTS) / FPS_EFF

    def center_edge(spatial):
        center = spatial[4:12, 4:12].mean()
        edge = (spatial.sum() - spatial[4:12, 4:12].sum()) / (GP - 64)
        return float(center), float(edge)

    c_enc, e_enc = center_edge(spatial_enc)
    c_pred, e_pred = center_edge(spatial_pred)
    c_topk, e_topk = center_edge(spatial_topk_pred)

    last10 = secs >= (secs.max() - 10)
    first10 = secs <= 10

    def recency_bias(temporal):
        return float(temporal[last10].sum() / temporal.sum()), \
               float(temporal[first10].sum() / temporal.sum())

    rb_enc = recency_bias(temporal_enc)
    rb_pred = recency_bias(temporal_pred)

    summary = {
        "n_samples": n_done, "keep_k": KEEP_K, "total_tokens": NTOK,
        "chunk_size": CHUNK_SIZE,
        "uniform_spatial": round(unif_s, 2),
        "uniform_temporal": round(unif_t, 2),
        "encoder_block24": {
            "spatial_center8x8": round(c_enc, 2),
            "spatial_edge_mean": round(e_enc, 2),
            "center_vs_edge_ratio": round(c_enc / e_enc, 3),
            "frac_kept_last10s": round(rb_enc[0], 4),
            "frac_kept_first10s": round(rb_enc[1], 4),
        },
        "predictor_block0_soft": {
            "spatial_center8x8": round(c_pred, 2),
            "spatial_edge_mean": round(e_pred, 2),
            "center_vs_edge_ratio": round(c_pred / e_pred, 3),
            "frac_kept_last10s": round(rb_pred[0], 4),
            "frac_kept_first10s": round(rb_pred[1], 4),
        },
        "predictor_block0_topk4096": {
            "spatial_center8x8": round(c_topk, 2),
            "spatial_edge_mean": round(e_topk, 2),
            "center_vs_edge_ratio": round(c_topk / e_topk, 3),
        },
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    # ── Figures ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vmax_s = max(spatial_enc.max(), spatial_pred.max(), spatial_topk_pred.max())

        # (A) spatial side-by-side: encoder vs predictor block-0 (soft) vs predictor top-K
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for ax, data, title in [
            (axes[0], spatial_enc, f"Encoder block-24 col-sum (top-{KEEP_K} hard)\n"
             f"center/edge={c_enc:.1f}/{e_enc:.1f} ratio={c_enc/e_enc:.3f}"),
            (axes[1], spatial_pred, f"Predictor block-0 col-sum (soft weighted)\n"
             f"center/edge={c_pred:.1f}/{e_pred:.1f} ratio={c_pred/e_pred:.3f}"),
            (axes[2], spatial_topk_pred, f"Predictor block-0 top-{KEEP_K} (hard)\n"
             f"center/edge={c_topk:.1f}/{e_topk:.1f} ratio={c_topk/e_topk:.3f}"),
        ]:
            im = ax.imshow(data, cmap="magma", vmin=0, vmax=vmax_s)
            fig.colorbar(im, ax=ax, label="avg kept / patch (uniform=16)")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("w"); ax.set_ylabel("h")
        fig.suptitle(f"Spatial importance: encoder vs predictor block-0 (n={n_done})", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "spatial_comparison.png"), dpi=130)
        plt.close(fig)

        # (B) temporal: encoder vs predictor block-0 (soft)
        fig, ax = plt.subplots(figsize=(12, 4))
        w = 1 / FPS_EFF * 0.45
        ax.bar(secs - w, temporal_enc, width=w, color="#3b7", alpha=0.8, label="encoder block-24")
        ax.bar(secs, temporal_pred, width=w, color="#c44", alpha=0.8, label="predictor block-0 (soft)")
        ax.axhline(unif_t, color="k", ls="--", lw=1, label=f"uniform ({unif_t:.1f})")
        ax.set_xlabel("time in 60s window (s)  [0=oldest, 60≈action]")
        ax.set_ylabel(f"avg importance / 0.25s slot  (norm to {KEEP_K} tokens)")
        ax.set_title(f"Temporal pattern: encoder block-24 vs predictor block-0 (n={n_done})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "temporal_comparison.png"), dpi=130)
        plt.close(fig)

        # (C) diagonal slices: center column (w=7,8) and edge column (w=0,15) per row h
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, label, s_enc, s_pred in [
            (axes[0], "center column (w=7-8)",
             spatial_enc[:, 7:9].mean(axis=1), spatial_pred[:, 7:9].mean(axis=1)),
            (axes[1], "edge column (w=0 or 15)",
             (spatial_enc[:, 0] + spatial_enc[:, 15]) / 2,
             (spatial_pred[:, 0] + spatial_pred[:, 15]) / 2),
        ]:
            ax.plot(np.arange(GRID), s_enc, "o-", color="#3b7", label="encoder block-24")
            ax.plot(np.arange(GRID), s_pred, "s-", color="#c44", label="predictor block-0")
            ax.axhline(unif_s, color="k", ls="--", lw=1, label=f"uniform ({unif_s:.1f})")
            ax.set_xlabel("h (row, 0=top)"); ax.set_ylabel("avg importance")
            ax.set_title(f"Spatial slice — {label}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "spatial_slice.png"), dpi=130)
        plt.close(fig)

        print(f"[plots] saved to {OUT}", flush=True)
    except Exception as e:
        print(f"[plots] failed ({e}); arrays already saved", flush=True)

    print(f"[done] {n_done} samples in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
