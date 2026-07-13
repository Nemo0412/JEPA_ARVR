#!/usr/bin/env python
"""Visualize Qwen2.5-VL LLM layer-0 attention vs vision-tower col-sum (Direction A analogue).

Context (2026-06-28): For V-JEPA2, predictor block-0 on OOD positions gave uniform attention
(center/edge=1.002). Qwen's LLM layer-0 is different: the M-RoPE position_ids for video tokens
are their REAL (temporal, height, width) coordinates — fully in-distribution.

We capture two importance signals per video token:
  (A) Vision tower block-31 col-sum   -- current pruning signal
  (B) LLM layer-0 attention from text-after-video tokens to video tokens
      importance[j] = sum_heads softmax(q_text @ k_video_j / scale)
      text-after = 17 tokens at positions 19455..19471 (question-end + assistant-start)

Note on RoPE: to avoid complexity of M-RoPE slicing across sequence positions, we apply the
projection-only (no-RoPE) dot-product: q_ta = q_proj(hs_ta), k_vid = k_proj(hs_vid), then
q_ta @ k_vid.T / scale. This measures CONTENT similarity (which video token features match
the text query representation) without positional modulation. For spatial bias analysis
(center vs edge) this is the dominant signal.

Architecture (Qwen2.5-VL-3B, EgoLifeExp overlay, transformers 5.6.2):
  LLM path: backbone.model.language_model.layers[0]  (Qwen2_5_VLTextModel, no .model attribute)
  self_attn: Qwen2_5_VLAttention  (num_heads=16, num_kv_heads=2, head_dim=128)
  position_embeddings=(cos,sin) passed via self_attn.forward kwarg

Sequence layout (480f, 256px): 15 text + 19440 video + 17 text = 19472 total

Spatial grid: 9x9 (14px patch + 2x2 merge on 256px -> 18->9 per side).
Temporal: 240 slots (480f / temporal_patch_size=2 = 240), 0.25s each.
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

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/path/to/VJEPA2-EXP")
CODE_ROOT = os.environ.get("PROJECT_ROOT", SHARED)
sys.path.insert(0, os.path.join(CODE_ROOT, "vjepa2"))
sys.path.insert(0, CODE_ROOT)

PREPROC_CACHE = os.path.join(SHARED, "data/preproc_cache_qwen/qwen25vl_nf480_pnf480_fps8.0_px256")
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUT = os.environ.get("OUT_DIR", os.path.join(SHARED, "outputs/qwen_llm_layer0_attn"))

N_SAMPLE = int(os.environ.get("N_SAMPLE", "50"))
KEEP_K = 1297

GRID_Q = 9
SLOTS = 240
GP_Q = GRID_Q * GRID_Q   # 81
NTOK_Q = SLOTS * GP_Q    # 19440

N_TEXT_BEFORE = 15
N_TEXT_AFTER = 17  # positions 19455..19471

os.makedirs(OUT, exist_ok=True)


# ── spatial / temporal helpers ────────────────────────────────────────────────

def decode_qwen(idx: np.ndarray):
    slot = idx // GP_Q
    within = idx % GP_Q
    return slot, within // GRID_Q, within % GRID_Q


def accum_hard(idx: np.ndarray, spatial: np.ndarray, temporal: np.ndarray):
    _, h, w = decode_qwen(idx)
    np.add.at(spatial, (h, w), 1)
    slot, _, _ = decode_qwen(idx)
    np.add.at(temporal, slot, 1)


def accum_soft(imp: np.ndarray, spatial: np.ndarray, temporal: np.ndarray):
    """Weighted accumulation normalised to KEEP_K total."""
    w = imp / imp.sum() * KEEP_K
    slot_all, h_all, ww = decode_qwen(np.arange(NTOK_Q))
    np.add.at(spatial, (h_all, ww), w)
    np.add.at(temporal, slot_all, w)


# ── vision-tower importance hook ──────────────────────────────────────────────

def install_vis_hook(backbone):
    from app.hdepic_lora_action_anticipation.qwen_token_pruning import (
        _patch_last_vision_attention, _PruneState, _merged_importance_canonical,
    )
    state = _PruneState(keep_ratio=0.0667, chunk_size=0)
    inner = backbone.model
    state.inner = inner
    _patch_last_vision_attention(inner.visual, state)
    return state, _merged_importance_canonical


# ── LLM layer-0 hook (content-based, no explicit M-RoPE) ─────────────────────

def install_llm_hook(backbone):
    """Patch LLM layer-0 self_attn.forward to capture text-after → video importance.

    Uses content-only dot product (q_proj @ k_proj.T, no M-RoPE) which avoids the
    cross-position M-RoPE complexity while still measuring which video token features
    align with the text query representation. Valid for spatial/temporal bias analysis.

    Architecture specifics (transformers 5.6.2, Qwen2.5-VL-3B):
      - LLM: Qwen2_5_VLTextModel at backbone.model.language_model
      - Layer 0: .layers[0] (NOT .model.layers[0])
      - Attention: Qwen2_5_VLAttention, num_heads=16, num_kv_heads=2 (GQA 8:1), head_dim=128
    """
    lm = backbone.model.language_model   # Qwen2_5_VLTextModel
    layer0 = lm.layers[0]
    self_attn = layer0.self_attn

    nh = self_attn.num_heads
    nkv = self_attn.num_key_value_heads
    nkvg = nh // nkv
    hd = self_attn.head_dim

    imp_store = [None]
    orig_fwd = self_attn.forward

    def _patched_fwd(hidden_states, attention_mask=None, position_ids=None,
                     past_key_value=None, output_attentions=False, use_cache=False,
                     cache_position=None, position_embeddings=None, **kwargs):
        out = orig_fwd(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        B, L, D = hidden_states.shape
        if L != N_TEXT_BEFORE + NTOK_Q + N_TEXT_AFTER:
            return out  # unexpected layout; skip

        text_after_start = N_TEXT_BEFORE + NTOK_Q   # 19455
        with torch.no_grad():
            # Content-only importance: raw q,k projections without RoPE.
            # For text→video attention, M-RoPE height/width dims are near-zero for text
            # tokens (text h=w=0), so positional modulation is minor for spatial analysis.
            hs_ta = hidden_states[:, text_after_start:, :]            # [B, 17, D]
            hs_vid = hidden_states[:, N_TEXT_BEFORE:N_TEXT_BEFORE + NTOK_Q, :]  # [B, 19440, D]

            q_ta = self_attn.q_proj(hs_ta).view(B, N_TEXT_AFTER, nh, hd).transpose(1, 2).float()
            k_vid = self_attn.k_proj(hs_vid).view(B, NTOK_Q, nkv, hd).transpose(1, 2).float()

            # GQA: expand k
            k_vid = k_vid.repeat_interleave(nkvg, dim=1)  # [B, nh, 19440, hd]

            # Attention scores (no causal mask: text_after > video in seq order)
            scale = hd ** -0.5
            logits = torch.matmul(q_ta, k_vid.transpose(-2, -1)) * scale  # [B, nh, 17, 19440]
            attn_w = logits.float().softmax(dim=-1)
            imp = attn_w.sum(dim=2).sum(dim=1)  # [B, 19440] — sum over heads and text queries
        imp_store[0] = imp.cpu()
        return out

    self_attn.forward = _patched_fwd

    def restore():
        self_attn.forward = orig_fwd

    return imp_store, restore


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}", flush=True)

    files = sorted(glob.glob(os.path.join(PREPROC_CACHE, "*.pt")))
    random.seed(42)
    samp = random.sample(files, min(N_SAMPLE, len(files)))
    print(f"[data] {len(samp)} / {len(files)} samples", flush=True)

    print("[model] loading Qwen2.5-VL-3B...", flush=True)
    t0 = time.time()
    from transformers import Qwen2_5_VLForConditionalGeneration
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(dev)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.config.use_cache = False
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    vis_state, _merge_imp = install_vis_hook(backbone)
    llm_imp_store, llm_restore = install_llm_hook(backbone)

    # Print structure sanity check
    lm = backbone.model.language_model
    attn = lm.layers[0].self_attn
    print(f"[attn] type={type(attn).__name__}  heads={attn.num_heads}  "
          f"kv_heads={attn.num_key_value_heads}  head_dim={attn.head_dim}", flush=True)

    spatial_vis = np.zeros((GRID_Q, GRID_Q))
    temporal_vis = np.zeros(SLOTS)
    spatial_llm = np.zeros((GRID_Q, GRID_Q))
    temporal_llm = np.zeros(SLOTS)

    n_done = 0
    t_start = time.time()

    for fpath in samp:
        t_s = time.time()
        try:
            batch = torch.load(fpath, map_location=dev, weights_only=True)
        except Exception as e:
            print(f"  [skip] {os.path.basename(fpath)}: {e}", flush=True)
            continue

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values_videos=batch["pixel_values_videos"],
                video_grid_thw=batch["video_grid_thw"],
                second_per_grid_ts=batch.get("second_per_grid_ts"),
                mm_token_type_ids=batch.get("mm_token_type_ids"),
                return_dict=True,
            )

        # Vision tower importance
        if vis_state.patch_importance is None:
            print(f"  [warn] no vis importance for {os.path.basename(fpath)}", flush=True)
            continue
        vis_imp = _merge_imp(
            backbone.model.visual, batch["video_grid_thw"], vis_state, dev
        ).float().cpu().numpy()
        vis_state.patch_importance = None
        topk_vis = np.argsort(vis_imp)[-KEEP_K:]
        accum_hard(topk_vis, spatial_vis, temporal_vis)

        # LLM layer-0 importance
        if llm_imp_store[0] is None:
            print(f"  [warn] no LLM importance for {os.path.basename(fpath)}", flush=True)
            continue
        llm_imp = llm_imp_store[0][0].numpy()
        llm_imp_store[0] = None
        accum_soft(llm_imp, spatial_llm, temporal_llm)

        n_done += 1
        if n_done % 5 == 0:
            print(f"  {n_done}/{len(samp)}  {time.time()-t_s:.1f}s/sample  "
                  f"total {time.time()-t_start:.0f}s", flush=True)

    llm_restore()
    vis_state.remove()

    if n_done == 0:
        print("[error] no samples processed", flush=True)
        return

    spatial_vis /= n_done
    temporal_vis /= n_done
    spatial_llm /= n_done
    temporal_llm /= n_done

    np.save(os.path.join(OUT, "spatial_vis.npy"), spatial_vis)
    np.save(os.path.join(OUT, "temporal_vis.npy"), temporal_vis)
    np.save(os.path.join(OUT, "spatial_llm.npy"), spatial_llm)
    np.save(os.path.join(OUT, "temporal_llm.npy"), temporal_llm)

    unif_s = KEEP_K / GP_Q
    unif_t = KEEP_K / SLOTS
    secs = np.arange(SLOTS) / 4.0

    def center_edge(s, lo=3, hi=6):
        c = s[lo:hi, lo:hi].mean()
        n_edge = GP_Q - (hi - lo) ** 2
        e = (s.sum() - s[lo:hi, lo:hi].sum()) / n_edge
        return float(c), float(e)

    def recency(t):
        last10 = secs >= secs.max() - 10
        first10 = secs <= 10
        return float(t[last10].sum() / t.sum()), float(t[first10].sum() / t.sum())

    c_vis, e_vis = center_edge(spatial_vis)
    c_llm, e_llm = center_edge(spatial_llm)
    rb_vis = recency(temporal_vis)
    rb_llm = recency(temporal_llm)

    summary = {
        "n_samples": n_done,
        "keep_k": KEEP_K,
        "total_video_tokens": NTOK_Q,
        "uniform_spatial": round(unif_s, 2),
        "uniform_temporal": round(unif_t, 2),
        "vis_tower_block31_colsum": {
            "spatial_center3x3": round(c_vis, 3),
            "spatial_edge_mean": round(e_vis, 3),
            "center_vs_edge_ratio": round(c_vis / e_vis, 3),
            "frac_kept_last10s": round(rb_vis[0], 4),
            "frac_kept_first10s": round(rb_vis[1], 4),
        },
        "llm_layer0_text_after_video": {
            "spatial_center3x3": round(c_llm, 3),
            "spatial_edge_mean": round(e_llm, 3),
            "center_vs_edge_ratio": round(c_llm / e_llm, 3),
            "frac_kept_last10s": round(rb_llm[0], 4),
            "frac_kept_first10s": round(rb_llm[1], 4),
            "note": "content-only (q_proj@k_proj.T, no M-RoPE)",
        },
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vmax = max(spatial_vis.max(), spatial_llm.max())

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, data, title in [
            (axes[0], spatial_vis,
             f"Vis-tower block-31 col-sum (top-{KEEP_K} hard)\n"
             f"center/edge={c_vis:.2f}/{e_vis:.2f}  ratio={c_vis/e_vis:.3f}"),
            (axes[1], spatial_llm,
             f"LLM layer-0: text-after → video (soft, no-RoPE)\n"
             f"center/edge={c_llm:.2f}/{e_llm:.2f}  ratio={c_llm/e_llm:.3f}"),
        ]:
            im = ax.imshow(data, cmap="magma", vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, label=f"avg importance (uniform≈{unif_s:.1f})")
            ax.set_title(title, fontsize=9); ax.set_xlabel("w"); ax.set_ylabel("h")
        fig.suptitle(f"Qwen spatial: vis-tower vs LLM layer-0 (n={n_done})", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "spatial_comparison.png"), dpi=130); plt.close(fig)

        w_bar = 1 / 4.0 * 0.45
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(secs - w_bar, temporal_vis, width=w_bar, color="#3b7", alpha=0.8, label="vis-tower block-31")
        ax.bar(secs, temporal_llm, width=w_bar, color="#c44", alpha=0.8, label="LLM layer-0 (text→video)")
        ax.axhline(unif_t, color="k", ls="--", lw=1, label=f"uniform ({unif_t:.1f})")
        ax.set_xlabel("time in 60s window (s)"); ax.set_ylabel(f"avg importance / slot")
        ax.set_title(f"Temporal pattern (n={n_done})"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "temporal_comparison.png"), dpi=130); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, label, sv, sl in [
            (axes[0], "center column (w=4)", spatial_vis[:, 4], spatial_llm[:, 4]),
            (axes[1], "edge column (w=0 or 8)",
             (spatial_vis[:, 0] + spatial_vis[:, 8]) / 2,
             (spatial_llm[:, 0] + spatial_llm[:, 8]) / 2),
        ]:
            ax.plot(np.arange(GRID_Q), sv, "o-", color="#3b7", label="vis-tower")
            ax.plot(np.arange(GRID_Q), sl, "s-", color="#c44", label="LLM layer-0")
            ax.axhline(unif_s, color="k", ls="--", lw=1, label=f"uniform ({unif_s:.1f})")
            ax.set_xlabel("h (row)"); ax.set_ylabel("avg importance")
            ax.set_title(label); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "spatial_slice.png"), dpi=130); plt.close(fig)

        print(f"[plots] saved to {OUT}", flush=True)
    except Exception as e:
        print(f"[plots] failed ({e}); arrays saved", flush=True)

    print(f"[done] {n_done} samples in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
