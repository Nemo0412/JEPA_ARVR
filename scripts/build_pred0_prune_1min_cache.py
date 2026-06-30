#!/usr/bin/env python
"""Build predictor-block-0-guided pruning cache from 1min full-context encoder cache.

Input : data/preproc_cache_vjepa/nf480_fps8.0_px256_keep61440_idx/
         each file = {tok: [61440, 1024] fp16, idx: [61440] int32}
         where idx[j] is the original encoder token position for tok[j].

Process: For each sample, run predictor block-0 with re-based positions
         0..61439 (OOD compared to predictor's training range 0..4095, but
         the same OOD regime seen in the 1min-block-11 visualization).
         Compute attention column-sum importance [61440] → select top KEEP_K=4096.

Output : data/preproc_cache_vjepa/nf480_fps8.0_px256_pred0_keep4096_idx/
          each file = {tok: [4096, 1024] fp16, idx: [4096] int32}
          (encoder tokens + original positions for the 4096 kept tokens)

This cache is consumed by train_vjepa_prune_anticipation.py via
--cache-key-override nf480_fps8.0_px256_pred0_keep4096_idx.

Runtime: ~2.4 s/sample → ~7375 total ≈ 5 h (use 4-shard job array, ~75 min each).
"""
from __future__ import annotations

import glob
import os
import sys
import time

import torch
import torch.nn.functional as F

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/path/to/VJEPA2-EXP")
CODE_ROOT = os.environ.get("PROJECT_ROOT", SHARED)
sys.path.insert(0, os.path.join(CODE_ROOT, "vjepa2"))
sys.path.insert(0, CODE_ROOT)

CKPT = os.path.join(SHARED, "checkpoints/vitl.pt")
SRC_CACHE = os.path.join(SHARED, "data/preproc_cache_vjepa/nf480_fps8.0_px256_keep61440_idx")
OUT_KEY = "nf480_fps8.0_px256_pred0_keep4096_idx"
DST_CACHE = os.path.join(SHARED, "data/preproc_cache_vjepa", OUT_KEY)

NTOK = 61440       # tokens in source cache per sample
KEEP_K = 4096      # tokens to keep
CHUNK_SIZE = 256   # attention chunking to control peak memory
GRID = 16
GP = GRID * GRID   # 256 spatial patches per temporal slot

SHARD = int(os.environ.get("CACHE_SHARD", "0"))
NUM_SHARDS = int(os.environ.get("CACHE_NUM_SHARDS", "1"))


def make_pred0_hook(blk_attn, chunk_size: int):
    """Hook predictor block-0 RoPEAttention to capture col-sum importance."""
    from src.models.utils.modules import rotate_queries_or_keys  # noqa

    m = blk_attn
    orig = m.forward
    imp_store: list = [None]

    def _fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
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


def build_predictor(device):
    import app.hdepic_lora_action_anticipation.train_vjepa_prune_anticipation as M
    model = M.build_model(device, frames_per_clip=480, fps=8, img_size=256, checkpoint=CKPT)
    # Lift num_patches so separate_positions doesn't error on positions 0..61439
    n_pred = GP * round(1.0 * 8 / 2)   # 1s future = 4 slots × 256
    new_cap = ((NTOK + n_pred) // GP + 8) * GP
    model.predictor.num_patches = new_cap
    print(f"  predictor.num_patches lifted to {new_cap}", flush=True)
    return model.predictor


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}", flush=True)
    print(f"[shard] {SHARD}/{NUM_SHARDS}  keep_k={KEEP_K}", flush=True)

    files = sorted(glob.glob(os.path.join(SRC_CACHE, "*.pt")))
    if not files:
        raise RuntimeError(f"No .pt files found in {SRC_CACHE}")
    files = [f for i, f in enumerate(files) if i % NUM_SHARDS == SHARD]
    print(f"[data] {len(files)} files in this shard", flush=True)

    os.makedirs(DST_CACHE, exist_ok=True)

    print("[model] loading predictor...", flush=True)
    t0 = time.time()
    predictor = build_predictor(dev)
    predictor.eval()
    blk0_attn = predictor.predictor_blocks[0].attn
    orig_fwd, imp_store = make_pred0_hook(blk0_attn, CHUNK_SIZE)
    print(f"  loaded+hooked in {time.time()-t0:.1f}s", flush=True)

    ctx_pos = torch.arange(NTOK, device=dev).unsqueeze(0)   # [1, 61440]

    n_done = n_skip = 0
    t_start = time.time()

    for fpath in files:
        fname = os.path.basename(fpath)
        dst = os.path.join(DST_CACHE, fname)
        if os.path.exists(dst):
            n_done += 1
            continue

        try:
            o = torch.load(fpath, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  [skip-load] {fname}: {e}", flush=True)
            n_skip += 1
            continue

        tok_fp16 = o["tok"]   # [61440, 1024] fp16
        orig_idx = o["idx"]   # [61440] int32 — original encoder positions
        if tok_fp16.shape[0] != NTOK:
            print(f"  [skip-shape] {fname}: tok.shape={tok_fp16.shape}", flush=True)
            n_skip += 1
            continue

        tok = tok_fp16.unsqueeze(0).to(device=dev, dtype=torch.bfloat16)  # [1, 61440, 1024]

        t_s = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = predictor.predictor_embed(tok)             # [1, 61440, 384]
            predictor.predictor_blocks[0](x, mask=ctx_pos, attn_mask=None)

        if imp_store[0] is None:
            print(f"  [warn-no-imp] {fname}", flush=True)
            n_skip += 1
            imp_store[0] = None
            del tok, x
            continue

        imp = imp_store[0][0]   # [61440] float32 on CPU
        imp_store[0] = None

        # Top-KEEP_K local indices (sorted ascending to preserve temporal order)
        top_local = imp.topk(KEEP_K).indices.sort().values   # [4096]

        # Map to original encoder positions
        kept_tok = tok_fp16[top_local]                                  # [4096, 1024] fp16
        kept_idx = orig_idx[top_local].to(torch.int32)                  # [4096] int32

        tmp = dst + f".tmp{os.getpid()}"
        torch.save({"tok": kept_tok, "idx": kept_idx}, tmp)
        os.replace(tmp, dst)

        del tok, x, imp, top_local, kept_tok, kept_idx

        n_done += 1
        if n_done % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  done={n_done}  skip={n_skip}  {elapsed/(n_done+n_skip):.1f}s/sample  "
                  f"total {elapsed:.0f}s", flush=True)

    blk0_attn.forward = orig_fwd
    elapsed = time.time() - t_start
    print(f"\n[done] shard={SHARD}/{NUM_SHARDS}  done={n_done}  skip={n_skip}  "
          f"total={elapsed:.0f}s  avg={elapsed/max(n_done+n_skip,1):.1f}s/sample", flush=True)
    print(f"[output] {DST_CACHE}", flush=True)


if __name__ == "__main__":
    main()
