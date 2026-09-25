#!/usr/bin/env python3
"""Paired nopred eval: one-shot 128-frame encode vs streaming KV-cache + last 16.

Same 16s RGB window (128 frames @ 8 fps) and same MTP probe heads.

  full128      encoder self-attn over all 64 slots (16384 tokens) at once
  stream_kv    16-frame chunks with a per-layer K/V cache:
                 - first 112 frames (7×16) already in cache (global RoPE 0..55)
                 - new 16 frames attend to cached K/V + themselves (RoPE 56..63)
                 - past hidden states stay frozen (cannot see the new chunk)

No predictor. Weights: p01_stream_mtp_nopred_vanilla_2_4_6.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
os.environ.setdefault("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
sys.path.insert(0, os.environ["VJEPA_ROOT"])

from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

import dump_probe_blk0_selfattn_16s_maps as P  # noqa: E402
from eval_nopred_128_vs_stream16x8 import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    classify,
    encode_full128,
    keep_unique_videos,
    per_clip_hits,
    strip_hier,
)
from eval_64slot_pred0_prune_vs_last16 import (  # noqa: E402
    Ctx64Dataset,
    collate,
    summarize,
    update_metrics,
)

logger = logging.getLogger("eval_128_vs_kv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CHUNK_FRAMES = 16
N_CHUNKS = 8
N_FRAMES = CHUNK_FRAMES * N_CHUNKS  # 128
CACHE_FRAMES = N_FRAMES - CHUNK_FRAMES  # 112 = 7×16 (128-16)


def rope_qkv(attn, x: torch.Tensor, pos_ids: torch.Tensor, h_patches: int, w_patches: int):
    """Q/K/V after RoPE. x: [B,N,C], pos_ids: [B,N] global token indices."""
    b, n, _c = x.size()
    qkv = attn.qkv(x).unflatten(-1, (3, attn.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    mask = pos_ids.unsqueeze(1).expand(b, attn.num_heads, n)
    d_mask, h_mask, w_mask = attn.separate_positions(mask, h_patches, w_patches)
    s = 0
    qd = rotate_queries_or_keys(q[..., s : s + attn.d_dim], pos=d_mask)
    kd = rotate_queries_or_keys(k[..., s : s + attn.d_dim], pos=d_mask)
    s += attn.d_dim
    qh = rotate_queries_or_keys(q[..., s : s + attn.h_dim], pos=h_mask)
    kh = rotate_queries_or_keys(k[..., s : s + attn.h_dim], pos=h_mask)
    s += attn.h_dim
    qw = rotate_queries_or_keys(q[..., s : s + attn.w_dim], pos=w_mask)
    kw = rotate_queries_or_keys(k[..., s : s + attn.w_dim], pos=w_mask)
    s += attn.w_dim
    if s < attn.head_dim:
        q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
        k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
    else:
        q = torch.cat([qd, qh, qw], dim=-1)
        k = torch.cat([kd, kh, kw], dim=-1)
    return q, k, v


@torch.no_grad()
def encode_stream_kvcache(encoder, clips: torch.Tensor, embed_dim: int, chunk_frames: int = CHUNK_FRAMES):
    """Incremental 16-frame stream with per-layer K/V cache and global RoPE.

    Returns (tokens [B,16384,D], seconds_fill_cache, seconds_last_chunk).
    """
    b, _c, t, h, w = clips.shape
    if t != N_FRAMES:
        raise RuntimeError(f"expected T={N_FRAMES}, got {t}")
    gh = h // encoder.patch_size
    gw = w // encoder.patch_size
    n_chunks = t // chunk_frames
    cache_k: list[torch.Tensor | None] = [None] * len(encoder.blocks)
    cache_v: list[torch.Tensor | None] = [None] * len(encoder.blocks)
    out_chunks = []
    n_past = 0
    t_fill = t_last = 0.0
    device = clips.device

    for ci in range(n_chunks):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        chunk = clips[:, :, ci * chunk_frames : (ci + 1) * chunk_frames]
        x = encoder.patch_embed(chunk)
        n_new = x.size(1)
        pos = torch.arange(n_past, n_past + n_new, device=x.device).unsqueeze(0).expand(b, -1)
        for li, blk in enumerate(encoder.blocks):
            attn = blk.attn
            q, k, v = rope_qkv(attn, blk.norm1(x), pos, gh, gw)
            if cache_k[li] is not None:
                k_all = torch.cat([cache_k[li], k], dim=2)
                v_all = torch.cat([cache_v[li], v], dim=2)
            else:
                k_all, v_all = k, v
            y = F.scaled_dot_product_attention(q, k_all, v_all, dropout_p=0.0, is_causal=False)
            y = y.transpose(1, 2).reshape(b, n_new, -1)
            y = attn.proj_drop(attn.proj(y))
            x = x + blk.drop_path(y)
            x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
            cache_k[li] = k_all
            cache_v[li] = v_all
        if encoder.norm is not None:
            x = encoder.norm(x)
        out_chunks.append(x)
        n_past += n_new
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t1
        if ci < n_chunks - 1:
            t_fill += dt
        else:
            t_last += dt

    tok = strip_hier(torch.cat(out_chunks, dim=1), embed_dim)
    if tok.size(1) != n_chunks * (chunk_frames // encoder.tubelet_size) * gh * gw:
        raise RuntimeError(f"stream_kv token count {tuple(tok.shape)}")
    return tok, t_fill, t_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", type=Path, default=Path(
        "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/HD_EPIC_val_stream_mtp.csv"
    ))
    ap.add_argument("--video-root", type=Path, default=Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"))
    ap.add_argument("--checkpoint", type=Path, default=Path("/scratch/ll5914/models/vjepa2/vitl.pt"))
    ap.add_argument("--nopred-ckpt", type=Path, default=P.NOPRED_CKPT)
    ap.add_argument("--out-dir", type=Path, default=Path(
        "/scratch/ll5914/experiments/nopred_128_vs_stream_kvcache_10clips"
    ))
    ap.add_argument("--copy-json", type=Path, default=Path(
        "/home/ll5914/Jepa_yifan/nopred_128_vs_stream_kvcache_10clips.json"
    ))
    ap.add_argument("--context-sec", type=float, default=16.0)
    ap.add_argument("--require-ctx-sec", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--unique-videos", action="store_true", default=True)
    ap.add_argument("--no-unique-videos", action="store_false", dest="unique_videos")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",") if x.strip()]
    n_frames = max(2, int(round(args.context_sec * args.fps)))
    if n_frames % 2:
        n_frames += 1
    if n_frames != N_FRAMES:
        raise ValueError(f"this script is for {N_FRAMES} frames, got {n_frames}")

    ds = Ctx64Dataset(
        args.val_csv,
        args.video_root,
        context_sec=args.context_sec,
        model_fps=args.fps,
        img_size=args.img_size,
        max_samples=0,
        stride=args.stride,
        require_ctx_sec=args.require_ctx_sec if args.require_ctx_sec > 0 else None,
    )
    if args.unique_videos:
        ds = keep_unique_videos(ds, args.max_samples)
    else:
        ds.rows = ds.rows[: args.max_samples]
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=False
    )

    model, mtp_clf, _pooler, ck_meta = P.load_nopred_pooler(
        device, n_frames, args.fps, args.img_size, args.nopred_ckpt, str(args.checkpoint)
    )
    verb_map, noun_map, action_map = ck_meta["verb_map"], ck_meta["noun_map"], ck_meta["action_map"]
    encoder = model.base.encoder
    embed_dim = int(encoder.embed_dim)
    encoder.eval()
    mtp_clf.eval()
    logger.info(
        "compare full128 vs stream_kv cache=%d frames then +%d (tubelet=%d patch=%d)",
        CACHE_FRAMES,
        CHUNK_FRAMES,
        encoder.tubelet_size,
        encoder.patch_size,
    )

    totals = defaultdict(float)
    counts = defaultdict(int)
    per_clip = []
    t0 = time.time()
    t_full = t_fill = t_last = 0.0

    with torch.no_grad():
        for it, batch in enumerate(loader):
            clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
            clips = clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))
            batch_dev = {
                "mtp_verbs": batch["mtp_verbs"].to(device),
                "mtp_nouns": batch["mtp_nouns"].to(device),
                "mtp_mask": batch["mtp_mask"].to(device),
            }
            vid = batch["video_id"][0]
            tick = int(batch["tick_frame"][0])

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                tok_full = encode_full128(encoder, clips, embed_dim)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_full += time.time() - t1

                tok_stream, dt_fill, dt_last = encode_stream_kvcache(encoder, clips, embed_dim)
                t_fill += dt_fill
                t_last += dt_last

                out_full = classify(mtp_clf, tok_full)
                out_stream = classify(mtp_clf, tok_stream)

            update_metrics(
                totals, counts, out_full, batch_dev, horizons, verb_map, noun_map, action_map, device, prefix="full128"
            )
            update_metrics(
                totals, counts, out_stream, batch_dev, horizons, verb_map, noun_map, action_map, device, prefix="stream_kv"
            )

            cos = (
                torch.nn.functional.cosine_similarity(
                    tok_full.float().flatten(1), tok_stream.float().flatten(1), dim=1
                )
                .mean()
                .item()
            )
            rec = {
                "idx": it,
                "video_id": vid,
                "tick_frame": tick,
                "token_cosine": round(float(cos), 6),
                "full128": per_clip_hits(out_full, batch_dev, horizons, verb_map, noun_map, action_map, device),
                "stream_kv": per_clip_hits(out_stream, batch_dev, horizons, verb_map, noun_map, action_map, device),
            }
            per_clip.append(rec)
            logger.info(
                "clip %d/%d %s tick=%d cos=%.4f full@2s_top5=%s stream_kv@2s_top5=%s",
                it + 1,
                len(loader),
                vid,
                tick,
                cos,
                rec["full128"]["2s"]["hit_top5"],
                rec["stream_kv"]["2s"]["hit_top5"],
            )

    metrics = summarize(totals, counts)
    table = {}
    for prefix in ("full128", "stream_kv"):
        table[prefix] = {
            f"@{h:g}s": round(100.0 * metrics.get(f"{prefix}/action_top5@{h:g}s", float("nan")), 4)
            for h in horizons
        }
    delta = {
        f"stream_kv_minus_full128@{h:g}s": round(
            100.0
            * (
                metrics.get(f"stream_kv/action_top5@{h:g}s", float("nan"))
                - metrics.get(f"full128/action_top5@{h:g}s", float("nan"))
            ),
            4,
        )
        for h in horizons
    }
    n_agree_2s = sum(
        1
        for r in per_clip
        if r["full128"]["2s"]["hit_top5"] is not None
        and r["full128"]["2s"]["hit_top5"] == r["stream_kv"]["2s"]["hit_top5"]
    )
    winner = "tie"
    a = table["full128"].get("@2s", float("nan"))
    b = table["stream_kv"].get("@2s", float("nan"))
    if a > b:
        winner = "full128"
    elif b > a:
        winner = "stream_kv"

    out = {
        "winner_action_top5@2s": winner,
        "table_action_top5": table,
        "delta_stream_minus_full": delta,
        "mean_token_cosine": round(float(np.mean([r["token_cosine"] for r in per_clip])), 6) if per_clip else None,
        "n_clips": len(per_clip),
        "n_agree_top5@2s": n_agree_2s,
        "encode_seconds": {
            "full128": round(t_full, 3),
            "stream_kv_fill_112": round(t_fill, 3),
            "stream_kv_last16": round(t_last, 3),
        },
        "per_clip": per_clip,
        "config": {
            "no_predictor": True,
            "n_frames_full": N_FRAMES,
            "cache_frames": CACHE_FRAMES,
            "new_chunk_frames": CHUNK_FRAMES,
            "n_chunks": N_CHUNKS,
            "stream_rope": "global_incremental_kvcache",
            "unique_videos": bool(args.unique_videos),
            "ckpt": str(args.nopred_ckpt),
            "ckpt_meta": {k: ck_meta[k] for k in ("epoch", "step", "phase")},
        },
        "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
        "seconds": time.time() - t0,
    }
    out_path = args.out_dir / "metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    args.copy_json.parent.mkdir(parents=True, exist_ok=True)
    args.copy_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_path)
    logger.info("table %s", json.dumps(table))
    logger.info("delta %s winner@2s=%s", json.dumps(delta), winner)


if __name__ == "__main__":
    main()
