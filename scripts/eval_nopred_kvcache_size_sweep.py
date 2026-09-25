#!/usr/bin/env python3
"""Sweep KV-cache history length vs nopred action top-5.

Newest-aligned 16s window (128 frames @ 8 fps). Last 2s (16 frames) is always
the new stream chunk. KV cache holds the preceding 0/2/…/14 s, streamed in 2s
chunks with global RoPE. Probe sees all latents of that (cache+2s) window.

Also scores one-shot full 16s (bidirectional) as a reference.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

import dump_probe_blk0_selfattn_16s_maps as P  # noqa: E402
from eval_nopred_128_vs_stream16x8 import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    classify,
    encode_full128,
    per_clip_hits,
    strip_hier,
)
from eval_nopred_128_vs_stream_kvcache import rope_qkv  # noqa: E402
from eval_64slot_pred0_prune_vs_last16 import (  # noqa: E402
    Ctx64Dataset,
    collate,
    summarize,
    update_metrics,
)

logger = logging.getLogger("kvcache_sweep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CHUNK_FRAMES = 16
N_FRAMES = 128
CACHE_FRAMES_LIST = [0, 16, 32, 48, 64, 80, 96, 112]


def sample_even_per_video(ds: Ctx64Dataset, clips_per_video: int, max_samples: int) -> Ctx64Dataset:
    by_vid: dict[str, list] = defaultdict(list)
    for r in ds.rows:
        by_vid[str(r["video_id"])].append(r)
    kept = []
    for vid in sorted(by_vid):
        rows = by_vid[vid]
        n_take = min(clips_per_video, len(rows))
        if n_take == 1:
            idxs = [0]
        else:
            idxs = np.linspace(0, len(rows) - 1, n_take).round().astype(int).tolist()
            seen = set()
            uniq = []
            for i in idxs:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            idxs = uniq
        kept.extend(rows[i] for i in idxs)
        if max_samples > 0 and len(kept) >= max_samples:
            kept = kept[:max_samples]
            break
    ds.rows = kept
    logger.info(
        "sampled %d clips from %d videos (clips_per_video=%d)",
        len(kept),
        len({str(r["video_id"]) for r in kept}),
        clips_per_video,
    )
    return ds


@torch.no_grad()
def encode_stream_kv_window(encoder, clips: torch.Tensor, embed_dim: int, chunk_frames: int = CHUNK_FRAMES):
    """Stream clips (T multiple of chunk_frames) with per-layer K/V cache."""
    b, _c, t, h, w = clips.shape
    if t % chunk_frames != 0:
        raise RuntimeError(f"T={t} not divisible by chunk={chunk_frames}")
    gh = h // encoder.patch_size
    gw = w // encoder.patch_size
    n_chunks = t // chunk_frames
    cache_k: list[torch.Tensor | None] = [None] * len(encoder.blocks)
    cache_v: list[torch.Tensor | None] = [None] * len(encoder.blocks)
    out_chunks = []
    n_past = 0
    for ci in range(n_chunks):
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
    tok = strip_hier(torch.cat(out_chunks, dim=1), embed_dim)
    return tok


def prefix_for_cache(cache_frames: int) -> str:
    return f"kv{cache_frames}"


def plot_sweep(out: dict, png_path: Path, copy_png: Path | None):
    xs = out["cache_sec"]
    table = out["table_action_top5"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = {"2s": "#1f4e79", "4s": "#c45911", "6s": "#548235"}
    y_all = []
    for h, c in colors.items():
        ys = [table[f"kv{int(sec * 8)}"][f"@{h}"] for sec in xs]
        y_all.extend(ys)
        ax.plot(xs, ys, marker="o", color=c, lw=2.0, ms=6, label=f"stream KV  +{h}")
        full = table["full128"][f"@{h}"]
        y_all.append(full)
        ax.axhline(full, color=c, ls="--", lw=1.1, alpha=0.75)
    ax.set_xlabel("KV cache history (seconds)")
    ax.set_ylabel("Action Top-5 (%)")
    n = out["n_clips"]
    nvid = out["config"]["n_videos"]
    ax.set_title(f"No-predictor  ·  KV cache size vs accuracy   ({n} clips, {nvid} videos)")
    ax.set_xticks(xs)
    ymax = max([y for y in y_all if y == y], default=30)
    ax.set_ylim(0, max(35, 5 + ymax))
    ax.grid(True, axis="y", alpha=0.35)
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#555555", ls="--", lw=1.1))
    labels.append("one-shot 16s (bidirectional)")
    ax.legend(handles, labels, loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    if copy_png is not None:
        copy_png.parent.mkdir(parents=True, exist_ok=True)
        copy_png.write_bytes(png_path.read_bytes())
        logger.info("copied plot %s", copy_png)
    logger.info("wrote plot %s", png_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", type=Path, default=Path(
        "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/HD_EPIC_val_stream_mtp.csv"
    ))
    ap.add_argument("--video-root", type=Path, default=Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"))
    ap.add_argument("--checkpoint", type=Path, default=Path("/scratch/ll5914/models/vjepa2/vitl.pt"))
    ap.add_argument("--nopred-ckpt", type=Path, default=P.NOPRED_CKPT)
    ap.add_argument("--out-dir", type=Path, default=Path(
        "/scratch/ll5914/experiments/nopred_kvcache_size_sweep"
    ))
    ap.add_argument("--copy-json", type=Path, default=Path(
        "/home/ll5914/Jepa_yifan/nopred_kvcache_size_sweep.json"
    ))
    ap.add_argument("--copy-png", type=Path, default=Path(
        "/home/ll5914/Jepa_yifan/nopred_kvcache_size_sweep.png"
    ))
    ap.add_argument("--context-sec", type=float, default=16.0)
    ap.add_argument("--require-ctx-sec", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=108)
    ap.add_argument("--clips-per-video", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",") if x.strip()]
    n_frames = max(2, int(round(args.context_sec * args.fps)))
    if n_frames % 2:
        n_frames += 1
    if n_frames != N_FRAMES:
        raise ValueError(f"need {N_FRAMES} frames, got {n_frames}")

    ds = Ctx64Dataset(
        args.val_csv,
        args.video_root,
        context_sec=args.context_sec,
        model_fps=args.fps,
        img_size=args.img_size,
        max_samples=0,
        stride=1,
        require_ctx_sec=args.require_ctx_sec if args.require_ctx_sec > 0 else None,
    )
    ds = sample_even_per_video(ds, args.clips_per_video, args.max_samples)
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

    totals = defaultdict(float)
    counts = defaultdict(int)
    t0 = time.time()
    n_videos = len({str(r["video_id"]) for r in ds.rows})

    with torch.no_grad():
        for it, batch in enumerate(loader):
            clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
            clips = clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))
            batch_dev = {
                "mtp_verbs": batch["mtp_verbs"].to(device),
                "mtp_nouns": batch["mtp_nouns"].to(device),
                "mtp_mask": batch["mtp_mask"].to(device),
            }
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                tok_full = encode_full128(encoder, clips, embed_dim)
                out_full = classify(mtp_clf, tok_full)
                update_metrics(
                    totals, counts, out_full, batch_dev, horizons, verb_map, noun_map, action_map, device, prefix="full128"
                )
                for cache_frames in CACHE_FRAMES_LIST:
                    win_t = cache_frames + CHUNK_FRAMES
                    window = clips[:, :, -win_t:]
                    tok = encode_stream_kv_window(encoder, window, embed_dim)
                    outs = classify(mtp_clf, tok)
                    update_metrics(
                        totals,
                        counts,
                        outs,
                        batch_dev,
                        horizons,
                        verb_map,
                        noun_map,
                        action_map,
                        device,
                        prefix=prefix_for_cache(cache_frames),
                    )

            if it % args.log_every == 0:
                partial = summarize(totals, counts)
                msg = {
                    k: round(100.0 * v, 2)
                    for k, v in partial.items()
                    if k.endswith("/action_top5@2s") and not str(k).startswith("n|")
                }
                logger.info("itr=%d/%d elapsed=%.0fs %s", it, len(loader), time.time() - t0, msg)

    metrics = summarize(totals, counts)
    cache_sec = [c / args.fps for c in CACHE_FRAMES_LIST]
    prefixes = ["full128"] + [prefix_for_cache(c) for c in CACHE_FRAMES_LIST]
    table = {}
    for prefix in prefixes:
        table[prefix] = {
            f"@{h:g}s": round(100.0 * metrics.get(f"{prefix}/action_top5@{h:g}s", float("nan")), 4)
            for h in horizons
        }
        table[prefix]["n@2s"] = int(metrics.get(f"n|{prefix}/action_top5@2s", 0))

    out = {
        "cache_sec": cache_sec,
        "cache_frames": CACHE_FRAMES_LIST,
        "table_action_top5": table,
        "n_clips": len(ds),
        "config": {
            "no_predictor": True,
            "chunk_frames": CHUNK_FRAMES,
            "n_frames_full": N_FRAMES,
            "clips_per_video": args.clips_per_video,
            "n_videos": n_videos,
            "newest_aligned": True,
            "probe_sees": "all_latents_in_cache_plus_new_2s",
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
    plot_sweep(out, args.out_dir / "kvcache_size_vs_top5.png", args.copy_png)
    logger.info("wrote %s", out_path)
    logger.info("table %s", json.dumps(table))


if __name__ == "__main__":
    main()
