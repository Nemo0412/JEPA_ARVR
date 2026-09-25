#!/usr/bin/env python3
"""Latency overhead of probe-entry PE on 128 KV-cache + 34 new frames.

Steady-state: cache 128f, encode 34f vs remaining K/V, attn-prune to 128,
then probe. Compare no-PE vs rel_rank PE (the scheme used under repeated prune).

Synthetic clips, bf16, CUDA events, L40S.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
os.environ.setdefault("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
sys.path.insert(0, os.environ["VJEPA_ROOT"])

import dump_probe_blk0_selfattn_16s_maps as P  # noqa: E402
from app.hdepic_lora_action_anticipation.stream_kvcache_attn_prune import (  # noqa: E402
    CACHE_FRAMES,
    NEW_FRAMES,
    StreamKVAttnPruneEncoder,
    add_probe_positional_encoding,
)
from eval_probe_posenc_kvcache_128_34 import GRID, HORIZONS, classify_independent  # noqa: E402

logger = logging.getLogger("probe_pe_lat")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _stats(xs: list[float]) -> dict:
    xs_sorted = sorted(float(x) for x in xs)
    n = len(xs_sorted)
    return {
        "n": n,
        "mean_ms": float(statistics.fmean(xs_sorted)),
        "median_ms": float(statistics.median(xs_sorted)),
        "p90_ms": float(xs_sorted[max(0, int(0.9 * (n - 1)))]),
        "min_ms": float(xs_sorted[0]),
        "max_ms": float(xs_sorted[-1]),
    }


def _cuda_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("/scratch/ll5914/models/vjepa2/vitl.pt"))
    ap.add_argument("--nopred-ckpt", type=Path, default=P.NOPRED_CKPT)
    ap.add_argument("--out-dir", type=Path, default=Path("/scratch/ll5914/experiments/probe_posenc_kvcache_128_34"))
    ap.add_argument("--copy-json", type=Path, default=Path("/home/ll5914/Jepa_yifan/probe_posenc_latency.json"))
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_frames = CACHE_FRAMES + NEW_FRAMES
    logger.info("loading nopred ViT-L for PE latency  T=%d", total_frames)
    model, mtp_clf, _pooler, _ck = P.load_nopred_pooler(
        device, total_frames, args.fps, args.img_size, args.nopred_ckpt, str(args.checkpoint)
    )
    encoder = model.base.encoder
    embed_dim = int(encoder.embed_dim)
    gp = int(getattr(model.base, "grid_size", GRID) ** 2)
    stream = StreamKVAttnPruneEncoder(
        encoder, gp=gp, tubelet_size=2, cache_frames=CACHE_FRAMES, new_frames=NEW_FRAMES, chunk=args.chunk
    )
    encoder.eval()
    mtp_clf.eval()

    clips = torch.randn(1, 3, total_frames, args.img_size, args.img_size, device=device)
    hist = clips[:, :, :CACHE_FRAMES]
    new = clips[:, :, CACHE_FRAMES:]

    logger.info("warmup fill+step …")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fill_state = stream.fill(hist)
        tok, slot_ids, pos_ids = None, None, None

        def tokens_from_filled():
            state = stream.step(fill_state, new, mode="attn")
            t = state.tokens
            if t.size(-1) != embed_dim:
                t = t[:, :, -embed_dim:]
            return t, state.slot_ids, state.pos, state

        tok, slot_ids, pos_ids, _st = tokens_from_filled()
        _ = classify_independent(mtp_clf, tok, HORIZONS)
        _ = classify_independent(
            mtp_clf,
            add_probe_positional_encoding(tok, slot_ids, gp, pos_ids=pos_ids, grid_size=GRID, mode="rel_rank"),
            HORIZONS,
        )

    parts: dict = {}

    def pe_only():
        parts["tok_pe"] = add_probe_positional_encoding(
            tok, slot_ids, gp, pos_ids=pos_ids, grid_size=GRID, mode="rel_rank"
        )

    def probe_no_pe():
        parts["out"] = classify_independent(mtp_clf, tok, HORIZONS)

    def probe_rel_rank():
        tok_pe = add_probe_positional_encoding(
            tok, slot_ids, gp, pos_ids=pos_ids, grid_size=GRID, mode="rel_rank"
        )
        parts["out"] = classify_independent(mtp_clf, tok_pe, HORIZONS)

    def encode34_prune():
        parts["state"] = stream.step(fill_state, new, mode="attn")

    def e2e_no_pe():
        state = stream.step(fill_state, new, mode="attn")
        t = state.tokens
        if t.size(-1) != embed_dim:
            t = t[:, :, -embed_dim:]
        classify_independent(mtp_clf, t, HORIZONS)

    def e2e_rel_rank():
        state = stream.step(fill_state, new, mode="attn")
        t = state.tokens
        if t.size(-1) != embed_dim:
            t = t[:, :, -embed_dim:]
        t = add_probe_positional_encoding(
            t, state.slot_ids, gp, pos_ids=state.pos, grid_size=GRID, mode="rel_rank"
        )
        classify_independent(mtp_clf, t, HORIZONS)

    keys = {
        "pe_rel_rank_ms": pe_only,
        "probe_no_pe_ms": probe_no_pe,
        "probe_rel_rank_ms": probe_rel_rank,
        "encode34_prune_ms": encode34_prune,
        "e2e_no_pe_ms": e2e_no_pe,
        "e2e_rel_rank_ms": e2e_rel_rank,
    }
    results = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name, fn in keys.items():
            logger.info("warmup %s …", name)
            for _ in range(args.warmup):
                fn()
            rows = [_cuda_ms(fn) for _ in range(args.repeats)]
            results[name] = _stats(rows)
            logger.info("  %s median %.3f ms", name, results[name]["median_ms"])

    pe = results["pe_rel_rank_ms"]["median_ms"]
    probe0 = results["probe_no_pe_ms"]["median_ms"]
    probe1 = results["probe_rel_rank_ms"]["median_ms"]
    e0 = results["e2e_no_pe_ms"]["median_ms"]
    e1 = results["e2e_rel_rank_ms"]["median_ms"]
    payload = {
        "device": torch.cuda.get_device_name(0),
        "cache_frames": CACHE_FRAMES,
        "new_frames": NEW_FRAMES,
        "n_probe_tokens": int(tok.size(1)),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
        "delta_median_ms": {
            "pe_add": pe,
            "probe_rel_rank_minus_no_pe": probe1 - probe0,
            "e2e_rel_rank_minus_no_pe": e1 - e0,
            "pe_over_probe_no_pe_pct": 100.0 * pe / max(probe0, 1e-6),
            "pe_over_e2e_no_pe_pct": 100.0 * pe / max(e0, 1e-6),
        },
    }
    out_path = args.out_dir / "latency.json"
    text = json.dumps(payload, indent=2) + "\n"
    out_path.write_text(text, encoding="utf-8")
    if args.copy_json is not None:
        args.copy_json.parent.mkdir(parents=True, exist_ok=True)
        args.copy_json.write_text(text, encoding="utf-8")
    logger.info("wrote %s", out_path)
    logger.info("deltas %s", json.dumps(payload["delta_median_ms"]))


if __name__ == "__main__":
    main()
