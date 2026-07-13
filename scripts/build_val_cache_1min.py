#!/usr/bin/env python
"""Pre-build the Qwen2.5-VL preprocessing cache for the val set (480-frame 1-min config).

Runs Qwen's processor on each val sample (decode 480 frames + tokenize) and saves the
result to disk — identical to what train_vlm_probe_lora.py does on first encounter.
No GPU required: decord + Qwen processor run on CPU.

Usage (inside singularity overlay):
  python scripts/build_val_cache_1min.py \
      --val-csv data/hdepic_vjepa_annotations/phd_split/HD_EPIC_val_vjepa.csv \
      --video-root data/hdepic_vjepa_videos \
      --cache-dir data/preproc_cache_qwen/qwen25vl_nf480_pnf480_fps8.0_px256 \
      --num-workers 12
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _cache_path(cache_dir: str, row: dict) -> str:
    return os.path.join(
        cache_dir,
        f"{row['participant_id']}__{row.get('video_id', 'x')}__{int(row['start_frame'])}.pt",
    )


def _build_one(row: dict, video_root: str, cache_dir: str,
               processor, decode_size: int = 256, num_frames: int = 480,
               target_fps: float = 8.0) -> str:
    from decord import VideoReader, cpu
    from app.hdepic_lora_action_anticipation.zeroshot_vlm_prompting import (
        compute_clip_window, decode_frames,
    )
    from app.hdepic_lora_action_anticipation.train_vlm_probe_lora import (
        BACKEND_BATCH_BUILDERS, _resample_frames,
    )

    out_path = _cache_path(cache_dir, row)
    if os.path.exists(out_path):
        return f"SKIP {out_path}"

    video_id = row.get("video_id", "?")
    video_path = str(Path(video_root) / row["participant_id"] / f"{video_id}.MP4")
    vr_probe = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    vfps = vr_probe.get_avg_fps()
    indices = compute_clip_window(int(row["start_frame"]), vfps, num_frames, target_fps)
    frames, _ = decode_frames(video_path, indices, decode_size=decode_size)
    frames = _resample_frames(frames, num_frames)

    inputs = BACKEND_BATCH_BUILDERS["qwen25vl"](processor, [frames])
    cached = {
        k: (v.half() if (k == "pixel_values_videos" and torch.is_floating_point(v)) else v)
        for k, v in dict(inputs).items()
    }

    tmp = f"{out_path}.tmp.{os.getpid()}"
    torch.save(cached, tmp)
    os.replace(tmp, out_path)
    return f"OK {out_path}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--num-frames", type=int, default=480)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--decode-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    with open(args.val_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # Check how many already exist
    existing = sum(1 for r in rows if os.path.exists(_cache_path(args.cache_dir, r)))
    logger.info("%d / %d samples already cached; building remaining %d",
                existing, len(rows), len(rows) - existing)
    if existing == len(rows):
        logger.info("All cached. Done.")
        return

    # Load processor on CPU
    from transformers import AutoProcessor
    import app.hdepic_lora_action_anticipation.train_vlm_probe_lora as train_mod
    train_mod._QWEN_FRAME_SIZE = args.decode_size
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", local_files_only=True
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    t0 = time.time()
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futs = {
            pool.submit(
                _build_one, row, args.video_root, args.cache_dir,
                processor, args.decode_size, args.num_frames, args.target_fps
            ): row
            for row in rows
        }
        for fut in as_completed(futs):
            try:
                result = fut.result()
                done += 1
                if done % 50 == 0 or done == len(rows):
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (len(rows) - done) / max(rate, 1e-6)
                    logger.info("  %d / %d done (%.1f s/sample, ETA %.0f min)",
                                done, len(rows), elapsed / done, eta / 60)
            except Exception as exc:
                errors += 1
                logger.warning("Error on %s: %s", futs[fut].get("video_id", "?"), exc)

    elapsed = time.time() - t0
    logger.info("Finished: %d ok, %d errors in %.1f min", done - errors, errors, elapsed / 60)


if __name__ == "__main__":
    main()
