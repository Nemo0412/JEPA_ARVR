#!/usr/bin/env python3
"""Prefill TRI_MODAL_FRAME_CACHE for P01 clip_split (CPU-only).

Decodes each train/val clip once at anticipation=1.0s with anticipation_point=1.0
(matches ``phd_reference`` + util train recipe) so GPU jobs resume with high
cache hit rates and AveUtil can exceed 60%.

Usage:
  export TRI_MODAL_FRAME_CACHE=/scratch/$USER/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1
  export PYTHONPATH=/path/JEPA_ARVR:/path/vjepa2:$PYTHONPATH
  python -m app.hdepic_lora_action_anticipation.prefill_clip_frame_cache \\
    --video-root /scratch/.../hdepic_vjepa_videos \\
    --train-csv .../HD_EPIC_train_vjepa.csv \\
    --val-csv .../HD_EPIC_val_vjepa.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from decord import VideoReader, cpu

from app.hdepic_lora_action_anticipation.clip_frame_cache import cache_path_for, load_or_decode_clip

logger = logging.getLogger("prefill_clip_frame_cache")


def _video_path(video_root: Path, video_id: str, file_format: int = 1) -> Path:
    pid = str(video_id).split("_")[0]
    if file_format == 0:
        return video_root / pid / "videos" / f"{video_id}.MP4"
    return video_root / pid / f"{video_id}.MP4"


def _indices_for_row(
    vr: VideoReader,
    start_frame: int,
    stop_frame: int,
    frames_per_clip: int,
    fps: float,
    anticipation_sec: float,
    anticipation_point: float = 1.0,
) -> np.ndarray:
    vfps = float(vr.get_avg_fps())
    nframes_total = len(vr)
    frame_step = max(1, int(vfps / fps))
    nframes = int(frames_per_clip * frame_step)
    aframes = int(anticipation_sec * vfps)
    sf = int(start_frame)
    ef = int(stop_frame)
    ap = float(anticipation_point)
    # Mirror ClipBalancedDecodeVideosToClips.
    af = int(sf * ap + (1.0 - ap) * ef - aframes)
    indices = np.arange(af - nframes, af, frame_step).astype(np.int64)
    indices[indices < 0] = 0
    if nframes_total > 0:
        indices[indices >= nframes_total] = nframes_total - 1
    return indices


def _prefill_csv(
    csv_path: Path,
    video_root: Path,
    *,
    frames_per_clip: int,
    fps: float,
    anticipation_sec: float,
    anticipation_point: float,
    file_format: int,
) -> tuple[int, int]:
    df = pd.read_csv(csv_path)
    wrote = 0
    skipped = 0
    reader_cache: dict[str, VideoReader] = {}

    for i, row in df.iterrows():
        video_id = str(row["video_id"])
        path = _video_path(video_root, video_id, file_format=file_format)
        if not path.is_file():
            logger.warning("missing video for row %s: %s", i, path)
            continue
        path_s = str(path)
        try:
            if path_s not in reader_cache:
                reader_cache.clear()
                reader_cache[path_s] = VideoReader(path_s, num_threads=1, ctx=cpu(0))
            vr = reader_cache[path_s]
            indices = _indices_for_row(
                vr,
                int(row["start_frame"]),
                int(row["stop_frame"]),
                frames_per_clip,
                fps,
                anticipation_sec,
                anticipation_point=anticipation_point,
            )
            dest = cache_path_for(video_id, indices)
            if dest is not None and dest.is_file():
                skipped += 1
                continue

            def _decode(vr=vr, indices=indices):
                return vr.get_batch(indices).asnumpy().copy()

            load_or_decode_clip(video_id=video_id, indices=indices, decode_fn=_decode)
            wrote += 1
            if wrote % 50 == 0:
                logger.info("prefilled %d clips (skipped existing %d) last=%s", wrote, skipped, video_id)
        except Exception:
            logger.exception("failed row %s video_id=%s path=%s", i, video_id, path)
    return wrote, skipped


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument("--frames-per-clip", type=int, default=32)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--anticipation-sec", type=float, default=1.0)
    p.add_argument("--anticipation-point", type=float, default=1.0)
    p.add_argument("--file-format", type=int, default=1)
    args = p.parse_args(argv)

    root = os.environ.get("TRI_MODAL_FRAME_CACHE", "").strip()
    if not root:
        print("ERROR: set TRI_MODAL_FRAME_CACHE to the cache directory", file=sys.stderr)
        return 2
    Path(root).mkdir(parents=True, exist_ok=True)
    logger.info("Prefilling frame cache at %s", root)

    total_w = total_s = 0
    for csv_path in [args.train_csv] + ([args.val_csv] if args.val_csv else []):
        logger.info("Scanning %s", csv_path)
        w, s = _prefill_csv(
            csv_path,
            args.video_root,
            frames_per_clip=args.frames_per_clip,
            fps=args.fps,
            anticipation_sec=args.anticipation_sec,
            anticipation_point=args.anticipation_point,
            file_format=args.file_format,
        )
        total_w += w
        total_s += s
        logger.info("done %s: wrote=%d skipped=%d", csv_path, w, s)
    logger.info("ALL DONE wrote=%d skipped=%d cache=%s", total_w, total_s, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
