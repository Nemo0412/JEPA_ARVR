#!/usr/bin/env python3
"""HD-EPIC streaming index: entire-scene ticks, 4s context, +2s label.

Sampling is the same algorithm as
``/home/ll5914/Jepa/JEPA_ARVR/scripts/make_hdepic_stream_half_split.py``:
temporal half-split per video, tick every 2s from origin+4s. The only protocol
change is a *fixed* 4s window (no 4→10s growth) and a single +2s horizon.

Clip intervals come from the all-participant HD-EPIC pool
(``clip_split_all``, P01–P09). Train/val are *not* the clip 80/20 split;
they are first-half / second-half of each full video (the scene).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from decord import VideoReader, cpu

JEPA_SCRIPTS = Path("/home/ll5914/Jepa/JEPA_ARVR/scripts")
sys.path.insert(0, str(JEPA_SCRIPTS))
sys.path.insert(0, str(Path("/home/ll5914/Jepa/JEPA_ARVR")))

from make_hdepic_stream_half_split import (  # noqa: E402
    _video_path,
    build_intervals_from_clips,
    build_rows_for_video,
)

DEFAULT_CLIPS = Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split_all")
DEFAULT_VIDEO_ROOT = Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos")
DEFAULT_OUT = Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_4s2s_all")


def _pool_all_clips(src: Path) -> pd.DataFrame:
    dfs = []
    for name in ("HD_EPIC_train_vjepa.csv", "HD_EPIC_val_vjepa.csv", "HD_EPIC_test_vjepa.csv"):
        path = src / name
        if path.is_file():
            dfs.append(pd.read_csv(path))
    if not dfs:
        raise FileNotFoundError(f"no HD_EPIC_*_vjepa.csv under {src}")
    return pd.concat(dfs, ignore_index=True).drop_duplicates(
        subset=["video_id", "start_frame", "stop_frame", "verb_class", "noun_class"]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    ap.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tick-sec", type=float, default=2.0)
    ap.add_argument("--obs-sec", type=float, default=4.0)
    ap.add_argument("--horizon-sec", type=float, default=2.0)
    ap.add_argument("--model-fps", type=float, default=8.0)
    ap.add_argument("--participant", type=str, default="all")
    ap.add_argument("--max-videos", type=int, default=0, help="0=all")
    args = ap.parse_args()

    obs = float(args.obs_sec)
    horizon = float(args.horizon_sec)
    clips = _pool_all_clips(args.clips)
    if args.participant.lower() not in {"all", "*"}:
        clips = clips[clips["participant_id"].astype(str) == args.participant]
    video_ids = sorted(clips["video_id"].astype(str).unique().tolist())
    if int(args.max_videos) > 0:
        video_ids = video_ids[: int(args.max_videos)]
    intervals_map = build_intervals_from_clips(clips)
    print(
        f"clips={len(clips)} videos={len(video_ids)} participants="
        f"{sorted(clips['participant_id'].astype(str).unique().tolist())}",
        flush=True,
    )

    all_rows: list[dict] = []
    for vid in video_ids:
        path = _video_path(args.video_root, vid)
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        n_frames = len(vr)
        vfps = float(vr.get_avg_fps())
        del vr
        rows = build_rows_for_video(
            video_id=vid,
            n_frames=n_frames,
            vfps=vfps,
            intervals=intervals_map.get(vid, []),
            tick_sec=float(args.tick_sec),
            min_context_sec=obs,
            max_context_sec=obs,
            context_schedule=[obs],
            horizons_sec=[horizon],
            model_fps=float(args.model_fps),
        )
        all_rows.extend(rows)
        print(f"{vid}: {len(rows)} ticks", flush=True)

    df = pd.DataFrame(all_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        sub = df[df["split"] == split].reset_index(drop=True)
        out_name = f"HD_EPIC_{split}_stream_mtp.csv"
        sub.to_csv(args.out / out_name, index=False)
        if split == "val":
            sub.to_csv(args.out / "HD_EPIC_test_stream_mtp.csv", index=False)
        print(
            f"wrote {out_name}: {len(sub)} rows, videos={sub['video_id'].nunique()}",
            flush=True,
        )

    stats = {
        "protocol": "temporal_half_split_streaming_fixed_4s_plus2s",
        "source_sampler": str(JEPA_SCRIPTS / "make_hdepic_stream_half_split.py"),
        "clip_pool": str(args.clips),
        "tick_sec": float(args.tick_sec),
        "obs_sec": obs,
        "horizon_sec": horizon,
        "context_schedule": [obs],
        "model_fps": float(args.model_fps),
        "participant": args.participant,
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "videos": int(df["video_id"].nunique()),
        "note": (
            "Same entire-scene sampler as JEPA stream_half_split (first half train, "
            "second half val). Fixed 4s observation, predict +2s only. All HD-EPIC "
            "participants when --participant all."
        ),
    }
    (args.out / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
