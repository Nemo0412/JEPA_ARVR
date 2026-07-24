#!/usr/bin/env python3
"""Build HD-EPIC P01 *temporal half-split* streaming MTP index.

Protocol
--------
For each video of length T frames (native fps, usually 30):
  - train region = [0, T/2)
  - val/test region = [T/2, T)

Inside a region, stream from the region origin:
  tick every ``tick_sec`` (default 2s), first tick at origin+``min_context_sec`` (4s).
  Context grows from the origin: 1–4s → 1–6s → 1–8s → 1–10s, then slides a
  ``max_context_sec`` (10s) window. At each tick, record action labels at
  tick+{2,4,6}s via narration intervals (same lookup as clip MTP).

Outputs CSVs under ``--out`` (default scratch stream_half_split/).
Does **not** touch the frozen clip_split/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from decord import VideoReader, cpu

from app.hdepic_lora_action_anticipation.mtp import lookup_action_at_frame

DEFAULT_STAGE2 = Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stage2")
DEFAULT_VIDEO_ROOT = Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos")
DEFAULT_OUT = Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split")


def _pool_clips(stage2: Path) -> pd.DataFrame:
    dfs = []
    for name in ("HD_EPIC_train_vjepa.csv", "HD_EPIC_val_vjepa.csv", "HD_EPIC_test_vjepa.csv"):
        dfs.append(pd.read_csv(stage2 / name))
    return pd.concat(dfs, ignore_index=True).drop_duplicates(
        subset=["video_id", "start_frame", "stop_frame", "verb_class", "noun_class"]
    )


def _video_path(video_root: Path, video_id: str) -> Path:
    # P01 videos live under video_root/P01/<id>.MP4
    pid = str(video_id).split("_")[0]
    for ext in (".MP4", ".mp4", ".mkv"):
        p = video_root / pid / f"{video_id}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"missing video for {video_id} under {video_root}/{pid}")


def _snap_context(elapsed_sec: float, schedule: list[float], max_context: float) -> float:
    if elapsed_sec >= max_context - 1e-6:
        return float(max_context)
    # Grow along schedule: pick largest s in schedule with s <= elapsed.
    chosen = schedule[0]
    for s in schedule:
        if s <= elapsed_sec + 1e-6:
            chosen = s
    return float(chosen)


def build_rows_for_video(
    video_id: str,
    n_frames: int,
    vfps: float,
    intervals: list[tuple[int, int, int, int]],
    tick_sec: float,
    min_context_sec: float,
    max_context_sec: float,
    context_schedule: list[float],
    horizons_sec: list[float],
    model_fps: float,
) -> list[dict]:
    mid = n_frames // 2
    regions = (("train", 0, mid), ("val", mid, n_frames))
    rows: list[dict] = []
    max_h = max(horizons_sec)
    for split, t0, t1 in regions:
        if t1 - t0 < int((min_context_sec + max_h) * vfps):
            continue
        origin = t0
        tick = origin + int(round(min_context_sec * vfps))
        step = max(1, int(round(tick_sec * vfps)))
        while tick < t1:
            # Need room for the farthest label inside the video (not necessarily the half).
            if tick + int(round(max_h * vfps)) >= n_frames:
                break
            elapsed = (tick - origin) / float(vfps)
            context_sec = _snap_context(elapsed, context_schedule, max_context_sec)
            if elapsed <= max_context_sec + 1e-6:
                start_frame = origin
            else:
                start_frame = max(origin, tick - int(round(max_context_sec * vfps)))
                context_sec = max_context_sec
            # Sample model_fps frames in [start_frame, tick].
            n_model = max(1, int(round(context_sec * model_fps)))
            # Ensure tubelet alignment (even frame count for tubelet_size=2).
            if n_model % 2 == 1:
                n_model += 1
            frame_idx = np.linspace(start_frame, max(start_frame, tick - 1), n_model)
            frame_idx = np.clip(np.round(frame_idx).astype(np.int64), 0, n_frames - 1)

            verbs, nouns, masks = [], [], []
            for h in horizons_sec:
                f = int(tick + round(float(h) * vfps))
                v, n, ok = lookup_action_at_frame(intervals, f)
                verbs.append(int(v) if ok else -1)
                nouns.append(int(n) if ok else -1)
                masks.append(1.0 if ok else 0.0)
            if sum(masks) < 1:
                tick += step
                continue
            rows.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "origin_frame": int(origin),
                    "tick_frame": int(tick),
                    "start_frame": int(start_frame),
                    "context_sec": float(context_sec),
                    "n_model_frames": int(n_model),
                    "vfps": float(vfps),
                    "n_frames": int(n_frames),
                    "frame_indices": ",".join(str(int(x)) for x in frame_idx.tolist()),
                    "mtp_verbs": ",".join(str(x) for x in verbs),
                    "mtp_nouns": ",".join(str(x) for x in nouns),
                    "mtp_mask": ",".join(str(x) for x in masks),
                }
            )
            tick += step
    return rows


def build_intervals_from_clips(clips: pd.DataFrame) -> dict[str, list[tuple[int, int, int, int]]]:
    out: dict[str, list[tuple[int, int, int, int]]] = {}
    for video_id, g in clips.groupby("video_id"):
        intervals = []
        for row in g.itertuples(index=False):
            intervals.append(
                (
                    int(row.start_frame),
                    int(row.stop_frame),
                    int(row.verb_class),
                    int(row.noun_class),
                )
            )
        intervals.sort(key=lambda x: (x[0], x[1]))
        out[str(video_id)] = intervals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    ap.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tick-sec", type=float, default=2.0)
    ap.add_argument("--min-context-sec", type=float, default=4.0)
    ap.add_argument("--max-context-sec", type=float, default=10.0)
    ap.add_argument("--context-schedule", type=str, default="4,6,8,10")
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--model-fps", type=float, default=8.0)
    ap.add_argument("--participant", type=str, default="P01")
    args = ap.parse_args()

    schedule = [float(x) for x in args.context_schedule.split(",") if x.strip()]
    horizons = [float(x) for x in args.horizons_sec.split(",") if x.strip()]

    clips = _pool_clips(args.stage2)
    clips = clips[clips["participant_id"].astype(str) == args.participant]
    video_ids = sorted(clips["video_id"].astype(str).unique().tolist())
    intervals_map = build_intervals_from_clips(clips)

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
            tick_sec=args.tick_sec,
            min_context_sec=args.min_context_sec,
            max_context_sec=args.max_context_sec,
            context_schedule=schedule,
            horizons_sec=horizons,
            model_fps=args.model_fps,
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
        print(f"wrote {out_name}: {len(sub)} rows, videos={sub['video_id'].nunique()}")

    stats = {
        "protocol": "temporal_half_split_streaming_mtp",
        "tick_sec": args.tick_sec,
        "min_context_sec": args.min_context_sec,
        "max_context_sec": args.max_context_sec,
        "context_schedule": schedule,
        "horizons_sec": horizons,
        "model_fps": args.model_fps,
        "participant": args.participant,
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "videos": int(df["video_id"].nunique()),
        "note": (
            "Train = first half of each video; val/test = second half. "
            "Context grows 4→6→8→10s from half origin, then slides 10s. "
            "Predict actions at +2/+4/+6s every tick_sec. "
            "Prune-before-predictor is applied at train time when tokens exceed budget."
        ),
    }
    (args.out / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
