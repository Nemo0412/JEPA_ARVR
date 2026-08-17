#!/usr/bin/env python3
"""Build the frozen B13 EGTEA temporal-half streaming index.

This is the EGTEA data adapter for ll's
``scripts/make_hdepic_stream_half_split.py`` protocol.  Model-facing temporal
semantics are intentionally unchanged:

* split every full session video at ``n_frames // 2``;
* first half is train and second half is val/test;
* grow context 4 -> 6 -> 8 -> 10 seconds, then slide a 10-second window every
  2 seconds;
* sample RGB at 8 fps and label actions at +2/+4/+6 seconds;
* use the action covering the target frame, otherwise the next action start;
* permit train future labels to cross the temporal midpoint;
* copy val byte-for-byte to test.

The input action CSVs are the V1/session-frame EGTEA split-1 CSVs created by
``scripts/egtea/build_egtea_csvs_v2.py``.  Their original membership is pooled
before the temporal split; it is not retained as the streaming train/val split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from decord import VideoReader, cpu

from app.hdepic_lora_action_anticipation.mtp import lookup_action_at_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data/egtea/vjepa_annotations/v1/split1"
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "data/egtea/session_videos"
DEFAULT_OUT = PROJECT_ROOT / "data/egtea/vjepa_annotations/stream_half_split/split1"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pool_actions(source: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frames = []
    source_rows: dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = source / f"EGTEA_{split}_vjepa.csv"
        frame = pd.read_csv(path)
        source_rows[split] = int(len(frame))
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["participant_id", "start_frame", "stop_frame", "verb_class", "noun_class"]
    )
    return pooled, source_rows


def _video_path(video_root: Path, session: str) -> Path:
    for ext in (".mp4", ".MP4", ".mkv"):
        path = video_root / f"{session}{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing full-session video for {session} under {video_root}")


def _snap_context(elapsed_sec: float, schedule: list[float], max_context: float) -> float:
    if elapsed_sec >= max_context - 1e-6:
        return float(max_context)
    chosen = schedule[0]
    for value in schedule:
        if value <= elapsed_sec + 1e-6:
            chosen = value
    return float(chosen)


def _build_rows_for_session(
    session: str,
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
    midpoint = n_frames // 2
    regions = (("train", 0, midpoint), ("val", midpoint, n_frames))
    rows: list[dict] = []
    max_horizon = max(horizons_sec)
    for split, origin, region_end in regions:
        if region_end - origin < int((min_context_sec + max_horizon) * vfps):
            continue
        tick = origin + int(round(min_context_sec * vfps))
        step = max(1, int(round(tick_sec * vfps)))
        while tick < region_end:
            # Labels may cross the train midpoint, but never the physical video end.
            if tick + int(round(max_horizon * vfps)) >= n_frames:
                break
            elapsed = (tick - origin) / float(vfps)
            context_sec = _snap_context(elapsed, context_schedule, max_context_sec)
            if elapsed <= max_context_sec + 1e-6:
                start_frame = origin
            else:
                start_frame = max(origin, tick - int(round(max_context_sec * vfps)))
                context_sec = max_context_sec

            n_model_frames = max(1, int(round(context_sec * model_fps)))
            if n_model_frames % 2 == 1:
                n_model_frames += 1
            frame_indices = np.linspace(
                start_frame, max(start_frame, tick - 1), n_model_frames
            )
            frame_indices = np.clip(
                np.round(frame_indices).astype(np.int64), 0, n_frames - 1
            )

            verbs: list[int] = []
            nouns: list[int] = []
            masks: list[float] = []
            for horizon in horizons_sec:
                target = int(tick + round(float(horizon) * vfps))
                verb, noun, valid = lookup_action_at_frame(intervals, target)
                verbs.append(int(verb) if valid else -1)
                nouns.append(int(noun) if valid else -1)
                masks.append(1.0 if valid else 0.0)
            if sum(masks) < 1:
                tick += step
                continue

            rows.append(
                {
                    "split": split,
                    "video_id": session,
                    "origin_frame": int(origin),
                    "tick_frame": int(tick),
                    "start_frame": int(start_frame),
                    "context_sec": float(context_sec),
                    "n_model_frames": int(n_model_frames),
                    "vfps": float(vfps),
                    "n_frames": int(n_frames),
                    "frame_indices": ",".join(str(int(x)) for x in frame_indices.tolist()),
                    "mtp_verbs": ",".join(str(x) for x in verbs),
                    "mtp_nouns": ",".join(str(x) for x in nouns),
                    "mtp_mask": ",".join(str(x) for x in masks),
                }
            )
            tick += step
    return rows


def _build_intervals(actions: pd.DataFrame) -> dict[str, list[tuple[int, int, int, int]]]:
    result: dict[str, list[tuple[int, int, int, int]]] = {}
    for session, group in actions.groupby("participant_id"):
        intervals = [
            (
                int(row.start_frame),
                int(row.stop_frame),
                int(row.verb_class),
                int(row.noun_class),
            )
            for row in group.itertuples(index=False)
        ]
        intervals.sort(key=lambda item: (item[0], item[1]))
        result[str(session)] = intervals
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tick-sec", type=float, default=2.0)
    parser.add_argument("--min-context-sec", type=float, default=4.0)
    parser.add_argument("--max-context-sec", type=float, default=10.0)
    parser.add_argument("--context-schedule", default="4,6,8,10")
    parser.add_argument("--horizons-sec", default="2,4,6")
    parser.add_argument("--model-fps", type=float, default=8.0)
    args = parser.parse_args()

    schedule = [float(x) for x in args.context_schedule.split(",") if x.strip()]
    horizons = [float(x) for x in args.horizons_sec.split(",") if x.strip()]
    actions, source_rows = _pool_actions(args.source)
    intervals = _build_intervals(actions)
    sessions = sorted(intervals)

    all_rows: list[dict] = []
    video_metadata: dict[str, dict[str, float | int]] = {}
    for session in sessions:
        reader = VideoReader(str(_video_path(args.video_root, session)), ctx=cpu(0), num_threads=1)
        n_frames = len(reader)
        vfps = float(reader.get_avg_fps())
        del reader
        rows = _build_rows_for_session(
            session=session,
            n_frames=n_frames,
            vfps=vfps,
            intervals=intervals[session],
            tick_sec=args.tick_sec,
            min_context_sec=args.min_context_sec,
            max_context_sec=args.max_context_sec,
            context_schedule=schedule,
            horizons_sec=horizons,
            model_fps=args.model_fps,
        )
        all_rows.extend(rows)
        video_metadata[session] = {"frames": n_frames, "fps": vfps, "rows": len(rows)}
        print(f"{session}: {len(rows)} ticks", flush=True)

    frame = pd.DataFrame(all_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "val"):
        subset = frame[frame["split"] == split].reset_index(drop=True)
        path = args.out / f"EGTEA_{split}_stream_mtp.csv"
        subset.to_csv(path, index=False)
        paths[split] = path
        if split == "val":
            test_path = args.out / "EGTEA_test_stream_mtp.csv"
            subset.to_csv(test_path, index=False)
            paths["test"] = test_path
        print(f"wrote {path.name}: {len(subset)} rows, sessions={subset['video_id'].nunique()}")

    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    context_counts_train = Counter(str(float(x)) for x in train["context_sec"])
    context_counts_val = Counter(str(float(x)) for x in val["context_sec"])
    crossing = {}
    for index, horizon in enumerate(horizons):
        count = 0
        for row in train.itertuples(index=False):
            mask = [float(x) for x in str(row.mtp_mask).split(",")]
            if mask[index] >= 0.5:
                midpoint = int(row.n_frames) // 2
                target = int(row.tick_frame + round(horizon * float(row.vfps)))
                count += int(target >= midpoint)
        crossing[f"{horizon:g}s"] = count

    stats = {
        "protocol": "temporal_half_split_streaming_mtp_ll_exact",
        "reference_repo_commit": "29063fea9ac1fa56d3cfbe4014e885d7cefdf4e7",
        "annotation_source": str(args.source),
        "session_video_source": str(args.video_root),
        "source_rows": source_rows,
        "pooled_deduplicated_actions": int(len(actions)),
        "dedup_key": [
            "participant_id", "start_frame", "stop_frame", "verb_class", "noun_class"
        ],
        "tick_sec": args.tick_sec,
        "min_context_sec": args.min_context_sec,
        "max_context_sec": args.max_context_sec,
        "context_schedule": schedule,
        "horizons_sec": horizons,
        "model_fps": args.model_fps,
        "sessions": len(sessions),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(val)),
        "context_counts_train": dict(context_counts_train),
        "context_counts_val": dict(context_counts_val),
        "train_future_labels_crossing_midpoint": crossing,
        "md5": {name: _md5(path) for name, path in paths.items()},
        "video_metadata": video_metadata,
        "note": (
            "Exact ll protocol: first half train; second half val/test; val copied to test; "
            "4->6->8->10s then sliding 10s; +2/+4/+6s labels; covering action else next "
            "start; train future labels are allowed to cross midpoint."
        ),
    }
    (args.out / "split_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
