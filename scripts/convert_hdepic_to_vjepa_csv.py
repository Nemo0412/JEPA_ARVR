#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Utility for adapting HD-EPIC annotations to the EK100-style CSV format used by
# evals/action_anticipation_frozen in this repository.

import argparse
import ast
import json
import logging
import os
import random
import shutil
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("convert_hdepic_to_vjepa_csv")

OUTPUT_COLUMNS = [
    "participant_id",
    "video_id",
    "original_video_id",
    "start_frame",
    "stop_frame",
    "verb_class",
    "noun_class",
    "start_timestamp",
    "end_timestamp",
    "narration",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert HD-EPIC narration/action annotations into V-JEPA action "
            "anticipation CSVs with video_id,start_frame,stop_frame,verb_class,noun_class."
        )
    )
    parser.add_argument(
        "--annotations-pkl",
        required=True,
        type=Path,
        help="Path to HD_EPIC_Narrations.pkl.",
    )
    parser.add_argument(
        "--video-root",
        required=True,
        type=Path,
        help=(
            "Path to HD-EPIC videos. Accepts either the dataset root containing "
            "Videos/ or the Videos directory itself."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where train/val CSVs and conversion_stats.json are written.",
    )
    parser.add_argument(
        "--train-name",
        default="HD_EPIC_train_vjepa.csv",
        help="Output train CSV filename.",
    )
    parser.add_argument(
        "--val-name",
        default="HD_EPIC_val_vjepa.csv",
        help="Output validation CSV filename.",
    )
    parser.add_argument(
        "--val-participants",
        nargs="*",
        default=None,
        help="Participant ids to place in val, e.g. P01 P02. Overrides --val-ratio.",
    )
    parser.add_argument(
        "--include-participants",
        nargs="*",
        default=None,
        help="Participant ids to include before splitting, e.g. P01 P02. Defaults to all participants.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Video-level random validation ratio used when --val-participants is omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for video-level train/val split.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Fallback FPS used to convert timestamps to frames when decord is unavailable "
            "or video probing is disabled."
        ),
    )
    parser.add_argument(
        "--no-video-probe",
        action="store_true",
        help="Do not open videos to read FPS. Requires --fps.",
    )
    parser.add_argument(
        "--skip-missing-videos",
        action="store_true",
        help="Skip rows whose video file is missing. By default missing videos are kept if --fps is set.",
    )
    parser.add_argument(
        "--video-ext",
        default=".mp4",
        help="Preferred video extension. The converter also tries the opposite .mp4/.MP4 case.",
    )
    parser.add_argument(
        "--vjepa-video-id-format",
        default="ek100_compatible",
        choices=["ek100_compatible", "original"],
        help=(
            "How to write video_id in the output CSV. ek100_compatible rewrites "
            "P01-20240202-110250 to P01_20240202-110250 so the unmodified EK100 "
            "decoder resolves participant folders with video_id.split('_')[0]."
        ),
    )
    parser.add_argument(
        "--link-root",
        type=Path,
        default=None,
        help=(
            "Optional directory where an EK100-compatible video tree is created. "
            "Use this path as experiment.data.base_path with dataset: EK100 and file_format: 1."
        ),
    )
    parser.add_argument(
        "--link-method",
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
        help="How to populate --link-root. symlink is recommended to avoid duplicating videos.",
    )
    parser.add_argument(
        "--keep-secondary-actions",
        action="store_true",
        help=(
            "Emit one row for every pair in main_action_classes. By default only the "
            "first main action pair is used."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def normalize_video_root(video_root):
    video_root = video_root.expanduser().resolve()
    if (video_root / "Videos").exists():
        return video_root / "Videos"
    return video_root


def get_participant_id(row):
    if "participant_id" in row and pd.notna(row["participant_id"]):
        return str(row["participant_id"])
    return str(row["video_id"]).split("-")[0]


def format_vjepa_video_id(video_id, participant_id, video_id_format):
    if video_id_format == "original":
        return video_id
    prefix = f"{participant_id}-"
    if video_id.startswith(prefix):
        return f"{participant_id}_{video_id[len(prefix):]}"
    return video_id.replace("-", "_", 1)


def parse_action_pairs(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, str):
        value = ast.literal_eval(value)

    if hasattr(value, "tolist"):
        value = value.tolist()

    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and not isinstance(value[0], (list, tuple, dict))
    ):
        value = [value]

    pairs = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, str):
            item = ast.literal_eval(item)
        if len(item) < 2:
            continue
        pairs.append((int(item[0]), int(item[1])))
    return pairs


def resolve_video_path(video_root, participant_id, video_id, video_ext):
    preferred = video_root / participant_id / f"{video_id}{video_ext}"
    if preferred.exists():
        return preferred

    alt_ext = ".MP4" if video_ext == ".mp4" else ".mp4"
    alternate = video_root / participant_id / f"{video_id}{alt_ext}"
    if alternate.exists():
        return alternate

    return preferred


def create_link(src, dst, method):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst):
        return

    if method == "symlink":
        os.symlink(src, dst)
    elif method == "hardlink":
        os.link(src, dst)
    elif method == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link method: {method}")


def import_decord():
    try:
        from decord import VideoReader, cpu
    except ImportError:
        return None, None
    return VideoReader, cpu


def get_video_fps(video_path, fps_cache, fallback_fps, no_video_probe):
    cache_key = str(video_path)
    if cache_key in fps_cache:
        return fps_cache[cache_key]

    if no_video_probe:
        if fallback_fps is None:
            raise ValueError("--no-video-probe requires --fps")
        fps_cache[cache_key] = fallback_fps
        return fallback_fps

    VideoReader, cpu = import_decord()
    if VideoReader is not None and video_path.exists():
        fps = float(VideoReader(str(video_path), ctx=cpu(0)).get_avg_fps())
        fps_cache[cache_key] = fps
        return fps

    if fallback_fps is None:
        raise RuntimeError(
            f"Cannot determine FPS for {video_path}. Install decord, provide --fps, "
            "or use --no-video-probe --fps."
        )

    fps_cache[cache_key] = fallback_fps
    return fallback_fps


def build_rows(args):
    annotations = pd.read_pickle(args.annotations_pkl)
    video_root = normalize_video_root(args.video_root)
    fps_cache = {}
    rows = []
    missing_videos = set()
    unreadable_videos = {}
    linked_videos = {}
    dropped_no_action = 0

    required = {"video_id", "start_timestamp", "end_timestamp", "main_action_classes"}
    missing_columns = sorted(required - set(annotations.columns))
    if missing_columns:
        raise ValueError(f"Missing required HD-EPIC columns: {missing_columns}")

    if args.include_participants:
        include_participants = set(args.include_participants)
        annotations = annotations[
            annotations.apply(lambda row: get_participant_id(row) in include_participants, axis=1)
        ].copy()
        LOGGER.info("Keeping %d annotation rows for participants %s", len(annotations), sorted(include_participants))

    for _, row in annotations.iterrows():
        action_pairs = parse_action_pairs(row["main_action_classes"])
        if not action_pairs:
            dropped_no_action += 1
            continue
        if not args.keep_secondary_actions:
            action_pairs = action_pairs[:1]

        video_id = str(row["video_id"])
        participant_id = get_participant_id(row)
        video_path = resolve_video_path(video_root, participant_id, video_id, args.video_ext)
        if not video_path.exists():
            missing_videos.add(str(video_path))
            if args.skip_missing_videos:
                continue

        try:
            fps = get_video_fps(
                video_path=video_path,
                fps_cache=fps_cache,
                fallback_fps=args.fps,
                no_video_probe=args.no_video_probe,
            )
        except Exception as exc:
            unreadable_videos[str(video_path)] = repr(exc)
            if args.skip_missing_videos:
                continue
            raise
        start_frame = max(0, int(round(float(row["start_timestamp"]) * fps)))
        stop_frame = max(start_frame + 1, int(round(float(row["end_timestamp"]) * fps)))
        output_video_id = format_vjepa_video_id(video_id, participant_id, args.vjepa_video_id_format)

        if args.link_root is not None and video_path.exists():
            link_ext = args.video_ext if args.video_ext.startswith(".") else f".{args.video_ext}"
            link_path = args.link_root / participant_id / f"{output_video_id}{link_ext.upper()}"
            create_link(video_path, link_path, args.link_method)
            linked_videos[output_video_id] = str(link_path)

        for verb_class, noun_class in action_pairs:
            rows.append(
                {
                    "participant_id": participant_id,
                    "video_id": output_video_id,
                    "original_video_id": video_id,
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                    "verb_class": verb_class,
                    "noun_class": noun_class,
                    "start_timestamp": float(row["start_timestamp"]),
                    "end_timestamp": float(row["end_timestamp"]),
                    "narration": row.get("narration", ""),
                }
            )

    stats = {
        "annotation_rows": int(len(annotations)),
        "converted_rows": int(len(rows)),
        "dropped_no_action": int(dropped_no_action),
        "missing_videos": sorted(missing_videos),
        "unreadable_videos": unreadable_videos,
        "linked_videos": linked_videos,
        "vjepa_base_path": None if args.link_root is None else str(args.link_root.resolve()),
        "vjepa_dataset": "EK100",
        "vjepa_file_format": 1,
        "unique_videos": int(len({row["video_id"] for row in rows})),
        "unique_verbs": int(len({row["verb_class"] for row in rows})),
        "unique_nouns": int(len({row["noun_class"] for row in rows})),
        "unique_actions": int(len({(row["verb_class"], row["noun_class"]) for row in rows})),
    }
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), stats


def split_dataframe(df, val_participants, val_ratio, seed):
    if df.empty:
        raise ValueError("No rows were converted; check annotations, video paths, and action fields.")

    if val_participants:
        val_participants = set(val_participants)
        val_mask = df["participant_id"].isin(val_participants)
        train_df, val_df = df[~val_mask].copy(), df[val_mask].copy()
        if not train_df.empty and not val_df.empty:
            return train_df, val_df
        LOGGER.warning(
            "Participant split produced empty train or val set; falling back to video-level val_ratio=%s",
            val_ratio,
        )

    rng = random.Random(seed)
    videos = sorted(df["video_id"].unique())
    rng.shuffle(videos)
    num_val = max(1, int(round(len(videos) * val_ratio)))
    val_videos = set(videos[:num_val])
    val_mask = df["video_id"].isin(val_videos)
    return df[~val_mask].copy(), df[val_mask].copy()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    df, stats = build_rows(args)
    train_df, val_df = split_dataframe(df, args.val_participants, args.val_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / args.train_name
    val_path = args.output_dir / args.val_name
    stats_path = args.output_dir / "conversion_stats.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    stats.update(
        {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_videos": int(train_df["video_id"].nunique()),
            "val_videos": int(val_df["video_id"].nunique()),
        }
    )
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    LOGGER.info("Wrote %s", train_path)
    LOGGER.info("Wrote %s", val_path)
    LOGGER.info("Wrote %s", stats_path)
    if stats["missing_videos"]:
        LOGGER.warning("Missing %d videos; see conversion_stats.json", len(stats["missing_videos"]))


if __name__ == "__main__":
    main()
