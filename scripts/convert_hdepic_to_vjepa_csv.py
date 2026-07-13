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


# Canonical P01 fixed train/val/test video lists (project default split).
P01_FIXED_SPLITS_DIR = Path(__file__).resolve().parent.parent / "data/hdepic_vjepa_annotations" / "splits"
P01_FIXED_TRAIN_LIST = P01_FIXED_SPLITS_DIR / "p01_fixed_train_videos.txt"
P01_FIXED_VAL_LIST = P01_FIXED_SPLITS_DIR / "p01_fixed_val_videos.txt"
P01_FIXED_TEST_LIST = P01_FIXED_SPLITS_DIR / "p01_fixed_test_videos.txt"


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
        "--test-name",
        default="HD_EPIC_test_vjepa.csv",
        help="Output test CSV filename. Written only when the selected split defines test videos.",
    )
    parser.add_argument(
        "--split-preset",
        default="p01_fixed",
        choices=["p01_fixed", "legacy", "custom_fixed"],
        help=(
            "Dataset split policy. p01_fixed uses the canonical fixed P01 "
            "train/val/test video lists under data/hdepic_vjepa_annotations/splits/; "
            "legacy preserves participant/random validation split; "
            "custom_fixed uses --train-video-ids/--val-video-ids/--test-video-ids or their file forms."
        ),
    )
    parser.add_argument(
        "--train-video-ids",
        nargs="*",
        default=None,
        help="Original HD-EPIC video ids for a custom fixed train split.",
    )
    parser.add_argument(
        "--val-video-ids",
        nargs="*",
        default=None,
        help="Original HD-EPIC video ids for a custom fixed validation split.",
    )
    parser.add_argument(
        "--test-video-ids",
        nargs="*",
        default=None,
        help="Original HD-EPIC video ids for a custom fixed test split.",
    )
    parser.add_argument(
        "--train-video-list",
        type=Path,
        default=None,
        help="Text file with one original HD-EPIC video id per line for custom fixed train split.",
    )
    parser.add_argument(
        "--val-video-list",
        type=Path,
        default=None,
        help="Text file with one original HD-EPIC video id per line for custom fixed validation split.",
    )
    parser.add_argument(
        "--test-video-list",
        type=Path,
        default=None,
        help="Text file with one original HD-EPIC video id per line for custom fixed test split.",
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
            "With --label-source main_action_classes, emit one row for every pair. "
            "By default only the first main action pair is used."
        ),
    )
    parser.add_argument(
        "--label-source",
        default="primary_verb_noun",
        choices=["primary_verb_noun", "main_action_classes"],
        help=(
            "Which HD-EPIC label fields to write to verb_class/noun_class. "
            "primary_verb_noun matches the PhD reference code: verb_classes[0]/noun_classes[0]. "
            "main_action_classes preserves the older converter behavior."
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


def normalize_original_video_id(video_id):
    video_id = str(video_id).strip()
    if not video_id:
        return video_id
    if "_" in video_id and "-" in video_id:
        participant, rest = video_id.split("_", 1)
        if participant.startswith("P"):
            return f"{participant}-{rest}"
    return video_id


def read_video_id_list(path):
    if path is None:
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(line)
    return values


def split_video_ids_from_args(args):
    if args.split_preset == "p01_fixed":
        return {
            "train": read_video_id_list(P01_FIXED_TRAIN_LIST),
            "val": read_video_id_list(P01_FIXED_VAL_LIST),
            "test": read_video_id_list(P01_FIXED_TEST_LIST),
        }
    if args.split_preset == "custom_fixed":
        return {
            "train": list(args.train_video_ids or []) + read_video_id_list(args.train_video_list),
            "val": list(args.val_video_ids or []) + read_video_id_list(args.val_video_list),
            "test": list(args.test_video_ids or []) + read_video_id_list(args.test_video_list),
        }
    return {"train": [], "val": [], "test": []}


def _dedupe_video_ids(video_ids):
    out = []
    seen = set()
    for video_id in video_ids:
        clean = normalize_original_video_id(video_id)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


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


def primary_verb_noun_pair(row):
    verb_classes = row["verb_classes"]
    noun_classes = row["noun_classes"]
    if isinstance(verb_classes, str):
        verb_classes = ast.literal_eval(verb_classes)
    if isinstance(noun_classes, str):
        noun_classes = ast.literal_eval(noun_classes)
    if hasattr(verb_classes, "tolist"):
        verb_classes = verb_classes.tolist()
    if hasattr(noun_classes, "tolist"):
        noun_classes = noun_classes.tolist()
    if not isinstance(verb_classes, list) or not isinstance(noun_classes, list):
        return []
    if not verb_classes or not noun_classes:
        return []
    return [(int(verb_classes[0]), int(noun_classes[0]))]


def label_pairs_for_row(row, args):
    if args.label_source == "primary_verb_noun":
        return primary_verb_noun_pair(row)
    action_pairs = parse_action_pairs(row["main_action_classes"])
    if not args.keep_secondary_actions:
        action_pairs = action_pairs[:1]
    return action_pairs


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

    required = {"video_id", "start_timestamp", "end_timestamp"}
    if args.label_source == "primary_verb_noun":
        required.update({"verb_classes", "noun_classes"})
    else:
        required.add("main_action_classes")
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
        action_pairs = label_pairs_for_row(row, args)
        if not action_pairs:
            dropped_no_action += 1
            continue

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
        "label_source": args.label_source,
    }
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), stats


def split_dataframe(df, args):
    if df.empty:
        raise ValueError("No rows were converted; check annotations, video paths, and action fields.")

    if args.split_preset in {"p01_fixed", "custom_fixed"}:
        fixed_ids = split_video_ids_from_args(args)
        train_ids = _dedupe_video_ids(fixed_ids["train"])
        val_ids = _dedupe_video_ids(fixed_ids["val"])
        test_ids = _dedupe_video_ids(fixed_ids["test"])
        if args.split_preset == "p01_fixed" and (not train_ids or not val_ids or not test_ids):
            raise ValueError(
                "p01_fixed requires non-empty canonical train/val/test video lists at "
                f"{P01_FIXED_SPLITS_DIR}; got train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
            )
        overlap = (
            (set(train_ids) & set(val_ids))
            | (set(train_ids) & set(test_ids))
            | (set(val_ids) & set(test_ids))
        )
        if overlap:
            raise ValueError(f"Fixed split video ids overlap across splits: {sorted(overlap)}")
        if not train_ids or not val_ids:
            raise ValueError(
                f"{args.split_preset} requires non-empty train and val video ids; "
                f"got train={len(train_ids)} val={len(val_ids)}"
            )

        original_ids = df["original_video_id"].map(normalize_original_video_id)
        train_mask = original_ids.isin(train_ids)
        val_mask = original_ids.isin(val_ids)
        test_mask = original_ids.isin(test_ids) if test_ids else pd.Series(False, index=df.index)
        unmatched_mask = ~(train_mask | val_mask | test_mask)

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()
        unmatched_df = df[unmatched_mask].copy()
        if train_df.empty or val_df.empty:
            raise ValueError(
                f"Fixed split produced empty train or val set: train_rows={len(train_df)} val_rows={len(val_df)}"
            )
        missing_by_split = {
            "train": sorted(set(train_ids) - set(original_ids[train_mask].unique())),
            "val": sorted(set(val_ids) - set(original_ids[val_mask].unique())),
            "test": sorted(set(test_ids) - set(original_ids[test_mask].unique())),
        }
        split_meta = {
            "split_preset": args.split_preset,
            "split_policy": (
                "p01_fixed_video_lists"
                if args.split_preset == "p01_fixed"
                else "custom_fixed_video_lists"
            ),
            "participant_scope": "P01",
            "train_video_ids": train_ids,
            "val_video_ids": val_ids,
            "test_video_ids": test_ids,
            "missing_split_video_ids": missing_by_split,
            "unmatched_rows": int(len(unmatched_df)),
            "unmatched_videos": sorted(set(original_ids[unmatched_mask].unique())),
        }
        return train_df, val_df, test_df, split_meta

    if args.val_participants:
        val_participants = set(args.val_participants)
        val_mask = df["participant_id"].isin(val_participants)
        train_df, val_df = df[~val_mask].copy(), df[val_mask].copy()
        if not train_df.empty and not val_df.empty:
            split_meta = {
                "split_preset": "legacy",
                "split_policy": "participant",
                "val_participants": sorted(val_participants),
            }
            return train_df, val_df, pd.DataFrame(columns=OUTPUT_COLUMNS), split_meta
        LOGGER.warning(
            "Participant split produced empty train or val set; falling back to video-level val_ratio=%s",
            args.val_ratio,
        )

    rng = random.Random(args.seed)
    videos = sorted(df["video_id"].unique())
    rng.shuffle(videos)
    num_val = max(1, int(round(len(videos) * args.val_ratio)))
    val_videos = set(videos[:num_val])
    val_mask = df["video_id"].isin(val_videos)
    split_meta = {
        "split_preset": "legacy",
        "split_policy": "video_random_val_ratio",
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "val_video_ids": sorted(val_videos),
    }
    return df[~val_mask].copy(), df[val_mask].copy(), pd.DataFrame(columns=OUTPUT_COLUMNS), split_meta


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    df, stats = build_rows(args)
    train_df, val_df, test_df, split_meta = split_dataframe(df, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / args.train_name
    val_path = args.output_dir / args.val_name
    test_path = args.output_dir / args.test_name
    stats_path = args.output_dir / "conversion_stats.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    if not test_df.empty:
        test_df.to_csv(test_path, index=False)
    stats.update(
        {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_videos": int(train_df["video_id"].nunique()),
            "val_videos": int(val_df["video_id"].nunique()),
            "test_videos": int(test_df["video_id"].nunique()) if not test_df.empty else 0,
            **split_meta,
        }
    )
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    LOGGER.info("Wrote %s", train_path)
    LOGGER.info("Wrote %s", val_path)
    if not test_df.empty:
        LOGGER.info("Wrote %s", test_path)
    LOGGER.info("Wrote %s", stats_path)
    if stats["missing_videos"]:
        LOGGER.warning("Missing %d videos; see conversion_stats.json", len(stats["missing_videos"]))


if __name__ == "__main__":
    main()
