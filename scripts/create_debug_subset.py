#!/usr/bin/env python
"""
Create a fixed, deterministic P01 debug subset from the HD-EPIC V-JEPA train CSV.

Run this script once on HPC (where the full CSV lives) and commit the output
JSON into configs/. Subsequent runs are reproducible because the selection is
seeded and the output is version-controlled.

Usage:
    python scripts/create_debug_subset.py \
        --train-csv  $PROJECT_ROOT/data/hdepic_vjepa_annotations/HD_EPIC_train_vjepa.csv \
        --val-csv    $PROJECT_ROOT/data/hdepic_vjepa_annotations/HD_EPIC_val_vjepa.csv \
        --output     configs/debug_subset_p01.json \
        [--num-videos 6] [--min-actions-per-class 2] [--seed 42]

The output JSON is the sole authoritative record of which videos/actions are in
the debug subset. Do NOT pass ad-hoc video lists via shell state — always load
the JSON at runtime.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("create_debug_subset")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", required=True, type=Path, help="HD_EPIC_train_vjepa.csv path")
    p.add_argument("--val-csv", required=True, type=Path, help="HD_EPIC_val_vjepa.csv path")
    p.add_argument("--output", default="configs/debug_subset_p01.json", type=Path, help="Output JSON path")
    p.add_argument("--participant", default="P01", help="Participant prefix to sample from (default: P01)")
    p.add_argument("--num-videos", type=int, default=6, help="Number of videos to include per split (default: 6)")
    p.add_argument(
        "--min-actions-per-class",
        type=int,
        default=2,
        help="Minimum number of actions required before a (verb,noun) class is considered representative (default: 2)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _select_diverse_videos(df: pd.DataFrame, participant: str, num_videos: int, min_actions_per_class: int, rng) -> list[str]:
    """
    Select `num_videos` video_ids that together cover a diverse set of action
    classes.  Strategy:
      1. Keep only the target participant.
      2. Count (verb_class, noun_class) occurrences; keep classes with >= min_actions_per_class samples.
      3. Greedily pick videos that maximise new class coverage, breaking ties randomly.
    """
    sub = df[df["video_id"].str.startswith(participant + "_")].copy()
    if sub.empty:
        sub = df[df["participant_id"] == participant].copy()
    if sub.empty:
        raise ValueError(f"No rows found for participant {participant!r}")

    action_counts = sub.groupby(["verb_class", "noun_class"]).size()
    valid_classes = set(action_counts[action_counts >= min_actions_per_class].index.tolist())
    logger.info(
        "Participant %s: %d total rows, %d videos, %d action classes with >= %d samples",
        participant,
        len(sub),
        sub["video_id"].nunique(),
        len(valid_classes),
        min_actions_per_class,
    )

    videos = sorted(sub["video_id"].unique().tolist())
    rng.shuffle(videos)

    covered: set[tuple] = set()
    selected: list[str] = []
    for vid in videos:
        if len(selected) >= num_videos:
            break
        vid_actions = set(
            map(tuple, sub[sub["video_id"] == vid][["verb_class", "noun_class"]].drop_duplicates().values.tolist())
        )
        new_classes = (vid_actions & valid_classes) - covered
        if new_classes or len(selected) < num_videos // 2:
            selected.append(vid)
            covered.update(vid_actions)

    # If greedy didn't fill the quota (e.g. very few videos), pad with remaining
    remaining = [v for v in videos if v not in selected]
    while len(selected) < num_videos and remaining:
        selected.append(remaining.pop(0))

    logger.info("Selected %d videos covering %d action classes", len(selected), len(covered & valid_classes))
    return selected


def _summarise(df: pd.DataFrame, video_ids: list[str]) -> dict:
    sub = df[df["video_id"].isin(video_ids)]
    return {
        "num_videos": len(video_ids),
        "num_rows": int(len(sub)),
        "num_verb_classes": int(sub["verb_class"].nunique()),
        "num_noun_classes": int(sub["noun_class"].nunique()),
        "num_action_classes": int(sub.groupby(["verb_class", "noun_class"]).ngroups),
    }


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    logger.info("Reading %s", args.train_csv)
    train_df = pd.read_csv(args.train_csv)
    logger.info("Reading %s", args.val_csv)
    val_df = pd.read_csv(args.val_csv)

    import random
    rng = random.Random(args.seed)

    train_videos = _select_diverse_videos(
        train_df,
        participant=args.participant,
        num_videos=args.num_videos,
        min_actions_per_class=args.min_actions_per_class,
        rng=rng,
    )
    val_videos = _select_diverse_videos(
        val_df,
        participant=args.participant,
        num_videos=max(2, args.num_videos // 2),
        min_actions_per_class=1,
        rng=rng,
    )

    output = {
        "_comment": (
            "Fixed deterministic P01 debug subset. "
            "Generated by scripts/create_debug_subset.py. "
            "Commit this file; do NOT pass ad-hoc video lists via shell state. "
            "Label every run using this subset with DEBUG_SUBSET in LORA_TAG. "
            "Final teacher-facing results must use the full train split."
        ),
        "schema_version": 1,
        "participant": args.participant,
        "seed": args.seed,
        "train": {
            "video_ids": sorted(train_videos),
            "summary": _summarise(train_df, train_videos),
        },
        "val": {
            "video_ids": sorted(val_videos),
            "summary": _summarise(val_df, val_videos),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.output)

    print("\nDebug subset summary:")
    print(f"  train: {output['train']['summary']}")
    print(f"  val:   {output['val']['summary']}")
    print(f"\nCommit {args.output} into git so the selection is reproducible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
