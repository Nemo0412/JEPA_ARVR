#!/usr/bin/env python
"""
Generate two-stage annotation splits for HD-EPIC P01.

P01 has 27 videos, all with narration annotations.
We assign them by chronological order, giving a natural temporal split where
evaluation is on the most recent recordings:

  Train  (20 videos): all 6 from 20240202 + all 12 from 20240203 + first 2 of 20240204
  Val    ( 2 videos): 20240204 videos 3–4
  Test   ( 5 videos): 20240204 last 5 videos  (held out from all training)

Stage 1 — probe vocab learning (learns ALL P01 action/verb/noun classes):
  stage1/HD_EPIC_train_vjepa.csv   all 27 videos in train
  stage1/HD_EPIC_val_vjepa.csv     the 2 val videos (for loss monitoring only)

Stage 2 — encoder fine-tuning (strict held-out test):
  stage2/HD_EPIC_train_vjepa.csv   20 train videos
  stage2/HD_EPIC_val_vjepa.csv      2 val videos
  stage2/HD_EPIC_test_vjepa.csv     5 test videos  ← held out from ALL training

Usage (reads existing converted CSVs, no video probing required):
  python scripts/make_p01_splits.py \\
      --input-csvs /path/to/HD_EPIC_train_vjepa.csv /path/to/HD_EPIC_val_vjepa.csv \\
      --output-dir /scratch/yh6416/VJEPA2-EXP/data/hdepic_vjepa_annotations
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# ─── deterministic P01 split (chronological order within 20240204) ───────────
# video_id column uses the ek100_compatible format: P01_YYYYMMDD-HHMMSS

TRAIN_VIDEOS = [
    # 20240202 — all 6
    "P01_20240202-110250",
    "P01_20240202-161354",
    "P01_20240202-161948",
    "P01_20240202-171220",
    "P01_20240202-175627",
    "P01_20240202-195538",
    # 20240203 — all 12
    "P01_20240203-093333",
    "P01_20240203-121517",
    "P01_20240203-123350",
    "P01_20240203-130505",
    "P01_20240203-132119",
    "P01_20240203-135502",
    "P01_20240203-150506",
    "P01_20240203-152323",
    "P01_20240203-152956",
    "P01_20240203-161757",
    "P01_20240203-184045",
    "P01_20240203-184214",
    # 20240204 — first 2
    "P01_20240204-095114",
    "P01_20240204-120411",
]

VAL_VIDEOS = [
    "P01_20240204-121042",
    "P01_20240204-124504",
]

TEST_VIDEOS = [
    "P01_20240204-130448",
    "P01_20240204-142301",
    "P01_20240204-145458",
    "P01_20240204-152537",
    "P01_20240204-160230",
]

# 27 videos: 20 train + 2 val + 5 test
assert len(TRAIN_VIDEOS) == 20
assert len(VAL_VIDEOS) == 2
assert len(TEST_VIDEOS) == 5
assert len(set(TRAIN_VIDEOS) | set(VAL_VIDEOS) | set(TEST_VIDEOS)) == 27, "Split must cover all 27 P01 videos"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csvs",
        nargs="+",
        required=True,
        type=Path,
        help="One or more existing converted P01 CSV files (will be merged).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root output directory; stage1/ and stage2/ subdirs are created here.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing any files.",
    )
    return p.parse_args()


def load_and_merge(csv_paths: list) -> pd.DataFrame:
    dfs = []
    for path in csv_paths:
        if path.exists():
            dfs.append(pd.read_csv(path))
        else:
            print(f"[warn] CSV not found, skipping: {path}", file=sys.stderr)
    if not dfs:
        print("[error] No input CSVs could be read.", file=sys.stderr)
        sys.exit(1)
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return df


def check_coverage(df: pd.DataFrame):
    """Verify every expected video_id appears in the data."""
    all_expected = set(TRAIN_VIDEOS) | set(VAL_VIDEOS) | set(TEST_VIDEOS)
    found = set(df["video_id"].unique())
    missing = all_expected - found
    extra = found - all_expected
    if missing:
        print(f"[warn] {len(missing)} expected video_ids missing from CSV: {sorted(missing)}", file=sys.stderr)
    if extra:
        print(f"[info] {len(extra)} extra video_ids in CSV (will be ignored in stage2): {sorted(extra)}", file=sys.stderr)


def write_split(df: pd.DataFrame, path: Path, dry_run: bool):
    if dry_run:
        print(f"  [dry-run] would write {len(df)} rows → {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  wrote {len(df):>5} rows → {path}")


def main():
    args = parse_args()

    df = load_and_merge(args.input_csvs)
    check_coverage(df)

    train_set = set(TRAIN_VIDEOS)
    val_set   = set(VAL_VIDEOS)
    test_set  = set(TEST_VIDEOS)

    train_df = df[df["video_id"].isin(train_set)].copy()
    val_df   = df[df["video_id"].isin(val_set)].copy()
    test_df  = df[df["video_id"].isin(test_set)].copy()

    # Stage 1: probe vocab learning — train on ALL 27 videos so every
    # P01 action/verb/noun class is seen during probe training.
    stage1_train_df = df.copy()
    stage1_val_df   = val_df.copy()  # same 2 val videos for loss monitoring

    # Stage 2: encoder fine-tuning — strict 20/2/5 split
    stage2_train_df = train_df
    stage2_val_df   = val_df
    stage2_test_df  = test_df

    base = args.output_dir
    print("\n[stage 1] probe vocab learning")
    write_split(stage1_train_df, base / "stage1" / "HD_EPIC_train_vjepa.csv", args.dry_run)
    write_split(stage1_val_df,   base / "stage1" / "HD_EPIC_val_vjepa.csv",   args.dry_run)

    print("\n[stage 2] encoder fine-tuning")
    write_split(stage2_train_df, base / "stage2" / "HD_EPIC_train_vjepa.csv", args.dry_run)
    write_split(stage2_val_df,   base / "stage2" / "HD_EPIC_val_vjepa.csv",   args.dry_run)
    write_split(stage2_test_df,  base / "stage2" / "HD_EPIC_test_vjepa.csv",  args.dry_run)

    # Stats
    stats = {
        "split_definition": {
            "train_videos": TRAIN_VIDEOS,
            "val_videos": VAL_VIDEOS,
            "test_videos": TEST_VIDEOS,
        },
        "stage1": {
            "train_videos": int(stage1_train_df["video_id"].nunique()),
            "val_videos":   int(stage1_val_df["video_id"].nunique()),
            "train_rows":   int(len(stage1_train_df)),
            "val_rows":     int(len(stage1_val_df)),
            "unique_verbs":   int(stage1_train_df["verb_class"].nunique()),
            "unique_nouns":   int(stage1_train_df["noun_class"].nunique()),
            "unique_actions": int(stage1_train_df.groupby(["verb_class", "noun_class"]).ngroups),
        },
        "stage2": {
            "train_videos": int(stage2_train_df["video_id"].nunique()),
            "val_videos":   int(stage2_val_df["video_id"].nunique()),
            "test_videos":  int(stage2_test_df["video_id"].nunique()),
            "train_rows":   int(len(stage2_train_df)),
            "val_rows":     int(len(stage2_val_df)),
            "test_rows":    int(len(stage2_test_df)),
        },
    }

    stats_path = base / "split_stats.json"
    if not args.dry_run:
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\n  stats → {stats_path}")

    print("\n[summary]")
    print(f"  stage1 train: {stats['stage1']['train_videos']} videos / {stats['stage1']['train_rows']} rows")
    print(f"  stage1 val:   {stats['stage1']['val_videos']} videos / {stats['stage1']['val_rows']} rows")
    print(f"  stage1 vocab: {stats['stage1']['unique_verbs']} verbs / {stats['stage1']['unique_nouns']} nouns / {stats['stage1']['unique_actions']} actions")
    print(f"  stage2 train: {stats['stage2']['train_videos']} videos / {stats['stage2']['train_rows']} rows")
    print(f"  stage2 val:   {stats['stage2']['val_videos']} videos / {stats['stage2']['val_rows']} rows")
    print(f"  stage2 test:  {stats['stage2']['test_videos']} videos / {stats['stage2']['test_rows']} rows  (held out)")


if __name__ == "__main__":
    main()
