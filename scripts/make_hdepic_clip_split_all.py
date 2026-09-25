#!/usr/bin/env python3
"""Build an EGTEA-style clip 80/20 split over ALL HD-EPIC participants.

Does not touch the frozen P01 clip_split/. Writes:
  /scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split_all/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_hdepic_clip_split import DEDUP_KEY, SEED, VAL_RATIO, md5_file, split_clips

DEFAULT_POOL = Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/full_pool")
DEFAULT_OUT = Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split_all")
SLAM_OFFICIAL = Path("/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/SLAM-and-Gaze")
SLAM_TRAIN = Path("/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze")


def pool_csvs(src: Path) -> pd.DataFrame:
    dfs = []
    for name in ("HD_EPIC_train_vjepa.csv", "HD_EPIC_val_vjepa.csv", "HD_EPIC_test_vjepa.csv"):
        path = src / name
        if path.exists():
            dfs.append(pd.read_csv(path))
    if not dfs:
        raise FileNotFoundError(f"no HD_EPIC_*_vjepa.csv under {src}")
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=DEDUP_KEY).reset_index(drop=True)


def merge_slam_mappings(official: Path, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    merged: dict[str, str] = {}
    sources = []
    for root in (dest, official):
        for p in sorted(root.glob("P0*/SLAM/multi/vrs_to_multi_slam.json")):
            text = p.read_text(encoding="utf-8").strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            merged.update(raw)
            sources.append(str(p))
    for pid_dir in sorted(official.glob("P0*")):
        train_p = dest / pid_dir.name
        if not train_p.exists():
            train_p.symlink_to(pid_dir)
    out_json = dest / "vrs_to_multi_slam_all.json"
    out_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return {"n_mappings": len(merged), "sources": sources, "mapping_json": str(out_json)}


def write_split(out: Path, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "HD_EPIC_train_vjepa.csv": train_df,
        "HD_EPIC_val_vjepa.csv": val_df,
        "HD_EPIC_test_vjepa.csv": test_df,
    }
    for name, df in files.items():
        df.to_csv(out / name, index=False)

    def _parts(df: pd.DataFrame) -> dict[str, int]:
        return {str(k): int(v) for k, v in df["participant_id"].value_counts().sort_index().items()}

    stats = {
        "split_policy": "clip_random_like_egtea_all_participants",
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "dedup_key": DEDUP_KEY,
        "train_clips": int(len(train_df)),
        "val_clips": int(len(val_df)),
        "test_clips": int(len(test_df)),
        "train_videos": int(train_df["video_id"].nunique()),
        "val_videos": int(val_df["video_id"].nunique()),
        "participants_train": _parts(train_df),
        "participants_val": _parts(val_df),
        "md5": {name: md5_file(out / name) for name in files},
        "note": (
            "All HD-EPIC participants P01-P09. Clip-level 80/20, test=copy(val). "
            "Does not overwrite the frozen P01 clip_split/."
        ),
    }
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# HD-EPIC clip_split_all (P01-P09)\n\n"
        "Clip-level 80/20 over every participant. Frozen P01 `clip_split/` is unchanged.\n\n"
        f"- seed={SEED}, val_ratio={VAL_RATIO}, test=val copy\n"
        f"- train={stats['train_clips']} / val={stats['val_clips']}\n"
        f"- train videos={stats['train_videos']} / val videos={stats['val_videos']}\n",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.glob("HD_EPIC_*.csv")) and not args.force:
        print(f"Existing {args.out} left unchanged (pass --force to overwrite).")
        print((args.out / "split_stats.json").read_text())
        return

    all_df = pool_csvs(args.pool)
    train_df, val_df, test_df = split_clips(all_df)
    stats = write_split(args.out, train_df, val_df, test_df)
    slam = merge_slam_mappings(SLAM_OFFICIAL, SLAM_TRAIN)
    stats["slam"] = slam
    (args.out / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
