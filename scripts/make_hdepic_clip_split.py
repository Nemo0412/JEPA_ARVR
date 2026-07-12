#!/usr/bin/env python3
"""Create / verify the frozen HD-EPIC action-clip 80/20 split (EGTEA-style).

Canonical split directory (DO NOT regenerate casually):
  /scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/

Protocol (must stay identical across video-only and gaze+pose runs):
  1. Pool P01 clips from stage2 train+val+test CSVs (7384 rows, 27 videos).
  2. Deduplicate on (video_id, start_frame, stop_frame, verb_class, noun_class).
  3. RNG = numpy.random.RandomState(42); idx = RNG.permutation(N).
  4. n_val = max(1, int(round(0.2 * N))); first n_val indices -> val; rest -> train.
  5. test_df = val_df.copy()  (same as EGTEA prepare_egtea_split1.py).
  6. Same video may appear in train and val (clip-level split, not video-level).

Frozen counts (seed=42): train=5907, val=test=1477, train∩val videos=27.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STAGE2 = Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stage2")
DEFAULT_OUT = Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split")
SEED = 42
VAL_RATIO = 0.2
DEDUP_KEY = ["video_id", "start_frame", "stop_frame", "verb_class", "noun_class"]

# md5 of frozen CSVs created 2026-07-08 (do not change without user approval)
EXPECTED_MD5 = {
    "HD_EPIC_train_vjepa.csv": "ede421358b00f1ca2646c210237fd337",
    "HD_EPIC_val_vjepa.csv": "9073a82e14852ca83513aff492b00306",
    "HD_EPIC_test_vjepa.csv": "9073a82e14852ca83513aff492b00306",
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pool_stage2(stage2: Path) -> pd.DataFrame:
    dfs = []
    for name in ("HD_EPIC_train_vjepa.csv", "HD_EPIC_val_vjepa.csv", "HD_EPIC_test_vjepa.csv"):
        path = stage2 / name
        if not path.exists():
            raise FileNotFoundError(path)
        dfs.append(pd.read_csv(path))
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df.drop_duplicates(subset=DEDUP_KEY).reset_index(drop=True)


def split_clips(all_df: pd.DataFrame, seed: int = SEED, val_ratio: float = VAL_RATIO):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_df))
    n_val = max(1, int(round(val_ratio * len(all_df))))
    mask = np.zeros(len(all_df), dtype=bool)
    mask[idx[:n_val]] = True
    val_df = all_df.loc[mask].copy().reset_index(drop=True)
    train_df = all_df.loc[~mask].copy().reset_index(drop=True)
    test_df = val_df.copy()
    return train_df, val_df, test_df


def write_split(out: Path, train_df, val_df, test_df, verify_only: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "HD_EPIC_train_vjepa.csv": train_df,
        "HD_EPIC_val_vjepa.csv": val_df,
        "HD_EPIC_test_vjepa.csv": test_df,
    }
    if verify_only:
        for name in files:
            path = out / name
            if not path.exists():
                raise FileNotFoundError(path)
            got = md5_file(path)
            exp = EXPECTED_MD5[name]
            if got != exp:
                raise SystemExit(f"MD5 mismatch {name}: got={got} expected={exp}")
        print("verify OK: frozen clip_split matches EXPECTED_MD5")
        return

    for name, df in files.items():
        df.to_csv(out / name, index=False)

    stats = {
        "split_policy": "clip_random_like_egtea",
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "dedup_key": DEDUP_KEY,
        "pool_source": "stage2/{train,val,test}",
        "train_clips": int(len(train_df)),
        "val_clips": int(len(val_df)),
        "test_clips": int(len(test_df)),
        "train_videos": int(train_df["video_id"].nunique()),
        "val_videos": int(val_df["video_id"].nunique()),
        "train_val_video_overlap": int(len(set(train_df["video_id"]) & set(val_df["video_id"]))),
        "md5": {name: md5_file(out / name) for name in files},
        "note": (
            "Action-clip 80/20 split: same video may appear in train and val. "
            "test = copy(val). All HD-EPIC video-only and gaze+pose runs MUST use this directory."
        ),
    }
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# HD-EPIC clip_split (frozen)\n\n"
        "Do **not** overwrite these CSVs without explicit approval.\n\n"
        f"- seed={SEED}, val_ratio={VAL_RATIO}, test=val copy\n"
        f"- train={stats['train_clips']} / val={stats['val_clips']}\n"
        f"- recreate: `python scripts/make_hdepic_clip_split.py --verify`\n"
        f"- regenerate only if intentional: `python scripts/make_hdepic_clip_split.py --force`\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))
    for name, digest in stats["md5"].items():
        exp = EXPECTED_MD5.get(name)
        if exp and digest != exp:
            print(f"WARNING: {name} md5 {digest} != frozen expected {exp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true", help="Only check frozen MD5s")
    ap.add_argument("--force", action="store_true", help="Overwrite existing split CSVs")
    args = ap.parse_args()

    if args.verify:
        write_split(args.out, None, None, None, verify_only=True)
        return

    if args.out.exists() and any(args.out.glob("HD_EPIC_*.csv")) and not args.force:
        write_split(args.out, None, None, None, verify_only=True)
        print("Existing clip_split left unchanged (pass --force to overwrite).")
        return

    all_df = pool_stage2(args.stage2)
    train_df, val_df, test_df = split_clips(all_df)
    write_split(args.out, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
