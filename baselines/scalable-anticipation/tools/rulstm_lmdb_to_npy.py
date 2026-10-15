"""Convert an RU-LSTM frame-keyed feature LMDB to per-video 4 Hz `.npy` files.

RU-LSTM keys are `{video}_frame_{:010d}.jpg` on a 30 fps clock (1-based). For
each video, step t (time t / FEAT_FPS seconds) takes key max(1, floor(t * 30 /
FEAT_FPS)); if that key is missing, the nearest earlier available key is used
(the first available key at the very start). The video length is set by its
last available key. Writes `<video>.npy` (float32, T x D) and `<video>.json`
(n_feat_frames, coverage counts) next to each other, like `rulstm_features/rgb_p01`.

Usage:
  python tools/rulstm_lmdb_to_npy.py --lmdb <dir with data.mdb> --out <dir> \
      --csv <annotation csv> [--csv ...]
Only videos referenced by the CSVs (column 2) are converted.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np

ANNO_FPS = 30.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--feat-fps", type=float, default=4.0)
    args = parser.parse_args()

    videos = set()
    for path in args.csv:
        with path.open() as stream:
            videos.update(row[1].strip() for row in csv.reader(stream) if row)
    args.out.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(args.lmdb), readonly=True, lock=False, readahead=False, max_readers=8)
    frames = defaultdict(list)
    with env.begin() as txn:
        for key in txn.cursor().iternext(keys=True, values=False):
            name = key.decode()
            video, _, rest = name.rpartition("_frame_")
            if video in videos:
                frames[video].append(int(rest.split(".")[0]))
    missing_videos = sorted(videos - set(frames))
    summary = {"lmdb": str(args.lmdb), "feat_fps": args.feat_fps, "videos": len(videos),
               "missing_videos": missing_videos, "per_video": {}}
    with env.begin() as txn:
        for video in sorted(frames):
            available = np.unique(np.asarray(frames[video], dtype=np.int64))
            n_steps = int(np.floor(available[-1] / ANNO_FPS * args.feat_fps)) + 1
            wanted = np.maximum(1, np.floor(np.arange(n_steps) * ANNO_FPS / args.feat_fps)).astype(np.int64)
            pos = np.searchsorted(available, wanted, side="right") - 1
            chosen = available[np.clip(pos, 0, None)]
            exact = int((chosen == wanted).sum())
            feats = None
            for t, frame in enumerate(chosen):
                value = txn.get(f"{video}_frame_{frame:010d}.jpg".encode())
                vector = np.frombuffer(value, dtype=np.float32)
                if feats is None:
                    feats = np.empty((n_steps, vector.size), dtype=np.float32)
                feats[t] = vector
            if not np.isfinite(feats).all():
                raise ValueError(f"Nonfinite feature in {video}")
            np.save(args.out / f"{video}.npy", feats)
            meta = {"n_feat_frames": n_steps, "feat_fps": args.feat_fps, "feat_dim": int(feats.shape[1]),
                    "last_key_frame": int(available[-1]), "exact_keys": exact,
                    "substituted_keys": n_steps - exact}
            (args.out / f"{video}.json").write_text(json.dumps(meta))
            summary["per_video"][video] = meta
    summary["total_steps"] = sum(m["n_feat_frames"] for m in summary["per_video"].values())
    summary["total_substituted"] = sum(m["substituted_keys"] for m in summary["per_video"].values())
    (args.out / "_conversion_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}))
    if missing_videos:
        raise SystemExit(f"{len(missing_videos)} videos have no features: {missing_videos[:5]}")


if __name__ == "__main__":
    main()
