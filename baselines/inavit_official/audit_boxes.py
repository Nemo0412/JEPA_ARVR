"""Audit the author box h5 against the frames and windows the trainer will request.

Reports the h5 schema (sample groups/datasets), the coordinate range (normalized vs pixel), the
frame-key clock (largest key per video vs the stored frame count), and, for every observable row,
how many requested frames have a box entry and how many carry at least one object / hand box.
Fails (exit 2) if a requested video is absent from the h5 or a field other than obj/hand appears
where the trainer reads.
Usage: audit_boxes.py --boxes-h5 H5 --frames DIR --csv TRAIN.csv --csv VAL.csv --out audit.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_inavit_official import box_keys, load_rows, window_indices  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes-h5", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--csv", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--obs-sec", type=float, default=4.0)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--box-clock", choices=("nominal", "native"), default="nominal")
    args = ap.parse_args()
    h5 = h5py.File(args.boxes_h5, "r")
    videos_h5 = set(h5.keys())
    report = {"h5_videos": len(videos_h5)}
    # Schema sample.
    v0 = sorted(videos_h5)[0]
    keys0 = sorted(h5[v0].keys(), key=lambda k: int(k) if k.isdigit() else -1)
    sample = h5[v0][keys0[len(keys0) // 2]]
    report["schema_sample"] = {"video": v0, "n_keys": len(keys0), "first_keys": keys0[:5], "last_keys": keys0[-3:],
                               "entry_type": type(sample).__name__,
                               "fields": {k: [list(sample[k].shape), str(sample[k].dtype)] for k in sample.keys()}
                               if isinstance(sample, h5py.Group) else None}
    field_names, coord_max, coord_min = Counter(), 0.0, 0.0
    per_split = {}
    missing_videos = set()
    clock = {}
    for path in args.csv:
        df = load_rows(path)
        df = df[df["mask"] > 0.5]
        stats = Counter()
        for video, rows in df.groupby("video_id"):
            if video not in videos_h5:
                missing_videos.add(video)
                stats["rows_video_missing"] += len(rows)
                continue
            meta = json.loads((args.frames / video / "_meta.json").read_text())
            group = h5[video]
            if video not in clock:
                numeric = [int(k) for k in group.keys() if k.isdigit()]
                clock[video] = {"n_frames": meta["n_frames"], "max_key": max(numeric) if numeric else None,
                                "min_key": min(numeric) if numeric else None, "n_keys": len(numeric)}
            for r in rows.itertuples():
                idx = window_indices(int(r.tick_frame), float(r.vfps), meta["n_frames"], args.obs_sec, args.num_frames)
                has_obj = has_hand = has_entry = 0
                for number in box_keys(video, idx, float(r.vfps), args.box_clock):
                    entry = group.get(str(int(number)))
                    if entry is None:
                        continue
                    has_entry += 1
                    field_names.update(entry.keys())
                    for key in ("obj", "hand"):
                        value = entry.get(key)
                        if value is not None and len(value):
                            arr = np.asarray(value)
                            coord_max = max(coord_max, float(arr[:, :4].max()))
                            coord_min = min(coord_min, float(arr[:, :4].min()))
                            if key == "obj":
                                has_obj += 1
                            else:
                                has_hand += 1
                stats["rows"] += 1
                stats["frames"] += len(idx)
                stats["frames_with_entry"] += has_entry
                stats["frames_with_obj"] += has_obj
                stats["frames_with_hand"] += has_hand
                stats["rows_any_obj"] += int(has_obj > 0)
                stats["rows_any_hand"] += int(has_hand > 0)
        per_split[str(path)] = dict(stats)
    report.update({"per_split": per_split, "entry_fields": dict(field_names), "coord_min": coord_min,
                   "coord_max": coord_max, "coords_look_normalized": coord_max <= 2.0,  # detector boxes may overhang the frame slightly
                   "box_clock": args.box_clock,
                   "missing_videos": sorted(missing_videos)})
    over = {v: c for v, c in clock.items() if c["max_key"] is not None and c["max_key"] > c["n_frames"]}
    ratio = [c["max_key"] / c["n_frames"] for c in clock.values() if c["max_key"]]
    report["clock"] = {"videos": len(clock), "max_key_over_n_frames": len(over), "examples_over": dict(list(over.items())[:5]),
                       "max_key_ratio_min": min(ratio) if ratio else None, "max_key_ratio_median": float(np.median(ratio)) if ratio else None,
                       "max_key_ratio_max": max(ratio) if ratio else None}
    unexpected = set(field_names) - {"obj", "hand"}
    report["status"] = "passed" if not missing_videos and not unexpected else "failed"
    report["unexpected_fields"] = sorted(unexpected)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
