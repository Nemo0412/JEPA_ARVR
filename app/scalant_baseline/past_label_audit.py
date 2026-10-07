"""Measure whether official action intervals can supervise ScalAnt past tokens.

Reports overlap/background rates only. It does not settle a training label
policy or materialize labels for use in an experiment.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .egtea_cache import ANNOTATIONS, MAIN_ROOT


def rows_for(split: int, population: str):
    path = ANNOTATIONS / f"{population}{split}.csv"
    with path.open(newline="") as stream:
        return [(int(r[0]), r[1], int(r[2]), int(r[3]), int(r[6]))
                for r in csv.reader(stream)]


def audit(split: int, horizon: int, population: str):
    rows = rows_for(split, population)
    by_video = defaultdict(list)
    for row in rows:
        by_video[row[1]].append(row)
    counts = {"background": 0, "unique": 0, "same_label_overlap": 0,
              "conflicting_overlap": 0, "negative_frame": 0,
              "target_instance_visible": 0}
    ambiguous_examples = []
    for uid, video, start, _end, target in rows:
        for index in range(16):
            frame = math.floor(start - (horizon + 3.75 - index * .25) * 30)
            if frame < 1:
                counts["negative_frame"] += 1
                frame = 1
            matches = [item for item in by_video[video]
                       if item[2] <= frame < item[3]]
            labels = {item[4] for item in matches}
            if any(item[0] == uid for item in matches):
                counts["target_instance_visible"] += 1
            if not matches:
                counts["background"] += 1
            elif len(labels) > 1:
                counts["conflicting_overlap"] += 1
                if len(ambiguous_examples) < 8:
                    ambiguous_examples.append({"uid": uid, "video": video,
                                               "frame": frame, "labels": sorted(labels)})
            elif len(matches) > 1:
                counts["same_label_overlap"] += 1
            else:
                counts["unique"] += 1
    return {"split": split, "horizon": horizon, "population": population,
            "rows": len(rows), "sampled_past_tokens": len(rows) * 16,
            "counts": counts, "ambiguous_examples": ambiguous_examples}


def main():
    results = [audit(split, horizon, population)
               for split in (1, 2, 3) for horizon in (2, 6)
               for population in ("training", "validation")]
    video_membership = {}
    for split in (1, 2, 3):
        train = {row[1] for row in rows_for(split, "training")}
        val = {row[1] for row in rows_for(split, "validation")}
        video_membership[split] = {"train_videos": len(train),
                                   "validation_videos": len(val),
                                   "shared_videos": len(train & val)}
    report = {"kind": "past_label_feasibility_not_training_policy",
              "interval_convention": "[start_frame,end_frame)",
              "sample_clock": "30 Hz, 4 Hz observations, first source-cache 16 positions",
              "video_membership": video_membership, "results": results}
    out = MAIN_ROOT / "outputs/b19_scalant/implementation/past_label_feasibility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print(json.dumps({key: result[key] for key in
                          ("split", "horizon", "population", "rows", "counts")}),
              flush=True)
    print(json.dumps(video_membership), flush=True)


if __name__ == "__main__":
    main()
