#!/usr/bin/env python
"""Extract human-readable paired predictor early-exit cases from B14 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _class_names(path: Path):
    return {int(row["id"]): row["key"] for row in _read_rows(path)}


def main(args) -> None:
    train_rows = _read_rows(args.train_csv)
    val_rows = _read_rows(args.val_csv)
    # Match vjepa2/evals/action_anticipation_frozen/epickitchens.py exactly.
    train_actions = set((int(row["verb_class"]), int(row["noun_class"])) for row in train_rows)
    action_classes = {pair: index for index, pair in enumerate(train_actions)}
    reverse_actions = {index: pair for pair, index in action_classes.items()}
    verb_names = _class_names(args.verb_classes_csv)
    noun_names = _class_names(args.noun_classes_csv)
    val_by_key = {
        (row["video_id"], row["start_frame"], row["stop_frame"]): row for row in val_rows
    }

    by_sample = {}
    for row in _read_rows(args.per_sample_csv):
        sample = int(row["sample_index"])
        by_sample.setdefault(sample, {})[int(row["depth"])] = row

    def decode(action_id: int):
        verb_id, noun_id = reverse_actions[action_id]
        return {
            "action_id": action_id,
            "verb_id": verb_id,
            "verb": verb_names.get(verb_id, str(verb_id)),
            "noun_id": noun_id,
            "noun": noun_names.get(noun_id, str(noun_id)),
        }

    cases = []
    for sample in sorted(by_sample):
        rows = by_sample[sample]
        d3, d12 = rows[3], rows[12]
        early_hit = int(d3["action_top1_hit"])
        final_hit = int(d12["action_top1_hit"])
        disagree = d3["action_top1"] != d12["action_top1"]
        categories = []
        if early_hit and not final_hit:
            categories.append("d3_early_only_correct")
        if final_hit and not early_hit:
            categories.append("d12_only_correct")
        if disagree and float(d3["action_entropy_normalized"]) < args.low_entropy_threshold:
            categories.append("d3_low_entropy_disagreement")
        if not categories:
            continue
        key = (d3["video_id"], d3["start_frame"], d3["stop_frame"])
        annotation = val_by_key[key]
        cases.append(
            {
                "categories": categories,
                "sample_index": sample,
                "video_id": d3["video_id"],
                "start_frame": int(d3["start_frame"]),
                "stop_frame": int(d3["stop_frame"]),
                "narration": annotation["narration"].strip(),
                "ground_truth": decode(int(d3["action_label"])),
                "d3_prediction": decode(int(d3["action_top1"])),
                "d12_prediction": decode(int(d12["action_top1"])),
                "d3_entropy": float(d3["action_entropy_normalized"]),
                "d12_entropy": float(d12["action_entropy_normalized"]),
                "d3_vs_d12_top5_jaccard": float(d3["full_depth_top5_jaccard"]),
            }
        )
    payload = {
        "low_entropy_threshold": args.low_entropy_threshold,
        "counts": {
            category: sum(category in case["categories"] for case in cases)
            for category in (
                "d3_early_only_correct",
                "d12_only_correct",
                "d3_low_entropy_disagreement",
            )
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload["counts"], sort_keys=True))
    print(args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample-csv", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--verb-classes-csv", type=Path, required=True)
    parser.add_argument("--noun-classes-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-entropy-threshold", type=float, default=0.35)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
