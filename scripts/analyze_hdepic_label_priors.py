#!/usr/bin/env python3
"""Analyze HD-EPIC action anticipation label priors in converted CSVs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    lower = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower:
            return lower[col.lower()]
    return None


def top_items(values, k=10):
    counter = Counter(values)
    total = sum(counter.values())
    return [
        {"label": str(label), "count": count, "pct": 100.0 * count / max(1, total)}
        for label, count in counter.most_common(k)
    ]


def topk_prior_accuracy(train_values, val_values, k):
    top = [label for label, _ in Counter(train_values).most_common(k)]
    if not val_values:
        return 0.0, top
    hits = sum(1 for label in val_values if label in top)
    return 100.0 * hits / len(val_values), top


def extract_participant(video_name: str) -> str:
    text = str(video_name)
    if "/" in text:
        text = text.split("/")[-1]
    if "\\" in text:
        text = text.split("\\")[-1]
    return text.split("_")[0].split("-")[0]


def summarize_split(df: pd.DataFrame, cols: dict[str, str | None]):
    video_col = cols.get("video")
    out = {"rows": int(len(df))}
    if video_col:
        videos = df[video_col].astype(str)
        out["unique_videos"] = int(videos.nunique())
        out["participants"] = top_items([extract_participant(v) for v in videos], k=20)
    for name in ["verb", "noun", "action"]:
        col = cols.get(name)
        if col:
            values = df[col].dropna().astype(str).tolist()
            out[f"unique_{name}s"] = int(len(set(values)))
            out[f"top_{name}s"] = top_items(values, k=15)
    return out


def per_video_majority(df: pd.DataFrame, video_col: str, label_col: str):
    rows = []
    for video, group in df.groupby(video_col):
        counts = Counter(group[label_col].dropna().astype(str))
        if not counts:
            continue
        label, count = counts.most_common(1)[0]
        rows.append(
            {
                "video": str(video),
                "majority_label": str(label),
                "count": int(count),
                "rows": int(len(group)),
                "pct": 100.0 * count / max(1, len(group)),
            }
        )
    return sorted(rows, key=lambda r: r["pct"], reverse=True)


def main():
    args = parse_args()
    train = pd.read_csv(args.train_csv)
    val = pd.read_csv(args.val_csv)

    cols = {
        "video": find_col(train, ["video", "video_id", "video_path", "narration_video_id"]),
        "verb": find_col(train, ["verb_class", "verb", "verb_label"]),
        "noun": find_col(train, ["noun_class", "noun", "noun_label"]),
        "action": find_col(train, ["action_class", "action", "action_label"]),
    }

    report = {
        "inputs": {"train_csv": str(args.train_csv), "val_csv": str(args.val_csv)},
        "columns": cols,
        "train": summarize_split(train, cols),
        "val": summarize_split(val, cols),
        "prior_baselines": {},
    }

    for name in ["verb", "noun", "action"]:
        col = cols.get(name)
        if not col:
            continue
        acc, top = topk_prior_accuracy(
            train[col].dropna().astype(str).tolist(),
            val[col].dropna().astype(str).tolist(),
            args.top_k,
        )
        report["prior_baselines"][f"{name}_top{args.top_k}_from_train_frequency"] = {
            "accuracy": acc,
            "labels": [str(x) for x in top],
        }

    video_col = cols.get("video")
    if video_col:
        report["video_majorities"] = {}
        for name in ["verb", "noun", "action"]:
            col = cols.get(name)
            if col:
                report["video_majorities"][name] = per_video_majority(val, video_col, col)[:20]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote: {args.output}")
    for key, value in report["prior_baselines"].items():
        print(f"{key}: {value['accuracy']:.2f}% labels={value['labels']}")


if __name__ == "__main__":
    main()
