"""Analyze validation prediction dumps for per-action correct/wrong patterns.

Input is the CSV produced by `PredictionDumper` in
`app.hdepic_lora_action_anticipation.gaze`.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_json_list(value: str):
    if value is None or value == "":
        return []
    try:
        out = json.loads(value)
    except json.JSONDecodeError:
        return []
    return out if isinstance(out, list) else []


def _as_int(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default


def analyze(path: Path, out_dir: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    out_dir.mkdir(parents=True, exist_ok=True)

    per_action = defaultdict(lambda: Counter(total=0, top1=0, top3=0, top5=0))
    confusion = Counter()
    examples = defaultdict(list)

    for row in rows:
        label = _as_int(row.get("action_label"))
        preds = _load_json_list(row.get("action_top5") or row.get("action_top10"))
        pred1 = _as_int(row.get("action_top1"), _as_int(preds[0] if preds else -1))
        top3_hit = _as_int(row.get("action_top3_hit"), int(label in preds[:3]))
        top5_hit = _as_int(row.get("action_top5_hit"), int(label in preds[:5]))

        counts = per_action[label]
        counts["total"] += 1
        counts["top1"] += int(pred1 == label)
        counts["top3"] += int(top3_hit)
        counts["top5"] += int(top5_hit)
        if pred1 != label:
            confusion[(label, pred1)] += 1
            if len(examples[(label, pred1)]) < 5:
                examples[(label, pred1)].append(
                    {
                        "video_id": row.get("video_id", ""),
                        "start_frame": row.get("start_frame", ""),
                        "stop_frame": row.get("stop_frame", ""),
                        "top5": preds[:5],
                    }
                )

    summary_rows = []
    for label, counts in sorted(per_action.items()):
        total = counts["total"]
        summary_rows.append(
            {
                "action_label": label,
                "total": total,
                "top1_hits": counts["top1"],
                "top3_hits": counts["top3"],
                "top5_hits": counts["top5"],
                "top1_misses": total - counts["top1"],
                "top3_misses": total - counts["top3"],
                "top5_misses": total - counts["top5"],
                "top1_accuracy": 100.0 * counts["top1"] / max(1, total),
                "top3_accuracy": 100.0 * counts["top3"] / max(1, total),
                "top5_accuracy": 100.0 * counts["top5"] / max(1, total),
            }
        )

    summary_path = out_dir / "per_action_correct_wrong.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "action_label",
            "total",
            "top1_hits",
            "top3_hits",
            "top5_hits",
            "top1_misses",
            "top3_misses",
            "top5_misses",
            "top1_accuracy",
            "top3_accuracy",
            "top5_accuracy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    confusion_path = out_dir / "action_top1_confusions.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["action_label", "pred_top1", "count", "examples_json"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (label, pred), count in confusion.most_common():
            writer.writerow(
                {
                    "action_label": label,
                    "pred_top1": pred,
                    "count": count,
                    "examples_json": json.dumps(examples[(label, pred)], ensure_ascii=False),
                }
            )

    total = sum(r["total"] for r in summary_rows)
    overall = {
        "input": str(path),
        "num_samples": total,
        "num_action_labels": len(summary_rows),
        "action_top1_accuracy": 100.0 * sum(r["top1_hits"] for r in summary_rows) / max(1, total),
        "action_top3_accuracy": 100.0 * sum(r["top3_hits"] for r in summary_rows) / max(1, total),
        "action_top5_accuracy": 100.0 * sum(r["top5_hits"] for r in summary_rows) / max(1, total),
        "worst_top3_actions": sorted(summary_rows, key=lambda r: (r["top3_accuracy"], -r["total"]))[:20],
        "best_top3_actions": sorted(summary_rows, key=lambda r: (-r["top3_accuracy"], -r["total"]))[:20],
        "top_confusions": [
            {
                "action_label": label,
                "pred_top1": pred,
                "count": count,
                "examples": examples[(label, pred)],
            }
            for (label, pred), count in confusion.most_common(20)
        ],
    }
    overall_path = out_dir / "overall.json"
    overall_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary_path, confusion_path, overall_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.prediction_csv.with_suffix("").with_name(args.prediction_csv.stem + "_analysis")
    outputs = analyze(args.prediction_csv, out_dir)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

