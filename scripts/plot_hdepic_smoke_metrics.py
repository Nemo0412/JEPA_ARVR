#!/usr/bin/env python
"""Plot epoch-level metrics from hdepic log_r0.csv smoke/full runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_log_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("epoch"):
                rows.append(row)
    return rows


def _to_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _metric_series(rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for row in rows:
        val = _to_float(row, key)
        if val is not None:
            out.append(val)
    return out


def plot_smoke_comparison(
    runs: list[tuple[str, Path]],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    metrics = [
        ("val-acc", "Val Top-1 (%)"),
        ("val-recall", "Val Recall@5 (%)"),
        ("train-acc", "Train Top-1 (%)"),
        ("train-recall", "Train Recall@5 (%)"),
    ]

    for metric_key, title in metrics:
        labels = []
        values = []
        for label, csv_path in runs:
            if not csv_path.is_file():
                continue
            rows = _read_log_csv(csv_path)
            series = _metric_series(rows, metric_key)
            if not series:
                continue
            labels.append(label)
            values.append(series[-1])

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4))
        bars = ax.bar(range(len(labels)), values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"][: len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"Smoke comparison — {title}")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out_path = out_dir / f"smoke_{metric_key.replace('-', '_')}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        written.append(out_path)

    # Epoch curves when multiple epochs exist
    curve_metrics = [("val-acc", "Val Top-1 (%)"), ("val-recall", "Val Recall@5 (%)")]
    for metric_key, ylabel in curve_metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        any_curve = False
        for label, csv_path in runs:
            if not csv_path.is_file():
                continue
            rows = _read_log_csv(csv_path)
            series = _metric_series(rows, metric_key)
            if len(series) < 2:
                continue
            any_curve = True
            ax.plot(range(1, len(series) + 1), series, marker="o", label=label)
        if not any_curve:
            plt.close(fig)
            continue
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Epoch curves — {ylabel}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"curves_{metric_key.replace('-', '_')}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        written.append(out_path)

    return written


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_out = root / "logs" / "figures" / "smoke_metrics"
    frozen = root / "outputs" / "hdepic_lora_action_anticipation" / "action_anticipation_frozen"

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=REL_PATH",
        help="Label and path relative to action_anticipation_frozen, e.g. fp32=tag/log_r0.csv",
    )
    args = parser.parse_args()

    if args.run:
        runs = []
        for spec in args.run:
            label, rel = spec.split("=", 1)
            runs.append((label, frozen / rel))
    else:
        runs = [
            ("fp32 gaze+pose w2", frozen / "hdepic-singleprobe-enclora-graddiag-smoke-fp32-i150-w2/log_r0.csv"),
            ("bf16 gaze+pose", frozen / "hdepic-singleprobe-enclora-graddiag-smoke-bf16-i150/log_r0.csv"),
            ("bf16 direct-bwd", frozen / "hdepic-singleprobe-enclora-direct-bwd-graddiag-smoke-bf16-i150/log_r0.csv"),
            ("bf16 RGB baseline", frozen / "hdepic-baseline-enclora-graddiag-smoke-bf16-i150/log_r0.csv"),
            ("fp32 full 10ep", frozen / "hdepic-singleprobe-fp32-full-enclora-gaze-pose-h100-r8-allblocks-bs4-10ep-w2/log_r0.csv"),
        ]

    written = plot_smoke_comparison(runs, args.out_dir)
    if not written:
        print("[plot-smoke] No figures written (missing or empty log_r0.csv)")
        return
    for path in written:
        print(f"[plot-smoke] wrote {path}")


if __name__ == "__main__":
    main()
