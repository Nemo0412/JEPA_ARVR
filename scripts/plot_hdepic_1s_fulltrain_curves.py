#!/usr/bin/env python3
"""Plot train/val curves for 1s singleprobe fulltrain (log_r0.csv + val loss from .out)."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("epoch"):
                continue
            rows.append({k: float(v) for k, v in row.items() if k != "epoch"} | {"epoch": int(row["epoch"])})
    return rows


def parse_val_losses(log_path: Path) -> list[dict[str, float]]:
    pat = re.compile(
        r"validate_with_binary_input_adapter\] \[ *173\].*"
        r"loss \(v/n\): ([0-9.]+) \(([0-9.]+) ([0-9.]+)\)"
    )
    out: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            total, verb, noun = map(float, m.groups())
            out.append(
                {
                    "total": total,
                    "verb": verb,
                    "noun": noun,
                    "action": total - verb - noun,
                }
            )
    return out


def plot_curves(rows: list[dict[str, float]], losses: list[dict[str, float]], out_dir: Path, tag: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in rows]
    written: list[Path] = []

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

    # 1) Action Top-3 + Recall@5
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train-acc"] for r in rows], "o-", label="train Top-3", color="#4c78a8", lw=2)
    ax.plot(epochs, [r["val-acc"] for r in rows], "s-", label="val Top-3", color="#f58518", lw=2)
    ax.plot(epochs, [r["train-recall"] for r in rows], "o--", label="train Recall@5", color="#4c78a8", alpha=0.55)
    ax.plot(epochs, [r["val-recall"] for r in rows], "s--", label="val Recall@5", color="#f58518", alpha=0.55)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Action (%)")
    ax.set_title(f"1s fulltrain — Action metrics ({tag})")
    ax.set_xticks(epochs)
    ax.legend(loc="best")
    ax.set_ylim(0, 100)
    p1 = out_dir / "action_metrics.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    written.append(p1)

    # 2) Verb / noun val Top-3
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, task, tr_k, va_k in [
        (axes[0], "Verb", "train-acc-verb", "val-acc-verb"),
        (axes[1], "Noun", "train-acc-noun", "val-acc-noun"),
    ]:
        ax.plot(epochs, [r[tr_k] for r in rows], "o-", label=f"train Top-3", color="#54a24b", lw=2)
        ax.plot(epochs, [r[va_k] for r in rows], "s-", label=f"val Top-3", color="#e45756", lw=2)
        ax.set_title(task)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Top-3 (%)")
        ax.set_xticks(epochs)
        ax.legend()
        ax.set_ylim(0, 100)
    fig.suptitle(f"1s fulltrain — Verb/Noun Top-3 ({tag})", y=1.02)
    p2 = out_dir / "verb_noun_top3.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(p2)

    # 3) Val loss
    if losses and len(losses) >= len(rows):
        le = epochs
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(le, [l["total"] for l in losses[: len(rows)]], "o-", label="total", color="#333", lw=2)
        ax.plot(le, [l["action"] for l in losses[: len(rows)]], "s-", label="action", color="#4c78a8")
        ax.plot(le, [l["verb"] for l in losses[: len(rows)]], "^-", label="verb", color="#54a24b")
        ax.plot(le, [l["noun"] for l in losses[: len(rows)]], "d-", label="noun", color="#e45756")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val loss (CE sum over heads)")
        ax.set_title(f"1s fulltrain — Val loss ({tag})")
        ax.set_xticks(le)
        ax.legend()
        p3 = out_dir / "val_loss.png"
        fig.tight_layout()
        fig.savefig(p3, dpi=160)
        plt.close(fig)
        written.append(p3)

    # 4) Dashboard 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(epochs, [r["train-acc"] for r in rows], "o-", label="train", color="#4c78a8")
    ax.plot(epochs, [r["val-acc"] for r in rows], "s-", label="val", color="#f58518")
    ax.set_title("Action Top-3")
    ax.set_xticks(epochs)
    ax.legend()
    ax.set_ylim(0, 100)

    ax = axes[0, 1]
    ax.plot(epochs, [r["train-recall"] for r in rows], "o-", label="train", color="#4c78a8")
    ax.plot(epochs, [r["val-recall"] for r in rows], "s-", label="val", color="#f58518")
    ax.set_title("Action Recall@5")
    ax.set_xticks(epochs)
    ax.legend()
    ax.set_ylim(0, 100)

    ax = axes[1, 0]
    ax.plot(epochs, [r["val-acc-verb"] for r in rows], "s-", label="verb", color="#54a24b")
    ax.plot(epochs, [r["val-acc-noun"] for r in rows], "s-", label="noun", color="#e45756")
    ax.set_title("Val Top-3 (verb/noun)")
    ax.set_xticks(epochs)
    ax.legend()
    ax.set_ylim(0, 100)

    ax = axes[1, 1]
    if losses:
        ax.plot(epochs, [l["total"] for l in losses[: len(rows)]], "o-", label="total", color="#333")
        ax.plot(epochs, [l["action"] for l in losses[: len(rows)]], "s-", label="action", color="#4c78a8")
        ax.set_title("Val loss")
        ax.set_xticks(epochs)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no val loss in log", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Val loss")

    fig.suptitle(
        "B11 1s concat_ar fulltrain — ViT-L fp32 bs8\n"
        f"native / metric_wise_max | {tag}",
        fontsize=13,
    )
    p4 = out_dir / "dashboard.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(p4)

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="1s-fulltrain")
    args = ap.parse_args()

    rows = read_csv(args.csv)
    losses = parse_val_losses(args.log) if args.log and args.log.is_file() else []
    paths = plot_curves(rows, losses, args.out_dir, args.tag)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
