#!/usr/bin/env python3
"""
Plot the Pareto front of Intervention Accuracy vs Absolute Time Error,
with frontier foundation models overlaid as reference baselines.

This merges two scripts onto a single set of axes:
  * plot_pareto.py                 -> Pareto-front curves from evaluation JSONs
  * plot_intervention_comparison.py -> single-point foundation-model baselines

Both share the same axes:
    x = last-intervention MAE / absolute time error (seconds, lower is better)
    y = ratio of accurate interventions (%, higher is better)
so the foundation-model scatter points drop straight onto the Pareto curves.

Usage:
    # Compare VLESA vs prompt-based, with foundation models overlaid
    python plot_pareto.py \
        -i run_vlesa/evaluation_1_intervention_accuracy_good.json \
        -i run_baseline/evaluation_1_intervention_accuracy.json \
        -l "VLESA" -l "Prompt-based" \
        -o pareto_compare.png --annotate

    # Hide the foundation-model reference points
    python plot_pareto.py -i run/eval.json --no-baselines

    # Auto-derive labels from parent directory names
    python plot_pareto.py -i run_a/eval.json -i run_b/eval.json
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# ── Frontier foundation-model baselines (from ASIMOV-2.0 Figure 5b) ─────────
# Each point: mae in seconds, rate in percent.
BASELINES = {
    "Gemini 2.5 Pro":        {"mae": 0.65, "rate": 56.0, "color": "#2ca02c", "marker": "o"},
    "Gemini 2.5 Flash":      {"mae": 1.35, "rate": 35.0, "color": "#ff7f0e", "marker": "o"},
    "Gemini 2.5 Flash-lite": {"mae": 1.40, "rate": 29.0, "color": "#1f77b4", "marker": "o"},
    "Claude Opus 4.1":       {"mae": 2.45, "rate": 32.0, "color": "#d62728", "marker": "o"},
    "Claude Sonnet 4":       {"mae": 1.85, "rate": 41.0, "color": "#9467bd", "marker": "o"},
    "GPT 5":                 {"mae": 1.75, "rate":  9.0, "color": "#7f7f7f", "marker": "o"},
    "GPT 5 Mini":            {"mae": 1.45, "rate": 20.0, "color": "#8c564b", "marker": "o"},
    "GPT 5 Nano":            {"mae": 1.85, "rate": 30.0, "color": "#e377c2", "marker": "o"},
}

# Custom label offsets to avoid overlap (name -> (dx, dy))
BASELINE_LABEL_OFFSETS = {
    "Claude Opus 4.1": (0.06, -2.5),
    "GPT 5 Nano":      (0.06, -2.5),
    "GPT 5":           (0.06, -2.5),
    "GPT 5 Mini":      (0.06, -2.5),
}


def load_pareto(path: str) -> Tuple[List[float], List[float], List[int], int]:
    """
    Load a pareto_front from an evaluation_1_*.json file.

    Returns:
        xs: abs_error values (seconds)
        ys: ratio_accurate_intervention values (in [0, 1])
        hits: successful_interventions counts
        total: total_videos_with_gt
    """
    with open(path, "r") as f:
        data = json.load(f)

    pareto = data.get("pareto_front", [])
    if not pareto:
        raise ValueError(f"No 'pareto_front' field found in {path}")

    # Sort by abs_error to be safe
    pareto = sorted(pareto, key=lambda r: r["abs_error"])

    xs = [r["abs_error"] for r in pareto]
    ys = [r["ratio_accurate_intervention"] for r in pareto]
    hits = [r["successful_interventions"] for r in pareto]
    total = pareto[0]["total_videos_with_gt"] if pareto else 0

    return xs, ys, hits, total


def derive_label(path: str) -> str:
    """Use the parent directory name as a fallback label."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent if parent else os.path.basename(path)


def plot_pareto(
    inputs: List[str],
    labels: List[str],
    output: str,
    title: str,
    annotate: bool,
    show_step: bool,
    figsize: Tuple[float, float],
    dpi: int,
    show_baselines: bool,
):
    fig, ax = plt.subplots(figsize=figsize)

    # Reasonable distinct markers/colors for multi-curve plots
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colors = plt.cm.tab10.colors

    max_x = 0.0

    # ── Pareto-front curves from JSON ────────────────────────────────────
    for i, (path, label) in enumerate(zip(inputs, labels)):
        xs, ys, hits, total = load_pareto(path)
        ys_pct = [y * 100 for y in ys]
        max_x = max(max_x, max(xs) if xs else 0.0)

        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]

        if show_step:
            # Step plot: intervention is non-decreasing as you allow more error
            ax.step(xs, ys_pct, where="post", color=color, linewidth=2.5,
                    label=label, marker=marker, markersize=8, zorder=4)
        else:
            ax.plot(xs, ys_pct, color=color, linewidth=2.5, marker=marker,
                    markersize=8, label=label, zorder=4)

        if annotate:
            for x, y, h in zip(xs, ys_pct, hits):
                ax.annotate(
                    f"{(h / total):.2f}",
                    xy=(x, y),
                    xytext=(6, -22),
                    textcoords="offset points",
                    fontsize=13,
                    ha="center",
                    color=color,
                )

    # ── Foundation-model baselines (single reference points) ─────────────
    if show_baselines:
        for name, d in BASELINES.items():
            ax.scatter(d["mae"], d["rate"], s=150, c=d["color"],
                       marker=d["marker"], zorder=3,
                       edgecolors="white", linewidths=0.6)
            dx, dy = BASELINE_LABEL_OFFSETS.get(name, (0.06, 0.8))
            ax.annotate(
                name, (d["mae"], d["rate"]),
                xytext=(d["mae"] + dx, d["rate"] + dy),
                fontsize=11, color=d["color"], weight="medium",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            )
            max_x = max(max_x, d["mae"])

        # Single legend proxy for the baseline cloud
        ax.scatter([], [], s=110, c="#555555", marker="o",
                   edgecolors="white", linewidths=0.6,
                   label="Foundation models")

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xlabel("Intervention Absolute Time Error (seconds)", fontsize=20)
    ax.set_ylabel("Intervention Accuracy (%)", fontsize=20)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.1, max_x + 0.3)
    ax.grid(True, linestyle="--", alpha=0.5)

    # x ticks every 0.5s
    n_ticks = int(max_x / 0.5) + 2
    ax.set_xticks([round(0.5 * i, 1) for i in range(n_ticks)])
    ax.tick_params(labelsize=12)

    ax.legend(loc="lower right", fontsize=16, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {output}")

    # Also save a PDF next to the PNG if the output is a PNG
    base, ext = os.path.splitext(output)
    if ext.lower() == ".png":
        pdf_out = base + ".pdf"
        plt.savefig(pdf_out, bbox_inches="tight")
        print(f"Saved plot to: {pdf_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Pareto front of Intervention Accuracy vs Abs-Error, "
                    "with foundation-model baselines overlaid"
    )
    parser.add_argument(
        "-i", "--input", action="append", required=True,
        help="Path to evaluation_1_intervention_accuracy.json. "
             "Repeat the flag to overlay multiple runs.",
    )
    parser.add_argument(
        "-l", "--label", action="append", default=None,
        help="Display label for the corresponding --input. "
             "If omitted, the parent directory name is used.",
    )
    parser.add_argument(
        "-o", "--output", default="pareto_intervention_accuracy.png",
        help="Output image path (default: pareto_intervention_accuracy.png)",
    )
    parser.add_argument(
        "--title", default="Intervention Accuracy vs. Absolute Time Error",
        help="Plot title",
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Annotate each Pareto point with 'hits/total'",
    )
    parser.add_argument(
        "--no-step", action="store_true",
        help="Use plain line plot instead of step plot (default is step plot, "
             "which more honestly reflects the cumulative-tolerance interpretation)",
    )
    parser.add_argument(
        "--no-baselines", action="store_true",
        help="Hide the frontier foundation-model reference points",
    )
    parser.add_argument("--figsize", default="9,6",
                        help="Figure size in inches as 'W,H' (default: 9,6)")
    parser.add_argument("--dpi", type=int, default=150)

    args = parser.parse_args()

    inputs = args.input
    labels = args.label or []

    if labels and len(labels) != len(inputs):
        print(f"ERROR: number of --label ({len(labels)}) must match "
              f"number of --input ({len(inputs)})", file=sys.stderr)
        sys.exit(1)

    # Auto-derive any missing labels
    while len(labels) < len(inputs):
        labels.append(derive_label(inputs[len(labels)]))

    # Parse figsize
    try:
        w, h = (float(x) for x in args.figsize.split(","))
        figsize = (w, h)
    except Exception:
        print(f"ERROR: --figsize must be 'W,H', got: {args.figsize}", file=sys.stderr)
        sys.exit(1)

    plot_pareto(
        inputs=inputs,
        labels=labels,
        output=args.output,
        title=args.title,
        annotate=args.annotate,
        show_step=not args.no_step,
        figsize=figsize,
        dpi=args.dpi,
        show_baselines=not args.no_baselines,
    )


if __name__ == "__main__":
    main()

# Examples:
#
# Two runs overlaid + foundation-model baselines (default):
#   python plot_pareto_merge.py -i /path/to/data/vlesa-code/output_Llama-4-Scout-17B-16E-Instruct-FP8_np3_er3.0_sf-qwen3_vl_2b_grpo_single_current_norule_th0.5_a2.0_mdNone/evaluation_1_intervention_accuracy_good.json -i /path/to/data/vlesa-code/output_baseline_Llama-4-Scout-17B-16E-Instruct-FP8_np3_er3.0_sf-qwen3_vl_2b_grpo_single_current_norule_th0.5_a2.0_mdNone/evaluation_1_intervention_accuracy.json -l "VLESA (K=3)" -l "Prompt-based (K=3)"  -o pareto_compare_merge_k3.png --annotate

# python plot_pareto_merge.py -i /path/to/data/vlesa-code/output_new_Llama-4-Scout-17B-16E-Instruct-FP8_np1_er3.0_sf-qwen3_vl_2b_grpo_single_current_norule_th0.5_a2.0_mdNone/evaluation_1_intervention_accuracy_old.json -i /path/to/data/vlesa-code/output_baseline_Llama-4-Scout-17B-16E-Instruct-FP8_np1_er3.0_sf-qwen3_vl_2b_grpo_single_current_norule_th0.5_a2.0_mdNone/evaluation_1_intervention_accuracy_old.json -l "VLESA (K=1)" -l "Prompt-based (K=1)"  -o pareto_compare_merge_k1.png --annotate

# python plot_pareto_merge.py -i /path/to/data/vlesa-code/output_new_Llama-4-Scout-17B-16E-Instruct-FP8_np5_er3.0_sf-qwen3_vl_2b_grpo_single_current_norule_th0.5_a2.0_mdNone/evaluation_1_intervention_accuracy.json -i /path/to/data/vlesa-code/output_baseline_Llama-4-Scout-17B-16E-Instruct-FP8_np5_er3.0_sf-qwen3_vl_2b_grpo_single_current_norule_th0.5_a2.0_mdNone/evaluation_1_intervention_accuracy.json -l "VLESA (K=5)" -l "Prompt-based (K=5)"  -o pareto_compare_merge_k5.png --annotate
