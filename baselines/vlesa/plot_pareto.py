#!/usr/bin/env python3
"""
Plot the Pareto front of Intervention Accuracy vs Abs-Error (seconds).

Reads one or more `evaluation_1_intervention_accuracy.json` files and draws
a step/line plot of `ratio_accurate_intervention` vs `abs_error`.

Usage:
    # Single run
    python plot_pareto.py --input output_dir/evaluation_1_intervention_accuracy.json

    # Compare multiple runs on the same axes
    python plot_pareto.py \
        --input run_qwen/evaluation_1_intervention_accuracy.json \
        --input run_llama/evaluation_1_intervention_accuracy.json \
        --label "Qwen3-VL (fine-tuned)" \
        --label "Llama-4-Scout (prompt)" \
        --output pareto_compare.png

    # Auto-derive labels from parent directory names
    python plot_pareto.py -i run_a/evaluation_1_intervention_accuracy.json \
                         -i run_b/evaluation_1_intervention_accuracy.json
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt


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
):
    fig, ax = plt.subplots(figsize=figsize)

    # Reasonable distinct markers for multi-curve plots
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colors = plt.cm.tab10.colors

    max_x = 0.0
    totals_for_subtitle = []

    for i, (path, label) in enumerate(zip(inputs, labels)):
        xs, ys, hits, total = load_pareto(path)
        ys_pct = [y * 100 for y in ys]
        max_x = max(max_x, max(xs) if xs else 0.0)
        totals_for_subtitle.append((label, total))

        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]

        if show_step:
            # Step plot: intervention is non-decreasing as you allow more error
            ax.step(xs, ys_pct, where="post", color=color, linewidth=2,
                    label=label, marker=marker, markersize=7)
        else:
            ax.plot(xs, ys_pct, color=color, linewidth=2, marker=marker,
                    markersize=7, label=label)

        if annotate:
            for x, y, h in zip(xs, ys_pct, hits):
                ax.annotate(
                    f"{(h / total):.2f}",
                    xy=(x, y),
                    xytext=(6, -22),
                    textcoords="offset points",
                    fontsize=16,
                    ha="center",
                    color=color,
                )

    ax.set_xlabel("Absolute Time Error (seconds)", fontsize=20)
    ax.set_ylabel("Intervention Accuracy (%)", fontsize=20)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.1, max_x + 0.3)
    ax.grid(True, linestyle="--", alpha=0.5)

    # x ticks every 0.5s
    n_ticks = int(max_x / 0.5) + 1
    ax.set_xticks([round(0.5 * i, 1) for i in range(n_ticks)])

    ax.legend(loc="lower right", fontsize=18, framealpha=0.9)

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
        description="Plot Pareto front of Intervention Accuracy vs Abs-Error"
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
        help="Annotate each point with 'hits/total'",
    )
    parser.add_argument(
        "--no-step", action="store_true",
        help="Use plain line plot instead of step plot (default is step plot, "
             "which more honestly reflects the cumulative-tolerance interpretation)",
    )
    parser.add_argument("--figsize", default="8,5",
                        help="Figure size in inches as 'W,H' (default: 8,5)")
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
    )


if __name__ == "__main__":
    main()

# Examples:
#
# Single run:
#   python plot_pareto.py -i out_dir/evaluation_1_intervention_accuracy.json
#
# Two runs overlaid, with custom labels and annotations:
#   python plot_pareto.py \
#       -i run_qwen/evaluation_1_intervention_accuracy.json \
#       -i run_llama/evaluation_1_intervention_accuracy.json \
#       -l "Qwen3-VL (fine-tuned)" \
#       -l "Llama-4-Scout (prompt-based)" \
#       -o pareto_compare.png \
#       --annotate

# python plot_pareto.py -i /path/to/data/vlesa-code/output_Llama-4-Scout-17B-16E-Instruct-FP8_np3_er3.0_sf-qwen3_vl_2b_grpo_single_current_norule_th0.5_a2.0_mdNone/evaluation_1_intervention_accuracy_good.json -i /path/to/data/vlesa-code/output_baseline_Llama-4-Scout-17B-16E-Instruct-FP8_np3_er3.0_sf-qwen3_vl_2b_grpo_single_current_norule_th0.5_a2.0_mdNone/evaluation_1_intervention_accuracy.json -l "VLESA" -l "Prompt-based" -o pareto_compare.png --annotate