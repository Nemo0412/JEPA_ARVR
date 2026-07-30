#!/usr/bin/env python3
"""
Generate Figure: Real-Time Intervention Performance comparison.
Plots VLESA vs frontier foundation models on the ASIMOV-2.0-Video benchmark.

Usage:
    python plot_intervention_comparison.py [--output fig_intervention.pdf]
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import argparse

# ── Baseline data (read from ASIMOV-2.0 Figure 5b) ──────────────────────────
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

# ── VLESA results from evaluation_1 ─────────────────────────────────────────
# abs_error=0  → 126/189 = 66.7%, MAE = 0.0
# abs_error≤0.5 → 182/189 = 96.3%, MAE ≈ 0.154
VLESA_POINTS = {
    r"VLESA": {"mae": 0.0, "rate": 66.7, "color": "#ff4500", "marker": "*"},
    r"Baseline": {"mae": 0.0, "rate": 28, "color": "#662720", "marker": "s"},
}


def main(output_path: str = "fig_intervention.pdf"):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # ── Plot baselines ───────────────────────────────────────────────────
    for name, d in BASELINES.items():
        ax.scatter(d["mae"], d["rate"], s=160, c=d["color"], marker=d["marker"],
                   zorder=3, edgecolors="white", linewidths=0.6)
        # Label
        offset_x, offset_y = 0.06, 0.8
        ha = "left"
        # Custom offsets to avoid overlap
        if name == "Claude Opus 4.1":
            offset_x, offset_y = 0.06, -2.5
        elif name == "GPT 5 Nano":
            offset_x, offset_y = 0.06, -2.5
        elif name == "GPT 5":
            offset_x, offset_y = 0.06, -2.5
        elif name == "GPT 5 Mini":
            offset_x, offset_y = 0.06, -2.5

        ax.annotate(name, (d["mae"], d["rate"]),
                    xytext=(d["mae"] + offset_x, d["rate"] + offset_y),
                    fontsize=16, color=d["color"], weight="medium",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # ── Plot VLESA ───────────────────────────────────────────────────────
    for name, d in VLESA_POINTS.items():
        ax.scatter(d["mae"], d["rate"], s=240, c=d["color"], marker=d["marker"],
                   zorder=5, edgecolors="white", linewidths=0.8)
        offset_x = 0.08
        offset_y = 2.5
        if "0.0s" in name:
            offset_y = -4.0
            offset_x = 0.05
        ax.annotate(name, (d["mae"], d["rate"]),
                    xytext=(d["mae"] + offset_x, d["rate"] + offset_y),
                    fontsize=16, color=d["color"], weight="bold",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # ── Connect VLESA points with dashed line ────────────────────────────
    vlesa_vals = list(VLESA_POINTS.values())
    ax.plot([vlesa_vals[0]["mae"]],
            [vlesa_vals[0]["rate"]],
            ls="--", color="#ff4500", alpha=0.5, lw=1.2, zorder=4)

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xlabel("Last Time of Intervention MAE (seconds, lower is better)", fontsize=16)
    ax.set_ylabel("Ratio of Accurate Interventions (%, higher is better)", fontsize=16)
    ax.set_xlim(-0.1, 2.8)
    ax.set_ylim(0, 75)
    ax.set_xticks(np.arange(0, 3.0, 0.25))
    ax.set_yticks(np.arange(0, 80, 10))
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")

    # Also save PNG
    png_path = output_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="fig_intervention.pdf")
    args = parser.parse_args()
    main(args.output)