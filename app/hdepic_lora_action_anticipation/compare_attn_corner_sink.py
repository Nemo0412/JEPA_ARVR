#!/usr/bin/env python3
# [ATTN-CORNER-SINK] Post-processing: recency annotation + cross-model comparison.
# Pure numpy/matplotlib over the saved per_head_thw.npy tensors -- no GPU/torch.
"""Annotate per-head recency (r, rec) + pattern label, and compare models A-D.

For each run dir (from analyze_encoder_head_attn_corners.py) reads
``per_head_thw.npy`` (heads, T_slots, grid, grid) and computes, per head:
  * r    = Spearman(token importance, temporal slot)   (recency correlation)
  * rec  = fraction of importance in the recent half of slots (uniform=0.5)
  * cornerblk_over_uniform, peak_over_uniform, center_over_edge (spatial)
  * a pattern label using the hdepic vocabulary: recency+peak / edge /
    recency / center-peak.

Outputs an annotated per-head figure per run (matching the hdepic L23 figure)
and a cross-model comparison figure (corner-mass curves + top-corner-head maps +
2.1 register/CLS mass).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rank(a):
    order = a.argsort()
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(a))
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    starts = csum - cnt
    avg = (starts + csum - 1) / 2.0
    return avg[inv]


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def corner_metrics(spatial):
    g = spatial.shape[0]
    total = float(spatial.sum()) + 1e-12
    uniform = 1.0 / (g * g)
    b = max(1, g // 8)
    cb = (spatial[:b, :b].sum() + spatial[:b, -b:].sum()
          + spatial[-b:, :b].sum() + spatial[-b:, -b:].sum())
    edge_ring = (spatial.sum() - spatial[1:-1, 1:-1].sum()) / (g * g - (g - 2) * (g - 2))
    q = g // 4
    center = spatial[g // 2 - q:g // 2 + q, g // 2 - q:g // 2 + q].mean()
    return {
        "cornerblk_over_uniform": float((cb / (4 * b * b)) / uniform / total),
        "peak_over_uniform": float(spatial.max() / uniform / total),
        "center_over_edge": float(center / (edge_ring + 1e-12)),
    }


def label_head(r, rec, cm):
    """hdepic-style label from recency + spatial stats."""
    peak = cm["peak_over_uniform"] > 3.0
    recent = r > 0.35 or rec > 0.5
    corner_or_edge = cm["cornerblk_over_uniform"] > 1.6 and cm["center_over_edge"] < 0.8
    if corner_or_edge and not peak:
        base = "edge"
    elif cm["center_over_edge"] > 1.2 and peak:
        base = "center-peak"
    else:
        base = "diffuse"
    parts = []
    if recent:
        parts.append("recency")
    if peak:
        parts.append("peak")
    return "+".join(parts) if parts else base if base != "diffuse" else (base)


def analyze_run(run_dir: Path):
    thw = np.load(run_dir / "per_head_thw.npy").astype(np.float64)  # (H, T, g, g)
    H, T, g, _ = thw.shape
    slot_idx = np.repeat(np.arange(T), g * g).astype(np.float64)    # per-token slot
    spatial = thw.sum(axis=1)                                       # (H, g, g)
    spatial = spatial / (spatial.sum(axis=(1, 2), keepdims=True) + 1e-12)
    per_head = []
    for h in range(H):
        flat = thw[h].reshape(-1)
        temporal = thw[h].reshape(T, -1).sum(axis=1)
        tot = temporal.sum() + 1e-12
        rec = float(temporal[T // 2:].sum() / tot)                 # recent-half fraction
        r = spearman(flat, slot_idx)
        cm = corner_metrics(spatial[h])
        per_head.append({"head": h, "r": round(r, 3), "rec": round(rec, 3),
                         "label": label_head(r, rec, cm), **{k: round(v, 3) for k, v in cm.items()}})
    return thw, spatial, per_head, (H, T, g)


def fig_per_head(run_dir, spatial, per_head, meta, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, T, g = meta
    ncol = 4
    nrow = int(np.ceil(H / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.2 * nrow))
    for h in range(nrow * ncol):
        ax = axes.flat[h]
        if h < H:
            ax.imshow(spatial[h], cmap="viridis")
            p = per_head[h]
            ax.set_title(f"h{h} {p['label']}\nr={p['r']:.2f} rec={p['rec']:.2f} cx{p['cornerblk_over_uniform']:.1f}",
                         fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out = run_dir / "fig_annotated_per_head.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="label=dir pairs, e.g. finetuned20=outputs/.../finetuned20_res256_f80")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for spec in args.runs:
        label, d = spec.split("=", 1)
        runs[label] = Path(d)

    report = {}
    store = {}
    for label, d in runs.items():
        if not (d / "per_head_thw.npy").exists():
            print(f"[skip] {label}: no per_head_thw.npy at {d}", flush=True)
            continue
        thw, spatial, per_head, meta = analyze_run(d)
        store[label] = (thw, spatial, per_head, meta)
        smry = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        title = f"{label}  L{smry.get('block','?')} per-head received attn (rec annotated)  grid{meta[2]} slots{meta[1]}"
        fpath = fig_per_head(d, spatial, per_head, meta, title)
        corner_sorted = sorted([p["cornerblk_over_uniform"] for p in per_head], reverse=True)
        n_corner = sum(1 for p in per_head if p["cornerblk_over_uniform"] > 1.6)
        report[label] = {
            "grid": meta[2], "slots": meta[1], "n_heads": meta[0],
            "register_cls_mass_per_head": smry.get("register_cls_mass_per_head"),
            "n_corner_heads(cb>1.6)": n_corner,
            "corner_sorted": [round(x, 2) for x in corner_sorted],
            "mean_recency_r": round(float(np.mean([p["r"] for p in per_head])), 3),
            "per_head": per_head,
            "annotated_fig": str(fpath),
        }
        print(f"[{label}] corner_heads={n_corner}  top_corner={corner_sorted[0]:.2f}  "
              f"mean_r={report[label]['mean_recency_r']}", flush=True)

    # cross-model comparison figure
    _compare_fig(args.out_dir, store, report)
    (args.out_dir / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {kk: report[k][kk] for kk in
          ("grid", "n_corner_heads(cb>1.6)", "corner_sorted", "register_cls_mass_per_head", "mean_recency_r")}
          for k in report}, indent=2), flush=True)


def _compare_fig(out_dir, store, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(store.keys())
    if not labels:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    # (left) sorted corner-mass curves
    ax = axes[0]
    for label in labels:
        cs = report[label]["corner_sorted"]
        ax.plot(range(len(cs)), cs, "o-", label=f"{label} (g{report[label]['grid']})")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform")
    ax.axhline(1.6, color="grey", ls=":", lw=1, label="corner-head thresh")
    ax.set_xlabel("head rank"); ax.set_ylabel("corner-block / uniform")
    ax.set_title("Per-head corner concentration (sorted)")
    ax.legend(fontsize=8)
    # (right) top-corner-head spatial map per model
    ax = axes[1]; ax.axis("off")
    n = len(labels)
    sub = fig.add_gridspec(1, n, left=0.55, right=0.98, top=0.85, bottom=0.15)
    for i, label in enumerate(labels):
        thw, spatial, per_head, meta = store[label]
        best = max(range(meta[0]), key=lambda h: per_head[h]["cornerblk_over_uniform"])
        a = fig.add_subplot(sub[0, i])
        a.imshow(spatial[best], cmap="viridis")
        a.set_title(f"{label}\nh{best} cx{per_head[best]['cornerblk_over_uniform']:.1f}", fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("V-JEPA2 vs 2.1 last-block corner sink comparison", fontsize=12)
    out = out_dir / "fig3_model_comparison.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[compare] saved {out}", flush=True)


if __name__ == "__main__":
    main()
