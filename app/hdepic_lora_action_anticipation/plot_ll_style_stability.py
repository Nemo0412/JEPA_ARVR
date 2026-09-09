#!/usr/bin/env python3
# [ATTN-CORNER-SINK] ll-style views: per-head cross-video STABILITY + stable-vs-content
# dichotomy. Ports the derived analysis from ll's JEPA_ARVR
# scripts/analyze_encoder_last_layer_multiclip.py + plot_L23_explainer_figures.py onto
# our saved per-clip maps (per_sample_spatial.npy). Our own figs are kept separately.
"""Cross-video head-stability + stable/content figures from per-clip spatial maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def head_masses(sp):
    """sp: (grid,grid) normalized. ll-style border(excl-corner)/corner/center masses."""
    g = sp.shape[0]
    s = sp / (sp.sum() + 1e-12)
    e = max(1, g // 8)
    corner = s[:e, :e].sum() + s[:e, -e:].sum() + s[-e:, :e].sum() + s[-e:, -e:].sum()
    border = (s[:e, :].sum() + s[-e:, :].sum() + s[:, :e].sum() + s[:, -e:].sum()) - corner
    center = s[g // 4:3 * g // 4, g // 4:3 * g // 4].sum()
    return float(corner), float(border), float(center)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stable-thresh", type=float, default=0.5)
    args = ap.parse_args()
    rd = Path(args.run_dir)
    ps = np.load(rd / "per_sample_spatial.npy").astype(np.float64)  # (n, heads, g, g)
    n, H, g, _ = ps.shape
    # normalize each clip/head map to sum 1
    ps = ps / (ps.sum(axis=(2, 3), keepdims=True) + 1e-12)

    mean_corr, corner_m, border_m, center_m, labels = [], [], [], [], []
    for h in range(H):
        stack = ps[:, h]                                    # (n, g, g)
        cs = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = stack[i].ravel(), stack[j].ravel()
                if a.std() > 1e-8 and b.std() > 1e-8:
                    cs.append(np.corrcoef(a, b)[0, 1])
        mc = float(np.mean(cs)) if cs else 0.0
        cm, bm, ce = head_masses(stack.mean(0))
        mean_corr.append(mc); corner_m.append(cm); border_m.append(bm); center_m.append(ce)
        if mc > args.stable_thresh and (cm + bm) > ce:
            labels.append("stable")
        elif mc < 0.35 and ce > (cm + bm):
            labels.append("content")
        else:
            labels.append("mixed")
    stable = [h for h in range(H) if labels[h] == "stable"]
    content = [h for h in range(H) if labels[h] == "content"]
    summary = {"n_clips": n, "n_heads": H, "stable_heads": stable, "content_heads": content,
               "mean_corr": [round(x, 3) for x in mean_corr],
               "corner_mass": [round(x, 3) for x in corner_m],
               "center_mass": [round(x, 3) for x in center_m], "labels": labels}
    (rd / "ll_stability.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("stable_heads", "content_heads")}, indent=2))
    print("mean_corr:", [round(x, 2) for x in mean_corr])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig B: cross-video consistency bar (ll L23_cross_video_corr)
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = ["#4CAF50" if c > 0.5 else "#FF9800" if c > 0.3 else "#F44336" for c in mean_corr]
    ax.bar(range(H), mean_corr, color=cols)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("head (green=stable/consistent, red=content/variable)")
    ax.set_ylabel("mean cross-video spatial corr")
    ax.set_title(f"Cross-video head consistency ({n} clips)  stable={stable} content={content}", fontsize=9)
    fig.tight_layout(); fig.savefig(rd / "ll_cross_video_corr.png", dpi=140); plt.close(fig)

    # Fig C: stable vs content — a stable head and a content head across 3 clips + mean
    def pick(cands, default):
        return cands[0] if cands else default
    hs = pick(sorted(stable, key=lambda h: -(corner_m[h] + border_m[h])), int(np.argmax([corner_m[i] + border_m[i] for i in range(H)])))
    hc = pick(sorted(content, key=lambda h: -center_m[h]), int(np.argmax(center_m)))
    cols_show = min(3, n)
    idxs = np.linspace(0, n - 1, cols_show).round().astype(int)
    fig, axes = plt.subplots(2, cols_show + 1, figsize=(3 * (cols_show + 1), 6), squeeze=False)
    for row, (h, tag) in enumerate([(hs, f"STABLE h{hs}"), (hc, f"CONTENT h{hc}")]):
        for c, si in enumerate(idxs):
            m = ps[si, h]; axes[row][c].imshow(m / (m.max() + 1e-12), cmap="viridis")
            axes[row][c].set_title(f"{tag}\nclip {si}", fontsize=8)
            axes[row][c].set_xticks([]); axes[row][c].set_yticks([])
        mm = ps[:, h].mean(0); axes[row][-1].imshow(mm / (mm.max() + 1e-12), cmap="viridis")
        axes[row][-1].set_title(f"{tag}\n{n}-clip mean (r={mean_corr[h]:.2f})", fontsize=8)
        axes[row][-1].set_xticks([]); axes[row][-1].set_yticks([])
    fig.suptitle(f"Stable (corner/border anchor) vs Content (center, varies) — {rd.name}", fontsize=11)
    fig.tight_layout(); fig.savefig(rd / "ll_stable_vs_content.png", dpi=140); plt.close(fig)
    print(f"[ll-style] saved ll_cross_video_corr.png + ll_stable_vs_content.png to {rd}")


if __name__ == "__main__":
    main()
