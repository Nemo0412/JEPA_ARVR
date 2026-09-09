#!/usr/bin/env python3
"""Classify no-register DINOv2 heads by position stability vs content equivariance.

Consumes the already-dumped original/horizontal-flip per-image received-attention
maps.  No model forward is performed.  A content-following head should have low
cross-image spatial consistency but high same-image flip-back equivariance; a
position-fixed head instead agrees with the unaligned flipped-input map.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def normalize_maps(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (x.sum(axis=(-2, -1), keepdims=True) + 1e-12)


def load_and_validate(args):
    orig_dir, flip_dir = Path(args.orig_dir), Path(args.flip_dir)
    orig_summary = json.loads((orig_dir / "summary.json").read_text())
    flip_summary = json.loads((flip_dir / "summary.json").read_text())
    for name, summary in (("original", orig_summary), ("hflip", flip_summary)):
        if summary.get("model") != "facebook/dinov2-large":
            raise RuntimeError(f"{name} input is not no-register DINOv2-large: {summary.get('model')}")
        if int(summary.get("lead_tokens(cls+reg)", -1)) != 1:
            raise RuntimeError(
                f"{name} input has {summary.get('lead_tokens(cls+reg)')} leading tokens; "
                "expected exactly one CLS token and zero register tokens"
            )
    orig = np.load(orig_dir / "per_sample_spatial.npy")
    flip = np.load(flip_dir / "per_sample_spatial.npy")
    if orig.shape != flip.shape or orig.ndim != 4:
        raise RuntimeError(f"input shape mismatch: original={orig.shape}, hflip={flip.shape}")
    images = sorted(
        p for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) != orig.shape[0]:
        raise RuntimeError(f"image/map count mismatch: images={len(images)} maps={orig.shape[0]}")
    return normalize_maps(orig), normalize_maps(flip), images, orig_summary, flip_summary


def compute_metrics(orig: np.ndarray, flip: np.ndarray) -> list[dict]:
    n, heads, grid, _ = orig.shape
    rows = []
    for h in range(heads):
        cross = [corr(orig[i, h], orig[j, h]) for i in range(n) for j in range(i + 1, n)]
        fixed = [corr(orig[i, h], flip[i, h]) for i in range(n)]
        equiv = [corr(orig[i, h], np.fliplr(flip[i, h])) for i in range(n)]
        peak = orig[:, h].max(axis=(-2, -1)) * (grid * grid)
        entropy = -(orig[:, h] * np.log(orig[:, h] + 1e-12)).sum(axis=(-2, -1))
        entropy_norm = entropy / math.log(grid * grid)
        effective = np.exp(entropy)
        fixed_mean, equiv_mean = float(np.mean(fixed)), float(np.mean(equiv))
        delta = equiv_mean - fixed_mean
        cross_mean = float(np.mean(cross))
        peak_mean = float(np.mean(peak))
        if equiv_mean >= 0.5 and delta >= 0.1 and cross_mean < 0.5:
            label = "confirmed-content-following"
        elif fixed_mean >= 0.5 and -delta >= 0.1:
            label = "position-fixed"
        elif cross_mean >= 0.5:
            label = "cross-image-stable"
        else:
            label = "mixed-or-moving-sink"
        rows.append({
            "head": h,
            "label": label,
            "cross_image_corr": round(cross_mean, 5),
            "fixed_position_corr": round(fixed_mean, 5),
            "flipback_equivariance": round(equiv_mean, 5),
            "content_delta": round(delta, 5),
            "peak_over_uniform_mean": round(peak_mean, 5),
            "peak_over_uniform_p90": round(float(np.quantile(peak, 0.9)), 5),
            "normalized_entropy_mean": round(float(np.mean(entropy_norm)), 5),
            "effective_tokens_mean": round(float(np.mean(effective)), 3),
        })
    delta = np.array([r["content_delta"] for r in rows])
    entropy = np.array([r["normalized_entropy_mean"] for r in rows])
    log_peak = np.log(np.array([r["peak_over_uniform_mean"] for r in rows]))
    z = lambda a: (a - a.mean()) / (a.std() + 1e-12)
    # Exploratory relative score: content equivariance plus a diffuse (rather
    # than single-token artifact) spatial distribution.  It ranks candidates;
    # it does not upgrade any head to a confirmed semantic-content head.
    score = z(delta) + z(entropy) - z(log_peak)
    for r, s in zip(rows, score):
        r["relative_diffuse_content_score"] = round(float(s), 5)
    return rows


def save_table(rows: list[dict], out_dir: Path, orig_summary: dict, flip_summary: dict):
    ranked_content = sorted(rows, key=lambda r: r["relative_diffuse_content_score"], reverse=True)
    ranked_least_content = sorted(rows, key=lambda r: r["content_delta"])
    ranked_peak = sorted(rows, key=lambda r: r["peak_over_uniform_mean"], reverse=True)
    report = {
        "analysis": "DINOv2 no-register per-head content equivariance",
        "model": orig_summary["model"],
        "register_tokens": 0,
        "lead_tokens": {"cls": 1, "register": 0},
        "block": orig_summary["block"],
        "n_images": orig_summary["n_images"],
        "grid": orig_summary["grid"],
        "classification_rule": {
            "confirmed-content-following": "flipback_equivariance >= 0.5, content_delta >= 0.1, cross_image_corr < 0.5",
            "position-fixed": "fixed_position_corr >= 0.5 and fixed_position_corr - flipback_equivariance >= 0.1",
            "cross-image-stable": "cross_image_corr >= 0.5 after the two rules above",
            "otherwise": "mixed-or-moving-sink",
        },
        "exploratory_relative_ranking": (
            "z(content_delta) + z(normalized_entropy) - z(log(mean peak/uniform)); "
            "used only to surface diffuse content-responsive candidates, not as a confirmed class"
        ),
        "normalization": "Each patch map sums to one; visualization color scale is shared across all images within a head.",
        "input_lead_received_mass": {
            "original_cls": orig_summary["lead_received_mass_frac"],
            "hflip_cls": flip_summary["lead_received_mass_frac"],
        },
        "content_candidates_ranked": [r["head"] for r in ranked_content],
        "least_content_aligned_ranked": [r["head"] for r in ranked_least_content],
        "sharp_sink_candidates_ranked": [r["head"] for r in ranked_peak],
        "heads": rows,
    }
    (out_dir / "head_classification.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out_dir / "head_classification.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def plot_taxonomy(rows: list[dict], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([r["content_delta"] for r in rows])
    y = np.array([r["cross_image_corr"] for r in rows])
    peak = np.array([r["peak_over_uniform_mean"] for r in rows])
    equiv = np.array([r["flipback_equivariance"] for r in rows])
    sizes = 45 + 175 * (peak - peak.min()) / (np.ptp(peak) + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(x, y, s=sizes, c=equiv, cmap="viridis",
                    edgecolor="black", linewidth=0.6)
    for r in rows:
        ax.annotate(f"h{r['head']}", (r["content_delta"], r["cross_image_corr"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(0, color="0.5", lw=1)
    ax.set_xlabel("content delta = flip-back equivariance − fixed-position correlation")
    ax.set_ylabel("cross-image spatial correlation")
    ax.set_title("DINOv2-large, no registers: per-head position/content taxonomy\n"
                 "zoomed view; no head reaches confirmed content (equiv≥0.5) or stable (cross-image≥0.5)")
    fig.colorbar(sc, ax=ax, label="flip-back equivariance")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_head_taxonomy.png", dpi=160)
    plt.close(fig)

    heads = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(heads - width / 2, [r["fixed_position_corr"] for r in rows], width,
           label="fixed coordinates: corr(A(x), A(flip(x)))")
    ax.bar(heads + width / 2, [r["flipback_equivariance"] for r in rows], width,
           label="content aligned: corr(A(x), flip-back A(flip(x)))")
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="confirmation threshold")
    ax.set_xticks(heads); ax.set_xlabel("head")
    ax.set_ylabel("same-image correlation")
    ax.set_title("DINOv2-large without registers: flip alignment by head")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_head_flip_scores.png", dpi=160)
    plt.close(fig)


def load_display_image(path: Path, processor) -> np.ndarray:
    with Image.open(path) as im:
        px = processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0].numpy()
    mean = np.asarray(processor.image_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(processor.image_std, dtype=np.float32)[:, None, None]
    return np.clip((px * std + mean).transpose(1, 2, 0), 0.0, 1.0)


def plot_candidate_overlays(orig: np.ndarray, images: list[Path], rows: list[dict], out_dir: Path, processor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(rows, key=lambda r: r["relative_diffuse_content_score"], reverse=True)[:4]
    sample_ids = np.linspace(0, len(images) - 1, 4).round().astype(int)
    fig, axes = plt.subplots(len(ranked), len(sample_ids), figsize=(14, 3.15 * len(ranked)), squeeze=False)
    for ri, row in enumerate(ranked):
        h = row["head"]
        vmax = float(np.quantile(orig[:, h], 0.995))
        for ci, si in enumerate(sample_ids):
            ax = axes[ri, ci]
            im = load_display_image(images[si], processor)
            ax.imshow(im)
            ax.imshow(orig[si, h], cmap="magma", alpha=0.58, interpolation="bilinear",
                      extent=(0, im.shape[1], im.shape[0], 0), vmin=0, vmax=vmax)
            ax.set_title(f"h{h} clip {si}\n{images[si].stem}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(
                    f"relative score={row['relative_diffuse_content_score']:.2f}\n"
                    f"Δcontent={row['content_delta']:.2f}\n"
                    f"equiv={row['flipback_equivariance']:.2f}\n"
                    f"peak={row['peak_over_uniform_mean']:.0f}×", fontsize=8
                )
    fig.suptitle("DINOv2 no-register: relative diffuse content-responsive candidates (not confirmed content heads)\n"
                 "Exact processor crop; shared color scale across images within each head", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_content_candidates_overlay.png", dpi=150)
    plt.close(fig)


def plot_flip_examples(orig: np.ndarray, flip: np.ndarray, images: list[Path], rows: list[dict], out_dir: Path, processor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    content = max(rows, key=lambda r: r["relative_diffuse_content_score"])
    sharp = max(rows, key=lambda r: r["peak_over_uniform_mean"])
    picked = []
    for r, tag in ((content, "relative diffuse/content-responsive"), (sharp, "sharpest moving sink")):
        if r["head"] not in [x[0]["head"] for x in picked]:
            picked.append((r, tag))
    si = len(images) // 2
    im = load_display_image(images[si], processor)
    im_flip = np.fliplr(im)
    fig, axes = plt.subplots(len(picked), 5, figsize=(16, 3.2 * len(picked)), squeeze=False)
    for ri, (row, tag) in enumerate(picked):
        h = row["head"]
        maps = [orig[si, h], flip[si, h], np.fliplr(flip[si, h])]
        vmax = float(np.quantile(np.concatenate([orig[:, h].ravel(), flip[:, h].ravel()]), 0.995))
        axes[ri, 0].imshow(im); axes[ri, 0].set_title(f"{tag}: h{h}\noriginal image", fontsize=8)
        axes[ri, 1].imshow(im); axes[ri, 1].imshow(maps[0], cmap="magma", alpha=.58,
            interpolation="bilinear", extent=(0, im.shape[1], im.shape[0], 0), vmin=0, vmax=vmax)
        axes[ri, 1].set_title("A(original)", fontsize=8)
        axes[ri, 2].imshow(im_flip); axes[ri, 2].imshow(maps[1], cmap="magma", alpha=.58,
            interpolation="bilinear", extent=(0, im.shape[1], im.shape[0], 0), vmin=0, vmax=vmax)
        axes[ri, 2].set_title("A(flipped input)", fontsize=8)
        axes[ri, 3].imshow(im); axes[ri, 3].imshow(maps[2], cmap="magma", alpha=.58,
            interpolation="bilinear", extent=(0, im.shape[1], im.shape[0], 0), vmin=0, vmax=vmax)
        axes[ri, 3].set_title("flip-back A(flipped)", fontsize=8)
        diff = maps[2] - maps[0]
        lim = float(np.quantile(np.abs(diff), 0.995)) + 1e-12
        axes[ri, 4].imshow(diff, cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="nearest")
        axes[ri, 4].set_title(
            f"aligned difference\nequiv={row['flipback_equivariance']:.2f}, fixed={row['fixed_position_corr']:.2f}",
            fontsize=8,
        )
        for ax in axes[ri]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("DINOv2-large without registers: same-image horizontal-flip diagnostic", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_flip_equivariance_examples.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--flip-dir", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orig, flip, images, orig_summary, flip_summary = load_and_validate(args)
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    rows = compute_metrics(orig, flip)
    report = save_table(rows, out_dir, orig_summary, flip_summary)
    plot_taxonomy(rows, out_dir)
    plot_candidate_overlays(orig, images, rows, out_dir, processor)
    plot_flip_examples(orig, flip, images, rows, out_dir, processor)
    print(json.dumps({
        "model": report["model"],
        "register_tokens": report["register_tokens"],
        "content_candidates_top4": report["content_candidates_ranked"][:4],
        "least_content_aligned_top4": report["least_content_aligned_ranked"][:4],
        "sharp_sink_candidates_top4": report["sharp_sink_candidates_ranked"][:4],
        "outputs": str(out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
