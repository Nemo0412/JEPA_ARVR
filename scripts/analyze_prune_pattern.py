#!/usr/bin/env python
"""Visualize the V-JEPA2 attention-importance prune pattern: which encoder tokens survive.

Pruning keeps the top-K tokens by attention RECEIVED in the encoder's last self-attention block
(importance[j] = sum_heads sum_queries attn[i->j]). The kept tokens' true positions are cached in
the `_idx` cache as {tok, idx}. We map each kept index -> (temporal slot, spatial h, w) and plot:
  (A) temporal distribution: avg kept tokens per 0.25s slot across the 60s window (vs uniform 17.07)
  (B) spatial heatmap: avg kept count per 16x16 patch location (vs uniform 16.0)
  (C) a few per-sample spatiotemporal keep maps (slot x spatial)
Index layout: idx = slot*256 + h*16 + w ; slot in 0..239 (slot s ~ s*0.25s, high=recent/near action).
"""
import glob
import json
import os
import random

import numpy as np
import torch

CACHE = os.environ.get("IDX_CACHE",
    "/path/to/VJEPA2-EXP/data/preproc_cache_vjepa/nf480_fps8.0_px256_keep4096_idx")
OUT = os.environ.get("OUT_DIR", "/path/to/VJEPA2-EXP/outputs/prune_pattern")
METHOD_LABEL = os.environ.get("METHOD_LABEL", "attention-importance")
GRID, SLOTS = 16, 240
GP = GRID * GRID
FPS, TUBELET = 8, 2
KEEP, NTOK = 4096, SLOTS * GP
N_SAMPLE = int(os.environ.get("N_SAMPLE", "400"))

os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
random.seed(0)
samp = random.sample(files, min(N_SAMPLE, len(files)))

temporal = np.zeros(SLOTS)        # summed kept count per temporal slot
spatial = np.zeros((GRID, GRID))  # summed kept count per spatial patch
examples = []
per_sample = []
secs = np.arange(SLOTS) * TUBELET / FPS
n = 0
for f in samp:
    o = torch.load(f, map_location="cpu")
    idx = (o["idx"] if isinstance(o, dict) else o).numpy().astype(np.int64)
    idx = idx[(idx >= 0) & (idx < NTOK)]
    slot = idx // GP
    within = idx % GP
    h, w = within // GRID, within % GRID
    np.add.at(temporal, slot, 1)
    np.add.at(spatial, (h, w), 1)
    if len(examples) < 6:
        m = np.zeros((SLOTS, GP), dtype=np.uint8); m[slot, within] = 1
        examples.append((os.path.basename(f), m))
    ps_temporal = np.bincount(slot, minlength=SLOTS).astype(np.float64)
    ps_spatial = np.zeros((GRID, GRID), dtype=np.float64)
    np.add.at(ps_spatial, (h, w), 1)
    ps_center = ps_spatial[4:12, 4:12].mean()
    ps_edge = (ps_spatial.sum() - ps_spatial[4:12, 4:12].sum()) / (GRID * GRID - 64)
    per_sample.append({
        "center": float(ps_center),
        "edge": float(ps_edge),
        "center_edge_ratio": float(ps_center / max(ps_edge, 1e-12)),
        "last10": float(ps_temporal[secs >= (secs.max() - 10)].sum() / max(ps_temporal.sum(), 1)),
        "first10": float(ps_temporal[secs <= 10].sum() / max(ps_temporal.sum(), 1)),
    })
    n += 1

temporal /= n
spatial /= n
unif_t = KEEP / SLOTS          # 17.07 kept/slot if uniform
unif_s = KEEP / GP             # 16.0 kept per spatial patch (summed over slots) if uniform

# ---- numeric summary ----
last10 = temporal[secs >= (secs.max() - 10)].sum() / temporal.sum()
first10 = temporal[secs <= 10].sum() / temporal.sum()
center = spatial[4:12, 4:12].mean()
edge = (spatial.sum() - spatial[4:12, 4:12].sum()) / (GRID * GRID - 64)


def summarize_samples(values, seed=123, n_boot=5000):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    if len(arr) <= 1:
        return {"mean": round(mean, 4), "ci95": [round(mean, 4), round(mean, 4)]}
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": round(mean, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


summary = {
    "method_label": METHOD_LABEL,
    "n_samples": n, "keep": KEEP, "tokens_total": NTOK,
    "uniform_per_slot": unif_t, "uniform_per_spatial": unif_s,
    "frac_kept_last_10s": round(float(last10), 4),
    "frac_kept_first_10s": round(float(first10), 4),
    "temporal_per_slot_min": round(float(temporal.min()), 2),
    "temporal_per_slot_max": round(float(temporal.max()), 2),
    "spatial_center8x8_mean": round(float(center), 2),
    "spatial_edge_mean": round(float(edge), 2),
    "spatial_center_edge_ratio": round(float(center / max(edge, 1e-12)), 4),
    "per_sample": {
        "center_edge_ratio": summarize_samples([x["center_edge_ratio"] for x in per_sample]),
        "frac_kept_last_10s": summarize_samples([x["last10"] for x in per_sample]),
        "frac_kept_first_10s": summarize_samples([x["first10"] for x in per_sample]),
    },
}
json.dump(summary, open(os.path.join(OUT, "prune_pattern_summary.json"), "w"), indent=2)
np.save(os.path.join(OUT, "temporal.npy"), temporal)
np.save(os.path.join(OUT, "spatial.npy"), spatial)
json.dump(per_sample, open(os.path.join(OUT, "per_sample_metrics.json"), "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)

# ---- figures ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (A) temporal
    plt.figure(figsize=(10, 3.6))
    plt.bar(secs, temporal, width=TUBELET / FPS, color="#3b7", align="edge")
    plt.axhline(unif_t, color="k", ls="--", lw=1, label=f"uniform ({unif_t:.1f})")
    plt.xlabel("time in 60s window (s)  [0 = oldest, 60 = ~1s before action]")
    plt.ylabel("avg kept tokens / 0.25s slot")
    plt.title(f"Temporal prune pattern ({METHOD_LABEL}, top-{KEEP}/{NTOK}, n={n})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "A_temporal.png"), dpi=130); plt.close()

    # (B) spatial heatmap
    plt.figure(figsize=(4.6, 4))
    im = plt.imshow(spatial, cmap="magma")
    plt.colorbar(im, label="avg kept count / patch (uniform=16)")
    plt.title(f"Spatial prune pattern ({METHOD_LABEL}, 16x16 patches, n={n})")
    plt.xlabel("w"); plt.ylabel("h"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "B_spatial.png"), dpi=130); plt.close()

    # (C) example per-sample maps: collapse spatial -> show slot (y) x spatial-index (x)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, (name, m) in zip(axes.ravel(), examples):
        ax.imshow(m, aspect="auto", cmap="Greens", interpolation="nearest")
        ax.set_title(name.replace("P01__", ""), fontsize=7)
        ax.set_xlabel("spatial patch (0-255)"); ax.set_ylabel("time slot (0-239)")
    fig.suptitle("Per-sample kept-token maps (green = kept)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "C_examples.png"), dpi=120); plt.close()
    print(f"[plot] wrote A_temporal.png, B_spatial.png, C_examples.png -> {OUT}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[plot] matplotlib unavailable ({e}); saved .npy + summary json instead", flush=True)
