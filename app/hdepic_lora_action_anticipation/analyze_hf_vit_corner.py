#!/usr/bin/env python3
# [ATTN-CORNER-SINK] Generic HF ViT control: is the last-block corner/artifact-token
# sink specific to JEPA, or general across visual encoders (supervised / contrastive /
# SSL, with/without registers)? Pure transformers/PIL/numpy; jepa-3dgs can3tok env.
"""Per-head last-block received-attention corner analysis for any HF ViT encoder.

Uses AutoImageProcessor (correct per-model preprocessing) + output_attentions. Patch
tokens are the LAST grid^2 tokens (CLS / register tokens are leading and skipped for
the spatial map, but their received-attention mass is reported separately). Supports a
block sweep. Handles DINOv2 (+/- registers), CLIP vision tower, supervised ViT, I-JEPA.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


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
        "border_over_center": float(edge_ring / (center * uniform * g * g + 1e-12)),
        "peak_over_uniform": float(spatial.max() / uniform / total),
    }


def load_model(model_id):
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(model_id)
    if "clip" in model_id.lower():
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained(model_id, attn_implementation="eager")
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id, attn_implementation="eager")
    return proc, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--block", type=int, default=-1)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-images", type=int, default=0)
    ap.add_argument("--hflip", action="store_true", help="horizontally mirror input (position-vs-content test)")
    ap.add_argument("--dump-per-sample", action="store_true", help="save per-image spatial maps (ll-style stability)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load_model(args.model_id)
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    files = sorted(sum([glob.glob(os.path.join(args.img_dir, e)) for e in ("*.jpg", "*.png", "*.jpeg")], []))
    if args.n_images > 0:
        files = files[: args.n_images]
    print(f"[{args.model_id}] {len(files)} images", flush=True)

    acc = None
    sweep_acc = None
    lead_mass = 0.0
    per_sample = []            # for ll-style cross-image stability
    n_used = 0
    n_heads = grid = off = n_layers = None
    for f in files:
        im = Image.open(f).convert("RGB")
        px = proc(images=im, return_tensors="pt")["pixel_values"].to(device)
        if args.hflip:
            px = torch.flip(px, dims=[-1])
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = model(pixel_values=px, output_attentions=True)
        attns = out.attentions                       # tuple[L] (B, heads, N, N)
        if n_heads is None:
            n_layers = len(attns)
            n_heads = attns[0].shape[1]
            N = attns[0].shape[-1]
            grid = int(math.isqrt(N))
            while grid * grid > N:
                grid -= 1
            # pick the largest square <= N that also leaves a small non-neg lead (cls/reg)
            off = N - grid * grid
            acc = np.zeros((n_heads, grid, grid), dtype=np.float64)
            if args.sweep:
                sweep_acc = np.zeros((n_layers, n_heads), dtype=np.float64)
            blk = args.block if args.block >= 0 else n_layers + args.block
            print(f"  layers={n_layers} heads={n_heads} N={N} grid={grid} lead(cls/reg)={off} block={blk}", flush=True)
        for li, a in enumerate(attns):
            recv = a[0].float().sum(dim=1)            # (heads, N) received per key
            lead = recv[:, :off].sum(dim=1)           # cls/register received mass
            patch = recv[:, off:off + grid * grid]
            denom = recv.sum(dim=1, keepdim=True) + 1e-12
            if li == blk:
                lead_mass += float((lead / denom.squeeze(1)).mean())
            pn = (patch / denom).reshape(n_heads, grid, grid).cpu().numpy()
            if args.sweep:
                for h in range(n_heads):
                    sweep_acc[li, h] += corner_metrics(pn[h])["cornerblk_over_uniform"]
            if li == blk:
                acc += pn
                if args.dump_per_sample:
                    per_sample.append(pn.astype(np.float32))
        n_used += 1
    if n_used == 0:
        raise SystemExit("no images")

    spatial = acc / n_used
    spatial = spatial / (spatial.sum(axis=(1, 2), keepdims=True) + 1e-12)
    metrics = [corner_metrics(spatial[h]) for h in range(n_heads)]
    order = sorted(range(n_heads), key=lambda h: metrics[h]["cornerblk_over_uniform"], reverse=True)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "per_head_spatial.npy", spatial.astype(np.float32))
    if args.dump_per_sample and per_sample:
        np.save(outdir / "per_sample_spatial.npy", np.stack(per_sample))  # (n, heads, grid, grid)
    summary = {
        "model": args.model_id, "grid": grid, "block": blk, "n_layers": n_layers,
        "n_heads": n_heads, "n_images": n_used, "lead_tokens(cls+reg)": off,
        "lead_received_mass_frac": round(lead_mass / n_used, 4),
        "corner_sorted": [round(metrics[h]["cornerblk_over_uniform"], 2) for h in order],
        "n_corner_heads": sum(1 for h in range(n_heads) if metrics[h]["cornerblk_over_uniform"] > 1.6),
        "per_head_corner": {str(h): {k: round(v, 3) for k, v in metrics[h].items()} for h in range(n_heads)},
    }
    if args.sweep:
        sw = sweep_acc / n_used
        summary["sweep_max_cornerblk_per_layer"] = [round(float(sw[li].max()), 2) for li in range(n_layers)]
        summary["sweep_ncorner_per_layer"] = [int((sw[li] > 1.6).sum()) for li in range(n_layers)]
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("grid", "block", "n_corner_heads", "corner_sorted",
          "lead_received_mass_frac") if k in summary}, indent=2), flush=True)
    if args.sweep:
        print("sweep max cornerblk/layer:", summary["sweep_max_cornerblk_per_layer"], flush=True)
    _figs(outdir, spatial, metrics, args, summary)


def _figs(outdir, spatial, metrics, args, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = spatial.shape[0]
    ncol = 4
    nrow = int(np.ceil(H / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.2 * nrow))
    for h in range(nrow * ncol):
        ax = axes.flat[h]
        if h < H:
            ax.imshow(spatial[h], cmap="viridis")
            cb = metrics[h]["cornerblk_over_uniform"]
            ax.set_title(f"h{h} corner×{cb:.1f}", fontsize=8, color=("red" if cb > 1.6 else "black"))
        ax.set_xticks([]); ax.set_yticks([])
    tag = args.model_id.split("/")[-1]
    fig.suptitle(f"{tag}  block {summary['block']} per-head received attn ({summary['n_images']} imgs, "
                 f"cls/reg mass {summary['lead_received_mass_frac']:.2f})")
    fig.tight_layout(); fig.savefig(outdir / "fig_per_head.png", dpi=120); plt.close(fig)
    if args.sweep:
        sw = summary["sweep_max_cornerblk_per_layer"]; nc = summary["sweep_ncorner_per_layer"]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(range(len(sw)), sw, "o-", color="C3"); ax1.set_xlabel("layer"); ax1.set_ylabel("max corner conc.", color="C3")
        ax1.axhline(1.0, color="grey", ls="--", lw=1)
        ax2 = ax1.twinx(); ax2.plot(range(len(nc)), nc, "s--", color="C0"); ax2.set_ylabel("# corner heads", color="C0")
        ax1.set_title(f"{tag} corner sink across layers")
        fig.tight_layout(); fig.savefig(outdir / "fig_block_sweep.png", dpi=130); plt.close(fig)
    print(f"[plots] saved to {outdir}", flush=True)


if __name__ == "__main__":
    main()
