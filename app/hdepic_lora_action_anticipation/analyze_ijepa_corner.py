#!/usr/bin/env python3
# [ATTN-CORNER-SINK] I-JEPA control: does the image-JEPA encoder show the same
# per-head last-block corner sink as V-JEPA2? Pure transformers/PIL/numpy; runs
# in a transformers-capable env (jepa-3dgs can3tok). Standalone, deletable.
"""Per-head last-block received-attention corner analysis for official I-JEPA.

Loads facebook/ijepa_vith14_1k (ViT-H/14: 32 layers, 16 heads, 224px/patch14 ->
16x16=256 patch tokens, NO cls) with eager attention, runs a folder of images,
and for a chosen block computes per-head received attention = column-sum of the
softmax attention over queries -> reshape 16x16. Reports corner concentration +
a block sweep. Analogue of analyze_encoder_head_attn_corners.py (V-JEPA2), but
attention comes straight from output_attentions (I-JEPA is SDPA/eager, no RoPE).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HOME", "/scratch/yh6416/.huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

IMG, PATCH = 224, 14
GRID = IMG // PATCH                       # 16
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


def load_img(path, device):
    im = Image.open(path).convert("RGB")
    if im.size != (IMG, IMG):
        im = im.resize((IMG, IMG), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(255.)
    return ((x.unsqueeze(0) - IMAGENET_MEAN) / IMAGENET_STD).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default="facebook/ijepa_vith14_1k")
    ap.add_argument("--block", type=int, default=-1, help="layer to read (-1=last)")
    ap.add_argument("--sweep", action="store_true", help="also report corner conc across all layers")
    ap.add_argument("--n-images", type=int, default=0)
    ap.add_argument("--hflip", action="store_true", help="horizontally mirror input (position-vs-content test)")
    ap.add_argument("--dump-per-sample", action="store_true", help="save per-image spatial maps (ll-style stability)")
    args = ap.parse_args()

    from transformers import AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model_id, attn_implementation="eager").eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    block = args.block if args.block >= 0 else n_layers + args.block
    print(f"[ijepa] layers={n_layers} heads={n_heads} grid={GRID} block={block}", flush=True)

    files = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg"))
                   + glob.glob(os.path.join(args.img_dir, "*.png")))
    if args.n_images > 0:
        files = files[: args.n_images]
    print(f"[data] {len(files)} images from {args.img_dir}", flush=True)

    acc = np.zeros((n_heads, GRID, GRID), dtype=np.float64)
    sweep_acc = np.zeros((n_layers, n_heads), dtype=np.float64) if args.sweep else None
    per_sample = []
    n_used = 0
    for f in files:
        x = load_img(f, device)
        if args.hflip:
            x = torch.flip(x, dims=[-1])
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = model(pixel_values=x, output_attentions=True)
        attns = out.attentions            # tuple[n_layers] each (B, heads, N, N)
        N = attns[0].shape[-1]
        # I-JEPA has no cls; N should be GRID*GRID
        off = N - GRID * GRID             # tolerate a leading token if present
        for li, a in enumerate(attns):
            recv = a[0].float().sum(dim=1)          # sum over queries -> (heads, N)
            recv = recv[:, off:]                    # keep patch tokens
            recv = recv / (recv.sum(dim=1, keepdim=True) + 1e-12)
            maps = recv.reshape(n_heads, GRID, GRID).cpu().numpy()
            if args.sweep:
                for h in range(n_heads):
                    sweep_acc[li, h] += corner_metrics(maps[h])["cornerblk_over_uniform"]
            if li == block:
                acc += maps
                if args.dump_per_sample:
                    per_sample.append(maps.astype(np.float32))
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
        np.save(outdir / "per_sample_spatial.npy", np.stack(per_sample))

    summary = {
        "model": args.model_id, "grid": GRID, "block": block, "n_heads": n_heads,
        "n_images": n_used,
        "per_head_corner": {str(h): {k: round(v, 3) for k, v in metrics[h].items()} for h in range(n_heads)},
        "heads_ranked_by_cornerblk": order,
        "corner_sorted": [round(metrics[h]["cornerblk_over_uniform"], 2) for h in order],
        "n_corner_heads": sum(1 for h in range(n_heads) if metrics[h]["cornerblk_over_uniform"] > 1.6),
    }
    if args.sweep:
        sw = sweep_acc / n_used
        summary["sweep_max_cornerblk_per_layer"] = [round(float(sw[li].max()), 2) for li in range(n_layers)]
        summary["sweep_ncorner_per_layer"] = [int((sw[li] > 1.6).sum()) for li in range(n_layers)]
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("grid", "block", "n_images", "n_corner_heads",
          "corner_sorted") if k in summary}, indent=2), flush=True)
    if args.sweep:
        print("sweep max cornerblk/layer:", summary["sweep_max_cornerblk_per_layer"], flush=True)

    _figs(outdir, spatial, metrics, order, args, summary)


def _figs(outdir, spatial, metrics, order, args, summary):
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
            ax.set_title(f"h{h} corner×{cb:.1f} c/e{metrics[h]['center_over_edge']:.1f}",
                         fontsize=8, color=("red" if cb > 1.6 else "black"))
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"I-JEPA {args.model_id.split('/')[-1]}  block {summary['block']} per-head received attn  ({summary['n_images']} imgs)")
    fig.tight_layout()
    fig.savefig(outdir / "fig_ijepa_per_head.png", dpi=120)
    plt.close(fig)
    if args.sweep:
        sw = summary["sweep_max_cornerblk_per_layer"]; nc = summary["sweep_ncorner_per_layer"]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(range(len(sw)), sw, "o-", color="C3"); ax1.set_xlabel("layer"); ax1.set_ylabel("max corner conc.", color="C3")
        ax1.axhline(1.0, color="grey", ls="--", lw=1)
        ax2 = ax1.twinx(); ax2.plot(range(len(nc)), nc, "s--", color="C0"); ax2.set_ylabel("# corner heads", color="C0")
        ax1.set_title("I-JEPA corner sink across layers")
        fig.tight_layout(); fig.savefig(outdir / "fig_ijepa_block_sweep.png", dpi=130); plt.close(fig)
    print(f"[plots] saved to {outdir}", flush=True)


if __name__ == "__main__":
    main()
