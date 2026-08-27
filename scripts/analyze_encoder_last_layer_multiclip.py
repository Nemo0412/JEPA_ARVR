#!/usr/bin/env python3
"""Multi-clip analysis of last encoder layer attention patterns (video-only)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

_spec = importlib.util.spec_from_file_location(
    "analyze_fastgen_heads", PROJECT_ROOT / "scripts" / "analyze_fastgen_heads.py"
)
_fastgen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fastgen)

_spec2 = importlib.util.spec_from_file_location(
    "analyze_fastgen_head_heatmaps", PROJECT_ROOT / "scripts" / "analyze_fastgen_head_heatmaps.py"
)
_hm = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_hm)

logger = logging.getLogger("enc_last_layer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_diverse_clips(args):
    """One clip per unique video_id (up to n_sample)."""
    df = pd.read_csv(args.val_csv)
    seen = set()
    rows = []
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        if vid in seen:
            continue
        seen.add(vid)
        rows.append(row)
        if len(rows) >= args.n_sample:
            break
    # Temporarily override n_sample for load_val_clips by filtering csv
    import tempfile

    tmp = Path(tempfile.gettempdir()) / f"fastgen_diverse_{os.getpid()}.csv"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    old = args.val_csv
    old_n = args.n_sample
    args.val_csv = str(tmp)
    args.n_sample = len(rows)
    samples = _fastgen.load_val_clips(args)
    args.val_csv = old
    args.n_sample = old_n
    tmp.unlink(missing_ok=True)
    return samples


class LastLayerCapture:
    def __init__(self, encoder, layer: int, chunk: int):
        self.layer = layer
        self.chunk = chunk
        self.imp: np.ndarray | None = None
        self._orig = None
        block = encoder.blocks[layer]
        orig = block.attn.forward

        def wrapped(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            q, k, v = _fastgen._rope_qk(block.attn, x, mask, T, H_patches, W_patches)
            self.imp = _hm.compute_head_key_importance(q, k, block.attn.scale, chunk=self.chunk)
            with torch.backends.cuda.sdp_kernel():
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=block.attn.is_causal, attn_mask=attn_mask
                )
            y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2])
            y = block.attn.proj(y)
            y = block.attn.proj_drop(y)
            return y

        block.attn.forward = wrapped
        self._orig = (block.attn, orig)

    def restore(self):
        mod, orig = self._orig
        mod.forward = orig


def head_features(imp_h: np.ndarray, grid: int) -> dict:
    """imp_h: [N] for one head."""
    gp = grid * grid
    t_slots = imp_h.size // gp
    vid = imp_h.reshape(t_slots, grid, grid)
    spatial = vid.mean(axis=0)
    temporal = vid.sum(axis=(1, 2))
    temporal = temporal / (temporal.sum() + 1e-12)
    s = spatial / (spatial.sum() + 1e-12)
    yy, xx = np.mgrid[0:grid, 0:grid]
    cy = (s * yy).sum()
    cx = (s * xx).sum()
    edge = 2
    edge_m = (
        s[:edge, :].sum()
        + s[-edge:, :].sum()
        + s[:, :edge].sum()
        + s[:, -edge:].sum()
        - (s[:edge, :edge].sum() + s[:edge, -edge:].sum() + s[-edge:, :edge].sum() + s[-edge:, -edge:].sum())
    )
    center = s[grid // 4 : 3 * grid // 4, grid // 4 : 3 * grid // 4].sum()
    recent = temporal[int(0.7 * t_slots) :].sum()
    ent = float(-(s[s > 1e-12] * np.log(s[s > 1e-12])).sum())
    peak = np.unravel_index(int(spatial.argmax()), spatial.shape)
    return {
        "spatial": spatial.astype(np.float32),
        "temporal": temporal.astype(np.float32),
        "center_mass": float(center),
        "edge_mass": float(edge_m),
        "recent_mass": float(recent),
        "entropy": ent,
        "peak_yx": (int(peak[0]), int(peak[1])),
        "peak_val": float(spatial.max()),
    }


def classify_head(feats: dict) -> str:
    if feats["recent_mass"] > 0.45 and feats["peak_val"] > 0.08:
        return "recency+peak"
    if feats["recent_mass"] > 0.40:
        return "recency"
    if feats["edge_mass"] > 0.35:
        return "edge"
    if feats["center_mass"] > 0.35:
        return "center"
    if feats["entropy"] > 4.5:
        return "diffuse"
    if feats["peak_val"] > 0.12:
        return "sparse_peak"
    return "mixed"


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_diverse_clips(args)
    logger.info("loaded %d clips from %d unique videos", len(samples), len(samples))

    model = _fastgen.load_video_model(args, device)
    encoder = model.base_model.encoder if hasattr(model, "base_model") else model.encoder
    n_layers = len(encoder.blocks)
    last = n_layers - 1
    grid = int(getattr(model.base_model if hasattr(model, "base_model") else model, "grid_size", 16))

    per_sample = []
    head_maps = {h: [] for h in range(16)}

    for si, (clip, meta) in enumerate(samples):
        vid = meta["video_id"]
        cap = LastLayerCapture(encoder, last, args.chunk)
        clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
        ant = torch.full((1,), float(args.anticipation), device=device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model(clip_b, ant)
        cap.restore()
        if out is None or cap.imp is None:
            logger.warning("skip %s", vid)
            continue
        heads = {}
        for h in range(cap.imp.shape[0]):
            f = head_features(cap.imp[h], grid)
            f["label"] = classify_head(f)
            heads[str(h)] = {k: v for k, v in f.items() if k != "spatial" and k != "temporal"}
            heads[str(h)]["spatial"] = f["spatial"].tolist()
            heads[str(h)]["temporal"] = f["temporal"].tolist()
            head_maps[h].append(f["spatial"])
        per_sample.append({"idx": si, "video_id": vid, "narration": meta.get("narration", ""), "heads": heads})
        logger.info("[%d/%d] %s", si + 1, len(samples), vid)

    # Cross-video stats per head
    head_stats = []
    for h in range(16):
        stack = np.stack(head_maps[h], axis=0)  # [S,16,16]
        mean_map = stack.mean(axis=0)
        std_map = stack.std(axis=0)
        # avg pairwise corr
        corrs = []
        for i in range(len(stack)):
            for j in range(i + 1, len(stack)):
                a = stack[i].ravel()
                b = stack[j].ravel()
                if a.std() > 1e-8 and b.std() > 1e-8:
                    corrs.append(float(np.corrcoef(a, b)[0, 1]))
        labels = [per_sample[s]["heads"][str(h)]["label"] for s in range(len(per_sample))]
        from collections import Counter

        lc = Counter(labels)
        head_stats.append(
            {
                "head": h,
                "mean_corr": float(np.mean(corrs)) if corrs else 0.0,
                "std_corr": float(np.std(corrs)) if corrs else 0.0,
                "label_mode": lc.most_common(1)[0][0],
                "label_counts": dict(lc),
                "mean_recent": float(np.mean([per_sample[s]["heads"][str(h)]["recent_mass"] for s in range(len(per_sample))])),
                "mean_center": float(np.mean([per_sample[s]["heads"][str(h)]["center_mass"] for s in range(len(per_sample))])),
                "mean_edge": float(np.mean([per_sample[s]["heads"][str(h)]["edge_mass"] for s in range(len(per_sample))])),
                "mean_entropy": float(np.mean([per_sample[s]["heads"][str(h)]["entropy"] for s in range(len(per_sample))])),
            }
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "layer": last,
        "n_clips": len(per_sample),
        "head_stats": head_stats,
        "samples": [{k: v for k, v in s.items() if k != "heads"} for s in per_sample],
    }
    (out_dir / "last_encoder_patterns.json").write_text(json.dumps(report, indent=2))

    # Fig: mean spatial maps 4x4
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for h in range(16):
        r, c = divmod(h, 4)
        mean_map = np.mean(np.stack(head_maps[h]), axis=0)
        mx = mean_map.max() or 1
        im = axes[r, c].imshow(mean_map / mx, cmap="viridis", vmin=0, vmax=1)
        st = head_stats[h]
        axes[r, c].set_title(
            f"h{h} {st['label_mode']}\nr={st['mean_corr']:.2f} rec={st['mean_recent']:.2f}",
            fontsize=8,
        )
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    fig.suptitle(f"Encoder L{last} mean spatial pattern ({len(per_sample)} videos)", fontsize=12)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "L23_mean_spatial_16heads.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig: head consistency bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    hs = [s["head"] for s in head_stats]
    corrs = [s["mean_corr"] for s in head_stats]
    colors = ["#4CAF50" if c > 0.5 else "#FF9800" if c > 0.3 else "#F44336" for c in corrs]
    ax.bar(hs, corrs, color=colors)
    ax.set_xlabel("Head")
    ax.set_ylabel("Mean cross-video spatial correlation")
    ax.set_title(f"Encoder L{last}: cross-video consistency ({len(per_sample)} clips)")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "L23_cross_video_corr.png", dpi=150)
    plt.close(fig)

    # Print summary
    print("\n=== Encoder L%d patterns (%d videos) ===" % (last, len(per_sample)))
    for st in head_stats:
        print(
            f"  h{st['head']:2d}: mode={st['label_mode']:14s} corr={st['mean_corr']:.3f} "
            f"recent={st['mean_recent']:.2f} center={st['mean_center']:.2f} edge={st['mean_edge']:.2f}"
        )
    consistent = [st["head"] for st in head_stats if st["mean_corr"] > 0.5]
    print(f"\nConsistent heads (corr>0.5): {consistent}")
    print(f"Wrote {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sample", type=int, default=20)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--anticipation", type=float, default=1.0)
    p.add_argument("--val-csv", default="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv")
    p.add_argument("--video-root", default="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos")
    p.add_argument("--vjepa-ckpt", default="/scratch/ll5914/models/vjepa2/vitl.pt")
    p.add_argument(
        "--video-enc-lora",
        default="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/encoder_lora_best.pt",
    )
    p.add_argument(
        "--video-pred-lora",
        default="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt",
    )
    p.add_argument("--out", default=str(PROJECT_ROOT / "experiments/fastgen_head_heatmaps/encoder_L23_multiclip"))
    args = p.parse_args()
    os.environ.setdefault("TRI_MODAL_FRAME_CACHE", "/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1")
    run(args)


if __name__ == "__main__":
    main()
