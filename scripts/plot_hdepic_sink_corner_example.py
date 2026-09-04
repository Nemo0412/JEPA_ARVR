#!/usr/bin/env python3
"""One HD-EPIC clip: RGB mid-frame + L23 key-attention (h3 sink vs h8 content) + corner numbers."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path("/home/ll5914/Jepa_yifan/JEPA_ARVR")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
os.environ.setdefault(
    "TRI_MODAL_FRAME_CACHE",
    "/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1",
)

_spec = importlib.util.spec_from_file_location(
    "mc", PROJECT_ROOT / "scripts/analyze_encoder_last_layer_multiclip.py"
)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)


def main():
    out_dir = Path("/home/ll5914/Jepa_yifan/encoder_L23_hdepic_sink_example")
    out_dir.mkdir(parents=True, exist_ok=True)

    class A:
        pass

    args = A()
    args.n_sample = 1
    args.chunk = 128
    args.frames = 32
    args.fps = 8
    args.img_size = 256
    args.anticipation = 1.0
    args.clips_dir = ""
    args.val_csv = (
        "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/"
        "clip_split/HD_EPIC_val_vjepa.csv"
    )
    args.video_root = "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
    args.vjepa_ckpt = "/scratch/ll5914/models/vjepa2/vitl.pt"
    args.video_enc_lora = (
        "/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/"
        "action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/"
        "encoder_lora_best.pt"
    )
    args.video_pred_lora = (
        "/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/"
        "action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/"
        "predictor_lora_best.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = mc.load_diverse_clips(args)
    clip, meta = samples[0]
    vid = meta["video_id"]
    print("video", vid, "clip", tuple(clip.shape))

    model = mc._fastgen.load_video_model(args, device)
    core = model.base_model if hasattr(model, "base_model") else model
    encoder = core.encoder
    grid = int(getattr(core, "grid_size", 16))
    last = len(encoder.blocks) - 1

    cap = mc.LastLayerCapture(encoder, last, args.chunk)
    clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
    ant = torch.full((1,), float(args.anticipation), device=device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _ = model(clip_b, ant)
    cap.restore()
    imp = cap.imp  # [H, N] already max-normalized per head
    assert imp is not None

    # Use RAW column-sum (recompute without max-norm) for numbers
    # Re-run capture... actually compute_head_key_importance max-norms.
    # Recover relative mass from max-normed map is OK for visualization;
    # also print corner vs center on the spatial mean (unnormalized within spatial).

    def spatial_of(h: int) -> np.ndarray:
        f = mc.head_features(imp[h], grid)
        # head_features renormalizes spatial to sum=1 for metrics; use raw reshape
        gp = grid * grid
        t_slots = imp[h].size // gp
        vid_map = imp[h].reshape(t_slots, grid, grid)
        # mean over time, then max-norm for display
        sp = vid_map.mean(axis=0)
        return sp

    sp3 = spatial_of(3)
    sp8 = spatial_of(8)
    sp3n = sp3 / (sp3.max() + 1e-12)
    sp8n = sp8 / (sp8.max() + 1e-12)

    # RGB mid frame of the tensor (denorm ImageNet)
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    t_mid = clip.shape[1] // 2
    rgb = clip[:, t_mid].cpu().numpy() * std + mean
    rgb = np.clip(rgb.transpose(1, 2, 0), 0, 1)

    # Also load the cached mid jpg if present for sharper view
    jpg = (
        PROJECT_ROOT
        / "experiments/fastgen_head_heatmaps/encoder_L23_multiclip/input_frames"
        / "00_P01_20240202-110250_mid.jpg"
    )
    if jpg.is_file() and "110250" in vid:
        rgb_show = np.asarray(Image.open(jpg).convert("RGB")) / 255.0
    else:
        rgb_show = rgb

    # Corner / center mass on probability-normalized spatial
    s3 = sp3 / (sp3.sum() + 1e-12)
    corners = [s3[0, 0], s3[0, -1], s3[-1, 0], s3[-1, -1]]
    center = s3[grid // 4 : 3 * grid // 4, grid // 4 : 3 * grid // 4].sum()
    corner_sum = float(sum(corners))
    # random baseline for 4 cells
    print(
        f"h3: 4-corner mass={corner_sum:.4f} (uniform 4/256={4/256:.4f}) "
        f"center4x4-area mass={center:.4f} (uniform≈0.25)"
    )
    print("h3 corner cells (TL,TR,BL,BR):", [float(x) for x in corners])

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(rgb_show)
    axes[0, 0].set_title(f"HD-EPIC RGB\n{vid}", fontsize=10)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    # mark 4 corners on RGB
    h, w = rgb_show.shape[:2]
    for yy, xx in [(2, 2), (2, w - 3), (h - 3, 2), (h - 3, w - 3)]:
        axes[0, 0].plot(xx, yy, "r*", markersize=14)

    im1 = axes[0, 1].imshow(sp3n, cmap="viridis", vmin=0, vmax=1)
    axes[0, 1].set_title("L23 stable head h3\nkey attention (time-mean)", fontsize=10)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    for r, c in [(0, 0), (0, 15), (15, 0), (15, 15)]:
        axes[0, 1].add_patch(
            plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="r", lw=1.5)
        )
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # overlay
    axes[0, 2].imshow(rgb_show)
    heat = np.array(
        Image.fromarray((sp3n * 255).astype(np.uint8)).resize(
            (rgb_show.shape[1], rgb_show.shape[0]), Image.NEAREST
        )
    ) / 255.0
    axes[0, 2].imshow(heat, cmap="viridis", alpha=0.55, vmin=0, vmax=1)
    axes[0, 2].set_title("RGB + h3 overlay\n(sink = 4 corners)", fontsize=10)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    axes[1, 0].imshow(rgb_show)
    axes[1, 0].set_title("same frame", fontsize=10)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    im2 = axes[1, 1].imshow(sp8n, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title("L23 content head h8\n(more center / object)", fontsize=10)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].imshow(rgb_show)
    heat8 = np.array(
        Image.fromarray((sp8n * 255).astype(np.uint8)).resize(
            (rgb_show.shape[1], rgb_show.shape[0]), Image.NEAREST
        )
    ) / 255.0
    axes[1, 2].imshow(heat8, cmap="viridis", alpha=0.55, vmin=0, vmax=1)
    axes[1, 2].set_title("RGB + h8 overlay", fontsize=10)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    fig.suptitle(
        f"Why sink is at corners (HD-EPIC): h3 dumps on 4 grid corners "
        f"(mass={corner_sum:.3f} vs uniform {4/256:.3f})",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = out_dir / "hdepic_sink_corner_example.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Print 16x16 numeric grid for h3 (rounded)
    np.savetxt(out_dir / "h3_spatial_mass.txt", s3, fmt="%.4f")
    # also a sparse print of only corners + center 4
    with open(out_dir / "h3_corner_numbers.txt", "w") as f:
        f.write(f"video={vid}\n")
        f.write(f"TL={s3[0,0]:.6f} TR={s3[0,-1]:.6f} BL={s3[-1,0]:.6f} BR={s3[-1,-1]:.6f}\n")
        f.write(f"sum4corners={corner_sum:.6f}  uniform4={4/256:.6f}  ratio={corner_sum/(4/256):.2f}x\n")
        f.write(f"center_half_mass={center:.6f} uniform≈0.25\n")
        # top-5 patches
        flat = s3.ravel()
        top = np.argsort(-flat)[:8]
        f.write("top8 patches (r,c,mass):\n")
        for idx in top:
            r, c = divmod(int(idx), grid)
            f.write(f"  ({r:2d},{c:2d}) {flat[idx]:.6f}\n")
    print("wrote", out_png)
    print((out_dir / "h3_corner_numbers.txt").read_text())


if __name__ == "__main__":
    main()
