#!/usr/bin/env python
"""Audit B11 gaze/pose auxiliary channels against normalized RGB clips.

This script does not initialize the V-JEPA model. It builds the same online
gaze+pose maps used by ``binary_input_adapter_gaze_pose_matrix`` and reports
shape/range statistics for RGB, gaze, pose, and the pose patch only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
VJEPA_ROOT = REPO_ROOT / "vjepa2"
for path in (REPO_ROOT, VJEPA_ROOT):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate, patch_metadata_dataloader  # noqa: E402
from app.hdepic_lora_action_anticipation.pose_map_builder import GazePoseInputMapBuilder  # noqa: E402
from evals.action_anticipation_frozen.dataloader import filter_annotations, init_data  # noqa: E402


def _save_adapter_input_viz(
    clips: torch.Tensor,
    aux: torch.Tensor,
    out_dir: Path,
    batch_idx: int,
    sample_idx: int = 0,
    pose_patch_hw: tuple[int, int] = (128, 9),
) -> None:
    """Visualize the exact 5-channel tensor fed to BinaryMapInputAdapter: cat(RGB, aux)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("[b11-pose-audit] matplotlib not available; skipping --save-viz-dir")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t_mid = clips.shape[2] // 2
    ph, pw = pose_patch_hw

    ch_names = ["ch0 R", "ch1 G", "ch2 B", "ch3 gaze", "ch4 pose"]
    panels = []
    for c in range(3):
        panels.append(clips[sample_idx, c, t_mid].detach().float().cpu().numpy())
    panels.append(aux[sample_idx, 0, t_mid].detach().float().cpu().numpy())
    panels.append(aux[sample_idx, 1, t_mid].detach().float().cpu().numpy())

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.ravel()
    for idx, (ax, name, data) in enumerate(zip(axes_flat[:5], ch_names, panels)):
        if idx < 3:
            vmin, vmax = float(data.min()), float(data.max())
            im = ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}\n384×384 (RGB norm clip)")
        elif idx == 3:
            im = ax.imshow(data, cmap="magma", vmin=0.0, vmax=max(1.0, float(data.max())))
            ax.set_title(f"{name}\n384×384 gaze map")
        else:
            im = ax.imshow(data, cmap="viridis", vmin=float(data.min()), vmax=float(data.max()))
            ax.add_patch(Rectangle((-0.5, -0.5), pw, ph, fill=False, edgecolor="red", linewidth=1.5))
            ax.set_title(f"{name}\n384×384 (red box = {ph}×{pw} pose patch)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    axes_flat[5].axis("off")
    axes_flat[5].text(
        0.0,
        0.95,
        "Adapter input: torch.cat([rgb, aux], dim=1)\n"
        f"shape = [5, T={clips.shape[2]}, H={clips.shape[3]}, W={clips.shape[4]}]\n"
        "aux = cat(gaze, pose) → ch3 + ch4\n"
        f"pose: {ph}×{pw} matrix pasted top-left; rest of 384×384 is zero",
        va="top",
        fontsize=10,
        family="monospace",
    )
    fig.suptitle(f"Adapter input (batch={batch_idx}, sample={sample_idx}, frame t={t_mid})")
    fig.tight_layout()
    out_path = out_dir / f"batch{batch_idx:02d}_sample{sample_idx}_adapter_input.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[b11-pose-audit] wrote {out_path}")


def _save_aux_viz(
    clips: torch.Tensor,
    gaze: torch.Tensor,
    pose_patch: torch.Tensor,
    out_dir: Path,
    batch_idx: int,
    sample_idx: int = 0,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[b11-pose-audit] matplotlib not available; skipping --save-viz-dir")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    # clips: (B, C, T, H, W)
    t_mid = clips.shape[2] // 2
    n_rgb = min(3, clips.shape[1])
    rgb = clips[sample_idx, :n_rgb, t_mid].detach().float().cpu()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    rgb_img = rgb.permute(1, 2, 0).numpy()
    if rgb_img.shape[-1] == 1:
        rgb_img = rgb_img[..., 0]

    gaze_map = gaze[sample_idx, 0, t_mid].detach().float().cpu().numpy()
    pose_map = pose_patch[sample_idx, 0, t_mid].detach().float().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB (mid frame, norm)")
    axes[0].axis("off")
    im1 = axes[1].imshow(gaze_map, cmap="magma", aspect="auto")
    axes[1].set_title("gaze channel")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(pose_map, cmap="viridis", aspect="auto")
    axes[2].set_title(f"pose patch ({pose_map.shape[0]}x{pose_map.shape[1]})")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle(f"batch={batch_idx} sample={sample_idx}")
    fig.tight_layout()
    out_path = out_dir / f"batch{batch_idx:02d}_sample{sample_idx}_aux_channels.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[b11-pose-audit] wrote {out_path}")


def _stats(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().float().cpu()
    finite = torch.isfinite(y)
    finite_y = y[finite]
    if finite_y.numel() == 0:
        return {
            "shape": list(y.shape),
            "finite": f"0/{y.numel()}",
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "absmax": None,
            "nonzero_frac": None,
        }
    return {
        "shape": list(y.shape),
        "finite": f"{int(finite.sum().item())}/{y.numel()}",
        "min": float(finite_y.min().item()),
        "max": float(finite_y.max().item()),
        "mean": float(finite_y.mean().item()),
        "std": float(finite_y.std(unbiased=False).item()),
        "absmax": float(finite_y.abs().max().item()),
        "nonzero_frac": float((finite_y != 0).float().mean().item()),
    }


def _make_gaze_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    args_eval = dict(cfg)
    data_cfg = dict(args_eval["experiment"]["data"])
    lora_cfg = dict(args_eval["experiment"].get("lora", {}))
    gaze_cfg = dict(lora_cfg.get("gaze", {}))
    gaze_cfg.setdefault("crop_size", data_cfg.get("resolution", 384))
    gaze_cfg.setdefault("frames_per_clip", data_cfg.get("frames_per_clip", 32))
    pretrain = args_eval.get("model_kwargs", {}).get("pretrain_kwargs", {})
    gaze_cfg.setdefault("patch_size", pretrain.get("encoder", {}).get("patch_size", 16))
    gaze_cfg.setdefault("tubelet_size", pretrain.get("encoder", {}).get("tubelet_size", 2))
    if str(gaze_cfg.get("mode", "")).lower() == "binary_input_adapter_gaze_pose_matrix":
        gaze_cfg.setdefault("input_adapter", {})
        gaze_cfg["input_adapter"].setdefault("in_channels", 5)
        gaze_cfg.setdefault("pose", {})
        gaze_cfg["pose"].setdefault("enabled", True)
        gaze_cfg["pose"].setdefault("interframe_k_max", 128)
        gaze_cfg.setdefault("pose_map", {})
        gaze_cfg["pose_map"].setdefault("patch_height", 128)
        gaze_cfg["pose_map"].setdefault("patch_width", 9)
        gaze_cfg["pose_map"].setdefault("layout", "topleft")
        gaze_cfg["pose_map"].setdefault("normalize", "none")
    return gaze_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training/eval yaml using binary_input_adapter_gaze_pose_matrix")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-subset-path", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--save-viz-dir",
        type=Path,
        default=None,
        help="Save RGB/gaze/pose PNG panels for the first sample of each batch",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp = cfg["experiment"]
    data_cfg = dict(exp["data"])
    opt_cfg = dict(exp["optimization"])
    batch_size = int(args.batch_size or opt_cfg.get("batch_size", 1))

    annotations = filter_annotations(
        data_cfg["dataset"],
        data_cfg["base_path"],
        data_cfg["dataset_train"],
        data_cfg["dataset_val"],
        file_format=data_cfg.get("file_format", 1),
    )
    annotations_path = annotations["train" if args.split == "train" else "val"]
    anticipation_time = (
        data_cfg.get("train_anticipation_time_sec")
        if args.split == "train"
        else data_cfg.get("anticipation_time_sec")
    )
    anticipation_point = (
        data_cfg.get("train_anticipation_point")
        if args.split == "train"
        else data_cfg.get("val_anticipation_point", [0.0, 0.0])
    )

    patch_metadata_dataloader(
        emit_binary_map=False,
        debug_subset_path=args.debug_subset_path.strip() or None,
    )
    _, loader, _ = init_data(
        dataset=data_cfg["dataset"],
        training=args.split == "train",
        base_path=data_cfg["base_path"],
        annotations_path=annotations_path,
        batch_size=batch_size,
        frames_per_clip=data_cfg["frames_per_clip"],
        fps=data_cfg["frames_per_second"],
        anticipation_time_sec=anticipation_time,
        anticipation_point=anticipation_point,
        random_resize_scale=data_cfg.get("random_resize_scale", (0.08, 1.0)),
        reprob=data_cfg.get("reprob", 0.0),
        auto_augment=data_cfg.get("auto_augment", False),
        motion_shift=data_cfg.get("motion_shift", False),
        crop_size=data_cfg.get("resolution", 384),
        world_size=1,
        rank=0,
        num_workers=args.num_workers,
        pin_mem=False,
        persistent_workers=False,
    )

    gaze_cfg = _make_gaze_cfg(cfg)
    builder = GazePoseInputMapBuilder(gaze_cfg, gate=GazeTokenGate(gaze_cfg))
    pose_h = int(builder.pose_builder.patch_height)
    pose_w = int(builder.pose_builder.patch_width)

    rows = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.batches:
            break
        clips = batch[0]
        metadata = batch[3]
        aux = builder.build(clips, metadata)
        gaze = aux[:, 0:1]
        pose = aux[:, 1:2]
        pose_patch = pose[:, :, :, : min(pose_h, pose.shape[-2]), : min(pose_w, pose.shape[-1])]
        rows.append(
            {
                "batch": batch_idx,
                "rgb": _stats(clips),
                "gaze": _stats(gaze),
                "pose_full_canvas": _stats(pose),
                "pose_patch": _stats(pose_patch),
                "pose_patch_hw": [pose_h, pose_w],
                "pose_feature_dim": int(builder.pose_builder.pose_loader.input_dim),
                "interframe_k_max": int(builder.pose_builder.k_max),
                "pose_normalize": builder.pose_builder.normalize,
            }
        )
        if args.save_viz_dir is not None:
            _save_adapter_input_viz(
                clips,
                aux,
                args.save_viz_dir,
                batch_idx,
                pose_patch_hw=(pose_h, pose_w),
            )

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    for row in rows:
        print(f"batch={row['batch']} pose_patch_hw={row['pose_patch_hw']} feature_dim={row['pose_feature_dim']} k={row['interframe_k_max']} normalize={row['pose_normalize']}")
        for key in ("rgb", "gaze", "pose_full_canvas", "pose_patch"):
            stats = row[key]
            print(
                "  {key:16s} shape={shape} finite={finite} min={min} max={max} mean={mean} std={std} absmax={absmax} nonzero={nonzero_frac}".format(
                    key=key,
                    **stats,
                )
            )


if __name__ == "__main__":
    main()
