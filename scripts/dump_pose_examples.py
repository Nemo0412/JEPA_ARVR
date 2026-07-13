#!/usr/bin/env python
"""Dump raw inter-frame pose matrix examples for inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "vjepa2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate, patch_metadata_dataloader  # noqa: E402
from app.hdepic_lora_action_anticipation.pose_map_builder import GazePoseInputMapBuilder  # noqa: E402
from app.hdepic_lora_action_anticipation.pose_slam import SlamPoseLoader  # noqa: E402
from evals.action_anticipation_frozen.dataloader import filter_annotations, init_data  # noqa: E402


def _make_gaze_cfg(cfg: dict) -> dict:
    args_eval = dict(cfg)
    data_cfg = dict(args_eval["experiment"]["data"])
    lora_cfg = dict(args_eval["experiment"].get("lora", {}))
    gaze_cfg = dict(lora_cfg.get("gaze", {}))
    gaze_cfg.setdefault("crop_size", data_cfg.get("resolution", 384))
    gaze_cfg.setdefault("frames_per_clip", data_cfg.get("frames_per_clip", 32))
    pretrain = args_eval.get("model_kwargs", {}).get("pretrain_kwargs", {})
    gaze_cfg.setdefault("patch_size", pretrain.get("encoder", {}).get("patch_size", 16))
    gaze_cfg.setdefault("tubelet_size", pretrain.get("encoder", {}).get("tubelet_size", 2))
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


def _summarize_matrix(mat: np.ndarray) -> dict:
    nz_rows = int((np.abs(mat).sum(axis=1) > 1e-8).sum())
    nz_cols = [int((np.abs(mat[:, c]) > 1e-8).sum()) for c in range(mat.shape[1])]
    return {
        "shape": list(mat.shape),
        "nonzero_rows": nz_rows,
        "nonzero_per_col": nz_cols,
        "col_names": ["dtx", "dty", "dtz", "r00", "r01", "r02", "r10", "r11", "r12"],
        "row0": mat[0].tolist(),
        "row1": mat[1].tolist() if mat.shape[0] > 1 else None,
        "last_nz_row_idx": int(nz_rows - 1) if nz_rows else None,
        "last_nz_row": mat[nz_rows - 1].tolist() if nz_rows else None,
        "min": float(mat.min()),
        "max": float(mat.max()),
        "mean": float(mat.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--frame-idx", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp = cfg["experiment"]
    data_cfg = dict(exp["data"])
    opt_cfg = dict(exp["optimization"])
    batch_size = int(opt_cfg.get("batch_size", 2))

    annotations = filter_annotations(
        data_cfg["dataset"],
        data_cfg["base_path"],
        data_cfg["dataset_train"],
        data_cfg["dataset_val"],
        file_format=data_cfg.get("file_format", 1),
    )
    patch_metadata_dataloader(emit_binary_map=False)
    _, loader, _ = init_data(
        dataset=data_cfg["dataset"],
        training=False,
        base_path=data_cfg["base_path"],
        annotations_path=annotations["val"],
        batch_size=batch_size,
        frames_per_clip=data_cfg["frames_per_clip"],
        fps=data_cfg["frames_per_second"],
        anticipation_time_sec=data_cfg.get("anticipation_time_sec"),
        anticipation_point=data_cfg.get("val_anticipation_point", [0.0, 0.0]),
        random_resize_scale=data_cfg.get("random_resize_scale", (0.08, 1.0)),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=data_cfg.get("resolution", 384),
        world_size=1,
        rank=0,
        num_workers=0,
        pin_mem=False,
        persistent_workers=False,
    )

    gaze_cfg = _make_gaze_cfg(cfg)
    builder = GazePoseInputMapBuilder(gaze_cfg)
    loader_pose = SlamPoseLoader(gaze_cfg["pose"], gate=GazeTokenGate(gaze_cfg))

    for batch_idx, batch in enumerate(loader):
        if batch_idx != args.batch_idx:
            continue
        metadata = batch[3]
        meta = metadata[args.sample_idx]
        pose_mats = loader_pose.query_interframe_matrices(meta, builder.pose_builder.k_max)
        frame_idx = min(args.frame_idx, pose_mats.shape[0] - 1) if pose_mats is not None else args.frame_idx

        report = {
            "video_id": meta.get("video_id"),
            "frame_indices_head": meta.get("frame_indices", [])[:5] if hasattr(meta.get("frame_indices", []), "__getitem__") else "n/a",
            "k_max": builder.pose_builder.k_max,
            "feature_dim": builder.pose_builder.pose_loader.input_dim,
            "feature_set": builder.pose_builder.pose_loader.feature_set,
            "layout": "topleft -> paste [128,9] into canvas [0:128, 0:9] of [384,384]",
            "dim_explanation": {
                "rows_128": "max SLAM samples between consecutive video frames (window-averaged)",
                "cols_9": "pose_6d: [dtx,dty,dtz, rot6d_r00,r01,r02,r10,r11,r12]",
            },
        }
        if pose_mats is not None:
            mat = pose_mats[frame_idx]
            report["frame_idx"] = frame_idx
            report["matrix_summary"] = _summarize_matrix(mat)
            report["first_5_rows"] = mat[:5].tolist()
            report["rows_60_65"] = mat[60:65].tolist()
            nz = int((np.abs(mat).sum(axis=1) > 1e-8).sum())
            report["padding_rows"] = f"rows {nz}..127 are zero-padded ({128 - nz} rows)"
        else:
            report["error"] = "no pose_mats for this clip"

        text = json.dumps(report, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
        break


if __name__ == "__main__":
    main()
