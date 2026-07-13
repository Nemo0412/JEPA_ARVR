#!/usr/bin/env python
"""Per-sample B11 pose coverage audit: trace SLAM zip/sync/clip/raster pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VJEPA_ROOT = REPO_ROOT / "vjepa2"
for path in (REPO_ROOT, VJEPA_ROOT):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate, patch_metadata_dataloader  # noqa: E402
from app.hdepic_lora_action_anticipation.pose_map_builder import (  # noqa: E402
    GazePoseInputMapBuilder,
    InterframePoseMapBuilder,
    rasterize_pose_matrix_to_patch,
)
from app.hdepic_lora_action_anticipation.pose_slam import SlamPoseLoader  # noqa: E402
from evals.action_anticipation_frozen.dataloader import filter_annotations, init_data  # noqa: E402


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


def _participant_source(video_id: str) -> str:
    text = str(video_id)
    if "_" in text:
        return text.split("_", 1)[0]
    if "-" in text:
        return text.split("-", 1)[0]
    return "unknown"


def _raster_nonzero_for_meta(pose_builder: InterframePoseMapBuilder, meta) -> int:
    loader = pose_builder.pose_loader
    pose_mats = loader.query_interframe_matrices(meta, pose_builder.k_max)
    if pose_mats is None:
        return 0
    total = 0
    nframes = int(pose_mats.shape[0])
    for t in range(nframes):
        patch = rasterize_pose_matrix_to_patch(
            pose_mats[t],
            pose_builder.patch_height,
            pose_builder.patch_width,
            normalize=pose_builder.normalize,
        )
        total += int(np.count_nonzero(patch))
    return total


def audit_sample(
    pose_builder: InterframePoseMapBuilder,
    meta: dict[str, Any],
    *,
    sample_idx: int,
    split: str,
) -> dict[str, Any]:
    loader: SlamPoseLoader = pose_builder.pose_loader
    video_id = str(meta.get("video_id", ""))
    row: dict[str, Any] = {
        "sample_idx": sample_idx,
        "split": split,
        "video_id": video_id,
        "source": _participant_source(video_id),
        "session_id": None,
        "zip_found": False,
        "inner_csv": None,
        "sync_found": False,
        "clip_window_ok": False,
        "clip_record_n": 0,
        "frame_ts_n": 0,
        "pose_mats_shape": None,
        "pose_mats_nonzero": 0,
        "raster_nonzero": 0,
        "status": "ok",
    }

    session_id = loader.resolve_session_id(video_id)
    row["session_id"] = session_id
    if session_id is None:
        row["status"] = "no_session_mapping"
        return row

    zip_path = loader._zip_path_for_session(session_id)  # noqa: SLF001
    row["zip_found"] = zip_path is not None
    if not row["zip_found"]:
        row["status"] = "no_zip"
        return row

    inner = loader._inner_csv_path(zip_path, session_id)  # noqa: SLF001
    row["inner_csv"] = inner
    if inner is None:
        row["status"] = "no_inner_csv"
        return row

    sync = loader._sync_for_video(video_id)  # noqa: SLF001
    row["sync_found"] = (
        sync is not None and {"mp4_time_ns", "vrs_device_time_ns"}.issubset(sync.columns)
    )

    window = loader._clip_time_us(meta)  # noqa: SLF001
    row["clip_window_ok"] = window is not None
    if not row["clip_window_ok"]:
        row["status"] = "no_clip_window"
        return row

    record = loader._load_clip_record(meta)  # noqa: SLF001
    if record is None:
        row["status"] = "no_clip_record"
        return row
    row["clip_record_n"] = int(record.timestamps_us.size)
    if row["clip_record_n"] < 2:
        row["status"] = "no_clip_record"
        return row

    frame_ts = loader.frame_timestamps_us(meta)
    if frame_ts is None:
        row["status"] = "no_frame_ts"
        return row
    row["frame_ts_n"] = int(frame_ts.shape[0])

    pose_mats = loader.query_interframe_matrices(meta, pose_builder.k_max)
    if pose_mats is None:
        row["status"] = "no_pose_mats"
        return row
    row["pose_mats_shape"] = list(pose_mats.shape)
    row["pose_mats_nonzero"] = int(np.count_nonzero(pose_mats))
    row["raster_nonzero"] = _raster_nonzero_for_meta(pose_builder, meta)

    if row["pose_mats_nonzero"] == 0:
        row["status"] = "zero_pose_mats"
    elif row["raster_nonzero"] == 0:
        row["status"] = "zero_raster"
    else:
        row["status"] = "ok"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-subset-path", default="")
    parser.add_argument("--json", action="store_true")
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
    gate = GazeTokenGate(gaze_cfg)
    builder = GazePoseInputMapBuilder(gaze_cfg, gate=gate)
    pose_builder = builder.pose_builder

    rows: list[dict[str, Any]] = []
    sample_idx = 0
    for batch in loader:
        metadata = batch[3]
        metas = metadata if isinstance(metadata, list) else [metadata]
        for meta in metas:
            if sample_idx >= args.max_samples:
                break
            rows.append(
                audit_sample(
                    pose_builder,
                    meta,
                    sample_idx=sample_idx,
                    split=args.split,
                )
            )
            sample_idx += 1
        if sample_idx >= args.max_samples:
            break

    status_counts = dict(Counter(row["status"] for row in rows))
    source_counts = dict(Counter(row["source"] for row in rows))
    payload = {
        "config": str(args.config),
        "split": args.split,
        "max_samples": args.max_samples,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "samples_seen": len(rows),
        "status_counts": status_counts,
        "source_counts": source_counts,
        "pose_k_max": int(pose_builder.k_max),
        "pose_feature_dim": int(pose_builder.pose_loader.input_dim),
        "pose_patch_hw": [int(pose_builder.patch_height), int(pose_builder.patch_width)],
        "rows": rows,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"split={args.split} samples={len(rows)} status_counts={status_counts} source_counts={source_counts}")
    for row in rows:
        print(
            "  idx={sample_idx} status={status} video_id={video_id} session_id={session_id} "
            "zip={zip_found} inner_csv={inner_csv} sync={sync_found} clip_n={clip_record_n} "
            "pose_nz={pose_mats_nonzero} raster_nz={raster_nonzero}".format(**row)
        )


if __name__ == "__main__":
    main()
