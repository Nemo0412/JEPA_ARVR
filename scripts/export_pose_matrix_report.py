#!/usr/bin/env python
"""Export a full inter-frame pose matrix [128,9] as a human-readable report file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "vjepa2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.dump_pose_examples import _make_gaze_cfg  # noqa: E402
from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate, patch_metadata_dataloader  # noqa: E402
from app.hdepic_lora_action_anticipation.pose_slam import SlamPoseLoader  # noqa: E402
from evals.action_anticipation_frozen.dataloader import filter_annotations, init_data  # noqa: E402

COL_NAMES = ["dtx", "dty", "dtz", "r00", "r01", "r02", "r10", "r11", "r12"]


def _format_matrix(mat: np.ndarray) -> str:
    header = "row\\col  " + "  ".join(f"{name:>10s}" for name in COL_NAMES)
    lines = [header, "-" * len(header)]
    for r in range(mat.shape[0]):
        vals = "  ".join(f"{mat[r, c]:10.6f}" for c in range(mat.shape[1]))
        lines.append(f"{r:3d}      {vals}")
    return "\n".join(lines)


def _write_report(
    out_path: Path,
    *,
    video_id: str,
    frame_idx: int,
    mat: np.ndarray,
    feature_set: str,
    k_max: int,
) -> None:
    nz_rows = int((np.abs(mat).sum(axis=1) > 1e-8).sum())
    lines = [
        "HD-EPIC Inter-frame Pose Matrix Example",
        "=" * 72,
        "",
        "1. What is this matrix?",
        "   Each video frame t (except the last) has one matrix [K, D] = [128, 9].",
        "   It encodes all SLAM pose samples between frame t and frame t+1.",
        "",
        "2. Dimensions",
        f"   - Rows (K={k_max}): max SLAM samples in one inter-frame interval,",
        "     window-averaged when SLAM rate is higher than K.",
        "     Unused rows are zero-padded at the bottom.",
        f"   - Cols (D=9): feature_set={feature_set!r}",
        "       [0:3] dtx,dty,dtz  relative translation delta (meters, vs segment start)",
        "       [3:9] rot6d         relative rotation (6D rep; identity ~ r00=r11=1)",
        "",
        "3. Normalization",
        "   pose_map.normalize = none  (no minmax scaling; values are physical/engineered features)",
        "",
        "4. Placement in model input",
        "   The [128,9] patch is pasted to the top-left corner of a 384x384 canvas;",
        "   the remaining canvas pixels are zero.",
        "",
        "5. This sample",
        f"   video_id  : {video_id}",
        f"   frame_idx : {frame_idx}  (matrix for interval frame[{frame_idx}] -> frame[{frame_idx+1}])",
        f"   shape     : {list(mat.shape)}",
        f"   nonzero rows: {nz_rows} / {mat.shape[0]}  (rows {nz_rows}..{mat.shape[0]-1} are padding)",
        f"   value range: min={mat.min():.6f}, max={mat.max():.6f}, mean={mat.mean():.6f}",
        "",
        "6. Full matrix (128 rows x 9 cols)",
        "",
        _format_matrix(mat),
        "",
        "7. Notes for interpretation",
        "   - Row 0 translation is [0,0,0] by definition (segment origin).",
        "   - Translation columns stay small (cm-scale) when head motion is mild.",
        "   - Rotation columns near 1.0 are expected for near-identity relative rotation.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "logs" / "pose_matrix_full_example.txt")
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--frame-idx", type=int, default=16)
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
    loader_pose = SlamPoseLoader(gaze_cfg["pose"], gate=GazeTokenGate(gaze_cfg))

    for batch_idx, batch in enumerate(loader):
        if batch_idx != args.batch_idx:
            continue
        meta = batch[3][args.sample_idx]
        pose_mats = loader_pose.query_interframe_matrices(meta, int(gaze_cfg["pose"]["interframe_k_max"]))
        if pose_mats is None:
            raise RuntimeError(f"no pose for video_id={meta.get('video_id')}")
        frame_idx = min(args.frame_idx, pose_mats.shape[0] - 1)
        mat = pose_mats[frame_idx]
        _write_report(
            args.out,
            video_id=str(meta.get("video_id")),
            frame_idx=frame_idx,
            mat=mat,
            feature_set=str(gaze_cfg["pose"].get("feature_set", "pose_6d")),
            k_max=int(gaze_cfg["pose"]["interframe_k_max"]),
        )
        print(f"wrote {args.out}")
        return
    raise RuntimeError(f"batch_idx={args.batch_idx} not found")


if __name__ == "__main__":
    main()
