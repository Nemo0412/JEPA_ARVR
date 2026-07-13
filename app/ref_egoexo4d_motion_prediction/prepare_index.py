#!/usr/bin/env python3
"""Build train/val index CSVs for Ego-Exo4D 3D Human Motion Prediction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from app.ref_egoexo4d_motion_prediction.joints import COCO_BODY_JOINTS

ANNOT_REL = Path("annotations/ego_pose")


def _load_take_index(takes_json: Path) -> dict[str, dict]:
    takes = json.loads(takes_json.read_text())
    if isinstance(takes, list):
        return {t["take_uid"]: t for t in takes if t.get("take_uid")}
    if isinstance(takes, dict):
        values = list(takes.values())
        if values and isinstance(values[0], dict) and "take_uid" in values[0]:
            return {t["take_uid"]: t for t in values}
        if values and isinstance(values[0], dict):
            return takes
    raise ValueError(f"Unrecognized takes.json structure in {takes_json}")


def _rgb_video_path(take: dict, egoexo_root: Path) -> Path | None:
    root_dir = take.get("root_dir")
    if not root_dir:
        return None
    rel = (
        take.get("frame_aligned_videos", {})
        .get("aria01", {})
        .get("rgb", {})
        .get("relative_path")
    )
    if not rel:
        return None
    take_root = egoexo_root / root_dir
    candidates = [take_root / rel]
    rgb_name = Path(rel).name
    downscaled_root = take_root / "frame_aligned_videos" / "downscaled"
    if downscaled_root.is_dir():
        for res_dir in sorted(downscaled_root.iterdir()):
            if res_dir.is_dir():
                candidates.append(res_dir / rgb_name)
    for path in candidates:
        if path.exists():
            return path
    return None


def _annotation_dir(egoexo_root: Path, split: str) -> Path:
    for candidate in (
        egoexo_root / ANNOT_REL / split / "body" / "annotation",
        egoexo_root / ANNOT_REL / split / "body_pose" / "annotation",
        egoexo_root / "annotations" / "ego_pose" / split / "body" / "annotation",
    ):
        if candidate.is_dir():
            return candidate
    return egoexo_root / ANNOT_REL / split / "body" / "annotation"


def _valid_frame_count(ann_path: Path, min_frames: int) -> int:
    data = json.loads(ann_path.read_text())
    ok = 0
    for frame_key, entries in data.items():
        if not str(frame_key).isdigit():
            continue
        if not entries:
            continue
        ann3d = entries[0].get("annotation3D", {})
        if all(j in ann3d for j in COCO_BODY_JOINTS):
            ok += 1
    return ok if ok >= min_frames else 0


def build_index(
    egoexo_root: Path,
    output_dir: Path,
    motion_fps: int = 30,
    motion_window: int = 20,
    stride: int = 5,
    selected_takes: Path | None = None,
) -> tuple[int, int]:
    takes_json = egoexo_root / "takes.json"
    if not takes_json.exists():
        raise FileNotFoundError(f"Missing takes.json under {egoexo_root}")

    take_by_uid = _load_take_index(takes_json)
    selected_uids: set[str] | None = None
    if selected_takes and selected_takes.exists():
        selected_uids = {
            line.strip()
            for line in selected_takes.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val"):
        ann_dir = _annotation_dir(egoexo_root, split)
        if not ann_dir.is_dir():
            raise FileNotFoundError(
                f"Body-pose annotations not found: {ann_dir}\n"
                "Download with: egoexo -o <root> --parts annotations --benchmarks bodypose"
            )

        rows = []
        for ann_path in sorted(ann_dir.glob("*.json")):
            take_uid = ann_path.stem
            if selected_uids is not None and take_uid not in selected_uids:
                continue
            take = take_by_uid.get(take_uid)
            if take is None or take.get("is_dropped"):
                continue
            video_path = _rgb_video_path(take, egoexo_root)
            if video_path is None:
                continue
            n_valid = _valid_frame_count(ann_path, motion_window)
            if n_valid < motion_window:
                continue

            max_start = n_valid - motion_window
            for start_idx in range(0, max_start + 1, stride):
                rows.append(
                    {
                        "take_uid": take_uid,
                        "take_name": take.get("take_name", ""),
                        "video_path": str(video_path),
                        "annotation_path": str(ann_path),
                        "motion_start_idx": start_idx,
                        "motion_fps": motion_fps,
                        "motion_window": motion_window,
                    }
                )

        out_csv = output_dir / f"egoexo_motion_{split}_fps{motion_fps}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "take_uid",
                    "take_name",
                    "video_path",
                    "annotation_path",
                    "motion_start_idx",
                    "motion_fps",
                    "motion_window",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        counts[split] = len(rows)
        print(f"[prepare_index] {split}: {len(rows)} windows -> {out_csv}")

    return counts.get("train", 0), counts.get("val", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egoexo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motion-fps", type=int, default=30, choices=(10, 30))
    parser.add_argument("--motion-window", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--selected-takes", type=Path, default=None)
    args = parser.parse_args()
    build_index(
        egoexo_root=args.egoexo_root,
        output_dir=args.output_dir,
        motion_fps=args.motion_fps,
        motion_window=args.motion_window,
        stride=args.stride,
        selected_takes=args.selected_takes,
    )


if __name__ == "__main__":
    main()
