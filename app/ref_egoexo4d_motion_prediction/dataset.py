"""Dataset for Ego-Exo4D 3D Human Motion Prediction (EgoAgent-style windows)."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset

from app.ref_egoexo4d_motion_prediction.joints import COCO_BODY_JOINTS, NUM_BODY_JOINTS
from evals.action_anticipation_frozen.dataloader import make_transforms

logger = logging.getLogger(__name__)


@dataclass
class MotionWindowSample:
    take_uid: str
    take_name: str
    video_path: str
    annotation_path: str
    motion_start_idx: int
    motion_fps: int
    motion_window: int


def load_index_csv(path: str | Path) -> list[MotionWindowSample]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                MotionWindowSample(
                    take_uid=row["take_uid"],
                    take_name=row.get("take_name", ""),
                    video_path=row["video_path"],
                    annotation_path=row["annotation_path"],
                    motion_start_idx=int(row["motion_start_idx"]),
                    motion_fps=int(row.get("motion_fps", 30)),
                    motion_window=int(row.get("motion_window", 20)),
                )
            )
    return rows


def _sorted_frame_keys(ann_data: dict) -> list[int]:
    keys = []
    for k in ann_data.keys():
        if str(k).isdigit():
            keys.append(int(k))
    return sorted(keys)


def _extract_pose_frame(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    ann3d = entry[0].get("annotation3D", {})
    coords = np.zeros((NUM_BODY_JOINTS, 3), dtype=np.float32)
    valid = np.zeros((NUM_BODY_JOINTS,), dtype=np.float32)
    for j, name in enumerate(COCO_BODY_JOINTS):
        if name not in ann3d:
            continue
        jdat = ann3d[name]
        coords[j, 0] = float(jdat["x"])
        coords[j, 1] = float(jdat["y"])
        coords[j, 2] = float(jdat["z"])
        valid[j] = 1.0
    return coords, valid


class EgoExoMotionPredictionDataset(Dataset):
    """
    Each item:
      - video: ``context_video_frames`` RGB frames ending at the last context motion frame
      - past_motion: first ``context_motion_frames`` 3D poses
      - future_motion: last ``future_motion_frames`` 3D poses (supervision)
    """

    def __init__(
        self,
        samples: list[MotionWindowSample],
        context_video_frames: int = 5,
        context_motion_frames: int = 5,
        future_motion_frames: int = 15,
        video_target_fps: float = 8.0,
        resolution: int = 384,
        training: bool = True,
        auto_augment: bool = True,
        reprob: float = 0.25,
        random_resize_scale: tuple[float, float] = (0.08, 1.0),
        max_samples: int | None = None,
    ):
        self.samples = samples[:max_samples] if max_samples else samples
        self.context_video_frames = int(context_video_frames)
        self.context_motion_frames = int(context_motion_frames)
        self.future_motion_frames = int(future_motion_frames)
        self.motion_window = self.context_motion_frames + self.future_motion_frames
        self.video_target_fps = float(video_target_fps)
        self.training = bool(training)
        self._ann_cache: dict[str, tuple[list[int], list]] = {}

        self.transform = make_transforms(
            training=training,
            crop_size=resolution,
            auto_augment=auto_augment,
            reprob=reprob,
            random_resize_scale=random_resize_scale,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_annotation(self, path: str) -> tuple[list[int], list]:
        if path not in self._ann_cache:
            data = json.loads(Path(path).read_text())
            frame_keys = _sorted_frame_keys(data)
            self._ann_cache[path] = (frame_keys, data)
        return self._ann_cache[path]

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frame_keys, ann_data = self._load_annotation(sample.annotation_path)
        start = sample.motion_start_idx
        end = start + self.motion_window
        if end > len(frame_keys):
            raise IndexError(f"Window out of range for {sample.take_uid}: {start}+{self.motion_window}>{len(frame_keys)}")

        motion_ids = frame_keys[start:end]
        poses = []
        mask = []
        for fid in motion_ids:
            entry = ann_data[str(fid)]
            c, m = _extract_pose_frame(entry)
            poses.append(c)
            mask.append(m)
        motion = np.stack(poses, axis=0)
        motion_mask = np.stack(mask, axis=0)

        past_motion = motion[: self.context_motion_frames]
        past_mask = motion_mask[: self.context_motion_frames]
        future_motion = motion[self.context_motion_frames :]
        future_mask = motion_mask[self.context_motion_frames :]

        anchor_fid = motion_ids[self.context_motion_frames - 1]
        video = self._decode_context_video(sample.video_path, anchor_fid)

        return {
            "video": video,
            "past_motion": torch.from_numpy(past_motion),
            "past_mask": torch.from_numpy(past_mask),
            "future_motion": torch.from_numpy(future_motion),
            "future_mask": torch.from_numpy(future_mask),
            "metadata": {
                "take_uid": sample.take_uid,
                "take_name": sample.take_name,
                "anchor_frame_id": anchor_fid,
                "motion_start_idx": sample.motion_start_idx,
            },
        }

    def _decode_context_video(self, video_path: str, anchor_frame_id: int) -> torch.Tensor:
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        vfps = float(vr.get_avg_fps())
        frame_step = max(1, int(round(vfps / self.video_target_fps)))
        nframes = self.context_video_frames * frame_step
        end_frame = int(anchor_frame_id)
        indices = np.arange(end_frame - nframes, end_frame, frame_step).astype(np.int64)
        indices[indices < 0] = 0
        n_total = len(vr)
        if n_total > 0:
            indices[indices >= n_total] = n_total - 1
        clip = self.transform(vr.get_batch(indices).asnumpy())
        return clip


def collate_motion_batch(batch: list[dict]) -> dict:
    return {
        "video": torch.stack([b["video"] for b in batch], dim=0),
        "past_motion": torch.stack([b["past_motion"] for b in batch], dim=0),
        "past_mask": torch.stack([b["past_mask"] for b in batch], dim=0),
        "future_motion": torch.stack([b["future_motion"] for b in batch], dim=0),
        "future_mask": torch.stack([b["future_mask"] for b in batch], dim=0),
        "metadata": [b["metadata"] for b in batch],
    }


def make_motion_dataloader(
    index_csv: str | Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    training: bool,
    **dataset_kwargs,
) -> tuple[EgoExoMotionPredictionDataset, DataLoader]:
    samples = load_index_csv(index_csv)
    ds = EgoExoMotionPredictionDataset(samples=samples, training=training, **dataset_kwargs)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_motion_batch,
        drop_last=training,
    )
    loader.num_batches = max(1, len(ds) // max(1, batch_size))
    loader.num_samples = len(ds)
    return ds, loader
