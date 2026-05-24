"""Project-local gaze utilities for HD-EPIC LoRA action anticipation."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import tempfile
import time
import zipfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import webdataset as wds
from decord import VideoReader, cpu
from torch.utils.data import get_worker_info
from evals.action_anticipation_frozen.epickitchens import DataInfo, split_by_node
from src.datasets.utils.worker_init_fn import pl_worker_init_function
from src.utils.logging import AverageMeter

logger = logging.getLogger(__name__)


def _as_path(value) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def _clean_video_id(video_id: str) -> str:
    text = str(video_id)
    if "_" in text:
        participant, rest = text.split("_", 1)
        return f"{participant}-{rest}"
    return text


def _vjepa_video_id(video_id: str) -> str:
    text = str(video_id)
    if "-" in text:
        participant, rest = text.split("-", 1)
        return f"{participant}_{rest}"
    return text


class ClipBalancedDecodeVideosToClips(wds.PipelineStage):
    def __init__(
        self,
        frames_per_clip=16,
        fps=5,
        transform=None,
        anticipation_time_sec=(0.0, 0.0),
        model_anticipation_time_sec=None,
        anticipation_point=(0.25, 0.75),
        emit_metadata=False,
        emit_binary_map=False,
        binary_map_cfg=None,
        drop_incomplete_history=False,
        epoch=None,
        label_horizon_schedule=None,
    ):
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.anticipation_time = anticipation_time_sec
        self.model_anticipation_time = model_anticipation_time_sec or anticipation_time_sec
        self.anticipation_point = anticipation_point
        self.emit_metadata = emit_metadata
        self.emit_binary_map = emit_binary_map
        self.binary_map_cfg = dict(binary_map_cfg or {})
        self.drop_incomplete_history = drop_incomplete_history
        self.epoch = epoch
        self.label_horizon_schedule = list(label_horizon_schedule or [])
        self._reader_path: str | None = None
        self._reader_state: tuple[VideoReader, float, int] | None = None
        self._binary_gate = None
        self._binary_grid_cache: dict[tuple[int, int, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _reader(self, path: str):
        if self._reader_path == path and self._reader_state is not None:
            return self._reader_state
        vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
        vr.seek(0)
        vfps = float(vr.get_avg_fps())
        nframes_total = len(vr)
        self._reader_path = path
        self._reader_state = (vr, vfps, nframes_total)
        return self._reader_state

    def _sample_anticipation_range(self):
        if not self.label_horizon_schedule:
            return self.anticipation_time
        epoch = int(self.epoch.get_value()) if self.epoch is not None else 0
        for stage in self.label_horizon_schedule:
            until_epoch = stage.get("until_epoch")
            if until_epoch is None or epoch < int(until_epoch):
                return tuple(float(x) for x in stage["label_horizon_sec"])
        return tuple(float(x) for x in self.label_horizon_schedule[-1]["label_horizon_sec"])

    def _get_binary_gate(self):
        if self._binary_gate is None:
            self._binary_gate = GazeTokenGate({**self.binary_map_cfg, "mode": "token_gate"})
        return self._binary_gate

    def _binary_grid(self, height: int, width: int, dtype: torch.dtype):
        key = (int(height), int(width), dtype)
        cached = self._binary_grid_cache.get(key)
        if cached is not None:
            return cached
        yy = torch.arange(height, dtype=dtype).view(1, height, 1)
        xx = torch.arange(width, dtype=dtype).view(1, 1, width)
        self._binary_grid_cache[key] = (yy, xx)
        return yy, xx

    def _query_binary_xy(self, meta):
        gate = self._get_binary_gate()
        record = gate._load_record(str(meta["video_id"]))  # noqa: SLF001 - reuse existing gaze sync logic
        if record is None:
            return None
        return gate._query_crop_xy(  # noqa: SLF001
            record,
            meta.get("frame_indices"),
            meta.get("vfps", 30.0),
            int(meta.get("height", self.binary_map_cfg.get("crop_size", 384))),
            int(meta.get("width", self.binary_map_cfg.get("crop_size", 384))),
        )

    def _build_binary_map(self, video: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
        _, frames, height, width = video.shape
        binary_map = video.new_zeros((1, frames, height, width))
        if bool(self.binary_map_cfg.get("force_zero_map", False)):
            return binary_map
        xy = self._query_binary_xy(meta)
        if xy is None:
            if bool(self.binary_map_cfg.get("fallback_full_frame", False)):
                binary_map.fill_(1.0)
            return binary_map

        nframes = min(frames, xy.shape[0])
        if nframes <= 0:
            return binary_map
        crop_size = int(self.binary_map_cfg.get("crop_size", max(height, width)))
        radius_px = float(self.binary_map_cfg.get("binary_radius_px", self.binary_map_cfg.get("binary_radius", 64.0)))
        yy, xx = self._binary_grid(height, width, video.dtype)
        xy_t = torch.as_tensor(xy[:nframes], dtype=video.dtype)
        x = xy_t[:, 0].view(nframes, 1, 1) * (width - 1) / max(1, crop_size - 1)
        y = xy_t[:, 1].view(nframes, 1, 1) * (height - 1) / max(1, crop_size - 1)
        binary_map[0, :nframes] = (((xx - x) ** 2 + (yy - y) ** 2) <= radius_px**2).to(video.dtype)
        return binary_map

    def run(self, src):
        for item in src:
            try:
                path = str(item["video_path"])
                video_id = str(item["video_id"])
                sf = int(item["start_frame"])
                ef = int(item["stop_frame"])
                labels_verb = int(item["verb_class"])
                labels_noun = int(item["noun_class"])
                vr, vfps, nframes_total = self._reader(path)
                frame_step = max(1, int(vfps / self.fps))
                nframes = int(self.frames_per_clip * frame_step)
                sample_at = float(np.random.uniform(*self._sample_anticipation_range()))
                model_at = float(np.random.uniform(*self.model_anticipation_time))
                aframes = int(sample_at * vfps)
                ap = float(np.random.uniform(*self.anticipation_point))
                af = int(sf * ap + (1 - ap) * ef - aframes)
                if self.drop_incomplete_history and af - nframes < 0:
                    continue
                indices = np.arange(af - nframes, af, frame_step).astype(np.int64)
                indices[indices < 0] = 0
                if nframes_total > 0:
                    indices[indices >= nframes_total] = nframes_total - 1
                buffer = vr.get_batch(indices).asnumpy()
            except Exception as exc:
                logging.info("Encountered exception decoding clip-balanced sample: %r", exc)
                continue

            height, width = int(buffer.shape[1]), int(buffer.shape[2])
            if self.transform is not None:
                buffer = self.transform(buffer)

            out = {
                "video": buffer,
                "verb": labels_verb,
                "noun": labels_noun,
                "anticipation_time": model_at,
            }
            if self.emit_metadata:
                out["metadata"] = {
                    "video_id": video_id,
                    "video_path": path,
                    "frame_indices": indices.tolist(),
                    "vfps": vfps,
                    "height": height,
                    "width": width,
                    "start_frame": sf,
                    "stop_frame": ef,
                    "anticipation_point": ap,
                    "sample_anticipation_time": sample_at,
                    "model_anticipation_time": model_at,
                }
            if self.emit_binary_map:
                meta = out.get("metadata")
                if meta is None:
                    meta = {
                        "video_id": video_id,
                        "frame_indices": indices.tolist(),
                        "vfps": vfps,
                        "height": height,
                        "width": width,
                    }
                out["binary_map"] = self._build_binary_map(buffer, meta)
            yield out


def _metadata_collate(batch):
    videos = torch.stack([item[0] for item in batch])
    verbs = torch.tensor([item[1] for item in batch], dtype=torch.long)
    nouns = torch.tensor([item[2] for item in batch], dtype=torch.long)
    metadata = [item[3] for item in batch]
    anticipation = torch.tensor([item[4] for item in batch], dtype=torch.float32)
    return videos, verbs, nouns, metadata, anticipation


def _metadata_binary_collate(batch):
    videos = torch.stack([item[0] for item in batch])
    verbs = torch.tensor([item[1] for item in batch], dtype=torch.long)
    nouns = torch.tensor([item[2] for item in batch], dtype=torch.long)
    metadata = [item[3] for item in batch]
    anticipation = torch.tensor([item[4] for item in batch], dtype=torch.float32)
    binary_maps = torch.stack([item[5] for item in batch])
    return videos, verbs, nouns, metadata, anticipation, binary_maps


def _build_clip_samples(paths, annotations):
    path_by_video_id = {Path(path).stem: str(path) for path in paths}
    samples = []
    for video_index, (video_id, df) in enumerate(annotations.items()):
        path = path_by_video_id.get(str(video_id))
        if path is None:
            continue
        for action_index, row in enumerate(df.itertuples(index=False)):
            samples.append(
                {
                    "video_path": path,
                    "video_id": str(video_id),
                    "video_index": int(video_index),
                    "action_index": int(action_index),
                    "start_frame": int(getattr(row, "start_frame")),
                    "stop_frame": int(getattr(row, "stop_frame")),
                    "verb_class": int(getattr(row, "verb_class")),
                    "noun_class": int(getattr(row, "noun_class")),
                }
            )
    return samples


def _filter_samples_with_full_history(samples, anticipation_time_sec, anticipation_point, frames_per_clip, fps):
    sample_horizon = max(float(x) for x in anticipation_time_sec)
    ap = max(float(x) for x in anticipation_point)
    kept = []
    dropped = 0
    fps_cache: dict[str, float | None] = {}
    for item in samples:
        path = str(item["video_path"])
        if path not in fps_cache:
            try:
                vr = VideoReader(path, num_threads=1, ctx=cpu(0))
                fps_cache[path] = float(vr.get_avg_fps())
            except Exception as exc:
                logger.warning("Could not read fps for past-window sample filter: path=%s error=%r", path, exc)
                fps_cache[path] = None
        vfps = fps_cache[path]
        if vfps is None:
            dropped += 1
            continue
        frame_step = max(1, int(vfps / float(fps)))
        nframes = int(int(frames_per_clip) * frame_step)
        sf = int(item["start_frame"])
        ef = int(item["stop_frame"])
        anchor = int(sf * ap + (1.0 - ap) * ef)
        observed_end = anchor - int(sample_horizon * vfps)
        if observed_end - nframes < 0:
            dropped += 1
            continue
        kept.append(item)
    logger.info(
        "Past-window full-history filter: kept=%d dropped=%d sample_horizon=%.3fs frames_per_clip=%d fps=%s",
        len(kept),
        dropped,
        sample_horizon,
        int(frames_per_clip),
        fps,
    )
    return kept


class ResampledItems(torch.utils.data.IterableDataset):
    def __init__(self, items, epoch, training):
        super().__init__()
        self.items = list(items)
        self.epoch = epoch
        self.training = training
        logger.info("Done initializing clip-balanced items: %d", len(self.items))

    def __iter__(self):
        if not self.training:
            order = range(len(self.items))
        else:
            epoch = self.epoch.get_value()
            gen = torch.Generator()
            gen.manual_seed(epoch)
            by_video: OrderedDict[int, list[int]] = OrderedDict()
            for idx, item in enumerate(self.items):
                by_video.setdefault(int(item["video_index"]), []).append(idx)
            videos = list(by_video.keys())
            perm = torch.randperm(len(videos), generator=gen).tolist()
            order = []
            for pos in perm:
                video_items = by_video[videos[int(pos)]]
                if len(video_items) > 1:
                    item_perm = torch.randperm(len(video_items), generator=gen).tolist()
                    order.extend(video_items[int(i)] for i in item_perm)
                else:
                    order.extend(video_items)
        for idx in order:
            yield self.items[int(idx)]


class ContiguousSplitByWorker(wds.PipelineStage):
    def run(self, src):
        items = list(src)
        info = get_worker_info()
        if info is None or info.num_workers <= 1:
            yield from items
            return
        n = len(items)
        chunk = int(math.ceil(n / float(info.num_workers)))
        start = min(n, info.id * chunk)
        end = min(n, start + chunk)
        yield from items[start:end]


def make_clip_balanced_webvid(
    base_path,
    annotations_path,
    batch_size,
    transform,
    frames_per_clip=16,
    fps=5,
    num_workers=8,
    world_size=1,
    rank=0,
    anticipation_time_sec=(0.0, 0.0),
    model_anticipation_time_sec=None,
    persistent_workers=True,
    pin_memory=True,
    training=True,
    anticipation_point=(0.1, 0.1),
    emit_metadata=False,
    emit_binary_map=False,
    binary_map_cfg=None,
    drop_incomplete_history=False,
    label_horizon_schedule=None,
    **kwargs,
):
    del base_path, kwargs
    if emit_binary_map and training and bool((binary_map_cfg or {}).get("disable_train_aug", False)):
        if hasattr(transform, "training"):
            transform.training = False
            logger.info("Disabled training video augmentation for binary-map alignment")
    paths, annotations = annotations_path
    samples = _build_clip_samples(paths, annotations)
    if drop_incomplete_history:
        samples = _filter_samples_with_full_history(
            samples,
            anticipation_time_sec=anticipation_time_sec,
            anticipation_point=anticipation_point,
            frames_per_clip=frames_per_clip,
            fps=fps,
        )

    from evals.action_anticipation_frozen.epickitchens import SharedEpoch

    epoch = SharedEpoch(epoch=0)
    decoder = ClipBalancedDecodeVideosToClips(
        frames_per_clip=frames_per_clip,
        fps=fps,
        transform=transform,
        anticipation_time_sec=anticipation_time_sec,
        model_anticipation_time_sec=model_anticipation_time_sec,
        anticipation_point=anticipation_point,
        emit_metadata=emit_metadata,
        emit_binary_map=emit_binary_map,
        binary_map_cfg=binary_map_cfg,
        drop_incomplete_history=drop_incomplete_history,
        epoch=epoch,
        label_horizon_schedule=label_horizon_schedule,
    )
    if emit_binary_map:
        tuple_keys = ("video", "verb", "noun", "metadata", "anticipation_time", "binary_map")
        collate = _metadata_binary_collate
    elif emit_metadata:
        tuple_keys = ("video", "verb", "noun", "metadata", "anticipation_time")
        collate = _metadata_collate
    else:
        tuple_keys = ("video", "verb", "noun", "anticipation_time")
        collate = torch.utils.data.default_collate
    pipeline = [
        ResampledItems(samples, epoch=epoch, training=training),
        split_by_node(rank=rank, world_size=world_size),
        ContiguousSplitByWorker(),
        decoder,
        wds.to_tuple(*tuple_keys),
        wds.batched(batch_size, partial=True, collation_fn=collate),
    ]
    dataset = wds.DataPipeline(*pipeline)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        worker_init_fn=pl_worker_init_function,
        pin_memory=pin_memory,
    )
    dataloader.num_batches = len(samples) // (world_size * batch_size)
    dataloader.num_samples = len(samples)
    logger.info(
        "Using clip-balanced HD-EPIC dataloader: samples=%d batch_size=%d workers=%d emit_metadata=%s emit_binary_map=%s",
        len(samples),
        batch_size,
        num_workers,
        emit_metadata,
        emit_binary_map,
    )
    return dataset, dataloader, DataInfo(dataloader=dataloader, shared_epoch=epoch)


def patch_metadata_dataloader(
    model_anticipation_time_sec=None,
    drop_incomplete_history=False,
    apply_to_train=False,
    train_label_horizon_schedule=None,
    emit_binary_map=False,
    binary_map_cfg=None,
):
    import evals.action_anticipation_frozen.dataloader as dl
    import evals.action_anticipation_frozen.epickitchens as ek

    def _make(*args, **kwargs):
        kwargs["emit_metadata"] = True
        kwargs["emit_binary_map"] = emit_binary_map
        kwargs["binary_map_cfg"] = binary_map_cfg
        should_apply = model_anticipation_time_sec is not None and (apply_to_train or not bool(kwargs.get("training", True)))
        if should_apply:
            kwargs["model_anticipation_time_sec"] = model_anticipation_time_sec
            kwargs["drop_incomplete_history"] = drop_incomplete_history
        if bool(kwargs.get("training", True)) and train_label_horizon_schedule:
            kwargs["label_horizon_schedule"] = train_label_horizon_schedule
        return make_clip_balanced_webvid(*args, **kwargs)

    ek.make_webvid = _make
    dl.ek100_make_webvid = _make


def patch_clip_balanced_dataloader(
    model_anticipation_time_sec=None,
    drop_incomplete_history=False,
    apply_to_train=False,
    train_label_horizon_schedule=None,
):
    import evals.action_anticipation_frozen.dataloader as dl
    import evals.action_anticipation_frozen.epickitchens as ek

    def _make(*args, **kwargs):
        kwargs["emit_metadata"] = False
        should_apply = model_anticipation_time_sec is not None and (apply_to_train or not bool(kwargs.get("training", True)))
        if should_apply:
            kwargs["model_anticipation_time_sec"] = model_anticipation_time_sec
            kwargs["drop_incomplete_history"] = drop_incomplete_history
        if bool(kwargs.get("training", True)) and train_label_horizon_schedule:
            kwargs["label_horizon_schedule"] = train_label_horizon_schedule
        return make_clip_balanced_webvid(*args, **kwargs)

    ek.make_webvid = _make
    dl.ek100_make_webvid = _make


def _find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _numeric(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _yaw_pitch_to_xy(yaw, pitch, fov_deg):
    fov = math.radians(max(1e-3, min(179.0, fov_deg)))
    scale = 2.0 * math.tan(fov / 2.0)
    x = 0.5 + math.tan(float(yaw)) / scale
    y = 0.5 - math.tan(float(pitch)) / scale
    return np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)


def _image_coords_after_resize_center_crop(u0, v0, h0: int, w0: int, out_size: int):
    short_side = int(256.0 / 224.0 * out_size)
    scale = short_side / float(max(1, min(h0, w0)))
    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))
    left = (new_w - out_size) // 2
    top = (new_h - out_size) // 2
    return u0.astype(np.float64) * scale - left, v0.astype(np.float64) * scale - top


@dataclass
class GazeRecord:
    timestamps_us: np.ndarray
    xy_norm: np.ndarray | None
    yaw: np.ndarray | None
    pitch: np.ndarray | None
    sync: pd.DataFrame | None


class GazeTokenGate:
    def __init__(self, cfg: dict[str, Any]):
        self.enabled = str(cfg.get("mode", "none")).lower() == "token_gate"
        self.gamma = float(cfg.get("gamma", 0.7))
        self.sigma_px = float(cfg.get("sigma_px", 40.0))
        self.fallback_fov_deg = float(cfg.get("fallback_fov_deg", 90.0))
        self.patch_size = int(cfg.get("patch_size", 16))
        self.tubelet_size = int(cfg.get("tubelet_size", 2))
        self.crop_size = int(cfg.get("crop_size", 384))
        self.gaze_root = _as_path(cfg.get("gaze_root"))
        self.extract_root = _as_path(cfg.get("extract_root"))
        self.sync_root = _as_path(cfg.get("sync_root"))
        self.use_motion = bool(cfg.get("use_motion", True))
        self.motion_weight = float(cfg.get("motion_weight", 0.15))
        self.cache: dict[str, GazeRecord | None] = {}

    def _extract_zip(self, video_id: str) -> Path | None:
        if self.gaze_root is None:
            return None
        clean_id = _clean_video_id(video_id)
        zip_path = _find_first(self.gaze_root, [f"mps_{clean_id}_vrs.zip", f"*{clean_id}*vrs.zip"])
        if zip_path is None:
            return None
        if self.extract_root is None:
            root = Path(tempfile.gettempdir()) / "hdepic_gaze_extract"
        else:
            root = self.extract_root
        out_dir = root / zip_path.stem
        gaze_csv = _find_first(out_dir, ["general_eye_gaze.csv"])
        if gaze_csv is not None:
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        return out_dir

    def _load_record(self, video_id: str) -> GazeRecord | None:
        if video_id in self.cache:
            return self.cache[video_id]
        clean_id = _clean_video_id(video_id)
        roots: list[tuple[Path, bool]] = []
        if self.extract_root is not None:
            roots.append((self.extract_root, False))
        extracted = self._extract_zip(clean_id)
        if extracted is not None:
            roots.append((extracted, True))
        gaze_csv = None
        for root, allow_generic in roots:
            patterns = [f"*{clean_id}*/**/general_eye_gaze.csv"]
            if allow_generic:
                patterns.extend(["**/general_eye_gaze.csv", "general_eye_gaze.csv"])
            gaze_csv = _find_first(root, patterns)
            if gaze_csv is not None:
                break
        if gaze_csv is None:
            self.cache[video_id] = None
            return None

        df = pd.read_csv(gaze_csv)
        lower = {c.lower(): c for c in df.columns}
        t_col = lower.get("tracking_timestamp_us") or next((c for c in df.columns if "timestamp" in c.lower()), None)
        if t_col is None:
            self.cache[video_id] = None
            return None

        left_yaw = lower.get("left_yaw_rads_cpf")
        right_yaw = lower.get("right_yaw_rads_cpf")
        pitch_col = lower.get("pitch_rads_cpf")
        yaw = None
        pitch = None
        xy = None
        if left_yaw and right_yaw and pitch_col:
            yaw = (
                pd.to_numeric(df[left_yaw], errors="coerce").to_numpy(dtype=np.float64)
                + pd.to_numeric(df[right_yaw], errors="coerce").to_numpy(dtype=np.float64)
            ) * 0.5
            pitch = pd.to_numeric(df[pitch_col], errors="coerce").to_numpy(dtype=np.float64)
        else:
            x_col = next((c for c in df.columns if c.lower() in {"gaze_x", "gaze_center_x", "x", "u"}), None)
            y_col = next((c for c in df.columns if c.lower() in {"gaze_y", "gaze_center_y", "y", "v"}), None)
            yaw_cols = [c for c in df.columns if "yaw" in c.lower()]
            pitch_cols = [c for c in df.columns if "pitch" in c.lower()]
            if yaw_cols and pitch_cols:
                yaw = df[yaw_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy(dtype=np.float64)
                pitch = pd.to_numeric(df[pitch_cols[0]], errors="coerce").to_numpy(dtype=np.float64)
            elif x_col and y_col:
                x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64)
                y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64)
                if np.nanmax(np.abs(x)) > 2.0 or np.nanmax(np.abs(y)) > 2.0:
                    x = x / max(1.0, np.nanmax(x))
                    y = y / max(1.0, np.nanmax(y))
                xy = np.stack([x, y], axis=1)
            else:
                self.cache[video_id] = None
                return None

        if yaw is not None and pitch is not None:
            xy = np.array([_yaw_pitch_to_xy(a, b, self.fallback_fov_deg) for a, b in zip(yaw, pitch)], dtype=np.float64)

        if xy is None:
            self.cache[video_id] = None
            return None

        ts = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(ts) & np.isfinite(xy).all(axis=1)
        if yaw is not None and pitch is not None:
            valid = valid & np.isfinite(yaw) & np.isfinite(pitch)
        sync = None
        if self.sync_root is not None:
            sync_path = _find_first(self.sync_root, [f"{clean_id}_mp4_to_vrs_time_ns.csv", f"*{clean_id}*mp4_to_vrs_time_ns.csv"])
            if sync_path is not None:
                sync = pd.read_csv(sync_path)
        ts = ts[valid]
        xy = xy[valid]
        yaw = yaw[valid] if yaw is not None else None
        pitch = pitch[valid] if pitch is not None else None
        order = np.argsort(ts, kind="stable")
        record = GazeRecord(
            timestamps_us=ts[order],
            xy_norm=np.clip(xy[order], 0.0, 1.0),
            yaw=yaw[order] if yaw is not None else None,
            pitch=pitch[order] if pitch is not None else None,
            sync=sync,
        )
        self.cache[video_id] = record
        return record

    def _query_indices(self, record: GazeRecord, frame_indices, vfps):
        mp4_ns = (np.asarray(frame_indices, dtype=np.float64) / max(float(vfps), 1e-6)) * 1e9
        if record.sync is not None and {"mp4_time_ns", "vrs_device_time_ns"}.issubset(record.sync.columns):
            vrs_ns = np.interp(
                mp4_ns,
                record.sync["mp4_time_ns"].to_numpy(dtype=np.float64),
                record.sync["vrs_device_time_ns"].to_numpy(dtype=np.float64),
            )
            q_us = vrs_ns / 1000.0
        else:
            q_us = mp4_ns / 1000.0

        ts = record.timestamps_us
        if len(ts) == 0:
            return None
        idx = np.searchsorted(ts, q_us)
        idx = np.clip(idx, 0, len(ts) - 1)
        idx2 = np.clip(idx - 1, 0, len(ts) - 1)
        choose = np.abs(ts[idx] - q_us) < np.abs(ts[idx2] - q_us)
        return np.where(choose, idx, idx2)

    def _query_crop_xy(self, record: GazeRecord, frame_indices, vfps, h0: int, w0: int):
        pick = self._query_indices(record, frame_indices, vfps)
        if pick is None:
            return None

        if record.yaw is not None and record.pitch is not None:
            yaw = record.yaw[pick]
            pitch = record.pitch[pick]
            h_half = np.radians(55.0)
            v_half = np.radians(45.0)
            xc = np.tan(np.clip(yaw, -1.4, 1.4)) / np.tan(h_half)
            yc = np.tan(np.clip(pitch, -1.2, 1.2)) / np.tan(v_half)
            u0 = (0.5 + 0.5 * np.clip(xc, -1.0, 1.0)) * (w0 - 1)
            v0 = (0.5 + 0.5 * np.clip(yc, -1.0, 1.0)) * (h0 - 1)
            u, v = _image_coords_after_resize_center_crop(u0, v0, h0, w0, self.crop_size)
            return np.stack(
                [
                    np.clip(u, 0, self.crop_size - 1),
                    np.clip(v, 0, self.crop_size - 1),
                ],
                axis=1,
            )

        if record.xy_norm is None:
            return None
        xy = record.xy_norm[pick]
        return np.stack([xy[:, 0] * (self.crop_size - 1), xy[:, 1] * (self.crop_size - 1)], axis=1)

    def _heatmap(self, xy_px, device):
        T = len(xy_px)
        H = W = self.crop_size
        yy = torch.arange(H, device=device).view(1, H, 1)
        xx = torch.arange(W, device=device).view(1, 1, W)
        x = torch.as_tensor(xy_px[:, 0], dtype=torch.float32, device=device).view(T, 1, 1)
        y = torch.as_tensor(xy_px[:, 1], dtype=torch.float32, device=device).view(T, 1, 1)
        heat = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * max(self.sigma_px, 1.0) ** 2))
        return heat / heat.amax().clamp(min=1e-6)

    def _importance(self, clips, metadata):
        device = clips.device
        rows = []
        for idx in range(clips.shape[0]):
            meta = metadata[idx] if isinstance(metadata, list) else metadata
            video_id = str(meta.get("video_id"))
            record = self._load_record(video_id)
            if record is None:
                rows.append(None)
                continue
            frame_indices = meta.get("frame_indices")
            if torch.is_tensor(frame_indices):
                frame_indices = frame_indices.detach().cpu().numpy()
            vfps = meta.get("vfps", 30.0)
            if torch.is_tensor(vfps):
                vfps = float(vfps.detach().cpu())
            h0 = int(meta.get("height", meta.get("H0", self.crop_size)))
            w0 = int(meta.get("width", meta.get("W0", self.crop_size)))
            xy = self._query_crop_xy(record, frame_indices, vfps, h0, w0)
            if xy is None:
                rows.append(None)
                continue
            heat = self._heatmap(xy, device=device).unsqueeze(0)
            T = heat.shape[1]
            Tuse = T - (T % self.tubelet_size)
            if Tuse <= 0:
                rows.append(None)
                continue
            heat = heat[:, :Tuse].view(1, Tuse // self.tubelet_size, self.tubelet_size, self.crop_size, self.crop_size).mean(dim=2)
            if self.use_motion:
                gray = clips[idx : idx + 1].float().mean(dim=1)
                diff = torch.zeros_like(gray)
                diff[:, 1:] = torch.abs(gray[:, 1:] - gray[:, :-1])
                diff = diff[:, :Tuse].view(1, Tuse // self.tubelet_size, self.tubelet_size, diff.shape[-2], diff.shape[-1]).mean(dim=2)
                diff = F.interpolate(diff, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
                diff = diff / diff.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
                heat = heat + self.motion_weight * diff
            grid = self.crop_size // self.patch_size
            patch = F.adaptive_avg_pool2d(heat.reshape(-1, 1, self.crop_size, self.crop_size), (grid, grid))
            rows.append(patch.view(1, -1))
        if all(row is None for row in rows):
            return None
        template = next((row for row in rows if row is not None), None)
        filled = []
        for row in rows:
            if row is None:
                filled.append(torch.ones(1, template.shape[1], device=device))
            else:
                filled.append(row)
        return torch.cat(filled, dim=0)

    def apply(self, tokens, clips, metadata):
        if not self.enabled or metadata is None:
            return tokens
        imp = self._importance(clips, metadata)
        if imp is None:
            return tokens
        if imp.shape[1] < tokens.shape[1]:
            pad = torch.ones(imp.shape[0], tokens.shape[1] - imp.shape[1], device=tokens.device, dtype=imp.dtype)
            imp = torch.cat([imp, pad], dim=1)
        elif imp.shape[1] > tokens.shape[1]:
            imp = imp[:, : tokens.shape[1]]
        imp = imp - imp.min(dim=1, keepdim=True).values
        imp = imp / imp.max(dim=1, keepdim=True).values.clamp(min=1e-6)
        gate = (1.0 - self.gamma) + self.gamma * imp
        return tokens * gate.unsqueeze(-1).to(tokens.dtype)


class PredictionDumper:
    def __init__(self, cfg: dict[str, Any], output_dir: str | os.PathLike | None, rank: int):
        self.enabled = bool(cfg.get("enabled", False)) and rank == 0
        self.topk = int(cfg.get("topk", 5))
        self.rows: list[dict[str, Any]] = []
        path = cfg.get("path")
        self.path = Path(path) if path else (Path(output_dir) / "val_predictions.csv" if output_dir else None)

    def add_batch(self, udata, outputs, labels, class_maps):
        if not self.enabled:
            return
        metadata = udata[3] if len(udata) > 4 else []
        out = outputs[0]
        for i in range(out["action"].shape[0]):
            row = {}
            meta = metadata[i] if isinstance(metadata, list) and i < len(metadata) else {}
            for key, value in meta.items():
                row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
            row.update(
                {
                    "verb_raw": int(udata[1][i]),
                    "noun_raw": int(udata[2][i]),
                    "verb_label": int(labels["verb"][i]),
                    "noun_label": int(labels["noun"][i]),
                    "action_label": int(labels["action"][i]),
                }
            )
            for name in ["verb", "noun", "action"]:
                logits = torch.sigmoid(out[name][i].float())
                k = min(self.topk, logits.numel())
                scores, preds = logits.topk(k)
                pred_list = [int(x) for x in preds.detach().cpu().tolist()]
                row[f"{name}_top{self.topk}"] = json.dumps(pred_list)
                row[f"{name}_scores_top{self.topk}"] = json.dumps([float(x) for x in scores.detach().cpu().tolist()])
                row[f"{name}_top1"] = pred_list[0] if pred_list else -1
                row[f"{name}_top3_hit"] = int(int(labels[name][i]) in pred_list[:3])
                row[f"{name}_top5_hit"] = int(int(labels[name][i]) in pred_list[:5])
            self.rows.append(row)

    def write(self):
        if not self.enabled or self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for row in self.rows for k in row.keys()})
        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        self._write_summary()
        logger.info("Wrote validation prediction dump: %s", self.path)

    def _write_summary(self):
        for name in ["verb", "noun", "action"]:
            counts = Counter()
            for row in self.rows:
                label = row.get(f"{name}_label")
                counts[(label, "hit")] += int(row.get(f"{name}_top3_hit", 0))
                counts[(label, "miss")] += 1 - int(row.get(f"{name}_top3_hit", 0))
            summary = self.path.with_name(f"{self.path.stem}_{name}_summary.csv")
            with summary.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["label", "top3_hits", "top3_misses", "total", "top3_accuracy"])
                writer.writeheader()
                labels = sorted({label for label, _ in counts})
                for label in labels:
                    hits = counts[(label, "hit")]
                    misses = counts[(label, "miss")]
                    total = hits + misses
                    writer.writerow(
                        {
                            "label": label,
                            "top3_hits": hits,
                            "top3_misses": misses,
                            "total": total,
                            "top3_accuracy": 100.0 * hits / max(1, total),
                        }
                    )


def labels_from_udata(udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes):
    if action_is_verb_noun:
        _verbs, _nouns = udata[1], udata[2]
        verb_labels, noun_labels, action_labels = [], [], []
        for v, n in zip(_verbs, _nouns):
            verb_labels.append(verb_classes[int(v)])
            noun_labels.append(noun_classes[int(n)])
            action_labels.append(action_classes[(int(v), int(n))])
        dtype = _verbs.dtype
        return {
            "verb": torch.tensor(verb_labels, device=device).to(dtype),
            "noun": torch.tensor(noun_labels, device=device).to(dtype),
            "action": torch.tensor(action_labels, device=device).to(dtype),
        }
    _actions = udata[1]
    action_labels = [action_classes[str(int(a))] for a in _actions]
    return {"action": torch.tensor(action_labels, device=device).to(_actions.dtype)}


def train_one_epoch_with_gaze(
    base_eval,
    gate: GazeTokenGate,
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
    traj_loader=None,
):
    _data_loader = iter(data_loader)
    for c in classifiers:
        c.train(mode=True)
    if action_is_verb_noun:
        verb_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]
    data_elapsed_time_meter = AverageMeter()
    try:
        max_train_iters = int(os.environ.get("EVAL_MAX_TRAIN_ITERS", os.environ.get("MAX_TRAIN_ITERS", "0")) or "0")
    except ValueError:
        max_train_iters = 0
    if max_train_iters > 0 and max_train_iters < ipe:
        logger.info("Limiting train_one_epoch_with_gaze to %d/%d iterations via EVAL_MAX_TRAIN_ITERS", max_train_iters, ipe)
        ipe = max_train_iters

    for itr in range(ipe):
        itr_start_time = time.time()
        fusion_monitor = {}
        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)
        [s.step() for s in scheduler]
        [wds_.step() for wds_ in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = udata[0].to(device)
            anticipation_times = udata[-1].to(device)
            labels = labels_from_udata(udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes)
            data_elapsed_time_meter.update((time.time() - itr_start_time) * 1000.0)
            metadata = udata[3] if len(udata) > 4 else None
            with torch.no_grad():
                tokens = model(clips, anticipation_times)
                tokens = gate.apply(tokens, clips, metadata)
            if traj_loader is not None and metadata is not None:
                from app.hdepic_lora_action_anticipation.gaze_rnn import (
                    call_classifier,
                    encode_gaze_tokens,
                    gaze_fusion_monitor,
                    load_gaze_batch,
                )

                gaze_batch = load_gaze_batch(metadata, traj_loader, device, video_tokens=tokens)
                outputs = []
                for idx, c in enumerate(classifiers):
                    gaze_tokens = encode_gaze_tokens(c, metadata, traj_loader, device, gaze_batch=gaze_batch)
                    outputs.append(call_classifier(c, tokens, gaze_tokens))
                    if idx == 0:
                        fusion_monitor = gaze_fusion_monitor(c)
            else:
                outputs = [c(tokens) for c in classifiers]

        if action_is_verb_noun:
            loss = [
                criterion(o["verb"], labels["verb"])
                + criterion(o["noun"], labels["noun"])
                + criterion(o["action"], labels["action"])
                for o in outputs
            ]
        else:
            loss = [criterion(o["action"], labels["action"]) for o in outputs]
        if traj_loader is not None:
            from app.hdepic_lora_action_anticipation.gaze_rnn import clip_gaze_encoder_grads

            grad_clip = float(os.environ.get("GAZE_RNN_GRAD_CLIP", "1.0"))
        else:
            clip_gaze_encoder_grads = None
            grad_clip = 0.0
        if use_bfloat16:
            for l, s, o, c in zip(loss, scaler, optimizer, classifiers):
                if not torch.isfinite(l.detach()):
                    logger.warning("Skipping optimizer step because gaze loss is non-finite: %s", float(l.detach().float()))
                    o.zero_grad()
                    continue
                s.scale(l).backward()
                if clip_gaze_encoder_grads is not None:
                    s.unscale_(o)
                    if not clip_gaze_encoder_grads(c, max_norm=grad_clip):
                        o.zero_grad()
                        s.update()
                        continue
                s.step(o)
                s.update()
        else:
            for L, o, c in zip(loss, optimizer, classifiers):
                if not torch.isfinite(L.detach()):
                    logger.warning("Skipping optimizer step because gaze loss is non-finite: %s", float(L.detach().float()))
                    o.zero_grad()
                    continue
                L.backward()
                if clip_gaze_encoder_grads is not None and not clip_gaze_encoder_grads(c, max_norm=grad_clip):
                    o.zero_grad()
                    continue
                o.step()
        [o.zero_grad() for o in optimizer]

        with torch.no_grad():
            action_metrics = [m(o["action"], labels["action"]) for o, m in zip(outputs, action_metric_loggers)]
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], labels["verb"]) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], labels["noun"]) for o, m in zip(outputs, noun_metric_loggers)]
        if itr % 10 == 0 or itr == ipe - 1:
            fusion_text = ""
            if fusion_monitor:
                fusion_text = " [fusion: " + " ".join(f"{k}={v:.4f}" for k, v in sorted(fusion_monitor.items())) + "]"
            if action_is_verb_noun:
                logger.info(
                    "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) [mem: %.2e] [data: %.1f ms]%s",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(v["accuracy"] for v in verb_metrics),
                    max(n["accuracy"] for n in noun_metrics),
                    max(a["recall"] for a in action_metrics),
                    max(v["recall"] for v in verb_metrics),
                    max(n["recall"] for n in noun_metrics),
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    data_elapsed_time_meter.avg,
                    fusion_text,
                )

    ret = {"action": {"accuracy": max(a["accuracy"] for a in action_metrics), "recall": max(a["recall"] for a in action_metrics)}}
    if action_is_verb_noun:
        ret.update(
            {
                "verb": {"accuracy": max(v["accuracy"] for v in verb_metrics), "recall": max(v["recall"] for v in verb_metrics)},
                "noun": {"accuracy": max(n["accuracy"] for n in noun_metrics), "recall": max(n["recall"] for n in noun_metrics)},
            }
        )
    return ret


@torch.no_grad()
def validate_with_gaze(
    base_eval,
    gate: GazeTokenGate,
    dumper: PredictionDumper,
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    data_loader,
    use_bfloat16,
    valid_nouns,
    valid_verbs,
    valid_actions,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
    traj_loader=None,
    hidden_dump=None,
):
    logger.info("Running val with project-local gaze/dump hooks...")
    _data_loader = iter(data_loader)
    for c in classifiers:
        c.train(mode=False)
    if action_is_verb_noun:
        verb_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]

    for itr in range(ipe):
        fusion_monitor = {}
        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = udata[0].to(device)
            anticipation_times = udata[-1].to(device)
            labels = labels_from_udata(udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes)
            metadata = udata[3] if len(udata) > 4 else None
            tokens = model(clips, anticipation_times)
            tokens = gate.apply(tokens, clips, metadata)
            if traj_loader is not None and metadata is not None:
                from app.hdepic_lora_action_anticipation.gaze_rnn import (
                    call_classifier,
                    encode_gaze_tokens,
                    gaze_fusion_monitor,
                    load_gaze_batch,
                )

                gaze_batch = load_gaze_batch(metadata, traj_loader, device, video_tokens=tokens)
                outputs = []
                for idx, c in enumerate(classifiers):
                    gaze_tokens = encode_gaze_tokens(c, metadata, traj_loader, device, gaze_batch=gaze_batch)
                    outputs.append(call_classifier(c, tokens, gaze_tokens))
                    if idx == 0:
                        fusion_monitor = gaze_fusion_monitor(c)
                    if hidden_dump is not None and idx == 0:
                        hidden_dump.add(c, metadata, gaze_tokens)
            else:
                outputs = [c(tokens) for c in classifiers]
            action_metrics = [m(o["action"], labels["action"], valid_actions) for o, m in zip(outputs, action_metric_loggers)]
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], labels["verb"], valid_verbs) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], labels["noun"], valid_nouns) for o, m in zip(outputs, noun_metric_loggers)]
                verb_loss = sum(criterion(o["verb"], labels["verb"]) for o in outputs)
                noun_loss = sum(criterion(o["noun"], labels["noun"]) for o in outputs)
                action_loss = sum(criterion(o["action"], labels["action"]) for o in outputs)
                loss = verb_loss + noun_loss + action_loss
            else:
                loss = sum(criterion(o["action"], labels["action"]) for o in outputs)
        dumper.add_batch(udata, outputs, labels, {"verb": verb_classes, "noun": noun_classes, "action": action_classes})
        if itr % 10 == 0 or itr == ipe - 1:
            fusion_text = ""
            if fusion_monitor:
                fusion_text = " [fusion: " + " ".join(f"{k}={v:.4f}" for k, v in sorted(fusion_monitor.items())) + "]"
            if action_is_verb_noun:
                logger.info(
                    "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) loss (v/n): %.3f (%.3f %.3f) [mem: %.2e]%s",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(v["accuracy"] for v in verb_metrics),
                    max(n["accuracy"] for n in noun_metrics),
                    max(a["recall"] for a in action_metrics),
                    max(v["recall"] for v in verb_metrics),
                    max(n["recall"] for n in noun_metrics),
                    loss,
                    verb_loss,
                    noun_loss,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    fusion_text,
                )
    dumper.write()
    if hidden_dump is not None:
        hidden_dump.flush()
    ret = {"action": {"accuracy": max(a["accuracy"] for a in action_metrics), "recall": max(a["recall"] for a in action_metrics)}}
    if action_is_verb_noun:
        ret.update(
            {
                "verb": {"accuracy": max(v["accuracy"] for v in verb_metrics), "recall": max(v["recall"] for v in verb_metrics)},
                "noun": {"accuracy": max(n["accuracy"] for n in noun_metrics), "recall": max(n["recall"] for n in noun_metrics)},
            }
        )
    return ret
