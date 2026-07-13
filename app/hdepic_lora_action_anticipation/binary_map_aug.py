"""Aug-aware joint transform for RGB + binary gaze map.

Background:
The original binary-input-adapter path disables V-JEPA2 training augmentation
(`disable_train_aug=True`) because the binary map is built from gaze pixel
coordinates in the *center-cropped* coordinate system, while RGB undergoes
random_resized_crop + horizontal_flip + autoaug + random_erasing. The two
streams desynchronize geometrically under train aug. The 2026-05-25
`gazefixed-gradisolate` run reached val action Top-3 50.75 with this aug
disabled, but zero-channel val (gaze forced to zero) matched at 50.4 and the
trained adapter's output-projection norm remained ~1e-2 (near-identity), which
together suggest the gain came from disabled aug rather than from gaze use.

This module replays V-JEPA2's training augmentation with the same random
parameters applied to RGB and binary jointly for the geometric ops
(random_resized_crop + horizontal_flip) and to RGB only for the pixel ops
(autoaugment + normalize + random_erasing). Eval uses Resize + CenterCrop on
both streams, matching V-JEPA2's eval path.

The pipeline takes raw video frames in (T, H, W, C) uint8 and the raw-frame
gaze (u, v) pixel coordinates per frame. It returns (rgb_CTHW float32,
binary_CTHW float32) already aligned.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

from app.hdepic_lora_action_anticipation.binary_map_utils import normalize_map_type, rasterize_gaze_disk
from src.datasets.utils.video.randerase import RandomErasing
from src.datasets.utils.video import transforms as video_transforms

logger = logging.getLogger(__name__)


def query_resized_xy(gate, record, frame_indices, vfps: float, h0: int, w0: int, out_size: int):
    """Return gaze pixel coords in the *resize-only* coordinate system.

    V-JEPA2's eval pipeline is Resize(short=out_size*256/224) -> CenterCrop(out_size).
    Training augmentation operates on the resize-only stage (before crop). This
    helper returns (u, v) in that intermediate frame, so downstream geometric
    aug (RandomResizedCrop, CenterCrop, HorizontalFlip) can transform it
    jointly with the RGB clip.

    Returns:
        ndarray of shape (T, 2): (u, v) in the resize-only frame, or None.
    """
    pick = gate._query_indices(record, frame_indices, vfps)  # noqa: SLF001
    if pick is None:
        return None

    short_side = int(256.0 / 224.0 * out_size)
    scale = short_side / float(max(1, min(h0, w0)))

    if record.yaw is not None and record.pitch is not None:
        yaw = record.yaw[pick]
        pitch = record.pitch[pick]
        h_half = np.radians(55.0)
        v_half = np.radians(45.0)
        xc = np.tan(np.clip(yaw, -1.4, 1.4)) / np.tan(h_half)
        yc = np.tan(np.clip(pitch, -1.2, 1.2)) / np.tan(v_half)
        u0 = (0.5 + 0.5 * np.clip(xc, -1.0, 1.0)) * (w0 - 1)
        v0 = (0.5 + 0.5 * np.clip(yc, -1.0, 1.0)) * (h0 - 1)
    elif record.xy_norm is not None:
        xy = record.xy_norm[pick]
        u0 = xy[:, 0] * (w0 - 1)
        v0 = xy[:, 1] * (h0 - 1)
    else:
        return None

    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))
    return np.stack([u0 * scale, v0 * scale], axis=1).astype(np.float64), (new_h, new_w)


def _resize_short_side(buffer_uint8: np.ndarray, short_side: int) -> torch.Tensor:
    """Resize a (T, H, W, C) uint8 clip so min(H, W) == short_side.

    Returns float tensor (T, C, H', W') in [0, 1].
    """
    t, h, w, c = buffer_uint8.shape
    scale = short_side / float(max(1, min(h, w)))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    tensor = torch.from_numpy(buffer_uint8).permute(0, 3, 1, 2).contiguous().float() / 255.0
    return F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _build_binary_clip(
    xy_resized: np.ndarray,
    new_h: int,
    new_w: int,
    t_frames: int,
    radius_px: float,
    dtype: torch.dtype,
    map_type: str = "binary",
) -> torch.Tensor:
    """Draw gaze maps at xy_resized in a (1, T, new_h, new_w) clip."""
    binary = torch.zeros((1, t_frames, new_h, new_w), dtype=dtype)
    n = min(t_frames, xy_resized.shape[0])
    if n <= 0:
        return binary
    yy = torch.arange(new_h, dtype=dtype).view(1, new_h, 1)
    xx = torch.arange(new_w, dtype=dtype).view(1, 1, new_w)
    xy_t = torch.as_tensor(xy_resized[:n], dtype=dtype)
    x = xy_t[:, 0].view(n, 1, 1)
    y = xy_t[:, 1].view(n, 1, 1)
    binary[0, :n] = rasterize_gaze_disk(xx, yy, x, y, radius_px, map_type=map_type, dtype=dtype)
    return binary


def _crop_clip(clip_tchw: torch.Tensor, i: int, j: int, h: int, w: int) -> torch.Tensor:
    return clip_tchw[:, :, i : i + h, j : j + w]


def _resize_clip(clip_tchw: torch.Tensor, target_h: int, target_w: int, mode: str) -> torch.Tensor:
    """Resize a (T, C, H, W) clip; uses requested interpolation mode."""
    if mode == "nearest":
        return F.interpolate(clip_tchw, size=(target_h, target_w), mode="nearest")
    align = False if mode in ("bilinear", "bicubic") else None
    return F.interpolate(clip_tchw, size=(target_h, target_w), mode=mode, align_corners=align)


def _autoaugment_rgb(rgb_thwc_uint8_pil: list[Image.Image], autoaug) -> list[Image.Image]:
    return autoaug(rgb_thwc_uint8_pil)


class BinaryMapAwareTransform:
    """Joint train/eval transform for RGB and binary gaze maps.

    Train path (training=True):
        1. autoaugment (PIL) - RGB only, pre-resize
        2. resize short side to crop_size  (RGB + binary, joint geometry)
        3. random_resized_crop -> crop_size (RGB + binary, joint geometry)
        4. random horizontal flip p=0.5 (RGB + binary, joint geometry)
        5. RGB-only normalize
        6. RGB-only RandomErasing (probability=reprob)

    Eval path (training=False):
        1. resize short side to crop_size*256/224 (RGB + binary, joint geometry)
        2. center crop -> crop_size  (RGB + binary, joint geometry)
        3. RGB-only normalize

    Note: V-JEPA2's train path resizes implicitly via random_resized_crop's
    interpolate-to-target. We follow that and skip a separate pre-resize on
    the train side. Step 2 in the train list above is just the input shape
    that random_resized_crop operates on; the buffer passes through as-is.
    """

    def __init__(
        self,
        training: bool,
        crop_size: int,
        radius_px: float,
        map_type: str = "binary",
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        random_resize_scale=(0.08, 1.0),
        random_resize_aspect_ratio=(3.0 / 4.0, 4.0 / 3.0),
        random_horizontal_flip: bool = True,
        auto_augment: bool = True,
        reprob: float = 0.25,
    ):
        self.training = bool(training)
        self.crop_size = int(crop_size)
        self.radius_px = float(radius_px)
        self.map_type = normalize_map_type(map_type)
        self.normalize_mean = torch.tensor(normalize_mean).view(3, 1, 1, 1)
        self.normalize_std = torch.tensor(normalize_std).view(3, 1, 1, 1)
        self.random_resize_scale = tuple(random_resize_scale)
        self.random_resize_aspect_ratio = tuple(random_resize_aspect_ratio)
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.auto_augment = bool(auto_augment)
        self.reprob = float(reprob)
        self.eval_short_side = int(256.0 / 224.0 * self.crop_size)

        if self.training and self.auto_augment:
            self.autoaug_transform = video_transforms.create_random_augment(
                input_size=(self.crop_size, self.crop_size),
                auto_augment="rand-m7-n4-mstd0.5-inc1",
                interpolation="bicubic",
            )
        else:
            self.autoaug_transform = None

        if self.training and self.reprob > 0:
            self.erase_transform = RandomErasing(
                self.reprob,
                mode="pixel",
                max_count=1,
                num_splits=1,
                device="cpu",
            )
        else:
            self.erase_transform = None

    def _sample_rrc_params(self, height: int, width: int) -> tuple[int, int, int, int]:
        scale = self.random_resize_scale
        ratio = self.random_resize_aspect_ratio
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        for _ in range(10):
            target_area = random.uniform(*scale) * height * width
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(
        self,
        buffer_thwc_uint8: np.ndarray,
        xy_resized: np.ndarray | None,
        resized_hw: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            buffer_thwc_uint8: raw frames (T, H_raw, W_raw, C) uint8.
            xy_resized: gaze pixel coords (T, 2) in the resize-only frame
                (returned by ``query_resized_xy``). May be None when gaze is
                missing for this clip; then the gaze map is all zeros.
            resized_hw: (new_h, new_w) the resize-only frame dimensions (also
                from ``query_resized_xy``). Required when xy_resized is not
                None; if None we fall back to short-side resize using
                self.eval_short_side.

        Returns:
            rgb: (3, T, crop_size, crop_size) float32, normalized.
            binary: (1, T, crop_size, crop_size) float32, in {0, 1}.
        """
        t, h_raw, w_raw, _ = buffer_thwc_uint8.shape
        if not self.training:
            return self._eval(buffer_thwc_uint8, xy_resized, resized_hw)
        return self._train(buffer_thwc_uint8, xy_resized, resized_hw)

    def _eval(
        self,
        buffer_thwc_uint8: np.ndarray,
        xy_resized: np.ndarray | None,
        resized_hw: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t, h_raw, w_raw, _ = buffer_thwc_uint8.shape
        if resized_hw is None:
            short = self.eval_short_side
            rgb_resized = _resize_short_side(buffer_thwc_uint8, short)
            new_h, new_w = rgb_resized.shape[-2:]
        else:
            new_h, new_w = resized_hw
            tensor = (
                torch.from_numpy(buffer_thwc_uint8).permute(0, 3, 1, 2).contiguous().float() / 255.0
            )
            rgb_resized = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

        binary_resized = (
            _build_binary_clip(xy_resized, new_h, new_w, t, self.radius_px, rgb_resized.dtype, self.map_type)
            if xy_resized is not None
            else torch.zeros((1, t, new_h, new_w), dtype=rgb_resized.dtype)
        )
        binary_resized = binary_resized.permute(1, 0, 2, 3)  # T 1 H W

        i = (new_h - self.crop_size) // 2
        j = (new_w - self.crop_size) // 2
        rgb_crop = _crop_clip(rgb_resized, i, j, self.crop_size, self.crop_size)
        binary_crop = _crop_clip(binary_resized, i, j, self.crop_size, self.crop_size)

        rgb = rgb_crop.permute(1, 0, 2, 3).contiguous()  # C T H W
        rgb = (rgb - self.normalize_mean) / self.normalize_std
        binary = binary_crop.permute(1, 0, 2, 3).contiguous()  # 1 T H W
        return rgb, binary

    def _train(
        self,
        buffer_thwc_uint8: np.ndarray,
        xy_resized: np.ndarray | None,
        resized_hw: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t, h_raw, w_raw, _ = buffer_thwc_uint8.shape

        rgb_input = buffer_thwc_uint8
        if self.autoaug_transform is not None:
            pil_frames = [Image.fromarray(rgb_input[i]) for i in range(t)]
            pil_frames = self.autoaug_transform(pil_frames)
            rgb_input = np.stack([np.asarray(p) for p in pil_frames], axis=0)

        if resized_hw is None:
            new_h, new_w = h_raw, w_raw
            rgb_resized = (
                torch.from_numpy(rgb_input).permute(0, 3, 1, 2).contiguous().float() / 255.0
            )
        else:
            new_h, new_w = resized_hw
            tensor = (
                torch.from_numpy(rgb_input).permute(0, 3, 1, 2).contiguous().float() / 255.0
            )
            rgb_resized = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

        binary_resized = (
            _build_binary_clip(xy_resized, new_h, new_w, t, self.radius_px, rgb_resized.dtype, self.map_type)
            if xy_resized is not None
            else torch.zeros((1, t, new_h, new_w), dtype=rgb_resized.dtype)
        )
        binary_resized = binary_resized.permute(1, 0, 2, 3)  # T 1 H W

        i, j, h, w = self._sample_rrc_params(new_h, new_w)
        rgb_cropped = _crop_clip(rgb_resized, i, j, h, w)
        binary_cropped = _crop_clip(binary_resized, i, j, h, w)
        rgb_out = _resize_clip(rgb_cropped, self.crop_size, self.crop_size, mode="bilinear")
        binary_out = _resize_clip(binary_cropped, self.crop_size, self.crop_size, mode="nearest")

        if self.random_horizontal_flip and random.random() < 0.5:
            rgb_out = rgb_out.flip(-1)
            binary_out = binary_out.flip(-1)

        rgb = rgb_out.permute(1, 0, 2, 3).contiguous()  # C T H W
        rgb = (rgb - self.normalize_mean) / self.normalize_std

        if self.erase_transform is not None:
            rgb_t_first = rgb.permute(1, 0, 2, 3)  # T C H W
            rgb_t_first = self.erase_transform(rgb_t_first)
            rgb = rgb_t_first.permute(1, 0, 2, 3).contiguous()

        binary = binary_out.permute(1, 0, 2, 3).contiguous()  # 1 T H W
        return rgb, binary
