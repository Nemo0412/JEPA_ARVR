"""
HD-EPIC gaze -> spatiotemporal importance for ViT token gating.

Ground-truth pixel-level gaze heatmaps are in the official slam-gaze release (~349 GB, separate download).
This module combines:
  1) Motion saliency: frame-difference magnitude (from RGB only)
  2) Temporal prior: gaze priming frames from eye-gaze-priming JSON

Mapped to the ViT patch grid and aligned with the encoder token count for multiplicative gating.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def load_priming(peaks_by_video_path: Optional[str]) -> Dict[str, List[int]]:
    """
    priming_info.json: video_id -> object_id -> start/end -> prime_stats -> frame_primed
    """
    peaks: Dict[str, List[int]] = {}
    if not peaks_by_video_path or not os.path.isfile(peaks_by_video_path):
        return peaks
    with open(peaks_by_video_path, "r") as f:
        root = json.load(f)
    for vid, objs in root.items():
        plist: List[int] = []
        if not isinstance(objs, dict):
            continue
        for _, blob in objs.items():
            for side in ("start", "end"):
                if side not in blob:
                    continue
                ps = blob[side].get("prime_stats")
                if not ps:
                    continue
                fp = ps.get("frame_primed")
                if fp is not None:
                    plist.append(int(fp))
        peaks[vid] = sorted(set(plist))
    return peaks


def temporal_prior_from_frames(
    frame_indices: np.ndarray | Sequence[int],
    peak_frames: List[int],
    sigma_frames: float = 12.0,
) -> np.ndarray:
    """Per-frame Gaussian weight in [0.35, 1.0+]; returns all-ones when no priming frames exist."""
    fi = np.asarray(frame_indices, dtype=np.float64)
    if not peak_frames:
        return np.ones(len(fi), dtype=np.float32)
    w = []
    sigma = max(sigma_frames, 1.0)
    for f in fi:
        m = 0.0
        for p in peak_frames:
            m = max(m, math.exp(-0.5 * ((f - p) / sigma) ** 2))
        w.append(max(0.35, float(m)))  # floor to avoid zeroing everything out
    return np.asarray(w, dtype=np.float32)


def motion_saliency_tubes(
    gray_bt_hw: torch.Tensor,
    tubelet_size: int = 2,
) -> torch.Tensor:
    """
    gray: [B, T, H, W]
    Returns [B, T//tubelet, H, W]: average frame-diff magnitude per temporal tubelet.
    """
    B, T, H, W = gray_bt_hw.shape
    d = torch.zeros_like(gray_bt_hw)
    d[:, 1:] = torch.abs(gray_bt_hw[:, 1:] - gray_bt_hw[:, :-1])
    Tuse = T - (T % tubelet_size)
    d = d[:, :Tuse]
    return d.view(B, Tuse // tubelet_size, tubelet_size, H, W).mean(dim=2)


def patch_importance_from_maps(
    imp_b_t_hw: torch.Tensor,
    *,
    spatial_size_hw: Tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    """
    imp_b_t_hw: [B, Ttube, H, W], aligned with ViT temporal tubelets
    Spatially pooled to patch grid (h_p, w_p), then flattened to [B, Ttube*h_p*w_p]
    """
    B, tt, _, _ = imp_b_t_hw.shape
    h_p, w_p = spatial_size_hw[0] // patch_size, spatial_size_hw[1] // patch_size
    x = torch.nn.functional.adaptive_avg_pool2d(imp_b_t_hw.reshape(B * tt, 1, imp_b_t_hw.shape[2], imp_b_t_hw.shape[3]), (h_p, w_p))
    x = x.view(B, tt, h_p, w_p).reshape(B, -1)
    return x


def gate_encoder_tokens(
    feat_bnd: torch.Tensor,
    imp_bn: torch.Tensor,
    *,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """feat [B,N,D], imp [B,N] -> multiply by (constant offset + normalised imp)."""
    imp = imp_bn - imp_bn.min(dim=1, keepdim=True)[0]
    imp = imp / (imp.max(dim=1, keepdim=True)[0] + eps)
    g = (1.0 - gamma) + gamma * imp
    return feat_bnd * g.unsqueeze(-1)


def splat_gaze_heatmap(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    H: int,
    W: int,
    sigma: float = 36.0,
) -> np.ndarray:
    """Splat a Gaussian at (u,v) for each frame; returns [T,H,W] float32."""
    T = len(u_coords)
    out = np.zeros((T, H, W), dtype=np.float32)
    yy, xx = np.ogrid[0:H, 0:W]
    sigma = max(sigma, 1.0)
    for t in range(T):
        u, v = float(u_coords[t]), float(v_coords[t])
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        out[t] = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma * sigma))
    m = out.max()
    if m > 1e-6:
        out /= m
    return out


def image_coords_after_resize_center_crop(
    u0: np.ndarray,
    v0: np.ndarray,
    H0: int,
    W0: int,
    out_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Matches V-JEPA video eval: short_side = 256/224*out_size then CenterCrop(out_size)."""
    short_side = int(256.0 / 224 * out_size)
    scale = short_side / float(min(H0, W0))
    new_w = int(round(W0 * scale))
    new_h = int(round(H0 * scale))
    u1 = u0.astype(np.float64) * scale
    v1 = v0.astype(np.float64) * scale
    left = (new_w - out_size) // 2
    top = (new_h - out_size) // 2
    return u1 - left, v1 - top


def build_gaze_maps_for_indices(
    frame_indices: np.ndarray,
    vfps: float,
    sync_df: pd.DataFrame,
    gaze_df: pd.DataFrame,
    H0: int,
    W0: int,
    out_size: int = 256,
    sigma_px: float = 36.0,
) -> np.ndarray:
    """
    Maps MP4 frame index -> VRS time -> nearest gaze sample; projects yaw/pitch to image coords
    after Resize+CenterCrop, returns [T, out_size, out_size].
    """
    mp4_ns = (frame_indices.astype(np.float64) / vfps) * 1e9
    vrs_ns = np.interp(
        mp4_ns,
        sync_df["mp4_time_ns"].values.astype(np.float64),
        sync_df["vrs_device_time_ns"].values.astype(np.float64),
    )
    q_us = vrs_ns / 1000.0

    gz = gaze_df.sort_values("tracking_timestamp_us").reset_index(drop=True)
    ts = gz["tracking_timestamp_us"].values.astype(np.float64)
    idx = np.searchsorted(ts, q_us)
    idx = np.clip(idx, 0, len(ts) - 1)
    idx2 = np.clip(idx - 1, 0, len(ts) - 1)
    choose = np.abs(ts[idx] - q_us) < np.abs(ts[idx2] - q_us)
    pick = np.where(choose, idx, idx2)
    rows = gz.iloc[pick]

    yaw = (rows["left_yaw_rads_cpf"].values + rows["right_yaw_rads_cpf"].values) * 0.5
    pitch = rows["pitch_rads_cpf"].values
    h_half = np.radians(55.0)
    v_half = np.radians(45.0)
    xc = np.tan(np.clip(yaw, -1.4, 1.4)) / np.tan(h_half)
    yc = np.tan(np.clip(pitch, -1.2, 1.2)) / np.tan(v_half)
    u0 = (0.5 + 0.5 * np.clip(xc, -1.0, 1.0)) * (W0 - 1)
    v0 = (0.5 + 0.5 * np.clip(yc, -1.0, 1.0)) * (H0 - 1)

    u_c, v_c = image_coords_after_resize_center_crop(u0, v0, H0, W0, out_size=out_size)
    u_c = np.clip(u_c, 0, out_size - 1)
    v_c = np.clip(v_c, 0, out_size - 1)
    return splat_gaze_heatmap(u_c.astype(np.float32), v_c.astype(np.float32), out_size, out_size, sigma=sigma_px)
