"""
HD-EPIC camera pose encoder.

Currently available data (per-video MPS zip extracted):
  hand_tracking/wrist_and_palm_poses.csv
    -> left/right wrist + palm 3D positions in device frame (12 dims)
  eye_gaze/general_eye_gaze.csv
    -> left eye yaw/pitch (2 dims)
Total POSE_DIM = 14

Time alignment: mp4 frame index -> vrs_device_time_ns (sync CSV) -> tracking_timestamp_us (nearest neighbour)


PoseEncoder (nn.Module): [B, T, 14] -> frame MLP + temporal Transformer -> [B, T, D]
  Output tokens are concatenated to ViT tokens before AttentivePooler.

When no MPS data exists for a video_id, returns all-zeros; the model can still forward normally.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── MPS data root ───────────────────────────────────────────────────
MPS_EXTRACT_ROOT = Path("/scratch/ll5914/datasets/HD-EPIC/_gaze_extract")
SYNC_CSV_DIR = Path("/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos/P01")
POSE_DIM = 14   # 12 hand + 2 gaze


# ── Path helpers ─────────────────────────────────────────────────────
def _mps_dir(video_id: str) -> Path:
    return MPS_EXTRACT_ROOT / f"mps_{video_id}_vrs"


def _hand_csv(video_id: str) -> Path:
    return _mps_dir(video_id) / "hand_tracking" / "wrist_and_palm_poses.csv"


def _gaze_csv(video_id: str) -> Path:
    return _mps_dir(video_id) / "eye_gaze" / "general_eye_gaze.csv"


def _sync_csv(video_id: str) -> Path:
    return SYNC_CSV_DIR / f"{video_id}_mp4_to_vrs_time_ns.csv"


def has_pose_data(video_id: str) -> bool:
    return _hand_csv(video_id).is_file() and _gaze_csv(video_id).is_file() and _sync_csv(video_id).is_file()


# ── PoseLoader ────────────────────────────────────────────────────────
class PoseLoader:
    """
    Cache MPS CSVs per video_id; call get_pose_for_frames to obtain [T, POSE_DIM].
    Returns all-zeros for videos without MPS data.
    """

    def __init__(self):
        self._cache: Dict[str, Optional[dict]] = {}

    def _load(self, video_id: str) -> Optional[dict]:
        if not has_pose_data(video_id):
            return None
        sync = pd.read_csv(_sync_csv(video_id))
        hand = pd.read_csv(_hand_csv(video_id))
        gaze = pd.read_csv(_gaze_csv(video_id))
        # timestamps: vrs_device_time_ns -> us
        sync_us = sync["vrs_device_time_ns"].values / 1000.0
        sync_mp4_ns = sync["mp4_time_ns"].values
        hand_ts = hand["tracking_timestamp_us"].values.astype(np.float64)
        gaze_ts = gaze["tracking_timestamp_us"].values.astype(np.float64)
        return dict(sync_us=sync_us, sync_mp4_ns=sync_mp4_ns,
                    hand=hand, gaze=gaze, hand_ts=hand_ts, gaze_ts=gaze_ts)

    def _ensure(self, video_id: str):
        if video_id not in self._cache:
            self._cache[video_id] = self._load(video_id)

    def get_pose_for_frames(self, video_id: str, frame_indices: np.ndarray,
                             vfps: float) -> np.ndarray:
        """
        Returns [T, POSE_DIM] float32; all-zeros when no MPS data.
        frame_indices: video frame numbers (int array), shape [T]
        vfps: video frame rate
        """
        self._ensure(video_id)
        T = len(frame_indices)
        data = self._cache.get(video_id)
        out = np.zeros((T, POSE_DIM), dtype=np.float32)
        if data is None:
            return out

        # mp4 frame -> mp4_time_ns -> interpolate sync -> vrs_us
        frame_time_ns = (frame_indices.astype(np.float64) / vfps) * 1e9
        vrs_us = np.interp(frame_time_ns, data["sync_mp4_ns"], data["sync_us"])

        hand_df = data["hand"]
        gaze_df = data["gaze"]
        hand_ts = data["hand_ts"]
        gaze_ts = data["gaze_ts"]

        hand_cols_l = ["tx_left_wrist_device", "ty_left_wrist_device", "tz_left_wrist_device",
                       "tx_left_palm_device",  "ty_left_palm_device",  "tz_left_palm_device"]
        hand_cols_r = ["tx_right_wrist_device", "ty_right_wrist_device", "tz_right_wrist_device",
                       "tx_right_palm_device",  "ty_right_palm_device",  "tz_right_palm_device"]
        hand_vals_l = hand_df[hand_cols_l].values.astype(np.float32)   # [N_hand, 6]
        hand_vals_r = hand_df[hand_cols_r].values.astype(np.float32)
        hand_conf_l = hand_df["left_tracking_confidence"].values
        hand_conf_r = hand_df["right_tracking_confidence"].values

        gaze_yaw   = gaze_df["left_yaw_rads_cpf"].values.astype(np.float32)
        gaze_pitch = gaze_df["left_pitch_rads_cpf"].values.astype(np.float32)

        for i, t_us in enumerate(vrs_us):
            hi = int(np.argmin(np.abs(hand_ts - t_us)))
            gi = int(np.argmin(np.abs(gaze_ts - t_us)))
            # hand left
            if hand_conf_l[hi] > 0:
                out[i, 0:6] = hand_vals_l[hi]
            # hand right
            if hand_conf_r[hi] > 0:
                out[i, 6:12] = hand_vals_r[hi]
            # gaze
            gy, gp = float(gaze_yaw[gi]), float(gaze_pitch[gi])
            if not (math.isnan(gy) or math.isnan(gp)):
                out[i, 12] = gy
                out[i, 13] = gp

        return out   # [T, 14]


# ── PoseEncoder ───────────────────────────────────────────────────────
class PoseEncoder(nn.Module):
    """
    [B, T, POSE_DIM=14] -> [B, T, embed_dim] pose tokens
    Architecture: per-frame Linear -> sincos pos encoding -> 2-layer TransformerEncoder
    Tokens are concatenated to ViT tokens before AttentivePooler.

    When input is all-zeros, output is near-zero and does not interfere with the visual path.
    """

    def __init__(self, embed_dim: int, num_frames: int = 32,
                 hidden_dim: int = 128, num_tf_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.input_proj = nn.Sequential(
            nn.Linear(POSE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(encoder_layer, num_layers=num_tf_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Fixed sincos positional encoding (non-learnable)
        pe = self._sincos_pe(num_frames, embed_dim)
        self.register_buffer("pe", pe)   # [1, T, D]

        # Normalization stats (filled by init_pose_stats at runtime; defaults to 0/1)
        self.register_buffer("pose_mean", torch.zeros(POSE_DIM))
        self.register_buffer("pose_std",  torch.ones(POSE_DIM))

    @staticmethod
    def _sincos_pe(length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[:dim // 2])
        return pe

    def init_pose_stats(self, mean: np.ndarray, std: np.ndarray):
        """Optional: fill normalization stats from training set statistics."""
        self.pose_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.pose_std.copy_(torch.from_numpy(np.maximum(std, 1e-6).astype(np.float32)))

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        pose: [B, T, POSE_DIM]  (all-zeros -> near-zero output)
        returns: [B, T, embed_dim]
        """
        T = pose.size(1)
        # Normalise only non-zero frames to avoid shifting zero-padded frames
        mask_nonzero = (pose.abs().sum(-1, keepdim=True) > 1e-6).float()
        pose_n = (pose - self.pose_mean) / self.pose_std * mask_nonzero
        x = self.input_proj(pose_n)            # [B, T, D]
        x = x + self.pe[:, :T, :]             # pos encoding
        x = self.tf(x)
        return self.norm(x)                    # [B, T, D]
