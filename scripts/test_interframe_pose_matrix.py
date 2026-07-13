#!/usr/bin/env python3
"""Unit tests for inter-frame pose matrix packing."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

gaze_stub = types.ModuleType("app.hdepic_lora_action_anticipation.gaze")


class _GazeTokenGate:
    def __init__(self, cfg):
        self.sync_root = None


def _as_path(value):
    return Path(value) if value else None


def _clean_video_id(video_id):
    return str(video_id)


def _find_first(root, patterns):
    return None


gaze_stub.GazeTokenGate = _GazeTokenGate
gaze_stub._as_path = _as_path
gaze_stub._clean_video_id = _clean_video_id
gaze_stub._find_first = _find_first
sys.modules.setdefault("app.hdepic_lora_action_anticipation.gaze", gaze_stub)

adapter_stub = types.ModuleType("app.hdepic_lora_action_anticipation.binary_input_adapter")


class _BinaryGazeMapBuilder:
    pass


adapter_stub.BinaryGazeMapBuilder = _BinaryGazeMapBuilder
sys.modules.setdefault("app.hdepic_lora_action_anticipation.binary_input_adapter", adapter_stub)

from app.hdepic_lora_action_anticipation.pose_map_builder import rasterize_pose_matrix_to_patch
from app.hdepic_lora_action_anticipation.pose_slam import (
    PoseRecord,
    SlamPoseLoader,
    pad_or_truncate_pose_matrix,
)


def test_pad_or_truncate_pose_matrix():
    feats = np.arange(20, dtype=np.float32).reshape(5, 4)
    out = pad_or_truncate_pose_matrix(feats, k_max=8)
    assert out.shape == (8, 4)
    np.testing.assert_allclose(out[:5], feats)
    np.testing.assert_allclose(out[5:], 0.0)

    smoothed = pad_or_truncate_pose_matrix(feats, k_max=3)
    assert smoothed.shape == (3, 4)
    expected = np.stack([feats[:1].mean(axis=0), feats[1:3].mean(axis=0), feats[3:5].mean(axis=0)])
    np.testing.assert_allclose(smoothed, expected)


def test_rasterize_pose_matrix_to_patch():
    mat = np.arange(18, dtype=np.float32).reshape(6, 3)
    patch = rasterize_pose_matrix_to_patch(mat, patch_height=8, patch_width=4)
    assert patch.shape == (8, 4)
    np.testing.assert_allclose(patch[:6, :3], mat)
    np.testing.assert_allclose(patch[6:, :], 0.0)


def test_query_interframe_matrices_synthetic():
    loader = SlamPoseLoader({"feature_set": "pose_6d"}, gate=None)
    ts = np.linspace(0.0, 3100.0, 32)
    record = PoseRecord(
        timestamps_us=ts,
        translation=np.stack([ts, ts * 0.1, ts * 0.2], axis=1),
        quaternion=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (32, 1)),
        angular_vel=None,
        linear_vel=None,
        quality=None,
    )
    loader._load_clip_record = lambda meta: record  # type: ignore[method-assign]
    loader.frame_timestamps_us = lambda meta: np.array([0.0, 1000.0, 2000.0, 3000.0])  # type: ignore[method-assign]

    meta = {"video_id": "P01_test", "frame_indices": [0, 8, 16, 24], "vfps": 8.0}
    mats = loader.query_interframe_matrices(meta, k_max=4)
    assert mats is not None
    assert mats.shape == (4, 4, 9)
    assert np.count_nonzero(mats[0]) > 0
    assert np.count_nonzero(mats[1]) > 0
    assert np.count_nonzero(mats[2]) > 0
    assert np.count_nonzero(mats[3]) == 0


if __name__ == "__main__":
    test_pad_or_truncate_pose_matrix()
    test_rasterize_pose_matrix_to_patch()
    test_query_interframe_matrices_synthetic()
    print("ok")
