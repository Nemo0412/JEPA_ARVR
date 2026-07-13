#!/usr/bin/env python
"""Regression tests for fp32 checkpoint scaler save/load (job 10742531)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from app.hdepic_lora_action_anticipation.checkpoint_utils import (
    restore_checkpoint_scalers,
    serialize_checkpoint_scalers,
)


class LegacySaveCheckpoint:
    """Minimal replica of the pre-fix upstream save path."""

    @staticmethod
    def broken(scaler):
        return None if scaler is None else [s.state_dict() for s in scaler]


class TestCheckpointScalerUtils(unittest.TestCase):
    def test_none_scaler_serializes_to_none(self):
        self.assertIsNone(serialize_checkpoint_scalers(None))

    def test_empty_scaler_list_serializes_to_none(self):
        self.assertIsNone(serialize_checkpoint_scalers([]))

    def test_fp32_placeholder_list_does_not_crash(self):
        states = serialize_checkpoint_scalers([None])
        self.assertIsNone(states)

    def test_legacy_save_path_crashed_on_fp32_placeholder(self):
        with self.assertRaises(AttributeError):
            LegacySaveCheckpoint.broken([None])

    def test_bf16_scaler_round_trip(self):
        scaler = torch.cuda.amp.GradScaler()
        states = serialize_checkpoint_scalers([scaler])
        self.assertIsNotNone(states)
        self.assertEqual(len(states), 1)

        restored = torch.cuda.amp.GradScaler()
        restore_checkpoint_scalers([restored], states)
        self.assertEqual(restored.state_dict(), states[0])

    def test_fp32_restore_is_noop(self):
        restore_checkpoint_scalers([], None)
        restore_checkpoint_scalers([], [])
        restore_checkpoint_scalers(None, None)

    def test_mixed_none_and_live_scaler_serializes_live_only(self):
        live = torch.cuda.amp.GradScaler()
        states = serialize_checkpoint_scalers([None, live, None])
        self.assertEqual(len(states), 1)
        restore_checkpoint_scalers([live], states)

    def test_save_dict_regression_10742531(self):
        """End-to-end: fp32 init_opt placeholder must not break torch.save payload."""
        optimizer = torch.optim.AdamW([torch.nn.Linear(4, 2).weight], lr=1e-4)
        classifier = torch.nn.Linear(4, 2)
        scaler = [None]  # historical fp32 return value

        payload = {
            "classifiers": [classifier.state_dict()],
            "opt": [optimizer.state_dict()],
            "scaler": serialize_checkpoint_scalers(scaler),
            "epoch": 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "latest.pt"
            torch.save(payload, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)

        self.assertIsNone(loaded["scaler"])
        restore_checkpoint_scalers([], loaded["scaler"])


if __name__ == "__main__":
    unittest.main()
