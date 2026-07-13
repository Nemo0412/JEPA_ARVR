#!/usr/bin/env python
"""Smoke test: sliding AR final_window returns fixed encoder-length tokens."""

from __future__ import annotations

import torch

from app.hdepic_lora_action_anticipation.modelcustom.vit_encoder_predictor_rollout import (
    AutoregressiveAnticipativeWrapper,
)


class _FakeEncoder(torch.nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.patch_size = 16
        self.tubelet_size = 2

    def forward(self, x):
        b = x.shape[0]
        # 2 temporal slots * 4 spatial tokens = 8 tokens (toy geometry)
        return torch.randn(b, 8, self.embed_dim, device=x.device, dtype=x.dtype)


class _FakePredictor(torch.nn.Module):
    def forward(self, x, masks_x=None, masks_y=None):
        n_pred = masks_y.shape[1]
        return torch.randn(x.shape[0], n_pred, x.shape[-1], device=x.device, dtype=x.dtype)


def test_final_window_keeps_context_length():
    wrapper = AutoregressiveAnticipativeWrapper(
        encoder=_FakeEncoder(),
        predictor=_FakePredictor(),
        frames_per_second=8,
        crop_size=32,
        patch_size=16,
        tubelet_size=2,
        num_output_frames=2,
        return_mode="final_window",
        max_rollout_steps=512,
    )
    clips = torch.randn(2, 3, 32, 32, 32)
    anticipation = torch.tensor([10.0, 10.0])
    out = wrapper(clips, anticipation)
    assert out.shape[0] == 2
    assert out.shape[1] == 8
    assert out.shape[2] == 8


def test_observed_plus_target_is_longer():
    wrapper = AutoregressiveAnticipativeWrapper(
        encoder=_FakeEncoder(),
        predictor=_FakePredictor(),
        frames_per_second=8,
        crop_size=32,
        patch_size=16,
        tubelet_size=2,
        num_output_frames=2,
        return_mode="observed_plus_target",
        max_rollout_steps=512,
    )
    clips = torch.randn(1, 3, 32, 32, 32)
    out = wrapper(clips, torch.tensor([10.0]))
    assert out.shape[1] > 8


def test_sliding_three_steps_calls_predictor_three_times():
    calls = {"n": 0}

    class _CountPredictor(_FakePredictor):
        def forward(self, x, masks_x=None, masks_y=None):
            calls["n"] += 1
            return super().forward(x, masks_x, masks_y)

    wrapper = AutoregressiveAnticipativeWrapper(
        encoder=_FakeEncoder(),
        predictor=_CountPredictor(),
        frames_per_second=8,
        crop_size=32,
        patch_size=16,
        tubelet_size=2,
        num_output_frames=2,
        num_steps=3,
        return_mode="final_window",
        max_rollout_steps=512,
    )
    clips = torch.randn(1, 3, 32, 32, 32)
    out = wrapper(clips, torch.tensor([10.0]))
    assert out.shape[1] == 8
    assert calls["n"] == 3


def test_sliding_three_steps_mixed_horizons_final_window():
    wrapper = AutoregressiveAnticipativeWrapper(
        encoder=_FakeEncoder(),
        predictor=_FakePredictor(),
        frames_per_second=8,
        crop_size=32,
        patch_size=16,
        tubelet_size=2,
        num_output_frames=2,
        num_steps=3,
        return_mode="final_window",
        max_rollout_steps=512,
    )
    clips = torch.randn(3, 3, 32, 32, 32)
    out = wrapper(clips, torch.tensor([8.0, 9.0, 10.0]))
    assert out.shape == (3, 8, 8)


if __name__ == "__main__":
    test_final_window_keeps_context_length()
    test_observed_plus_target_is_longer()
    test_sliding_three_steps_calls_predictor_three_times()
    test_sliding_three_steps_mixed_horizons_final_window()
    print("sliding_ar smoke tests passed")
