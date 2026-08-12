#!/usr/bin/env python3
"""Smoke: Ego4D-style gaze-only CA (KV=gaze, no IMU) shapes + finite outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.hdepic_lora_action_anticipation.binary_input_adapter import BinaryMapInputAdapter  # noqa: E402
from app.hdepic_lora_action_anticipation.concat_plus_cross_attn import (  # noqa: E402
    ConcatPlusCrossAttnAdaptedModel,
)
from app.hdepic_lora_action_anticipation.tri_modal_fusion import (  # noqa: E402
    GazeSpatialEncoder,
    ProjectedTriModalCrossAttention,
    compute_token_budgets,
)


class _FakeCore(torch.nn.Module):
    """Minimal encoder/predictor stand-in for shape checks."""

    def __init__(self, embed_dim=64, grid_size=16, tubelet=2, fps=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.tubelet_size = tubelet
        self.frames_per_second = fps
        self.num_output_frames = 2
        self.num_steps = 1
        self.no_predictor = False
        self.no_encoder = False
        self.return_mode = "observed_plus_target"
        n = grid_size * grid_size
        self.encoder = torch.nn.Linear(3, embed_dim)  # unused; we override forward
        self.encoder.embed_dim = embed_dim
        self._n = n

        class _Pred(torch.nn.Module):
            def forward(self, x, masks_x=None, masks_y=None):
                b = x.size(0)
                n_pred = int(grid_size * grid_size * (2 // tubelet))
                return torch.randn(b, n_pred, embed_dim, device=x.device, dtype=x.dtype)

        self.predictor = _Pred()

    def encoder_forward_override(self, clips):
        b, _, t, _, _ = clips.shape
        slots = max(1, t // self.tubelet_size)
        n = self.grid_size * self.grid_size * slots
        return torch.randn(b, n, self.embed_dim, device=clips.device, dtype=clips.dtype)


def main():
    torch.manual_seed(0)
    D, G = 64, 16
    B, T, H, W = 1, 16, 32, 32
    n_v, n_g, _ = compute_token_budgets(G * G, gaze_grid_size=10, gaze_token_ratio=0.5, imu_token_ratio=0.1)

    core = _FakeCore(embed_dim=D, grid_size=G)

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = D

        def forward(self, clips):
            return core.encoder_forward_override(clips)

    core.encoder = Enc()

    adapter = BinaryMapInputAdapter(
        hidden_dim=8, scale=1.0, temporal_kernel=1, binary_center=0.0, residual_clamp=1.0, in_channels=4
    )
    fusion = ProjectedTriModalCrossAttention(
        embed_dim=D,
        attn_dim=D,
        num_heads=4,
        num_layers=1,
        use_gated_residual=True,
        use_gaze_branch=True,
        use_imu_branch=False,
        gate_bias_init=-2.0,
    )
    gaze_enc = GazeSpatialEncoder(embed_dim=D, grid_size=10)
    model = ConcatPlusCrossAttnAdaptedModel(
        core,
        input_adapter=adapter,
        fusion=fusion,
        gaze_encoder=gaze_enc,
        fusion_cfg={
            "use_gaze_branch": True,
            "use_imu_branch": False,
            "keep_aux_tokens_in_predictor": True,
            "gaze_grid_size": 10,
            "gaze_token_ratio": 0.5,
            "imu_token_ratio": 0.1,
        },
        ca_aux="gaze",
    )

    clips = torch.randn(B, 3, T, H, W)
    gaze = torch.rand(B, 1, T, H, W)
    ant = torch.full((B,), 2.0)
    out = model(clips, ant, gaze_map=gaze)
    assert out is not None and torch.isfinite(out).all(), "non-finite or None output"
    assert model.imu_encoder is None
    assert model.gaze_encoder is not None
    assert model.fusion.use_gaze_branch and not model.fusion.use_imu_branch
    print(
        f"PASS gaze-only CA: out={tuple(out.shape)} n_v_spatial={n_v} n_gaze={n_g} "
        f"ca_aux={model.ca_aux}"
    )


if __name__ == "__main__":
    main()
