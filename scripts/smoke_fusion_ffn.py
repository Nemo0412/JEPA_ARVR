#!/usr/bin/env python3
"""Smoke: fusion CA + optional FFN shapes and near-identity when FFN last-layer zeroed."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.hdepic_lora_action_anticipation.tri_modal_fusion import (  # noqa: E402
    ProjectedTriModalCrossAttention,
)


def main():
    torch.manual_seed(0)
    B, T, Nv, Na, D = 2, 2, 256, 26, 64
    z_v = torch.randn(B, T, Nv, D)
    z_i = torch.randn(B, T, Na, D)

    for use_ffn in (False, True):
        m = ProjectedTriModalCrossAttention(
            embed_dim=D,
            attn_dim=D,
            num_heads=4,
            num_layers=2,
            use_gated_residual=True,
            use_gaze_branch=False,
            use_imu_branch=True,
            gate_bias_init=-2.0,
            video_query_side="video",
            use_ffn=use_ffn,
            ffn_mult=4,
        )
        # Keep W_o zero → CA residual ~0; with FFN last zero → near identity
        out_v, _, _ = m(z_v, None, z_i)
        assert out_v.shape == z_v.shape
        delta = (out_v - z_v).abs().max().item()
        print(f"[ok] use_ffn={use_ffn} max|ΔZ_v|={delta:.2e} (expect ~0 with W_o+FFN-last zero)")
        assert delta < 1e-4, delta

        # Break identity: xavier FFN last
        if use_ffn:
            with torch.no_grad():
                torch.nn.init.xavier_uniform_(m.layers[0].ffn[3].weight)
            out2, _, _ = m(z_v, None, z_i)
            assert not torch.allclose(out2, z_v, atol=1e-4)
            print("[ok] FFN can change Z_v when last Linear is non-zero")

    print("ALL SMOKE PASS")


if __name__ == "__main__":
    main()
