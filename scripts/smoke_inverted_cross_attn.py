#!/usr/bin/env python3
"""Smoke test: video_query_side=aux (Q=aux KV=video + attn^T writeback) shape / residual."""
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
    B, T, Nv, Na, D = 2, 3, 256, 26, 64
    z_v = torch.randn(B, T, Nv, D)
    z_i = torch.randn(B, T, Na, D)

    for side in ("video", "aux"):
        m = ProjectedTriModalCrossAttention(
            embed_dim=D,
            attn_dim=D,
            num_heads=4,
            num_layers=2,
            use_gated_residual=True,
            use_gaze_branch=False,
            use_imu_branch=True,
            gate_bias_init=-2.0,
            video_query_side=side,
        )
        # Non-zero out_proj so residual can move (default zero-init leaves near-identity)
        with torch.no_grad():
            nn_init = torch.nn.init
            nn_init.xavier_uniform_(m.video_proj.w_o.weight)
            nn_init.zeros_(m.video_proj.w_o.bias)

        out_v, out_g, out_i = m(z_v, z_gaze=None, z_imu=z_i)
        assert out_g is None
        assert out_v.shape == z_v.shape, (side, out_v.shape)
        assert out_i is not None and out_i.shape == z_i.shape
        delta = (out_v - z_v).abs().mean().item()
        print(f"[ok] side={side:5s} out_v={tuple(out_v.shape)} mean|ΔZ_v|={delta:.6f}")

    # inverted path must be able to change Z_v (not identity when W_o non-zero)
    m = ProjectedTriModalCrossAttention(
        embed_dim=D,
        attn_dim=D,
        num_heads=4,
        num_layers=1,
        use_gated_residual=False,
        use_gaze_branch=False,
        use_imu_branch=True,
        video_query_side="aux",
    )
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(m.video_proj.w_o.weight)
        torch.nn.init.zeros_(m.video_proj.w_o.bias)
    out_v, _, _ = m(z_v, None, z_i)
    assert not torch.allclose(out_v, z_v, atol=1e-5), "inverted writeback left Z_v unchanged"
    print("[ok] inverted residual changes Z_v")
    print("ALL SMOKE PASS")


if __name__ == "__main__":
    main()
