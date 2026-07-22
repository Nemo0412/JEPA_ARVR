"""Idea 1 hybrid: early 5ch gaze+pose concat + late IMU cross-attention.

Keeps the proven BinaryMapInputAdapter path (P01 concat SOTA 42.74%) as the
encoder backbone, then refines encoder tokens with IMU-only projected
cross-attention before the predictor — same fusion core as tri-modal, but
``use_gaze_branch=False`` because gaze already entered via the adapter.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation.binary_input_adapter import BinaryMapInputAdapter
from app.hdepic_lora_action_anticipation.tri_modal_fusion import (
    ImuTemporalEncoder,
    ProjectedTriModalCrossAttention,
    TriModalFusionAdaptedModel,
)


class ConcatPlusCrossAttnAdaptedModel(nn.Module):
    """5ch concat adapter → video encoder → IMU cross-attn → predictor AR."""

    def __init__(
        self,
        base_model: nn.Module,
        input_adapter: BinaryMapInputAdapter,
        fusion: ProjectedTriModalCrossAttention,
        imu_encoder: Optional[ImuTemporalEncoder],
        fusion_cfg: dict[str, Any],
    ):
        super().__init__()
        self.input_adapter = input_adapter
        # Reuse tri-modal AR / fuse path with gaze branch disabled.
        self.tri = TriModalFusionAdaptedModel(
            base_model,
            fusion=fusion,
            gaze_encoder=None,
            imu_encoder=imu_encoder,
            fusion_cfg={**dict(fusion_cfg), "use_gaze_branch": False, "use_imu_branch": True},
        )
        # Expose attributes expected by tri-modal train/val + sidecar savers.
        self.base_model = self.tri.base_model
        self.fusion = self.tri.fusion
        self.gaze_encoder = None
        self.imu_encoder = self.tri.imu_encoder
        self.fusion_cfg = self.tri.fusion_cfg
        self.embed_dim = int(base_model.embed_dim)

    def forward(
        self,
        clips: torch.Tensor,
        anticipation_times: torch.Tensor,
        gaze_map: Optional[torch.Tensor] = None,
        imu_batch: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        binary_map: Optional[torch.Tensor] = None,
    ):
        # Prefetcher reuses the tri-modal API and passes the 2ch gaze+pose map
        # as ``gaze_map``; accept either name.
        aux = binary_map if binary_map is not None else gaze_map
        if aux is not None:
            clips = self.input_adapter(clips, aux)
        return self.tri.forward(clips, anticipation_times, gaze_map=None, imu_batch=imu_batch)
