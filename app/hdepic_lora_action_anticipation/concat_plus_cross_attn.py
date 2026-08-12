"""Idea 1 hybrid: early binary-map concat + late projected cross-attention.

HD-EPIC default (``ca_aux='imu'``):
  5ch gaze+pose BinaryMapInputAdapter → encoder → IMU-only CA
  (``use_gaze_branch=False``, ``use_imu_branch=True``).

Ego4D / gaze-only (``ca_aux='gaze'``):
  4ch RGB+gaze adapter → encoder → Gaze tokens as K/V (no IMU concat)
  (``use_gaze_branch=True``, ``use_imu_branch=False``).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation.binary_input_adapter import BinaryMapInputAdapter
from app.hdepic_lora_action_anticipation.tri_modal_fusion import (
    GazeSpatialEncoder,
    ImuTemporalEncoder,
    ProjectedTriModalCrossAttention,
    TriModalFusionAdaptedModel,
)


class ConcatPlusCrossAttnAdaptedModel(nn.Module):
    """Binary-map concat adapter → video encoder → aux cross-attn → predictor AR."""

    def __init__(
        self,
        base_model: nn.Module,
        input_adapter: BinaryMapInputAdapter,
        fusion: ProjectedTriModalCrossAttention,
        imu_encoder: Optional[ImuTemporalEncoder] = None,
        gaze_encoder: Optional[GazeSpatialEncoder] = None,
        fusion_cfg: Optional[dict[str, Any]] = None,
        ca_aux: str = "imu",
    ):
        super().__init__()
        self.input_adapter = input_adapter
        self.ca_aux = str(ca_aux).lower().strip()
        if self.ca_aux not in ("imu", "gaze"):
            raise ValueError(f"ca_aux must be 'imu' or 'gaze', got {ca_aux!r}")

        cfg = dict(fusion_cfg or {})
        if self.ca_aux == "gaze":
            cfg["use_gaze_branch"] = True
            cfg["use_imu_branch"] = False
            if gaze_encoder is None:
                raise ValueError("ca_aux='gaze' requires gaze_encoder")
            imu_encoder = None
        else:
            cfg["use_gaze_branch"] = False
            cfg["use_imu_branch"] = True
            if imu_encoder is None:
                raise ValueError("ca_aux='imu' requires imu_encoder")
            gaze_encoder = None

        self.tri = TriModalFusionAdaptedModel(
            base_model,
            fusion=fusion,
            gaze_encoder=gaze_encoder,
            imu_encoder=imu_encoder,
            fusion_cfg=cfg,
        )
        # Expose attributes expected by tri-modal train/val + sidecar savers.
        self.base_model = self.tri.base_model
        self.fusion = self.tri.fusion
        self.gaze_encoder = self.tri.gaze_encoder
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
        # Prefetcher may pass the aux map as ``gaze_map`` or ``binary_map``.
        aux = binary_map if binary_map is not None else gaze_map
        if aux is not None:
            clips = self.input_adapter(clips, aux)

        if self.ca_aux == "gaze":
            # Late CA: Q=video, KV=gaze tokens. Pass 1ch gaze (take first aux ch).
            ca_gaze = None
            if aux is not None:
                # aux is [B, C, T, H, W]; GazeSpatialEncoder expects C=1.
                ca_gaze = aux[:, :1]
            return self.tri.forward(clips, anticipation_times, gaze_map=ca_gaze, imu_batch=None)

        # HD-EPIC: gaze already entered via adapter; CA uses IMU only.
        return self.tri.forward(clips, anticipation_times, gaze_map=None, imu_batch=imu_batch)
