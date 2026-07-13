"""Attentive probe head for 3D motion regression (trained from scratch)."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.ref_egoexo4d_motion_prediction.joints import NUM_BODY_JOINTS
from src.models.attentive_pooler import AttentivePooler


class AttentiveMotionHead(nn.Module):
    """
    Mirrors HD-EPIC / EK100 AttentiveClassifier pooler settings, but regresses
    future 3D body joints instead of action logits.

    Optional past-motion conditioning follows the EgoAgent 3D motion prediction
    protocol (first ``context_motion_frames`` poses as extra input).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 16,
        num_probe_blocks: int = 4,
        num_joints: int = NUM_BODY_JOINTS,
        context_motion_frames: int = 5,
        future_motion_frames: int = 15,
        use_past_pose: bool = True,
        use_activation_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_joints = int(num_joints)
        self.context_motion_frames = int(context_motion_frames)
        self.future_motion_frames = int(future_motion_frames)
        self.use_past_pose = bool(use_past_pose)

        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=num_probe_blocks,
            use_activation_checkpointing=use_activation_checkpointing,
        )

        pose_dim = self.context_motion_frames * self.num_joints * 3
        if self.use_past_pose:
            self.pose_encoder = nn.Sequential(
                nn.LayerNorm(pose_dim),
                nn.Linear(pose_dim, embed_dim),
                nn.GELU(),
            )
            head_in = embed_dim * 2
        else:
            self.pose_encoder = None
            head_in = embed_dim

        out_dim = self.future_motion_frames * self.num_joints * 3
        self.motion_head = nn.Linear(head_in, out_dim)

    def forward(self, tokens: torch.Tensor, past_motion: torch.Tensor | None = None) -> torch.Tensor:
        """
        tokens: [B, S, D] frozen V-JEPA encoder tokens.
        past_motion: [B, T_ctx, J, 3] optional context poses.
        Returns: [B, T_future, J, 3]
        """
        pooled = self.pooler(tokens).squeeze(1)
        if self.use_past_pose:
            if past_motion is None:
                raise ValueError("use_past_pose=True requires past_motion")
            pose_flat = past_motion.reshape(past_motion.size(0), -1)
            pose_emb = self.pose_encoder(pose_flat)
            fused = torch.cat([pooled, pose_emb], dim=-1)
        else:
            fused = pooled

        out = self.motion_head(fused)
        b = out.size(0)
        return out.view(b, self.future_motion_frames, self.num_joints, 3)
