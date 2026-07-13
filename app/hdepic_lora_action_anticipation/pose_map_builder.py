"""Inter-frame SLAM pose matrix rasterization for binary-map-style input adapters."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from app.hdepic_lora_action_anticipation.binary_input_adapter import BinaryGazeMapBuilder
from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate
from app.hdepic_lora_action_anticipation.pose_slam import SlamPoseLoader

logger = logging.getLogger(__name__)


def rasterize_pose_matrix_to_patch(
    pose_mat: np.ndarray,
    patch_height: int,
    patch_width: int,
    normalize: str = "none",
) -> np.ndarray:
    """Layout ``[K, D]`` as a 2D patch ``[patch_height, patch_width]``."""
    patch = np.zeros((int(patch_height), int(patch_width)), dtype=np.float32)
    if pose_mat.size == 0:
        return patch
    mat = np.asarray(pose_mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"Expected pose_mat.ndim==2, got {mat.ndim}")
    norm = str(normalize).lower()
    if norm == "minmax":
        lo = float(np.min(mat))
        hi = float(np.max(mat))
        if hi > lo:
            mat = (mat - lo) / (hi - lo)
        else:
            mat = np.zeros_like(mat)
    elif norm not in {"none", ""}:
        raise ValueError(f"Unsupported pose_map.normalize={normalize!r}")
    kh = min(int(mat.shape[0]), int(patch_height))
    kw = min(int(mat.shape[1]), int(patch_width))
    patch[:kh, :kw] = mat[:kh, :kw]
    return patch


class InterframePoseMapBuilder:
    """Rasterize inter-frame pose matrices into per-frame spatial maps ``[B,1,T,H,W]``."""

    def __init__(self, cfg: dict[str, Any], gate: GazeTokenGate | None = None):
        self.cfg = dict(cfg)
        pose_cfg = dict(cfg.get("pose", {}))
        pose_cfg.setdefault("gaze_root", cfg.get("gaze_root"))
        pose_cfg.setdefault("sync_root", cfg.get("sync_root"))
        if gate is None:
            gate = GazeTokenGate({"mode": "none", "gaze_root": cfg.get("gaze_root"), "sync_root": cfg.get("sync_root")})
        self.pose_loader = SlamPoseLoader(pose_cfg, gate=gate)
        self.k_max = int(pose_cfg.get("interframe_k_max", 128))
        map_cfg = dict(cfg.get("pose_map", {}))
        self.patch_height = int(map_cfg.get("patch_height", 128))
        configured_patch_width = int(map_cfg.get("patch_width", self.pose_loader.input_dim))
        if configured_patch_width < self.pose_loader.input_dim:
            logger.warning(
                "pose_map.patch_width=%d is smaller than pose feature dim=%d; expanding patch width to avoid dropping pose channels",
                configured_patch_width,
                self.pose_loader.input_dim,
            )
            configured_patch_width = self.pose_loader.input_dim
        self.patch_width = configured_patch_width
        self.layout = str(map_cfg.get("layout", "topleft")).lower()
        self.normalize = str(map_cfg.get("normalize", "none"))
        self.force_zero_pose = bool(map_cfg.get("force_zero_pose", False))

    def build(self, clips: torch.Tensor, metadata) -> torch.Tensor:
        bsz, _, frames, height, width = clips.shape
        maps = clips.new_zeros((bsz, 1, frames, height, width))
        if self.force_zero_pose:
            return maps
        ph = min(self.patch_height, int(height))
        pw = min(self.patch_width, int(width))
        if self.layout != "topleft":
            raise ValueError(f"Unsupported pose_map.layout={self.layout!r}; expected topleft")

        for idx in range(bsz):
            meta = metadata[idx] if isinstance(metadata, list) else metadata
            pose_mats = self.pose_loader.query_interframe_matrices(meta, self.k_max)
            if pose_mats is None:
                continue
            nframes = min(int(frames), int(pose_mats.shape[0]))
            for t in range(nframes):
                patch = rasterize_pose_matrix_to_patch(
                    pose_mats[t],
                    self.patch_height,
                    self.patch_width,
                    normalize=self.normalize,
                )
                patch_t = torch.as_tensor(patch, device=clips.device, dtype=clips.dtype)
                if ph != self.patch_height or pw != self.patch_width:
                    patch_t = patch_t.unsqueeze(0).unsqueeze(0)
                    patch_t = F.interpolate(patch_t, size=(ph, pw), mode="nearest").squeeze(0).squeeze(0)
                maps[idx, 0, t, :ph, :pw] = patch_t
        return maps


class GazePoseInputMapBuilder:
    """Build combined gaze + inter-frame pose auxiliary maps for the input adapter."""

    def __init__(self, cfg: dict[str, Any], gate: GazeTokenGate | None = None):
        self.gaze_builder = BinaryGazeMapBuilder(cfg, gate=gate)
        self.pose_builder = InterframePoseMapBuilder(cfg, gate=gate)
        self.adapter_checkpoint_path = self.gaze_builder.adapter_checkpoint_path
        self.rank = self.gaze_builder.rank

    def build(self, clips: torch.Tensor, metadata) -> torch.Tensor:
        gaze_map = self.gaze_builder.build(clips, metadata)
        pose_map = self.pose_builder.build(clips, metadata)
        return torch.cat([gaze_map, pose_map], dim=1)
