"""Tri-modal encoder-output fusion for HD-EPIC action anticipation.

Three separate modality encoders (video / gaze map / head-IMU proxy) feed a
``ProjectedTriModalCrossAttention`` block between the frozen V-JEPA encoder
output and the predictor input.  Video tokens keep their original flat shape
``[B, N_v, D]`` for the downstream predictor interface.

Fusion path (video branch, per user spec)::

    Z_attn = MultiHeadAttention(Q_v, concat(K_aux), concat(V_aux))
    Z_fused = W_o(Z_attn)
    Z_final = Z_video + gate(Z_video, Z_attn) * Z_fused   # if use_gated_residual
    Z_final = Z_video + Z_fused                           # otherwise

HD-EPIC does not ship raw accelerometer CSVs; the IMU branch uses SLAM
angular velocity + device linear velocity as a 6D proxy
``[gyro_x, gyro_y, gyro_z, vel_x, vel_y, vel_z]``.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hdepic_lora_action_anticipation.binary_input_adapter import BinaryGazeMapBuilder
from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate, labels_from_udata
from app.hdepic_lora_action_anticipation.pose_slam import SlamPoseLoader, window_smooth_pose_matrix
from app.hdepic_lora_action_anticipation.val_metrics import summarize_val_metrics
from src.utils.logging import AverageMeter

logger = logging.getLogger(__name__)


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def compute_token_budgets(
    n_video_spatial: int,
    gaze_grid_size: int = 10,
    gaze_token_ratio: float = 0.5,
    imu_token_ratio: float = 0.1,
) -> tuple[int, int, int]:
    """Return ``(n_video, n_gaze, n_imu)`` token counts per temporal slot."""
    n_gaze = int(gaze_grid_size * gaze_grid_size)
    n_imu = max(1, round(float(imu_token_ratio) * float(n_video_spatial)))
    return int(n_video_spatial), n_gaze, n_imu


def _reshape_video_tokens(x: torch.Tensor, temporal_slots: int) -> torch.Tensor:
    """``[B, N, D]`` -> ``[B, T, N_v, D]`` with shape assertions."""
    bsz, n_tok, dim = x.shape
    if temporal_slots <= 0 or n_tok % temporal_slots != 0:
        raise ValueError(f"Cannot reshape video tokens: N={n_tok}, temporal_slots={temporal_slots}")
    n_spatial = n_tok // temporal_slots
    out = x.view(bsz, temporal_slots, n_spatial, dim)
    assert out.shape[1:] == (temporal_slots, n_spatial, dim), (
        f"Z_video shape mismatch: got {tuple(out.shape)}, expected (*, {temporal_slots}, {n_spatial}, {dim})"
    )
    return out


def _flatten_modal_tokens(x: torch.Tensor) -> torch.Tensor:
    """``[B, T, N, D]`` -> ``[B, T*N, D]``."""
    bsz, t_slots, n_tok, dim = x.shape
    return x.reshape(bsz, t_slots * n_tok, dim)


def _pool_frames_to_tubelets(x: torch.Tensor, tubelet_size: int) -> torch.Tensor:
    """Average-pool frame dimension ``T_frames`` into ``T_frames // tubelet_size`` slots."""
    if tubelet_size <= 1:
        return x
    bsz, channels, frames, height, width = x.shape
    if frames % tubelet_size != 0:
        pad = tubelet_size - (frames % tubelet_size)
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        frames = x.shape[2]
    t_slots = frames // tubelet_size
    x = x.view(bsz, channels, t_slots, tubelet_size, height, width).mean(dim=3)
    return x


class GazeSpatialEncoder(nn.Module):
    """Lightweight spatial encoder for rasterized gaze maps ``[B, 1, H, W]``."""

    def __init__(self, embed_dim: int, grid_size: int = 10, in_channels: int = 1):
        super().__init__()
        self.grid_size = int(grid_size)
        self.n_tokens = self.grid_size * self.grid_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
        )
        self.proj = nn.Linear(64, embed_dim)
        self.queries = nn.Parameter(torch.randn(1, self.n_tokens, 64) * 0.02)

    def forward(self, gaze_maps: torch.Tensor) -> torch.Tensor:
        """Input ``[B*T, 1, H, W]`` -> ``[B*T, N_g, D]``."""
        feat = self.stem(gaze_maps)
        bt, _, fh, fw = feat.shape
        flat = feat.flatten(2).transpose(1, 2)
        q = self.queries.expand(bt, -1, -1)
        attn = torch.softmax(torch.bmm(q, flat.transpose(1, 2)) / (flat.shape[-1] ** 0.5), dim=-1)
        tokens = torch.bmm(attn, flat)
        return self.proj(tokens)


class ImuTemporalEncoder(nn.Module):
    """GRU/LSTM over per-frame IMU samples, pooled to ``N_imu`` motion tokens."""

    def __init__(
        self,
        embed_dim: int,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_imu_tokens: int = 20,
        encoder_type: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_imu_tokens = int(num_imu_tokens)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        rnn_type = str(encoder_type).lower()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        self.token_queries = nn.Parameter(torch.randn(1, self.num_imu_tokens, hidden_dim) * 0.02)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, imu: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Input ``[B*T, K, 6]`` -> ``[B*T, N_imu, D]``."""
        x = self.input_proj(imu)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
            )
            if isinstance(self.rnn, nn.LSTM):
                packed_out, _ = self.rnn(packed)
            else:
                packed_out, _ = self.rnn(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            h, _ = self.rnn(x)
        bt = h.shape[0]
        q = self.token_queries.expand(bt, -1, -1)
        attn = torch.softmax(torch.bmm(q, h.transpose(1, 2)) / (h.shape[-1] ** 0.5), dim=-1)
        tokens = torch.bmm(attn, h)
        return self.out_proj(tokens)


class ModalityProjections(nn.Module):
    """Learned Q/K/V/O projections for one modality."""

    def __init__(self, embed_dim: int, attn_dim: int):
        super().__init__()
        self.w_q = nn.Linear(embed_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, attn_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, attn_dim, bias=False)
        self.w_o = nn.Linear(attn_dim, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Start nearly closed: sigmoid(-4)≈0.018 so early updates stay near identity
        # until aux encoders become useful (avoids injecting random gaze/IMU noise).
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, -4.0)

    def project_qkv(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.w_q(z), self.w_k(z), self.w_v(z)


class ProjectedCrossAttentionUpdate(nn.Module):
    """One cross-attention update: query modality attends to auxiliary K/V."""

    def __init__(self, embed_dim: int, attn_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(f"attn_dim={attn_dim} must be divisible by num_heads={num_heads}")
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim

    def forward(
        self,
        z_query: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out_proj: nn.Linear,
        gate_mlp: nn.Sequential,
        use_gated_residual: bool,
    ) -> torch.Tensor:
        z_attn, _ = self.attn(q, k, v)
        z_fused = out_proj(z_attn)
        if use_gated_residual:
            gate = torch.sigmoid(gate_mlp(torch.cat([z_query, z_attn], dim=-1)))
            return z_query + gate * z_fused
        return z_query + z_fused


class ProjectedTriModalCrossAttention(nn.Module):
    """Projected tri-modal cross-attention with optional stacked layers."""

    def __init__(
        self,
        embed_dim: int,
        attn_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_gated_residual: bool = True,
        use_gaze_branch: bool = True,
        use_imu_branch: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_gaze_branch = bool(use_gaze_branch)
        self.use_imu_branch = bool(use_imu_branch)
        self.use_gated_residual = bool(use_gated_residual)
        self.video_proj = ModalityProjections(embed_dim, attn_dim)
        self.gaze_proj = ModalityProjections(embed_dim, attn_dim) if self.use_gaze_branch else None
        self.imu_proj = ModalityProjections(embed_dim, attn_dim) if self.use_imu_branch else None
        self.layers = nn.ModuleList(
            [
                ProjectedCrossAttentionUpdate(embed_dim, attn_dim, num_heads, dropout=dropout)
                for _ in range(int(num_layers))
            ]
        )
        for proj in (self.video_proj, self.gaze_proj, self.imu_proj):
            if proj is not None:
                nn.init.zeros_(proj.w_o.weight)
                nn.init.zeros_(proj.w_o.bias)

    def _update_branch(
        self,
        z_m: torch.Tensor,
        aux_z: list[torch.Tensor],
        aux_proj: list[ModalityProjections],
        self_proj: ModalityProjections,
        layer: ProjectedCrossAttentionUpdate,
    ) -> torch.Tensor:
        if not aux_z:
            return z_m
        q, _, _ = self_proj.project_qkv(z_m)
        k_parts = []
        v_parts = []
        for z_aux, proj in zip(aux_z, aux_proj):
            _, k_i, v_i = proj.project_qkv(z_aux)
            k_parts.append(k_i)
            v_parts.append(v_i)
        k = torch.cat(k_parts, dim=1)
        v = torch.cat(v_parts, dim=1)
        return layer(
            z_m,
            q,
            k,
            v,
            self_proj.w_o,
            self_proj.gate,
            self.use_gated_residual,
        )

    def forward(
        self,
        z_video: torch.Tensor,
        z_gaze: Optional[torch.Tensor] = None,
        z_imu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert z_video.ndim == 4, f"Z_video must be [B,T,N_v,D], got {tuple(z_video.shape)}"
        if z_gaze is not None:
            assert z_gaze.shape[:2] == z_video.shape[:2] and z_gaze.shape[-1] == z_video.shape[-1], (
                f"Z_gaze shape {tuple(z_gaze.shape)} incompatible with Z_video {tuple(z_video.shape)}"
            )
        if z_imu is not None:
            assert z_imu.shape[:2] == z_video.shape[:2] and z_imu.shape[-1] == z_video.shape[-1], (
                f"Z_imu shape {tuple(z_imu.shape)} incompatible with Z_video {tuple(z_video.shape)}"
            )

        bsz, t_slots, n_v, dim = z_video.shape
        flat_b = bsz * t_slots
        z_v = z_video.reshape(flat_b, n_v, dim)
        z_g = z_gaze.reshape(flat_b, z_gaze.shape[2], dim) if z_gaze is not None else None
        z_i = z_imu.reshape(flat_b, z_imu.shape[2], dim) if z_imu is not None else None

        for layer in self.layers:
            gaze_aux = [z_g] if z_g is not None and self.use_gaze_branch else []
            imu_aux = [z_i] if z_i is not None and self.use_imu_branch else []
            gaze_projs = [self.gaze_proj] if gaze_aux else []
            imu_projs = [self.imu_proj] if imu_aux else []

            if self.use_gaze_branch and z_g is not None:
                video_aux = [z_v] + ([z_i] if z_i is not None and self.use_imu_branch else [])
                video_projs = [self.video_proj] + ([self.imu_proj] if z_i is not None and self.use_imu_branch else [])
                z_g = self._update_branch(z_g, video_aux, video_projs, self.gaze_proj, layer)

            if self.use_imu_branch and z_i is not None:
                video_aux = [z_v] + ([z_g] if z_g is not None and self.use_gaze_branch else [])
                video_projs = [self.video_proj] + ([self.gaze_proj] if z_g is not None and self.use_gaze_branch else [])
                z_i = self._update_branch(z_i, video_aux, video_projs, self.imu_proj, layer)

            video_aux = gaze_aux + imu_aux
            video_projs = gaze_projs + imu_projs
            z_v = self._update_branch(z_v, video_aux, video_projs, self.video_proj, layer)

        z_video_fused = z_v.view(bsz, t_slots, n_v, dim)
        assert z_video_fused.shape == z_video.shape, (
            f"Z_video_fused shape {tuple(z_video_fused.shape)} != Z_video {tuple(z_video.shape)}"
        )
        z_gaze_fused = z_g.view(bsz, t_slots, z_gaze.shape[2], dim) if z_g is not None else None
        z_imu_fused = z_i.view(bsz, t_slots, z_imu.shape[2], dim) if z_i is not None else None
        return z_video_fused, z_gaze_fused, z_imu_fused


class ImuTrajectoryLoader:
    """Load per-frame inter-frame 6D IMU proxy arrays from SLAM trajectories."""

    def __init__(self, cfg: dict[str, Any], gate: GazeTokenGate | None = None):
        pose_cfg = dict(cfg.get("pose", {}))
        pose_cfg.setdefault("enabled", True)
        self.pose_loader = SlamPoseLoader({**cfg, "pose": pose_cfg}, gate=gate)
        self.k_max = int(pose_cfg.get("interframe_k_max", 128))
        # Clip-level cache: IMU features are deterministic given (video, frames).
        # Avoids re-slicing SLAM zips every epoch (main CPU stall → low GPU util).
        self._imu_cache: dict[tuple, np.ndarray | None] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.Lock()

    @staticmethod
    def _cache_key(meta) -> tuple:
        frame_indices = meta.get("frame_indices")
        if hasattr(frame_indices, "detach"):
            frame_indices = frame_indices.detach().cpu().numpy()
        if frame_indices is None:
            fi = ()
        else:
            fi = tuple(np.asarray(frame_indices, dtype=np.int64).tolist())
        return (str(meta.get("video_id", "")), fi, int(meta.get("start_frame", -1) or -1))

    def load_batch(self, metadata, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        if metadata is None:
            return None
        if not isinstance(metadata, list):
            metadata = [metadata]
        arrays = []
        lengths = []
        for meta in metadata:
            imu6 = self._query_imu_6d(meta)
            if imu6 is None:
                imu6 = np.zeros((1, self.k_max, 6), dtype=np.float32)
                valid_len = 0
            else:
                valid_len = int((np.abs(imu6).sum(axis=-1) > 0).sum())
            arrays.append(imu6)
            lengths.append(max(valid_len, 1))
        max_t = max(a.shape[0] for a in arrays)
        batch = np.zeros((len(arrays), max_t, self.k_max, 6), dtype=np.float32)
        out_lengths = np.zeros((len(arrays), max_t), dtype=np.int64)
        for idx, arr in enumerate(arrays):
            t_len = arr.shape[0]
            batch[idx, :t_len] = arr
            out_lengths[idx, :t_len] = lengths[idx]
        imu_t = torch.as_tensor(batch, device=device, dtype=torch.float32)
        len_t = torch.as_tensor(out_lengths, device=device, dtype=torch.int64)
        return imu_t, len_t

    def _query_imu_6d(self, meta) -> np.ndarray | None:
        key = self._cache_key(meta)
        with self._lock:
            if key in self._imu_cache:
                self._cache_hits += 1
                cached = self._imu_cache[key]
                return None if cached is None else cached.copy()
            self._cache_misses += 1
        # Do NOT call query_interframe_matrices first — it reloads the same clip
        # record and was only used as a null-check, doubling SLAM zip work.
        record = self.pose_loader._load_clip_record(meta)  # noqa: SLF001
        frame_ts = self.pose_loader.frame_timestamps_us(meta)
        if record is None or frame_ts is None:
            with self._lock:
                self._imu_cache[key] = None
            return None
        t_vid = int(frame_ts.shape[0])
        out = np.zeros((t_vid, self.k_max, 6), dtype=np.float32)
        for i in range(max(t_vid - 1, 0)):
            seg = self.pose_loader._slice_record_interval(  # noqa: SLF001
                record, float(frame_ts[i]), float(frame_ts[i + 1])
            )
            if seg is None or seg.timestamps_us.size < 1:
                continue
            gyro = seg.angular_vel if seg.angular_vel is not None else np.zeros((seg.timestamps_us.size, 3), np.float32)
            vel = seg.linear_vel if seg.linear_vel is not None else np.zeros((seg.timestamps_us.size, 3), np.float32)
            feats = np.concatenate([gyro.astype(np.float32), vel.astype(np.float32)], axis=1)
            out[i] = window_smooth_pose_matrix(feats, self.k_max)
        with self._lock:
            self._imu_cache[key] = out
            if (self._cache_hits + self._cache_misses) % 500 == 0:
                logger.info(
                    "ImuTrajectoryLoader cache: hits=%d misses=%d size=%d",
                    self._cache_hits,
                    self._cache_misses,
                    len(self._imu_cache),
                )
        return out.copy()


class TriModalFusionAdaptedModel(nn.Module):
    """Wrap V-JEPA with tri-modal encoder-output fusion before the predictor."""

    def __init__(
        self,
        base_model: nn.Module,
        fusion: ProjectedTriModalCrossAttention,
        gaze_encoder: Optional[GazeSpatialEncoder],
        imu_encoder: Optional[ImuTemporalEncoder],
        fusion_cfg: dict[str, Any],
    ):
        super().__init__()
        self.base_model = base_model
        self.fusion = fusion
        self.gaze_encoder = gaze_encoder
        self.imu_encoder = imu_encoder
        self.fusion_cfg = dict(fusion_cfg)
        self.embed_dim = int(base_model.encoder.embed_dim)
        self.tubelet_size = int(base_model.tubelet_size)
        self.grid_size = int(base_model.grid_size)
        self.use_gaze_branch = bool(fusion_cfg.get("use_gaze_branch", True))
        self.use_imu_branch = bool(fusion_cfg.get("use_imu_branch", True))
        self.gaze_grid_size = int(fusion_cfg.get("gaze_grid_size", 10))
        n_v, n_g, n_i = compute_token_budgets(
            self.grid_size * self.grid_size,
            gaze_grid_size=self.gaze_grid_size,
            gaze_token_ratio=float(fusion_cfg.get("gaze_token_ratio", 0.5)),
            imu_token_ratio=float(fusion_cfg.get("imu_token_ratio", 0.1)),
        )
        self.n_video_spatial = n_v
        self.n_gaze_tokens = n_g
        self.n_imu_tokens = n_i

    def _temporal_slots(self, num_frames: int) -> int:
        if num_frames % self.tubelet_size != 0:
            num_frames = num_frames + (self.tubelet_size - num_frames % self.tubelet_size)
        return max(1, num_frames // self.tubelet_size)

    def _encode_modalities(
        self,
        x_full: torch.Tensor,
        gaze_map: Optional[torch.Tensor],
        imu_batch: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, n_tok, dim = x_full.shape
        t_slots = self._temporal_slots(int(gaze_map.shape[2]) if gaze_map is not None else n_tok // self.n_video_spatial)
        z_video = _reshape_video_tokens(x_full[:, :, -dim:], t_slots)

        z_gaze = None
        if self.use_gaze_branch and gaze_map is not None and self.gaze_encoder is not None:
            gaze_slots = _pool_frames_to_tubelets(gaze_map, self.tubelet_size)
            bt = bsz * gaze_slots.shape[2]
            g_in = gaze_slots.permute(0, 2, 1, 3, 4).reshape(bt, 1, gaze_slots.shape[3], gaze_slots.shape[4])
            z_gaze = self.gaze_encoder(g_in).view(bsz, t_slots, self.n_gaze_tokens, dim)

        z_imu = None
        if self.use_imu_branch and imu_batch is not None and self.imu_encoder is not None:
            imu_t, imu_lens = imu_batch
            # Align IMU temporal axis to video tubelet slots before flattening.
            imu_slots = self._pool_imu_to_tubelets(imu_t, imu_lens, target_slots=t_slots)
            assert imu_slots.shape[:2] == (bsz, t_slots), (
                f"IMU slots {tuple(imu_slots.shape)} != expected ({bsz}, {t_slots}, K, 6)"
            )
            bt = bsz * t_slots
            k = imu_slots.shape[2]
            imu_flat = imu_slots.reshape(bt, k, 6).to(x_full.dtype)
            lens_flat = imu_slots.new_full((bt,), k, dtype=torch.long)
            z_imu = self.imu_encoder(imu_flat, lengths=lens_flat).view(bsz, t_slots, self.n_imu_tokens, dim)

        return z_video, z_gaze, z_imu

    def _pool_imu_to_tubelets(
        self,
        imu: torch.Tensor,
        lengths: torch.Tensor,
        target_slots: Optional[int] = None,
    ) -> torch.Tensor:
        """Pool per-frame IMU ``[B,T,K,6]`` into tubelet slots ``[B,T_slots,K,6]``.

        If ``target_slots`` is given (video tubelet count), temporally resample so
        IMU slots match the video timeline even when frame counts differ.
        """
        bsz, t_frames, k, d = imu.shape
        if t_frames <= 0:
            slots = int(target_slots or 1)
            return imu.new_zeros((bsz, slots, k, d))

        if t_frames % self.tubelet_size != 0:
            pad = self.tubelet_size - (t_frames % self.tubelet_size)
            imu = F.pad(imu, (0, 0, 0, 0, 0, pad))
            lengths = F.pad(lengths, (0, pad))
            t_frames = imu.shape[1]
        src_slots = t_frames // self.tubelet_size
        imu = imu.view(bsz, src_slots, self.tubelet_size, k, d).mean(dim=2)

        if target_slots is None or target_slots == src_slots:
            return imu
        # [B, src_slots, K, 6] -> [B*K, 6, src_slots] -> interpolate -> [B, target, K, 6]
        x = imu.permute(0, 2, 3, 1).reshape(bsz * k, d, src_slots)
        x = F.interpolate(x, size=int(target_slots), mode="linear", align_corners=False)
        return x.view(bsz, k, d, int(target_slots)).permute(0, 3, 1, 2).contiguous()

    def _fuse_for_predictor(
        self,
        x_full: torch.Tensor,
        gaze_map: Optional[torch.Tensor],
        imu_batch: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if not self.use_gaze_branch and not self.use_imu_branch:
            return x_full
        embed_dim = self.embed_dim
        use_hierarchical = x_full.size(-1) > embed_dim
        x_last = x_full[:, :, -embed_dim:] if use_hierarchical else x_full
        z_video, z_gaze, z_imu = self._encode_modalities(x_full, gaze_map, imu_batch)
        z_video_fused, _, _ = self.fusion(z_video, z_gaze, z_imu)
        z_flat = _flatten_modal_tokens(z_video_fused)
        if use_hierarchical:
            return torch.cat([x_full[:, :, :-embed_dim], z_flat], dim=-1)
        return z_flat

    def forward(
        self,
        clips: torch.Tensor,
        anticipation_times: torch.Tensor,
        gaze_map: Optional[torch.Tensor] = None,
        imu_batch: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        base = self.base_model
        x_full = base.encoder(clips)
        if torch.is_tensor(x_full) and not torch.isfinite(x_full).all():
            return None
        # Fusion-only recipes freeze the video encoder: drop its activations from the
        # autograd graph so backward only touches gaze/IMU/fusion (saves H100 memory
        # → larger batches → more GPU work per decode).
        if not any(p.requires_grad for p in base.encoder.parameters()):
            x_full = x_full.detach()
        if base.no_predictor:
            return x_full

        embed_dim = base.encoder.embed_dim
        use_hierarchical = x_full.size(-1) > embed_dim
        x_last_obs = x_full[:, :, -embed_dim:] if use_hierarchical else x_full
        bsz = x_full.size(0)
        if base.no_encoder:
            x_accumulate = torch.rand(bsz, 0, embed_dim, device=x_full.device)
        else:
            x_accumulate = x_last_obs.clone()

        x_pred_input = self._fuse_for_predictor(x_full, gaze_map, imu_batch)
        if torch.is_tensor(x_pred_input) and not torch.isfinite(x_pred_input).all():
            return None

        if int(getattr(base, "num_steps", 1)) > 1 and hasattr(base, "_forward_sliding_window"):
            return base._forward_sliding_window(x_pred_input, anticipation_times, x_accumulate)

        return self._forward_single_step(base, x_pred_input, x_accumulate, anticipation_times)

    def _forward_single_step(self, base, x_pred_input, x_accumulate, anticipation_times):
        bsz, n_ctx, _ = x_pred_input.size()
        embed_dim = base.encoder.embed_dim
        device = x_pred_input.device
        spatial_tokens = int(base.grid_size**2)
        chunk_tokens = int(spatial_tokens * (base.num_output_frames // base.tubelet_size))
        local_ctxt_positions = torch.arange(n_ctx, device=device).unsqueeze(0).repeat(bsz, 1)
        local_tgt_positions = torch.arange(chunk_tokens, device=device).unsqueeze(0).repeat(bsz, 1)
        local_tgt_positions += n_ctx
        horizon_chunks = (anticipation_times * base.frames_per_second / base.tubelet_size).to(torch.int64)
        rollout_steps = (horizon_chunks + (base.num_output_frames // base.tubelet_size)).clamp(min=1)
        max_steps = int(rollout_steps.max().item())
        x_window = x_pred_input
        target_by_sample = [None for _ in range(bsz)]
        rollout_for_classifier = []
        for step in range(max_steps):
            pred_out = base.predictor(x_window, masks_x=local_ctxt_positions, masks_y=local_tgt_positions)
            pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
            pred_for_classifier = pred_full[:, :, -embed_dim:] if pred_full.size(-1) != embed_dim else pred_full
            rollout_for_classifier.append(pred_for_classifier)
            for b in range(bsz):
                if step == int(rollout_steps[b].item()) - 1:
                    target_by_sample[b] = pred_for_classifier[b : b + 1]
            pred_for_input = pred_full if pred_full.size(-1) == x_window.size(-1) else pred_for_classifier
            x_window = torch.cat([x_window[:, chunk_tokens:, :], pred_for_input], dim=1)
        target_tokens = torch.cat(target_by_sample, dim=0)
        final_window = x_window[:, :, -embed_dim:] if x_window.size(-1) != embed_dim else x_window
        return_mode = getattr(base, "return_mode", "observed_plus_target")
        if return_mode == "target_only":
            return target_tokens
        if return_mode == "final_window":
            return final_window
        if return_mode == "observed_plus_rollout":
            return torch.cat([x_accumulate, *rollout_for_classifier], dim=1)
        return torch.cat([x_accumulate, target_tokens], dim=1)


def tri_modal_fusion_param_names(model: nn.Module) -> set[str]:
    model = unwrap_ddp(model)
    prefixes = ("fusion.", "gaze_encoder.", "imu_encoder.")
    names: set[str] = set()
    for name, _ in model.named_parameters():
        bare = name.split("module.", 1)[-1] if name.startswith("module.") else name
        if bare.startswith(prefixes):
            names.add(name)
    return names


def trainable_tri_modal_fusion_params(model: nn.Module) -> list[nn.Parameter]:
    model = unwrap_ddp(model)
    params: list[nn.Parameter] = []
    for module in (model.fusion, model.gaze_encoder, model.imu_encoder):
        if module is None:
            continue
        for p in module.parameters():
            if p.requires_grad:
                params.append(p)
    return params


def _fusion_grads_finite(model: nn.Module) -> bool:
    model = unwrap_ddp(model)
    for module in (model.fusion, model.gaze_encoder, model.imu_encoder):
        if module is None:
            continue
        for param in module.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False
    return True


def _zero_fusion_grads(model: nn.Module) -> None:
    model = unwrap_ddp(model)
    for module in (model.fusion, model.gaze_encoder, model.imu_encoder):
        if module is None:
            continue
        for param in module.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()


def _classifier_grads_finite(classifier: nn.Module) -> bool:
    for param in classifier.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


def _zero_classifier_grads(classifier: nn.Module) -> None:
    for param in classifier.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def save_tri_modal_fusion_checkpoint(model: nn.Module, path: str) -> None:
    model = unwrap_ddp(model)
    payload = {"fusion": model.fusion.state_dict()}
    if model.gaze_encoder is not None:
        payload["gaze_encoder"] = model.gaze_encoder.state_dict()
    if model.imu_encoder is not None:
        payload["imu_encoder"] = model.imu_encoder.state_dict()
    torch.save(payload, path)


def load_tri_modal_fusion_checkpoint(model: nn.Module, path: str) -> None:
    model = unwrap_ddp(model)
    payload = torch.load(path, map_location="cpu")
    model.fusion.load_state_dict(payload["fusion"], strict=False)
    if model.gaze_encoder is not None and "gaze_encoder" in payload:
        model.gaze_encoder.load_state_dict(payload["gaze_encoder"], strict=False)
    if model.imu_encoder is not None and "imu_encoder" in payload:
        model.imu_encoder.load_state_dict(payload["imu_encoder"], strict=False)


def _metadata_from_udata(udata):
    metadata = udata[3] if len(udata) > 4 else None
    if metadata is None:
        raise ValueError("projected_tri_modal_cross_attention requires metadata-aware dataloader")
    return metadata


def _prepare_aux_cpu(
    gaze_map_builder: BinaryGazeMapBuilder,
    imu_loader: ImuTrajectoryLoader,
    udata,
):
    """CPU-side gaze rasterize + IMU load (overlapped with GPU via prefetch)."""
    clips = udata[0]
    metadata = _metadata_from_udata(udata)
    _, _, frames, height, width = clips.shape
    gaze_cpu = gaze_map_builder.build_cpu(metadata, int(frames), int(height), int(width))
    imu_cpu = imu_loader.load_batch(metadata, torch.device("cpu"))
    return gaze_cpu, imu_cpu


class _TriModalBatchPrefetcher:
    """Two-stage producer: decode ‖ gaze/IMU, then feed GPU.

    Stage A: ``next(dataloader)`` only (video decode / collate).
    Stage B: gaze rasterize + IMU load on a thread pool.
    Overlapping A and B cuts wall-clock fetch when aux is a large fraction of
    the old sequential ``next+aux`` path (needed for >60% GPU util).
    """

    def __init__(
        self,
        data_loader,
        gaze_map_builder: BinaryGazeMapBuilder,
        imu_loader: ImuTrajectoryLoader,
        *,
        depth: int = 3,
        aux_workers: int | None = None,
    ):
        self.data_loader = data_loader
        self.gaze_map_builder = gaze_map_builder
        self.imu_loader = imu_loader
        self.depth = max(2, int(depth))
        if aux_workers is None:
            aux_workers = int(os.environ.get("TRI_MODAL_AUX_WORKERS", "2") or "2")
        self.aux_workers = max(1, int(aux_workers))
        # Raw video batches waiting for aux; keep at least 1 in flight.
        self._raw_q: queue.Queue = queue.Queue(maxsize=max(2, self.depth))
        self._q: queue.Queue = queue.Queue(maxsize=self.depth)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._aux_pool = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(
            max_workers=self.aux_workers, thread_name_prefix="tri-aux"
        )
        self._decode_thread = threading.Thread(target=self._run_decode, name="tri-modal-decode", daemon=True)
        self._aux_thread = threading.Thread(target=self._run_aux, name="tri-modal-aux", daemon=True)
        self._decode_thread.start()
        self._aux_thread.start()
        logger.info(
            "TriModalBatchPrefetcher pipeline: ready_depth=%d raw_depth=%d aux_workers=%d",
            self.depth,
            self._raw_q.maxsize,
            self.aux_workers,
        )

    def _fail(self, exc: BaseException) -> None:
        self._error = exc
        self._stop.set()
        for q in (self._raw_q, self._q):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

    def _run_decode(self) -> None:
        _data_loader = iter(self.data_loader)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                try:
                    udata = next(_data_loader)
                except StopIteration:
                    _data_loader = iter(self.data_loader)
                    udata = next(_data_loader)
                decode_ms = (time.time() - t0) * 1000.0
                self._raw_q.put((udata, decode_ms))
            except BaseException as exc:  # noqa: BLE001
                self._fail(exc)
                return

    def _run_aux(self) -> None:
        pending: list = []
        while not self._stop.is_set():
            try:
                # Reap completed aux jobs first so ready_q stays full.
                still = []
                for fut in pending:
                    if not fut.done():
                        still.append(fut)
                        continue
                    try:
                        item = fut.result()
                    except BaseException as exc:  # noqa: BLE001
                        self._fail(exc)
                        return
                    self._q.put(item)
                pending = still

                if len(pending) >= self.aux_workers:
                    time.sleep(0.001)
                    continue

                try:
                    raw = self._raw_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if raw is None:
                    self._q.put(None)
                    return
                udata, decode_ms = raw

                def _job(u=udata, d_ms=decode_ms):
                    t1 = time.time()
                    gaze_cpu, imu_cpu = _prepare_aux_cpu(self.gaze_map_builder, self.imu_loader, u)
                    aux_ms = (time.time() - t1) * 1000.0
                    # fetch_ms ≈ decode + aux wall if sequential; report sum for logs
                    # but pipeline hides aux behind next decode.
                    fetch_ms = float(d_ms) + float(aux_ms)
                    return (u, gaze_cpu, imu_cpu, fetch_ms, float(d_ms), float(aux_ms))

                pending.append(self._aux_pool.submit(_job))
            except BaseException as exc:  # noqa: BLE001
                self._fail(exc)
                return

    def get(self):
        item = self._q.get()
        if item is None:
            if self._error is not None:
                raise RuntimeError("tri-modal batch prefetcher failed") from self._error
            raise RuntimeError("tri-modal batch prefetcher stopped unexpectedly")
        # Back-compat: allow 4-tuple or 6-tuple
        if len(item) == 4:
            return item
        udata, gaze_cpu, imu_cpu, fetch_ms, _decode_ms, _aux_ms = item
        return udata, gaze_cpu, imu_cpu, fetch_ms

    def get_detailed(self):
        item = self._q.get()
        if item is None:
            if self._error is not None:
                raise RuntimeError("tri-modal batch prefetcher failed") from self._error
            raise RuntimeError("tri-modal batch prefetcher stopped unexpectedly")
        if len(item) == 4:
            udata, gaze_cpu, imu_cpu, fetch_ms = item
            return udata, gaze_cpu, imu_cpu, fetch_ms, fetch_ms, 0.0
        return item

    def close(self):
        self._stop.set()
        for q in (self._raw_q, self._q):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        self._decode_thread.join(timeout=5.0)
        self._aux_thread.join(timeout=5.0)
        self._aux_pool.shutdown(wait=False)


def train_one_epoch_with_tri_modal_fusion(
    base_eval,
    gaze_map_builder: BinaryGazeMapBuilder,
    imu_loader: ImuTrajectoryLoader,
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
):
    model_inner = unwrap_ddp(model)
    model_inner.base_model.eval()
    model_inner.fusion.train(mode=True)
    if model_inner.gaze_encoder is not None:
        model_inner.gaze_encoder.train(mode=True)
    if model_inner.imu_encoder is not None:
        model_inner.imu_encoder.train(mode=True)
    for c in classifiers:
        c.train(mode=True)
    if action_is_verb_noun:
        verb_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5)
            for _ in classifiers
        ]
        noun_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5)
            for _ in classifiers
        ]
    action_metric_loggers = [
        base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5)
        for _ in classifiers
    ]
    batch_wait_meter = AverageMeter()
    fetch_meter = AverageMeter()
    try:
        max_train_iters = int(
            os.environ.get("EVAL_MAX_TRAIN_ITERS", os.environ.get("MAX_TRAIN_ITERS", "0")) or "0"
        )
    except ValueError:
        max_train_iters = 0
    if max_train_iters > 0 and max_train_iters < ipe:
        logger.info(
            "Limiting train_one_epoch_with_tri_modal_fusion to %d/%d iterations via EVAL_MAX_TRAIN_ITERS",
            max_train_iters,
            ipe,
        )
        ipe = max_train_iters

    # Depth>=2 so next(dataloader)+gaze/IMU stay off the main thread while GPU runs.
    prefetch_depth = max(2, int(os.environ.get("TRI_MODAL_PREFETCH_DEPTH", "3")))
    logger.info(
        "tri_modal train async prefetch: depth=%d (decode‖aux pipeline)",
        prefetch_depth,
    )
    prefetcher = _TriModalBatchPrefetcher(
        data_loader, gaze_map_builder, imu_loader, depth=prefetch_depth
    )
    decode_meter = AverageMeter()
    aux_meter = AverageMeter()
    try:
        for itr in range(ipe):
            itr_start_time = time.time()
            [s.step() for s in scheduler]
            [wds_.step() for wds_ in wd_scheduler]

            wait_t0 = time.time()
            udata, gaze_cpu, imu_cpu, fetch_ms, decode_ms, aux_ms = prefetcher.get_detailed()
            batch_wait_meter.update((time.time() - wait_t0) * 1000.0)
            fetch_meter.update(float(fetch_ms))
            decode_meter.update(float(decode_ms))
            aux_meter.update(float(aux_ms))

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                clips = udata[0].to(device, non_blocking=True)
                anticipation_times = udata[4].to(device, non_blocking=True)
                labels = labels_from_udata(
                    udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes
                )
                gaze_map = gaze_cpu.to(device, non_blocking=True)
                if imu_cpu is None:
                    imu_batch = None
                else:
                    imu_batch = (
                        imu_cpu[0].to(device, non_blocking=True),
                        imu_cpu[1].to(device, non_blocking=True),
                    )
                tokens = model(clips, anticipation_times, gaze_map=gaze_map, imu_batch=imu_batch)
                if tokens is None:
                    logger.warning(
                        "Skipping tri_modal_fusion optimizer step because tokens are non-finite at itr=%d",
                        itr,
                    )
                    optimizer[0].zero_grad()
                    continue
                tokens_proxy = tokens.detach().requires_grad_(True)
                outputs = [c(tokens_proxy) for c in classifiers]

            if action_is_verb_noun:
                loss = [
                    criterion(o["verb"], labels["verb"])
                    + criterion(o["noun"], labels["noun"])
                    + criterion(o["action"], labels["action"])
                    for o in outputs
                ]
            else:
                loss = [criterion(o["action"], labels["action"]) for o in outputs]

            tokens_grad_accum = torch.zeros_like(tokens_proxy)
            healthy_heads = 0
            for head_idx, (l, c) in enumerate(zip(loss, classifiers)):
                if not torch.isfinite(l.detach()):
                    _zero_classifier_grads(c)
                    continue
                if tokens_proxy.grad is not None:
                    tokens_proxy.grad.zero_()
                scaled = scaler[0].scale(l) if use_bfloat16 else l
                scaled.backward(retain_graph=(head_idx < len(loss) - 1))
                head_token_grad = tokens_proxy.grad
                if head_token_grad is None or not torch.isfinite(head_token_grad).all():
                    _zero_classifier_grads(c)
                    continue
                if not _classifier_grads_finite(c):
                    _zero_classifier_grads(c)
                    continue
                tokens_grad_accum.add_(head_token_grad)
                healthy_heads += 1

            if healthy_heads == 0:
                optimizer[0].zero_grad()
                if use_bfloat16:
                    scaler[0].update()
                continue

            tokens_grad_accum.mul_(1.0 / float(healthy_heads))
            tokens.backward(gradient=tokens_grad_accum)
            fusion_ok = _fusion_grads_finite(model)
            if not fusion_ok:
                _zero_fusion_grads(model)
            if use_bfloat16:
                scaler[0].step(optimizer[0])
                scaler[0].update()
            else:
                optimizer[0].step()
            optimizer[0].zero_grad()

            with torch.no_grad():
                action_metrics = [
                    m(o["action"], labels["action"]) for o, m in zip(outputs, action_metric_loggers)
                ]
                if action_is_verb_noun:
                    verb_metrics = [
                        m(o["verb"], labels["verb"]) for o, m in zip(outputs, verb_metric_loggers)
                    ]
                    noun_metrics = [
                        m(o["noun"], labels["noun"]) for o, m in zip(outputs, noun_metric_loggers)
                    ]
            if itr % 10 == 0 or itr == ipe - 1:
                step_ms = (time.time() - itr_start_time) * 1000.0
                if action_is_verb_noun:
                    logger.info(
                        "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                        "healthy_heads=%d/%d fusion_ok=%s [mem: %.2e] "
                        "[batch_wait: %.1f ms] [fetch: %.0f ms] [decode: %.0f ms] [aux: %.0f ms] [step: %.0f ms]",
                        itr,
                        max(a["accuracy"] for a in action_metrics),
                        max(v["accuracy"] for v in verb_metrics),
                        max(n["accuracy"] for n in noun_metrics),
                        max(a["recall"] for a in action_metrics),
                        max(v["recall"] for v in verb_metrics),
                        max(n["recall"] for n in noun_metrics),
                        healthy_heads,
                        len(loss),
                        fusion_ok,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                        batch_wait_meter.avg,
                        fetch_meter.avg,
                        decode_meter.avg,
                        aux_meter.avg,
                        step_ms,
                    )
    finally:
        prefetcher.close()

    ret = {
        "action": {
            "accuracy": max(a["accuracy"] for a in action_metrics),
            "recall": max(a["recall"] for a in action_metrics),
        }
    }
    if action_is_verb_noun:
        ret.update(
            {
                "verb": {
                    "accuracy": max(v["accuracy"] for v in verb_metrics),
                    "recall": max(v["recall"] for v in verb_metrics),
                },
                "noun": {
                    "accuracy": max(n["accuracy"] for n in noun_metrics),
                    "recall": max(n["recall"] for n in noun_metrics),
                },
            }
        )
    return ret


@torch.no_grad()
def validate_with_tri_modal_fusion(
    base_eval,
    dumper,
    gaze_map_builder: BinaryGazeMapBuilder,
    imu_loader: ImuTrajectoryLoader,
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    data_loader,
    use_bfloat16,
    valid_nouns,
    valid_verbs,
    valid_actions,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
    val_metric_scope: str = "native",
    val_metric_aggregation: str = "metric_wise_max",
    val_fixed_head_index: int | None = None,
):
    metric_scope = str(val_metric_scope).lower()
    if metric_scope not in {"native", "filtered"}:
        raise ValueError(f"Unsupported val_metric_scope={val_metric_scope!r}; expected native or filtered")
    use_valid_filter = metric_scope == "filtered"
    logger.info(
        "Running val with tri-modal projected cross-attention fusion (metric_scope=%s, aggregation=%s)...",
        metric_scope,
        val_metric_aggregation,
    )
    model_inner = unwrap_ddp(model)
    model_inner.base_model.eval()
    model_inner.fusion.eval()
    if model_inner.gaze_encoder is not None:
        model_inner.gaze_encoder.eval()
    if model_inner.imu_encoder is not None:
        model_inner.imu_encoder.eval()
    for c in classifiers:
        c.train(mode=False)
    if action_is_verb_noun:
        verb_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5)
            for _ in classifiers
        ]
        noun_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5)
            for _ in classifiers
        ]
    action_metric_loggers = [
        base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5)
        for _ in classifiers
    ]

    prefetch_depth = max(2, int(os.environ.get("TRI_MODAL_PREFETCH_DEPTH", "3")))
    prefetcher = _TriModalBatchPrefetcher(
        data_loader, gaze_map_builder, imu_loader, depth=prefetch_depth
    )
    try:
        for itr in range(ipe):
            udata, gaze_cpu, imu_cpu, _fetch_ms = prefetcher.get()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                clips = udata[0].to(device, non_blocking=True)
                labels = labels_from_udata(
                    udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes
                )
                anticipation_times = udata[4].to(device, non_blocking=True)
                gaze_map = gaze_cpu.to(device, non_blocking=True)
                if imu_cpu is None:
                    imu_batch = None
                else:
                    imu_batch = (
                        imu_cpu[0].to(device, non_blocking=True),
                        imu_cpu[1].to(device, non_blocking=True),
                    )
                tokens = model(clips, anticipation_times, gaze_map=gaze_map, imu_batch=imu_batch)
                if tokens is None:
                    continue
                outputs = [c(tokens) for c in classifiers]
                valid_actions_arg = valid_actions if use_valid_filter else None
                valid_verbs_arg = valid_verbs if use_valid_filter else None
                valid_nouns_arg = valid_nouns if use_valid_filter else None
                action_metrics = [
                    m(o["action"], labels["action"], valid_actions_arg)
                    for o, m in zip(outputs, action_metric_loggers)
                ]
                if action_is_verb_noun:
                    verb_metrics = [
                        m(o["verb"], labels["verb"], valid_verbs_arg)
                        for o, m in zip(outputs, verb_metric_loggers)
                    ]
                    noun_metrics = [
                        m(o["noun"], labels["noun"], valid_nouns_arg)
                        for o, m in zip(outputs, noun_metric_loggers)
                    ]
                else:
                    verb_metrics = noun_metrics = None
                if action_is_verb_noun:
                    verb_loss = sum(criterion(o["verb"], labels["verb"]) for o in outputs)
                    noun_loss = sum(criterion(o["noun"], labels["noun"]) for o in outputs)
                    action_loss = sum(criterion(o["action"], labels["action"]) for o in outputs)
                    loss = verb_loss + noun_loss + action_loss
                else:
                    verb_loss = noun_loss = None
                    loss = sum(criterion(o["action"], labels["action"]) for o in outputs)
            best_head_idx = max(range(len(action_metrics)), key=lambda i: action_metrics[i]["accuracy"])
            dumper.add_batch(
                udata,
                [outputs[best_head_idx]],
                labels,
                {"verb": verb_classes, "noun": noun_classes, "action": action_classes},
            )
            if itr % 10 == 0 or itr == ipe - 1:
                if action_is_verb_noun:
                    logger.info(
                        "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                        "loss (v/n): %.3f (%.3f %.3f) [mem: %.2e]",
                        itr,
                        max(a["accuracy"] for a in action_metrics),
                        max(v["accuracy"] for v in verb_metrics),
                        max(n["accuracy"] for n in noun_metrics),
                        max(a["recall"] for a in action_metrics),
                        max(v["recall"] for v in verb_metrics),
                        max(n["recall"] for n in noun_metrics),
                        float(loss.detach()),
                        float(verb_loss.detach()),
                        float(noun_loss.detach()),
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )
    finally:
        prefetcher.close()

    dumper.write()
    return summarize_val_metrics(
        action_metrics,
        verb_metrics if action_is_verb_noun else None,
        noun_metrics if action_is_verb_noun else None,
        metric_scope,
        metric_aggregation=val_metric_aggregation,
        val_fixed_head_index=val_fixed_head_index,
    )
