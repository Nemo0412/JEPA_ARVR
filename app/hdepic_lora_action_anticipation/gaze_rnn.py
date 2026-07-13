"""RNN-based gaze trajectory fusion at the probe.

This module adds a third gaze-fusion variant on top of the existing LoRA probe
pipeline. The frozen V-JEPA2 encoder/predictor and the LoRA-finetuned pooler
are left untouched. The gaze trajectory of the observed 4s window is encoded
by a BiGRU (or per-frame MLP for ablation) into a fixed number of gaze tokens.
Optionally, each gaze sample is conditioned on observed video tokens from the
V-JEPA token grid before it enters the GRU/MLP. Supported first-pass fusion
variants include nearest-token concat, gated nearest-token conditioning, local
neighborhood attention, and a residual-conditioned path with a monitored
learnable alpha.
Those tokens are concatenated to the K/V of the probe's cross-attention block
(after the probe self-attn blocks), so the probe queries attend to the union
of predictor-output video tokens and the gaze tokens.

Trajectory loading reuses ``GazeTokenGate._load_record`` from ``gaze.py`` so
that the existing zip extraction / yaw-pitch / sync logic is shared with the
token-gate variant.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evals.action_anticipation_frozen.models import AttentiveClassifier
from src.models.attentive_pooler import AttentivePooler
from src.utils.tensors import trunc_normal_

from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate

logger = logging.getLogger(__name__)

_DIAG_SEEN: set[str] = set()
_DIAG_LINES = 0
_DIAG_LIMIT_REPORTED = False


def _diag_enabled() -> bool:
    return os.environ.get("GAZE_RNN_DIAG", "0").lower() in {"1", "true", "yes", "on"}


def _diag_verbose() -> bool:
    return os.environ.get("GAZE_RNN_DIAG_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}


def _diag_max_lines() -> int:
    try:
        return int(os.environ.get("GAZE_RNN_DIAG_MAX_LINES", "256"))
    except ValueError:
        return 256


def _tensor_diag(name: str, tensor: Optional[torch.Tensor]) -> bool:
    """Log compact finite/range stats; return True when a non-finite value exists."""
    global _DIAG_LINES, _DIAG_LIMIT_REPORTED
    if tensor is None:
        return False
    with torch.no_grad():
        x = tensor.detach()
        finite = torch.isfinite(x)
        bad = not bool(finite.all().item())
        if not bad and not (_diag_enabled() and _diag_verbose()):
            return False
        if name in _DIAG_SEEN:
            return bad
        max_lines = _diag_max_lines()
        if max_lines >= 0 and _DIAG_LINES >= max_lines:
            if not _DIAG_LIMIT_REPORTED:
                logger.warning(
                    "Gaze RNN diag line cap reached (%d); suppressing additional tensor diagnostics. "
                    "Increase GAZE_RNN_DIAG_MAX_LINES or use a scratch JSONL trace for longer runs.",
                    max_lines,
                )
                _DIAG_LIMIT_REPORTED = True
            return bad
        _DIAG_SEEN.add(name)
        _DIAG_LINES += 1
        xf = x.float()
        finite_count = int(finite.sum().item())
        total = x.numel()
        if finite_count:
            vals = xf[finite]
            min_val = float(vals.min().item())
            max_val = float(vals.max().item())
            mean_val = float(vals.mean().item())
            absmax_val = float(vals.abs().max().item())
        else:
            min_val = max_val = mean_val = absmax_val = float("nan")
        logger.warning(
            "Gaze RNN diag tensor=%s shape=%s dtype=%s finite=%d/%d min=%.6g max=%.6g mean=%.6g absmax=%.6g",
            name,
            tuple(x.shape),
            x.dtype,
            finite_count,
            total,
            min_val,
            max_val,
            mean_val,
            absmax_val,
        )
        return bad


def _sanitize_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Replace non-finite values after logging the first compact diagnostic."""
    bad = _tensor_diag(name, tensor)
    if bad:
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def _module_param_diag(name: str, module: Optional[nn.Module]) -> bool:
    if module is None:
        return False
    bad = False
    for param_name, param in module.named_parameters(recurse=True):
        bad = _tensor_diag(f"{name}.{param_name}", param) or bad
    return bad


def _scalar_for_log(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().reshape(-1)[0].item())
    return float(value)


def _first_nonfinite_grad(enc: nn.Module) -> tuple[str, torch.Tensor] | tuple[None, None]:
    for name, param in enc.named_parameters(recurse=True):
        if param.grad is not None and not bool(torch.isfinite(param.grad.detach().float()).all().item()):
            return name, param
    return None, None


def _adaptive_segment_pool(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Pool a variable-length sequence to ``k`` tokens with a validity mask.

    Args:
        x: ``[B, T, D]`` token sequence.
        mask: ``[B, T]`` per-step validity (1.0 inside the valid prefix, 0.0 on padding).
        k: target number of output tokens.
    """
    B, T, D = x.shape
    if T <= k:
        # Right-pad to k by repeating the last valid token; cheap fallback rarely hit
        # because we pre-pad to traj_len >= K.
        pad = x[:, -1:, :].expand(B, k - T, D)
        return torch.cat([x, pad], dim=1)
    x_t = x.transpose(1, 2)  # [B, D, T]
    m = mask.unsqueeze(1).to(x_t.dtype)  # [B, 1, T]
    num = F.adaptive_avg_pool1d(x_t * m, k)
    den = F.adaptive_avg_pool1d(m, k).clamp(min=1e-6)
    pooled = (num / den).transpose(1, 2)  # [B, k, D]
    return pooled


class GazeTrajectoryEncoder(nn.Module):
    """BiGRU (or per-frame MLP) over a gaze trajectory, projected to ``embed_dim``.

    Output: ``[B, num_tokens, embed_dim]`` with a learnable modality embedding and
    LayerNorm applied. Samples whose gaze record is missing receive a learnable
    null token replicated to ``num_tokens`` so the downstream cross-attention
    sequence length stays uniform across the batch.
    """

    def __init__(
        self,
        embed_dim: int,
        mode: str = "rnn",
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        num_tokens: int = 64,
        modality_embed_std: float = 0.02,
        video_feat_dim: int = 0,
        video_proj_dim: int = 128,
        video_fusion: str = "nearest_concat",
        residual_alpha_init: float = 0.01,
    ):
        super().__init__()
        assert mode in {"rnn", "mlp"}, mode
        video_fusion = str(video_fusion).lower()
        supported_fusions = {"nearest_concat", "gated_nearest", "local_attention", "residual_conditioned"}
        if video_feat_dim > 0 and video_fusion not in supported_fusions:
            raise ValueError(f"Unsupported gaze RNN video_fusion={video_fusion}")
        self.mode = mode
        self.num_tokens = int(num_tokens)
        self.gaze_input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.video_feat_dim = int(video_feat_dim)
        self.video_proj_dim = int(video_proj_dim) if self.video_feat_dim > 0 else 0
        self.video_fusion = video_fusion if self.video_feat_dim > 0 else "none"
        if self.video_feat_dim > 0:
            self.video_proj = nn.Sequential(
                nn.LayerNorm(self.video_feat_dim),
                nn.Linear(self.video_feat_dim, self.video_proj_dim),
                nn.GELU(),
            )
        else:
            self.video_proj = None
        self.input_dim = self.gaze_input_dim + self.video_proj_dim

        def make_sequence_encoder(in_dim: int):
            if mode == "rnn":
                encoder = nn.GRU(
                    input_size=in_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                out_dim = hidden_dim * (2 if bidirectional else 1)
            else:
                encoder = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                out_dim = hidden_dim
            return encoder, out_dim

        if self.video_fusion == "residual_conditioned":
            self.encoder, self._out_dim = make_sequence_encoder(self.gaze_input_dim)
            self.cond_encoder, self._cond_out_dim = make_sequence_encoder(self.gaze_input_dim + self.video_proj_dim)
            self.cond_proj = nn.Linear(self._cond_out_dim, embed_dim)
            alpha = float(residual_alpha_init)
            alpha = min(max(alpha, 1e-6), 1.0 - 1e-6)
            self.fusion_alpha_logit = nn.Parameter(torch.tensor(np.log(alpha / (1.0 - alpha)), dtype=torch.float32))
        else:
            self.encoder, self._out_dim = make_sequence_encoder(self.input_dim)
            self.cond_encoder = None
            self.cond_proj = None
            self.fusion_alpha_logit = None

        if self.video_fusion == "gated_nearest":
            self.video_gate = nn.Sequential(
                nn.Linear(self.gaze_input_dim + self.video_proj_dim, self.video_proj_dim),
                nn.Sigmoid(),
            )
        else:
            self.video_gate = None

        if self.video_fusion == "local_attention":
            self.local_query = nn.Linear(self.gaze_input_dim, self.video_proj_dim)
        else:
            self.local_query = None

        self.proj = nn.Linear(self._out_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.modality_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.modality_embed, std=modality_embed_std)
        trunc_normal_(self.null_token, std=modality_embed_std)

        self._last_hidden: Optional[torch.Tensor] = None
        self._last_fusion_alpha: Optional[float] = None
        self._last_video_gate_mean: Optional[float] = None

    def _encode_sequence(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        with torch.cuda.amp.autocast(enabled=False):
            if self.mode == "rnn":
                h, _ = encoder(x)
                return h.float()
            return encoder(x).float()

    def _video_condition(self, traj: torch.Tensor, video_features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.video_proj is None:
            self._last_video_gate_mean = None
            return None
        B, T, _ = traj.shape
        if video_features is None:
            video_features = traj.new_zeros(B, T, self.video_feat_dim)
        traj = torch.nan_to_num(traj.float(), nan=0.0, posinf=0.0, neginf=0.0)
        video_features = torch.nan_to_num(video_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if _diag_enabled():
            _tensor_diag("traj.after_nan_to_num", traj)
            _tensor_diag("video_features.after_nan_to_num", video_features)
            _module_param_diag("video_proj", self.video_proj)
            _module_param_diag("video_gate", self.video_gate)
            _module_param_diag("local_query", self.local_query)

        if video_features.ndim == 4:
            # [B, T, K, D] -> local attention over K nearby observed tokens.
            Bv, Tv, K, D = video_features.shape
            with torch.cuda.amp.autocast(enabled=False):
                video_proj = self.video_proj(video_features.reshape(Bv * Tv * K, D)).reshape(Bv, Tv, K, self.video_proj_dim)
            video_proj = _sanitize_tensor("video_proj.local", video_proj)
            if self.video_fusion == "local_attention":
                with torch.cuda.amp.autocast(enabled=False):
                    query = self.local_query(traj).unsqueeze(2)
                    query = _sanitize_tensor("local_query", query)
                    scores = (query * video_proj).sum(dim=-1) / max(1.0, float(self.video_proj_dim) ** 0.5)
                    scores = _sanitize_tensor("local_scores", scores)
                    weights = torch.softmax(scores, dim=-1)
                weights = _sanitize_tensor("local_weights", weights)
                self._last_video_gate_mean = weights.max(dim=-1).values.detach().float().mean().item()
                out = (weights.unsqueeze(-1) * video_proj).sum(dim=2)
                return _sanitize_tensor("local_video_cond", out)
            video_cond = video_proj[:, :, 0, :]
        else:
            with torch.cuda.amp.autocast(enabled=False):
                video_cond = self.video_proj(video_features)
            video_cond = _sanitize_tensor("video_proj.nearest", video_cond)

        if self.video_fusion == "gated_nearest":
            with torch.cuda.amp.autocast(enabled=False):
                gate = self.video_gate(torch.cat([traj, video_cond], dim=-1))
            gate = _sanitize_tensor("gated_gate", gate).clamp(0.0, 1.0)
            self._last_video_gate_mean = gate.detach().float().mean().item()
            video_cond = gate * video_cond
            video_cond = _sanitize_tensor("gated_video_cond", video_cond)
        else:
            self._last_video_gate_mean = None
        return video_cond

    def forward(
        self,
        traj: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        sample_valid: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            traj: ``[B, T, gaze_input_dim]`` padded trajectory (zeros on padding).
            lengths: ``[B]`` valid length within ``T``; defaults to full ``T``.
            sample_valid: ``[B]`` bool; ``False`` means no gaze record at all for that sample.
            video_features: optional ``[B, T, video_feat_dim]`` nearest observed
                V-JEPA token per gaze sample, or ``[B, T, K, video_feat_dim]``
                local observed-token neighborhoods for local attention.
        """
        traj = torch.nan_to_num(traj.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if _diag_enabled():
            _module_param_diag("encoder", self.encoder)
            _module_param_diag("cond_encoder", self.cond_encoder)
            _module_param_diag("proj", self.proj)
            _module_param_diag("cond_proj", self.cond_proj)
            _module_param_diag("norm", self.norm)
        B, T, _ = traj.shape
        device = traj.device
        video_cond = self._video_condition(traj, video_features)

        if self.video_fusion == "residual_conditioned":
            h_base = self._encode_sequence(self.encoder, traj)
            h_base = _sanitize_tensor("residual.h_base_raw", h_base)
            with torch.cuda.amp.autocast(enabled=False):
                h_base = self.proj(h_base)
            h_base = _sanitize_tensor("residual.h_base_proj", h_base)
            if video_cond is None:
                video_cond = traj.new_zeros(B, T, self.video_proj_dim)
            h_cond = self._encode_sequence(self.cond_encoder, torch.cat([traj, video_cond], dim=-1))
            h_cond = _sanitize_tensor("residual.h_cond_raw", h_cond)
            with torch.cuda.amp.autocast(enabled=False):
                h_cond = self.cond_proj(h_cond)
            h_cond = _sanitize_tensor("residual.h_cond_proj", h_cond)
            alpha = torch.sigmoid(self.fusion_alpha_logit)
            alpha = _sanitize_tensor("residual.alpha", alpha).clamp(0.0, 1.0)
            self._last_fusion_alpha = alpha.detach().float().item()
        else:
            if video_cond is not None:
                traj = torch.cat([traj, video_cond], dim=-1)
            h = self._encode_sequence(self.encoder, traj)
            h = _sanitize_tensor("encoder.h_raw", h)
            with torch.cuda.amp.autocast(enabled=False):
                h = self.proj(h)
            h = _sanitize_tensor("encoder.h_proj", h)
            self._last_fusion_alpha = None

        if self.video_fusion == "residual_conditioned":
            mask_dtype = h_base.dtype
        else:
            mask_dtype = h.dtype
        if lengths is None:
            mask = torch.ones(B, T, device=device, dtype=mask_dtype)
        else:
            arange = torch.arange(T, device=device).unsqueeze(0)
            mask = (arange < lengths.clamp(min=1).unsqueeze(1)).to(mask_dtype)

        if self.video_fusion == "residual_conditioned":
            pooled_base = _adaptive_segment_pool(h_base, mask, self.num_tokens)
            pooled_cond = _adaptive_segment_pool(h_cond, mask, self.num_tokens)
            pooled = self.norm(pooled_base + alpha.to(pooled_base.dtype) * pooled_cond + self.modality_embed)
            pooled = _sanitize_tensor("residual.pooled", pooled)
        else:
            pooled = _adaptive_segment_pool(h, mask, self.num_tokens)
            pooled = self.norm(pooled + self.modality_embed)
            pooled = _sanitize_tensor("pooled", pooled)

        if sample_valid is not None:
            null = self.norm(self.null_token + self.modality_embed)  # [1, 1, D]
            null_expand = null.expand(B, self.num_tokens, -1)
            keep = sample_valid.view(B, 1, 1).to(pooled.dtype)
            pooled = keep * pooled + (1.0 - keep) * null_expand

        # Keep a detached copy for optional offline dumping; freed each forward.
        self._last_hidden = pooled.detach()
        return pooled


class GazeTrajectoryLoader:
    """Build per-batch gaze trajectories from dataloader metadata.

    Reuses ``GazeTokenGate._load_record`` so that yaw/pitch decoding, zip
    extraction and MPS-to-MP4 sync are not duplicated. Returns padded tensors
    sized to ``traj_len``; samples that have more than ``traj_len`` raw points
    are uniformly subsampled, samples without any gaze record receive
    ``sample_valid=False``.
    """

    def __init__(self, cfg: dict, gate: Optional[GazeTokenGate] = None):
        self.cfg = dict(cfg)
        if gate is None:
            gate_cfg = dict(cfg)
            gate_cfg["mode"] = "rnn_fuse"
            gate = GazeTokenGate(gate_cfg)
        self.gate = gate
        self.traj_len = int(cfg.get("traj_len", 1024))
        rnn_cfg = dict(cfg.get("rnn", {}))
        self.use_video_tokens = bool(rnn_cfg.get("use_video_tokens", False))
        self.history_sec = float(rnn_cfg.get("history_sec", 0.0) or 0.0)
        if self.history_sec < 0.0:
            raise ValueError(f"gaze.rnn.history_sec must be non-negative, got {self.history_sec}")
        if self.history_sec > 0.0 and self.use_video_tokens:
            raise ValueError(
                "gaze.rnn.history_sec currently supports the gaze-only branch only; "
                "disable gaze.rnn.use_video_tokens or implement overlap-only video conditioning."
            )
        self.video_fusion = str(rnn_cfg.get("video_fusion", "nearest_concat")).lower()
        supported_fusions = {"nearest_concat", "gated_nearest", "local_attention", "residual_conditioned"}
        if self.video_fusion not in supported_fusions:
            raise ValueError(f"Unsupported gaze RNN video_fusion={self.video_fusion}")
        self.local_temporal_radius = max(0, int(rnn_cfg.get("local_temporal_radius", 0)))
        self.local_spatial_radius = max(0, int(rnn_cfg.get("local_spatial_radius", 1)))
        self.local_token_count = (2 * self.local_temporal_radius + 1) * (2 * self.local_spatial_radius + 1) ** 2
        self.frames_per_clip = int(cfg.get("frames_per_clip", 32))
        self.patch_size = int(cfg.get("patch_size", 16))
        self.tubelet_size = int(cfg.get("tubelet_size", 2))
        self.crop_size = int(cfg.get("crop_size", 384))
        self.grid_size = max(1, self.crop_size // max(1, self.patch_size))
        self.observed_token_count = int(
            cfg.get(
                "observed_token_count",
                max(1, self.frames_per_clip // max(1, self.tubelet_size)) * self.grid_size * self.grid_size,
            )
        )

    def load_batch(self, metadata, device: torch.device, video_tokens: Optional[torch.Tensor] = None):
        records = []
        for meta in metadata:
            traj = self._load_one(meta)
            records.append(traj)
        B = len(records)
        T = self.traj_len
        input_dim = int(self.cfg.get("rnn", {}).get("input_dim", 3))
        padded = torch.zeros(B, T, input_dim, dtype=torch.float32)
        lengths = torch.zeros(B, dtype=torch.long)
        sample_valid = torch.zeros(B, dtype=torch.bool)
        video_features = None
        if self.use_video_tokens:
            if video_tokens is None:
                raise ValueError("gaze.rnn.use_video_tokens=true requires video tokens")
            if self.video_fusion == "local_attention":
                video_features = torch.zeros(B, T, self.local_token_count, video_tokens.shape[-1], device=device, dtype=torch.float32)
            else:
                video_features = torch.zeros(B, T, video_tokens.shape[-1], device=device, dtype=torch.float32)
            video_tokens_detached = video_tokens.detach().to(torch.float32)
        for i, traj in enumerate(records):
            if traj is None or traj.shape[0] < 2:
                continue
            sample_valid[i] = True
            n_raw = traj.shape[0]
            if n_raw > T:
                idx = np.linspace(0, n_raw - 1, T).astype(np.int64)
                padded[i] = torch.from_numpy(traj[idx])
                lengths[i] = T
                if video_features is not None:
                    if self.video_fusion == "local_attention":
                        video_features[i] = self._local_video_features(video_tokens_detached[i], traj[idx], idx, n_raw)
                    else:
                        video_features[i] = self._nearest_video_features(video_tokens_detached[i], traj[idx], idx, n_raw)
            else:
                padded[i, :n_raw] = torch.from_numpy(traj)
                lengths[i] = n_raw
                if video_features is not None:
                    idx = np.arange(n_raw, dtype=np.int64)
                    if self.video_fusion == "local_attention":
                        video_features[i, :n_raw] = self._local_video_features(video_tokens_detached[i], traj, idx, n_raw)
                    else:
                        video_features[i, :n_raw] = self._nearest_video_features(video_tokens_detached[i], traj, idx, n_raw)
        return padded.to(device), lengths.to(device), sample_valid.to(device), video_features

    def _token_indices(self, traj: np.ndarray, raw_indices: np.ndarray, n_raw: int, n_obs: int):
        grid = self.grid_size
        spatial = grid * grid
        time_bins = max(1, n_obs // spatial)
        if n_raw <= 1:
            t_idx = np.zeros(traj.shape[0], dtype=np.int64)
        else:
            t_idx = np.rint(raw_indices.astype(np.float64) / float(n_raw - 1) * (time_bins - 1)).astype(np.int64)
        xy = np.clip(traj[:, :2] + 0.5, 0.0, 1.0)
        x_idx = np.rint(xy[:, 0] * (grid - 1)).astype(np.int64)
        y_idx = np.rint(xy[:, 1] * (grid - 1)).astype(np.int64)
        return t_idx, x_idx, y_idx, time_bins

    def _nearest_video_features(
        self,
        tokens_1: torch.Tensor,
        traj: np.ndarray,
        raw_indices: np.ndarray,
        n_raw: int,
    ) -> torch.Tensor:
        """Select nearest observed V-JEPA tubelet/patch token for each gaze point."""
        n_obs = min(int(tokens_1.shape[0]), self.observed_token_count)
        if n_obs <= 0:
            return torch.zeros(traj.shape[0], tokens_1.shape[-1], device=tokens_1.device, dtype=torch.float32)

        grid = self.grid_size
        spatial = grid * grid
        t_idx, x_idx, y_idx, _time_bins = self._token_indices(traj, raw_indices, n_raw, n_obs)
        token_idx = t_idx * spatial + y_idx * grid + x_idx
        token_idx = np.clip(token_idx, 0, n_obs - 1)
        return tokens_1[torch.from_numpy(token_idx).long().to(tokens_1.device)]

    def _local_video_features(
        self,
        tokens_1: torch.Tensor,
        traj: np.ndarray,
        raw_indices: np.ndarray,
        n_raw: int,
    ) -> torch.Tensor:
        """Select a small spatiotemporal token neighborhood around each gaze point."""
        n_obs = min(int(tokens_1.shape[0]), self.observed_token_count)
        if n_obs <= 0:
            return torch.zeros(
                traj.shape[0],
                self.local_token_count,
                tokens_1.shape[-1],
                device=tokens_1.device,
                dtype=torch.float32,
            )

        grid = self.grid_size
        spatial = grid * grid
        t_idx, x_idx, y_idx, time_bins = self._token_indices(traj, raw_indices, n_raw, n_obs)
        offsets = []
        for dt in range(-self.local_temporal_radius, self.local_temporal_radius + 1):
            for dy in range(-self.local_spatial_radius, self.local_spatial_radius + 1):
                for dx in range(-self.local_spatial_radius, self.local_spatial_radius + 1):
                    tt = np.clip(t_idx + dt, 0, time_bins - 1)
                    yy = np.clip(y_idx + dy, 0, grid - 1)
                    xx = np.clip(x_idx + dx, 0, grid - 1)
                    offsets.append(tt * spatial + yy * grid + xx)
        token_idx = np.stack(offsets, axis=1)
        token_idx = np.clip(token_idx, 0, n_obs - 1)
        return tokens_1[torch.from_numpy(token_idx).long().to(tokens_1.device)]

    def _load_one(self, meta) -> Optional[np.ndarray]:
        if meta is None:
            return None
        video_id = str(meta.get("video_id"))
        record = self.gate._load_record(video_id)  # noqa: SLF001 - intentional reuse
        if record is None or record.xy_norm is None or record.timestamps_us.size == 0:
            return None
        frame_indices = meta.get("frame_indices")
        if torch.is_tensor(frame_indices):
            frame_indices = frame_indices.detach().cpu().numpy()
        frame_indices = np.asarray(frame_indices, dtype=np.float64)
        vfps = meta.get("vfps", 30.0)
        if torch.is_tensor(vfps):
            vfps = float(vfps.detach().cpu())
        if frame_indices.size < 2 or vfps <= 0:
            return None

        mp4_t1_ns = float(frame_indices.max()) / vfps * 1e9
        if self.history_sec > 0.0:
            # Extend gaze history before the observed video clip, but never before
            # the source video start. The final part of the gaze window overlaps
            # the actual V-JEPA input clip.
            mp4_t0_ns = max(0.0, mp4_t1_ns - self.history_sec * 1e9)
        else:
            mp4_t0_ns = float(frame_indices.min()) / vfps * 1e9
        if record.sync is not None and {"mp4_time_ns", "vrs_device_time_ns"}.issubset(record.sync.columns):
            sync_mp4 = record.sync["mp4_time_ns"].to_numpy(dtype=np.float64)
            sync_vrs = record.sync["vrs_device_time_ns"].to_numpy(dtype=np.float64)
            vrs = np.interp([mp4_t0_ns, mp4_t1_ns], sync_mp4, sync_vrs)
            q_us = vrs / 1000.0
        else:
            q_us = np.array([mp4_t0_ns, mp4_t1_ns]) / 1000.0
        q_us_0, q_us_1 = float(q_us[0]), float(q_us[1])

        ts = record.timestamps_us
        mask = (ts >= q_us_0) & (ts <= q_us_1)
        if not mask.any():
            return None
        xy = record.xy_norm[mask]  # [N, 2] in [0,1]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if xy.shape[0] < 2:
            return None
        traj = np.concatenate(
            [xy.astype(np.float32) - 0.5, np.ones((xy.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        return traj


class PoseTrajectoryLoader:
    """Build per-batch SLAM pose trajectories from dataloader metadata."""

    def __init__(self, cfg: dict, gate: Optional[GazeTokenGate] = None):
        self.cfg = dict(cfg)
        pose_cfg = dict(cfg.get("pose", {}))
        pose_cfg.setdefault("gaze_root", cfg.get("gaze_root"))
        pose_cfg.setdefault("sync_root", cfg.get("sync_root"))
        if gate is None:
            gate = GazeTokenGate({"mode": "none", "gaze_root": cfg.get("gaze_root"), "sync_root": cfg.get("sync_root")})
        from app.hdepic_lora_action_anticipation.pose_slam import SlamPoseLoader

        self.pose_loader = SlamPoseLoader(pose_cfg, gate=gate)
        self.traj_len = int(cfg.get("traj_len", 1024))
        self.input_dim = self.pose_loader.input_dim
        self.history_sec = float(pose_cfg.get("history_sec", 0.0) or 0.0)

    def _load_one(self, meta) -> Optional[np.ndarray]:
        return self.pose_loader.query_clip_features(meta)

    def load_batch(
        self,
        metadata,
        device: torch.device,
        video_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        del video_tokens
        records = [self._load_one(meta) for meta in metadata]
        B = len(records)
        T = self.traj_len
        D = self.input_dim
        padded = torch.zeros(B, T, D, dtype=torch.float32)
        lengths = torch.zeros(B, dtype=torch.long)
        sample_valid = torch.zeros(B, dtype=torch.bool)
        for i, traj in enumerate(records):
            if traj is None or traj.shape[0] < 2:
                continue
            sample_valid[i] = True
            n_raw = traj.shape[0]
            if n_raw > T:
                # Avg pooling downsample (audio-style) instead of point picking.
                # This reduces aliasing when raw SLAM sampling rate >> traj_len.
                # traj: [N, D] -> padded: [T, D]
                edges = np.linspace(0, n_raw, T + 1, dtype=np.float64)
                out = np.zeros((T, D), dtype=np.float32)
                for t in range(T):
                    s = int(edges[t])
                    e = int(edges[t + 1])
                    if e <= s:
                        e = min(s + 1, n_raw)
                    seg = traj[s:e]
                    # seg is never empty due to guard above
                    out[t] = seg.mean(axis=0, dtype=np.float32)
                padded[i] = torch.from_numpy(out)
                lengths[i] = T
            else:
                padded[i, :n_raw] = torch.from_numpy(traj)
                lengths[i] = n_raw
        return padded.to(device), lengths.to(device), sample_valid.to(device), None


class GazeFusedAttentivePooler(AttentivePooler):
    """``AttentivePooler`` whose cross-attn K/V can be augmented with extra tokens.

    The self-attention blocks still operate only on the video token stream; gaze
    tokens are concatenated immediately before cross-attention so they are not
    mixed by LoRA-finetuned self-attn weights that were trained for video only.
    """

    def forward(self, x, extra_kv: Optional[torch.Tensor] = None):  # type: ignore[override]
        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x, False, None, use_reentrant=False)
                else:
                    x = blk(x)
        if extra_kv is not None:
            x = torch.cat([x, extra_kv.to(x.dtype)], dim=1)
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        return q


class GazeFusedAttentiveClassifier(AttentiveClassifier):
    """Action-anticipation classifier whose pooler accepts gaze tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.pooler
        new_pooler = GazeFusedAttentivePooler.__new__(GazeFusedAttentivePooler)
        new_pooler.__dict__ = old.__dict__
        new_pooler.__class__ = GazeFusedAttentivePooler
        self.pooler = new_pooler
        self.gaze_encoder: Optional[GazeTrajectoryEncoder] = None
        self.pose_encoder: Optional[GazeTrajectoryEncoder] = None

    def attach_gaze_encoder(self, encoder: GazeTrajectoryEncoder):
        self.gaze_encoder = encoder

    def attach_pose_encoder(self, encoder: GazeTrajectoryEncoder):
        self.pose_encoder = encoder

    def forward(self, x, gaze_tokens: Optional[torch.Tensor] = None):  # type: ignore[override]
        if torch.isnan(x).any():
            print("Nan detected at output of encoder")
            exit(1)
        x = self.pooler(x, extra_kv=gaze_tokens)
        if not self.action_only:
            x_verb, x_noun, x_action = x[:, 0, :], x[:, 1, :], x[:, 2, :]
            return dict(
                verb=self.verb_classifier(x_verb),
                noun=self.noun_classifier(x_noun),
                action=self.action_classifier(x_action),
            )
        x_action = x[:, 0, :]
        return dict(action=self.action_classifier(x_action))


def call_classifier(c: nn.Module, tokens: torch.Tensor, gaze_tokens: Optional[torch.Tensor]):
    """Forward a classifier with optional gaze tokens, transparent for non-RNN modes."""
    inner = c.module if hasattr(c, "module") else c
    if gaze_tokens is not None and isinstance(inner, GazeFusedAttentiveClassifier):
        return c(tokens, gaze_tokens=gaze_tokens)
    return c(tokens)


def encode_gaze_tokens(
    classifier: nn.Module,
    metadata,
    traj_loader: GazeTrajectoryLoader,
    device: torch.device,
    video_tokens: Optional[torch.Tensor] = None,
    gaze_batch: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = None,
) -> Optional[torch.Tensor]:
    """Run the gaze RNN/MLP encoder for a batch; returns ``[B, K, D]`` or ``None``.

    If the classifier has no attached gaze encoder, returns ``None`` so the
    caller can fall back to the plain probe call.
    """
    if metadata is None:
        return None
    inner = classifier.module if hasattr(classifier, "module") else classifier
    encoder = getattr(inner, "gaze_encoder", None)
    if encoder is None:
        return None
    if gaze_batch is None:
        gaze_batch = load_gaze_batch(metadata, traj_loader, device, video_tokens=video_tokens)
    traj, lengths, sample_valid, video_features = gaze_batch
    return encoder(traj, lengths=lengths, sample_valid=sample_valid, video_features=video_features)


def load_gaze_batch(
    metadata,
    traj_loader: GazeTrajectoryLoader,
    device: torch.device,
    video_tokens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load gaze trajectory and optional nearest video-token features once per batch."""
    return traj_loader.load_batch(metadata, device, video_tokens=video_tokens)


def load_pose_batch(
    metadata,
    traj_loader: PoseTrajectoryLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    return traj_loader.load_batch(metadata, device)


def _encode_with_encoder(
    classifier: nn.Module,
    encoder: GazeTrajectoryEncoder,
    traj: torch.Tensor,
    lengths: torch.Tensor,
    sample_valid: torch.Tensor,
    video_features: Optional[torch.Tensor],
) -> torch.Tensor:
    return encoder(traj, lengths=lengths, sample_valid=sample_valid, video_features=video_features)


def encode_pose_tokens(
    classifier: nn.Module,
    metadata,
    traj_loader: PoseTrajectoryLoader,
    device: torch.device,
    pose_batch: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]] = None,
) -> Optional[torch.Tensor]:
    if metadata is None:
        return None
    inner = classifier.module if hasattr(classifier, "module") else classifier
    encoder = getattr(inner, "pose_encoder", None)
    if encoder is None:
        return None
    if pose_batch is None:
        pose_batch = load_pose_batch(metadata, traj_loader, device)
    traj, lengths, sample_valid, _ = pose_batch
    return _encode_with_encoder(classifier, encoder, traj, lengths, sample_valid, None)


def encode_fusion_tokens(
    classifier: nn.Module,
    metadata,
    device: torch.device,
    gaze_loader: Optional[GazeTrajectoryLoader] = None,
    pose_loader: Optional[PoseTrajectoryLoader] = None,
    video_tokens: Optional[torch.Tensor] = None,
    gaze_batch: Optional[tuple] = None,
    pose_batch: Optional[tuple] = None,
) -> Optional[torch.Tensor]:
    """Encode gaze and/or pose branches; concat tokens on the sequence axis."""
    parts: list[torch.Tensor] = []
    if gaze_loader is not None:
        gt = encode_gaze_tokens(
            classifier,
            metadata,
            gaze_loader,
            device,
            video_tokens=video_tokens,
            gaze_batch=gaze_batch,
        )
        if gt is not None:
            parts.append(gt)
    if pose_loader is not None:
        pt = encode_pose_tokens(classifier, metadata, pose_loader, device, pose_batch=pose_batch)
        if pt is not None:
            parts.append(pt)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=1)


def _build_trajectory_encoder(embed_dim: int, rnn_cfg: dict) -> GazeTrajectoryEncoder:
    return GazeTrajectoryEncoder(
        embed_dim=embed_dim,
        mode=str(rnn_cfg.get("mode_impl", "rnn")),
        input_dim=int(rnn_cfg.get("input_dim", 3)),
        hidden_dim=int(rnn_cfg.get("hidden_dim", 256)),
        num_layers=int(rnn_cfg.get("num_layers", 2)),
        bidirectional=bool(rnn_cfg.get("bidirectional", True)),
        dropout=float(rnn_cfg.get("dropout", 0.1)),
        num_tokens=int(rnn_cfg.get("num_tokens", 64)),
        modality_embed_std=float(rnn_cfg.get("modality_embed_std", 0.02)),
        video_feat_dim=embed_dim if bool(rnn_cfg.get("use_video_tokens", False)) else 0,
        video_proj_dim=int(rnn_cfg.get("video_proj_dim", 128)),
        video_fusion=str(rnn_cfg.get("video_fusion", "nearest_concat")),
        residual_alpha_init=float(rnn_cfg.get("residual_alpha_init", 0.01)),
    )


def attach_gaze_encoder_to_classifier(
    classifier: GazeFusedAttentiveClassifier,
    embed_dim: int,
    rnn_cfg: dict,
) -> GazeTrajectoryEncoder:
    encoder = _build_trajectory_encoder(embed_dim, rnn_cfg)
    classifier.attach_gaze_encoder(encoder)
    return encoder


def attach_pose_encoder_to_classifier(
    classifier: GazeFusedAttentiveClassifier,
    embed_dim: int,
    rnn_cfg: dict,
) -> GazeTrajectoryEncoder:
    encoder = _build_trajectory_encoder(embed_dim, rnn_cfg)
    classifier.attach_pose_encoder(encoder)
    return encoder


def gaze_fusion_monitor(classifier: nn.Module) -> dict[str, float]:
    """Return lightweight diagnostics from the most recent gaze encoder forward."""
    inner = classifier.module if hasattr(classifier, "module") else classifier
    enc = getattr(inner, "gaze_encoder", None)
    if enc is None:
        return {}
    stats = {}
    if getattr(enc, "_last_fusion_alpha", None) is not None:
        stats["alpha"] = float(enc._last_fusion_alpha)
    if getattr(enc, "_last_video_gate_mean", None) is not None:
        key = "attn_max" if getattr(enc, "video_fusion", None) == "local_attention" else "gate"
        stats[key] = float(enc._last_video_gate_mean)
    return stats


class GazeHiddenDump:
    """Save a small held-out batch of gaze encoder outputs to disk.

    Intended for offline visualization of GRU hidden trajectories around
    saccade / fixation boundaries. Only rank 0 writes, and only the first
    ``max_batches`` batches of a single validation pass are recorded.
    """

    def __init__(self, cfg: dict, output_dir, rank: int):
        from pathlib import Path

        self.enabled = bool(cfg.get("enabled", False)) and rank == 0
        self.max_batches = int(cfg.get("max_batches", 4))
        self.path = None
        if self.enabled:
            base = Path(cfg.get("path") or (output_dir or "."))
            base.mkdir(parents=True, exist_ok=True)
            self.path = base / "gaze_rnn_hidden_states.pt"
        self.records: list = []

    def add(self, classifier, metadata, gaze_tokens):
        if not self.enabled or gaze_tokens is None or len(self.records) >= self.max_batches:
            return
        ids = [str((m or {}).get("video_id")) for m in metadata]
        starts = [int((m or {}).get("start_frame", -1)) for m in metadata]
        self.records.append(
            {
                "video_ids": ids,
                "start_frames": starts,
                "gaze_tokens": gaze_tokens.detach().to(torch.float32).cpu(),
            }
        )

    def flush(self):
        if not self.enabled or not self.records:
            return
        torch.save(self.records, self.path)
        logger.info("Wrote gaze RNN hidden-state dump to %s (%d batches)", self.path, len(self.records))
        self.records = []


def gaze_encoder_param_names(classifier: nn.Module) -> set[str]:
    """Return parameter names belonging to gaze/pose trajectory encoders."""
    names: set[str] = set()
    for name, _p in classifier.named_parameters():
        bare = name.split("module.", 1)[-1] if name.startswith("module.") else name
        if bare.startswith("gaze_encoder.") or bare.startswith("pose_encoder."):
            names.add(name)
    return names


def clip_gaze_encoder_grads(
    classifier: nn.Module,
    max_norm: float = 1.0,
    *,
    head_idx: Optional[int] = None,
    itr: Optional[int] = None,
    loss_info: Optional[dict[str, Any]] = None,
    scaler_scale: Optional[float] = None,
) -> bool:
    """Clip gaze encoder gradients; return False if non-finite grads were found."""
    inner = classifier.module if hasattr(classifier, "module") else classifier
    encoders = [getattr(inner, "gaze_encoder", None), getattr(inner, "pose_encoder", None)]
    encoders = [e for e in encoders if e is not None]
    if not encoders:
        return True
    params = [p for e in encoders for p in e.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return True
    bad_name, _bad_param = None, None
    for enc in encoders:
        bad_name, _bad_param = _first_nonfinite_grad(enc)
        if bad_name is not None:
            break
    if bad_name is not None:
        logger.warning(
            "Non-finite gaze encoder gradient detected; itr=%s head=%s param=%s loss=%s scaler_scale=%s; "
            "skipping optimizer step for this classifier",
            itr,
            head_idx,
            bad_name,
            None if loss_info is None else _scalar_for_log(loss_info.get("total")),
            scaler_scale,
        )
        return False
    if max_norm and max_norm > 0:
        total_norm = torch.nn.utils.clip_grad_norm_(params, float(max_norm), error_if_nonfinite=False)
        if not torch.isfinite(total_norm):
            logger.warning(
                "Non-finite gaze encoder grad norm detected; itr=%s head=%s loss=%s scaler_scale=%s total_norm=%s; "
                "skipping optimizer step for this classifier",
                itr,
                head_idx,
                None if loss_info is None else _scalar_for_log(loss_info.get("total")),
                scaler_scale,
                _scalar_for_log(total_norm),
            )
            return False
    return True
