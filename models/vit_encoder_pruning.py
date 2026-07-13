"""ViT encoder forward extensions for loss-aware token pruning."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from loss_aware_pruning import (
    LossAwarePruningConfig,
    LossAwareScoreCollector,
    build_prune_ratio_schedule,
    expected_final_keep_count,
    lookup_calibrated_scores,
    resolve_prune_layers,
    topk_token_prune,
)


def _resolve_video_geometry(encoder: nn.Module, x: torch.Tensor):
    if x.ndim == 4:
        _, _, h, w = x.shape
        t = 1
    elif x.ndim == 5:
        _, _, t, h, w = x.shape
        t = t // encoder.tubelet_size
    else:
        raise ValueError(f"expected image/video tensor, got shape {tuple(x.shape)}")
    h_patches = h // encoder.patch_size
    w_patches = w // encoder.patch_size
    if not encoder.handle_nonsquare_inputs:
        t = h_patches = w_patches = None
    return t, h_patches, w_patches


def forward_encoder_with_hooks(
    encoder: nn.Module,
    x: torch.Tensor,
    masks=None,
    *,
    prune_layers: set[int] | None = None,
    prune_ratio_schedule: dict[int, float] | None = None,
    score_provider: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    score_collector: LossAwareScoreCollector | None = None,
    gp: int | None = None,
    round_to_frame_tokens: bool = True,
    protected_indices: torch.Tensor | None = None,
    return_intermediate: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """Encoder forward with optional cascade pruning and score collection."""
    from src.masks.utils import apply_masks

    if masks is not None and not isinstance(masks, list):
        masks = [masks]

    t, h_patches, w_patches = _resolve_video_geometry(encoder, x)

    if not encoder.use_rope:
        pos_embed = encoder.interpolate_pos_encoding(x, encoder.pos_embed)
        x = encoder.patch_embed(x)
        x += pos_embed
    else:
        x = encoder.patch_embed(x)

    if masks is not None:
        x = apply_masks(x, masks)
        token_pos = torch.cat(masks, dim=0).to(x.device)
    else:
        token_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)

    if encoder.out_layers is not None:
        raise RuntimeError("loss-aware pruning does not support encoder.out_layers")

    prune_layers = prune_layers or set()
    prune_ratio_schedule = prune_ratio_schedule or {}
    return_intermediate = return_intermediate or set()
    intermediates: dict[int, torch.Tensor] = {}

    for layer_idx, blk in enumerate(encoder.blocks):
        if encoder.use_activation_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                blk,
                x,
                token_pos,
                None,
                T=t,
                H_patches=h_patches,
                W_patches=w_patches,
                use_reentrant=False,
            )
        else:
            x = blk(
                x,
                mask=token_pos,
                attn_mask=None,
                T=t,
                H_patches=h_patches,
                W_patches=w_patches,
            )

        if score_collector is not None:
            x = score_collector.maybe_retain(layer_idx, x)
        if layer_idx in return_intermediate:
            intermediates[layer_idx] = x

        if layer_idx not in prune_layers:
            continue

        if score_provider is None:
            raise RuntimeError(
                f"pruning at layer {layer_idx} requires score_provider (calibrated scores)"
            )
        score = score_provider(layer_idx, x, token_pos)
        r_l = float(prune_ratio_schedule[layer_idx])
        x, keep_idx = topk_token_prune(
            x,
            score,
            conditional_prune_ratio=r_l,
            gp=gp,
            round_to_frame_tokens=round_to_frame_tokens,
            protected_indices=protected_indices,
        )
        token_pos = token_pos.gather(1, keep_idx)

    if encoder.norm is not None:
        x = encoder.norm(x)
    return x, token_pos, intermediates


class LossAwareEncoderPruner:
    """Runtime encoder patch using a saved calibrated cascade pruning policy."""

    def __init__(
        self,
        encoder: nn.Module,
        config: LossAwarePruningConfig,
        *,
        gp: int,
        num_tokens_full: int,
    ):
        if not config.per_layer_prune_ratios:
            raise ValueError("LossAwareEncoderPruner requires calibrated per_layer_prune_ratios")
        if not config.calibration_scores_path and not hasattr(config, "_calibrated_scores_cache"):
            raise ValueError(
                "LossAwareEncoderPruner requires calibration_scores_path "
                "(run offline calibration with JEPA gradients first)"
            )

        self.encoder = encoder
        self.config = config
        self.gp = gp
        self.num_tokens_full = num_tokens_full
        self._last_idx: torch.Tensor | None = None
        self._last_token_count: int | None = None
        self._orig_encoder_forward = encoder.forward
        self._score_collector: LossAwareScoreCollector | None = None

        self.prune_layers = set(resolve_prune_layers(config))
        self.prune_ratio_schedule = build_prune_ratio_schedule(config)

        if hasattr(config, "_calibrated_scores_cache"):
            self._calibrated_scores = config._calibrated_scores_cache  # type: ignore[attr-defined]
        else:
            device = next(encoder.parameters()).device
            self._calibrated_scores = config.load_calibrated_scores(device)

        pruner = self

        def _forward(x, masks=None):
            return pruner._forward(x, masks=masks)

        encoder.forward = _forward

    @property
    def keep_count(self) -> int:
        if self._last_token_count is not None:
            return self._last_token_count
        return expected_final_keep_count(
            self.num_tokens_full,
            self.prune_ratio_schedule,
            gp=self.gp,
            round_to_frame_tokens=self.config.round_to_frame_tokens,
        )

    def begin_score_collection(self) -> LossAwareScoreCollector:
        self._score_collector = LossAwareScoreCollector(sorted(self.prune_layers))
        return self._score_collector

    def _score_provider(self, layer_idx: int, z: torch.Tensor, token_pos: torch.Tensor) -> torch.Tensor:
        return lookup_calibrated_scores(layer_idx, token_pos, self._calibrated_scores)

    def _forward(self, x, masks=None):
        protected = None
        if self.config.protected_token_indices:
            protected = torch.tensor(self.config.protected_token_indices, device=x.device, dtype=torch.long)

        out, token_pos, _ = forward_encoder_with_hooks(
            self.encoder,
            x,
            masks=masks,
            prune_layers=self.prune_layers,
            prune_ratio_schedule=self.prune_ratio_schedule,
            score_provider=self._score_provider,
            score_collector=self._score_collector,
            gp=self.gp,
            round_to_frame_tokens=self.config.round_to_frame_tokens,
            protected_indices=protected,
        )
        self._last_idx = token_pos
        self._last_token_count = out.shape[1]
        return out

    def encode(self, clips: torch.Tensor):
        toks = self.encoder(clips)
        if self._last_idx is None:
            raise RuntimeError("loss-aware pruner did not record kept token positions")
        return toks, self._last_idx

    def remove(self):
        self.encoder.forward = self._orig_encoder_forward


def extend_vision_transformer_forward(encoder: nn.Module) -> None:
    def forward_with_pruning(
        x,
        masks=None,
        *,
        prune_layers=None,
        prune_ratio_schedule=None,
        score_provider=None,
        score_collector=None,
        gp=None,
        round_to_frame_tokens=True,
        protected_indices=None,
    ):
        out, token_pos, _ = forward_encoder_with_hooks(
            encoder,
            x,
            masks=masks,
            prune_layers=set(prune_layers or []),
            prune_ratio_schedule=prune_ratio_schedule or {},
            score_provider=score_provider,
            score_collector=score_collector,
            gp=gp,
            round_to_frame_tokens=round_to_frame_tokens,
            protected_indices=protected_indices,
        )
        return out, token_pos

    def forward_with_intermediate(x, masks=None, layers=None):
        _, token_pos, intermediates = forward_encoder_with_hooks(
            encoder,
            x,
            masks=masks,
            return_intermediate=set(layers or []),
        )
        return intermediates, token_pos

    encoder.forward_with_pruning = forward_with_pruning
    encoder.forward_with_intermediate = forward_with_intermediate


def make_single_layer_config(layer: int, prune_ratio: float) -> LossAwarePruningConfig:
    return LossAwarePruningConfig(
        enable_loss_aware_pruning=True,
        pruning_mode="single_layer",
        single_prune_layer=layer,
        global_prune_ratio=prune_ratio,
        per_layer_prune_ratios={layer: prune_ratio},
        importance_source="calibrated",
    )
