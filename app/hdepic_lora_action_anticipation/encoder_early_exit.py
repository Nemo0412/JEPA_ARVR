"""Intermediate encoder outputs for early-exit diagnostics.

Each selected encoder prefix uses the encoder's shared final normalization,
then follows the unchanged one-step predictor and classifier path. This keeps
the existing checkpoint usable without training encoder-specific exit heads.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


class EncoderExitProjector(nn.Module):
    """Per-token map from an encoder prefix into the final encoder space."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(tokens))


def encoder_prefix_outputs(
    encoder,
    clips: torch.Tensor,
    exit_depths: Iterable[int],
) -> dict[int, torch.Tensor]:
    """Return normalized encoder tokens after selected 1-indexed depths."""
    depths = tuple(sorted({int(depth) for depth in exit_depths}))
    total_depth = len(encoder.blocks)
    if not depths or depths[0] < 1 or depths[-1] > total_depth:
        raise ValueError(f"exit depths must be within [1, {total_depth}], got {depths}")
    if getattr(encoder, "out_layers", None) is not None:
        raise NotImplementedError("encoder early exit does not support hierarchical out_layers")

    if clips.ndim == 4:
        _, _, height, width = clips.shape
        time_patches = 1
    elif clips.ndim == 5:
        _, _, frames, height, width = clips.shape
        time_patches = frames // encoder.tubelet_size
    else:
        raise ValueError(f"expected 4D/5D visual input, got shape={tuple(clips.shape)}")
    height_patches = height // encoder.patch_size
    width_patches = width // encoder.patch_size
    if not encoder.handle_nonsquare_inputs:
        time_patches = height_patches = width_patches = None

    if encoder.use_rope:
        x = encoder.patch_embed(clips)
    else:
        pos_embed = encoder.interpolate_pos_encoding(clips, encoder.pos_embed)
        x = encoder.patch_embed(clips) + pos_embed

    outputs: dict[int, torch.Tensor] = {}
    wanted = set(depths)
    for depth, block in enumerate(encoder.blocks, start=1):
        x = block(
            x,
            mask=None,
            attn_mask=None,
            T=time_patches,
            H_patches=height_patches,
            W_patches=width_patches,
        )
        if depth in wanted:
            outputs[depth] = encoder.norm(x) if encoder.norm is not None else x
    return outputs


def _anticipation_tokens_from_encoded(
    model,
    encoded_full: torch.Tensor,
    anticipation_times: torch.Tensor,
) -> torch.Tensor:
    if model.no_predictor or model.num_steps != 1:
        raise ValueError("encoder early exit requires no_predictor=False and num_steps=1")

    batch_size, n_context, width = encoded_full.shape
    embed_dim = int(model.encoder.embed_dim)
    encoded_for_classifier = encoded_full[:, :, -embed_dim:] if width > embed_dim else encoded_full
    if model.no_encoder:
        encoded_for_classifier = encoded_for_classifier[:, :0]

    masks_x = torch.arange(n_context, device=encoded_full.device).unsqueeze(0).repeat(batch_size, 1)
    anticipation_steps = (
        anticipation_times * model.frames_per_second / model.tubelet_size
    ).to(torch.int64)
    target_start = n_context + int(model.grid_size**2) * anticipation_steps
    n_target = int(model.grid_size**2 * (model.num_output_frames // model.tubelet_size))
    masks_y = torch.arange(n_target, device=encoded_full.device).unsqueeze(0).repeat(batch_size, 1)
    masks_y = masks_y + target_start.unsqueeze(1)

    predicted = model.predictor(encoded_full, masks_x=masks_x, masks_y=masks_y)
    predicted_full = predicted[0] if isinstance(predicted, tuple) else predicted
    predicted_for_classifier = (
        predicted_full[:, :, -embed_dim:]
        if predicted_full.shape[-1] != embed_dim
        else predicted_full
    )
    return torch.cat([encoded_for_classifier, predicted_for_classifier], dim=1)


def anticipation_tokens_from_encoded(
    model,
    encoded_full: torch.Tensor,
    anticipation_times: torch.Tensor,
) -> torch.Tensor:
    """Run the unchanged predictor path from externally supplied encoder tokens."""
    return _anticipation_tokens_from_encoded(model, encoded_full, anticipation_times)


def anticipation_tokens_by_encoder_depth(
    model,
    clips: torch.Tensor,
    anticipation_times: torch.Tensor,
    exit_depths: Iterable[int],
) -> dict[int, torch.Tensor]:
    """Expose encoder prefix exits, each followed by the complete predictor."""
    encoded = encoder_prefix_outputs(model.encoder, clips, exit_depths)
    return {
        depth: _anticipation_tokens_from_encoded(model, tokens, anticipation_times)
        for depth, tokens in encoded.items()
    }
