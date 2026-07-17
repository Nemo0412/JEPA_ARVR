"""Intermediate predictor outputs for early-exit diagnostics.

The implementation mirrors ``src.models.predictor.VisionTransformerPredictor``
and the one-step action-anticipation wrapper.  It intentionally keeps the
predictor's shared final norm/projection at every candidate exit so an existing
final classifier can be evaluated without training new exit heads.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch


def _project_exit(predictor, x, argsort, n_context: int) -> torch.Tensor:
    x = predictor.predictor_norm(x)
    if not predictor.return_all_tokens:
        reverse_argsort = torch.argsort(argsort, dim=1)
        x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
        x = x[:, n_context:]
    return predictor.predictor_proj(x)


def predictor_prefix_outputs(
    predictor,
    context_tokens: torch.Tensor,
    masks_x: torch.Tensor,
    masks_y: torch.Tensor,
    exit_depths: Iterable[int],
    *,
    mask_index: int = 1,
) -> dict[int, torch.Tensor]:
    """Return predictor outputs after selected 1-indexed transformer depths."""
    depths = tuple(sorted({int(depth) for depth in exit_depths}))
    total_depth = len(predictor.predictor_blocks)
    if not depths or depths[0] < 1 or depths[-1] > total_depth:
        raise ValueError(f"exit depths must be within [1, {total_depth}], got {depths}")
    if predictor.chop_last_n_tokens > 0:
        raise NotImplementedError("early-exit diagnostic does not support chop_last_n_tokens")

    masks_x_list = [masks_x] if not isinstance(masks_x, list) else masks_x
    masks_y_list = [masks_y] if not isinstance(masks_y, list) else masks_y
    batch_size = len(context_tokens) // len(masks_x_list)

    x = predictor.predictor_embed(context_tokens)
    n_context = x.shape[1]
    if not predictor.use_rope:
        pos = predictor.predictor_pos_embed.repeat(batch_size, 1, 1)
        x = x + apply_masks(pos, masks_x_list)

    mask_index %= predictor.num_mask_tokens
    target = predictor.mask_tokens[mask_index].repeat(batch_size, predictor.num_patches, 1)
    target = apply_masks(target, masks_y_list)
    if not predictor.use_rope:
        pos = predictor.predictor_pos_embed.repeat(batch_size, 1, 1)
        target_pos = apply_masks(pos, masks_y_list)
        target_pos = repeat_interleave_batch(target_pos, batch_size, repeat=len(masks_x_list))
        target = target + target_pos

    x = x.repeat(len(masks_x_list), 1, 1)
    x = torch.cat([x, target], dim=1)
    masks_x_cat = torch.cat(masks_x_list, dim=0)
    masks_y_cat = torch.cat(masks_y_list, dim=0)
    masks = torch.cat([masks_x_cat, masks_y_cat], dim=1)
    argsort = torch.argsort(masks, dim=1)
    masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
    x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)

    outputs: dict[int, torch.Tensor] = {}
    wanted = set(depths)
    for depth, block in enumerate(predictor.predictor_blocks, start=1):
        x = block(x, mask=masks, attn_mask=None)
        if depth in wanted:
            outputs[depth] = _project_exit(predictor, x, argsort, n_context)
    return outputs


def anticipation_tokens_by_predictor_depth(
    model,
    clips: torch.Tensor,
    anticipation_times: torch.Tensor,
    exit_depths: Iterable[int],
) -> dict[int, torch.Tensor]:
    """Run the encoder once and expose one-step predictor prefix exits."""
    if model.no_predictor or model.num_steps != 1:
        raise ValueError("predictor early exit requires no_predictor=False and num_steps=1")

    encoded_full = model.encoder(clips)
    batch_size, n_context, width = encoded_full.shape
    embed_dim = int(model.encoder.embed_dim)
    encoded_for_classifier = encoded_full[:, :, -embed_dim:] if width > embed_dim else encoded_full
    if model.no_encoder:
        encoded_for_classifier = encoded_for_classifier[:, :0]

    masks_x = torch.arange(n_context, device=clips.device).unsqueeze(0).repeat(batch_size, 1)
    anticipation_steps = (
        anticipation_times * model.frames_per_second / model.tubelet_size
    ).to(torch.int64)
    target_start = n_context + int(model.grid_size**2) * anticipation_steps
    n_target = int(model.grid_size**2 * (model.num_output_frames // model.tubelet_size))
    masks_y = torch.arange(n_target, device=clips.device).unsqueeze(0).repeat(batch_size, 1)
    masks_y = masks_y + target_start.unsqueeze(1)

    predicted = predictor_prefix_outputs(
        model.predictor,
        encoded_full,
        masks_x,
        masks_y,
        exit_depths,
    )
    outputs = {}
    for depth, target in predicted.items():
        target_last = target[:, :, -embed_dim:] if target.shape[-1] != embed_dim else target
        outputs[depth] = torch.cat([encoded_for_classifier, target_last], dim=1)
    return outputs
