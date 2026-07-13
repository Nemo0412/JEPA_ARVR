"""Temporal RoPE position remapping for out-of-distribution direct-mask horizons.

The ViT-g/384 predictor is trained with ``num_frames=64`` (32 temporal chunks).
Direct-mask future targets can request flat position indices beyond that table;
this module applies NTK-style temporal scaling so depth/frame RoPE angles stay
within the pretrained range while preserving spatial (H/W) indices.
"""

from __future__ import annotations

import torch


def ntk_temporal_scale(flat_positions: torch.Tensor, spatial: int, trained_grid_depth: int) -> float:
    """Return scale factor ``alpha`` such that ``t_eff = t / alpha`` maps into trained depth."""
    if flat_positions.numel() == 0:
        return 1.0
    max_t = int((flat_positions.max().item()) // spatial)
    max_trained_t = max(int(trained_grid_depth) - 1, 1)
    if max_t <= max_trained_t:
        return 1.0
    return float(max_t) / float(max_trained_t)


def remap_flat_positions_ntk_temporal(
    flat_ids: torch.Tensor,
    spatial: int,
    scale: float,
) -> torch.Tensor:
    """Remap flat patch indices by dividing only the temporal (depth) component."""
    if scale <= 1.0 + 1e-6:
        return flat_ids
    t = flat_ids // spatial
    s = flat_ids % spatial
    t_eff = torch.div(t.to(torch.float32), scale, rounding_mode="floor").to(torch.long)
    return t_eff * spatial + s


def remap_mask_pair_ntk_temporal(
    masks_x: torch.Tensor,
    masks_y: torch.Tensor,
    spatial: int,
    trained_grid_depth: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Remap context + target mask indices with a shared temporal NTK scale."""
    combined = torch.cat([masks_x.reshape(-1), masks_y.reshape(-1)])
    scale = ntk_temporal_scale(combined, spatial, trained_grid_depth)
    if scale <= 1.0 + 1e-6:
        return masks_x, masks_y, scale
    return (
        remap_flat_positions_ntk_temporal(masks_x, spatial, scale),
        remap_flat_positions_ntk_temporal(masks_y, spatial, scale),
        scale,
    )
