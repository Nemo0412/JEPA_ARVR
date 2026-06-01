"""Small helpers for gaze-map rasterization."""

from __future__ import annotations

import torch


def normalize_map_type(map_type: str | None) -> str:
    value = str(map_type or "binary").strip().lower()
    aliases = {
        "disk": "binary",
        "hard": "binary",
        "hard_disk": "binary",
        "distance_transform": "distance",
        "dist": "distance",
        "linear_distance": "distance",
    }
    value = aliases.get(value, value)
    if value not in {"binary", "distance"}:
        raise ValueError(f"Unsupported binary gaze map type={map_type!r}; expected binary or distance")
    return value


def rasterize_gaze_disk(
    xx: torch.Tensor,
    yy: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    radius_px: float,
    map_type: str = "binary",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Rasterize per-frame gaze maps.

    Args:
        xx, yy: broadcastable image grids.
        x, y: per-frame gaze centers shaped like ``(T, 1, 1)``.
        radius_px: disk radius in output pixels.
        map_type: ``binary`` gives a hard 0/1 disk. ``distance`` gives a
            clipped linear distance transform, 1 at the gaze point and 0 at or
            beyond ``radius_px``.
    """
    dtype = dtype or xx.dtype
    radius = max(float(radius_px), 1e-6)
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    if normalize_map_type(map_type) == "binary":
        return (dist2 <= radius * radius).to(dtype)
    dist = torch.sqrt(torch.clamp(dist2, min=0.0))
    return torch.clamp(1.0 - dist / radius, min=0.0, max=1.0).to(dtype)
