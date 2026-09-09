"""Within-slot spatial selectors for fixed-budget token pruning.

This module decides only which spatial tokens survive after a temporal quota
has been supplied.  Temporal allocation algorithms live in
``temporal_budget_pruning.py`` and the training path remains independent.
"""
from __future__ import annotations

import hashlib
from typing import Literal

import torch


SelectorMode = Literal["uniform", "calib", "attention", "gaze", "gaze_shift", "random"]


def valid_gaze_types(types: torch.Tensor) -> torch.Tensor:
    """Return validity for the parser's two explicitly tracked gaze states.

    ``parse_gtea_gaze`` uses 1=Fixation, 2=Saccade, 3=all other raw labels
    (including Blink and ``-``), and 4=out-of-bounds.  Consequently type 3 is
    deliberately rejected rather than treated as a generic tracked state.
    """
    rounded = torch.round(types).to(torch.long)
    return (rounded == 1) | (rounded == 2)


def _uniform_farthest_order(grid_size: int) -> torch.Tensor:
    """Deterministic 2-D coverage ordering for a content-free spatial control."""
    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    yy, xx = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float64),
        torch.arange(grid_size, dtype=torch.float64),
        indexing="ij",
    )
    points = torch.stack([yy.flatten(), xx.flatten()], dim=1)
    center = torch.tensor([(grid_size - 1) / 2.0, (grid_size - 1) / 2.0])
    first = int(torch.argmin(((points - center) ** 2).sum(dim=1)))
    selected = [first]
    min_dist = ((points - points[first]) ** 2).sum(dim=1)
    min_dist[first] = -1
    while len(selected) < points.shape[0]:
        nxt = int(torch.argmax(min_dist))
        selected.append(nxt)
        dist = ((points - points[nxt]) ** 2).sum(dim=1)
        min_dist = torch.minimum(min_dist, dist)
        min_dist[torch.tensor(selected, dtype=torch.long)] = -1
    return torch.tensor(selected, dtype=torch.long)


class QuotaSpatialTokenSelector:
    """Select exactly ``q_t`` spatial tokens inside every temporal slot.

    Gaze coordinates are normalized ``(x, y)`` values aligned to the exact two
    decoded frames that form each tubelet.  If a tubelet has no valid tracked
    gaze, the gaze variants fall back locally to received-attention ranking;
    they never synthesize a center gaze point.
    """

    def __init__(self, grid_size: int = 16, tubelet_size: int = 2):
        self.grid_size = int(grid_size)
        self.tubelet_size = int(tubelet_size)
        if self.grid_size <= 0 or self.tubelet_size <= 0:
            raise ValueError("grid_size and tubelet_size must be positive")
        self.gp = self.grid_size**2
        self._uniform_order = _uniform_farthest_order(self.grid_size)
        yy, xx = torch.meshgrid(
            (torch.arange(self.grid_size, dtype=torch.float64) + 0.5) / self.grid_size,
            (torch.arange(self.grid_size, dtype=torch.float64) + 0.5) / self.grid_size,
            indexing="ij",
        )
        # Flattening matches encoder token order: h * grid + w.
        self._patch_xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    def select(
        self,
        *,
        mode: SelectorMode,
        quotas: torch.Tensor,
        attention_scores: torch.Tensor,
        calibrated_scores: torch.Tensor | None = None,
        gaze_xy: torch.Tensor | None = None,
        gaze_valid: torch.Tensor | None = None,
        random_keys: list[str] | tuple[str, ...] | None = None,
        random_seed: int = 17,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Return sorted original flat token indices, shape ``[B, sum(q)]``."""
        if mode not in ("uniform", "calib", "attention", "gaze", "gaze_shift", "random"):
            raise ValueError(f"unsupported selector mode: {mode}")
        if attention_scores.ndim != 2:
            raise ValueError("attention_scores must have shape [B, N]")
        batch, n_tokens = attention_scores.shape
        if n_tokens % self.gp:
            raise ValueError(f"token count {n_tokens} is not divisible by gp={self.gp}")
        n_slots = n_tokens // self.gp
        quotas = quotas.detach().to(dtype=torch.long, device="cpu")
        if quotas.shape == (n_slots,):
            quotas = quotas.unsqueeze(0).expand(batch, -1)
        elif quotas.shape != (batch, n_slots):
            raise ValueError(
                f"quotas must have shape ({n_slots},) or ({batch},{n_slots}), got {tuple(quotas.shape)}"
            )
        if bool((quotas < 0).any()) or bool((quotas > self.gp).any()):
            raise ValueError("every slot quota must be in [0, gp]")
        row_totals = quotas.sum(dim=1)
        if not bool((row_totals == row_totals[0]).all()):
            raise ValueError("every batch row must have the same total token budget")

        if mode == "calib":
            if calibrated_scores is None:
                raise ValueError("calib mode requires calibrated_scores")
            if calibrated_scores.ndim == 1:
                calibrated_scores = calibrated_scores.unsqueeze(0).expand(batch, -1)
            if tuple(calibrated_scores.shape) != (batch, n_tokens):
                raise ValueError(
                    f"calibrated_scores must be [N] or [B,N], got {tuple(calibrated_scores.shape)}"
                )

        if mode.startswith("gaze"):
            expected_frames = n_slots * self.tubelet_size
            if gaze_xy is None or gaze_valid is None:
                raise ValueError("gaze modes require gaze_xy and gaze_valid")
            if tuple(gaze_xy.shape) != (batch, expected_frames, 2):
                raise ValueError(
                    f"gaze_xy must be [B,{expected_frames},2], got {tuple(gaze_xy.shape)}"
                )
            if tuple(gaze_valid.shape) != (batch, expected_frames):
                raise ValueError(
                    f"gaze_valid must be [B,{expected_frames}], got {tuple(gaze_valid.shape)}"
                )
        if mode == "random":
            if random_keys is None or len(random_keys) != batch:
                raise ValueError(f"random mode requires exactly {batch} stable row keys")
            if len(set(str(key) for key in random_keys)) != batch:
                raise ValueError("random row keys must be unique within a batch")

        device = attention_scores.device
        uniform_order = self._uniform_order.to(device)
        patch_xy = self._patch_xy.to(device)
        per_batch: list[torch.Tensor] = []
        invalid_slots = 0
        gaze_slots = 0
        random_slots = 0
        for b in range(batch):
            chosen: list[torch.Tensor] = []
            for slot, quota_value in enumerate(quotas[b].tolist()):
                q = int(quota_value)
                if q == 0:
                    continue
                base = slot * self.gp
                if q == self.gp:
                    local = torch.arange(self.gp, device=device)
                elif mode == "uniform":
                    local = uniform_order[:q]
                elif mode == "attention":
                    scores = attention_scores[b, base : base + self.gp]
                    local = torch.argsort(scores, descending=True, stable=True)[:q]
                elif mode == "calib":
                    scores = calibrated_scores[b, base : base + self.gp]
                    local = torch.argsort(scores, descending=True, stable=True)[:q]
                elif mode == "random":
                    # A row/slot-keyed CPU generator makes the sample uniform
                    # without replacement and invariant to batch order/device.
                    payload = f"{int(random_seed)}|{str(random_keys[b])}|{slot}".encode()
                    keyed_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(keyed_seed % (2**63 - 1))
                    local = torch.randperm(self.gp, generator=generator)[:q].to(device)
                    random_slots += 1
                else:
                    frame_start = slot * self.tubelet_size
                    valid = gaze_valid[b, frame_start : frame_start + self.tubelet_size].bool()
                    if bool(valid.any()):
                        points = gaze_xy[b, frame_start : frame_start + self.tubelet_size][valid]
                        points = points.to(device=device, dtype=patch_xy.dtype)
                        if mode == "gaze_shift":
                            points = points.clone()
                            points[:, 0] = torch.remainder(points[:, 0] + 0.5, 1.0)
                        # max Gaussian is rank-equivalent to negative distance to
                        # the nearest valid gaze sample.
                        dist2 = ((patch_xy[:, None, :] - points[None, :, :]) ** 2).sum(dim=2)
                        scores = -dist2.min(dim=1).values
                        local = torch.argsort(scores, descending=True, stable=True)[:q]
                        gaze_slots += 1
                    else:
                        scores = attention_scores[b, base : base + self.gp]
                        local = torch.argsort(scores, descending=True, stable=True)[:q]
                        invalid_slots += 1
                chosen.append(base + local)
            flat = torch.cat(chosen).sort().values if chosen else torch.empty(0, device=device, dtype=torch.long)
            if flat.numel() != int(row_totals[b]):
                raise RuntimeError(
                    f"selector returned {flat.numel()} tokens, expected {int(row_totals[b])}"
                )
            if flat.numel() > 1 and not bool((flat[1:] > flat[:-1]).all()):
                raise RuntimeError("selected token indices must be unique and strictly increasing")
            per_batch.append(flat)
        stats = {
            "gaze_selected_slots": int(gaze_slots),
            "attention_fallback_slots": int(invalid_slots),
            "total_slots": int(batch * n_slots),
            "full_capacity_slots": int((quotas == self.gp).sum()),
        }
        if mode == "random":
            stats["random_sampled_slots"] = int(random_slots)
        return torch.stack(per_batch, dim=0), stats


def gather_tokens(tokens: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or keep_idx.ndim != 2 or tokens.shape[0] != keep_idx.shape[0]:
        raise ValueError("tokens must be [B,N,D] and keep_idx [B,K]")
    return tokens.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
