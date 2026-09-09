"""Temporal quota allocators for exact-budget post-encoder token pruning.

This module decides only how many tokens each temporal slot may retain.  It is
independent of the within-slot spatial selector and of the training path.  The
registered B17 comparison uses exact K=3840; historical K=4096 attention runs
are not budget-matched arms.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


# Mean final survivors per 0.25 s slot from
# outputs/egtea_stream_mtp_kvcache_prune/pattern/full/temporal_loss_aware.npy.
# Oldest -> newest, 40 slots, sum == 3840; the newest 16 slots sum to 2764
# (71.979% of the budget).
LOSS_AWARE_10S_TEMPORAL_QUOTAS: tuple[int, ...] = (
    85, 49, 54, 57, 58, 42, 47, 45, 39, 29,
    34, 41, 44, 23, 28, 25, 27, 27, 35, 69,
    42, 46, 71, 59, 71, 108, 123, 138, 164, 178,
    156, 186, 180, 169, 189, 190, 173, 230, 253, 256,
)
LOSS_AWARE_PRIMARY_BUDGET = 3840


def _capacity_project(weights: torch.Tensor, total: int, capacity: int) -> torch.Tensor:
    """Project non-negative weights to capped integer quotas with exact sum."""
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("weights must be a non-empty 1-D tensor")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    total = max(0, min(int(total), int(weights.numel()) * int(capacity)))
    if total == 0:
        return torch.zeros_like(weights, dtype=torch.long)
    if total == weights.numel() * capacity:
        return torch.full_like(weights, int(capacity), dtype=torch.long)

    w = weights.detach().to(dtype=torch.float64, device="cpu").clamp_min(0)
    if float(w.sum()) <= 0:
        w = torch.ones_like(w)
    continuous = torch.zeros_like(w)
    active = torch.ones(w.numel(), dtype=torch.bool)
    remaining = float(total)
    while bool(active.any()):
        active_w = w[active]
        if float(active_w.sum()) <= 0:
            active_w = torch.ones_like(active_w)
        proposal = remaining * active_w / active_w.sum()
        saturated_local = proposal >= float(capacity)
        active_idx = active.nonzero(as_tuple=False).flatten()
        if not bool(saturated_local.any()):
            continuous[active_idx] = proposal
            break
        saturated_idx = active_idx[saturated_local]
        continuous[saturated_idx] = float(capacity)
        active[saturated_idx] = False
        remaining = float(total) - float(continuous.sum())

    quotas = torch.floor(continuous).to(torch.long).clamp_(0, int(capacity))
    need = int(total - int(quotas.sum()))
    if need > 0:
        remainder = continuous - quotas.to(continuous.dtype)
        eligible = quotas < int(capacity)
        order = torch.argsort(
            remainder.masked_fill(~eligible, -1), descending=True, stable=True
        )
        for idx in order.tolist():
            if need <= 0:
                break
            if quotas[idx] < capacity:
                quotas[idx] += 1
                need -= 1
    if need != 0 or int(quotas.sum()) != total:
        raise RuntimeError(f"failed exact quota projection: got {int(quotas.sum())}, want {total}")
    return quotas


@dataclass(frozen=True)
class LossAwareTemporalQuotaProvider:
    """Relative-recency temporal budget initialized from loss-aware calibration."""

    total_budget: int = LOSS_AWARE_PRIMARY_BUDGET
    slot_capacity: int = 256
    reference_quotas: tuple[int, ...] = LOSS_AWARE_10S_TEMPORAL_QUOTAS

    def quotas(self, n_slots: int) -> torch.Tensor:
        n_slots = int(n_slots)
        if n_slots <= 0:
            raise ValueError("n_slots must be positive")
        if n_slots > len(self.reference_quotas):
            raise ValueError(
                f"loss-aware reference covers at most {len(self.reference_quotas)} slots, got {n_slots}"
            )
        target = min(int(self.total_budget), n_slots * int(self.slot_capacity))
        reference = torch.tensor(self.reference_quotas, dtype=torch.float64)
        weights = reference[-n_slots:]
        if n_slots == len(self.reference_quotas) and target == sum(self.reference_quotas):
            return reference.to(torch.long)
        return _capacity_project(weights, target, int(self.slot_capacity))


def uniform_temporal_quotas(n_slots: int, *, total: int, capacity: int) -> torch.Tensor:
    """Allocate an exact budget as evenly as possible across temporal slots."""
    return _capacity_project(torch.ones(int(n_slots)), int(total), int(capacity))


def recent_temporal_quotas(n_slots: int, *, total: int, capacity: int) -> torch.Tensor:
    """Fill newest slots first, with at most one partial boundary slot."""
    n_slots, total, capacity = int(n_slots), int(total), int(capacity)
    if n_slots <= 0 or capacity <= 0:
        raise ValueError("n_slots and capacity must be positive")
    remaining = min(max(total, 0), n_slots * capacity)
    quotas = torch.zeros(n_slots, dtype=torch.long)
    for slot in range(n_slots - 1, -1, -1):
        keep = min(capacity, remaining)
        quotas[slot] = keep
        remaining -= keep
    if remaining != 0:
        raise RuntimeError("recent quota allocation did not exhaust budget")
    return quotas


def permuted_temporal_quotas(
    base_quotas: torch.Tensor,
    row_keys: list[str] | tuple[str, ...],
    *,
    seed: int,
) -> torch.Tensor:
    """Independently permute one quota multiset per stable row key."""
    base = base_quotas.detach().to(dtype=torch.long, device="cpu")
    if base.ndim != 1 or base.numel() == 0:
        raise ValueError("base_quotas must be a non-empty vector")
    if not row_keys or len(set(str(key) for key in row_keys)) != len(row_keys):
        raise ValueError("row_keys must be non-empty and unique within a batch")
    rows = []
    for key in row_keys:
        payload = f"{int(seed)}|{str(key)}|temporal-quota".encode()
        keyed_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(keyed_seed % (2**63 - 1))
        rows.append(base[torch.randperm(base.numel(), generator=generator)])
    return torch.stack(rows)


def attention_sum_temporal_quotas(
    attention_scores: torch.Tensor,
    *,
    spatial_tokens: int,
    total: int,
    capacity: int,
) -> torch.Tensor:
    """Allocate each row's budget proportional to per-slot attention mass."""
    if attention_scores.ndim != 2 or attention_scores.shape[1] % int(spatial_tokens):
        raise ValueError("attention_scores must be [B,N] with N divisible by spatial_tokens")
    if not bool(torch.isfinite(attention_scores).all()):
        raise ValueError("attention scores contain non-finite values")
    batch, n_tokens = attention_scores.shape
    weights = attention_scores.detach().float().reshape(
        batch, n_tokens // spatial_tokens, spatial_tokens
    )
    weights = weights.clamp_min(0).sum(dim=2).cpu()
    return torch.stack(
        [_capacity_project(row, int(total), int(capacity)) for row in weights]
    )
