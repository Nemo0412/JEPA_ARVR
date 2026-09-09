"""Task-protected JEPA redundancy pruning primitives.

This module deliberately separates offline target construction from online
selection.  Offline counterfactual task utility is a protection target; JEPA
residual is a redundancy target.  The online policy is lexicographic rather
than a weighted sum of incomparable raw scores.
"""

from __future__ import annotations

import csv
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from app.hdepic_lora_action_anticipation.vit_encoder_loss_aware_pruning import (
    _resolve_video_geometry,
)


ORACLE_PROTOCOL_ID = "egtea-stream-mtp-pruning/task-protected-jepa-oracle-gate-l16-v1"
ONLINE_PROTOCOL_ID = "egtea-stream-mtp-pruning/task-protected-jepa-online-l16-v1"
SWAP_ORACLE_PROTOCOL_ID = (
    "egtea-stream-mtp-pruning/attention-jepa-disagreement-swap-oracle-l16-v2"
)
COMPLETION_E0_PROTOCOL_ID = (
    "egtea-stream-mtp-pruning/predictor-compensated-completion-l16-e0-v1"
)
EXPECTED_TRAIN_CSV_SHA256 = "6d604dded2f8ba875caace4d9266b87616bcfddb40c522f443a316786cdd2844"


def _as_batched_scores(scores: torch.Tensor, name: str) -> Tuple[torch.Tensor, bool]:
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
        squeezed = True
    elif scores.ndim == 2:
        squeezed = False
    else:
        raise ValueError(f"{name} must have shape [G] or [B,G], got {tuple(scores.shape)}")
    if not torch.is_floating_point(scores):
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(scores).all():
        raise ValueError(f"{name} contains non-finite values")
    return scores, squeezed


def _stable_topk_from_mask(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Select descending scores, breaking exact ties by lower group index."""
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if scores.shape != eligible.shape:
        raise ValueError("scores and eligible must have identical shapes")
    if count == 0:
        return torch.empty((scores.shape[0], 0), device=scores.device, dtype=torch.long)
    available = eligible.sum(dim=1)
    if (available < count).any():
        raise ValueError(
            f"requested {count} entries but minimum eligible count is {int(available.min().item())}"
        )
    masked = scores.masked_fill(~eligible, -torch.inf)
    order = torch.argsort(masked, dim=1, descending=True, stable=True)
    chosen = order[:, :count]
    if not eligible.gather(1, chosen).all():
        raise RuntimeError("internal selector error: an ineligible group was selected")
    return chosen


@dataclass(frozen=True)
class SpatiotemporalGroupLayout:
    """Regular groups over flattened (tubelet, row, column) token positions."""

    num_slots: int = 40
    grid_size: int = 16
    group_side: int = 4
    recent_slots: int = 4

    def __post_init__(self) -> None:
        if self.num_slots <= 0 or self.grid_size <= 0 or self.group_side <= 0:
            raise ValueError("num_slots, grid_size, and group_side must be positive")
        if self.grid_size % self.group_side:
            raise ValueError("grid_size must be divisible by group_side")
        if not 0 <= self.recent_slots <= self.num_slots:
            raise ValueError("recent_slots must be between zero and num_slots")

    @property
    def tokens_per_slot(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def tokens_per_group(self) -> int:
        return self.group_side * self.group_side

    @property
    def groups_per_axis(self) -> int:
        return self.grid_size // self.group_side

    @property
    def groups_per_slot(self) -> int:
        return self.groups_per_axis * self.groups_per_axis

    @property
    def num_groups(self) -> int:
        return self.num_slots * self.groups_per_slot

    @property
    def num_tokens(self) -> int:
        return self.num_slots * self.tokens_per_slot

    @property
    def num_recent_groups(self) -> int:
        return self.recent_slots * self.groups_per_slot

    def groups_for_budget(self, token_budget: int) -> int:
        if token_budget <= 0 or token_budget > self.num_tokens:
            raise ValueError(f"token budget must be in [1,{self.num_tokens}], got {token_budget}")
        if token_budget % self.tokens_per_group:
            raise ValueError(
                f"token budget {token_budget} is not divisible by group size {self.tokens_per_group}"
            )
        groups = token_budget // self.tokens_per_group
        if groups < self.num_recent_groups:
            raise ValueError(
                f"token budget keeps {groups} groups but recent core needs {self.num_recent_groups}"
            )
        return groups

    def token_indices(self, device: torch.device | str | None = None) -> torch.Tensor:
        groups: List[List[int]] = []
        for slot in range(self.num_slots):
            slot_offset = slot * self.tokens_per_slot
            for group_row in range(self.groups_per_axis):
                for group_col in range(self.groups_per_axis):
                    tokens: List[int] = []
                    row0 = group_row * self.group_side
                    col0 = group_col * self.group_side
                    for row in range(row0, row0 + self.group_side):
                        for col in range(col0, col0 + self.group_side):
                            tokens.append(slot_offset + row * self.grid_size + col)
                    groups.append(tokens)
        result = torch.tensor(groups, dtype=torch.long, device=device)
        if result.shape != (self.num_groups, self.tokens_per_group):
            raise RuntimeError(f"invalid group layout shape: {tuple(result.shape)}")
        return result

    def recent_group_mask(self, device: torch.device | str | None = None) -> torch.Tensor:
        group_slots = torch.arange(self.num_groups, device=device) // self.groups_per_slot
        return group_slots >= (self.num_slots - self.recent_slots)


def _validate_group_scores(
    scores: torch.Tensor,
    layout: SpatiotemporalGroupLayout,
    name: str,
) -> Tuple[torch.Tensor, bool]:
    scores, squeezed = _as_batched_scores(scores, name)
    if scores.shape[1] != layout.num_groups:
        raise ValueError(f"{name} has {scores.shape[1]} groups, expected {layout.num_groups}")
    return scores, squeezed


def _finalize_group_selection(keep: torch.Tensor, expected: int, squeezed: bool) -> torch.Tensor:
    keep = torch.sort(keep, dim=1).values
    if keep.shape[1] != expected:
        raise RuntimeError(f"selector returned {keep.shape[1]} groups, expected {expected}")
    if keep.shape[1] > 1 and not (keep[:, 1:] > keep[:, :-1]).all():
        raise RuntimeError("selector returned duplicate group indices")
    return keep.squeeze(0) if squeezed else keep


def select_task_protected_jepa(
    task_utility: torch.Tensor,
    jepa_residual: torch.Tensor,
    layout: SpatiotemporalGroupLayout,
    token_budget: int,
) -> torch.Tensor:
    """Recent core -> task protection -> high-residual fill, with exact K."""
    task_utility, task_squeezed = _validate_group_scores(task_utility, layout, "task_utility")
    jepa_residual, jepa_squeezed = _validate_group_scores(jepa_residual, layout, "jepa_residual")
    if task_squeezed != jepa_squeezed or task_utility.shape != jepa_residual.shape:
        raise ValueError("task_utility and jepa_residual must have matching shapes")
    if task_utility.device != jepa_residual.device:
        raise ValueError("task_utility and jepa_residual must be on the same device")

    batch_size = task_utility.shape[0]
    keep_groups = layout.groups_for_budget(token_budget)
    core = layout.recent_group_mask(task_utility.device).unsqueeze(0).expand(batch_size, -1)
    protected = core.clone()
    task_quota = (keep_groups - layout.num_recent_groups) // 2
    task_keep = _stable_topk_from_mask(task_utility, ~protected, task_quota)
    protected.scatter_(1, task_keep, True)

    residual_quota = keep_groups - int(protected[0].sum().item())
    residual_keep = _stable_topk_from_mask(jepa_residual, ~protected, residual_quota)
    protected.scatter_(1, residual_keep, True)
    keep = torch.nonzero(protected, as_tuple=False)[:, 1].reshape(batch_size, keep_groups)
    return _finalize_group_selection(keep, keep_groups, task_squeezed)


def _select_single_signal(
    scores: torch.Tensor,
    layout: SpatiotemporalGroupLayout,
    token_budget: int,
    name: str,
) -> torch.Tensor:
    scores, squeezed = _validate_group_scores(scores, layout, name)
    batch_size = scores.shape[0]
    keep_groups = layout.groups_for_budget(token_budget)
    core = layout.recent_group_mask(scores.device).unsqueeze(0).expand(batch_size, -1)
    quota = keep_groups - layout.num_recent_groups
    selected = _stable_topk_from_mask(scores, ~core, quota)
    keep = torch.cat((torch.nonzero(core, as_tuple=False)[:, 1].reshape(batch_size, -1), selected), dim=1)
    return _finalize_group_selection(keep, keep_groups, squeezed)


def select_task_only(
    task_utility: torch.Tensor,
    layout: SpatiotemporalGroupLayout,
    token_budget: int,
) -> torch.Tensor:
    return _select_single_signal(task_utility, layout, token_budget, "task_utility")


def select_jepa_only(
    jepa_residual: torch.Tensor,
    layout: SpatiotemporalGroupLayout,
    token_budget: int,
) -> torch.Tensor:
    return _select_single_signal(jepa_residual, layout, token_budget, "jepa_residual")


def select_attention_group(
    attention_score: torch.Tensor,
    layout: SpatiotemporalGroupLayout,
    token_budget: int,
) -> torch.Tensor:
    return _select_single_signal(attention_score, layout, token_budget, "attention_score")


def select_attention_groups_exact(
    attention_score: torch.Tensor,
    layout: SpatiotemporalGroupLayout,
    token_budget: int,
) -> torch.Tensor:
    """Select the unmodified group-attention exact-K set.

    Unlike the v1 comparison helper, this diagnostic selector does not inject a
    recent core.  The v2 swap oracle must start from the attention set itself so
    that the only intervention is a same-budget group exchange.
    """
    attention_score, squeezed = _validate_group_scores(
        attention_score, layout, "attention_score"
    )
    keep_groups = layout.groups_for_budget(token_budget)
    eligible = torch.ones_like(attention_score, dtype=torch.bool)
    keep = _stable_topk_from_mask(attention_score, eligible, keep_groups)
    return _finalize_group_selection(keep, keep_groups, squeezed)


def build_attention_jepa_swap_candidates(
    attention_keep: torch.Tensor,
    attention_score: torch.Tensor,
    jepa_residual: torch.Tensor,
    layout: SpatiotemporalGroupLayout,
    *,
    candidate_count: int = 8,
) -> Dict[str, torch.Tensor]:
    """Build matched-count one-for-one swap candidates around an attention set.

    Each returned tensor has shape ``[M,2]`` and stores ``(remove, add)`` group
    IDs.  ``jepa_disagreement`` rescues attention-dropped/high-residual groups
    while evicting attention-kept/low-residual groups.  ``anti_jepa`` reverses
    that direction.  ``attention_boundary`` swaps groups immediately across the
    attention cutoff and controls for a same-size local oracle search.
    """
    if attention_keep.ndim != 1 or attention_keep.dtype != torch.long:
        raise ValueError("attention_keep must be a one-dimensional LongTensor")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    attention_score, attention_squeezed = _validate_group_scores(
        attention_score, layout, "attention_score"
    )
    jepa_residual, residual_squeezed = _validate_group_scores(
        jepa_residual, layout, "jepa_residual"
    )
    if not attention_squeezed or not residual_squeezed:
        raise ValueError("swap candidate construction accepts one row at a time")
    attention_score = attention_score.squeeze(0)
    jepa_residual = jepa_residual.squeeze(0)
    if attention_keep.numel() == 0:
        raise ValueError("attention_keep must be non-empty")
    if int(attention_keep.min().item()) < 0 or int(attention_keep.max().item()) >= layout.num_groups:
        raise ValueError("attention_keep contains an out-of-range group")
    if torch.unique(attention_keep).numel() != attention_keep.numel():
        raise ValueError("attention_keep contains duplicate groups")

    kept = torch.zeros(layout.num_groups, dtype=torch.bool, device=attention_keep.device)
    kept[attention_keep] = True
    dropped = ~kept
    if candidate_count > min(int(kept.sum().item()), int(dropped.sum().item())):
        raise ValueError("candidate_count exceeds the smaller side of the attention partition")

    def ranked(scores: torch.Tensor, mask: torch.Tensor, descending: bool) -> torch.Tensor:
        ranked_scores = scores if descending else -scores
        return _stable_topk_from_mask(
            ranked_scores.unsqueeze(0), mask.unsqueeze(0), candidate_count
        ).squeeze(0)

    add_high_residual = ranked(jepa_residual, dropped, True)
    remove_low_residual = ranked(jepa_residual, kept, False)
    add_low_residual = ranked(jepa_residual, dropped, False)
    remove_high_residual = ranked(jepa_residual, kept, True)
    add_attention_boundary = ranked(attention_score, dropped, True)
    remove_attention_boundary = ranked(attention_score, kept, False)

    return {
        "jepa_disagreement": torch.stack(
            (remove_low_residual, add_high_residual), dim=1
        ),
        "anti_jepa": torch.stack((remove_high_residual, add_low_residual), dim=1),
        "attention_boundary": torch.stack(
            (remove_attention_boundary, add_attention_boundary), dim=1
        ),
    }


def apply_group_swaps(
    attention_keep: torch.Tensor,
    swap_pairs: torch.Tensor,
    layout: SpatiotemporalGroupLayout,
    token_budget: int,
) -> torch.Tensor:
    """Apply one remove/add pair per row while preserving the exact group budget."""
    if attention_keep.ndim != 1 or attention_keep.dtype != torch.long:
        raise ValueError("attention_keep must be a one-dimensional LongTensor")
    if swap_pairs.ndim != 2 or swap_pairs.shape[1] != 2 or swap_pairs.dtype != torch.long:
        raise ValueError("swap_pairs must be a LongTensor with shape [M,2]")
    expected = layout.groups_for_budget(token_budget)
    if attention_keep.numel() != expected:
        raise ValueError(
            f"attention_keep has {attention_keep.numel()} groups, expected {expected}"
        )
    if swap_pairs.shape[0] == 0:
        raise ValueError("swap_pairs must be non-empty")
    if int(swap_pairs.min().item()) < 0 or int(swap_pairs.max().item()) >= layout.num_groups:
        raise ValueError("swap_pairs contains an out-of-range group")
    kept = torch.zeros(layout.num_groups, dtype=torch.bool, device=attention_keep.device)
    kept[attention_keep] = True
    remove_ids, add_ids = swap_pairs[:, 0], swap_pairs[:, 1]
    if not bool(kept[remove_ids].all()):
        raise ValueError("every removed group must belong to the attention set")
    if bool(kept[add_ids].any()):
        raise ValueError("every added group must be outside the attention set")
    variants = attention_keep.unsqueeze(0).expand(swap_pairs.shape[0], -1).clone()
    for row, (remove_id, add_id) in enumerate(swap_pairs.tolist()):
        location = torch.nonzero(variants[row] == remove_id, as_tuple=False).flatten()
        if location.numel() != 1:
            raise RuntimeError("removed group does not occur exactly once")
        variants[row, location.item()] = add_id
    variants = torch.sort(variants, dim=1).values
    if not bool((variants[:, 1:] > variants[:, :-1]).all()):
        raise RuntimeError("swap produced duplicate group IDs")
    return variants


def aggregate_token_scores_to_groups(
    token_scores: torch.Tensor,
    group_token_indices: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    token_scores, squeezed = _as_batched_scores(token_scores, "token_scores")
    if group_token_indices.ndim != 2 or group_token_indices.dtype != torch.long:
        raise ValueError("group_token_indices must be a LongTensor with shape [G,S]")
    if group_token_indices.numel() == 0:
        raise ValueError("group_token_indices must be non-empty")
    if int(group_token_indices.min().item()) < 0 or int(group_token_indices.max().item()) >= token_scores.shape[1]:
        raise ValueError("group_token_indices contains an out-of-range token")
    group_token_indices = group_token_indices.to(token_scores.device)
    gathered = token_scores[:, group_token_indices]
    if reduction == "mean":
        result = gathered.mean(dim=-1)
    elif reduction == "sum":
        result = gathered.sum(dim=-1)
    else:
        raise ValueError(f"unsupported reduction: {reduction}")
    return result.squeeze(0) if squeezed else result


def expand_group_indices(
    group_indices: torch.Tensor,
    group_token_indices: torch.Tensor,
) -> torch.Tensor:
    squeezed = group_indices.ndim == 1
    if squeezed:
        group_indices = group_indices.unsqueeze(0)
    if group_indices.ndim != 2 or group_indices.dtype != torch.long:
        raise ValueError("group_indices must be a LongTensor with shape [Kg] or [B,Kg]")
    if group_token_indices.ndim != 2 or group_token_indices.dtype != torch.long:
        raise ValueError("group_token_indices must be a LongTensor with shape [G,S]")
    if group_indices.numel() == 0:
        raise ValueError("group_indices must be non-empty")
    if int(group_indices.min().item()) < 0 or int(group_indices.max().item()) >= group_token_indices.shape[0]:
        raise ValueError("group_indices contains an out-of-range group")
    expanded = group_token_indices.to(group_indices.device)[group_indices].flatten(1)
    expanded = torch.sort(expanded, dim=1).values
    if expanded.shape[1] > 1 and not (expanded[:, 1:] > expanded[:, :-1]).all():
        raise ValueError("groups overlap or a group was selected more than once")
    return expanded.squeeze(0) if squeezed else expanded


def delete_group_batch(
    features: torch.Tensor,
    token_positions: torch.Tensor,
    deleted_group_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one deletion variant per group from a single unpruned sample."""
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError("features must have shape [1,N,D]")
    if token_positions.ndim != 2 or token_positions.shape != features.shape[:2]:
        raise ValueError("token_positions must have shape [1,N]")
    if deleted_group_tokens.ndim != 2 or deleted_group_tokens.dtype != torch.long:
        raise ValueError("deleted_group_tokens must be a LongTensor with shape [M,S]")
    num_tokens = features.shape[1]
    if deleted_group_tokens.numel() == 0:
        raise ValueError("deleted_group_tokens must be non-empty")
    if int(deleted_group_tokens.min().item()) < 0 or int(deleted_group_tokens.max().item()) >= num_tokens:
        raise ValueError("deleted_group_tokens contains an out-of-range token")
    if deleted_group_tokens.shape[1] > 1:
        sorted_deleted = torch.sort(deleted_group_tokens, dim=1).values
        if not (sorted_deleted[:, 1:] > sorted_deleted[:, :-1]).all():
            raise ValueError("each deleted group must contain unique token indices")

    num_variants, group_size = deleted_group_tokens.shape
    keep_mask = torch.ones((num_variants, num_tokens), dtype=torch.bool, device=features.device)
    keep_mask.scatter_(1, deleted_group_tokens.to(features.device), False)
    keep_indices = torch.arange(num_tokens, device=features.device).expand(num_variants, -1)[keep_mask]
    keep_indices = keep_indices.reshape(num_variants, num_tokens - group_size)
    feature_batch = features.expand(num_variants, -1, -1)
    position_batch = token_positions.expand(num_variants, -1)
    kept_features = feature_batch.gather(1, keep_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
    kept_positions = position_batch.gather(1, keep_indices)
    target_positions = position_batch.gather(1, deleted_group_tokens.to(features.device))
    target_features = feature_batch.gather(
        1, deleted_group_tokens.to(features.device).unsqueeze(-1).expand(-1, -1, features.shape[-1])
    )
    return kept_features, kept_positions, target_features, target_positions


class EncoderL16Continuation:
    """Run a ViT encoder to an inclusive pruning block, then continue later."""

    def __init__(self, encoder: torch.nn.Module, prune_layer: int = 16) -> None:
        self.encoder = encoder
        self.prune_layer = int(prune_layer)
        if not hasattr(encoder, "blocks"):
            raise ValueError("encoder must expose a blocks sequence")
        if self.prune_layer < 0 or self.prune_layer >= len(encoder.blocks):
            raise ValueError(
                f"prune_layer must be in [0,{len(encoder.blocks) - 1}], got {self.prune_layer}"
            )

    def encode_to_prune(
        self,
        clips: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        encoder = self.encoder
        if getattr(encoder, "out_layers", None) is not None:
            raise ValueError("out_layers encoders are unsupported for continuation")
        t, h_patches, w_patches = _resolve_video_geometry(encoder, clips)
        if not getattr(encoder, "use_rope", False):
            pos_embed = encoder.interpolate_pos_encoding(clips, encoder.pos_embed)
            x = encoder.patch_embed(clips)
            x = x + pos_embed
        else:
            x = encoder.patch_embed(clips)
        batch_size, num_tokens, _ = x.shape
        if t is not None and num_tokens != int(t * h_patches * w_patches):
            raise ValueError(
                f"patch embedding returned {num_tokens} tokens, geometry implies "
                f"{int(t * h_patches * w_patches)}"
            )
        token_positions = torch.arange(num_tokens, device=x.device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        for block_idx in range(self.prune_layer + 1):
            block = encoder.blocks[block_idx]
            if getattr(encoder, "use_rope", False):
                x = block(
                    x,
                    mask=token_positions,
                    attn_mask=None,
                    T=t,
                    H_patches=h_patches,
                    W_patches=w_patches,
                )
            else:
                x = block(x, mask=None, attn_mask=None)
        return x, token_positions, (t, h_patches, w_patches)

    def continue_from(
        self,
        features: torch.Tensor,
        token_positions: torch.Tensor,
        geometry: Tuple[int, int, int],
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,N,D]")
        if token_positions.shape != features.shape[:2] or token_positions.dtype != torch.long:
            raise ValueError("token_positions must be a LongTensor with shape [B,N]")
        t, h_patches, w_patches = geometry
        x = features
        for block_idx in range(self.prune_layer + 1, len(self.encoder.blocks)):
            block = self.encoder.blocks[block_idx]
            if getattr(self.encoder, "use_rope", False):
                x = block(
                    x,
                    mask=token_positions,
                    attn_mask=None,
                    T=t,
                    H_patches=h_patches,
                    W_patches=w_patches,
                )
            else:
                x = block(x, mask=None, attn_mask=None)
        return self.encoder.norm(x)

    def forward_full(self, clips: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features, positions, geometry = self.encode_to_prune(clips)
        return self.continue_from(features, positions, geometry), positions


def masked_jepa_residual(
    predictor: torch.nn.Module,
    context_latents: torch.Tensor,
    context_positions: torch.Tensor,
    target_positions: torch.Tensor,
    target_latents: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Predict held-out group latents without exposing their values to predictor."""
    if context_latents.ndim != 3 or target_latents.ndim != 3:
        raise ValueError("context_latents and target_latents must have shape [B,N,D]")
    if context_positions.shape != context_latents.shape[:2] or context_positions.dtype != torch.long:
        raise ValueError("context_positions must be a LongTensor matching context_latents")
    if target_positions.shape != target_latents.shape[:2] or target_positions.dtype != torch.long:
        raise ValueError("target_positions must be a LongTensor matching target_latents")
    if context_latents.shape[0] != target_latents.shape[0]:
        raise ValueError("context and target batch sizes must match")
    if context_latents.shape[-1] != target_latents.shape[-1]:
        raise ValueError("context and target latent dimensions must match")
    if not torch.isfinite(context_latents).all() or not torch.isfinite(target_latents).all():
        raise ValueError("context and target latents must be finite")
    for row in range(context_latents.shape[0]):
        context_set = set(context_positions[row].tolist())
        target_set = set(target_positions[row].tolist())
        if len(context_set) != context_positions.shape[1] or len(target_set) != target_positions.shape[1]:
            raise ValueError("context and target positions must each be unique")
        if context_set.intersection(target_set):
            raise ValueError("JEPA target leakage: context and target positions overlap")
    max_position = int(torch.maximum(context_positions.max(), target_positions.max()).item())
    predictor_capacity = getattr(predictor, "num_patches", None)
    if predictor_capacity is not None and max_position >= int(predictor_capacity):
        raise ValueError(
            f"predictor position capacity {predictor_capacity} is too small for position {max_position}"
        )

    predicted = predict_masked_latents(
        predictor,
        context_latents,
        context_positions,
        target_positions,
    )
    if predicted.shape != target_latents.shape:
        raise ValueError(
            f"predictor returned {tuple(predicted.shape)}, expected {tuple(target_latents.shape)}"
        )
    predicted_unit = F.normalize(predicted.float(), dim=-1)
    target_unit = F.normalize(target_latents.float(), dim=-1)
    cosine = 1.0 - (predicted_unit * target_unit).sum(dim=-1)
    predicted_norm = F.layer_norm(predicted.float(), (predicted.shape[-1],))
    target_norm = F.layer_norm(target_latents.float(), (target_latents.shape[-1],))
    normalized_mse = (predicted_norm - target_norm).square().mean(dim=-1)
    return {
        "cosine_residual": cosine.mean(dim=-1),
        "normalized_mse": normalized_mse.mean(dim=-1),
        "predicted_latents": predicted,
    }


def predict_masked_latents(
    predictor: torch.nn.Module,
    context_latents: torch.Tensor,
    context_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """Predict target-position latents from context without accepting targets.

    Keeping target values out of this interface makes the E0 no-leakage
    contract structural: the recovery call can receive target positions but no
    teacher latent tensor.
    """
    if context_latents.ndim != 3:
        raise ValueError("context_latents must have shape [B,N,D]")
    if context_positions.shape != context_latents.shape[:2]:
        raise ValueError("context_positions must align with context_latents")
    if context_positions.dtype != torch.long or target_positions.dtype != torch.long:
        raise ValueError("context and target positions must be LongTensors")
    if target_positions.ndim != 2 or target_positions.shape[0] != context_latents.shape[0]:
        raise ValueError("target_positions must have shape [B,M]")
    if context_latents.shape[1] == 0 or target_positions.shape[1] == 0:
        raise ValueError("context and target position sets must be non-empty")
    if not bool(torch.isfinite(context_latents).all()):
        raise ValueError("context_latents must be finite")
    for row in range(context_latents.shape[0]):
        context_set = set(context_positions[row].tolist())
        target_set = set(target_positions[row].tolist())
        if len(context_set) != context_positions.shape[1]:
            raise ValueError("context positions must be unique")
        if len(target_set) != target_positions.shape[1]:
            raise ValueError("target positions must be unique")
        if context_set.intersection(target_set):
            raise ValueError("JEPA target leakage: context and target positions overlap")
    max_position = int(torch.maximum(context_positions.max(), target_positions.max()).item())
    predictor_capacity = getattr(predictor, "num_patches", None)
    if predictor_capacity is not None and max_position >= int(predictor_capacity):
        raise ValueError(
            f"predictor position capacity {predictor_capacity} is too small for position {max_position}"
        )
    predicted = predictor(
        context_latents,
        masks_x=context_positions,
        masks_y=target_positions,
    )
    if isinstance(predicted, (tuple, list)):
        if len(predicted) != 1:
            raise ValueError("predictor returned multiple outputs for a single mask pair")
        predicted = predicted[0]
    if predicted.ndim != 3 or predicted.shape[:2] != target_positions.shape:
        raise ValueError(
            "predictor output must align with target positions, got "
            f"{tuple(predicted.shape)} for {tuple(target_positions.shape)}"
        )
    if not bool(torch.isfinite(predicted).all()):
        raise ValueError("predictor returned non-finite completion latents")
    return predicted


def predict_mask_only_latents(
    predictor: torch.nn.Module,
    target_positions: torch.Tensor,
    *,
    output_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run the same frozen predictor on target mask tokens without context."""
    if target_positions.ndim != 2 or target_positions.dtype != torch.long:
        raise ValueError("target_positions must be a LongTensor with shape [B,M]")
    if target_positions.shape[1] == 0 or output_dim <= 0:
        raise ValueError("mask-only completion requires targets and a positive output_dim")
    empty_context = torch.empty(
        target_positions.shape[0],
        0,
        output_dim,
        device=target_positions.device,
        dtype=dtype,
    )
    empty_positions = torch.empty(
        target_positions.shape[0], 0, device=target_positions.device, dtype=torch.long
    )
    predicted = predictor(
        empty_context,
        masks_x=empty_positions,
        masks_y=target_positions,
    )
    if isinstance(predicted, (tuple, list)):
        if len(predicted) != 1:
            raise ValueError("predictor returned multiple mask-only outputs")
        predicted = predicted[0]
    if predicted.ndim != 3 or predicted.shape[:2] != target_positions.shape:
        raise ValueError("mask-only predictor output does not align with target positions")
    if not bool(torch.isfinite(predicted).all()):
        raise ValueError("mask-only predictor returned non-finite latents")
    return predicted


def gather_latents_by_positions(
    latents: torch.Tensor,
    positions: torch.Tensor,
    requested_positions: torch.Tensor,
) -> torch.Tensor:
    """Gather unique requested positions from a position-labelled latent set."""
    if latents.ndim != 3 or positions.shape != latents.shape[:2]:
        raise ValueError("latents/positions must have aligned [B,N] leading dimensions")
    if requested_positions.ndim != 2 or requested_positions.shape[0] != latents.shape[0]:
        raise ValueError("requested_positions must have shape [B,M]")
    if positions.dtype != torch.long or requested_positions.dtype != torch.long:
        raise ValueError("positions must be LongTensors")
    matches = requested_positions.unsqueeze(-1).eq(positions.unsqueeze(1))
    if not bool(matches.any(dim=-1).all()) or not bool((matches.sum(dim=-1) == 1).all()):
        raise ValueError("each requested position must occur exactly once")
    indices = matches.to(torch.long).argmax(dim=-1)
    return latents.gather(1, indices.unsqueeze(-1).expand(-1, -1, latents.shape[-1]))


def merge_positioned_latents(
    retained_latents: torch.Tensor,
    retained_positions: torch.Tensor,
    completion_latents: torch.Tensor,
    completion_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge disjoint retained/completion latents in original-position order."""
    if retained_latents.ndim != 3 or completion_latents.ndim != 3:
        raise ValueError("retained and completion latents must have shape [B,N,D]")
    if retained_latents.shape[0] != completion_latents.shape[0]:
        raise ValueError("retained and completion batch sizes must match")
    if retained_latents.shape[-1] != completion_latents.shape[-1]:
        raise ValueError("retained and completion latent dimensions must match")
    if retained_positions.shape != retained_latents.shape[:2]:
        raise ValueError("retained_positions must align with retained_latents")
    if completion_positions.shape != completion_latents.shape[:2]:
        raise ValueError("completion_positions must align with completion_latents")
    if retained_positions.dtype != torch.long or completion_positions.dtype != torch.long:
        raise ValueError("positions must be LongTensors")
    merged_latents = torch.cat((retained_latents, completion_latents), dim=1)
    merged_positions = torch.cat((retained_positions, completion_positions), dim=1)
    order = torch.argsort(merged_positions, dim=1, stable=True)
    sorted_positions = merged_positions.gather(1, order)
    sorted_latents = merged_latents.gather(
        1, order.unsqueeze(-1).expand(-1, -1, merged_latents.shape[-1])
    )
    if sorted_positions.shape[1] > 1 and not bool(
        (sorted_positions[:, 1:] > sorted_positions[:, :-1]).all()
    ):
        raise ValueError("retained and completion positions must be unique and disjoint")
    if not bool(torch.isfinite(sorted_latents).all()):
        raise ValueError("merged completion contains non-finite latents")
    return sorted_latents, sorted_positions


def same_slot_mean_completion(
    retained_latents: torch.Tensor,
    retained_positions: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    tokens_per_slot: int,
) -> torch.Tensor:
    """Fill each target with the mean retained latent from its temporal slot.

    If attention retained no token in a target's slot, use the row-global mean.
    This is the E0 deterministic position-aware pooling placebo.
    """
    if retained_latents.ndim != 3 or retained_positions.shape != retained_latents.shape[:2]:
        raise ValueError("retained latents and positions must align")
    if target_positions.ndim != 2 or target_positions.shape[0] != retained_latents.shape[0]:
        raise ValueError("target_positions must have shape [B,M]")
    if tokens_per_slot <= 0:
        raise ValueError("tokens_per_slot must be positive")
    output = torch.empty(
        target_positions.shape[0],
        target_positions.shape[1],
        retained_latents.shape[-1],
        device=retained_latents.device,
        dtype=retained_latents.dtype,
    )
    retained_slots = retained_positions // tokens_per_slot
    target_slots = target_positions // tokens_per_slot
    for row in range(retained_latents.shape[0]):
        global_mean = retained_latents[row].mean(dim=0)
        for slot in torch.unique(target_slots[row]).tolist():
            target_mask = target_slots[row] == slot
            retained_mask = retained_slots[row] == slot
            pooled = (
                retained_latents[row, retained_mask].mean(dim=0)
                if bool(retained_mask.any())
                else global_mean
            )
            output[row, target_mask] = pooled
    if not bool(torch.isfinite(output).all()):
        raise ValueError("same-slot pooling produced non-finite latents")
    return output


@contextmanager
def predictor_lora_disabled(model: torch.nn.Module):
    """Temporarily expose the pretrained predictor path without task LoRA."""
    modules = [
        module
        for module in model.modules()
        if bool(getattr(module, "_is_predictor_lora", False))
    ]
    saved_scaling = []
    try:
        for module in modules:
            if not hasattr(module, "scaling"):
                raise ValueError("predictor-LoRA module has no scaling attribute")
            saved_scaling.append(module.scaling)
            module.scaling = 0.0
        yield len(modules)
    finally:
        for module, scaling in zip(modules, saved_scaling):
            module.scaling = scaling


def predict_future_tokens_rebased(
    core: torch.nn.Module,
    selected_full: torch.Tensor,
    anticipation_times: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen streaming parent predictor with its registered rebase convention."""
    if selected_full.ndim != 3:
        raise ValueError("selected_full must have shape [B,N,D]")
    batch_size, num_tokens, full_dim = selected_full.shape
    if anticipation_times.shape != (batch_size,):
        raise ValueError("anticipation_times must have shape [B]")
    grid_tokens = int(core.grid_size**2)
    embed_dim = int(core.encoder.embed_dim)
    selected = selected_full[:, :, -embed_dim:] if full_dim > embed_dim else selected_full
    accumulated = selected.clone()
    context_positions = torch.arange(num_tokens, device=selected_full.device, dtype=torch.long)
    context_positions = context_positions.unsqueeze(0).expand(batch_size, -1)
    anticipation_steps = (
        anticipation_times * core.frames_per_second / core.tubelet_size
    ).to(torch.int64)
    target_start = num_tokens + grid_tokens * anticipation_steps
    num_predicted = int(grid_tokens * (core.num_output_frames // core.tubelet_size))
    target_positions = torch.arange(num_predicted, device=selected_full.device, dtype=torch.long)
    target_positions = target_positions.unsqueeze(0).expand(batch_size, -1) + target_start.unsqueeze(1)
    capacity = getattr(core.predictor, "num_patches", None)
    if capacity is not None and int(target_positions.max().item()) >= int(capacity):
        raise ValueError(
            f"predictor position capacity {capacity} is too small for target position "
            f"{int(target_positions.max().item())}"
        )

    predictor_input = selected_full
    for _ in range(int(core.num_steps)):
        predicted = core.predictor(
            predictor_input,
            masks_x=context_positions,
            masks_y=target_positions,
        )
        predicted_full = predicted[0] if isinstance(predicted, tuple) else predicted
        predicted_base = (
            predicted_full[:, :, -embed_dim:]
            if predicted_full.shape[-1] != embed_dim
            else predicted_full
        )
        accumulated = torch.cat((accumulated, predicted_base), dim=1)
        next_input = (
            predicted_full
            if predicted_full.shape[-1] == predictor_input.shape[-1]
            else predicted_base
        )
        predictor_input = torch.cat((predictor_input[:, num_predicted:, :], next_input), dim=1)
    return accumulated


def weighted_mtp_loss_per_row(
    outputs: Mapping[object, Mapping[str, torch.Tensor]],
    raw_verbs: torch.Tensor,
    raw_nouns: torch.Tensor,
    target_mask: torch.Tensor,
    horizons: Sequence[object],
    horizon_weights: Sequence[float],
    verb_map: Mapping[int, int],
    noun_map: Mapping[int, int],
    action_map: Mapping[Tuple[int, int], int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Actual weighted multi-horizon verb+noun+action CE for each variant."""
    if raw_verbs.shape != raw_nouns.shape or raw_verbs.shape != target_mask.shape:
        raise ValueError("raw_verbs, raw_nouns, and target_mask must have identical [B,H] shapes")
    if raw_verbs.ndim != 2 or raw_verbs.shape[1] != len(horizons):
        raise ValueError("target tensors must have shape [B,len(horizons)]")
    if len(horizon_weights) != len(horizons):
        raise ValueError("horizon_weights must align with horizons")
    if (torch.as_tensor(horizon_weights) < 0).any():
        raise ValueError("horizon_weights must be non-negative")

    first_logits = outputs[horizons[0]]["action"]
    batch_size = raw_verbs.shape[0]
    if first_logits.shape[0] != batch_size:
        raise ValueError("output and target batch sizes do not match")
    total = torch.zeros(batch_size, device=first_logits.device, dtype=torch.float32)
    valid_horizons = torch.zeros(batch_size, device=first_logits.device, dtype=torch.long)

    for horizon_idx, (horizon, weight) in enumerate(zip(horizons, horizon_weights)):
        horizon_output = outputs[horizon]
        selected_rows: List[int] = []
        mapped_verbs: List[int] = []
        mapped_nouns: List[int] = []
        mapped_actions: List[int] = []
        for row in range(batch_size):
            if float(target_mask[row, horizon_idx].item()) < 0.5:
                continue
            raw_verb = int(raw_verbs[row, horizon_idx].item())
            raw_noun = int(raw_nouns[row, horizon_idx].item())
            if raw_verb not in verb_map or raw_noun not in noun_map or (raw_verb, raw_noun) not in action_map:
                continue
            selected_rows.append(row)
            mapped_verbs.append(int(verb_map[raw_verb]))
            mapped_nouns.append(int(noun_map[raw_noun]))
            mapped_actions.append(int(action_map[(raw_verb, raw_noun)]))
        if not selected_rows:
            continue
        row_index = torch.tensor(selected_rows, device=first_logits.device, dtype=torch.long)
        verb_target = torch.tensor(mapped_verbs, device=first_logits.device, dtype=torch.long)
        noun_target = torch.tensor(mapped_nouns, device=first_logits.device, dtype=torch.long)
        action_target = torch.tensor(mapped_actions, device=first_logits.device, dtype=torch.long)
        loss = F.cross_entropy(horizon_output["verb"][row_index].float(), verb_target, reduction="none")
        loss = loss + F.cross_entropy(
            horizon_output["noun"][row_index].float(), noun_target, reduction="none"
        )
        loss = loss + F.cross_entropy(
            horizon_output["action"][row_index].float(), action_target, reduction="none"
        )
        total.index_add_(0, row_index, float(weight) * loss)
        valid_horizons.index_add_(0, row_index, torch.ones_like(row_index))
    return total, valid_horizons


def weighted_action_margin_per_row(
    outputs: Mapping[object, Mapping[str, torch.Tensor]],
    raw_verbs: torch.Tensor,
    raw_nouns: torch.Tensor,
    target_mask: torch.Tensor,
    horizons: Sequence[object],
    horizon_weights: Sequence[float],
    action_map: Mapping[Tuple[int, int], int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted true-action logit minus strongest competing action logit."""
    if raw_verbs.shape != raw_nouns.shape or raw_verbs.shape != target_mask.shape:
        raise ValueError("raw_verbs, raw_nouns, and target_mask must have identical shapes")
    if raw_verbs.ndim != 2 or raw_verbs.shape[1] != len(horizons):
        raise ValueError("target tensors must have shape [B,len(horizons)]")
    if len(horizon_weights) != len(horizons):
        raise ValueError("horizon_weights must align with horizons")
    first_logits = outputs[horizons[0]]["action"]
    margins = torch.zeros(first_logits.shape[0], device=first_logits.device, dtype=torch.float32)
    valid_horizons = torch.zeros(first_logits.shape[0], device=first_logits.device, dtype=torch.long)
    for horizon_idx, (horizon, weight) in enumerate(zip(horizons, horizon_weights)):
        logits = outputs[horizon]["action"].float()
        rows: List[int] = []
        labels: List[int] = []
        for row in range(logits.shape[0]):
            if float(target_mask[row, horizon_idx].item()) < 0.5:
                continue
            pair = (int(raw_verbs[row, horizon_idx].item()), int(raw_nouns[row, horizon_idx].item()))
            if pair not in action_map:
                continue
            rows.append(row)
            labels.append(int(action_map[pair]))
        if not rows:
            continue
        row_index = torch.tensor(rows, device=logits.device, dtype=torch.long)
        labels_tensor = torch.tensor(labels, device=logits.device, dtype=torch.long)
        selected_logits = logits[row_index]
        true_logits = selected_logits.gather(1, labels_tensor.unsqueeze(1)).squeeze(1)
        competitors = selected_logits.clone()
        competitors.scatter_(1, labels_tensor.unsqueeze(1), -torch.inf)
        horizon_margin = true_logits - competitors.max(dim=1).values
        margins.index_add_(0, row_index, float(weight) * horizon_margin)
        valid_horizons.index_add_(0, row_index, torch.ones_like(row_index))
    return margins, valid_horizons


def weighted_action_loss_per_row(
    outputs: Mapping[object, Mapping[str, torch.Tensor]],
    raw_verbs: torch.Tensor,
    raw_nouns: torch.Tensor,
    target_mask: torch.Tensor,
    horizons: Sequence[object],
    horizon_weights: Sequence[float],
    action_map: Mapping[Tuple[int, int], int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted action-head-only cross entropy for target-stability audits."""
    if raw_verbs.shape != raw_nouns.shape or raw_verbs.shape != target_mask.shape:
        raise ValueError("raw_verbs, raw_nouns, and target_mask must have identical shapes")
    if raw_verbs.ndim != 2 or raw_verbs.shape[1] != len(horizons):
        raise ValueError("target tensors must have shape [B,len(horizons)]")
    if len(horizon_weights) != len(horizons):
        raise ValueError("horizon_weights must align with horizons")
    first_logits = outputs[horizons[0]]["action"]
    losses = torch.zeros(first_logits.shape[0], device=first_logits.device, dtype=torch.float32)
    valid_horizons = torch.zeros(first_logits.shape[0], device=first_logits.device, dtype=torch.long)
    for horizon_idx, (horizon, weight) in enumerate(zip(horizons, horizon_weights)):
        logits = outputs[horizon]["action"].float()
        rows: List[int] = []
        labels: List[int] = []
        for row in range(logits.shape[0]):
            if float(target_mask[row, horizon_idx].item()) < 0.5:
                continue
            pair = (int(raw_verbs[row, horizon_idx].item()), int(raw_nouns[row, horizon_idx].item()))
            if pair not in action_map:
                continue
            rows.append(row)
            labels.append(int(action_map[pair]))
        if not rows:
            continue
        row_index = torch.tensor(rows, device=logits.device, dtype=torch.long)
        labels_tensor = torch.tensor(labels, device=logits.device, dtype=torch.long)
        horizon_loss = F.cross_entropy(logits[row_index], labels_tensor, reduction="none")
        losses.index_add_(0, row_index, float(weight) * horizon_loss)
        valid_horizons.index_add_(0, row_index, torch.ones_like(row_index))
    return losses, valid_horizons


class OfflineOracleTargetAdapter:
    """Generate coarse leave-one-group-out task and JEPA targets for one row."""

    def __init__(
        self,
        core: torch.nn.Module,
        classifier: torch.nn.Module,
        verb_map: Mapping[int, int],
        noun_map: Mapping[int, int],
        action_map: Mapping[Tuple[int, int], int],
        *,
        horizons: Sequence[float] = (2.0, 4.0, 6.0),
        horizon_weights: Sequence[float] = (1.0, 0.7, 0.5),
        anticipation_sec: float = 2.0,
        prune_layer: int = 16,
    ) -> None:
        if len(horizons) != len(horizon_weights):
            raise ValueError("horizons and horizon_weights must align")
        self.core = core
        self.classifier = classifier
        self.verb_map = verb_map
        self.noun_map = noun_map
        self.action_map = action_map
        self.horizons = tuple(float(value) for value in horizons)
        self.horizon_weights = tuple(float(value) for value in horizon_weights)
        self.anticipation_sec = float(anticipation_sec)
        self.continuation = EncoderL16Continuation(core.encoder, prune_layer=prune_layer)

    def _score_final_tokens(
        self,
        final_tokens: torch.Tensor,
        raw_verbs: torch.Tensor,
        raw_nouns: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = final_tokens.shape[0]
        anticipation = torch.full(
            (batch_size,),
            self.anticipation_sec,
            device=final_tokens.device,
            dtype=torch.float32,
        )
        classifier_tokens = predict_future_tokens_rebased(self.core, final_tokens, anticipation)
        outputs = self.classifier(classifier_tokens)
        losses, valid_loss = weighted_mtp_loss_per_row(
            outputs,
            raw_verbs,
            raw_nouns,
            target_mask,
            self.horizons,
            self.horizon_weights,
            self.verb_map,
            self.noun_map,
            self.action_map,
        )
        margins, valid_margin = weighted_action_margin_per_row(
            outputs,
            raw_verbs,
            raw_nouns,
            target_mask,
            self.horizons,
            self.horizon_weights,
            self.action_map,
        )
        action_losses, valid_action_loss = weighted_action_loss_per_row(
            outputs,
            raw_verbs,
            raw_nouns,
            target_mask,
            self.horizons,
            self.horizon_weights,
            self.action_map,
        )
        if (
            not torch.equal(valid_loss, valid_margin)
            or not torch.equal(valid_loss, valid_action_loss)
            or not bool((valid_loss == len(self.horizons)).all())
        ):
            raise ValueError("oracle row does not have all mapped horizons")
        return losses, action_losses, margins, valid_loss

    @torch.no_grad()
    def generate_row_targets(
        self,
        clips: torch.Tensor,
        raw_verbs: torch.Tensor,
        raw_nouns: torch.Tensor,
        target_mask: torch.Tensor,
        deletion_groups: torch.Tensor,
        *,
        deletion_batch_size: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """Generate targets; caller must pass only non-core groups for one row."""
        if clips.shape[0] != 1:
            raise ValueError("offline oracle target export accepts exactly one source row")
        if raw_verbs.shape != (1, len(self.horizons)):
            raise ValueError("raw_verbs must have shape [1,H]")
        if raw_nouns.shape != raw_verbs.shape or target_mask.shape != raw_verbs.shape:
            raise ValueError("raw_nouns and target_mask must match raw_verbs")
        if deletion_batch_size <= 0:
            raise ValueError("deletion_batch_size must be positive")
        if self.core.training or self.classifier.training:
            raise ValueError("core and classifier must be in eval mode for frozen target export")

        l16_tokens, full_positions, geometry = self.continuation.encode_to_prune(clips)
        full_final = self.continuation.continue_from(l16_tokens, full_positions, geometry)
        baseline_loss, baseline_action_loss, baseline_margin, _ = self._score_final_tokens(
            full_final, raw_verbs, raw_nouns, target_mask
        )

        deletion_losses: List[torch.Tensor] = []
        deletion_action_losses: List[torch.Tensor] = []
        deletion_margins: List[torch.Tensor] = []
        cosine_residuals: List[torch.Tensor] = []
        normalized_mses: List[torch.Tensor] = []
        lora_module_counts: List[int] = []
        for start in range(0, deletion_groups.shape[0], deletion_batch_size):
            group_chunk = deletion_groups[start : start + deletion_batch_size]
            context_l16, context_positions, _, target_positions = delete_group_batch(
                l16_tokens, full_positions, group_chunk
            )
            context_final = self.continuation.continue_from(context_l16, context_positions, geometry)
            group_count = context_final.shape[0]
            repeated_verbs = raw_verbs.expand(group_count, -1)
            repeated_nouns = raw_nouns.expand(group_count, -1)
            repeated_mask = target_mask.expand(group_count, -1)
            losses, action_losses, margins, _ = self._score_final_tokens(
                context_final, repeated_verbs, repeated_nouns, repeated_mask
            )
            deletion_losses.append(losses)
            deletion_action_losses.append(action_losses)
            deletion_margins.append(margins)

            target_final = full_final.expand(group_count, -1, -1).gather(
                1,
                group_chunk.to(full_final.device).unsqueeze(-1).expand(-1, -1, full_final.shape[-1]),
            ).detach()
            with predictor_lora_disabled(self.core) as disabled_count:
                residual = masked_jepa_residual(
                    self.core.predictor,
                    context_final,
                    context_positions,
                    target_positions,
                    target_final,
                )
            lora_module_counts.append(disabled_count)
            cosine_residuals.append(residual["cosine_residual"])
            normalized_mses.append(residual["normalized_mse"])

        deletion_loss = torch.cat(deletion_losses)
        deletion_action_loss = torch.cat(deletion_action_losses)
        deletion_margin = torch.cat(deletion_margins)
        if len(set(lora_module_counts)) != 1:
            raise RuntimeError("predictor-LoRA module count changed during target export")
        return {
            "baseline_task_loss": baseline_loss,
            "baseline_action_loss": baseline_action_loss,
            "baseline_action_margin": baseline_margin,
            "deletion_task_loss": deletion_loss,
            "task_utility": counterfactual_marginal_utility(baseline_loss, deletion_loss),
            "deletion_action_loss": deletion_action_loss,
            "action_loss_utility": counterfactual_marginal_utility(
                baseline_action_loss, deletion_action_loss
            ),
            "deletion_action_margin": deletion_margin,
            "action_margin_drop": baseline_margin - deletion_margin,
            "jepa_cosine_residual": torch.cat(cosine_residuals),
            "jepa_normalized_mse": torch.cat(normalized_mses),
            "predictor_lora_modules_disabled": torch.tensor(
                lora_module_counts[0], device=deletion_loss.device, dtype=torch.long
            ),
        }


class OfflineJEPARedundancyAdapter:
    """Generate JEPA residuals only, without singleton task-loss targets."""

    def __init__(
        self,
        core: torch.nn.Module,
        *,
        prune_layer: int = 16,
    ) -> None:
        self.core = core
        self.continuation = EncoderL16Continuation(core.encoder, prune_layer=prune_layer)

    @torch.no_grad()
    def residuals_from_l16(
        self,
        l16_tokens: torch.Tensor,
        full_positions: torch.Tensor,
        full_final: torch.Tensor,
        geometry: Tuple[int, int, int],
        groups: torch.Tensor,
        *,
        group_batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        if l16_tokens.shape[0] != 1 or full_final.shape[0] != 1:
            raise ValueError("JEPA swap diagnostic accepts exactly one source row")
        if full_positions.shape != l16_tokens.shape[:2]:
            raise ValueError("full_positions must align with L16 tokens")
        if groups.ndim != 2 or groups.dtype != torch.long or groups.shape[0] == 0:
            raise ValueError("groups must be a non-empty LongTensor with shape [G,S]")
        if group_batch_size <= 0:
            raise ValueError("group_batch_size must be positive")
        if self.core.training:
            raise ValueError("core must be in eval mode for frozen JEPA residual export")

        cosine_residuals: List[torch.Tensor] = []
        normalized_mses: List[torch.Tensor] = []
        lora_module_counts: List[int] = []
        for start in range(0, groups.shape[0], group_batch_size):
            group_chunk = groups[start : start + group_batch_size]
            context_l16, context_positions, _, target_positions = delete_group_batch(
                l16_tokens, full_positions, group_chunk
            )
            context_final = self.continuation.continue_from(
                context_l16, context_positions, geometry
            )
            group_count = context_final.shape[0]
            target_final = full_final.expand(group_count, -1, -1).gather(
                1,
                group_chunk.to(full_final.device)
                .unsqueeze(-1)
                .expand(-1, -1, full_final.shape[-1]),
            ).detach()
            with predictor_lora_disabled(self.core) as disabled_count:
                residual = masked_jepa_residual(
                    self.core.predictor,
                    context_final,
                    context_positions,
                    target_positions,
                    target_final,
                )
            lora_module_counts.append(disabled_count)
            cosine_residuals.append(residual["cosine_residual"])
            normalized_mses.append(residual["normalized_mse"])
        if len(set(lora_module_counts)) != 1:
            raise RuntimeError("predictor-LoRA module count changed during JEPA residual export")
        return {
            "jepa_cosine_residual": torch.cat(cosine_residuals),
            "jepa_normalized_mse": torch.cat(normalized_mses),
            "predictor_lora_modules_disabled": torch.tensor(
                lora_module_counts[0], device=l16_tokens.device, dtype=torch.long
            ),
        }


def counterfactual_marginal_utility(
    baseline_loss: torch.Tensor,
    deletion_loss: torch.Tensor,
) -> torch.Tensor:
    if baseline_loss.ndim != 1 or deletion_loss.ndim != 1:
        raise ValueError("baseline_loss and deletion_loss must be one-dimensional")
    if baseline_loss.numel() not in (1, deletion_loss.numel()):
        raise ValueError("baseline_loss must be scalar-per-sample or match deletion_loss")
    if not torch.isfinite(baseline_loss).all() or not torch.isfinite(deletion_loss).all():
        raise ValueError("losses must be finite")
    return deletion_loss - baseline_loss


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def oracle_row_key(row: Mapping[str, str]) -> str:
    return f"{row['video_id']}\t{int(row['tick_frame'])}"


def _parse_csv_list(raw: str, name: str) -> List[float]:
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        values = json.loads(text)
        if not isinstance(values, list):
            raise ValueError(f"{name} must encode a list")
        return [float(value) for value in values]
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def oracle_row_is_eligible(row: Mapping[str, str]) -> bool:
    if abs(float(row["context_sec"]) - 10.0) > 1e-9:
        return False
    target_mask = _parse_csv_list(row["mtp_mask"], "mtp_mask")
    verbs = _parse_csv_list(row["mtp_verbs"], "mtp_verbs")
    nouns = _parse_csv_list(row["mtp_nouns"], "mtp_nouns")
    if len(target_mask) != 3 or len(verbs) != 3 or len(nouns) != 3:
        return False
    return all(mask >= 0.5 for mask in target_mask) and all(value >= 0 for value in verbs + nouns)


def _stable_hash_order(values: Iterable[str]) -> List[str]:
    return sorted(values, key=lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value))


def _sha256_lines(values: Iterable[str]) -> str:
    """Hash UTF-8 lines with one trailing newline per value."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _round_robin_rows(
    rows_by_video: Mapping[str, Sequence[Mapping[str, str]]],
    video_ids: Sequence[str],
    target_rows: int,
) -> List[Mapping[str, str]]:
    selected: List[Mapping[str, str]] = []
    depth = 0
    while len(selected) < target_rows:
        made_progress = False
        for video_id in video_ids:
            video_rows = rows_by_video[video_id]
            if depth < len(video_rows):
                selected.append(video_rows[depth])
                made_progress = True
                if len(selected) == target_rows:
                    break
        if not made_progress:
            raise ValueError(f"only found {len(selected)} eligible rows, need {target_rows}")
        depth += 1
    return selected


def partition_oracle_rows(
    rows: Sequence[Mapping[str, str]],
    expected_video_count: int = 86,
    calibration_video_count: int = 64,
    calibration_rows: int = 512,
    heldout_rows: int = 2048,
) -> Dict[str, object]:
    eligible = [row for row in rows if oracle_row_is_eligible(row)]
    rows_by_video: MutableMapping[str, List[Mapping[str, str]]] = {}
    seen_keys = set()
    for row in eligible:
        key = oracle_row_key(row)
        if key in seen_keys:
            raise ValueError(f"duplicate oracle row key: {key}")
        seen_keys.add(key)
        rows_by_video.setdefault(row["video_id"], []).append(row)
    video_ids = _stable_hash_order(rows_by_video.keys())
    if len(video_ids) != expected_video_count:
        raise ValueError(f"expected {expected_video_count} eligible videos, found {len(video_ids)}")
    if not 0 < calibration_video_count < expected_video_count:
        raise ValueError("calibration_video_count must split the video set")
    calibration_videos = video_ids[:calibration_video_count]
    heldout_videos = video_ids[calibration_video_count:]
    for video_id, video_rows in rows_by_video.items():
        video_rows.sort(
            key=lambda row: (
                hashlib.sha256(oracle_row_key(row).encode("utf-8")).hexdigest(),
                int(row["tick_frame"]),
            )
        )
    calibration = _round_robin_rows(rows_by_video, calibration_videos, calibration_rows)
    heldout = _round_robin_rows(rows_by_video, heldout_videos, heldout_rows)
    return {
        "calibration_videos": calibration_videos,
        "heldout_videos": heldout_videos,
        "calibration_rows": calibration,
        "heldout_rows": heldout,
    }


def write_oracle_manifests(
    source_csv: str | Path,
    output_dir: str | Path,
    expected_source_sha256: str = EXPECTED_TRAIN_CSV_SHA256,
) -> Dict[str, object]:
    source_csv = Path(source_csv)
    output_dir = Path(output_dir)
    actual_source_sha256 = sha256_file(source_csv)
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {expected_source_sha256}, got {actual_source_sha256}"
        )
    output_paths = {
        "calibration": output_dir / "oracle_calibration_512.csv",
        "heldout": output_dir / "oracle_heldout_2048.csv",
        "manifest": output_dir / "oracle_manifest.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing oracle manifests: {existing}")

    with source_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("source CSV has no header")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    partition = partition_oracle_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("calibration", "heldout"):
        path = output_paths[split_name]
        split_rows = partition[f"{split_name}_rows"]
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

    manifest = {
        "protocol_id": ORACLE_PROTOCOL_ID,
        "source_csv": str(source_csv),
        "source_csv_sha256": actual_source_sha256,
        "eligibility": {
            "context_sec": 10.0,
            "all_three_mtp_targets_valid": True,
        },
        "video_partition": {
            "order": "ascending sha256(utf8(video_id)); lexical video_id tie-break",
            "membership_hash_format": "sha256(concat(utf8(video_id) + newline)) in stored order",
            "calibration_count": len(partition["calibration_videos"]),
            "heldout_count": len(partition["heldout_videos"]),
            "calibration_video_ids": partition["calibration_videos"],
            "heldout_video_ids": partition["heldout_videos"],
            "calibration_video_ids_sha256": _sha256_lines(partition["calibration_videos"]),
            "heldout_video_ids_sha256": _sha256_lines(partition["heldout_videos"]),
        },
        "row_selection": {
            "within_video_order": "ascending sha256(utf8(video_id + tab + int(tick_frame))); tick tie-break",
            "sampling": "round-robin across the fixed split video order",
            "row_key_hash_format": "sha256(concat(utf8(video_id + tab + int(tick_frame)) + newline))",
            "calibration_rows": len(partition["calibration_rows"]),
            "heldout_rows": len(partition["heldout_rows"]),
            "calibration_row_keys_sha256": _sha256_lines(
                oracle_row_key(row) for row in partition["calibration_rows"]
            ),
            "heldout_row_keys_sha256": _sha256_lines(
                oracle_row_key(row) for row in partition["heldout_rows"]
            ),
        },
        "artifacts": {
            "calibration_csv": output_paths["calibration"].name,
            "calibration_sha256": sha256_file(output_paths["calibration"]),
            "heldout_csv": output_paths["heldout"].name,
            "heldout_sha256": sha256_file(output_paths["heldout"]),
        },
    }
    with output_paths["manifest"].open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
