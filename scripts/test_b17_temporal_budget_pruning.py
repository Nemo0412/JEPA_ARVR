#!/usr/bin/env python3
"""Semantic contracts for B17 temporal allocation under uniform spatial pruning."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from app.hdepic_lora_action_anticipation.eval_stream_mtp_fixed_budget_prune import (
    PROTOCOL_HORIZONS,
    TEMPORAL_PROTOCOL_ID,
    TEMPORAL_PROTOCOL_VARIANTS,
    validate_protocol_args,
)
from app.hdepic_lora_action_anticipation.gaze_spatial_pruning import (
    QuotaSpatialTokenSelector,
)
from app.hdepic_lora_action_anticipation.temporal_budget_pruning import (
    LOSS_AWARE_PRIMARY_BUDGET,
    LossAwareTemporalQuotaProvider,
    attention_sum_temporal_quotas,
    permuted_temporal_quotas,
    recent_temporal_quotas,
    uniform_temporal_quotas,
)


def test_all_allocators_exact_k_across_contexts() -> None:
    provider = LossAwareTemporalQuotaProvider()
    for n_slots in (16, 24, 32, 40):
        lossaware = provider.quotas(n_slots)
        uniform = uniform_temporal_quotas(n_slots, total=3840, capacity=256)
        recent = recent_temporal_quotas(n_slots, total=3840, capacity=256)
        for quotas in (lossaware, uniform, recent):
            assert quotas.shape == (n_slots,)
            assert int(quotas.sum()) == LOSS_AWARE_PRIMARY_BUDGET == 3840
            assert bool((quotas >= 0).all()) and bool((quotas <= 256).all())

    assert uniform_temporal_quotas(40, total=3840, capacity=256).tolist() == [96] * 40
    assert uniform_temporal_quotas(16, total=3840, capacity=256).tolist() == [240] * 16
    assert recent_temporal_quotas(40, total=3840, capacity=256).tolist() == (
        [0] * 25 + [256] * 15
    )
    assert recent_temporal_quotas(16, total=3840, capacity=256).tolist() == (
        [0] + [256] * 15
    )


def test_row_keyed_random_quota_permutations() -> None:
    base = LossAwareTemporalQuotaProvider().quotas(40)
    keys = ["video-a|10", "video-b|20"]
    seed17 = permuted_temporal_quotas(base, keys, seed=17)
    repeated = permuted_temporal_quotas(base, keys, seed=17)
    swapped = permuted_temporal_quotas(base, keys[::-1], seed=17)
    seed29 = permuted_temporal_quotas(base, keys, seed=29)
    assert torch.equal(seed17, repeated)
    assert torch.equal(seed17[0], swapped[1]) and torch.equal(seed17[1], swapped[0])
    assert not torch.equal(seed17, seed29)
    assert all(sorted(row.tolist()) == sorted(base.tolist()) for row in seed17)
    assert bool((seed17.sum(dim=1) == 3840).all())


def test_attention_mass_capacity_projection() -> None:
    attention = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    quota = attention_sum_temporal_quotas(
        attention, spatial_tokens=4, total=6, capacity=4
    )
    assert quota.tolist() == [[4, 2], [2, 4]]


def test_per_row_quotas_feed_one_uniform_spatial_selector() -> None:
    selector = QuotaSpatialTokenSelector(grid_size=4, tubelet_size=2)
    attention = torch.arange(64, dtype=torch.float32).view(2, 32)
    quotas = torch.tensor([[1, 3], [2, 2]])
    indices, stats = selector.select(
        mode="uniform", quotas=quotas, attention_scores=attention
    )
    assert torch.bincount(indices[0] // 16, minlength=2).tolist() == [1, 3]
    assert torch.bincount(indices[1] // 16, minlength=2).tolist() == [2, 2]
    assert stats["gaze_selected_slots"] == 0
    assert stats["attention_fallback_slots"] == 0


def test_temporal_protocol_is_fail_closed() -> None:
    args = SimpleNamespace(
        budget=3840,
        position_mode="true_full",
        max_frames=80,
        fps=8,
        src_fps=8,
        img_size=256,
        anticipation_sec=2.0,
        primary_horizon_sec=2.0,
        max_val_batches=0,
        only_context_sec=0.0,
        protocol_id=TEMPORAL_PROTOCOL_ID,
        random_seed=17,
        evaluation_population="full",
        val_subset_n=0,
    )
    validate_protocol_args(
        args, list(TEMPORAL_PROTOCOL_VARIANTS), list(PROTOCOL_HORIZONS)
    )
    args.budget = 4096
    try:
        validate_protocol_args(
            args, list(TEMPORAL_PROTOCOL_VARIANTS), list(PROTOCOL_HORIZONS)
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("temporal protocol accepted K=4096 drift")
    args.budget = 3840
    args.val_subset_n = 2048
    try:
        validate_protocol_args(
            args, list(TEMPORAL_PROTOCOL_VARIANTS), list(PROTOCOL_HORIZONS)
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("full temporal protocol accepted a subset artifact")
    args.evaluation_population = "smoke"
    validate_protocol_args(
        args, list(TEMPORAL_PROTOCOL_VARIANTS), list(PROTOCOL_HORIZONS)
    )
    args.val_subset_n = 0
    try:
        validate_protocol_args(
            args, list(TEMPORAL_PROTOCOL_VARIANTS), list(PROTOCOL_HORIZONS)
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("smoke temporal protocol accepted the full population")
    args.evaluation_population = "unspecified"
    try:
        validate_protocol_args(
            args, list(TEMPORAL_PROTOCOL_VARIANTS), list(PROTOCOL_HORIZONS)
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("temporal protocol accepted an unspecified population")


def main() -> None:
    test_all_allocators_exact_k_across_contexts()
    test_row_keyed_random_quota_permutations()
    test_attention_mass_capacity_projection()
    test_per_row_quotas_feed_one_uniform_spatial_selector()
    test_temporal_protocol_is_fail_closed()
    print("B17 temporal budget pruning contracts: PASS")


if __name__ == "__main__":
    main()
