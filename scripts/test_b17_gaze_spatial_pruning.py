#!/usr/bin/env python3
"""Semantic contract tests for B17 temporal-quota gaze spatial pruning."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from app.hdepic_lora_action_anticipation.egtea_gaze import parse_gtea_gaze
from app.hdepic_lora_action_anticipation.eval_stream_mtp_fixed_budget_prune import (
    PROTOCOL_HORIZONS,
    PROTOCOL_VARIANTS,
    RANDOM_PROTOCOL_ID,
    RANDOM_PROTOCOL_VARIANTS,
    _build_calibration_ranking,
    _paired_summary,
    _validate_checkpoint_contract,
    _validate_stream_timeline,
    predict_from_selected_tokens,
    validate_protocol_args,
)
from app.hdepic_lora_action_anticipation.gaze_spatial_pruning import (
    QuotaSpatialTokenSelector,
    gather_tokens,
    valid_gaze_types,
)
from app.hdepic_lora_action_anticipation.temporal_budget_pruning import (
    LOSS_AWARE_10S_TEMPORAL_QUOTAS,
    LOSS_AWARE_PRIMARY_BUDGET,
    LossAwareTemporalQuotaProvider,
)


def test_primary_quota_identity() -> None:
    provider = LossAwareTemporalQuotaProvider()
    quotas = provider.quotas(40)
    assert quotas.tolist() == list(LOSS_AWARE_10S_TEMPORAL_QUOTAS)
    assert int(quotas.sum()) == LOSS_AWARE_PRIMARY_BUDGET == 3840
    assert int(quotas[-16:].sum()) == 2764
    ratio = int(quotas[-16:].sum()) / int(quotas.sum())
    assert abs(ratio - 0.7197916667) < 1e-8


def test_growing_context_quota_projection() -> None:
    provider = LossAwareTemporalQuotaProvider()
    expected = {
        16: [124, 188, 215, 241, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256],
        24: [35, 34, 45, 88, 54, 59, 91, 75, 91, 138, 157, 177, 210, 228, 200, 238,
             230, 216, 242, 243, 221, 256, 256, 256],
        32: [45, 33, 39, 47, 51, 27, 32, 29, 31, 31, 40, 80, 48, 53, 82, 68,
             82, 125, 142, 159, 189, 205, 180, 214, 208, 195, 218, 219, 200, 256, 256, 256],
        40: list(LOSS_AWARE_10S_TEMPORAL_QUOTAS),
    }
    for n_slots in (16, 24, 32, 40):
        quotas = provider.quotas(n_slots)
        assert quotas.shape == (n_slots,)
        assert int(quotas.sum()) == min(3840, n_slots * 256)
        assert bool((quotas >= 0).all()) and bool((quotas <= 256).all())
        assert quotas.tolist() == expected[n_slots]


def test_raw_gaze_type_semantics() -> None:
    payload = "\n".join(
        [
            "Time\tType\tDummy",
            "0 0 0 640 480 1 Fixation",
            "0 0 0 640 480 2 Saccade",
            "0 0 0 640 480 3 Blink",
            "0 0 0 640 480 4 -",
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "gaze.txt"
        path.write_text(payload + "\n", encoding="latin-1")
        gaze = parse_gtea_gaze(path)
    types = torch.from_numpy(gaze[1:5, 2])
    assert types.tolist() == [1.0, 2.0, 3.0, 3.0]
    assert valid_gaze_types(types).tolist() == [True, True, False, False]


def test_spatial_modes_and_missing_fallback() -> None:
    selector = QuotaSpatialTokenSelector(grid_size=4, tubelet_size=2)
    quotas = torch.tensor([1, 3])
    # Make attention choose local index 15 in slot 0 and 12,13,14 in slot 1.
    attention = torch.arange(32, dtype=torch.float32).view(1, 32)
    gaze_xy = torch.tensor([[[0.05, 0.05], [0.06, 0.05], [0.95, 0.95], [0.94, 0.95]]])
    all_valid = torch.ones(1, 4, dtype=torch.bool)

    gaze_idx, gaze_stats = selector.select(
        mode="gaze", quotas=quotas, attention_scores=attention,
        gaze_xy=gaze_xy, gaze_valid=all_valid,
    )
    assert int(gaze_idx[0, 0]) == 0, gaze_idx
    assert set(gaze_idx[0, 1:].tolist()) <= {26, 27, 30, 31}
    assert gaze_stats == {
        "gaze_selected_slots": 2,
        "attention_fallback_slots": 0,
        "total_slots": 2,
        "full_capacity_slots": 0,
    }

    calibrated = torch.arange(32, dtype=torch.float32).flip(0)
    calib_idx, _ = selector.select(
        mode="calib", quotas=quotas, attention_scores=attention,
        calibrated_scores=calibrated,
    )
    assert calib_idx.tolist() == [[0, 16, 17, 18]]

    shifted_idx, _ = selector.select(
        mode="gaze_shift", quotas=quotas, attention_scores=attention,
        gaze_xy=gaze_xy, gaze_valid=all_valid,
    )
    assert int(shifted_idx[0, 0]) != int(gaze_idx[0, 0])

    missing = all_valid.clone()
    missing[:, :2] = False
    missing_idx, missing_stats = selector.select(
        mode="gaze", quotas=quotas, attention_scores=attention,
        gaze_xy=gaze_xy, gaze_valid=missing,
    )
    attention_idx, _ = selector.select(
        mode="attention", quotas=quotas, attention_scores=attention,
    )
    assert int(missing_idx[0, 0]) == int(attention_idx[0, 0]) == 15
    assert missing_stats["attention_fallback_slots"] == 1

    for idx in (gaze_idx, shifted_idx, missing_idx, attention_idx, calib_idx):
        assert idx.shape == (1, 4)
        assert bool((idx[:, 1:] > idx[:, :-1]).all())
        counts = torch.bincount(idx[0] // 16, minlength=2)
        assert counts.tolist() == quotas.tolist()


def test_keyed_random_semantics() -> None:
    selector = QuotaSpatialTokenSelector(grid_size=4, tubelet_size=2)
    quotas = torch.tensor([3, 5])
    attention_a = torch.arange(64, dtype=torch.float32).view(2, 32)
    attention_b = attention_a.flip(1)
    keys = ["video-a|100", "video-b|200"]
    first, stats = selector.select(
        mode="random", quotas=quotas, attention_scores=attention_a,
        random_keys=keys, random_seed=17,
    )
    repeated, _ = selector.select(
        mode="random", quotas=quotas, attention_scores=attention_b,
        random_keys=keys, random_seed=17,
    )
    swapped, _ = selector.select(
        mode="random", quotas=quotas, attention_scores=attention_a.flip(0),
        random_keys=keys[::-1], random_seed=17,
    )
    changed_seed, _ = selector.select(
        mode="random", quotas=quotas, attention_scores=attention_a,
        random_keys=keys, random_seed=29,
    )
    assert torch.equal(first, repeated)  # independent of RGB/attention values
    assert torch.equal(first[0], swapped[1]) and torch.equal(first[1], swapped[0])
    assert not torch.equal(first, changed_seed)
    assert stats["random_sampled_slots"] == 4
    for row in first:
        assert row.unique().numel() == 8
        assert torch.bincount(row // 16, minlength=2).tolist() == quotas.tolist()

class _RecordingPredictor:
    def __init__(self):
        self.masks_x = None
        self.masks_y = None

    def __call__(self, x, *, masks_x, masks_y):
        self.masks_x = masks_x.detach().clone()
        self.masks_y = masks_y.detach().clone()
        return torch.zeros(x.shape[0], masks_y.shape[1], x.shape[2], dtype=x.dtype)


def test_true_full_position_contract() -> None:
    predictor = _RecordingPredictor()
    core = SimpleNamespace(
        grid_size=2,
        encoder=SimpleNamespace(embed_dim=3),
        frames_per_second=8,
        tubelet_size=2,
        num_output_frames=2,
        num_steps=1,
        predictor=predictor,
    )
    full = torch.arange(24, dtype=torch.float32).view(1, 8, 3)
    keep_idx = torch.tensor([[0, 3, 4, 7]])
    selected = gather_tokens(full, keep_idx)
    out = predict_from_selected_tokens(
        core,
        selected,
        keep_idx,
        torch.tensor([2.0]),
        num_full_tokens=8,
        position_mode="true_full",
    )
    assert predictor.masks_x.tolist() == keep_idx.tolist()
    # Two observed slots + eight anticipation slots => target starts at slot 10.
    assert predictor.masks_y.tolist() == [[40, 41, 42, 43]]
    assert out.shape == (1, 8, 3)


def test_protocol_and_statistical_contracts() -> None:
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
        protocol_id="egtea-stream-mtp-pruning/lossaware3840-gaze-spatial-native-v2",
        random_seed=17,
    )
    validate_protocol_args(args, list(PROTOCOL_VARIANTS), list(PROTOCOL_HORIZONS))
    args.position_mode = "rebase"
    try:
        validate_protocol_args(args, list(PROTOCOL_VARIANTS), list(PROTOCOL_HORIZONS))
    except SystemExit:
        pass
    else:
        raise AssertionError("registered v1 must reject rebased positions")
    args.position_mode = "true_full"
    args.protocol_id = RANDOM_PROTOCOL_ID
    validate_protocol_args(args, list(RANDOM_PROTOCOL_VARIANTS), list(PROTOCOL_HORIZONS))
    args.random_seed = 29
    try:
        validate_protocol_args(args, list(RANDOM_PROTOCOL_VARIANTS), list(PROTOCOL_HORIZONS))
    except SystemExit:
        pass
    else:
        raise AssertionError("registered random protocol must reject seed drift")
    _validate_stream_timeline(np.asarray([1, 4, 7]), tick_frame=8, vfps=24.0)
    for bad_frames, bad_tick, bad_fps in (([1, 4, 8], 8, 24.0), ([1, 1, 2], 8, 24.0), ([1, 4], 8, 30.0)):
        try:
            _validate_stream_timeline(np.asarray(bad_frames), tick_frame=bad_tick, vfps=bad_fps)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid stream timeline was accepted")

    summary = _paired_summary(
        [1, 1, 0, 0], [0, 1, 1, 0], ["session-a", "session-a", "session-b", "session-b"],
        mask=[True, False, True, False],
    )
    assert summary["n"] == 2 and summary["clusters"] == 2
    assert summary["a_correct_b_wrong"] == summary["a_wrong_b_correct"] == 1
    assert summary["row_level_mcnemar_exact_two_sided_p"] == 1.0

    verb_map, noun_map, action_map = {1: 0}, {2: 0}, {(1, 2): 0}
    checkpoint = {
        "verb_map": verb_map,
        "noun_map": noun_map,
        "action_map": {"1,2": 0},
        "horizons": [2.0, 4.0, 6.0],
    }
    _validate_checkpoint_contract(checkpoint, verb_map, noun_map, action_map, [2.0, 4.0, 6.0])
    checkpoint["horizons"] = [2.0]
    try:
        _validate_checkpoint_contract(checkpoint, verb_map, noun_map, action_map, [2.0, 4.0, 6.0])
    except RuntimeError:
        pass
    else:
        raise AssertionError("checkpoint horizon drift was accepted")


def test_calibration_mask_with_continuous_tie_break() -> None:
    selector = QuotaSpatialTokenSelector(grid_size=2, tubelet_size=2)
    mask = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw = torch.tensor([0.0, 0.2, 0.9, 0.5])
    ranking = _build_calibration_ranking(mask, raw, torch.tensor([1]), gp=4)
    original_idx, _ = selector.select(
        mode="calib",
        quotas=torch.tensor([1]),
        attention_scores=torch.zeros(1, 4),
        calibrated_scores=ranking,
    )
    assert original_idx.tolist() == [[0]]  # exact original mask
    expanded_idx, _ = selector.select(
        mode="calib",
        quotas=torch.tensor([2]),
        attention_scores=torch.zeros(1, 4),
        calibrated_scores=ranking,
    )
    assert expanded_idx.tolist() == [[0, 2]]  # extra token follows raw score, not raster tie


def main() -> None:
    test_primary_quota_identity()
    test_growing_context_quota_projection()
    test_raw_gaze_type_semantics()
    test_spatial_modes_and_missing_fallback()
    test_keyed_random_semantics()
    test_true_full_position_contract()
    test_protocol_and_statistical_contracts()
    test_calibration_mask_with_continuous_tie_break()
    print("B17 gaze spatial pruning contracts: PASS")


if __name__ == "__main__":
    main()
