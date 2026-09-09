#!/usr/bin/env python3
"""CPU contract tests for B17 task-protected JEPA pruning."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation.task_protected_jepa_pruning import (
    EncoderL16Continuation,
    OfflineOracleTargetAdapter,
    SpatiotemporalGroupLayout,
    aggregate_token_scores_to_groups,
    apply_group_swaps,
    build_attention_jepa_swap_candidates,
    counterfactual_marginal_utility,
    delete_group_batch,
    expand_group_indices,
    gather_latents_by_positions,
    masked_jepa_residual,
    merge_positioned_latents,
    oracle_row_key,
    partition_oracle_rows,
    predict_mask_only_latents,
    predict_masked_latents,
    predictor_lora_disabled,
    same_slot_mean_completion,
    select_jepa_only,
    select_attention_groups_exact,
    select_task_only,
    select_task_protected_jepa,
    weighted_mtp_loss_per_row,
    weighted_action_loss_per_row,
)
from scripts.analyze_b17_tpjepa_calibration_targets import (
    protection_overlap_rows,
    spearman_rows,
)
from scripts.analyze_b17_tpjepa_swap_oracle import cluster_bootstrap_mean_interval


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_layout_and_exact_k() -> None:
    formal = SpatiotemporalGroupLayout(group_side=4)
    coarse = SpatiotemporalGroupLayout(group_side=8)
    require(formal.num_tokens == 10240, "formal token count")
    require(formal.num_groups == 640, "formal group count")
    require(formal.num_recent_groups == 64, "formal recent-core count")
    require(coarse.num_groups == 160, "coarse group count")
    require(coarse.num_recent_groups == 16, "coarse recent-core count")
    require(formal.groups_for_budget(4096) == 256, "K4096 group count")
    require(formal.groups_for_budget(3072) == 192, "K3072 group count")
    groups = formal.token_indices()
    require(torch.unique(groups).numel() == formal.num_tokens, "groups must partition all tokens")
    require(int(groups.min()) == 0 and int(groups.max()) == 10239, "group token range")


def test_lexicographic_policy() -> None:
    layout = SpatiotemporalGroupLayout(num_slots=3, grid_size=4, group_side=2, recent_slots=1)
    # 12 groups total, 4 recent; K=8 groups -> protect 2 old task groups, then 2 residual groups.
    task = torch.tensor([9.0, 8.0, 1.0, 0.0, 7.0, 6.0, 5.0, 4.0, 100.0, 100.0, 100.0, 100.0])
    residual = torch.tensor([0.0, 1.0, 8.0, 9.0, 2.0, 3.0, 7.0, 6.0, -1.0, -1.0, -1.0, -1.0])
    keep = select_task_protected_jepa(task, residual, layout, token_budget=32)
    require(keep.tolist() == [0, 1, 2, 3, 8, 9, 10, 11], f"unexpected lexicographic keep: {keep}")
    task_only = select_task_only(task, layout, token_budget=32)
    require(task_only.tolist() == [0, 1, 4, 5, 8, 9, 10, 11], "task-only policy")
    jepa_only = select_jepa_only(residual, layout, token_budget=32)
    require(jepa_only.tolist() == [2, 3, 6, 7, 8, 9, 10, 11], "JEPA-only policy")

    tied = torch.zeros(layout.num_groups)
    tie_keep = select_task_only(tied, layout, token_budget=32)
    require(tie_keep.tolist() == [0, 1, 2, 3, 8, 9, 10, 11], "tie-break must favor lower group index")
    tokens = expand_group_indices(keep, layout.token_indices())
    require(tokens.numel() == 32 and bool((tokens[1:] > tokens[:-1]).all()), "exact unique token K")
    group_scores = aggregate_token_scores_to_groups(torch.arange(layout.num_tokens).float(), layout.token_indices())
    require(group_scores.shape == (layout.num_groups,), "group score shape")


def test_attention_jepa_swap_candidates() -> None:
    layout = SpatiotemporalGroupLayout(
        num_slots=3, grid_size=4, group_side=2, recent_slots=1
    )
    attention = torch.tensor(
        [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    )
    residual = torch.tensor(
        [9.0, 8.0, 7.0, 0.0, 1.0, 2.0, 6.0, 5.0, 10.0, 11.0, 3.0, 4.0]
    )
    keep = select_attention_groups_exact(attention, layout, token_budget=32)
    require(keep.tolist() == list(range(8)), "raw group-attention exact-K set")
    pairs = build_attention_jepa_swap_candidates(
        keep, attention, residual, layout, candidate_count=2
    )
    require(
        pairs["jepa_disagreement"].tolist() == [[3, 9], [4, 8]],
        "JEPA disagreement direction",
    )
    require(
        pairs["anti_jepa"].tolist() == [[0, 10], [1, 11]],
        "anti-JEPA direction",
    )
    require(
        pairs["attention_boundary"].tolist() == [[7, 8], [6, 9]],
        "attention boundary control",
    )
    swapped = apply_group_swaps(
        keep, pairs["jepa_disagreement"], layout, token_budget=32
    )
    require(swapped.shape == (2, 8), "one exact-K set per candidate")
    require(swapped[0].tolist() == [0, 1, 2, 4, 5, 6, 7, 9], "first swap")
    require(swapped[1].tolist() == [0, 1, 2, 3, 5, 6, 7, 8], "second swap")
    require(bool((swapped[:, 1:] > swapped[:, :-1]).all()), "unique sorted groups")


def test_delete_group_batch() -> None:
    features = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
    positions = torch.arange(8).unsqueeze(0)
    deleted = torch.tensor([[1, 2], [5, 7]])
    kept, kept_pos, target, target_pos = delete_group_batch(features, positions, deleted)
    require(kept.shape == (2, 6, 3), "deletion batch feature shape")
    require(kept_pos.tolist() == [[0, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 6]], "kept positions")
    require(target_pos.tolist() == deleted.tolist(), "target positions")
    require(torch.equal(target[0], features[0, deleted[0]]), "target latent gather")


class DummyPatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = 1
        self.tubelet_size = 1

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return clips.permute(0, 2, 3, 4, 1).reshape(clips.shape[0], -1, clips.shape[1])


class DummyBlock(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        if mask is None:
            return x + self.amount
        return x + self.amount + mask.unsqueeze(-1).to(x.dtype) * 0.001


class DummyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = DummyPatchEmbed()
        self.blocks = nn.ModuleList([DummyBlock(float(i + 1)) for i in range(4)])
        self.norm = nn.LayerNorm(3)
        self.use_rope = True
        self.handle_nonsquare_inputs = True
        self.patch_size = 1
        self.tubelet_size = 1
        self.out_layers = None
        self.pos_drop = nn.Identity()
        self.embed_dim = 3

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(clips)
        t, h, w = clips.shape[2:]
        positions = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
        for block in self.blocks:
            x = block(x, mask=positions, T=t, H_patches=h, W_patches=w)
        return self.norm(x)


def test_l16_continuation_parity() -> None:
    torch.manual_seed(17)
    encoder = DummyEncoder().eval()
    clips = torch.randn(2, 3, 2, 2, 2)
    split = EncoderL16Continuation(encoder, prune_layer=1)
    split_output, positions = split.forward_full(clips)
    native_output = encoder(clips)
    torch.testing.assert_close(split_output, native_output, rtol=0, atol=0)
    require(positions.tolist() == [list(range(8)), list(range(8))], "full token positions")


class CopyByPositionPredictor(nn.Module):
    num_patches = 64

    def forward(self, context, masks_x, masks_y):
        # Target values are intentionally unavailable. This deterministic mock
        # depends only on context summary and requested target positions.
        base = context.mean(dim=1, keepdim=True)
        return base.expand(-1, masks_y.shape[1], -1) + masks_y.unsqueeze(-1).float() * 0.0


class MaskAwarePredictor(nn.Module):
    num_patches = 64

    def forward(self, context, masks_x, masks_y):
        return masks_y.unsqueeze(-1).to(context.dtype).expand(-1, -1, context.shape[-1])


class DummyCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = DummyEncoder()
        self.predictor = CopyByPositionPredictor()
        self.grid_size = 2
        self.frames_per_second = 2
        self.tubelet_size = 1
        self.num_output_frames = 1
        self.num_steps = 1


class DummyMTPClassifier(nn.Module):
    def forward(self, tokens: torch.Tensor):
        signal = tokens.mean(dim=(1, 2))
        logits = torch.stack((signal, -signal), dim=1)
        return {
            horizon: {"verb": logits, "noun": logits, "action": logits}
            for horizon in (2.0, 4.0, 6.0)
        }


class FakeLoRA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._is_predictor_lora = True
        self.scaling = 2.5


def test_masked_jepa_no_leakage() -> None:
    predictor = CopyByPositionPredictor()
    context = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    context_pos = torch.tensor([[0, 2]])
    target_pos = torch.tensor([[1]])
    target = torch.tensor([[[1.0, 0.0]]])
    result = masked_jepa_residual(predictor, context, context_pos, target_pos, target)
    torch.testing.assert_close(result["cosine_residual"], torch.zeros(1))
    try:
        masked_jepa_residual(predictor, context, torch.tensor([[0, 1]]), target_pos, target)
    except ValueError as exc:
        require("leakage" in str(exc), "overlap must fail as leakage")
    else:
        raise AssertionError("overlapping JEPA masks did not fail closed")

    fake = FakeLoRA()
    with predictor_lora_disabled(fake) as count:
        require(count == 1 and fake.scaling == 0.0, "predictor LoRA must be disabled in scope")
    require(fake.scaling == 2.5, "predictor LoRA scaling must be restored")


def test_e0_completion_contracts() -> None:
    retained = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]]])
    retained_pos = torch.tensor([[0, 2, 5]])
    target_pos = torch.tensor([[1, 3, 4]])
    predictor = MaskAwarePredictor()
    predicted = predict_masked_latents(predictor, retained, retained_pos, target_pos)
    require(predicted.shape == (1, 3, 2), "E0 predictor output shape")
    mask_only = predict_mask_only_latents(
        predictor, target_pos, output_dim=2, dtype=retained.dtype
    )
    torch.testing.assert_close(mask_only, predicted)
    merged, merged_pos = merge_positioned_latents(
        retained, retained_pos, predicted, target_pos
    )
    require(merged.shape == (1, 6, 2), "E0 merged token count")
    require(merged_pos.tolist() == [[0, 1, 2, 3, 4, 5]], "E0 original positions")
    gathered = gather_latents_by_positions(merged, merged_pos, target_pos)
    torch.testing.assert_close(gathered, predicted)

    pooled = same_slot_mean_completion(
        retained,
        retained_pos,
        target_pos,
        tokens_per_slot=3,
    )
    expected = torch.tensor([[[2.0, 1.0], [5.0, 4.0], [5.0, 4.0]]])
    torch.testing.assert_close(pooled, expected)
    try:
        predict_masked_latents(
            predictor, retained, retained_pos, torch.tensor([[1, 2, 4]])
        )
    except ValueError as exc:
        require("leakage" in str(exc), "E0 overlap must fail as target leakage")
    else:
        raise AssertionError("E0 predictor accepted overlapping target positions")


def test_offline_target_adapter() -> None:
    torch.manual_seed(19)
    core = DummyCore().eval()
    classifier = DummyMTPClassifier().eval()
    adapter = OfflineOracleTargetAdapter(
        core,
        classifier,
        verb_map={10: 0},
        noun_map={20: 0},
        action_map={(10, 20): 0},
        prune_layer=1,
    )
    targets = adapter.generate_row_targets(
        clips=torch.randn(1, 3, 2, 2, 2),
        raw_verbs=torch.tensor([[10, 10, 10]]),
        raw_nouns=torch.tensor([[20, 20, 20]]),
        target_mask=torch.ones(1, 3),
        deletion_groups=torch.tensor([[0, 1], [2, 3]]),
        deletion_batch_size=1,
    )
    for key in (
        "deletion_task_loss",
        "task_utility",
        "deletion_action_loss",
        "action_loss_utility",
        "action_margin_drop",
        "jepa_cosine_residual",
        "jepa_normalized_mse",
    ):
        require(targets[key].shape == (2,), f"{key} shape")
        require(bool(torch.isfinite(targets[key]).all()), f"{key} finite")
    require(int(targets["predictor_lora_modules_disabled"].item()) == 0, "dummy has no LoRA")


def test_weighted_task_utility() -> None:
    outputs = {}
    for horizon in ("2", "4", "6"):
        outputs[horizon] = {
            "verb": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            "noun": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            "action": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        }
    raw_verbs = torch.tensor([[10, 10, 10], [11, 11, 11]])
    raw_nouns = torch.tensor([[20, 20, 20], [21, 21, 21]])
    mask = torch.ones_like(raw_verbs, dtype=torch.float32)
    losses, valid = weighted_mtp_loss_per_row(
        outputs,
        raw_verbs,
        raw_nouns,
        mask,
        horizons=("2", "4", "6"),
        horizon_weights=(1.0, 0.7, 0.5),
        verb_map={10: 0, 11: 1},
        noun_map={20: 0, 21: 1},
        action_map={(10, 20): 0, (11, 21): 1},
    )
    require(valid.tolist() == [3, 3], "all three horizons must contribute")
    require(bool((losses > 0).all()), "cross entropy must be positive")
    action_losses, action_valid = weighted_action_loss_per_row(
        outputs,
        raw_verbs,
        raw_nouns,
        mask,
        horizons=("2", "4", "6"),
        horizon_weights=(1.0, 0.7, 0.5),
        action_map={(10, 20): 0, (11, 21): 1},
    )
    require(action_valid.tolist() == [3, 3], "all action-only horizons must contribute")
    require(bool((action_losses > 0).all()), "action-only cross entropy must be positive")
    delta = counterfactual_marginal_utility(losses[:1], losses + torch.tensor([1.0, -0.5]))
    torch.testing.assert_close(delta, torch.tensor([1.0, -0.5]))


def make_manifest_row(video_id: str, tick: int, eligible: bool = True) -> dict[str, str]:
    return {
        "video_id": video_id,
        "tick_frame": str(tick),
        "context_sec": "10.0" if eligible else "8.0",
        "mtp_mask": json.dumps([1.0, 1.0, 1.0]),
        "mtp_verbs": json.dumps([1, 2, 3]),
        "mtp_nouns": json.dumps([4, 5, 6]),
    }


def test_manifest_partition() -> None:
    rows = []
    for video_idx in range(86):
        for tick in range(110):
            rows.append(make_manifest_row(f"video_{video_idx:03d}", tick))
        rows.append(make_manifest_row(f"video_{video_idx:03d}", 1000, eligible=False))
    partition_a = partition_oracle_rows(rows)
    partition_b = partition_oracle_rows(list(reversed(rows)))
    require(len(partition_a["calibration_videos"]) == 64, "calibration video count")
    require(len(partition_a["heldout_videos"]) == 22, "held-out video count")
    require(len(partition_a["calibration_rows"]) == 512, "calibration row count")
    require(len(partition_a["heldout_rows"]) == 2048, "held-out row count")
    cal_videos = set(partition_a["calibration_videos"])
    heldout_videos = set(partition_a["heldout_videos"])
    require(cal_videos.isdisjoint(heldout_videos), "video split leakage")
    require(
        [oracle_row_key(row) for row in partition_a["calibration_rows"]]
        == [oracle_row_key(row) for row in partition_b["calibration_rows"]],
        "manifest selection must be input-order invariant",
    )
    require(
        [oracle_row_key(row) for row in partition_a["heldout_rows"]]
        == [oracle_row_key(row) for row in partition_b["heldout_rows"]],
        "held-out selection must be input-order invariant",
    )


def test_target_stability_math() -> None:
    reference = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    identical = reference.clone()
    reversed_signal = torch.flip(reference, dims=(1,))
    torch.testing.assert_close(spearman_rows(reference, identical), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(
        spearman_rows(reference, reversed_signal), -torch.ones(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        protection_overlap_rows(reference, identical, quota=2), torch.ones(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        protection_overlap_rows(reference, reversed_signal, quota=2), torch.zeros(2, dtype=torch.float64)
    )
    tied_left = torch.tensor([[3.0, 3.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    tied_right = torch.tensor([[8.0, 8.0, 4.0, 4.0], [5.0, 4.0, 3.0, 2.0]])
    tied_spearman = spearman_rows(tied_left, tied_right)
    torch.testing.assert_close(tied_spearman[:1], torch.ones(1, dtype=torch.float64))
    require(bool(torch.isnan(tied_spearman[1])), "constant row Spearman must be undefined")


def test_cluster_bootstrap_interval() -> None:
    positive = torch.tensor([1.0, 2.0, 3.0, 4.0])
    lower, upper = cluster_bootstrap_mean_interval(
        positive, ["a", "a", "b", "b"], seed=17, samples=1000
    )
    require(0.0 < lower <= upper, "positive cluster-bootstrap interval")


def main() -> None:
    tests = [
        test_layout_and_exact_k,
        test_lexicographic_policy,
        test_attention_jepa_swap_candidates,
        test_delete_group_batch,
        test_l16_continuation_parity,
        test_masked_jepa_no_leakage,
        test_e0_completion_contracts,
        test_offline_target_adapter,
        test_weighted_task_utility,
        test_manifest_partition,
        test_target_stability_math,
        test_cluster_bootstrap_interval,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all={len(tests)}")


if __name__ == "__main__":
    main()
