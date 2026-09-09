#!/usr/bin/env python3
"""Evaluate B17 E0 L16 predictor-compensated completion feasibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.attn_midlayer_pruning import _MidLayerAttnCapture
from app.hdepic_lora_action_anticipation.eval_stream_mtp_fixed_budget_prune import (
    IndexedStreamMTPDataset,
    _validate_checkpoint_contract,
    collate_indexed_stream,
)
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (
    enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.evaluate_task_protected_jepa_oracle import (
    _evaluate_selected,
)
from app.hdepic_lora_action_anticipation.export_task_protected_jepa_targets import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_ENCODER_LORA_SHA256,
    EXPECTED_PARENT_SHA256,
    EXPECTED_PREDICTOR_LORA_SHA256,
    HORIZONS,
)
from app.hdepic_lora_action_anticipation.task_protected_jepa_pruning import (
    COMPLETION_E0_PROTOCOL_ID,
    EXPECTED_TRAIN_CSV_SHA256,
    EncoderL16Continuation,
    SpatiotemporalGroupLayout,
    aggregate_token_scores_to_groups,
    expand_group_indices,
    gather_latents_by_positions,
    merge_positioned_latents,
    predict_mask_only_latents,
    predict_masked_latents,
    predictor_lora_disabled,
    same_slot_mean_completion,
    select_attention_groups_exact,
    sha256_file,
)


REFERENCE_BUDGET = 4096
RETAINED_BUDGET = 3072
BRIDGE_TOKENS = REFERENCE_BUDGET - RETAINED_BUDGET
SCIENTIFIC_START = 128
SCIENTIFIC_ROWS = 128
LATENCY_ROWS = 4
LATENCY_WARMUP = 1
LATENCY_REPEATS = 5
TEACHER_ARMS = (
    "reference_k4096",
    "retained_k3072",
    "teacher_completion",
    "mask_only_completion",
    "zero_completion",
    "mean_completion",
    "same_slot_pool_completion",
)
JEPA_ARMS = ("reference_k4096", "retained_k3072", "jepa_completion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--stage", choices=("teacher", "jepa"), required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-lora", type=Path, required=True)
    parser.add_argument("--predictor-lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-start", type=int, default=SCIENTIFIC_START)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "unknown"))
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--code-snapshot-sha256", required=True)
    return parser.parse_args()


def _require_hash(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _gather_l16_groups(
    l16_tokens: torch.Tensor,
    positions: torch.Tensor,
    group_keep: torch.Tensor,
    group_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_keep = expand_group_indices(group_keep, group_tokens).unsqueeze(0)
    selected = l16_tokens.gather(
        1, token_keep.unsqueeze(-1).expand(-1, -1, l16_tokens.shape[-1])
    )
    return selected, positions.gather(1, token_keep)


def _new_metric_store() -> dict[str, list[torch.Tensor]]:
    return {
        "task_loss": [],
        "action_loss": [],
        "action_margin": [],
        "action_top1_correct": [],
        "action_top5_correct": [],
    }


def _append_metrics(store: dict[str, list[torch.Tensor]], metrics: dict[str, torch.Tensor]) -> None:
    for key, value in metrics.items():
        store[key].append(value.detach().cpu())


def _completion_cosine(completion: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (F.normalize(completion.float(), dim=-1) * F.normalize(teacher.float(), dim=-1)).sum(
        dim=-1
    ).mean(dim=-1)


def _predictor_dense_macs(
    *,
    context_tokens: int,
    target_tokens: int,
    input_dim: int,
    predictor_dim: int,
    output_dim: int,
    depth: int,
) -> int:
    total_tokens = context_tokens + target_tokens
    return int(
        context_tokens * input_dim * predictor_dim
        + depth
        * (
            12 * total_tokens * predictor_dim * predictor_dim
            + 2 * total_tokens * total_tokens * predictor_dim
        )
        + target_tokens * predictor_dim * output_dim
    )


def _dense_mac_contract(base: torch.nn.Module) -> dict[str, object]:
    encoder_dim = int(base.encoder.embed_dim)
    late_depth = len(base.encoder.blocks) - 17
    predictor = base.predictor
    predictor_dim = int(predictor.predictor_embed.out_features)
    predictor_depth = len(predictor.predictor_blocks)
    output_dim = int(predictor.predictor_proj.out_features)
    target_tokens = int(base.grid_size**2 * (base.num_output_frames // base.tubelet_size))
    steps = int(base.num_steps)

    def late(tokens: int) -> int:
        return int(
            late_depth
            * (12 * tokens * encoder_dim * encoder_dim + 2 * tokens * tokens * encoder_dim)
        )

    def future(context_tokens: int) -> int:
        return steps * _predictor_dense_macs(
            context_tokens=context_tokens,
            target_tokens=target_tokens,
            input_dim=encoder_dim,
            predictor_dim=predictor_dim,
            output_dim=output_dim,
            depth=predictor_depth,
        )

    recovery = _predictor_dense_macs(
        context_tokens=RETAINED_BUDGET,
        target_tokens=BRIDGE_TOKENS,
        input_dim=encoder_dim,
        predictor_dim=predictor_dim,
        output_dim=output_dim,
        depth=predictor_depth,
    )
    routes = {
        "reference_k4096": late(REFERENCE_BUDGET) + future(REFERENCE_BUDGET),
        "retained_k3072": late(RETAINED_BUDGET) + future(RETAINED_BUDGET),
        "teacher_completion": late(RETAINED_BUDGET) + future(REFERENCE_BUDGET),
        "mask_only_completion": late(RETAINED_BUDGET)
        + _predictor_dense_macs(
            context_tokens=0,
            target_tokens=BRIDGE_TOKENS,
            input_dim=encoder_dim,
            predictor_dim=predictor_dim,
            output_dim=output_dim,
            depth=predictor_depth,
        )
        + future(REFERENCE_BUDGET),
        "zero_completion": late(RETAINED_BUDGET) + future(REFERENCE_BUDGET),
        "mean_completion": late(RETAINED_BUDGET) + future(REFERENCE_BUDGET),
        "same_slot_pool_completion": late(RETAINED_BUDGET) + future(REFERENCE_BUDGET),
        "jepa_completion": late(RETAINED_BUDGET) + recovery + future(REFERENCE_BUDGET),
    }
    return {
        "unit": "dense multiply-accumulates",
        "scope": "blocks 17-23 plus recovery/future predictors; input encoder and task probe excluded",
        "formula": "block=12*N*D^2+2*N^2*D; linear projections counted once per weight multiply",
        "encoder_dim": encoder_dim,
        "late_block_count": late_depth,
        "predictor_dim": predictor_dim,
        "predictor_depth": predictor_depth,
        "future_target_tokens_per_step": target_tokens,
        "future_steps": steps,
        "routes": routes,
    }


def main() -> None:
    args = parse_args()
    if args.protocol_id != COMPLETION_E0_PROTOCOL_ID:
        raise SystemExit(
            f"protocol mismatch: expected {COMPLETION_E0_PROTOCOL_ID}, got {args.protocol_id}"
        )
    if args.shard_start != SCIENTIFIC_START:
        raise ValueError("E0 is frozen to manifest offset 128")
    if args.shard_count not in (2, 8, SCIENTIFIC_ROWS):
        raise ValueError("shard-count must be a registered smoke size (2/8) or 128")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    sidecar_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.output} or {sidecar_path}")

    hashes = {
        "manifest_csv": _require_hash(
            args.manifest_csv, args.expected_manifest_sha256, "held-out manifest"
        ),
        "train_csv": _require_hash(args.train_csv, EXPECTED_TRAIN_CSV_SHA256, "train CSV"),
        "checkpoint": _require_hash(
            args.checkpoint, EXPECTED_CHECKPOINT_SHA256, "V-JEPA checkpoint"
        ),
        "parent_checkpoint": _require_hash(
            args.parent_checkpoint, EXPECTED_PARENT_SHA256, "stream-MTP parent"
        ),
        "encoder_lora": _require_hash(
            args.encoder_lora, EXPECTED_ENCODER_LORA_SHA256, "encoder LoRA"
        ),
        "predictor_lora": _require_hash(
            args.predictor_lora, EXPECTED_PREDICTOR_LORA_SHA256, "predictor LoRA"
        ),
    }
    dataset = IndexedStreamMTPDataset(args.manifest_csv, args.video_root, 256, src_fps=8, fps=8)
    if len(dataset.rows) != 2048:
        raise RuntimeError(f"expected 2048-row held-out manifest, got {len(dataset.rows)}")
    dataset.rows = dataset.rows[args.shard_start : args.shard_start + args.shard_count]
    expected_row_keys = [f"{row['video_id']}\t{int(row['tick_frame'])}" for row in dataset.rows]
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_indexed_stream,
        "pin_memory": False,
        "persistent_workers": False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    if not torch.cuda.is_available():
        raise RuntimeError("E0 completion evaluation requires CUDA")
    device = torch.device("cuda")
    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    base = T.build_model(device, max_frames=80, fps=8, img_size=256, checkpoint=str(args.checkpoint))
    T.load_lora_sidecars(base, str(args.encoder_lora), str(args.predictor_lora))
    layout = SpatiotemporalGroupLayout(group_side=8, recent_slots=0)
    enlarge_predictor_budget(base, layout.num_tokens, layout.tokens_per_slot)
    load_wrapper = T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).to(device)
    classifier = T.AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=int(base.encoder.embed_dim),
        num_heads=16,
        depth=4,
        use_activation_checkpointing=True,
    ).to(device)
    mtp_classifier = T.CommunicatingMLPMTPClassifier(
        classifier, horizons_sec=HORIZONS, comm_layers=2, comm_heads=4
    ).to(device)
    checkpoint = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    _validate_checkpoint_contract(checkpoint, verb_map, noun_map, action_map, list(HORIZONS))
    load_wrapper.load_state_dict(checkpoint["model"], strict=True)
    mtp_classifier.load_state_dict(checkpoint["mtp_classifier"], strict=True)
    parent_best = checkpoint.get("best")
    del checkpoint
    base = load_wrapper.base
    base.eval()
    mtp_classifier.eval()
    for module in (base, mtp_classifier):
        for parameter in module.parameters():
            parameter.requires_grad = False

    continuation = EncoderL16Continuation(base.encoder, prune_layer=16)
    attention_capture = _MidLayerAttnCapture(base.encoder.blocks[16].attn)
    group_tokens = layout.token_indices(device)
    arms = TEACHER_ARMS if args.stage == "teacher" else JEPA_ARMS
    metrics = {arm: _new_metric_store() for arm in arms}
    completion_cosine: dict[str, list[torch.Tensor]] = {
        arm: [] for arm in arms if arm.endswith("completion")
    }
    latency_ms: dict[str, list[float]] = {
        arm: []
        for arm in (("reference_k4096", "retained_k3072") if args.stage == "teacher" else ("reference_k4096", "jepa_completion"))
    }
    row_keys: list[str] = []
    video_ids: list[str] = []
    tick_frames: list[int] = []
    parity_max_abs: list[float] = []
    reference_roundtrip_max_abs: list[float] = []
    disabled_counts: list[int] = []
    contract_counts = {
        "reference_tokens": [],
        "retained_tokens": [],
        "bridge_tokens": [],
        "retained_is_reference_attention_prefix": [],
        "retained_bridge_disjoint": [],
    }
    macs = _dense_mac_contract(base)
    started = time.time()

    def build_arm(
        clips: torch.Tensor,
        arm: str,
    ) -> torch.Tensor:
        l16, positions, geometry = continuation.encode_to_prune(clips)
        if attention_capture.importance is None:
            raise RuntimeError("L16 attention capture did not emit token importance")
        attention_score = aggregate_token_scores_to_groups(
            attention_capture.importance[0], group_tokens
        ).float()
        reference_groups = select_attention_groups_exact(
            attention_score, layout, REFERENCE_BUDGET
        )
        retained_groups = select_attention_groups_exact(
            attention_score, layout, RETAINED_BUDGET
        )
        bridge_groups = reference_groups[~torch.isin(reference_groups, retained_groups)]
        if bridge_groups.numel() != (
            layout.groups_for_budget(REFERENCE_BUDGET)
            - layout.groups_for_budget(RETAINED_BUDGET)
        ):
            raise RuntimeError("K3072 attention set is not nested inside K4096")
        reference_l16, reference_positions = _gather_l16_groups(
            l16, positions, reference_groups, group_tokens
        )
        retained_l16, retained_positions = _gather_l16_groups(
            l16, positions, retained_groups, group_tokens
        )
        _, bridge_positions = _gather_l16_groups(l16, positions, bridge_groups, group_tokens)
        reference_final = continuation.continue_from(reference_l16, reference_positions, geometry)
        if arm == "reference_k4096":
            return reference_final
        retained_final = continuation.continue_from(retained_l16, retained_positions, geometry)
        if arm == "retained_k3072":
            return retained_final
        teacher = gather_latents_by_positions(reference_final, reference_positions, bridge_positions)
        if arm == "teacher_completion":
            completion = teacher
        elif arm == "zero_completion":
            completion = torch.zeros_like(teacher)
        elif arm == "mean_completion":
            completion = retained_final.mean(dim=1, keepdim=True).expand_as(teacher)
        elif arm == "same_slot_pool_completion":
            completion = same_slot_mean_completion(
                retained_final,
                retained_positions,
                bridge_positions,
                tokens_per_slot=layout.tokens_per_slot,
            )
        elif arm == "mask_only_completion":
            with predictor_lora_disabled(base) as disabled_count:
                if disabled_count <= 0:
                    raise RuntimeError("mask-only control did not disable predictor LoRA")
                completion = predict_mask_only_latents(
                    base.predictor,
                    bridge_positions,
                    output_dim=retained_final.shape[-1],
                    dtype=retained_final.dtype,
                )
        elif arm == "jepa_completion":
            with predictor_lora_disabled(base) as disabled_count:
                if disabled_count <= 0:
                    raise RuntimeError("JEPA completion did not disable predictor LoRA")
                completion = predict_masked_latents(
                    base.predictor, retained_final, retained_positions, bridge_positions
                )
        else:
            raise ValueError(f"unknown E0 arm {arm}")
        completed, completed_positions = merge_positioned_latents(
            retained_final, retained_positions, completion, bridge_positions
        )
        if completed.shape[1] != REFERENCE_BUDGET:
            raise RuntimeError("completion arm violated exact K4096 task-interface count")
        if not torch.equal(completed_positions, reference_positions):
            raise RuntimeError("completion arm did not restore the exact reference positions")
        return completed

    def time_arm(
        clips: torch.Tensor,
        arm: str,
        raw_verbs: torch.Tensor,
        raw_nouns: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> float:
        def execute() -> None:
            final_tokens = build_arm(clips, arm)
            _evaluate_selected(
                base,
                mtp_classifier,
                final_tokens,
                raw_verbs,
                raw_nouns,
                target_mask,
                verb_map,
                noun_map,
                action_map,
            )

        for _ in range(LATENCY_WARMUP):
            execute()
        torch.cuda.synchronize()
        samples = []
        for _ in range(LATENCY_REPEATS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            execute()
            end_event.record()
            torch.cuda.synchronize()
            samples.append(float(start_event.elapsed_time(end_event)))
        return float(torch.tensor(samples, dtype=torch.float64).median().item())

    try:
        for row_index, batch in enumerate(loader):
            clips = batch["clip"].to(device).float().div_(255.0)
            clips = clips.sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
            raw_verbs = batch["mtp_verbs"].to(device)
            raw_nouns = batch["mtp_nouns"].to(device)
            target_mask = batch["mtp_mask"].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                l16_tokens, positions, geometry = continuation.encode_to_prune(clips)
                if attention_capture.importance is None:
                    raise RuntimeError("L16 attention capture did not emit token importance")
                attention_score = aggregate_token_scores_to_groups(
                    attention_capture.importance[0], group_tokens
                ).float()
                reference_groups = select_attention_groups_exact(
                    attention_score, layout, REFERENCE_BUDGET
                )
                independently_selected = select_attention_groups_exact(
                    attention_score, layout, RETAINED_BUDGET
                )
                retained_groups = independently_selected
                bridge_groups = reference_groups[
                    ~torch.isin(reference_groups, retained_groups)
                ]
                if bridge_groups.numel() != (
                    layout.groups_for_budget(REFERENCE_BUDGET)
                    - layout.groups_for_budget(RETAINED_BUDGET)
                ):
                    raise RuntimeError("K3072 attention set is not nested inside K4096")
                reference_l16, reference_positions = _gather_l16_groups(
                    l16_tokens, positions, reference_groups, group_tokens
                )
                retained_l16, retained_positions = _gather_l16_groups(
                    l16_tokens, positions, retained_groups, group_tokens
                )
                _, bridge_positions = _gather_l16_groups(
                    l16_tokens, positions, bridge_groups, group_tokens
                )
                reference_final = continuation.continue_from(
                    reference_l16, reference_positions, geometry
                )
                retained_final = continuation.continue_from(
                    retained_l16, retained_positions, geometry
                )
                teacher = gather_latents_by_positions(
                    reference_final, reference_positions, bridge_positions
                )
                reference_retained = gather_latents_by_positions(
                    reference_final, reference_positions, retained_positions
                )
                roundtrip, roundtrip_positions = merge_positioned_latents(
                    reference_retained, retained_positions, teacher, bridge_positions
                )
                roundtrip_error = float(
                    (roundtrip.float() - reference_final.float()).abs().max().item()
                )
                if not torch.equal(roundtrip_positions, reference_positions) or roundtrip_error != 0.0:
                    raise RuntimeError("untouched K4096 reference roundtrip parity failed")
                if row_index == 0:
                    native_final = base.encoder(clips)
                    full_final = continuation.continue_from(l16_tokens, positions, geometry)
                    parity = float((full_final.float() - native_final.float()).abs().max().item())
                    if parity > 1e-5:
                        raise RuntimeError(f"native/L16 no-delete parity max_abs={parity}")
                    parity_max_abs.append(parity)
                    del native_final, full_final
                reference_roundtrip_max_abs.append(roundtrip_error)

                if not bool(torch.isin(retained_groups, reference_groups).all()):
                    raise RuntimeError(
                        "K3072 is not the exact attention-ranking prefix set of K4096"
                    )
                if bool(torch.isin(retained_positions, bridge_positions).any()):
                    raise RuntimeError("retained and recovery bridge positions overlap")
                contract_counts["reference_tokens"].append(reference_positions.shape[1])
                contract_counts["retained_tokens"].append(retained_positions.shape[1])
                contract_counts["bridge_tokens"].append(bridge_positions.shape[1])
                contract_counts["retained_is_reference_attention_prefix"].append(True)
                contract_counts["retained_bridge_disjoint"].append(True)

                arm_tokens = {
                    "reference_k4096": reference_final,
                    "retained_k3072": retained_final,
                }
                if args.stage == "teacher":
                    with predictor_lora_disabled(base) as disabled_count:
                        disabled_counts.append(disabled_count)
                        mask_only = predict_mask_only_latents(
                            base.predictor,
                            bridge_positions,
                            output_dim=retained_final.shape[-1],
                            dtype=retained_final.dtype,
                        )
                    completions = {
                        "teacher_completion": teacher,
                        "mask_only_completion": mask_only,
                        "zero_completion": torch.zeros_like(teacher),
                        "mean_completion": retained_final.mean(dim=1, keepdim=True).expand_as(
                            teacher
                        ),
                        "same_slot_pool_completion": same_slot_mean_completion(
                            retained_final,
                            retained_positions,
                            bridge_positions,
                            tokens_per_slot=layout.tokens_per_slot,
                        ),
                    }
                else:
                    with predictor_lora_disabled(base) as disabled_count:
                        disabled_counts.append(disabled_count)
                        predicted = predict_masked_latents(
                            base.predictor,
                            retained_final,
                            retained_positions,
                            bridge_positions,
                        )
                    completions = {"jepa_completion": predicted}

                for arm, completion in completions.items():
                    completed, completed_positions = merge_positioned_latents(
                        retained_final, retained_positions, completion, bridge_positions
                    )
                    if completed.shape[1] != REFERENCE_BUDGET:
                        raise RuntimeError(f"{arm} count mismatch")
                    if not torch.equal(completed_positions, reference_positions):
                        raise RuntimeError(f"{arm} original-position parity failed")
                    arm_tokens[arm] = completed
                    completion_cosine[arm].append(
                        _completion_cosine(completion, teacher).detach().cpu()
                    )
                for arm in arms:
                    result = _evaluate_selected(
                        base,
                        mtp_classifier,
                        arm_tokens[arm],
                        raw_verbs,
                        raw_nouns,
                        target_mask,
                        verb_map,
                        noun_map,
                        action_map,
                    )
                    _append_metrics(metrics[arm], result)

                if row_index < LATENCY_ROWS:
                    timing_arms = list(latency_ms)
                    if row_index % 2:
                        timing_arms.reverse()
                    for arm in timing_arms:
                        latency_ms[arm].append(
                            time_arm(clips, arm, raw_verbs, raw_nouns, target_mask)
                        )

            video_id = batch["video_id"][0]
            tick_frame = int(batch["tick_frame"][0].item())
            row_keys.append(f"{video_id}\t{tick_frame}")
            video_ids.append(video_id)
            tick_frames.append(tick_frame)
            torch.cuda.synchronize()
            print(
                f"[completion-e0:{args.stage}] row={row_index + 1}/{args.shard_count} "
                f"key={row_keys[-1]} elapsed_sec={time.time() - started:.1f}",
                flush=True,
            )
    finally:
        attention_capture.remove()

    if row_keys != expected_row_keys:
        raise RuntimeError("processed row keys do not match frozen manifest slice")
    if not disabled_counts or min(disabled_counts) <= 0 or len(set(disabled_counts)) != 1:
        raise RuntimeError("predictor-LoRA disable count is missing or unstable")
    serialized_metrics = {
        arm: {key: torch.cat(values, dim=0) for key, values in store.items()}
        for arm, store in metrics.items()
    }
    serialized_cosine = {
        arm: torch.cat(values, dim=0) for arm, values in completion_cosine.items()
    }
    payload = {
        "protocol_id": COMPLETION_E0_PROTOCOL_ID,
        "diagnostic_schema_version": 1,
        "stage": args.stage,
        "manifest_csv": str(args.manifest_csv),
        "manifest_rows": 2048,
        "shard_start": args.shard_start,
        "shard_count": args.shard_count,
        "row_keys": row_keys,
        "video_ids": video_ids,
        "tick_frames": tick_frames,
        "hashes": hashes,
        "group_layout": {
            "num_slots": layout.num_slots,
            "grid_size": layout.grid_size,
            "group_side": layout.group_side,
            "recent_slots": layout.recent_slots,
            "num_groups": layout.num_groups,
            "tokens_per_group": layout.tokens_per_group,
        },
        "prune_layer": 16,
        "reference_budget": REFERENCE_BUDGET,
        "retained_budget": RETAINED_BUDGET,
        "bridge_tokens": BRIDGE_TOKENS,
        "teacher_target": "bridge positions gathered from the K4096 attention-route block-23 output",
        "recovery_interface": "separate predictor pass at original positions; task predictor then uses registered rebase path",
        "arms": arms,
        "metrics": serialized_metrics,
        "completion_cosine_to_teacher": serialized_cosine,
        "native_l16_parity_max_abs": parity_max_abs,
        "reference_roundtrip_max_abs": reference_roundtrip_max_abs,
        "predictor_lora_modules_disabled": disabled_counts[0],
        "contract_counts": contract_counts,
        "latency_contract": {
            "scope": "model-only clip tensor through encoder, completion/future predictor, and communicating task probe",
            "device_local": True,
            "paired_rows": min(args.shard_count, LATENCY_ROWS),
            "warmup_per_row_arm": LATENCY_WARMUP,
            "repeats_per_row_arm": LATENCY_REPEATS,
            "row_summary": "median CUDA-event milliseconds across repeats",
            "arm_latency_ms": latency_ms,
        },
        "dense_mac_contract": macs,
        "parent_best": parent_best,
        "job_id": str(args.job_id),
        "code_commit": args.code_commit,
        "code_snapshot_sha256": args.code_snapshot_sha256,
        "elapsed_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{args.job_id}")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    artifact_hash = sha256_file(args.output)
    sidecar = {
        "complete": True,
        "protocol_id": COMPLETION_E0_PROTOCOL_ID,
        "diagnostic_schema_version": 1,
        "stage": args.stage,
        "row_count": len(row_keys),
        "row_keys_sha256": hashlib.sha256(
            "".join(f"{key}\n" for key in row_keys).encode("utf-8")
        ).hexdigest(),
        "artifact": str(args.output),
        "artifact_sha256": artifact_hash,
        "code_snapshot_sha256": args.code_snapshot_sha256,
        "job_id": str(args.job_id),
    }
    sidecar_temporary = sidecar_path.with_suffix(sidecar_path.suffix + f".tmp.{args.job_id}")
    with sidecar_temporary.open("x", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(sidecar_temporary, sidecar_path)
    print(json.dumps(sidecar, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
