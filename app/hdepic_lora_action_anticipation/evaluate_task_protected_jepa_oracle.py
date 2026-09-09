#!/usr/bin/env python3
"""Evaluate actual multi-group oracle selections from frozen B17 target shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
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
from app.hdepic_lora_action_anticipation.export_task_protected_jepa_targets import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_ENCODER_LORA_SHA256,
    EXPECTED_PARENT_SHA256,
    EXPECTED_PREDICTOR_LORA_SHA256,
    HORIZONS,
    HORIZON_WEIGHTS,
)
from app.hdepic_lora_action_anticipation.task_protected_jepa_pruning import (
    EXPECTED_TRAIN_CSV_SHA256,
    ORACLE_PROTOCOL_ID,
    EncoderL16Continuation,
    SpatiotemporalGroupLayout,
    expand_group_indices,
    predict_future_tokens_rebased,
    select_attention_group,
    select_jepa_only,
    select_task_only,
    select_task_protected_jepa,
    sha256_file,
    weighted_action_margin_per_row,
    weighted_action_loss_per_row,
    weighted_mtp_loss_per_row,
)


GROUP_VARIANTS = (
    "attention_group",
    "counterfactual_task_only",
    "jepa_only",
    "task_protected_jepa",
)
BUDGETS = (4096, 3072)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--target-shard", type=Path, required=True)
    parser.add_argument("--expected-target-sha256", required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-lora", type=Path, required=True)
    parser.add_argument("--predictor-lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def _full_group_scores(
    compact: torch.Tensor,
    deletion_group_ids: torch.Tensor,
    layout: SpatiotemporalGroupLayout,
    device: torch.device,
) -> torch.Tensor:
    if compact.shape != (deletion_group_ids.numel(),):
        raise ValueError("compact target score shape does not match deletion group IDs")
    scores = torch.zeros(layout.num_groups, device=device, dtype=torch.float32)
    scores[deletion_group_ids.to(device)] = compact.to(device=device, dtype=torch.float32)
    return scores


def _mapped_action_targets(
    raw_verbs: torch.Tensor,
    raw_nouns: torch.Tensor,
    target_mask: torch.Tensor,
    action_map: dict[tuple[int, int], int],
) -> torch.Tensor:
    labels = []
    for horizon_index in range(len(HORIZONS)):
        if float(target_mask[0, horizon_index].item()) < 0.5:
            raise ValueError("oracle evaluator requires all horizons")
        pair = (
            int(raw_verbs[0, horizon_index].item()),
            int(raw_nouns[0, horizon_index].item()),
        )
        if pair not in action_map:
            raise ValueError(f"unmapped action label {pair}")
        labels.append(int(action_map[pair]))
    return torch.tensor(labels, device=raw_verbs.device, dtype=torch.long)


@torch.no_grad()
def _evaluate_selected(
    core: torch.nn.Module,
    classifier: torch.nn.Module,
    final_tokens: torch.Tensor,
    raw_verbs: torch.Tensor,
    raw_nouns: torch.Tensor,
    target_mask: torch.Tensor,
    verb_map: dict[int, int],
    noun_map: dict[int, int],
    action_map: dict[tuple[int, int], int],
) -> dict[str, torch.Tensor]:
    anticipation = torch.full(
        (final_tokens.shape[0],), 2.0, device=final_tokens.device, dtype=torch.float32
    )
    outputs = classifier(predict_future_tokens_rebased(core, final_tokens, anticipation))
    loss, valid_loss = weighted_mtp_loss_per_row(
        outputs,
        raw_verbs,
        raw_nouns,
        target_mask,
        HORIZONS,
        HORIZON_WEIGHTS,
        verb_map,
        noun_map,
        action_map,
    )
    margin, valid_margin = weighted_action_margin_per_row(
        outputs,
        raw_verbs,
        raw_nouns,
        target_mask,
        HORIZONS,
        HORIZON_WEIGHTS,
        action_map,
    )
    action_loss, valid_action_loss = weighted_action_loss_per_row(
        outputs,
        raw_verbs,
        raw_nouns,
        target_mask,
        HORIZONS,
        HORIZON_WEIGHTS,
        action_map,
    )
    if (
        not torch.equal(valid_loss, valid_margin)
        or not torch.equal(valid_loss, valid_action_loss)
        or not bool((valid_loss == len(HORIZONS)).all())
    ):
        raise ValueError("not all horizons contributed to oracle evaluation")
    labels = _mapped_action_targets(raw_verbs, raw_nouns, target_mask, action_map)
    top1 = []
    top5 = []
    for horizon_index, horizon in enumerate(HORIZONS):
        logits = outputs[horizon]["action"].float()
        label = labels[horizon_index]
        top1.append(logits.argmax(dim=1).eq(label))
        top5.append(logits.topk(min(5, logits.shape[1]), dim=1).indices.eq(label).any(dim=1))
    return {
        "task_loss": loss,
        "action_loss": action_loss,
        "action_margin": margin,
        "action_top1_correct": torch.stack(top1, dim=1),
        "action_top5_correct": torch.stack(top5, dim=1),
    }


def main() -> None:
    args = parse_args()
    if args.protocol_id != ORACLE_PROTOCOL_ID:
        raise SystemExit(f"protocol mismatch: expected {ORACLE_PROTOCOL_ID}, got {args.protocol_id}")
    sidecar_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.output} or {sidecar_path}")
    hashes = {
        "target_shard": _require_hash(
            args.target_shard, args.expected_target_sha256, "target shard"
        ),
        "manifest_csv": _require_hash(
            args.manifest_csv, args.expected_manifest_sha256, "manifest CSV"
        ),
        "train_csv": _require_hash(args.train_csv, EXPECTED_TRAIN_CSV_SHA256, "train CSV"),
        "checkpoint": _require_hash(args.checkpoint, EXPECTED_CHECKPOINT_SHA256, "V-JEPA checkpoint"),
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
    targets = torch.load(args.target_shard, map_location="cpu", weights_only=False)
    if targets.get("protocol_id") != ORACLE_PROTOCOL_ID:
        raise RuntimeError("target shard protocol mismatch")
    if int(targets.get("target_schema_version", -1)) != 2:
        raise RuntimeError("oracle evaluator requires target_schema_version=2")
    layout_config = targets.get("group_layout", {})
    layout = SpatiotemporalGroupLayout(
        num_slots=int(layout_config.get("num_slots", -1)),
        grid_size=int(layout_config.get("grid_size", -1)),
        group_side=int(layout_config.get("group_side", -1)),
        recent_slots=int(layout_config.get("recent_slots", -1)),
    )
    if layout.group_side != 8 or layout.num_groups != 160 or layout.num_recent_groups != 16:
        raise RuntimeError("oracle evaluator requires the registered coarse 8x8 layout")
    shard_start = int(targets["shard_start"])
    shard_count = int(targets["shard_count"])
    if len(targets["row_keys"]) != shard_count:
        raise RuntimeError("target shard row-key count mismatch")

    dataset = IndexedStreamMTPDataset(
        args.manifest_csv, args.video_root, 256, src_fps=8, fps=8
    )
    if shard_start + shard_count > len(dataset.rows):
        raise RuntimeError("target shard range exceeds manifest")
    dataset.rows = dataset.rows[shard_start : shard_start + shard_count]
    expected_row_keys = [f"{row['video_id']}\t{int(row['tick_frame'])}" for row in dataset.rows]
    if expected_row_keys != targets["row_keys"]:
        raise RuntimeError("target shard row keys do not match manifest slice")
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
    loader_iter = iter(loader)

    if not torch.cuda.is_available():
        raise RuntimeError("oracle evaluation requires CUDA")
    device = torch.device("cuda")
    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    base = T.build_model(device, max_frames=80, fps=8, img_size=256, checkpoint=str(args.checkpoint))
    T.load_lora_sidecars(base, str(args.encoder_lora), str(args.predictor_lora))
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
    all_group_tokens = layout.token_indices(device)
    deletion_group_ids = layout_config["deletion_group_ids"].long()
    if not torch.equal(
        deletion_group_ids,
        torch.nonzero(~layout.recent_group_mask(), as_tuple=False).flatten(),
    ):
        raise RuntimeError("target shard deletion groups are not exactly all non-core groups")

    arm_names = ["baseline_full", "token_attention_anchor_k4096"]
    arm_names.extend(
        f"{variant}_k{budget}" for budget in BUDGETS for variant in GROUP_VARIANTS
    )
    results = {
        arm: {
            "task_loss": [],
            "action_loss": [],
            "action_margin": [],
            "action_top1_correct": [],
            "action_top5_correct": [],
            "selected_token_count": [],
        }
        for arm in arm_names
    }
    selected_group_ids: dict[str, list[torch.Tensor]] = {
        arm: [] for arm in arm_names if arm not in ("baseline_full", "token_attention_anchor_k4096")
    }
    started = time.time()
    try:
        for row_index, batch in enumerate(loader_iter):
            clips = batch["clip"].to(device).float().div_(255.0)
            clips = clips.sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
            raw_verbs = batch["mtp_verbs"].to(device)
            raw_nouns = batch["mtp_nouns"].to(device)
            target_mask = batch["mtp_mask"].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                l16_tokens, positions, geometry = continuation.encode_to_prune(clips)
                full_final = continuation.continue_from(l16_tokens, positions, geometry)
                arm_final: dict[str, torch.Tensor] = {"baseline_full": full_final}
                if attention_capture.importance is None:
                    raise RuntimeError("L16 attention capture did not emit importance")
                token_keep = attention_capture.importance.topk(4096, dim=1).indices.sort(dim=1).values
                token_l16 = l16_tokens.gather(
                    1, token_keep.unsqueeze(-1).expand(-1, -1, l16_tokens.shape[-1])
                )
                token_positions = positions.gather(1, token_keep)
                arm_final["token_attention_anchor_k4096"] = continuation.continue_from(
                    token_l16, token_positions, geometry
                )

                task_score = _full_group_scores(
                    targets["task_utility"][row_index], deletion_group_ids, layout, device
                )
                jepa_score = _full_group_scores(
                    targets["jepa_cosine_residual"][row_index], deletion_group_ids, layout, device
                )
                attention_score = _full_group_scores(
                    targets["attention_group_score"][row_index], deletion_group_ids, layout, device
                )
                for budget in BUDGETS:
                    keep_by_variant = {
                        "attention_group": select_attention_group(attention_score, layout, budget),
                        "counterfactual_task_only": select_task_only(task_score, layout, budget),
                        "jepa_only": select_jepa_only(jepa_score, layout, budget),
                        "task_protected_jepa": select_task_protected_jepa(
                            task_score, jepa_score, layout, budget
                        ),
                    }
                    for variant, group_keep in keep_by_variant.items():
                        arm = f"{variant}_k{budget}"
                        token_indices = expand_group_indices(group_keep, all_group_tokens).unsqueeze(0)
                        selected_l16 = l16_tokens.gather(
                            1,
                            token_indices.unsqueeze(-1).expand(-1, -1, l16_tokens.shape[-1]),
                        )
                        selected_positions = positions.gather(1, token_indices)
                        arm_final[arm] = continuation.continue_from(
                            selected_l16, selected_positions, geometry
                        )
                        selected_group_ids[arm].append(group_keep.detach().cpu())

                for arm, final_tokens in arm_final.items():
                    scored = _evaluate_selected(
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
                    for key, value in scored.items():
                        results[arm][key].append(value.detach().cpu())
                    results[arm]["selected_token_count"].append(final_tokens.shape[1])
            torch.cuda.synchronize()
            print(
                f"[oracle] row={row_index + 1}/{shard_count} key={expected_row_keys[row_index]} "
                f"elapsed_sec={time.time() - started:.1f}",
                flush=True,
            )
    finally:
        attention_capture.remove()

    serialized_results = {}
    for arm, arm_result in results.items():
        serialized_results[arm] = {
            "task_loss": torch.cat(arm_result["task_loss"]),
            "action_loss": torch.cat(arm_result["action_loss"]),
            "action_margin": torch.cat(arm_result["action_margin"]),
            "action_top1_correct": torch.cat(arm_result["action_top1_correct"], dim=0),
            "action_top5_correct": torch.cat(arm_result["action_top5_correct"], dim=0),
            "selected_token_count": torch.tensor(arm_result["selected_token_count"]),
        }
    payload = {
        "protocol_id": ORACLE_PROTOCOL_ID,
        "target_shard": str(args.target_shard),
        "manifest_csv": str(args.manifest_csv),
        "hashes": hashes,
        "row_keys": expected_row_keys,
        "video_ids": targets["video_ids"],
        "tick_frames": targets["tick_frames"],
        "group_layout": layout_config,
        "prune_layer": 16,
        "budgets": BUDGETS,
        "group_variants": GROUP_VARIANTS,
        "parent_best": parent_best,
        "results": serialized_results,
        "selected_group_ids": {
            arm: torch.stack(rows) for arm, rows in selected_group_ids.items()
        },
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
        "protocol_id": ORACLE_PROTOCOL_ID,
        "row_count": shard_count,
        "row_keys_sha256": hashlib.sha256(
            "".join(f"{key}\n" for key in expected_row_keys).encode("utf-8")
        ).hexdigest(),
        "artifact": str(args.output),
        "artifact_sha256": artifact_hash,
        "target_shard_sha256": hashes["target_shard"],
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
