#!/usr/bin/env python3
"""Evaluate a bounded same-budget attention/JEPA group-swap oracle at L16."""

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
    EXPECTED_TRAIN_CSV_SHA256,
    SWAP_ORACLE_PROTOCOL_ID,
    OfflineJEPARedundancyAdapter,
    SpatiotemporalGroupLayout,
    aggregate_token_scores_to_groups,
    apply_group_swaps,
    build_attention_jepa_swap_candidates,
    expand_group_indices,
    select_attention_groups_exact,
    sha256_file,
)


BUDGETS = (4096, 3072)
STRATEGIES = ("jepa_disagreement", "anti_jepa", "attention_boundary")
SCIENTIFIC_ROWS = 128
CANDIDATE_COUNT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-lora", type=Path, required=True)
    parser.add_argument("--predictor-lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-start", type=int, default=0)
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


def _append_metrics(store: dict[str, list[torch.Tensor]], metrics: dict[str, torch.Tensor]) -> None:
    for key, value in metrics.items():
        store[key].append(value.detach().cpu())


def _new_metric_store() -> dict[str, list[torch.Tensor]]:
    return {
        "task_loss": [],
        "action_loss": [],
        "action_margin": [],
        "action_top1_correct": [],
        "action_top5_correct": [],
    }


def main() -> None:
    args = parse_args()
    if args.protocol_id != SWAP_ORACLE_PROTOCOL_ID:
        raise SystemExit(
            f"protocol mismatch: expected {SWAP_ORACLE_PROTOCOL_ID}, got {args.protocol_id}"
        )
    if args.shard_start != 0:
        raise ValueError("v2 diagnostic is frozen to the leading held-out slice")
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
    dataset.rows = dataset.rows[: args.shard_count]
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
        raise RuntimeError("swap oracle requires CUDA")
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

    adapter = OfflineJEPARedundancyAdapter(base, prune_layer=16)
    continuation = adapter.continuation
    attention_capture = _MidLayerAttnCapture(base.encoder.blocks[16].attn)
    group_tokens = layout.token_indices(device)
    results: dict[int, dict[str, object]] = {}
    for budget in BUDGETS:
        results[budget] = {
            "attention_keep": [],
            "baseline": _new_metric_store(),
            "swap_pairs": {strategy: [] for strategy in STRATEGIES},
            "candidates": {strategy: _new_metric_store() for strategy in STRATEGIES},
        }
    row_keys: list[str] = []
    video_ids: list[str] = []
    tick_frames: list[int] = []
    attention_rows: list[torch.Tensor] = []
    residual_rows: list[torch.Tensor] = []
    residual_mse_rows: list[torch.Tensor] = []
    parity_max_abs: list[float] = []
    disabled_counts: list[int] = []
    started = time.time()
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
                attention_token = attention_capture.importance[0].clone()
                full_final = continuation.continue_from(l16_tokens, positions, geometry)
                if row_index == 0:
                    native_final = base.encoder(clips)
                    parity = float((full_final.float() - native_final.float()).abs().max().item())
                    if parity > 1e-5:
                        raise RuntimeError(f"native/L16 no-delete parity max_abs={parity}")
                    parity_max_abs.append(parity)
                    del native_final
                residual = adapter.residuals_from_l16(
                    l16_tokens,
                    positions,
                    full_final,
                    geometry,
                    group_tokens,
                    group_batch_size=1,
                )
                jepa_score = residual["jepa_cosine_residual"].float()
                attention_score = aggregate_token_scores_to_groups(
                    attention_token, group_tokens
                ).float()
                if not bool(torch.isfinite(jepa_score).all()):
                    raise RuntimeError("non-finite JEPA residual")
                disabled_counts.append(int(residual["predictor_lora_modules_disabled"].item()))
                attention_rows.append(attention_score.detach().cpu())
                residual_rows.append(jepa_score.detach().cpu())
                residual_mse_rows.append(residual["jepa_normalized_mse"].detach().cpu())

                for budget in BUDGETS:
                    budget_store = results[budget]
                    attention_keep = select_attention_groups_exact(
                        attention_score, layout, budget
                    )
                    budget_store["attention_keep"].append(attention_keep.detach().cpu())
                    baseline_l16, baseline_positions = _gather_l16_groups(
                        l16_tokens, positions, attention_keep, group_tokens
                    )
                    baseline_final = continuation.continue_from(
                        baseline_l16, baseline_positions, geometry
                    )
                    baseline_metrics = _evaluate_selected(
                        base,
                        mtp_classifier,
                        baseline_final,
                        raw_verbs,
                        raw_nouns,
                        target_mask,
                        verb_map,
                        noun_map,
                        action_map,
                    )
                    _append_metrics(budget_store["baseline"], baseline_metrics)
                    candidates = build_attention_jepa_swap_candidates(
                        attention_keep,
                        attention_score,
                        jepa_score,
                        layout,
                        candidate_count=CANDIDATE_COUNT,
                    )
                    for strategy in STRATEGIES:
                        pairs = candidates[strategy]
                        budget_store["swap_pairs"][strategy].append(pairs.detach().cpu())
                        variants = apply_group_swaps(
                            attention_keep, pairs, layout, budget
                        )
                        row_candidate_metrics = _new_metric_store()
                        for candidate_keep in variants:
                            candidate_l16, candidate_positions = _gather_l16_groups(
                                l16_tokens, positions, candidate_keep, group_tokens
                            )
                            candidate_final = continuation.continue_from(
                                candidate_l16, candidate_positions, geometry
                            )
                            candidate_metrics = _evaluate_selected(
                                base,
                                mtp_classifier,
                                candidate_final,
                                raw_verbs,
                                raw_nouns,
                                target_mask,
                                verb_map,
                                noun_map,
                                action_map,
                            )
                            _append_metrics(row_candidate_metrics, candidate_metrics)
                        for key, values in row_candidate_metrics.items():
                            budget_store["candidates"][strategy][key].append(
                                torch.cat(values, dim=0)
                            )
            video_id = batch["video_id"][0]
            tick_frame = int(batch["tick_frame"][0].item())
            row_keys.append(f"{video_id}\t{tick_frame}")
            video_ids.append(video_id)
            tick_frames.append(tick_frame)
            torch.cuda.synchronize()
            print(
                f"[swap-oracle] row={row_index + 1}/{args.shard_count} "
                f"key={row_keys[-1]} elapsed_sec={time.time() - started:.1f}",
                flush=True,
            )
    finally:
        attention_capture.remove()

    if row_keys != expected_row_keys:
        raise RuntimeError("processed row keys do not match the frozen manifest prefix")
    if not disabled_counts or min(disabled_counts) <= 0 or len(set(disabled_counts)) != 1:
        raise RuntimeError("predictor-LoRA disable count is missing or unstable")

    serialized: dict[str, object] = {}
    for budget in BUDGETS:
        budget_store = results[budget]
        serialized_budget = {
            "attention_keep": torch.stack(budget_store["attention_keep"]),
            "baseline": {},
            "swap_pairs": {},
            "candidates": {},
        }
        for key, values in budget_store["baseline"].items():
            serialized_budget["baseline"][key] = torch.cat(values, dim=0)
        for strategy in STRATEGIES:
            serialized_budget["swap_pairs"][strategy] = torch.stack(
                budget_store["swap_pairs"][strategy]
            )
            serialized_budget["candidates"][strategy] = {}
            for key, values in budget_store["candidates"][strategy].items():
                serialized_budget["candidates"][strategy][key] = torch.stack(values)
        serialized[str(budget)] = serialized_budget

    payload = {
        "protocol_id": SWAP_ORACLE_PROTOCOL_ID,
        "diagnostic_schema_version": 1,
        "manifest_csv": str(args.manifest_csv),
        "manifest_rows": 2048,
        "shard_start": 0,
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
        "predictor_position_mode": "rebase",
        "budgets": BUDGETS,
        "candidate_count": CANDIDATE_COUNT,
        "strategies": STRATEGIES,
        "parent_best": parent_best,
        "native_l16_parity_max_abs": parity_max_abs,
        "predictor_lora_modules_disabled": disabled_counts[0],
        "attention_group_score": torch.stack(attention_rows),
        "jepa_cosine_residual": torch.stack(residual_rows),
        "jepa_normalized_mse": torch.stack(residual_mse_rows),
        "results": serialized,
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
        "protocol_id": SWAP_ORACLE_PROTOCOL_ID,
        "diagnostic_schema_version": 1,
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
