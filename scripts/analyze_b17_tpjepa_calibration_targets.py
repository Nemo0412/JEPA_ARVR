#!/usr/bin/env python3
"""Audit schema-v2 B17 calibration targets before held-out unlock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from app.hdepic_lora_action_anticipation.task_protected_jepa_pruning import (
    ORACLE_PROTOCOL_ID,
    sha256_file,
)


TARGET_SCHEMA_VERSION = 2
EXPECTED_ROWS = 512
EXPECTED_GROUPS = 144
NONDEGENERATE_FRACTION_MIN = 0.99
TASK_ACTION_LOSS_SPEARMAN_MEDIAN_MIN = 0.50
TASK_MARGIN_SPEARMAN_MEDIAN_MIN = 0.30
TASK_ACTION_LOSS_SIGN_AGREEMENT_MIN = 0.60
ACTION_LOSS_PROTECTION_OVERLAP_MEDIAN_MIN = 0.50
MARGIN_PROTECTION_OVERLAP_MEDIAN_MIN = 0.35
PROTECTION_QUOTAS = (16, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-shard", type=Path, action="append", required=True)
    parser.add_argument("--expected-target-sha256", action="append", required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _ranks_descending(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("rank input must have shape [R,G]")
    ranks = torch.empty_like(values, dtype=torch.float64)
    group_count = values.shape[1]
    for row_index in range(values.shape[0]):
        _, inverse, counts = torch.unique(
            values[row_index].double(), sorted=True, return_inverse=True, return_counts=True
        )
        cumulative = counts.cumsum(dim=0)
        average_descending_rank = (
            group_count - cumulative.double() + (counts.double() - 1.0) / 2.0
        )
        ranks[row_index] = average_descending_rank[inverse]
    return ranks


def spearman_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Spearman inputs must have identical [R,G] shapes")
    left_rank = _ranks_descending(left.double())
    right_rank = _ranks_descending(right.double())
    left_centered = left_rank - left_rank.mean(dim=1, keepdim=True)
    right_centered = right_rank - right_rank.mean(dim=1, keepdim=True)
    denominator = left_centered.norm(dim=1) * right_centered.norm(dim=1)
    result = (left_centered * right_centered).sum(dim=1) / denominator.clamp_min(1e-12)
    result[denominator <= 1e-12] = torch.nan
    return result


def protection_overlap_rows(
    reference: torch.Tensor,
    comparison: torch.Tensor,
    quota: int,
) -> torch.Tensor:
    if reference.shape != comparison.shape or reference.ndim != 2:
        raise ValueError("protection inputs must have identical [R,G] shapes")
    if quota <= 0 or quota > reference.shape[1]:
        raise ValueError("invalid protection quota")
    reference_top = torch.argsort(reference, dim=1, descending=True, stable=True)[:, :quota]
    comparison_top = torch.argsort(comparison, dim=1, descending=True, stable=True)[:, :quota]
    overlap = (reference_top.unsqueeze(2) == comparison_top.unsqueeze(1)).any(dim=2).sum(dim=1)
    return overlap.double() / float(quota)


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    finite = values[torch.isfinite(values)].double()
    if finite.numel() == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "median": float(finite.median().item()),
        "q10": float(torch.quantile(finite, 0.10).item()),
        "q90": float(torch.quantile(finite, 0.90).item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
    }


def _row_keys_from_manifest(path: Path) -> list[str]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return [f"{row['video_id']}\t{int(row['tick_frame'])}" for row in csv.DictReader(handle)]


def main() -> None:
    args = parse_args()
    if len(args.target_shard) != len(args.expected_target_sha256):
        raise ValueError("target shard and expected hash counts differ")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    manifest_hash = sha256_file(args.manifest_csv)
    if manifest_hash != args.expected_manifest_sha256:
        raise RuntimeError("calibration manifest SHA-256 mismatch")

    loaded = []
    for path, expected_hash in zip(args.target_shard, args.expected_target_sha256):
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"target shard SHA-256 mismatch for {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard.get("protocol_id") != ORACLE_PROTOCOL_ID:
            raise RuntimeError(f"protocol mismatch in {path}")
        if int(shard.get("target_schema_version", -1)) != TARGET_SCHEMA_VERSION:
            raise RuntimeError(f"target schema mismatch in {path}")
        if shard.get("manifest_split") != "calibration":
            raise RuntimeError(f"non-calibration shard supplied: {path}")
        loaded.append((int(shard["shard_start"]), path, actual_hash, shard))
    loaded.sort(key=lambda item: item[0])
    sorted_paths = [item[1] for item in loaded]
    shard_hashes = [item[2] for item in loaded]
    loaded_shards = [item[3] for item in loaded]

    expected_start = 0
    for shard in loaded_shards:
        if int(shard["shard_start"]) != expected_start:
            raise RuntimeError(f"shard coverage gap/overlap at row {expected_start}")
        expected_start += int(shard["shard_count"])
    if expected_start != EXPECTED_ROWS:
        raise RuntimeError(f"target shards cover {expected_start} rows, expected {EXPECTED_ROWS}")

    row_keys = [key for shard in loaded_shards for key in shard["row_keys"]]
    manifest_row_keys = _row_keys_from_manifest(args.manifest_csv)
    if row_keys != manifest_row_keys:
        raise RuntimeError("ordered target row keys do not equal calibration manifest")
    row_keys_hash = hashlib.sha256(
        "".join(f"{key}\n" for key in row_keys).encode("utf-8")
    ).hexdigest()

    tensor_keys = (
        "task_utility",
        "action_loss_utility",
        "action_margin_drop",
        "jepa_cosine_residual",
        "jepa_normalized_mse",
        "attention_group_score",
    )
    tensors = {}
    for key in tensor_keys:
        tensors[key] = torch.cat([shard[key].float() for shard in loaded_shards], dim=0)
        if tensors[key].shape != (EXPECTED_ROWS, EXPECTED_GROUPS):
            raise RuntimeError(f"{key} shape is {tuple(tensors[key].shape)}")
        if not torch.isfinite(tensors[key]).all():
            raise RuntimeError(f"{key} contains non-finite values")

    task = tensors["task_utility"]
    action_loss = tensors["action_loss_utility"]
    margin = tensors["action_margin_drop"]
    jepa_cosine = tensors["jepa_cosine_residual"]
    jepa_mse = tensors["jepa_normalized_mse"]
    task_action_spearman = spearman_rows(task, action_loss)
    task_margin_spearman = spearman_rows(task, margin)
    jepa_metric_spearman = spearman_rows(jepa_cosine, jepa_mse)

    nondegenerate = {
        key: float((value.std(dim=1) > 1e-6).float().mean().item())
        for key, value in tensors.items()
    }
    sign_mask = (task.abs() > 1e-6) & (action_loss.abs() > 1e-6)
    sign_agreement = float(
        ((task.sign() == action_loss.sign()) & sign_mask).sum().double().div(sign_mask.sum().clamp_min(1)).item()
    )
    overlaps = {}
    for quota in PROTECTION_QUOTAS:
        overlaps[str(quota)] = {
            "action_loss": _summary(protection_overlap_rows(task, action_loss, quota)),
            "action_margin": _summary(protection_overlap_rows(task, margin, quota)),
        }

    diagnostics = {
        "nondegenerate_row_fraction": nondegenerate,
        "task_action_loss_spearman": _summary(task_action_spearman),
        "task_action_margin_spearman": _summary(task_margin_spearman),
        "jepa_cosine_mse_spearman": _summary(jepa_metric_spearman),
        "task_action_loss_sign_agreement": sign_agreement,
        "protection_overlap": overlaps,
        "value_ranges": {
            key: {"min": float(value.min().item()), "max": float(value.max().item())}
            for key, value in tensors.items()
        },
    }
    thresholds = {
        "nondegenerate_fraction_min": NONDEGENERATE_FRACTION_MIN,
        "task_action_loss_spearman_median_min": TASK_ACTION_LOSS_SPEARMAN_MEDIAN_MIN,
        "task_margin_spearman_median_min": TASK_MARGIN_SPEARMAN_MEDIAN_MIN,
        "task_action_loss_sign_agreement_min": TASK_ACTION_LOSS_SIGN_AGREEMENT_MIN,
        "action_loss_protection_overlap_median_min": ACTION_LOSS_PROTECTION_OVERLAP_MEDIAN_MIN,
        "margin_protection_overlap_median_min": MARGIN_PROTECTION_OVERLAP_MEDIAN_MIN,
        "protection_quotas": list(PROTECTION_QUOTAS),
    }
    failures = []
    for key in ("task_utility", "action_loss_utility", "action_margin_drop"):
        if nondegenerate[key] < NONDEGENERATE_FRACTION_MIN:
            failures.append(f"nondegenerate:{key}")
    if diagnostics["task_action_loss_spearman"]["median"] < TASK_ACTION_LOSS_SPEARMAN_MEDIAN_MIN:
        failures.append("task_action_loss_spearman")
    if diagnostics["task_action_margin_spearman"]["median"] < TASK_MARGIN_SPEARMAN_MEDIAN_MIN:
        failures.append("task_action_margin_spearman")
    if sign_agreement < TASK_ACTION_LOSS_SIGN_AGREEMENT_MIN:
        failures.append("task_action_loss_sign_agreement")
    for quota in PROTECTION_QUOTAS:
        quota_result = overlaps[str(quota)]
        if quota_result["action_loss"]["median"] < ACTION_LOSS_PROTECTION_OVERLAP_MEDIAN_MIN:
            failures.append(f"action_loss_protection_overlap_q{quota}")
        if quota_result["action_margin"]["median"] < MARGIN_PROTECTION_OVERLAP_MEDIAN_MIN:
            failures.append(f"margin_protection_overlap_q{quota}")

    output = {
        "protocol_id": ORACLE_PROTOCOL_ID,
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "auditor": str(Path(__file__).resolve()),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "rank_tie_method": "average",
        "population": "calibration",
        "row_count": len(row_keys),
        "groups_per_row": EXPECTED_GROUPS,
        "manifest_csv": str(args.manifest_csv),
        "manifest_sha256": manifest_hash,
        "row_keys_sha256": row_keys_hash,
        "target_shards": [str(path) for path in sorted_paths],
        "target_shard_sha256": shard_hashes,
        "thresholds_frozen_before_schema_v2_export": thresholds,
        "diagnostics": diagnostics,
        "gate_pass": not failures,
        "gate_failures": failures,
        "heldout_unlock": not failures,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
