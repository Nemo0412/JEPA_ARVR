#!/usr/bin/env python3
"""Audit and summarize the bounded B17 attention/JEPA swap oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from app.hdepic_lora_action_anticipation.task_protected_jepa_pruning import (
    SWAP_ORACLE_PROTOCOL_ID,
    sha256_file,
)


BUDGETS = (4096, 3072)
STRATEGIES = ("jepa_disagreement", "anti_jepa", "attention_boundary")
SCIENTIFIC_ROWS = 128
CANDIDATE_COUNT = 8
BOOTSTRAP_SEED = 170217
BOOTSTRAP_SAMPLES = 20000
MIN_RELATIVE_TASK_LOSS_REDUCTION = 0.005
MIN_IMPROVED_ROW_FRACTION = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-artifact-sha256", required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cluster_bootstrap_mean_interval(
    values: torch.Tensor,
    cluster_ids: list[str],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    values = values.detach().double().flatten().cpu()
    if values.numel() != len(cluster_ids) or values.numel() == 0:
        raise ValueError("values and non-empty cluster_ids must align")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("bootstrap values must be finite")
    clusters = sorted(set(cluster_ids))
    if len(clusters) < 2:
        raise ValueError("cluster bootstrap needs at least two clusters")
    sums = torch.tensor(
        [values[[index for index, item in enumerate(cluster_ids) if item == cluster]].sum() for cluster in clusters],
        dtype=torch.float64,
    )
    counts = torch.tensor(
        [sum(item == cluster for item in cluster_ids) for cluster in clusters],
        dtype=torch.float64,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    draws = torch.randint(
        len(clusters), (samples, len(clusters)), generator=generator
    )
    means = sums[draws].sum(dim=1) / counts[draws].sum(dim=1)
    lower, upper = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(lower.item()), float(upper.item())


def _manifest_row_keys(path: Path) -> tuple[list[str], list[str]]:
    import csv

    row_keys: list[str] = []
    video_ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = row["video_id"]
            row_keys.append(f"{video_id}\t{int(row['tick_frame'])}")
            video_ids.append(video_id)
            if len(row_keys) == SCIENTIFIC_ROWS:
                break
    if len(row_keys) != SCIENTIFIC_ROWS:
        raise RuntimeError("manifest does not contain the frozen 128-row prefix")
    return row_keys, video_ids


def _best_improvement(
    baseline: torch.Tensor, candidates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if baseline.ndim != 1 or candidates.ndim != 2:
        raise ValueError("baseline must be [R] and candidates [R,M]")
    if candidates.shape != (baseline.numel(), CANDIDATE_COUNT):
        raise ValueError("candidate loss tensor violates the frozen 128x8 contract")
    choices = torch.cat((baseline.unsqueeze(1), candidates), dim=1)
    best_loss, best_index = choices.min(dim=1)
    return baseline - best_loss, best_index


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    artifact_hash = sha256_file(args.artifact)
    if artifact_hash != args.expected_artifact_sha256:
        raise RuntimeError(
            f"artifact SHA-256 mismatch: expected {args.expected_artifact_sha256}, got {artifact_hash}"
        )
    manifest_hash = sha256_file(args.manifest_csv)
    if manifest_hash != args.expected_manifest_sha256:
        raise RuntimeError(
            f"manifest SHA-256 mismatch: expected {args.expected_manifest_sha256}, got {manifest_hash}"
        )
    payload = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if payload.get("protocol_id") != SWAP_ORACLE_PROTOCOL_ID:
        raise RuntimeError("swap artifact protocol mismatch")
    if int(payload.get("diagnostic_schema_version", -1)) != 1:
        raise RuntimeError("unsupported swap diagnostic schema")
    if int(payload.get("shard_start", -1)) != 0 or int(payload.get("shard_count", -1)) != SCIENTIFIC_ROWS:
        raise RuntimeError("scientific audit requires exactly the leading 128 held-out rows")
    if tuple(payload.get("budgets", ())) != BUDGETS:
        raise RuntimeError("budget contract mismatch")
    if tuple(payload.get("strategies", ())) != STRATEGIES:
        raise RuntimeError("strategy contract mismatch")
    if int(payload.get("candidate_count", -1)) != CANDIDATE_COUNT:
        raise RuntimeError("candidate-count contract mismatch")
    expected_keys, expected_videos = _manifest_row_keys(args.manifest_csv)
    if payload.get("row_keys") != expected_keys or payload.get("video_ids") != expected_videos:
        raise RuntimeError("artifact rows do not match the frozen held-out manifest prefix")
    if len(set(expected_videos)) != 22:
        raise RuntimeError("the frozen prefix must cover all 22 held-out videos")
    if float(max(payload.get("native_l16_parity_max_abs", [float("inf")]))) > 1e-5:
        raise RuntimeError("native/L16 parity failed")
    if int(payload.get("predictor_lora_modules_disabled", 0)) <= 0:
        raise RuntimeError("JEPA residual did not disable predictor LoRA")

    budget_results: dict[str, object] = {}
    budget_passes: dict[str, bool] = {}
    directional_points: dict[str, bool] = {}
    for budget in BUDGETS:
        raw = payload["results"][str(budget)]
        baseline = raw["baseline"]["task_loss"].double()
        if baseline.shape != (SCIENTIFIC_ROWS,) or not bool(torch.isfinite(baseline).all()):
            raise RuntimeError(f"invalid K{budget} baseline task loss")
        expected_groups = budget // 64
        attention_keep = raw["attention_keep"]
        if attention_keep.shape != (SCIENTIFIC_ROWS, expected_groups):
            raise RuntimeError(f"invalid K{budget} attention exact-K shape")
        if not bool((attention_keep[:, 1:] > attention_keep[:, :-1]).all()):
            raise RuntimeError(f"K{budget} attention sets are not sorted unique groups")

        improvements: dict[str, torch.Tensor] = {}
        summaries: dict[str, object] = {}
        for strategy_index, strategy in enumerate(STRATEGIES):
            candidate_loss = raw["candidates"][strategy]["task_loss"].double()
            if not bool(torch.isfinite(candidate_loss).all()):
                raise RuntimeError(f"non-finite candidate loss for {strategy} K{budget}")
            swap_pairs = raw["swap_pairs"][strategy]
            if swap_pairs.shape != (SCIENTIFIC_ROWS, CANDIDATE_COUNT, 2):
                raise RuntimeError(f"invalid swap-pair shape for {strategy} K{budget}")
            improvement, best_index = _best_improvement(baseline, candidate_loss)
            improvements[strategy] = improvement
            ci = cluster_bootstrap_mean_interval(
                improvement,
                expected_videos,
                seed=BOOTSTRAP_SEED + budget + strategy_index,
            )
            summaries[strategy] = {
                "mean_task_loss_improvement": float(improvement.mean().item()),
                "relative_mean_task_loss_reduction": float(
                    improvement.mean().item() / baseline.mean().item()
                ),
                "cluster_bootstrap_95ci": list(ci),
                "improved_row_fraction": float((improvement > 1e-8).double().mean().item()),
                "no_op_row_fraction": float((best_index == 0).double().mean().item()),
            }

        advantages: dict[str, object] = {}
        advantage_passes = []
        for control_index, control in enumerate(("anti_jepa", "attention_boundary")):
            advantage = improvements["jepa_disagreement"] - improvements[control]
            ci = cluster_bootstrap_mean_interval(
                advantage,
                expected_videos,
                seed=BOOTSTRAP_SEED + 10000 + budget + control_index,
            )
            advantages[control] = {
                "mean_incremental_improvement": float(advantage.mean().item()),
                "cluster_bootstrap_95ci": list(ci),
            }
            advantage_passes.append(ci[0] > 0.0)

        jepa = summaries["jepa_disagreement"]
        directional_points[str(budget)] = all(
            advantages[control]["mean_incremental_improvement"] > 0.0
            for control in ("anti_jepa", "attention_boundary")
        )
        budget_pass = bool(
            jepa["relative_mean_task_loss_reduction"] >= MIN_RELATIVE_TASK_LOSS_REDUCTION
            and jepa["cluster_bootstrap_95ci"][0] > 0.0
            and jepa["improved_row_fraction"] >= MIN_IMPROVED_ROW_FRACTION
            and all(advantage_passes)
        )
        budget_passes[str(budget)] = budget_pass
        budget_results[str(budget)] = {
            "baseline_mean_task_loss": float(baseline.mean().item()),
            "strategies": summaries,
            "jepa_increment_over_controls": advantages,
            "budget_gate_pass": budget_pass,
        }

    gate_pass = any(budget_passes.values()) and all(directional_points.values())
    decision = {
        "gate_pass": gate_pass,
        "scorer_consideration_unlocked": gate_pass,
        "decision": (
            "oracle-swap-supports-jepa-incremental-information"
            if gate_pass
            else "end-jepa-scoring-direction"
        ),
        "rule": (
            "At least one K must pass all absolute and matched-control criteria, "
            "and both K values must show positive point direction against both controls."
        ),
    }
    result = {
        "protocol_id": SWAP_ORACLE_PROTOCOL_ID,
        "artifact": str(args.artifact),
        "artifact_sha256": artifact_hash,
        "manifest_csv": str(args.manifest_csv),
        "manifest_sha256": manifest_hash,
        "row_count": SCIENTIFIC_ROWS,
        "video_count": len(set(expected_videos)),
        "row_keys_sha256": hashlib.sha256(
            "".join(f"{key}\n" for key in expected_keys).encode("utf-8")
        ).hexdigest(),
        "bootstrap": {
            "unit": "video_id",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "thresholds": {
            "min_relative_task_loss_reduction": MIN_RELATIVE_TASK_LOSS_REDUCTION,
            "min_improved_row_fraction": MIN_IMPROVED_ROW_FRACTION,
            "jepa_improvement_cluster_ci_lower_gt": 0.0,
            "jepa_minus_each_control_cluster_ci_lower_gt": 0.0,
        },
        "budgets": budget_results,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
