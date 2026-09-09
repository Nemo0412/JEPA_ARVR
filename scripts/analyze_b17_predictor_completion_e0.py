#!/usr/bin/env python3
"""Fail-closed audit for B17 E0 predictor-compensated completion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from app.hdepic_lora_action_anticipation.evaluate_predictor_compensated_completion_e0 import (
    BRIDGE_TOKENS,
    JEPA_ARMS,
    LATENCY_REPEATS,
    LATENCY_ROWS,
    REFERENCE_BUDGET,
    RETAINED_BUDGET,
    SCIENTIFIC_ROWS,
    SCIENTIFIC_START,
    TEACHER_ARMS,
)
from app.hdepic_lora_action_anticipation.task_protected_jepa_pruning import (
    COMPLETION_E0_PROTOCOL_ID,
    sha256_file,
)
from scripts.analyze_b17_tpjepa_swap_oracle import cluster_bootstrap_mean_interval


BOOTSTRAP_SEED = 170317
MIN_TEACHER_RELATIVE_REDUCTION = 0.005
MIN_TEACHER_IMPROVED_ROW_FRACTION = 0.50
MIN_JEPA_TEACHER_GAP_RECOVERY = 0.25
MIN_E2E_LATENCY_REDUCTION = 0.05
PLACEBO_ARMS = (
    "mask_only_completion",
    "zero_completion",
    "mean_completion",
    "same_slot_pool_completion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("teacher", "final"), required=True)
    parser.add_argument("--teacher-artifact", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--jepa-artifact", type=Path)
    parser.add_argument("--expected-jepa-sha256")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_bound(path: Path, expected_hash: str, name: str) -> tuple[dict, str]:
    actual = sha256_file(path)
    if actual != expected_hash:
        raise RuntimeError(f"{name} SHA-256 mismatch: expected {expected_hash}, got {actual}")
    return torch.load(path, map_location="cpu", weights_only=False), actual


def _manifest_slice(path: Path) -> tuple[list[str], list[str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = rows[SCIENTIFIC_START : SCIENTIFIC_START + SCIENTIFIC_ROWS]
    if len(rows) != 2048 or len(selected) != SCIENTIFIC_ROWS:
        raise RuntimeError("manifest population does not match frozen 2048/[128:256] contract")
    row_keys = [f"{row['video_id']}\t{int(row['tick_frame'])}" for row in selected]
    videos = [row["video_id"] for row in selected]
    if len(set(videos)) != 22:
        raise RuntimeError("frozen E0 population must contain all 22 train-held-out videos")
    return row_keys, videos


def _validate_payload(payload: dict, *, stage: str, row_keys: list[str], videos: list[str]) -> None:
    if payload.get("protocol_id") != COMPLETION_E0_PROTOCOL_ID:
        raise RuntimeError(f"{stage} artifact protocol mismatch")
    if int(payload.get("diagnostic_schema_version", -1)) != 1:
        raise RuntimeError(f"{stage} artifact schema mismatch")
    if payload.get("stage") != stage:
        raise RuntimeError(f"expected {stage} artifact")
    if int(payload.get("shard_start", -1)) != SCIENTIFIC_START:
        raise RuntimeError("scientific artifact has wrong manifest offset")
    if int(payload.get("shard_count", -1)) != SCIENTIFIC_ROWS:
        raise RuntimeError("scientific artifact must contain exactly 128 rows")
    if payload.get("row_keys") != row_keys or payload.get("video_ids") != videos:
        raise RuntimeError(f"{stage} rows do not match frozen manifest slice")
    expected_arms = TEACHER_ARMS if stage == "teacher" else JEPA_ARMS
    if tuple(payload.get("arms", ())) != expected_arms:
        raise RuntimeError(f"{stage} arm contract mismatch")
    if int(payload.get("reference_budget", -1)) != REFERENCE_BUDGET:
        raise RuntimeError("reference budget mismatch")
    if int(payload.get("retained_budget", -1)) != RETAINED_BUDGET:
        raise RuntimeError("retained budget mismatch")
    if int(payload.get("bridge_tokens", -1)) != BRIDGE_TOKENS:
        raise RuntimeError("bridge count mismatch")
    if max(payload.get("native_l16_parity_max_abs", [float("inf")])) > 1e-5:
        raise RuntimeError("native/L16 no-delete parity failed")
    roundtrip = payload.get("reference_roundtrip_max_abs", [])
    if len(roundtrip) != SCIENTIFIC_ROWS or max(roundtrip, default=float("inf")) != 0.0:
        raise RuntimeError("untouched K4096 reference roundtrip parity failed")
    if int(payload.get("predictor_lora_modules_disabled", 0)) <= 0:
        raise RuntimeError("recovery/control predictor did not disable task predictor-LoRA")
    counts = payload.get("contract_counts", {})
    exact = {
        "reference_tokens": REFERENCE_BUDGET,
        "retained_tokens": RETAINED_BUDGET,
        "bridge_tokens": BRIDGE_TOKENS,
        "retained_is_reference_attention_prefix": True,
        "retained_bridge_disjoint": True,
    }
    for key, expected in exact.items():
        values = counts.get(key, [])
        if len(values) != SCIENTIFIC_ROWS or any(value != expected for value in values):
            raise RuntimeError(f"contract field {key} failed")
    for arm in expected_arms:
        arm_metrics = payload.get("metrics", {}).get(arm, {})
        for key in ("task_loss", "action_loss", "action_margin"):
            values = arm_metrics.get(key)
            if not isinstance(values, torch.Tensor) or values.shape != (SCIENTIFIC_ROWS,):
                raise RuntimeError(f"{stage}/{arm}/{key} shape mismatch")
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError(f"{stage}/{arm}/{key} contains non-finite values")
        for key in ("action_top1_correct", "action_top5_correct"):
            values = arm_metrics.get(key)
            if not isinstance(values, torch.Tensor) or values.shape != (SCIENTIFIC_ROWS, 3):
                raise RuntimeError(f"{stage}/{arm}/{key} shape mismatch")


def _paired_summary(
    baseline_loss: torch.Tensor,
    arm_loss: torch.Tensor,
    videos: list[str],
    *,
    seed: int,
) -> tuple[dict[str, object], torch.Tensor]:
    improvement = baseline_loss.double() - arm_loss.double()
    interval = cluster_bootstrap_mean_interval(improvement, videos, seed=seed)
    return (
        {
            "mean_task_loss": float(arm_loss.double().mean().item()),
            "mean_improvement_over_k3072": float(improvement.mean().item()),
            "relative_reduction_over_k3072": float(
                improvement.mean().item() / baseline_loss.double().mean().item()
            ),
            "cluster_bootstrap_95ci": list(interval),
            "improved_row_fraction": float((improvement > 1e-8).double().mean().item()),
        },
        improvement,
    )


def _teacher_result(payload: dict, videos: list[str]) -> tuple[dict[str, object], bool]:
    metrics = payload["metrics"]
    baseline = metrics["retained_k3072"]["task_loss"].double()
    reference = metrics["reference_k4096"]["task_loss"].double()
    arms: dict[str, object] = {
        "reference_k4096": {
            "mean_task_loss": float(reference.mean().item()),
            "mean_improvement_over_k3072": float((baseline - reference).mean().item()),
        },
        "retained_k3072": {"mean_task_loss": float(baseline.mean().item())},
    }
    improvements: dict[str, torch.Tensor] = {}
    for index, arm in enumerate(("teacher_completion",) + PLACEBO_ARMS):
        summary, improvement = _paired_summary(
            baseline, metrics[arm]["task_loss"], videos, seed=BOOTSTRAP_SEED + index
        )
        arms[arm] = summary
        improvements[arm] = improvement
    teacher = arms["teacher_completion"]
    teacher_gate = bool(
        teacher["relative_reduction_over_k3072"] >= MIN_TEACHER_RELATIVE_REDUCTION
        and teacher["cluster_bootstrap_95ci"][0] > 0.0
        and teacher["improved_row_fraction"] >= MIN_TEACHER_IMPROVED_ROW_FRACTION
    )
    latency = payload["latency_contract"]["arm_latency_ms"]
    if any(len(latency[arm]) != LATENCY_ROWS for arm in ("reference_k4096", "retained_k3072")):
        raise RuntimeError("teacher artifact latency population mismatch")
    ref_latency = torch.tensor(latency["reference_k4096"], dtype=torch.float64)
    retained_latency = torch.tensor(latency["retained_k3072"], dtype=torch.float64)
    mac_contract = payload.get("dense_mac_contract", {})
    route_macs = mac_contract.get("routes", {})
    for arm in TEACHER_ARMS:
        if arm not in route_macs or int(route_macs[arm]) <= 0:
            raise RuntimeError(f"missing positive dense-MAC accounting for {arm}")
    reference_macs = int(route_macs["reference_k4096"])
    retained_macs = int(route_macs["retained_k3072"])
    teacher_macs = int(route_macs["teacher_completion"])
    result = {
        "arms": arms,
        "teacher_gate_pass": teacher_gate,
        "frozen_jepa_stage_authorized": teacher_gate,
        "teacher_gate_rule": {
            "min_relative_task_loss_reduction": MIN_TEACHER_RELATIVE_REDUCTION,
            "cluster_bootstrap_95ci_lower_gt": 0.0,
            "min_improved_row_fraction": MIN_TEACHER_IMPROVED_ROW_FRACTION,
        },
        "latency_reference_vs_uncompensated": {
            "paired_rows": LATENCY_ROWS,
            "repeats_per_row_arm": LATENCY_REPEATS,
            "reference_median_ms": float(ref_latency.median().item()),
            "retained_median_ms": float(retained_latency.median().item()),
            "relative_reduction": float(
                (ref_latency.median() - retained_latency.median()).item()
                / ref_latency.median().item()
            ),
        },
        "dense_mac_accounting": {
            "unit": mac_contract.get("unit"),
            "scope": mac_contract.get("scope"),
            "reference_k4096": reference_macs,
            "retained_k3072": retained_macs,
            "teacher_completion_inference_route": teacher_macs,
            "retained_relative_reduction_vs_reference": (
                reference_macs - retained_macs
            )
            / reference_macs,
            "teacher_completion_relative_reduction_vs_reference": (
                reference_macs - teacher_macs
            )
            / reference_macs,
            "all_teacher_stage_routes": {
                arm: int(route_macs[arm]) for arm in TEACHER_ARMS
            },
        },
    }
    return result, teacher_gate


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest_hash = sha256_file(args.manifest_csv)
    if manifest_hash != args.expected_manifest_sha256:
        raise RuntimeError("manifest SHA-256 mismatch")
    row_keys, videos = _manifest_slice(args.manifest_csv)
    teacher, teacher_hash = _load_bound(
        args.teacher_artifact, args.expected_teacher_sha256, "teacher artifact"
    )
    _validate_payload(teacher, stage="teacher", row_keys=row_keys, videos=videos)
    teacher_result, teacher_gate = _teacher_result(teacher, videos)
    result: dict[str, object] = {
        "protocol_id": COMPLETION_E0_PROTOCOL_ID,
        "audit_mode": args.mode,
        "manifest_csv": str(args.manifest_csv),
        "manifest_sha256": manifest_hash,
        "population": {
            "slice": [SCIENTIFIC_START, SCIENTIFIC_START + SCIENTIFIC_ROWS],
            "rows": SCIENTIFIC_ROWS,
            "video_clusters": len(set(videos)),
        },
        "teacher_artifact": str(args.teacher_artifact),
        "teacher_artifact_sha256": teacher_hash,
        "teacher_stage": teacher_result,
    }
    if args.mode == "teacher":
        result["decision"] = (
            "run-frozen-jepa-completion" if teacher_gate else "close-e0-no-recoverable-teacher-gap"
        )
        result["e0_complete"] = not teacher_gate
    else:
        if not teacher_gate:
            raise RuntimeError("final JEPA audit is forbidden because teacher gate did not pass")
        if args.jepa_artifact is None or not args.expected_jepa_sha256:
            raise ValueError("final audit requires a bound JEPA artifact and SHA-256")
        jepa, jepa_hash = _load_bound(
            args.jepa_artifact, args.expected_jepa_sha256, "JEPA artifact"
        )
        _validate_payload(jepa, stage="jepa", row_keys=row_keys, videos=videos)
        for arm in ("reference_k4096", "retained_k3072"):
            delta = (
                teacher["metrics"][arm]["task_loss"].double()
                - jepa["metrics"][arm]["task_loss"].double()
            ).abs().max()
            if float(delta.item()) > 1e-4:
                raise RuntimeError(f"teacher/JEPA rerun parity failed for {arm}: {delta.item()}")

        baseline = teacher["metrics"]["retained_k3072"]["task_loss"].double()
        teacher_improvement = baseline - teacher["metrics"]["teacher_completion"]["task_loss"].double()
        jepa_summary, jepa_improvement = _paired_summary(
            baseline,
            jepa["metrics"]["jepa_completion"]["task_loss"],
            videos,
            seed=BOOTSTRAP_SEED + 100,
        )
        recovery_fraction = float(
            jepa_improvement.mean().item() / teacher_improvement.mean().item()
        )
        placebo_advantages: dict[str, object] = {}
        placebo_passes = []
        for index, arm in enumerate(PLACEBO_ARMS):
            placebo_improvement = baseline - teacher["metrics"][arm]["task_loss"].double()
            advantage = jepa_improvement - placebo_improvement
            interval = cluster_bootstrap_mean_interval(
                advantage, videos, seed=BOOTSTRAP_SEED + 200 + index
            )
            placebo_advantages[arm] = {
                "mean_incremental_improvement": float(advantage.mean().item()),
                "cluster_bootstrap_95ci": list(interval),
            }
            placebo_passes.append(interval[0] > 0.0)
        representation_gate = bool(
            jepa_summary["cluster_bootstrap_95ci"][0] > 0.0
            and recovery_fraction >= MIN_JEPA_TEACHER_GAP_RECOVERY
            and all(placebo_passes)
        )

        latency = jepa["latency_contract"]["arm_latency_ms"]
        if any(len(latency[arm]) != LATENCY_ROWS for arm in ("reference_k4096", "jepa_completion")):
            raise RuntimeError("JEPA latency population mismatch")
        reference_latency = torch.tensor(latency["reference_k4096"], dtype=torch.float64)
        jepa_latency = torch.tensor(latency["jepa_completion"], dtype=torch.float64)
        latency_reduction = float(
            (reference_latency.median() - jepa_latency.median()).item()
            / reference_latency.median().item()
        )
        macs = jepa["dense_mac_contract"]
        reference_macs = int(macs["routes"]["reference_k4096"])
        jepa_macs = int(macs["routes"]["jepa_completion"])
        mac_reduction = (reference_macs - jepa_macs) / reference_macs
        efficiency_gate = bool(
            mac_reduction > 0.0 and latency_reduction >= MIN_E2E_LATENCY_REDUCTION
        )
        e0_pass = representation_gate and efficiency_gate
        result.update(
            {
                "jepa_artifact": str(args.jepa_artifact),
                "jepa_artifact_sha256": jepa_hash,
                "jepa_completion": {
                    **jepa_summary,
                    "teacher_gap_recovery_fraction": recovery_fraction,
                    "increment_over_placebos": placebo_advantages,
                    "representation_gate_pass": representation_gate,
                    "representation_gate_rule": {
                        "improvement_cluster_95ci_lower_gt": 0.0,
                        "min_teacher_gap_recovery_fraction": MIN_JEPA_TEACHER_GAP_RECOVERY,
                        "increment_over_every_placebo_cluster_95ci_lower_gt": 0.0,
                    },
                },
                "cost": {
                    "paired_latency_rows": LATENCY_ROWS,
                    "repeats_per_row_arm": LATENCY_REPEATS,
                    "reference_median_ms": float(reference_latency.median().item()),
                    "jepa_median_ms": float(jepa_latency.median().item()),
                    "e2e_latency_relative_reduction": latency_reduction,
                    "reference_post_l16_dense_macs": reference_macs,
                    "jepa_post_l16_dense_macs": jepa_macs,
                    "post_l16_dense_mac_relative_reduction": mac_reduction,
                    "efficiency_gate_pass": efficiency_gate,
                    "min_e2e_latency_relative_reduction": MIN_E2E_LATENCY_REDUCTION,
                },
                "e0_pass": e0_pass,
                "e0_complete": True,
                "decision": (
                    "advance-to-compact-residual-set-memory"
                    if e0_pass
                    else (
                        "retain-as-representation-analysis-no-efficiency-claim"
                        if representation_gate
                        else "close-frozen-jepa-completion"
                    )
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
