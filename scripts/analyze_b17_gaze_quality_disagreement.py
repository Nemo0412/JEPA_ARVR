#!/usr/bin/env python3
"""Post-hoc gaze-quality/disagreement analysis for B17 v2 frozen results.

This analysis never changes or re-selects a pruning method. It joins the fixed
v2 +2s paired-correctness rows to their causal observed gaze samples, computes
predeclared quality/dynamics features, and reports descriptive video/session-
cluster bootstrap intervals for hypothesis generation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.hdepic_lora_action_anticipation.egtea_gaze import parse_gtea_gaze


PROTOCOL_ID = "egtea-stream-mtp-pruning/gaze-quality-disagreement-posthoc-v1"
EXPECTED_VAL_SHA256 = "d02af8297b1f845a644fd617642ca612fb553f8fa3300f9ae3903227a9dd6393"
EXPECTED_PAIRED_SHA256 = "d4e66d4e3377c996bc438827a69e6c9117b5f9e0a9ab087a57a3002f18234bd7"
VARIANTS = ("uniform", "calib", "attention", "gaze", "gaze_shift")
COMPARISONS = ("attention", "gaze_shift", "calib", "uniform")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in str(value).split(",") if item.strip()]


def gaze_features(frame_indices: np.ndarray, gaze: np.ndarray) -> dict[str, float | int]:
    """Compute causal quality/dynamics features on the exact observed frames."""
    if frame_indices.ndim != 1 or frame_indices.size == 0:
        raise ValueError("frame_indices must be a non-empty vector")
    in_range = (frame_indices >= 0) & (frame_indices < len(gaze))
    safe = np.clip(frame_indices, 0, max(0, len(gaze) - 1))
    types = np.rint(gaze[safe, 2]).astype(np.int64) if len(gaze) else np.zeros_like(frame_indices)
    xy = gaze[safe, :2].astype(np.float64) if len(gaze) else np.zeros((len(frame_indices), 2))
    finite = np.isfinite(xy).all(axis=1)
    valid = in_range & finite & np.isin(types, (1, 2))
    fixation = valid & (types == 1)
    saccade = valid & (types == 2)

    def window_stats(indices: np.ndarray) -> tuple[float, float, float, int]:
        win_valid = valid[indices]
        count = int(win_valid.sum())
        valid_fraction = count / max(1, len(indices))
        fixation_share = float(fixation[indices].sum() / count) if count else float("nan")
        pair_valid = win_valid[1:] & win_valid[:-1]
        if bool(pair_valid.any()):
            steps = np.linalg.norm(xy[indices][1:] - xy[indices][:-1], axis=1)[pair_valid]
            median_step = float(np.median(steps))
        else:
            median_step = float("nan")
        return valid_fraction, fixation_share, median_step, count

    all_indices = np.arange(len(frame_indices))
    recent_indices = np.arange(max(0, len(frame_indices) - 4), len(frame_indices))
    valid_fraction, fixation_share, median_step, valid_count = window_stats(all_indices)
    recent_valid_fraction, recent_fixation_share, recent_median_step, _ = window_stats(recent_indices)

    if valid_count:
        valid_xy = xy[valid]
        center = np.median(valid_xy, axis=0)
        dispersion = float(np.sqrt(np.mean(np.sum((valid_xy - center) ** 2, axis=1))))
    else:
        dispersion = float("nan")
    landing_count = int(((types[:-1] == 2) & (types[1:] == 1) & valid[:-1] & valid[1:]).sum())

    return {
        "valid_fraction": float(valid_fraction),
        "fixation_share_valid": float(fixation_share),
        "saccade_share_valid": float(1.0 - fixation_share) if math.isfinite(fixation_share) else float("nan"),
        "median_step_norm": float(median_step),
        "dispersion_rms_norm": float(dispersion),
        "recent_valid_fraction": float(recent_valid_fraction),
        "recent_fixation_share_valid": float(recent_fixation_share),
        "recent_median_step_norm": float(recent_median_step),
        "landing_count": landing_count,
        "valid_count": valid_count,
        "fixation_count": int(fixation.sum()),
        "saccade_count": int(saccade.sum()),
    }


def cluster_paired_summary(
    a: np.ndarray,
    b: np.ndarray,
    clusters: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.shape != a.shape:
        raise ValueError("stratum mask must match correctness shape")
    aa, bb, cc = a[mask].astype(np.float64), b[mask].astype(np.float64), clusters[mask]
    if aa.size == 0:
        return {"n": 0, "clusters": 0, "status": "empty stratum"}
    diff = aa - bb
    unique = np.unique(cc)
    sums = np.asarray([diff[cc == cluster].sum() for cluster in unique], dtype=np.float64)
    counts = np.asarray([(cc == cluster).sum() for cluster in unique], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 256):
        size = min(256, bootstrap_samples - start)
        sampled = rng.integers(0, len(unique), size=(size, len(unique)))
        boot[start : start + size] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    low, high = np.percentile(boot, (2.5, 97.5)) * 100.0
    a_only = int(((aa == 1) & (bb == 0)).sum())
    b_only = int(((aa == 0) & (bb == 1)).sum())
    return {
        "n": int(diff.size),
        "clusters": int(len(unique)),
        "delta_pp": float(diff.mean() * 100.0),
        "cluster_bootstrap_ci95_pp": [float(low), float(high)],
        "gaze_correct_control_wrong": a_only,
        "gaze_wrong_control_correct": b_only,
    }


def finite_summary(values: np.ndarray, mask: np.ndarray) -> dict:
    selected = values[mask & np.isfinite(values)]
    if selected.size == 0:
        return {"n_finite": 0}
    return {
        "n_finite": int(selected.size),
        "mean": float(selected.mean()),
        "median": float(np.median(selected)),
        "q25": float(np.quantile(selected, 0.25)),
        "q75": float(np.quantile(selected, 0.75)),
    }


def build_strata(features: dict[str, np.ndarray], effective: np.ndarray) -> dict[str, np.ndarray]:
    valid = features["valid_fraction"]
    fixation = features["fixation_share_valid"]
    step = features["median_step_norm"]
    dispersion = features["dispersion_rms_norm"]
    recent_valid = features["recent_valid_fraction"]
    recent_fixation = features["recent_fixation_share_valid"]
    recent_step = features["recent_median_step_norm"]
    context = features["context_sec"]
    landing = features["landing_count"]
    finite_fix = np.isfinite(fixation)
    finite_step = np.isfinite(step)
    finite_disp = np.isfinite(dispersion)
    finite_recent = np.isfinite(recent_fixation) & np.isfinite(recent_step)

    strata: dict[str, np.ndarray] = {
        "all_rows": np.ones_like(effective, dtype=np.bool_),
        "effective_gaze_rows": effective,
    }
    for seconds in (4, 6, 8, 10):
        strata[f"context_{seconds}s"] = context == float(seconds)
    strata.update(
        {
            "valid_low_lt50": valid < 0.50,
            "valid_mid_50to90": (valid >= 0.50) & (valid < 0.90),
            "valid_high_ge90": valid >= 0.90,
            "fixation_low_lt50": finite_fix & (fixation < 0.50),
            "fixation_mid_50to80": finite_fix & (fixation >= 0.50) & (fixation < 0.80),
            "fixation_high_ge80": finite_fix & (fixation >= 0.80),
            "motion_stable_le1patch": finite_step & (step <= 1.0 / 16.0),
            "motion_moderate_1to2patch": finite_step & (step > 1.0 / 16.0) & (step <= 2.0 / 16.0),
            "motion_dynamic_gt2patch": finite_step & (step > 2.0 / 16.0),
            "dispersion_compact_le1patch": finite_disp & (dispersion <= 1.0 / 16.0),
            "dispersion_mid_1to2patch": finite_disp & (dispersion > 1.0 / 16.0) & (dispersion <= 2.0 / 16.0),
            "dispersion_broad_gt2patch": finite_disp & (dispersion > 2.0 / 16.0),
            "has_saccade_to_fixation_landing": landing > 0,
            "no_sampled_landing": landing == 0,
            "reliable_stable_fixation": effective & (valid >= 0.75) & finite_fix & (fixation >= 0.80)
            & finite_step & (step <= 2.0 / 16.0),
            "recent_reliable_stable_fixation": effective & (recent_valid >= 0.75) & finite_recent
            & (recent_fixation >= 0.75) & (recent_step <= 2.0 / 16.0),
        }
    )
    return strata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--gaze-root", type=Path, required=True)
    parser.add_argument("--paired-npz", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--code-snapshot-sha256", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    feature_path = args.out_json.with_suffix(".features.npz")
    existing = [path for path in (args.out_json, feature_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")
    if sha256_file(args.val_csv) != EXPECTED_VAL_SHA256:
        raise RuntimeError("validation CSV does not match registered SHA-256")
    if sha256_file(args.paired_npz) != EXPECTED_PAIRED_SHA256:
        raise RuntimeError("paired v2 artifact does not match registered SHA-256")

    with args.val_csv.open() as handle:
        csv_rows = list(csv.DictReader(handle))
    row_map: dict[str, dict] = {}
    for row in csv_rows:
        key = f"{row['video_id']}|{int(row['tick_frame'])}"
        if key in row_map:
            raise RuntimeError(f"duplicate validation row key: {key}")
        row_map[key] = row

    paired = np.load(args.paired_npz)
    row_keys = paired["row_key__2s"].astype(str)
    clusters = paired["cluster__2s"].astype(str)
    effective = paired["effective_gaze__2s"].astype(np.bool_)
    correctness = {variant: paired[f"{variant}__2s"].astype(np.int8) for variant in VARIANTS}
    expected_shape = (22741,)
    arrays = [row_keys, clusters, effective, *correctness.values()]
    if any(array.shape != expected_shape for array in arrays):
        raise RuntimeError(f"registered +2s population requires shape {expected_shape}")
    if len(set(row_keys.tolist())) != len(row_keys) or set(row_keys.tolist()) != set(row_map):
        raise RuntimeError("paired row keys do not bijectively match validation CSV rows")

    gaze_cache: dict[str, np.ndarray] = {}
    gaze_hashes: dict[str, str] = {}
    records: list[dict] = []
    for index, key in enumerate(row_keys.tolist()):
        row = row_map[key]
        video_id = str(row["video_id"])
        if video_id not in gaze_cache:
            gaze_path = args.gaze_root / f"{video_id}.txt"
            if not gaze_path.is_file():
                raise FileNotFoundError(gaze_path)
            gaze_hashes[video_id] = sha256_file(gaze_path)
            gaze_cache[video_id] = parse_gtea_gaze(gaze_path)
        frame_indices = np.asarray(parse_int_list(row["frame_indices"]), dtype=np.int64)
        if frame_indices[-1] >= int(row["tick_frame"]):
            raise RuntimeError(f"causality violation for {key}")
        record = gaze_features(frame_indices, gaze_cache[video_id])
        record.update(
            {
                "context_sec": float(row["context_sec"]),
                "tick_frame": int(row["tick_frame"]),
                "primary_verb": parse_int_list(row["mtp_verbs"])[0],
                "primary_noun": parse_int_list(row["mtp_nouns"])[0],
                "row_index": index,
            }
        )
        records.append(record)

    feature_names = tuple(records[0].keys())
    features = {
        name: np.asarray([record[name] for record in records], dtype=np.float64)
        for name in feature_names
    }
    strata = build_strata(features, effective)
    analysis = {}
    for stratum_index, (name, mask) in enumerate(strata.items()):
        comparisons = {
            f"gaze_minus_{control}": cluster_paired_summary(
                correctness["gaze"], correctness[control], clusters, mask,
                seed=args.seed + stratum_index, bootstrap_samples=args.bootstrap_samples,
            )
            for control in COMPARISONS
        }
        analysis[name] = {
            "n": int(mask.sum()),
            "clusters": int(len(np.unique(clusters[mask]))) if bool(mask.any()) else 0,
            "quality_summary": {
                feature: finite_summary(features[feature], mask)
                for feature in (
                    "valid_fraction", "fixation_share_valid", "median_step_norm",
                    "dispersion_rms_norm", "recent_valid_fraction",
                    "recent_fixation_share_valid", "recent_median_step_norm", "landing_count",
                )
            },
            "paired": comparisons,
        }

    discordance_profiles = {}
    for control in COMPARISONS:
        gaze_correct = correctness["gaze"] == 1
        control_correct = correctness[control] == 1
        categories = {
            "gaze_only": gaze_correct & ~control_correct,
            "control_only": ~gaze_correct & control_correct,
            "both_correct": gaze_correct & control_correct,
            "both_wrong": ~gaze_correct & ~control_correct,
        }
        discordance_profiles[f"gaze_vs_{control}"] = {
            category: {
                "n": int(mask.sum()),
                "quality_summary": {
                    feature: finite_summary(features[feature], mask)
                    for feature in (
                        "valid_fraction", "fixation_share_valid", "median_step_norm",
                        "dispersion_rms_norm", "recent_fixation_share_valid", "landing_count",
                    )
                },
            }
            for category, mask in categories.items()
        }

    gaze_manifest = hashlib.sha256()
    for video_id, digest in sorted(gaze_hashes.items()):
        gaze_manifest.update(f"{video_id}\t{digest}\n".encode())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        row_key=row_keys,
        cluster=clusters,
        effective_gaze=effective,
        **{f"feature__{name}": values for name, values in features.items()},
        **{f"correct__{name}": values for name, values in correctness.items()},
    )
    report = {
        "branch": "B17",
        "group": "B17-gaze-quality-disagreement-analysis",
        "evaluation_protocol": PROTOCOL_ID,
        "analysis_kind": "post-hoc descriptive hypothesis generation",
        "selection_boundary": (
            "fixed v2 full outputs only; strata cannot promote a method or support a superiority claim"
        ),
        "population": {
            "horizon": "+2s",
            "rows": int(len(row_keys)),
            "clusters": int(len(np.unique(clusters))),
            "effective_gaze_rows": int(effective.sum()),
        },
        "fixed_thresholds": {
            "valid_fraction": [0.50, 0.90],
            "fixation_share_valid": [0.50, 0.80],
            "motion_and_dispersion_patch_units": [1.0, 2.0],
            "patch_width_normalized": 1.0 / 16.0,
            "reliable_stable": "effective & valid>=0.75 & fixation_share>=0.80 & median_step<=2 patches",
            "recent_window": "last four 8-FPS observations (0.5 s)",
        },
        "strata": analysis,
        "discordance_profiles": discordance_profiles,
        "feature_artifact": str(feature_path),
        "provenance": {
            "job_id": args.job_id,
            "code_root": str(args.code_root.resolve()),
            "git_commit": args.code_commit,
            "code_snapshot_sha256": args.code_snapshot_sha256,
            "val_csv_sha256": EXPECTED_VAL_SHA256,
            "paired_npz_sha256": EXPECTED_PAIRED_SHA256,
            "gaze_files": len(gaze_hashes),
            "gaze_manifest_sha256": gaze_manifest.hexdigest(),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
    }
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
