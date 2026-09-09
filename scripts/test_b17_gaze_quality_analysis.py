#!/usr/bin/env python3
"""Semantic contracts for B17 post-hoc gaze-quality analysis."""
from __future__ import annotations

import numpy as np

from scripts.analyze_b17_gaze_quality_disagreement import (
    build_strata,
    cluster_paired_summary,
    gaze_features,
)


def test_gaze_feature_semantics() -> None:
    gaze = np.zeros((8, 3), dtype=np.float32)
    gaze[1] = [0.10, 0.10, 2]  # saccade
    gaze[2] = [0.12, 0.10, 1]  # landing into fixation
    gaze[3] = [0.13, 0.11, 1]
    gaze[4] = [0.90, 0.90, 3]  # Blink/unknown: invalid
    gaze[5] = [0.20, 0.20, 2]
    features = gaze_features(np.asarray([1, 2, 3, 4, 5]), gaze)
    assert features["valid_count"] == 4
    assert features["fixation_count"] == 2
    assert features["saccade_count"] == 2
    assert abs(features["valid_fraction"] - 0.8) < 1e-12
    assert abs(features["fixation_share_valid"] - 0.5) < 1e-12
    assert features["landing_count"] == 1
    assert np.isfinite(features["median_step_norm"])
    assert np.isfinite(features["dispersion_rms_norm"])


def test_cluster_bootstrap_and_strata() -> None:
    gaze_correct = np.asarray([1, 1, 0, 0, 1, 0], dtype=np.int8)
    control_correct = np.asarray([0, 1, 1, 0, 0, 0], dtype=np.int8)
    clusters = np.asarray(["a", "a", "b", "b", "c", "c"])
    mask = np.asarray([True, True, True, True, False, False])
    summary = cluster_paired_summary(
        gaze_correct, control_correct, clusters, mask, seed=17, bootstrap_samples=200
    )
    assert summary["n"] == 4
    assert summary["clusters"] == 2
    assert summary["delta_pp"] == 0.0
    assert summary["gaze_correct_control_wrong"] == 1
    assert summary["gaze_wrong_control_correct"] == 1

    features = {
        "valid_fraction": np.asarray([1.0, 0.8, 0.4]),
        "fixation_share_valid": np.asarray([0.9, 0.6, np.nan]),
        "median_step_norm": np.asarray([0.05, 0.10, np.nan]),
        "dispersion_rms_norm": np.asarray([0.03, 0.08, np.nan]),
        "recent_valid_fraction": np.asarray([1.0, 0.75, 0.0]),
        "recent_fixation_share_valid": np.asarray([1.0, 0.75, np.nan]),
        "recent_median_step_norm": np.asarray([0.02, 0.12, np.nan]),
        "context_sec": np.asarray([10.0, 8.0, 4.0]),
        "landing_count": np.asarray([1.0, 0.0, 0.0]),
    }
    strata = build_strata(features, np.asarray([True, True, False]))
    assert strata["reliable_stable_fixation"].tolist() == [True, False, False]
    assert strata["recent_reliable_stable_fixation"].tolist() == [True, True, False]
    assert strata["context_4s"].tolist() == [False, False, True]
    assert strata["valid_low_lt50"].tolist() == [False, False, True]


def main() -> None:
    test_gaze_feature_semantics()
    test_cluster_bootstrap_and_strata()
    print("B17 gaze-quality analysis contracts: PASS")


if __name__ == "__main__":
    main()
