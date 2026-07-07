#!/usr/bin/env python
"""Smoke test for cascade-aware loss-aware pruning."""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from loss_aware_pruning import (
    LossAwarePruningConfig,
    allocate_cascade_prune_ratios,
    cascade_final_keep_ratio,
    compute_token_importance,
    conditional_keep_count,
    finalize_calibration_config,
    lookup_calibrated_scores,
    save_calibration_artifacts,
    simulate_cascade_keep_counts,
    topk_token_prune,
    verify_cascade_global_ratio,
)
from models.vit_encoder_pruning import make_single_layer_config


def main():
    z = torch.randn(2, 256, 64, requires_grad=True)
    grad = torch.randn_like(z)
    score = compute_token_importance(z, grad)
    assert score.shape == (2, 256)

    z2, idx = topk_token_prune(z, score, conditional_prune_ratio=0.5, gp=16)
    assert z2.shape[1] == 128

    ratios = allocate_cascade_prune_ratios({8: 1.0, 12: 2.0, 16: 0.5}, 0.5)
    verify_cascade_global_ratio(ratios, 0.5)
    assert math.isclose(cascade_final_keep_ratio(ratios), 0.5, rel_tol=1e-5)

    n = 4096
    for layer in sorted(ratios):
        n = conditional_keep_count(n, ratios[layer], gp=256)
    cascade = simulate_cascade_keep_counts(4096, ratios, gp=256)
    assert cascade[max(cascade)] == n

    mean_scores = {8: torch.randn(4096), 12: torch.randn(4096), 16: torch.randn(4096)}
    token_pos = torch.arange(128).unsqueeze(0).expand(2, -1)
    looked = lookup_calibrated_scores(8, token_pos, mean_scores)
    assert looked.shape == (2, 128)

    cfg = make_single_layer_config(12, 0.5)
    cfg2 = finalize_calibration_config(cfg, 4096, {12: mean_scores[8][:4096]}, {12: 1.5}, gp=256)

    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "lap.json"
        cfg2 = save_calibration_artifacts(
            cfg2,
            {12: mean_scores[8][:4096]},
            {12: 1.5},
            cfg_path,
        )
        loaded = LossAwarePruningConfig.load(cfg_path)
        scores = loaded.load_calibrated_scores("cpu")
        assert 12 in scores
        assert scores[12].shape[0] == 4096

    print("[smoke] cascade loss-aware pruning ok")


if __name__ == "__main__":
    main()
