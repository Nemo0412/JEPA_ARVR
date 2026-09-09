#!/usr/bin/env python3
"""CPU logic tests for the B13 gating probe math (no model, no GPU).

Validates: (1) spearman on monotone/anti-monotone/uncorrelated inputs; (2) top-K
overlap; (3) the exact ridge accumulate->solve used in gating_probe_saliency.main
recovers a planted low-layer->saliency signal (held-out Spearman high) and a pure-noise
feature does NOT (held-out Spearman ~0) -- i.e. the probe discriminates.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.hdepic_lora_action_anticipation.gating_probe_saliency import (  # noqa: E402
    spearman, topk_overlap,
)


def test_spearman():
    x = np.arange(100, dtype=float)
    assert spearman(x, x) > 0.999, "monotone identical -> +1"
    assert spearman(x, -x) < -0.999, "anti-monotone -> -1"
    rng = np.random.default_rng(0)
    vals = [spearman(rng.standard_normal(200), rng.standard_normal(200)) for _ in range(50)]
    assert abs(np.mean(vals)) < 0.1, f"uncorrelated -> ~0, got {np.mean(vals):.3f}"
    # monotone nonlinear transform preserves rank corr
    assert spearman(x, np.exp(x / 20)) > 0.999, "monotone nonlinear -> +1"
    print("test_spearman OK")


def test_topk_overlap():
    true = np.arange(1000, dtype=float)  # top-K = the largest indices
    assert abs(topk_overlap(true, true, 100) - 1.0) < 1e-9, "identical -> 1.0"
    assert topk_overlap(-true, true, 100) < 1e-9, "reversed -> 0.0"
    rng = np.random.default_rng(1)
    ov = np.mean([topk_overlap(rng.standard_normal(1000), true, 100) for _ in range(200)])
    assert 0.05 < ov < 0.15, f"random top-100/1000 overlap ~0.1, got {ov:.3f}"
    print("test_topk_overlap OK")


def _fit_ridge(A, b, Dp, lam_frac):
    """Exact replica of gating_probe_saliency.main ridge solve."""
    diag_feat = A.diagonal()[:-1].mean().item()
    lam = lam_frac * max(diag_feat, 1e-6)
    reg = torch.eye(Dp, dtype=torch.float64) * lam
    reg[-1, -1] = 0.0
    return torch.linalg.solve(A + reg, b).numpy()


def _accumulate(samples, D):
    Dp = D + 1
    A = torch.zeros(Dp, Dp, dtype=torch.float64)
    b = torch.zeros(Dp, dtype=torch.float64)
    for F, s in samples:
        F = torch.from_numpy(F).double()
        F1 = torch.cat([F, torch.ones(F.shape[0], 1, dtype=torch.float64)], dim=1)
        A += F1.t() @ F1
        s = (s - s.mean()) / (s.std() + 1e-8)  # per-sample z-score, as in main
        b += F1.t() @ torch.from_numpy(s).double()
    return A, b, Dp


def test_ridge_recovers_signal():
    rng = np.random.default_rng(2)
    D = 16
    w_true = rng.standard_normal(D)
    def make(n):
        F = rng.standard_normal((n, D))
        s = F @ w_true + 0.1 * rng.standard_normal(n)  # planted linear signal
        return F, s
    train = [make(rng.integers(300, 600)) for _ in range(30)]
    test = [make(rng.integers(300, 600)) for _ in range(10)]

    A, b, Dp = _accumulate(train, D)
    w = _fit_ridge(A, b, Dp, 1e-2)
    sp = np.mean([spearman(F @ w[:-1] + w[-1], s) for F, s in test])
    assert sp > 0.9, f"signal probe held-out Spearman should be high, got {sp:.3f}"

    # noise features (uninformative) -> held-out Spearman ~0
    def make_noise(n):
        return rng.standard_normal((n, D)), rng.standard_normal(n)
    ntr = [make_noise(rng.integers(300, 600)) for _ in range(30)]
    nte = [make_noise(rng.integers(300, 600)) for _ in range(10)]
    An, bn, _ = _accumulate(ntr, D)
    wn = _fit_ridge(An, bn, Dp, 1e-2)
    spn = np.mean([spearman(F @ wn[:-1] + wn[-1], s) for F, s in nte])
    assert abs(spn) < 0.15, f"noise probe held-out Spearman should be ~0, got {spn:.3f}"
    print(f"test_ridge_recovers_signal OK (signal sp={sp:.3f}, noise sp={spn:.3f})")


if __name__ == "__main__":
    test_spearman()
    test_topk_overlap()
    test_ridge_recovers_signal()
    print("ALL GATING-PROBE MATH TESTS PASSED")
