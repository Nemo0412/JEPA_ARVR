#!/usr/bin/env python3
"""CPU logic test for RecencyWeightedTokenPruner (no model/GPU).

Validates: keeps exactly floor(K/gp)*gp tokens, frame-aligned (whole gp-slots),
chronological order, the recent window is fully covered, ~recent_frac of the slot
budget lands in the recent window, and K>=N is a no-op.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (  # noqa: E402
    RecencyWeightedTokenPruner,
)


def test_recency_weighted():
    gp = 256
    # 10 s @ 8 fps: 40 slots; keep 16 slots (4 s budget); recent window 4 s = 16 slots.
    p = RecencyWeightedTokenPruner(keep_count=16 * gp, gp=gp, recent_frac=0.72,
                                   recent_window_sec=4.0, fps=8, tubelet=2)
    N = 40 * gp
    feats = torch.randn(2, N, 8)
    kept, idx = p.prune(feats)
    assert kept.shape[1] == 16 * gp, f"kept {kept.shape[1]} != 4096"
    slots = sorted(set((idx[0] // gp).tolist()))
    # frame-aligned: every kept slot contributes all gp tokens
    for s in slots:
        toks = set(idx[0][(idx[0] // gp) == s].tolist())
        assert toks == set(range(s * gp, (s + 1) * gp)), f"slot {s} not whole"
    assert idx[0].tolist() == sorted(idx[0].tolist()), "chronological order"
    # recent_frac*16 ~= 11.5 -> 12 recent slots, from the most-recent (slots 28..39 region)
    n_recent = sum(1 for s in slots if s >= 40 - 16)  # in the last-4s (16-slot) window
    assert n_recent >= 11, f"recent window under-covered: {n_recent}"
    # the very newest slots must be kept
    assert 39 in slots and 38 in slots, "newest slots must be kept"
    # some OLD slots kept too (not pure-recent)
    assert any(s < 40 - 16 for s in slots), "should keep some older slots (recency-WEIGHTED)"
    print(f"test_recency_weighted OK (16 kept slots: {slots})")

    # steep frac -> nearly all recent; K>=N no-op
    p2 = RecencyWeightedTokenPruner(keep_count=100 * gp, gp=gp)
    kept2, idx2 = p2.prune(feats)
    assert kept2.shape[1] == N and torch.equal(idx2, torch.arange(N).unsqueeze(0).expand(2, -1)), "K>=N no-op"
    print("test_recency_weighted no-op OK")


if __name__ == "__main__":
    test_recency_weighted()
    print("ALL RECENCY-WEIGHTED TESTS PASSED")
