#!/usr/bin/env python3
"""CPU logic tests for the B13 learned-early-pruner scorer module (no encoder/GPU).

Validates: (1) MidLayerTokenScorer forward shapes + save/load round-trip; (2)
LearnedMidLayerEncoderPruner._prune keeps exactly floor(K/gp)*gp tokens, preserves
chronological order, carries TRUE token positions, and is a no-op when K>=N.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.hdepic_lora_action_anticipation.midlayer_scorer import (  # noqa: E402
    MidLayerTokenScorer, LearnedMidLayerEncoderPruner,
)


def test_scorer_shapes_and_roundtrip(tmp=Path("/tmp/b13_scorer_test.pt")):
    D = 64
    s = MidLayerTokenScorer(D, hidden=32)
    x = torch.randn(2, 100, D)
    out = s(x)
    assert out.shape == (2, 100), f"scorer out shape {out.shape}"
    assert s.embed_dim == D
    torch.save({"state_dict": s.state_dict(), "embed_dim": D, "hidden": 32}, tmp)
    ck = torch.load(tmp, weights_only=False)
    s2 = MidLayerTokenScorer(ck["embed_dim"], hidden=ck["hidden"])
    s2.load_state_dict(ck["state_dict"])
    assert torch.allclose(s(x), s2(x)), "round-trip mismatch"
    tmp.unlink()
    print("test_scorer_shapes_and_roundtrip OK")


def _bare_pruner(keep_count, gp):
    p = object.__new__(LearnedMidLayerEncoderPruner)
    p.gp = gp
    p.keep_count = max(gp, (keep_count // gp) * gp)
    return p


def test_prune_math():
    gp = 4
    p = _bare_pruner(keep_count=8, gp=gp)  # keep 8 (2 slots)
    B, N, D = 1, 16, 3
    x = torch.arange(N, dtype=torch.float32).view(1, N, 1).expand(B, N, D).contiguous()
    token_pos = torch.arange(N).view(1, N)
    # importance: make the LAST 8 tokens most important (indices 8..15)
    imp = torch.arange(N, dtype=torch.float32).view(1, N)
    xk, posk = p._prune(x, token_pos, imp)
    assert xk.shape[1] == 8, f"kept {xk.shape[1]} != 8"
    kept = posk[0].tolist()
    assert kept == sorted(kept), "kept positions must stay in chronological order"
    assert kept == list(range(8, 16)), f"should keep the 8 most-important (8..15), got {kept}"
    # feature carried matches the true index (x[i]==i here)
    assert torch.allclose(xk[0, :, 0], torch.tensor(kept, dtype=torch.float32)), "feature/pos misaligned"

    # scattered importance: top-8 by value, order preserved
    imp2 = torch.tensor([[9., 1, 8, 2, 7, 3, 6, 4, 5, 0, 9.5, 0.5, 8.5, 1.5, 7.5, 2.5]])
    top8 = set(imp2[0].topk(8).indices.tolist())
    _, pos2 = p._prune(x, token_pos, imp2)
    assert set(pos2[0].tolist()) == top8, "prune must keep the top-8 importance set"
    assert pos2[0].tolist() == sorted(pos2[0].tolist()), "order preserved"

    # no-op when K>=N
    p_big = _bare_pruner(keep_count=64, gp=gp)
    xk3, pos3 = p_big._prune(x, token_pos, imp)
    assert xk3.shape[1] == N and torch.equal(pos3, token_pos), "K>=N must be a no-op"
    print("test_prune_math OK")


if __name__ == "__main__":
    test_scorer_shapes_and_roundtrip()
    test_prune_math()
    print("ALL MIDLAYER-SCORER TESTS PASSED")
