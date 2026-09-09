"""CPU logic tests for B13 future-conditioned / hybrid KV-cache prune selectors.

Validates the index math of ``future_conditioned_pruning`` in isolation (no model
/ video), so silent top-K / slicing bugs are caught before a GPU smoke:

* pure helpers (``_round_keep``/``_normalize_per_sample``/``_topk_gather``);
* ``_capture_last_sdpa`` grabs the LAST predictor block's q/k and restores;
* ``FutureAttnTokenPruner`` reproduces a reference future->context importance +
  target-position rebasing, against a fake single-block predictor;
* ``HybridTokenPruner`` score = w*norm(prior) + (1-w)*norm(content) top-K.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hdepic_lora_action_anticipation import future_conditioned_pruning as fcp


def test_pure_fns():
    assert fcp._round_keep(4100, 256) == 4096  # floor to multiple of gp
    assert fcp._round_keep(3900, 256) == 3840  # 15 * 256
    assert fcp._round_keep(100, 256) == 256     # >= gp floor
    x = torch.tensor([[1.0, 3.0, 2.0, 3.0]])
    assert torch.allclose(fcp._normalize_per_sample(x), torch.tensor([[0.0, 1.0, 0.5, 1.0]]), atol=1e-4)
    assert torch.allclose(fcp._normalize_per_sample(torch.full((1, 5), 2.0)), torch.zeros(1, 5))
    feats = torch.arange(6 * 3, dtype=torch.float32).reshape(1, 6, 3)
    score = torch.tensor([[0.0, 5.0, 1.0, 9.0, 2.0, 7.0]])
    kept, idx = fcp._topk_gather(feats, score, keep_count=4, gp=2)
    assert idx.tolist() == [[1, 3, 4, 5]]
    assert torch.equal(kept, feats[:, [1, 3, 4, 5], :])
    _, idx2 = fcp._topk_gather(feats, score, keep_count=100, gp=2)
    assert idx2.tolist() == [[0, 1, 2, 3, 4, 5]]


def test_sdpa_capture():
    store = {}
    q1, k1, v1 = (torch.randn(1, 2, 4, 8) for _ in range(3))
    q2, k2, v2 = (torch.randn(1, 2, 4, 8) for _ in range(3))
    with fcp._capture_last_sdpa(store):
        F.scaled_dot_product_attention(q1, k1, v1)
        F.scaled_dot_product_attention(q2, k2, v2)
    assert torch.equal(store["q"], q2) and torch.equal(store["k"], k2)


class _FakePredictor(nn.Module):
    def __init__(self, S, d, heads=2):
        super().__init__()
        self.q = torch.randn(1, heads, S, d)
        self.k = torch.randn(1, heads, S, d)
        self.v = torch.randn(1, heads, S, d)

    def forward(self, feats, masks_x, masks_y):
        return F.scaled_dot_product_attention(self.q, self.k, self.v)


class _FakeBase(nn.Module):
    def __init__(self, N, n_pred, d, heads=2):
        super().__init__()
        self.frames_per_second = 8
        self.tubelet_size = 2
        self.grid_size = 16
        self.num_output_frames = 2
        self.predictor = _FakePredictor(N + n_pred, d, heads)


def test_future_attn():
    N, n_pred, d, heads = 512, 256, 8, 2
    base = _FakeBase(N, n_pred, d, heads)
    pr = fcp.FutureAttnTokenPruner(base, keep_count=256, gp=256, anticipation_sec=2.0)
    feats = torch.randn(1, N, 5)
    kept, idx = pr.prune(feats)
    q, k = base.predictor.q, base.predictor.k
    scale = 1.0 / math.sqrt(d)
    attn = ((q[:, :, N : N + n_pred, :] @ k.transpose(-2, -1)) * scale).softmax(-1)
    imp = attn[:, :, :, :N].sum(2).mean(1).float()
    _, ref_idx = imp.topk(256, dim=1)
    ref_idx = ref_idx.sort(1).values
    assert torch.equal(idx, ref_idx)
    assert torch.equal(kept, feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, 5)))
    tp = pr._target_positions(N, feats.device)
    assert tp[0, 0].item() == N + 256 * int(round(2.0 * 8 / 2)) and tp.shape == (1, 256)


class _Hybrid(fcp.HybridTokenPruner):
    def __init__(self, w, table):
        self.gp = 256
        self.keep_count = 256
        self.weight = w

        class _A:
            _importance = None

        self._attn = _A()
        self._calibrated = {8: table}
        self.prior_layer = 8


def test_hybrid():
    N = 512
    table = torch.linspace(0, 1, steps=1024)
    hp = _Hybrid(0.5, table)
    content = torch.randn(1, N)
    hp._attn._importance = content
    feats = torch.randn(1, N, 4)
    _, idx = hp.prune(feats)
    prior = table[:N].unsqueeze(0)
    sc = 0.5 * fcp._normalize_per_sample(prior) + 0.5 * fcp._normalize_per_sample(content[:, :N])
    _, rid = sc.topk(256, 1)
    rid = rid.sort(1).values
    assert torch.equal(idx, rid)


if __name__ == "__main__":
    test_pure_fns()
    print("pure-fn tests OK")
    test_sdpa_capture()
    print("sdpa-capture tests OK")
    test_future_attn()
    print("future-attn tests OK")
    test_hybrid()
    print("hybrid tests OK")
    print("ALL TESTS PASSED")
