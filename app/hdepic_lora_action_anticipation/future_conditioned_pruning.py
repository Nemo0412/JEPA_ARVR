"""Future-conditioned / anticipation-aware KV-cache prune selectors (B13 TODO 3).

Two post-encoder token-memory pruners that score usefulness *for the future
action*, closing the gap identified by the keep-pattern diagnostic
([[b13-kvcache-prune-pattern-diagnostic]]): the existing ``attention`` selector
is content-adaptive but uses only *current* encoder self-attention, while the
winning ``loss_aware`` cascade is a fixed position-indexed mask -- neither scores
tokens by their value to the anticipated future.

Both are drop-in for ``train_stream_mtp.PrunedAnticipativeModel``: they expose the
same ``prune(feats) -> (kept_feats, kept_idx)`` interface as ``TokenPruner`` and
keep the top-K in original chronological order (multiple of ``gp``).

* ``FutureAttnTokenPruner`` (route 1) -- future-conditioned analog of attention.
  Runs ONE extra predictor probe pass over the full (unpruned) context with the
  same rebased target positions the rollout uses, and scores each context token
  by how much the future target/mask tokens attend to it. RoPE-agnostic: the
  attention is captured at the SDPA boundary (final rotated q/k) rather than
  re-deriving the rotation. Cheap: one forward, no gradients.

* ``HybridTokenPruner`` (route 3/2) -- fuses the calibrated loss-aware position
  *prior* (recency-heavy fixed mask that wins accuracy) with the per-sample
  attention *content* score (adaptive spread). ``w * prior + (1-w) * content``
  after per-sample min-max normalization. Motivated by the diagnostic's Jaccard
  0.36 between the two methods = headroom for a selector that is both adaptive
  and anticipation-calibrated.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.hdepic_lora_action_anticipation.loss_aware_pruning import lookup_calibrated_scores


def _round_keep(keep_count: int, gp: int) -> int:
    return max(int(gp), (int(keep_count) // int(gp)) * int(gp))


def _normalize_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize each row of [B, N] to [0, 1] (constant rows -> 0)."""
    lo = x.amin(dim=1, keepdim=True)
    hi = x.amax(dim=1, keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def _topk_gather(feats: torch.Tensor, score: torch.Tensor, keep_count: int, gp: int):
    """Keep the top-K positions by ``score`` (kept in original order)."""
    N = feats.shape[1]
    K = min(_round_keep(keep_count, gp), (N // gp) * gp)
    if K >= N:
        idx = torch.arange(N, device=feats.device).unsqueeze(0).expand(feats.size(0), -1)
        return feats, idx
    _, idx = score.topk(K, dim=1)
    idx = idx.sort(dim=1).values
    gathered = feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
    return gathered, idx


@contextlib.contextmanager
def _capture_last_sdpa(store: dict):
    """Temporarily wrap ``F.scaled_dot_product_attention`` to stash the final
    (post-RoPE) q, k of the *last* call. Only active around the probe forward, so
    every intercepted call is a predictor block; the last call is the last block.
    """
    orig = F.scaled_dot_product_attention

    def _wrapped(q, k, v, *args, **kwargs):
        store["q"] = q.detach()
        store["k"] = k.detach()
        return orig(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = _wrapped
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig


class FutureAttnTokenPruner:
    """Route 1: score context tokens by predictor attention received from the
    future target tokens (anticipation analog of the attention top-K selector).

    ``base`` is the anticipative wrapper (encoder + predictor). The prune runs a
    single predictor probe over the full context with target positions rebased
    exactly as ``PrunedAnticipativeModel`` does for the primary anticipation
    horizon, so the score reflects the same future the model rolls out to.
    """

    def __init__(self, base: nn.Module, keep_count: int, gp: int, *, anticipation_sec: float):
        self.base = base
        self.gp = int(gp)
        self.keep_count = _round_keep(keep_count, gp)
        self.anticipation_sec = float(anticipation_sec)

    def _target_positions(self, N: int, device) -> torch.Tensor:
        core = self.base
        steps = int(round(self.anticipation_sec * core.frames_per_second / core.tubelet_size))
        gp = int(core.grid_size ** 2)
        n_pred = int(gp * (core.num_output_frames // core.tubelet_size))
        skip = N + gp * steps
        return torch.arange(n_pred, device=device).unsqueeze(0) + skip

    @torch.no_grad()
    def prune(self, feats: torch.Tensor):
        core = self.base
        B, N, _ = feats.shape
        K = min(self.keep_count, (N // self.gp) * self.gp)
        if K >= N:
            idx = torch.arange(N, device=feats.device).unsqueeze(0).expand(B, -1)
            return feats, idx

        ctxt_positions = torch.arange(N, device=feats.device).unsqueeze(0).repeat(B, 1)
        tgt_positions = self._target_positions(N, feats.device).repeat(B, 1)
        n_pred = tgt_positions.shape[1]

        store: dict = {}
        with _capture_last_sdpa(store):
            core.predictor(feats, masks_x=ctxt_positions, masks_y=tgt_positions)
        q = store["q"]  # [B, heads, S, d]; S = N + n_pred, context sorted first
        k = store["k"]
        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        q_tgt = q[:, :, N : N + n_pred, :]  # future target queries
        logits = (q_tgt @ k.transpose(-2, -1)) * scale  # [B, heads, n_pred, S]
        attn = logits.softmax(dim=-1)
        # received attention from future targets onto each context key
        importance = attn[:, :, :, :N].sum(dim=2).mean(dim=1).float()  # [B, N]

        return _topk_gather(feats, importance, self.keep_count, self.gp)

    def remove(self):
        return None


class HybridTokenPruner:
    """Route 3/2: fuse the calibrated loss-aware position prior with the
    per-sample attention content score. ``score = w*prior + (1-w)*content``.

    Holds an inner ``TokenPruner`` so the encoder's final-block attention patch is
    installed (populating the per-sample content importance); the loss-aware prior
    is a fixed position-indexed table (one calibrated layer) looked up per token.
    """

    def __init__(
        self,
        encoder: nn.Module,
        keep_count: int,
        gp: int,
        *,
        calibrated_scores: dict,
        prior_layer: int,
        weight: float,
    ):
        from app.hdepic_lora_action_anticipation.train_stream_mtp import TokenPruner

        self.gp = int(gp)
        self.keep_count = _round_keep(keep_count, gp)
        self.weight = float(weight)
        self._attn = TokenPruner(encoder, keep_count=keep_count, gp=gp)
        self._calibrated = calibrated_scores
        self.prior_layer = int(prior_layer)

    def prune(self, feats: torch.Tensor):
        N = feats.shape[1]
        K = min(self.keep_count, (N // self.gp) * self.gp)
        if K >= N:
            idx = torch.arange(N, device=feats.device).unsqueeze(0).expand(feats.size(0), -1)
            return feats, idx

        content = self._attn._importance  # [B, N] received attention (per sample)
        if content is None:
            raise RuntimeError("Run encoder before HybridTokenPruner.prune()")
        content = content[:, :N].float()

        positions = torch.arange(N, device=feats.device).unsqueeze(0)  # [1, N]
        prior = lookup_calibrated_scores(self.prior_layer, positions, self._calibrated).float()
        prior = prior.to(feats.device).expand(feats.size(0), -1)  # [B, N]

        score = self.weight * _normalize_per_sample(prior) + (1.0 - self.weight) * _normalize_per_sample(content)
        return _topk_gather(feats, score, self.keep_count, self.gp)

    def remove(self):
        return self._attn.remove()
