#!/usr/bin/env python3
# =============================================================================
# [B17-LEARNED-EARLY-PRUNER]  ADD-ON MODULE -- easy to merge or delete.
# -----------------------------------------------------------------------------
# Candidate #1: a LEARNED token scorer on block-L* encoder features that prunes
# INSIDE the encoder (blocks L*+1..23 see only the K survivors -> real encoder-FLOP
# saving), replacing the fixed received-attention signal of attn_midlayer_pruning
# with a small MLP distilled to per-token TASK saliency (see gating probe).
#
# Shared by Stage 1 (distill_midlayer_scorer.py -> trains + saves the MLP) and
# Stage 2 (train_stream_mtp_learned_prune.py -> loads the frozen MLP, co-adapts
# predictor-LoRA + heads). Feature consistency: block-L* OUTPUT features over the
# FULL context, exactly as extracted in Stage 1.
#
# To REMOVE: git rm this file (+ distill/train wrappers). grep B17-LEARNED-EARLY-PRUNER.
# =============================================================================
"""Learned intermediate-layer token scorer + encoder-internal prune (B13 add-on)."""
from __future__ import annotations

import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: F401  (sys.path side effect)
from app.hdepic_lora_action_anticipation.vit_encoder_loss_aware_pruning import (
    _resolve_video_geometry,
)
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (
    enlarge_predictor_budget,
)


class MidLayerTokenScorer(nn.Module):
    """Small per-token MLP: block-L* feature (D) -> scalar keep-score.

    LayerNorm front-end makes it robust to the block-L* feature scale (self-
    contained, no external normalization stats to ship). Deliberately tiny so the
    encoder-FLOP saving is not eaten by the scorer.
    """

    def __init__(self, embed_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [..., D] -> score [...]
        return self.net(feats).squeeze(-1)

    @property
    def embed_dim(self) -> int:
        return self.net[0].normalized_shape[0]


class LearnedMidLayerEncoderPruner:
    """Patch ``encoder.forward`` to prune to a fixed ``keep_count`` at block L*
    using a learned ``MidLayerTokenScorer`` on the block-L* OUTPUT features.

    Fork of ``attn_midlayer_pruning.AttnMidLayerEncoderPruner``: same own-block-loop
    + true-position threading, but the importance signal is ``scorer(block_L*_out)``
    instead of received attention. Interface-compatible (``_last_idx``, ``remove``)
    so the downstream ``PrunedAnticipativeModel(pruner=None)`` wrapper is reused.

    The scorer is held here (NOT in the model tree) so it stays out of the Stage-2
    optimizer/state_dict; move it to the right device via ``.to()`` before use.
    """

    def __init__(self, encoder, scorer: MidLayerTokenScorer, *, prune_layer: int,
                 keep_count: int, gp: int):
        self.encoder = encoder
        self.scorer = scorer
        self.prune_layer = int(prune_layer)
        self.gp = int(gp)
        self.keep_count = max(self.gp, (int(keep_count) // self.gp) * self.gp)
        self._last_idx: torch.Tensor | None = None
        self._orig_forward = encoder.forward
        n_blocks = len(encoder.blocks)
        if not (0 <= self.prune_layer < n_blocks):
            raise ValueError(f"prune_layer {self.prune_layer} out of range [0,{n_blocks})")
        pruner = self
        encoder.forward = lambda x, masks=None: pruner._forward(x, masks=masks)

    def to(self, device):
        self.scorer = self.scorer.to(device)
        return self

    def _prune(self, x, token_pos, imp):
        B, N, D = x.shape
        K = min(self.keep_count, (N // self.gp) * self.gp)
        if K >= N:
            return x, token_pos
        idx = imp.topk(K, dim=1).indices.sort(dim=1).values          # keep original order
        x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        token_pos = token_pos.gather(1, idx)                         # TRUE positions (no rebase)
        return x, token_pos

    def _forward(self, x, masks=None):
        enc = self.encoder
        if masks is not None:
            raise RuntimeError("[B17-LEARNED-EARLY-PRUNER] mask path not supported")
        t, h_patches, w_patches = _resolve_video_geometry(enc, x)
        if enc.use_rope:
            x = enc.patch_embed(x)
        else:
            pos_embed = enc.interpolate_pos_encoding(x, enc.pos_embed)
            x = enc.patch_embed(x) + pos_embed
        token_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        for layer_idx, blk in enumerate(enc.blocks):
            x = blk(x, mask=token_pos, attn_mask=None, T=t, H_patches=h_patches, W_patches=w_patches)
            if layer_idx == self.prune_layer:
                imp = self.scorer(x.detach().to(next(self.scorer.parameters()).dtype))
                x, token_pos = self._prune(x, token_pos, imp)
        if enc.norm is not None:
            x = enc.norm(x)
        self._last_idx = token_pos
        return x

    def remove(self):
        self.encoder.forward = self._orig_forward


def build_learned_midlayer_model(base, scorer: MidLayerTokenScorer, *, prune_layer: int,
                                 keep_count: int, gp: int, num_tokens_full: int, device):
    """Mirror of ``attn_midlayer_pruning.build_attn_midlayer_model`` but with the
    learned scorer. Returns a REBASE-mode ``PrunedAnticipativeModel`` (pruner=None)
    whose encoder prunes internally at L* by the MLP score."""
    enlarge_predictor_budget(base, num_tokens_full, gp)
    pruner = LearnedMidLayerEncoderPruner(base.encoder, scorer, prune_layer=prune_layer,
                                          keep_count=keep_count, gp=gp)
    pruner.to(device)
    model = T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).to(device)
    return model, pruner
