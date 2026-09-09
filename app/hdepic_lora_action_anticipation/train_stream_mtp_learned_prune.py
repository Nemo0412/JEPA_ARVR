#!/usr/bin/env python3
# [B17-LEARNED-EARLY-PRUNER] Stage 2: co-adapt reader with a FROZEN learned L* scorer.
"""Stage-2 co-adaptation launcher for candidate #1.

Thin wrapper around the canonical ``train_stream_mtp.main()`` (unchanged). It
monkeypatches the model build so the trainer prunes INSIDE the encoder at block L*
with a frozen, distilled ``MidLayerTokenScorer`` (Stage 1) instead of the default
post-encoder ``TokenPruner(attention)``. Trainable set is unchanged by us
(predictor-LoRA + MTP heads, per the canonical trainer / budget-finetune recipe);
the scorer and encoder stay frozen and out of the optimizer/state_dict.

Config via env:
  SCORER_CKPT   path to Stage-1 scorer_{task,attn}.pt  (required)
  PRUNE_LAYER   encoder block index L*                  (default 8)

All other flags flow through to ``train_stream_mtp.main()`` argv unchanged.
"""
from __future__ import annotations

import os

import torch

from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (
    enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.midlayer_scorer import (
    MidLayerTokenScorer, LearnedMidLayerEncoderPruner,
)

_SCORER_CKPT = os.environ.get("SCORER_CKPT")
_PRUNE_LAYER = int(os.environ.get("PRUNE_LAYER", "8"))

# Launch gate #1 (budget-finetune card) + [[bf16-gradscaler-lora-nan]]: the canonical
# trainer builds ``GradScaler(enabled=True)`` under bf16 autocast, which is the known
# nonfinite-grad/NaN risk (bf16 has fp32 exponent range -> loss-scaling is unnecessary
# and harmful). encoder_lora.make_grad_scaler returns enabled=False for bf16; mirror it
# by forcing the trainer's scaler disabled. Opt back in with EVAL_BF16_GRAD_SCALER=1.
if os.environ.get("EVAL_BF16_GRAD_SCALER", "0") != "1":
    _OrigScaler = torch.cuda.amp.GradScaler
    torch.cuda.amp.GradScaler = lambda *a, **k: _OrigScaler(enabled=False)

_ORIG_BUILD_MODEL = T.build_model
_ORIG_PAM = T.PrunedAnticipativeModel


def _load_scorer(embed_dim, device):
    if not _SCORER_CKPT or not os.path.isfile(_SCORER_CKPT):
        raise SystemExit(f"[learned-prune] SCORER_CKPT missing/invalid: {_SCORER_CKPT!r}")
    ck = torch.load(_SCORER_CKPT, map_location="cpu", weights_only=False)
    if int(ck["embed_dim"]) != int(embed_dim):
        raise SystemExit(f"[learned-prune] scorer embed_dim {ck['embed_dim']} != encoder {embed_dim}")
    scorer = MidLayerTokenScorer(int(ck["embed_dim"]), hidden=int(ck["hidden"]))
    scorer.load_state_dict(ck["state_dict"])
    scorer.eval()
    for p in scorer.parameters():
        p.requires_grad = False
    print(f"[learned-prune] loaded scorer L*={_PRUNE_LAYER} target={ck.get('target')} "
          f"from {_SCORER_CKPT}", flush=True)
    return scorer.to(device)


def _build_model_learned(device, max_frames, fps, img_size, checkpoint):
    base = _ORIG_BUILD_MODEL(device, max_frames, fps, img_size, checkpoint)
    gp = int(base.grid_size**2)
    full_context_tokens = (int(max_frames) // int(base.tubelet_size)) * gp
    enlarge_predictor_budget(base, full_context_tokens, gp)
    return base


class _TokenPrunerShim:
    """Stands in for ``T.TokenPruner`` during Stage-2 build. Side effect: patch the
    encoder to prune internally at L* with the frozen learned scorer. The returned
    object is discarded by the PAM shim below."""

    def __new__(cls, encoder, keep_count, gp, chunk_size=256):  # noqa: D401
        device = next(encoder.parameters()).device
        embed_dim = int(encoder.embed_dim)
        scorer = _load_scorer(embed_dim, device)
        pruner = LearnedMidLayerEncoderPruner(
            encoder, scorer, prune_layer=_PRUNE_LAYER, keep_count=keep_count, gp=gp
        )
        pruner.to(device)
        return pruner  # discarded by _PAM_shim; encoder is already patched


def _PAM_shim(base, pruner, prune_threshold):
    # Encoder prunes internally (pruner patched onto encoder.forward); the predictor
    # side stays default rebase(arange(K)). Ignore the passed pruner, run pruner=None.
    return _ORIG_PAM(base, None, prune_threshold=10**9)


T.build_model = _build_model_learned
T.TokenPruner = _TokenPrunerShim
T.PrunedAnticipativeModel = _PAM_shim


if __name__ == "__main__":
    T.main()
