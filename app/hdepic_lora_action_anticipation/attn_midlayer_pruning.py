#!/usr/bin/env python3
# =============================================================================
# [B13-ATTN-MIDLAYER]  ADD-ON MODULE -- easy to merge or delete.
# -----------------------------------------------------------------------------
# Status      : experimental scaffold (main-agent, 2026-08-22). NOT part of the
#               in-flight b13 diagnostic runs; touches NO existing file.
# Purpose     : run the SAME received-attention top-K signal as the last-layer
#               ``TokenPruner`` (train_stream_mtp.py), but prune INSIDE the
#               encoder at an intermediate layer L* so blocks L*+1..L (and their
#               O(N^2) attention) only ever see the K survivors -- i.e. this
#               actually saves encoder FLOPs, unlike the post-encoder variant.
# Design       (agreed w/ user):
#   * encoder-internal prune  -> NO rebase: true (t,h,w) of survivors are
#     threaded into every later block's RoPE (via ``token_pos``).
#   * before predictor        -> REBASE to arange(K): we return a plain
#     ``PrunedAnticipativeModel`` in its default rebase mode, so the predictor
#     side is byte-for-byte identical to attention/recent -> FAIR comparison.
#   * budget parity           -> fixed ``keep_count`` (4 s), pruned only when
#     N > keep_count, exactly mirroring last-layer ``TokenPruner`` semantics.
# Relation     : this is the received-attention twin of ``loss_aware`` /
#               ``LossAwareEncoderPruner`` (cross-layer cascade). It deliberately
#               does NOT reuse ``forward_encoder_with_hooks`` because that path
#               prunes by a static per-layer RATIO; we need a per-sample fixed
#               COUNT to match the 4 s token budget across 4/6/8/10 s contexts.
# To REMOVE    : ``git rm`` this single file. No other file imports it unless a
#               driver explicitly does; grep ``B13-ATTN-MIDLAYER`` to confirm.
# To MERGE     : promote as a new ``--prune-strategy attention_midlayer`` branch
#               in eval_stream_mtp_kvcache_prune.build_pruned_model (one elif +
#               a --prune-layer arg). The standalone ``main`` below already runs
#               it end-to-end for the layer sweep in the meantime.
# =============================================================================
"""Intermediate-layer received-attention token pruning (B13 add-on).

Math (identical importance signal to ``train_stream_mtp.TokenPruner``, just read
at block L* instead of block L):

    A^{(h)}_{ij} = softmax_j( q_i^{(h)} . k_j^{(h)} / sqrt(d) )     # RoPE q,k
    s_j         = sum_h sum_i A^{(h)}_{ij}                          # received attn
    keep        = TopK_j(s_j), |keep| = floor(K/gp)*gp, original order

Cascade forward: run blocks 0..L* on all N tokens (RoPE = true positions), read
``s`` from block L*'s attention, prune the block-L* output to K, thread the
survivors' TRUE positions into blocks L*+1..L, then ``encoder.norm``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np  # noqa: F401  (kept for parity w/ eval module; harmless)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import the train module FIRST: it inserts VJEPA_ROOT onto sys.path so the
# ``src.*`` imports below resolve.
from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402
from app.hdepic_lora_action_anticipation.vit_encoder_loss_aware_pruning import (  # noqa: E402
    _resolve_video_geometry,
)
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (  # noqa: E402
    FpsSubsampledStreamMTPDataset,
    _LimitedLoader,
    enlarge_predictor_budget,
)


# ── received-attention capture on ONE block's attn ───────────────────────────
class _MidLayerAttnCapture:
    """Patch ``block.attn.forward`` to stash per-token received attention.

    Ported verbatim from ``train_stream_mtp.TokenPruner._patched_forward``
    (lines ~69-113); the ONLY change is the target module is an arbitrary block
    ``L*`` rather than ``encoder.blocks[-1]``. Keep in sync with that source if
    the encoder attention internals change.
    """

    def __init__(self, block_attn):
        self._m = block_attn
        self._orig_forward = block_attn.forward
        self.importance: torch.Tensor | None = None
        self.chunk_size = 256
        cap = self

        def _patched_forward(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            m = cap._m
            B, N, C = x.size()
            grid_depth = int(N // (m.grid_size * m.grid_size))
            qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if mask is not None:
                mask_p = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
                d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)
            else:
                if T is None or H_patches is None or W_patches is None:
                    mask_p = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=x.device)
                else:
                    mask_p = torch.arange(int(T * H_patches * W_patches), device=x.device)
                d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)
            s = 0
            qd = rotate_queries_or_keys(q[..., s : s + m.d_dim], pos=d_mask)
            kd = rotate_queries_or_keys(k[..., s : s + m.d_dim], pos=d_mask)
            s += m.d_dim
            qh = rotate_queries_or_keys(q[..., s : s + m.h_dim], pos=h_mask)
            kh = rotate_queries_or_keys(k[..., s : s + m.h_dim], pos=h_mask)
            s += m.h_dim
            qw = rotate_queries_or_keys(q[..., s : s + m.w_dim], pos=w_mask)
            kw = rotate_queries_or_keys(k[..., s : s + m.w_dim], pos=w_mask)
            s += m.w_dim
            if s < m.head_dim:
                q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
                k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
            else:
                q = torch.cat([qd, qh, qw], dim=-1)
                k = torch.cat([kd, kh, kw], dim=-1)
            with torch.no_grad():
                imp = torch.zeros(B, N, device=x.device, dtype=torch.float32)
                for ci in range(0, N, cap.chunk_size):
                    q_c = q[:, :, ci : ci + cap.chunk_size, :]
                    logits = (q_c @ k.transpose(-2, -1)) * m.scale
                    imp += logits.softmax(dim=-1).sum(dim=2).sum(dim=1).float()
                cap.importance = imp
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=m.proj_drop_prob, is_causal=m.is_causal, attn_mask=attn_mask
                )
            x = x.transpose(1, 2).reshape(B, N, C)
            x = m.proj(x)
            x = m.proj_drop(x)
            return x

        block_attn.forward = _patched_forward

    def remove(self):
        self._m.forward = self._orig_forward


# ── encoder-internal fixed-count prune at layer L* ───────────────────────────
class AttnMidLayerEncoderPruner:
    """Patch ``encoder.forward`` to prune to a fixed ``keep_count`` at block L*.

    Interface-compatible with ``LossAwareEncoderPruner`` (``_last_idx``,
    ``remove``) so the downstream ``PrunedAnticipativeModel(pruner=None)`` wrapper
    is reused unchanged. Runs its own block loop (no activation checkpointing:
    frozen ``no_grad`` eval, so checkpointing only risks a double stash write).
    """

    def __init__(self, encoder, *, prune_layer: int, keep_count: int, gp: int):
        self.encoder = encoder
        self.prune_layer = int(prune_layer)
        self.gp = int(gp)
        self.keep_count = max(self.gp, (int(keep_count) // self.gp) * self.gp)
        self._last_idx: torch.Tensor | None = None
        self._orig_forward = encoder.forward
        n_blocks = len(encoder.blocks)
        if not (0 <= self.prune_layer < n_blocks):
            raise ValueError(f"prune_layer {self.prune_layer} out of range [0,{n_blocks})")
        # Capture received attention at exactly block L*.
        self._capture = _MidLayerAttnCapture(encoder.blocks[self.prune_layer].attn)
        pruner = self
        encoder.forward = lambda x, masks=None: pruner._forward(x, masks=masks)

    def _prune(self, x, token_pos, imp):
        B, N, D = x.shape
        K = min(self.keep_count, (N // self.gp) * self.gp)
        if K >= N:
            return x, token_pos
        idx = imp.topk(K, dim=1).indices.sort(dim=1).values          # keep original order
        x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        token_pos = token_pos.gather(1, idx)                         # TRUE positions carried (no rebase)
        return x, token_pos

    def _forward(self, x, masks=None):
        enc = self.encoder
        if masks is not None:
            raise RuntimeError("[B13-ATTN-MIDLAYER] mask path not supported in this add-on")
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
                x, token_pos = self._prune(x, token_pos, self._capture.importance)
        if enc.norm is not None:
            x = enc.norm(x)
        self._last_idx = token_pos
        return x

    def remove(self):
        self._capture.remove()
        self.encoder.forward = self._orig_forward


def build_attn_midlayer_model(base, *, prune_layer: int, keep_count: int, gp: int,
                              num_tokens_full: int, device):
    """Factory mirroring ``eval_stream_mtp_kvcache_prune.build_pruned_model``.

    Returns a REBASE-mode ``PrunedAnticipativeModel`` (pruner=None) whose encoder
    is patched to prune internally -- predictor input == arange(K), identical to
    attention/recent (fair comparison).
    """
    enlarge_predictor_budget(base, num_tokens_full, gp)
    AttnMidLayerEncoderPruner(base.encoder, prune_layer=prune_layer,
                              keep_count=keep_count, gp=gp)
    return T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).to(device)


# ── standalone driver (delete with the file) ─────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="[B13-ATTN-MIDLAYER] intermediate-layer "
                                             "received-attention top-K prune eval")
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--init-from-ckpt", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--prune-layer", type=int, required=True,
                    help="encoder block index L* to prune at (sweep this vs last layer)")
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--primary-horizon-sec", type=float, default=2.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-val-batches", type=int, default=0)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    primary_h = float(args.primary_horizon_sec)
    primary_idx = horizons.index(primary_h) if primary_h in horizons else 0
    weights = [1.0] * len(horizons)

    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    val_ds = FpsSubsampledStreamMTPDataset(
        args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps
    )
    val_sampler = T.ContextBucketBatchSampler(val_ds, args.batch_size, shuffle=False, seed=0)
    loader_kwargs = dict(num_workers=args.num_workers, collate_fn=T.collate_stream,
                         pin_memory=False, persistent_workers=False)
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kwargs)
    if args.max_val_batches > 0:
        val_loader = _LimitedLoader(val_loader, args.max_val_batches)

    base = T.build_model(device, args.max_frames, args.fps, args.img_size, str(args.checkpoint))
    for p in base.encoder.parameters():
        p.requires_grad = False
    T.load_lora_sidecars(
        base,
        str(args.encoder_lora) if args.encoder_lora else None,
        str(args.predictor_lora) if args.predictor_lora else None,
    )
    gp = int(base.grid_size**2)
    num_tokens_full = (int(args.max_frames) // int(base.tubelet_size)) * gp

    model = build_attn_midlayer_model(
        base, prune_layer=args.prune_layer, keep_count=args.keep_count, gp=gp,
        num_tokens_full=num_tokens_full, device=device,
    )

    classifier = T.AttentiveClassifier(
        verb_classes=verb_map, noun_classes=noun_map, action_classes=action_map,
        embed_dim=int(base.encoder.embed_dim), num_heads=16, depth=4,
        use_activation_checkpointing=True,
    ).to(device)
    mtp_clf = T.CommunicatingMLPMTPClassifier(
        classifier, horizons_sec=horizons, comm_layers=2, comm_heads=4
    ).to(device)

    ck = torch.load(args.init_from_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
    del ck

    for p in model.parameters():
        p.requires_grad = False
    for p in mtp_clf.parameters():
        p.requires_grad = False

    with torch.no_grad():
        metrics = T.run_epoch(
            model, mtp_clf, val_loader, device, horizons, weights, primary_idx,
            verb_map, noun_map, action_map,
            optimizer=None, scaler=None, train=False, anticipation_sec=args.anticipation_sec,
        )

    report = {
        "prune_strategy": "attention_midlayer",
        "prune_layer": args.prune_layer,
        "fps": args.fps, "src_fps": args.src_fps,
        "max_frames": args.max_frames, "keep_count": args.keep_count,
        "position_mode": "rebase (encoder-internal true-pos, predictor arange(K))",
        "metric_scope": "native",
        "eval_path": "frozen EGTEA split1 stream-MTP val; native Action Top-5 @ 2/4/6",
        "action_top5": {f"{h:g}s": metrics.get(f"action_top5@{h:g}s") for h in horizons},
        "n": {f"{h:g}s": metrics["_metric_state"]["counts"].get(f"action_top5@{h:g}s") for h in horizons},
        "primary_action_top5": metrics.get("primary_action_top5"),
        "seconds": metrics.get("seconds"),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
