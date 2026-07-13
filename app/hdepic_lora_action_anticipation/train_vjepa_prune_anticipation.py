#!/usr/bin/env python
"""
V-JEPA2 1-minute-window action anticipation with attention-importance token pruning.

B12 extension (2026-06-26): 60s observation window, performance comparison vs Qwen2.5-VL-3B.
Adapted from the PhD's reference code (authoritative source, NOT his doc):
  * refer_repo/JEPA_ARVR/vjepa/train_hdepic_probe.py     -> TokenPruner (attention importance)
  * refer_repo/JEPA_ARVR/hdepic_anticipation_ar.py       -> encoder+predictor config, AR rollout, probe

Pipeline:
  frozen ViT-L/256 encoder  (480 frames -> N=61440 tokens)
    -> attention-importance TokenPruner: keep top-K=4096 tokens (~6.7%)
    -> re-base kept tokens to a compact W=16-slot context (positions 0..K-1)   [PhD in-regime choice]
    -> V-JEPA2 predictor (LoRA-finetuned) AR rollout, 1s anticipation
    -> AttentivePooler probe -> verb / noun / action heads.

Deviations from PhD code (user-approved 2026-06-26, see memory b12-1min-vjepa2-side):
  * predictor is LoRA-finetuned (PhD freezes it); encoder stays frozen.
  * pruning is composed with the predictor (PhD never combined them).
  * data = our phd_split CSVs (identical samples to the Qwen 1-min side), not the PhD pkl loader.
  * boundary clips (<num_frames of history) are edge-padded to num_frames so the encoder batches;
    immaterial under re-basing + fixed keep_count, and keeps the sample set identical to Qwen.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/path/to/VJEPA2-EXP")  # data/ckpt/outputs (main)
CODE_ROOT = os.environ.get("PROJECT_ROOT", SHARED)                            # code (worktree under slurm)
sys.path.insert(0, os.path.join(CODE_ROOT, "vjepa2"))
sys.path.insert(0, CODE_ROOT)

from src.models.attentive_pooler import AttentivePooler  # noqa: E402
from src.masks.utils import apply_masks  # noqa: E402
from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402
from evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar import (  # noqa: E402
    init_module as init_anticipative_module,
)
from loss_aware_pruning import LossAwarePruningConfig, simulate_cascade_keep_counts  # noqa: E402
from models.vit_encoder_pruning import LossAwareEncoderPruner  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


# ── Frame extraction (mirrors zeroshot_vlm_prompting.compute_clip_window, so the
#    V-JEPA2 side decodes the *same* observation window as the Qwen 1-min side) ──
def compute_clip_window(start_frame: int, video_fps: float, num_frames: int, target_fps: float) -> np.ndarray:
    """1s anticipation, drop out-of-bounds (boundary) frames instead of clamping."""
    aframes = int(1.0 * video_fps)
    af = int(start_frame) - aframes
    fstp = max(1, int(video_fps / target_fps))
    nframes = int(num_frames * fstp)
    indices = np.arange(af - nframes, af, fstp).astype(np.int64)
    return indices[indices >= 0]


# ── LoRA (matches the PhD's hand-rolled LoRALinear) ──────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank
        dev, dtype = linear.weight.device, linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(rank, linear.in_features, device=dev, dtype=dtype) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank, device=dev, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora_to_predictor(predictor: nn.Module, rank: int, alpha: float) -> int:
    """Inject LoRA into every predictor block's attention qkv & proj; freeze everything else."""
    for p in predictor.parameters():
        p.requires_grad = False
    for blk in predictor.predictor_blocks:
        attn = blk.attn
        attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha)
        attn.proj = LoRALinear(attn.proj, rank=rank, alpha=alpha)
    for name, p in predictor.named_parameters():
        p.requires_grad = ("lora_A" in name) or ("lora_B" in name)
    n_lora = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in predictor.parameters())
    print(f"  Predictor LoRA: rank={rank} alpha={alpha:.1f} | trainable {n_lora/1e6:.3f}M / {n_total/1e6:.1f}M")
    return n_lora


# ── Attention-importance Token Pruner (ported from train_hdepic_probe.py) ────
class TokenPruner:
    """Prune encoder tokens by attention received: importance[j] = Σ_{h,i} attn[i,j].

    Monkey-patches the last encoder block's RoPEAttention.forward to compute importance
    (chunked over queries, no full N² matrix) while still returning the real SDPA output.
    ``prune`` then keeps the top ``keep_count`` tokens in original order (rounded down to a
    multiple of the per-frame token count so the re-based context tiles whole temporal slots).
    """

    def __init__(self, encoder: nn.Module, keep_count: int, gp: int, chunk_size: int = 256):
        self.keep_count = (keep_count // gp) * gp
        self.gp = gp
        self.chunk_size = chunk_size
        self._importance: torch.Tensor | None = None

        m = encoder.blocks[-1].attn
        self._attn_module = m
        self._orig_forward = m.forward
        pruner = self

        def _patched_forward(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
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
            qd = rotate_queries_or_keys(q[..., s:s + m.d_dim], pos=d_mask)
            kd = rotate_queries_or_keys(k[..., s:s + m.d_dim], pos=d_mask)
            s += m.d_dim
            qh = rotate_queries_or_keys(q[..., s:s + m.h_dim], pos=h_mask)
            kh = rotate_queries_or_keys(k[..., s:s + m.h_dim], pos=h_mask)
            s += m.h_dim
            qw = rotate_queries_or_keys(q[..., s:s + m.w_dim], pos=w_mask)
            kw = rotate_queries_or_keys(k[..., s:s + m.w_dim], pos=w_mask)
            s += m.w_dim
            if s < m.head_dim:
                q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
                k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
            else:
                q = torch.cat([qd, qh, qw], dim=-1)
                k = torch.cat([kd, kh, kw], dim=-1)

            with torch.no_grad():
                imp = torch.zeros(B, N, device=x.device, dtype=torch.float32)
                for ci in range(0, N, pruner.chunk_size):
                    q_c = q[:, :, ci:ci + pruner.chunk_size, :]
                    logits = (q_c @ k.transpose(-2, -1)) * m.scale
                    imp += logits.softmax(dim=-1).sum(dim=2).sum(dim=1).float()
                pruner._importance = imp

            if attn_mask is not None or m.use_sdpa:
                with torch.backends.cuda.sdp_kernel():
                    x = F.scaled_dot_product_attention(
                        q, k, v, dropout_p=m.proj_drop_prob, is_causal=m.is_causal, attn_mask=attn_mask
                    )
            else:
                attn = (q @ k.transpose(-2, -1)) * m.scale
                attn = attn.softmax(dim=-1)
                attn = m.attn_drop(attn)
                x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = m.proj(x)
            x = m.proj_drop(x)
            return x

        m.forward = _patched_forward

    def prune(self, feats: torch.Tensor):
        """Returns (pruned [B,K,D], idx [B,K]) — idx = the kept tokens' TRUE positions (0..N-1)."""
        if self._importance is None:
            raise RuntimeError("Run encoder.forward() before pruner.prune().")
        N = feats.shape[1]
        K = min(self.keep_count, (N // self.gp) * self.gp)
        _, idx = self._importance.topk(K, dim=1)   # indices of top-K (topk returns values, indices)
        idx = idx.sort(dim=1).values                # restore original token order
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])
        return feats.gather(1, idx_exp), idx

    def remove(self):
        if self._orig_forward is not None:
            self._attn_module.forward = self._orig_forward
            self._orig_forward = None


def _install_attention_importance_capture(attn_module: nn.Module, on_importance, chunk_size: int):
    """Patch one RoPEAttention module to report attention-received column sums."""
    m = attn_module
    orig_forward = m.forward

    def _patched_forward(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
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
        qd = rotate_queries_or_keys(q[..., s:s + m.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s:s + m.d_dim], pos=d_mask)
        s += m.d_dim
        qh = rotate_queries_or_keys(q[..., s:s + m.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s:s + m.h_dim], pos=h_mask)
        s += m.h_dim
        qw = rotate_queries_or_keys(q[..., s:s + m.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s:s + m.w_dim], pos=w_mask)
        s += m.w_dim
        if s < m.head_dim:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        with torch.no_grad():
            imp = torch.zeros(B, N, device=x.device, dtype=torch.float32)
            for ci in range(0, N, chunk_size):
                q_c = q[:, :, ci:ci + chunk_size, :]
                logits = (q_c @ k.transpose(-2, -1)) * m.scale
                imp += logits.softmax(dim=-1).sum(dim=2).sum(dim=1).float()
            on_importance(imp)

        if attn_mask is not None or m.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=m.proj_drop_prob, is_causal=m.is_causal, attn_mask=attn_mask
                )
        else:
            attn = (q @ k.transpose(-2, -1)) * m.scale
            attn = attn.softmax(dim=-1)
            attn = m.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = m.proj(x)
        x = m.proj_drop(x)
        return x

    m.forward = _patched_forward
    return orig_forward


def parse_encoder_prune_schedule(spec: str, full_n: int, gp: int) -> list[tuple[int, int]]:
    """Parse 1-indexed ``block:keep`` items; keep <= 1.0 means ratio of full tokens."""
    schedule: list[tuple[int, int]] = []
    for item in [p.strip() for p in spec.split(",") if p.strip()]:
        if ":" not in item:
            raise ValueError(f"bad --encoder-prune-schedule item {item!r}; expected block:keep")
        block_s, keep_s = item.split(":", 1)
        block_idx = int(block_s) - 1
        keep_v = float(keep_s)
        keep = int(round(full_n * keep_v)) if keep_v <= 1.0 else int(round(keep_v))
        keep = max(gp, (keep // gp) * gp)
        schedule.append((block_idx, keep))
    if any(b < 0 for b, _ in schedule):
        raise ValueError("--encoder-prune-schedule uses 1-indexed block numbers; got block <= 0")
    return schedule


def build_loss_aware_pruning_config(args, full_n: int) -> LossAwarePruningConfig:
    if not args.loss_aware_pruning_config:
        raise ValueError(
            "--enable-loss-aware-pruning requires --loss-aware-pruning-config "
            "(run scripts/calibrate_loss_aware_pruning.py first)"
        )
    cfg = LossAwarePruningConfig.load(args.loss_aware_pruning_config)
    cfg.enable_loss_aware_pruning = True
    if cfg.num_tokens_full and cfg.num_tokens_full != full_n:
        raise ValueError(
            f"calibrated num_tokens_full={cfg.num_tokens_full} != runtime full_n={full_n}"
        )
    if not cfg.num_tokens_full:
        cfg.num_tokens_full = full_n
    return cfg


def build_loss_aware_pruner(encoder, args, full_n: int, gp: int) -> LossAwareEncoderPruner:
    cfg = build_loss_aware_pruning_config(args, full_n)
    return LossAwareEncoderPruner(
        encoder,
        cfg,
        gp=gp,
        num_tokens_full=full_n,
    )


class MidEncoderPruner:
    """Patch encoder.forward to gather tokens between transformer blocks.

    ``metric=next_attn`` uses the named block's own attention-received score, so a schedule
    item ``9:0.5`` means: run block 9 on the full stream, use its attention to score the block-8
    output after one consumer layer, then keep 50% for block 10 onward.
    """

    def __init__(self, encoder: nn.Module, schedule: list[tuple[int, int]], gp: int,
                 metric: str = "next_attn", chunk_size: int = 256):
        self.encoder = encoder
        self.schedule = dict(schedule)
        self.gp = gp
        self.metric = metric
        self.chunk_size = chunk_size
        self._last_idx: torch.Tensor | None = None
        self._importance: dict[int, torch.Tensor] = {}
        self._orig_encoder_forward = encoder.forward
        self._attn_originals: list[tuple[nn.Module, object]] = []

        if metric == "next_attn":
            for block_idx in self.schedule:
                attn = encoder.blocks[block_idx].attn

                def _save(imp, bi=block_idx):
                    self._importance[bi] = imp

                orig = _install_attention_importance_capture(attn, _save, chunk_size)
                self._attn_originals.append((attn, orig))

        pruner = self

        def _forward(x, masks=None):
            return pruner._forward(x, masks=masks)

        encoder.forward = _forward

    def _select(self, x: torch.Tensor, token_pos: torch.Tensor, block_idx: int):
        K = min(self.schedule[block_idx], (x.shape[1] // self.gp) * self.gp)
        if K >= x.shape[1]:
            return x, token_pos
        if self.metric == "feature_norm":
            score = x.float().norm(dim=-1)
        elif self.metric == "next_attn":
            if block_idx not in self._importance:
                raise RuntimeError(f"missing next-attn importance for encoder block {block_idx + 1}")
            score = self._importance.pop(block_idx)
        else:
            raise ValueError(f"unknown mid-encoder prune metric: {self.metric}")
        idx = score.topk(K, dim=1).indices.sort(dim=1).values
        x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        token_pos = token_pos.gather(1, idx)
        return x, token_pos

    def _forward(self, x, masks=None):
        enc = self.encoder
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        elif x.ndim == 5:
            _, _, T, H, W = x.shape
            T = T // enc.tubelet_size
        else:
            raise ValueError(f"expected image/video tensor, got shape {tuple(x.shape)}")
        H_patches = H // enc.patch_size
        W_patches = W // enc.patch_size
        if not enc.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        if not enc.use_rope:
            pos_embed = enc.interpolate_pos_encoding(x, enc.pos_embed)
            x = enc.patch_embed(x)
            x += pos_embed
        else:
            x = enc.patch_embed(x)

        if masks is not None:
            x = apply_masks(x, masks)
            token_pos = torch.cat(masks, dim=0).to(x.device)
        else:
            token_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)

        if enc.out_layers is not None:
            raise RuntimeError("MidEncoderPruner does not support encoder.out_layers")

        for i, blk in enumerate(enc.blocks):
            if enc.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, token_pos, None, T=T, H_patches=H_patches, W_patches=W_patches, use_reentrant=False
                )
            else:
                x = blk(x, mask=token_pos, attn_mask=None, T=T, H_patches=H_patches, W_patches=W_patches)
            if i in self.schedule:
                x, token_pos = self._select(x, token_pos, i)

        if enc.norm is not None:
            x = enc.norm(x)
        self._last_idx = token_pos
        return x

    def encode(self, clips: torch.Tensor):
        toks = self.encoder(clips)
        if self._last_idx is None:
            raise RuntimeError("mid-encoder pruner did not record kept token positions")
        return toks, self._last_idx

    @property
    def keep_count(self) -> int:
        return list(self.schedule.values())[-1]

    def remove(self):
        self.encoder.forward = self._orig_encoder_forward
        for attn, orig in self._attn_originals:
            attn.forward = orig


# ── Model build (ViT-L/256, frozen encoder + predictor; PhD config) ──────────
def build_model(device, frames_per_clip, fps, img_size, checkpoint):
    model_kwargs = {
        "use_v2_1": False,
        "encoder": {
            "model_name": "vit_large", "checkpoint_key": "target_encoder",
            "tubelet_size": 2, "patch_size": 16, "uniform_power": True, "use_rope": True,
        },
        "predictor": {
            "model_name": "vit_predictor", "checkpoint_key": "predictor",
            "num_frames": 64, "depth": 12, "num_heads": 12, "predictor_embed_dim": 384,
            "num_mask_tokens": 10, "uniform_power": True, "use_mask_tokens": True,
            "use_sdpa": True, "use_silu": False, "wide_silu": False, "use_rope": True,
        },
    }
    wrapper_kwargs = {"no_predictor": False, "num_output_frames": 2, "num_steps": 1}
    model = init_anticipative_module(
        frames_per_clip=frames_per_clip, frames_per_second=fps, resolution=img_size,
        checkpoint=checkpoint, model_kwargs=model_kwargs, wrapper_kwargs=wrapper_kwargs,
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.encoder.eval()
    return model


@torch.no_grad()
def encode_and_prune(model, pruner, clips):
    """Frozen encoder forward + attention-importance pruning. Returns pruned tokens [B, K, D].

    Deterministic (frozen encoder, no augmentation) -> safe to cache to disk.
    """
    if hasattr(pruner, "encode"):
        return pruner.encode(clips)
    x = model.encoder(clips)                         # [B, N, D]; importance captured by pruner
    return pruner.prune(x)                           # [B, K, D], K multiple of gp


def anticipate_from_ctx(model, ctx, anticipation_sec, ctx_idx=None, num_steps=1):
    """(LoRA) predictor AR rollout from pruned tokens ``ctx`` [B, K, D].

    position_mode (read from ``model._position_mode``, default 'rebased'):
      - 'rebased': context positions re-based to 0..K-1 (PhD in-regime sliding window).
      - 'true':    context kept at its REAL positions ``ctx_idx`` (0..N_full-1, spanning the
                   whole 60s); a single predictor step predicts the 1s after the real last frame
                   at positions N_full.. (requires predictor.num_patches lifted, set in main).
    Returns the slid window [B, W*gp, D] (probe input). Grad flows through predictor LoRA.
    """
    prd = model.predictor
    grid, tube, fps = int(model.grid_size), int(model.tubelet_size), int(model.frames_per_second)
    gp = grid * grid
    B, K, D = ctx.shape
    W = max(1, K // gp)
    total_slabs = max(1, int(round(float(anticipation_sec) * fps / tube)))
    mode = getattr(model, "_position_mode", "rebased")

    if mode == "true":
        n_full = int(getattr(model, "_full_n"))                  # full pre-prune token count (=240*256)
        ctx_pos = ctx_idx                                         # [B, K] true positions
        n_pred = gp * total_slabs                                 # predict the 1s after the real last frame
        tgt_pos = torch.arange(n_pred, device=ctx.device).unsqueeze(0).repeat(B, 1) + n_full
        pred = prd(ctx, masks_x=ctx_pos, masks_y=tgt_pos)
        if isinstance(pred, tuple):
            pred = pred[0]
        full = torch.cat([ctx, pred], dim=1)
        return full[:, -(W * gp):, :]

    # -- re-based (default): PhD-style sliding window, positions 0..K-1 --
    ctx_pos = torch.arange(W * gp, device=ctx.device).unsqueeze(0).repeat(B, 1)
    cap_slabs = getattr(prd, "num_frames", 64) // tube
    max_adv = max(1, cap_slabs - W)
    Ksteps = max(int(num_steps), (total_slabs + max_adv - 1) // max_adv)
    advanced = 0
    for kk in range(1, Ksteps + 1):
        target = int(round(total_slabs * kk / Ksteps))
        adv = min(target - advanced, max_adv)
        if adv <= 0:
            continue
        n_pred = gp * adv
        tgt_pos = torch.arange(n_pred, device=ctx.device).unsqueeze(0).repeat(B, 1) + (W * gp)
        pred = prd(ctx, masks_x=ctx_pos, masks_y=tgt_pos)
        if isinstance(pred, tuple):
            pred = pred[0]
        full = torch.cat([ctx, pred], dim=1)
        ctx = full[:, -(W * gp):, :]
        advanced += adv
    return ctx


# ── Probe head (PhD HDEpicProbe) ─────────────────────────────────────────────
class HDEpicProbe(nn.Module):
    def __init__(self, embed_dim, num_verbs, num_nouns, num_actions):
        super().__init__()
        self.pooler = AttentivePooler(num_queries=3, embed_dim=embed_dim, num_heads=16,
                                      depth=4, use_activation_checkpointing=False)
        self.verb_head = nn.Linear(embed_dim, num_verbs)
        self.noun_head = nn.Linear(embed_dim, num_nouns)
        self.action_head = nn.Linear(embed_dim, num_actions)

    def forward(self, x):
        x = self.pooler(x)
        return self.verb_head(x[:, 0]), self.noun_head(x[:, 1]), self.action_head(x[:, 2])


# ── Data ─────────────────────────────────────────────────────────────────────
def load_class_vocab(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    return {int(r["id"]): r["key"] for r in rows}


def build_action_map(*csv_paths):
    pairs = set()
    for p in csv_paths:
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            pairs.add((int(r["verb_class"]), int(r["noun_class"])))
    return {k: i for i, k in enumerate(sorted(pairs))}


def read_rows(csv_path):
    return list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))


def cache_key(args):
    # `_idx` = new cache format storing a dict {tok, idx} (idx = true token positions). The old
    # tokens-only caches (no suffix) from the re-based runs are left untouched / still loadable.
    suffix = ""
    if args.encoder_prune_schedule:
        sched = args.encoder_prune_schedule.replace(":", "x").replace(",", "_").replace(".", "p")
        suffix = f"_mid{args.encoder_prune_metric}_{sched}"
    elif getattr(args, "enable_loss_aware_pruning", False):
        cfg_tag = Path(args.loss_aware_pruning_config).stem
        suffix = f"_lapcfg_{cfg_tag}"
    return f"nf{args.num_frames}_fps{args.target_fps}_px{args.img_size}_keep{args.keep_count}{suffix}_idx"


def cache_path_for(cache_dir, key, row):
    return os.path.join(cache_dir, key, f"{row['participant_id']}__{row['video_id']}__{int(row['start_frame'])}.pt")


def _decode_clip(row, video_root, num_frames, target_fps, img_size, decode_size):
    """Decode the observation clip as **uint8** [C, T, H, W] (front edge-pad for batching).

    Returns uint8 (94 MB vs 377 MB float) to keep dataloader prefetch/pin memory small;
    /255 + ImageNet normalization is done on the GPU in build_cache (avoids CPU-RAM OOM).
    """
    from decord import VideoReader, cpu
    path = str(Path(video_root) / row["participant_id"] / f"{row['video_id']}.MP4")
    vr = VideoReader(path, num_threads=1, ctx=cpu(0), width=decode_size, height=decode_size)
    vfps = vr.get_avg_fps()
    idx = compute_clip_window(int(row["start_frame"]), vfps, num_frames, target_fps)
    idx = np.clip(idx, 0, len(vr) - 1)
    frames = vr.get_batch(idx).asnumpy()                          # [T,H,W,C] uint8 @ decode_size
    clip = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()  # [C,T,H,W] uint8
    if clip.shape[1] < num_frames:
        pad = clip[:, :1].repeat(1, num_frames - clip.shape[1], 1, 1)
        clip = torch.cat([pad, clip], dim=1)
    return clip                                                   # uint8, normalized on GPU


class ClipDataset(Dataset):
    """Decodes clips; used for the one-time cache build. Returns (clip, row_index)."""

    def __init__(self, rows, indices, video_root, num_frames, target_fps, img_size, decode_size):
        self.rows = rows
        self.indices = indices
        self.video_root = video_root
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.img_size = img_size
        self.decode_size = decode_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = self.indices[j]
        try:
            clip = _decode_clip(self.rows[i], self.video_root, self.num_frames,
                                self.target_fps, self.img_size, self.decode_size)
        except Exception as exc:  # noqa: BLE001
            print(f"  [decode-fail] {self.rows[i].get('video_id','?')}: {exc}", flush=True)
            clip = torch.zeros(3, self.num_frames, self.img_size, self.img_size, dtype=torch.uint8)
        return clip, i


class TokenDataset(Dataset):
    """Loads cached frozen-encoder pruned tokens [K, D]. Returns (tokens, v, n, a)."""

    def __init__(self, rows, cache_dir, key, action_map):
        self.cache_dir = cache_dir
        self.key = key
        self.action_map = action_map
        # Keep only rows whose cache file exists (skips any decode failures during build).
        self.rows = [r for r in rows if os.path.exists(cache_path_for(cache_dir, key, r))]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        v = int(r["verb_class"]); n = int(r["noun_class"])
        a = self.action_map.get((v, n), -1)
        obj = torch.load(cache_path_for(self.cache_dir, self.key, r), map_location="cpu")
        if isinstance(obj, dict):                       # new format {tok, idx}
            tokens, idx = obj["tok"], obj["idx"].long()
        else:                                           # legacy tokens-only tensor
            tokens = obj
            idx = torch.arange(tokens.shape[0], dtype=torch.long)
        return tokens.float(), idx, v, n, a


@torch.no_grad()
def build_cache(model, pruner, rows, cache_dir, key, video_root, args, device, use_bf16):
    """One-time pass: decode -> frozen encoder -> prune -> write fp16 tokens [K, D] per sample."""
    out = os.path.join(cache_dir, key)
    os.makedirs(out, exist_ok=True)
    missing = [i for i, r in enumerate(rows) if not os.path.exists(cache_path_for(cache_dir, key, r))]
    if args.cache_num_shards > 1:
        missing = [i for i in missing if (i % args.cache_num_shards) == args.cache_shard]
        print(f"  [cache] shard {args.cache_shard}/{args.cache_num_shards}", flush=True)
    if not missing:
        print(f"  [cache] all present (this shard) at {out}", flush=True)
        return
    print(f"  [cache] building {len(missing)}/{len(rows)} -> {out}", flush=True)
    ds = ClipDataset(rows, missing, video_root, args.num_frames, args.target_fps, args.img_size, args.img_size)
    loader = DataLoader(ds, batch_size=args.cache_build_batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)
    done = 0; t0 = time.time()
    for clips, idxs in loader:
        # uint8 [B,C,T,H,W] -> GPU -> float/255 -> ImageNet normalize (kept off CPU to bound RAM)
        clips = clips.to(device, non_blocking=True).float().div_(255.0).sub_(mean).div_(std)
        with autocast:
            if pruner is not None:
                toks, kept = encode_and_prune(model, pruner, clips)  # [B,K,D], [B,K] true positions
            else:                                                    # no-prune: keep ALL tokens
                toks = model.encoder(clips)                          # [B,N,D]
                kept = torch.arange(toks.shape[1], device=device).unsqueeze(0).expand(toks.shape[0], -1)
        toks = toks.to(torch.float16).cpu()
        kept = kept.to(torch.int32).cpu()
        for b in range(toks.shape[0]):
            p = cache_path_for(cache_dir, key, rows[int(idxs[b])])
            tmp = p + f".tmp{os.getpid()}"
            torch.save({"tok": toks[b].clone(), "idx": kept[b].clone()}, tmp)
            os.replace(tmp, p)
        done += toks.shape[0]
        del clips, toks, kept
        if done % (args.cache_build_batch * 10) < args.cache_build_batch:
            import gc; gc.collect()   # decord leaks native buffers across many VideoReaders
            print(f"    cached {done}/{len(missing)} ({(time.time()-t0)/max(done,1):.2f}s/sample)", flush=True)
    print(f"  [cache] done {done} in {time.time()-t0:.0f}s", flush=True)


# ── Eval ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, probe, loader, device, anticipation_sec, use_bf16):
    probe.eval(); model.predictor.eval()
    v_c = defaultdict(int); v_t = defaultdict(int)
    n_c = defaultdict(int); n_t = defaultdict(int)
    a_c = defaultdict(int); a_t = defaultdict(int)
    v_top1 = v_top3 = v_top5 = n_top1 = n_top3 = n_top5 = a_top1 = a_top3 = a_top5 = total = 0
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)
    for tokens, idx, v_ids, n_ids, a_ids in loader:
        tokens = tokens.to(device); idx = idx.to(device)
        with autocast:
            feats = anticipate_from_ctx(model, tokens, anticipation_sec, ctx_idx=idx)
            vl, nl, al = probe(feats.float())
        for i in range(len(v_ids)):
            vi, ni, ai = int(v_ids[i]), int(n_ids[i]), int(a_ids[i])
            v_t[vi] += 1; n_t[ni] += 1
            v5 = vl[i].topk(5).indices.tolist(); n5 = nl[i].topk(5).indices.tolist()
            if vi == v5[0]: v_top1 += 1
            if vi in v5[:3]: v_top3 += 1
            if vi in v5: v_top5 += 1; v_c[vi] += 1
            if ni == n5[0]: n_top1 += 1
            if ni in n5[:3]: n_top3 += 1
            if ni in n5: n_top5 += 1; n_c[ni] += 1
            if ai != -1:
                a_t[ai] += 1
                a5 = al[i].topk(5).indices.tolist()
                if ai == a5[0]: a_top1 += 1
                if ai in a5[:3]: a_top3 += 1
                if ai in a5: a_top5 += 1; a_c[ai] += 1
            total += 1

    def cmr(c, t):
        r = [c.get(k, 0) / val for k, val in t.items()]
        return float(np.mean(r) * 100) if r else 0.0

    n_act = max(sum(a_t.values()), 1)
    return {
        "verb_top1": 100 * v_top1 / max(total, 1), "verb_top3": 100 * v_top3 / max(total, 1),
        "verb_top5": 100 * v_top5 / max(total, 1),
        "noun_top1": 100 * n_top1 / max(total, 1), "noun_top3": 100 * n_top3 / max(total, 1),
        "noun_top5": 100 * n_top5 / max(total, 1),
        "action_top1": 100 * a_top1 / n_act, "action_top3": 100 * a_top3 / n_act,
        "action_top5": 100 * a_top5 / n_act,
        "verb_r5": cmr(v_c, v_t), "noun_r5": cmr(n_c, n_t), "action_r5": cmr(a_c, a_t),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ann = f"{SHARED}/data/hdepic_vjepa_annotations/phd_split"
    ap.add_argument("--train-csv", default=f"{ann}/HD_EPIC_train_vjepa.csv")
    ap.add_argument("--val-csv", default=f"{ann}/HD_EPIC_val_vjepa.csv")
    ap.add_argument("--test-csv", default=f"{ann}/HD_EPIC_test_vjepa.csv")
    ap.add_argument("--verb-classes-csv",
                    default=f"{SHARED}/data/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_verb_classes.csv")
    ap.add_argument("--noun-classes-csv",
                    default=f"{SHARED}/data/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_noun_classes.csv")
    ap.add_argument("--video-root", default=f"{SHARED}/data/hdepic_vjepa_videos")
    ap.add_argument("--checkpoint", default=f"{SHARED}/checkpoints/vitl.pt")
    ap.add_argument("--out-dir", default=f"{SHARED}/outputs/vjepa_prune_anticipation/b12_1min")
    ap.add_argument("--cache-dir", default=f"{SHARED}/data/preproc_cache_vjepa",
                    help="frozen-encoder pruned-token cache root (built once, reused across epochs)")
    ap.add_argument("--cache-build-batch", type=int, default=4,
                    help="batch size for the one-time encoder+prune cache build")
    ap.add_argument("--num-frames", type=int, default=480)
    ap.add_argument("--target-fps", type=float, default=8.0)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--anticipation-sec", type=float, default=1.0)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--prune-chunk-size", type=int, default=256)
    ap.add_argument("--encoder-prune-schedule", default="",
                    help="optional mid-encoder pruning schedule, 1-indexed block:keep pairs. "
                         "keep <= 1 is a ratio of the full token count; keep > 1 is an absolute "
                         "token count rounded to whole frames. Example for next-layer pruning: "
                         "'9:0.5,17:4096' scores with blocks 9 and 17, pruning for later blocks.")
    ap.add_argument("--encoder-prune-metric", choices=["next_attn", "feature_norm"], default="next_attn",
                    help="mid-encoder score. next_attn uses the scheduled block's attention-received "
                         "column sum; feature_norm uses ||h_i||_2 after the scheduled block.")
    ap.add_argument("--enable-loss-aware-pruning", action="store_true",
                    help="use offline-calibrated loss-aware cascade pruning (requires --loss-aware-pruning-config)")
    ap.add_argument("--loss-aware-pruning-config", default="",
                    help="calibrated LossAwarePruningConfig JSON from calibrate_loss_aware_pruning.py")
    ap.add_argument("--no-prune", action="store_true",
                    help="full-context experiment: keep ALL encoder tokens (no pruning), forces "
                         "position-mode=true. Cache stores the full ~61440-token output per sample.")
    ap.add_argument("--build-cache-only", action="store_true",
                    help="build the frozen-encoder token cache (optionally sharded) and exit; no training")
    ap.add_argument("--cache-shard", type=int, default=0,
                    help="this task's shard id for a parallel (Slurm-array) cache build")
    ap.add_argument("--cache-num-shards", type=int, default=1,
                    help="total number of shards; each task builds rows where index %% num_shards == shard")
    ap.add_argument("--cache-key-override", default="",
                    help="skip cache_key() computation and use this literal key (for externally-built caches, "
                         "e.g. pred0-guided pruning); cache_build is a no-op when all files are already present")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-lr", type=float, default=5e-5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--num-epochs", type=int, default=10)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-eval-samples", type=int, default=0)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--run-tag", default="b12_vjepa_1min")
    ap.add_argument("--eval-only", default="", help="path to a ckpt: skip training, eval val+test, exit")
    ap.add_argument("--best-metric", default="action_top5",
                    help="validation metric used to write *-best.pt; action_top5 is the primary "
                         "HD-EPIC action-anticipation metric")
    ap.add_argument("--position-mode", choices=["rebased", "true"], default="rebased",
                    help="rebased = pruned tokens re-based to 0..K-1 (PhD in-regime); "
                         "true = keep real positions 0..N-1 over the whole window (lifts num_patches)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (not args.no_bf16) and device.type == "cuda"
    print("=" * 70)
    print(f"V-JEPA2 prune->predictor(LoRA)->probe | tag={args.run_tag}")
    print(f"  frames={args.num_frames} fps={args.target_fps} keep_count={args.keep_count} "
          f"anticipation={args.anticipation_sec}s lora_rank={args.lora_rank} bf16={use_bf16}")
    print("=" * 70, flush=True)

    verb_names = load_class_vocab(args.verb_classes_csv)
    noun_names = load_class_vocab(args.noun_classes_csv)
    action_map = build_action_map(args.train_csv, args.val_csv, args.test_csv)
    num_verbs = max(verb_names) + 1
    num_nouns = max(noun_names) + 1
    num_actions = len(action_map)
    print(f"  verbs={num_verbs} nouns={num_nouns} actions={num_actions}", flush=True)

    # ── Model first (needed to build the frozen-encoder pruned-token cache) ──
    print("\n[build] encoder+predictor (ViT-L/256)...", flush=True)
    model = build_model(device, args.num_frames, int(args.target_fps), args.img_size, args.checkpoint)
    gp = int(model.grid_size) ** 2
    full_n = (args.num_frames // int(model.tubelet_size)) * gp        # full pre-prune token count
    if args.no_prune:
        if args.encoder_prune_schedule:
            raise ValueError("--no-prune and --encoder-prune-schedule are mutually exclusive")
        if args.enable_loss_aware_pruning:
            raise ValueError("--no-prune and --enable-loss-aware-pruning are mutually exclusive")
        # full-context experiment: keep every token, real positions over the whole window.
        args.keep_count = full_n            # -> distinct cache key keep{full_n}; W = full_n//gp slots
        args.position_mode = "true"
        pruner = None
        print(f"  NO-PRUNE: keeping all {full_n} tokens, position_mode forced to 'true'", flush=True)
    elif args.enable_loss_aware_pruning:
        if args.encoder_prune_schedule:
            raise ValueError("--enable-loss-aware-pruning and --encoder-prune-schedule are mutually exclusive")
        pruner = build_loss_aware_pruner(model.encoder, args, full_n, gp)
        args.keep_count = pruner.keep_count
        cascade = simulate_cascade_keep_counts(
            full_n,
            pruner.prune_ratio_schedule,
            gp=gp,
            round_to_frame_tokens=pruner.config.round_to_frame_tokens,
        )
        ratio_desc = ", ".join(
            f"block{l}:r={pruner.prune_ratio_schedule[l]:.4f}" for l in sorted(pruner.prune_ratio_schedule)
        )
        cascade_desc = ", ".join(f"after{l}->{k}" for l, k in cascade.items())
        print(f"  LOSS-AWARE calibrated cascade | {ratio_desc}", flush=True)
        print(f"  cascade keep counts | {cascade_desc}", flush=True)
        print(f"  final keep_count={args.keep_count} -> W={args.keep_count // gp} slots", flush=True)
    elif args.encoder_prune_schedule:
        schedule = parse_encoder_prune_schedule(args.encoder_prune_schedule, full_n=full_n, gp=gp)
        max_block = len(model.encoder.blocks) - 1
        bad = [b + 1 for b, _ in schedule if b > max_block]
        if bad:
            raise ValueError(f"--encoder-prune-schedule references missing blocks: {bad}")
        args.keep_count = schedule[-1][1]    # final cache/probe length follows last mid-prune stage
        pruner = MidEncoderPruner(
            model.encoder, schedule=schedule, gp=gp,
            metric=args.encoder_prune_metric, chunk_size=args.prune_chunk_size,
        )
        desc = ", ".join(f"block{b + 1}->keep{k}" for b, k in schedule)
        print(f"  MID-ENCODER prune metric={args.encoder_prune_metric}: {desc}", flush=True)
        print(f"  final keep_count={args.keep_count} -> W={args.keep_count // gp} slots", flush=True)
    else:
        pruner = TokenPruner(model.encoder, keep_count=args.keep_count, gp=gp, chunk_size=args.prune_chunk_size)
        print(f"  keep_count(rounded)={pruner.keep_count} -> W={pruner.keep_count // gp} slots", flush=True)
    apply_lora_to_predictor(model.predictor, rank=args.lora_rank, alpha=args.lora_alpha)
    model.predictor.train()
    probe = HDEpicProbe(model.embed_dim, num_verbs, num_nouns, num_actions).to(device)

    # ── Position mode (re-based vs true) ──
    model._position_mode = args.position_mode
    model._full_n = full_n
    if args.position_mode == "true":
        # context spans real positions 0..full_n-1; target = 1s after the real last frame.
        anticip_steps = int(round(args.anticipation_sec * int(model.frames_per_second) / int(model.tubelet_size)))
        max_tgt = full_n + gp * (anticip_steps + 1) + gp   # generous upper bound on target index
        new_cap = ((max_tgt // gp) + 4) * gp
        model.predictor.num_patches = new_cap              # config buffer, no new params under RoPE
        print(f"  position_mode=true | full_n={full_n} -> predictor.num_patches lifted to {new_cap}", flush=True)
    else:
        print(f"  position_mode=rebased | full_n={full_n} (unused)", flush=True)

    # ── Rows + one-time pruned-token cache (frozen encoder, no aug -> deterministic) ──
    train_rows = read_rows(args.train_csv)
    val_rows = read_rows(args.val_csv)
    test_rows = read_rows(args.test_csv)
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_eval_samples:
        val_rows = val_rows[: args.max_eval_samples]
        test_rows = test_rows[: args.max_eval_samples]
    key = args.cache_key_override if args.cache_key_override else cache_key(args)
    print(f"\n[cache] key={key} dir={args.cache_dir}", flush=True)
    build_cache(model, pruner, train_rows, args.cache_dir, key, args.video_root, args, device, use_bf16)
    build_cache(model, pruner, val_rows, args.cache_dir, key, args.video_root, args, device, use_bf16)
    build_cache(model, pruner, test_rows, args.cache_dir, key, args.video_root, args, device, use_bf16)

    if args.build_cache_only:
        print(f"\n[build-cache-only] shard {args.cache_shard}/{args.cache_num_shards} done; exiting.", flush=True)
        return

    train_ds = TokenDataset(train_rows, args.cache_dir, key, action_map)
    val_ds = TokenDataset(val_rows, args.cache_dir, key, action_map)
    test_ds = TokenDataset(test_rows, args.cache_dir, key, action_map)
    print(f"  samples train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    def fmt(tag, m):
        return (f"  [{tag}] verb  top1={m['verb_top1']:.1f} top3={m['verb_top3']:.1f} top5={m['verb_top5']:.1f} r5={m['verb_r5']:.1f}\n"
                f"  [{tag}] noun  top1={m['noun_top1']:.1f} top3={m['noun_top3']:.1f} top5={m['noun_top5']:.1f} r5={m['noun_r5']:.1f}\n"
                f"  [{tag}] action top1={m['action_top1']:.1f} top3={m['action_top3']:.1f} top5={m['action_top5']:.1f} r5={m['action_r5']:.1f}")

    if args.eval_only:
        print(f"\n[eval-only] loading {args.eval_only}", flush=True)
        ck = torch.load(args.eval_only, map_location=device, weights_only=False)
        probe.load_state_dict(ck["probe"])
        model.predictor.load_state_dict(ck["predictor_lora"], strict=False)
        vm = evaluate(model, probe, val_loader, device, args.anticipation_sec, use_bf16)
        tm = evaluate(model, probe, test_loader, device, args.anticipation_sec, use_bf16)
        print("[eval-only] VAL:\n" + fmt("val", vm), flush=True)
        print("[eval-only] TEST:\n" + fmt("test", tm), flush=True)
        print("=" * 70, flush=True)
        return

    lora_params = [p for p in model.predictor.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        [{"params": list(probe.parameters()), "lr": args.lr},
         {"params": lora_params, "lr": args.lora_lr}],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)

    best_metric = -float("inf")
    last_ckpt = os.path.join(args.out_dir, f"{args.run_tag}-last.pt")
    best_ckpt = os.path.join(args.out_dir, f"{args.run_tag}-best.pt")

    for epoch in range(args.num_epochs):
        probe.train(); model.predictor.train()
        t0 = time.time(); running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for bi, (tokens, idx, v_ids, n_ids, a_ids) in enumerate(train_loader):
            tokens = tokens.to(device, non_blocking=True); idx = idx.to(device, non_blocking=True)
            v_ids = v_ids.to(device); n_ids = n_ids.to(device); a_ids = a_ids.to(device)
            with autocast():
                feats = anticipate_from_ctx(model, tokens, args.anticipation_sec, ctx_idx=idx)
                vl, nl, al = probe(feats.float())
                loss = criterion(vl, v_ids) + criterion(nl, n_ids)
                valid = a_ids >= 0
                if valid.any():
                    loss = loss + criterion(al[valid], a_ids[valid])
            (loss / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            if (bi + 1) % args.log_every == 0:
                print(f"  ep{epoch+1} step {bi+1}/{len(train_loader)} loss={loss.item():.3f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        print(f"\nEpoch {epoch+1}/{args.num_epochs} avg_loss={running/max(len(train_loader),1):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        m = evaluate(model, probe, val_loader, device, args.anticipation_sec, use_bf16)
        print(f"  [val] verb_top3={m['verb_top3']:.1f} noun_top3={m['noun_top3']:.1f} "
              f"action_top3={m['action_top3']:.1f} action_top5={m['action_top5']:.1f} | "
              f"verb_r5={m['verb_r5']:.1f} noun_r5={m['noun_r5']:.1f} "
              f"action_r5={m['action_r5']:.1f}", flush=True)

        ckpt = {"epoch": epoch + 1, "probe": probe.state_dict(),
                "predictor_lora": {k: v for k, v in model.predictor.state_dict().items()
                                   if "lora_A" in k or "lora_B" in k},
                "metrics": m, "action_map": action_map, "args": vars(args)}
        torch.save(ckpt, last_ckpt)
        if args.best_metric not in m:
            raise KeyError(f"--best-metric {args.best_metric!r} is not an evaluate() metric; got {sorted(m)}")
        if m[args.best_metric] > best_metric:
            best_metric = m[args.best_metric]
            torch.save(ckpt, best_ckpt)
            print(f"  ✓ best {args.best_metric}={best_metric:.1f} -> {best_ckpt}", flush=True)

    print("\n[test] final eval with best checkpoint...", flush=True)
    if os.path.isfile(best_ckpt):
        b = torch.load(best_ckpt, map_location=device, weights_only=False)
        probe.load_state_dict(b["probe"])
        model.predictor.load_state_dict(b["predictor_lora"], strict=False)
    tm = evaluate(model, probe, test_loader, device, args.anticipation_sec, use_bf16)
    print(f"  [test] verb_top3={tm['verb_top3']:.1f} noun_top3={tm['noun_top3']:.1f} "
          f"action_top3={tm['action_top3']:.1f} action_top5={tm['action_top5']:.1f} | "
          f"verb_r5={tm['verb_r5']:.1f} noun_r5={tm['noun_r5']:.1f} "
          f"action_r5={tm['action_r5']:.1f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
