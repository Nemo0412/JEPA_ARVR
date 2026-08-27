"""L23 head-pattern KV pruning for video-only V-JEPA.

Policies from last-layer encoder profiling (20 HD-EPIC clips):

  * stable heads  {0,1,3,4,6,9,11,12}: keep border/corners ∪ recent slots
  * content heads {8,10,14}: keep gaze (else spatial center) ∪ recent slots
  * other heads: keep all keys

Keys are masked inside last-layer RoPE attention (SDPA attn_mask).
Sequence length and RoPE positions are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from src.models.utils.modules import rotate_queries_or_keys

STABLE_HEADS = (0, 1, 3, 4, 6, 9, 11, 12)
CONTENT_HEADS = (8, 10, 14)


@dataclass
class L23PruneConfig:
    grid: int = 16
    border: int = 2
    recent_frac: float = 0.30
    center_frac: float = 0.50
    stable_heads: tuple[int, ...] = STABLE_HEADS
    content_heads: tuple[int, ...] = CONTENT_HEADS
    keep_recent_on_content: bool = True
    keep_other_full: bool = True


def build_head_keep_mask(
    n_tok: int,
    n_heads: int,
    cfg: L23PruneConfig,
    gaze_flat: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Bool keep mask [H, N]; True = keep key."""
    gp = cfg.grid * cfg.grid
    t_slots = max(n_tok // gp, 1)
    idx = torch.arange(n_tok)
    t = idx // gp
    inner = idx % gp
    y = inner // cfg.grid
    x = inner % cfg.grid
    b = int(cfg.border)
    g = int(cfg.grid)
    border = (y < b) | (y >= g - b) | (x < b) | (x >= g - b)
    t_cut = int((1.0 - cfg.recent_frac) * t_slots)
    recent = t >= t_cut
    half = cfg.center_frac / 2.0
    lo = int(round((0.5 - half) * g))
    hi = int(round((0.5 + half) * g))
    center = (y >= lo) & (y < hi) & (x >= lo) & (x < hi)

    gaze = torch.zeros(n_tok, dtype=torch.bool)
    if gaze_flat is not None and int(gaze_flat.numel()) == n_tok:
        gaze = gaze_flat.bool().reshape(-1).cpu()

    keep = torch.ones(n_heads, n_tok, dtype=torch.bool)
    stable_keep = border | recent
    content_keep = gaze | center
    if cfg.keep_recent_on_content:
        content_keep = content_keep | recent

    for h in cfg.stable_heads:
        if 0 <= h < n_heads:
            keep[h] = stable_keep
    for h in cfg.content_heads:
        if 0 <= h < n_heads:
            keep[h] = content_keep
    if not cfg.keep_other_full:
        used = set(cfg.stable_heads) | set(cfg.content_heads)
        for h in range(n_heads):
            if h not in used:
                keep[h] = recent
    if device is not None:
        keep = keep.to(device)
    return keep


class L23PatternKVPruner:
    """Monkey-patch encoder.blocks[-1].attn to drop keys per head."""

    def __init__(self, encoder, cfg: L23PruneConfig | None = None):
        self.cfg = cfg or L23PruneConfig()
        self.encoder = encoder
        self.attn = encoder.blocks[-1].attn
        self._orig = self.attn.forward
        self.gaze_flat: torch.Tensor | None = None
        self.last_keep: torch.Tensor | None = None
        self._install()

    def set_gaze(self, gaze_flat: torch.Tensor | None):
        self.gaze_flat = gaze_flat

    def _install(self):
        pruner = self
        m = self.attn

        def _forward(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
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

            keep = build_head_keep_mask(N, m.num_heads, pruner.cfg, pruner.gaze_flat, device=x.device)
            pruner.last_keep = keep
            drop = ~keep
            additive = torch.zeros(1, m.num_heads, 1, N, device=x.device, dtype=q.dtype)
            additive = additive.masked_fill(drop.view(1, m.num_heads, 1, N), torch.finfo(q.dtype).min)
            if attn_mask is not None:
                additive = additive + attn_mask

            with torch.backends.cuda.sdp_kernel():
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=m.is_causal, attn_mask=additive
                )
            y = y.transpose(1, 2).reshape(B, N, C)
            y = m.proj(y)
            y = m.proj_drop(y)
            return y

        m.forward = _forward

    def remove(self):
        self.attn.forward = self._orig
        self._orig = None

    def keep_stats(self) -> dict:
        k = self.last_keep
        if k is None:
            return {}
        return {
            "all_keep": float(k.float().mean()),
            "stable_keep": float(k[list(self.cfg.stable_heads)].float().mean()),
            "content_keep": float(k[list(self.cfg.content_heads)].float().mean()),
            "n_tok": int(k.shape[1]),
            "n_heads": int(k.shape[0]),
        }
