"""Qwen-style decoder-only on frozen V-JEPA ViT-L latents.

Aligned with V-JEPA stream MTP on the vision / protocol / heads side; the
future module follows a **Qwen-like causal LM decoder**:
  - vision tokens as a **prefix** (decoder-only, not JEPA mask-token predictor)
  - **autoregressive** generation of future tokens
  - **KV cache** during AR decode (inference-style path)

Capacity matched to JEPA predictor / Exp A (d=384, L=12, H=12).
Future length defaults to 16 (LM-style slots), not 256 spatial tubelet tokens —
AR over 256 steps would be impractical; Qwen also generates short token spans.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalBlock(nn.Module):
    def __init__(self, d: int, nhead: int, ff: int, dropout: float = 0.0):
        super().__init__()
        self.nhead = int(nhead)
        self.head_dim = d // nhead
        assert d % nhead == 0
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, d),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """
        Args:
            x: [B, T, D] — full sequence (no cache) or only **new** tokens (with cache)
            attn_mask: optional additive mask [T, S] or [B, H, T, S]
            kv_cache: (K, V) each [B, H, S_past, Dh]
        """
        B, T, D = x.shape
        h, dh = self.nhead, self.head_dim
        y = self.norm1(x)
        qkv = self.qkv(y).view(B, T, 3, h, dh).permute(2, 0, 3, 1, 4)  # 3,B,H,T,Dh
        q, k, v = qkv[0], qkv[1], qkv[2]
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        # SDPA: [B,H,T,Dh]
        # Build causal mask when no cache (full seq) or for new tokens vs past+new
        S = k.size(2)
        if attn_mask is None:
            # causal over [past | new]: query i (abs=past+i) sees keys 0..past+i
            past = S - T
            # [T,S] additive mask
            attn_mask = torch.zeros(T, S, device=x.device, dtype=q.dtype)
            if T == S and past == 0:
                attn_mask = torch.triu(
                    torch.full((T, S), float("-inf"), device=x.device, dtype=q.dtype), diagonal=1
                )
            else:
                # allow all past keys; among new tokens, standard causal
                # positions with key_idx > past + query_idx → -inf
                q_idx = torch.arange(T, device=x.device).view(T, 1) + past
                k_idx = torch.arange(S, device=x.device).view(1, S)
                attn_mask = torch.where(
                    k_idx > q_idx,
                    torch.full((), float("-inf"), device=x.device, dtype=q.dtype),
                    torch.zeros((), device=x.device, dtype=q.dtype),
                )
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.out_proj(attn))
        x = x + self.mlp(self.norm2(x))
        new_cache = (k, v) if use_cache else None
        return x, new_cache


class QwenStyleARDecoder(nn.Module):
    """Decoder-only causal LM over [vision_prefix | future_tokens], with KV cache AR."""

    def __init__(
        self,
        embed_dim: int = 1024,
        d_model: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_future_tokens: int = 16,
        max_prefix: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        d = int(d_model)
        ff = int(d * mlp_ratio)
        self.embed_dim = int(embed_dim)
        self.d_model = d
        self.num_future_tokens = int(num_future_tokens)
        self.max_prefix = int(max_prefix)

        self.vision_proj = nn.Linear(embed_dim, d)
        self.vision_norm = nn.LayerNorm(d)
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        self.future_embed = nn.Parameter(torch.zeros(1, self.num_future_tokens, d))
        self.time_mlp = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([_CausalBlock(d, num_heads, ff, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, embed_dim)
        # Next-token head in latent space (train AR consistency)
        self.pred_lat = nn.Linear(d, d)

        nn.init.trunc_normal_(self.bos, std=0.02)
        nn.init.trunc_normal_(self.future_embed, std=0.02)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _prefix(self, vision: torch.Tensor, anticipation_times: torch.Tensor) -> torch.Tensor:
        """vision [B,N,E] → prefix [B,1+N,d] = BOS(+time) + projected vision."""
        B, N, _ = vision.shape
        if N > self.max_prefix:
            # uniform subsample for LM-style context budget
            idx = torch.linspace(0, N - 1, self.max_prefix, device=vision.device).round().long()
            vision = vision.index_select(1, idx)
        mem = self.vision_norm(self.vision_proj(vision))
        t = (anticipation_times.float().view(B, 1, 1) / 6.0).clamp(0.0, 2.0)
        bos = self.bos.expand(B, -1, -1) + self.time_mlp(t)
        return torch.cat([bos, mem], dim=1)

    def forward_teacher(
        self, vision: torch.Tensor, anticipation_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parallel teacher-forcing (Qwen train-style). Returns future latents + aux loss."""
        B = vision.size(0)
        prefix = self._prefix(vision, anticipation_times)
        # Teacher inputs: future_embed positions 0..K-1 as inputs predicting shifted targets
        fut_in = self.future_embed.expand(B, -1, -1)
        seq = torch.cat([prefix, fut_in], dim=1)  # B, P+K, d
        P = prefix.size(1)
        K = self.num_future_tokens
        # causal mask on full sequence
        L = seq.size(1)
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=seq.device, dtype=seq.dtype), diagonal=1
        )
        x = seq
        for blk in self.blocks:
            x, _ = blk(x, attn_mask=causal, use_cache=False)
        fut_h = self.out_norm(x[:, P : P + K])
        future = self.out_proj(fut_h)
        # Aux: predict next future_embed from previous hidden (AR consistency)
        # input fut_in[:, :-1] should predict fut_in[:, 1:] in d-space via pred_lat
        if K > 1:
            pred = self.pred_lat(fut_h[:, :-1])
            tgt = fut_in[:, 1:].detach()
            aux = F.mse_loss(pred, tgt)
        else:
            aux = future.new_zeros(())
        return future, aux

    def forward_ar_kv(
        self, vision: torch.Tensor, anticipation_times: torch.Tensor
    ) -> torch.Tensor:
        """Autoregressive decode of K future tokens with **KV cache** (Qwen infer-style)."""
        B = vision.size(0)
        prefix = self._prefix(vision, anticipation_times)
        # Prefill: run prefix once, build cache
        caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.blocks)
        x = prefix
        for i, blk in enumerate(self.blocks):
            x, caches[i] = blk(x, kv_cache=None, use_cache=True)

        futures = []
        # First AR step: feed future_embed[0] (or bos already in prefix — use slot 0)
        prev = self.future_embed[:, :1, :].expand(B, -1, -1)  # B,1,d
        for t in range(self.num_future_tokens):
            x = prev
            for i, blk in enumerate(self.blocks):
                x, caches[i] = blk(x, kv_cache=caches[i], use_cache=True)
            h = self.out_norm(x)
            futures.append(self.out_proj(h))
            if t + 1 < self.num_future_tokens:
                # next input = learned slot embed t+1 (+ optional pred from h)
                nxt = self.future_embed[:, t + 1 : t + 2, :].expand(B, -1, -1)
                nxt = nxt + self.pred_lat(h)
                prev = nxt
        return torch.cat(futures, dim=1)  # B,K,E

    def forward(
        self,
        vision: torch.Tensor,
        anticipation_times: torch.Tensor,
        *,
        mode: str = "teacher",
    ):
        if mode == "ar_kv":
            return self.forward_ar_kv(vision, anticipation_times), vision.new_zeros(())
        return self.forward_teacher(vision, anticipation_times)


class QwenStyleStreamModel(nn.Module):
    def __init__(
        self,
        base,
        decoder: QwenStyleARDecoder,
        pruner=None,
        prune_threshold: int = 4096,
        decode_mode: str = "teacher",
        train_encoder: bool = False,
    ):
        super().__init__()
        self.base = base
        self.decoder = decoder
        self.pruner = pruner
        self.prune_threshold = int(prune_threshold)
        self.decode_mode = str(decode_mode)
        self.train_encoder = bool(train_encoder)
        self.last_aux = None

    def forward(self, x, anticipation_times):
        # Frozen encoder: skip autograd graph. LoRA / block FT needs grads.
        if self.train_encoder and self.training:
            x_full = self.base.encoder(x)
        else:
            with torch.no_grad():
                x_full = self.base.encoder(x)
        B, N, D_full = x_full.size()
        embed_dim = self.base.encoder.embed_dim
        if self.pruner is not None and N > self.prune_threshold:
            x_full, _ = self.pruner.prune(x_full)
        x_ctx = x_full[:, :, -embed_dim:] if x_full.size(-1) > embed_dim else x_full
        mode = self.decode_mode
        # During training prefer teacher forcing; caller may set decode_mode=ar_kv for val
        future, aux = self.decoder(x_ctx, anticipation_times, mode=mode)
        self.last_aux = aux
        return torch.cat([x_ctx, future], dim=1)
