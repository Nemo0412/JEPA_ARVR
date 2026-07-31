"""Causal Transformer decoder on frozen V-JEPA encoder latents (experiment A).

Matches V-JEPA ``vit_predictor`` capacity (d=384, depth=12, heads=12, mlp=4×)
but uses a standard causal / cross-attn decoder instead of the anticipative
JEPA predictor. Same interface as stream MTP: ``forward(x, anticipation_times)
→ tokens`` for CommunicatingMLPMTPClassifier.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CausalFutureDecoder(nn.Module):
    """Cross-attn decoder: future queries (causal) attend to encoder memory."""

    def __init__(
        self,
        embed_dim: int = 1024,
        predictor_embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_future_tokens: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        d = int(predictor_embed_dim)
        ff = int(d * mlp_ratio)
        self.embed_dim = int(embed_dim)
        self.predictor_embed_dim = d
        self.num_future_tokens = int(num_future_tokens)

        self.memory_proj = nn.Linear(embed_dim, d)
        self.memory_norm = nn.LayerNorm(d)
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_future_tokens, d))
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=int(num_heads),
            dim_feedforward=ff,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(depth))
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, embed_dim)

        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, memory: torch.Tensor, anticipation_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: [B, N, embed_dim] encoder tokens (frozen latents)
            anticipation_times: [B] seconds
        Returns:
            future tokens [B, num_future_tokens, embed_dim]
        """
        B = memory.size(0)
        mem = self.memory_norm(self.memory_proj(memory))
        # Normalize anticipation by 6s so +2/+4/+6 sit in a stable range.
        t = (anticipation_times.float().view(B, 1, 1) / 6.0).clamp(0.0, 2.0)
        t_emb = self.time_mlp(t)  # B,1,d
        tgt = self.query_tokens.expand(B, -1, -1) + t_emb
        # Causal mask among future queries (standard decoder autoregression).
        n = self.num_future_tokens
        causal = torch.triu(
            torch.full((n, n), float("-inf"), device=memory.device, dtype=mem.dtype),
            diagonal=1,
        )
        out = self.decoder(tgt, mem, tgt_mask=causal)
        out = self.out_proj(self.out_norm(out))
        return out


class CausalDecoderStreamModel(nn.Module):
    """Frozen V-JEPA encoder (+ optional prune) → trainable causal future decoder."""

    def __init__(
        self,
        base: nn.Module,
        decoder: CausalFutureDecoder,
        pruner=None,
        prune_threshold: int = 4096,
    ):
        super().__init__()
        self.base = base
        self.decoder = decoder
        self.pruner = pruner
        self.prune_threshold = int(prune_threshold)
        self.embed_dim = int(base.encoder.embed_dim)

    def forward(self, x: torch.Tensor, anticipation_times: torch.Tensor) -> torch.Tensor:
        core = self.base
        with torch.no_grad():
            x_full = core.encoder(x)
        B, N, D_full = x_full.size()
        embed_dim = core.encoder.embed_dim
        if self.pruner is not None and N > self.prune_threshold:
            x_full, _ = self.pruner.prune(x_full)
            B, N, D_full = x_full.size()
        use_hierarchical = D_full > embed_dim
        x_ctx = x_full[:, :, -embed_dim:] if use_hierarchical else x_full
        # Decoder sees last-layer latents only (fair vs classifier pooler dim).
        future = self.decoder(x_ctx, anticipation_times)
        return torch.cat([x_ctx, future], dim=1)


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
