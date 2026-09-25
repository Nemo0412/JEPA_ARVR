"""GrayFreeze: freeze old-old (gray) logits in every probe self-attn block.

Steady-state probe step
-----------------------
Cache holds 128 frames. 34 new frames arrive. Prune 34 from the cache (keep 94).
Probe sees ``94 + 34 = 128`` tokens.

In each AttentivePooler self-attn block the QK matrix is

        K_old (94)     K_new (34)
      ┌─────────────┬─────────────┐
Q_old │  gray       │  old→new    │
      │  FROZEN     │  compute    │
      ├─────────────┼─────────────┤
Q_new │  new→old    │  new→new    │
      │  compute    │  compute    │
      └─────────────┴─────────────┘

Gray logits are not materialized. They are recovered by *evicting* the dropped
34 keys from the previous block's cached (out, log-sum-exp) and *merging* the
new 34 keys — online softmax, same as FlashAttention's incremental combine.

Old tokens still attend to the new 34 (softmax support is the full 128).
New tokens attend to all 128 via SDPA. The final 3-query cross-attn is unchanged.

Block 0 is exact vs full self-attn on the same 94+34 encoder tokens (QKV is a
per-token linear). Later blocks are approximate: they reuse last step's QKV for
the kept tokens instead of recomputing from this step's updated residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def fused_attn_lse(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """SDPA + log-sum-exp. q/k/v: [B,H,S,D] → out [B,H,S,D], lse [B,H,S] fp32.

    Uses the fused CUDA kernel (flash / memory-efficient). Falls back to a
    chunked matmul on CPU or if the fused op rejects the shapes.
    """
    if q.is_cuda:
        kwargs = {"scale": float(scale)}
        try:
            try:
                out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                    q, k, v, 0.0, False, False, **kwargs
                )
            except Exception:
                out, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
                    q, k, v, None, True, 0.0, False, **kwargs
                )
            lse = lse.float()
            nq = q.size(2)
            if lse.dim() == 2:
                if lse.size(0) == q.size(0):
                    lse = lse.unsqueeze(1).expand(-1, q.size(1), -1)
                else:
                    lse = lse.unsqueeze(0)
            if lse.size(-1) != nq:
                lse = lse[..., :nq]
            if tuple(lse.shape[:3]) != (q.size(0), q.size(1), nq) and lse.shape == (
                q.size(0),
                nq,
                q.size(1),
            ):
                lse = lse.transpose(1, 2)
            return out, lse
        except Exception:
            return online_attn(q, k, v, scale)
    return online_attn(q, k, v, scale)


def _qkv(attn, x: torch.Tensor):
    b, n, c = x.shape
    qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, c // attn.num_heads).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def _proj(attn, o: torch.Tensor, b: int, n: int, c: int) -> torch.Tensor:
    y = o.transpose(1, 2).reshape(b, n, c)
    return attn.proj_drop(attn.proj(y))


def _gather_bn(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather token dim of [B,H,N,...] with idx [B,K]."""
    extra = t.shape[3:]
    b, h, _n = t.shape[:3]
    k = idx.size(1)
    idx_e = idx[:, None, :].reshape(b, 1, k, *([1] * len(extra))).expand(b, h, k, *extra)
    return t.gather(2, idx_e)


def online_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q,k,v: [B,H,N,D] → (out [B,H,Nq,D] in q dtype, lse [B,H,Nq] fp32)."""
    qf, kf, vf = q.float(), k.float(), v.float()
    b, h, nq, d = qf.shape
    nk = kf.size(2)
    m = qf.new_full((b, h, nq), float("-inf"))
    se = qf.new_zeros(b, h, nq)
    acc = qf.new_zeros(b, h, nq, d)
    for c0 in range(0, nk, chunk):
        k_c = kf[:, :, c0 : c0 + chunk]
        v_c = vf[:, :, c0 : c0 + chunk]
        logits = torch.matmul(qf, k_c.transpose(-2, -1)) * scale
        m_c = logits.amax(dim=-1)
        m_new = torch.maximum(m, m_c)
        alpha = torch.exp(m - m_new)
        p = torch.exp(logits - m_new.unsqueeze(-1))
        acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, v_c)
        se = se * alpha + p.sum(dim=-1)
        m = m_new
    lse = m + torch.log(se.clamp_min(1e-20))
    o = acc / se.clamp_min(1e-20).unsqueeze(-1)
    return o.to(dtype=q.dtype), lse


def lse_only(q: torch.Tensor, k: torch.Tensor, scale: float, chunk: int = 256) -> torch.Tensor:
    qf, kf = q.float(), k.float()
    b, h, nq, _d = qf.shape
    m = qf.new_full((b, h, nq), float("-inf"))
    se = qf.new_zeros(b, h, nq)
    nk = kf.size(2)
    for c0 in range(0, nk, chunk):
        logits = torch.matmul(qf, kf[:, :, c0 : c0 + chunk].transpose(-2, -1)) * scale
        m_c = logits.amax(dim=-1)
        m_new = torch.maximum(m, m_c)
        se = se * torch.exp(m - m_new) + torch.exp(logits - m_new.unsqueeze(-1)).sum(dim=-1)
        m = m_new
    return m + torch.log(se.clamp_min(1e-20))


def softmax_remove_keys(
    q: torch.Tensor,
    o_full: torch.Tensor,
    lse_full: torch.Tensor,
    k_drop: torch.Tensor,
    v_drop: torch.Tensor,
    scale: float,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evict dropped keys from a cached softmax. q attends to the *kept* queries."""
    o_drop, lse_drop = fused_attn_lse(q, k_drop, v_drop, scale)
    lse_full_f = lse_full.float()
    lse_drop_f = lse_drop.float()
    delta = (lse_drop_f - lse_full_f).clamp(max=0.0)
    ratio = torch.exp(delta).clamp(max=1.0 - 1e-4)
    lse_keep = lse_full_f + torch.log1p(-ratio)
    zf = torch.exp(lse_full_f - lse_keep)
    zd = torch.exp(lse_drop_f - lse_keep)
    o_keep = o_full.float() * zf.unsqueeze(-1) - o_drop.float() * zd.unsqueeze(-1)
    return o_keep.to(dtype=q.dtype), lse_keep


def softmax_add_keys(
    q: torch.Tensor,
    o_old: torch.Tensor,
    lse_old: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    scale: float,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    o_new, lse_new = fused_attn_lse(q, k_new, v_new, scale)
    lse_old_f = lse_old.float()
    lse = torch.logaddexp(lse_old_f, lse_new.float())
    o = o_old.float() * torch.exp(lse_old_f - lse).unsqueeze(-1) + o_new.float() * torch.exp(
        lse_new.float() - lse
    ).unsqueeze(-1)
    return o.to(dtype=q.dtype), lse


@dataclass
class BlockCache:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    lse: torch.Tensor

    def gather(self, idx: torch.Tensor) -> "BlockCache":
        return BlockCache(
            q=_gather_bn(self.q, idx),
            k=_gather_bn(self.k, idx),
            v=_gather_bn(self.v, idx),
            o=_gather_bn(self.o, idx),
            lse=_gather_bn(self.lse, idx),
        )


def slot_importance_from_qk(
    q: torch.Tensor, k: torch.Tensor, scale: float, gp: int, chunk: int = 256
) -> torch.Tensor:
    """Column-sum received mass, mean-pooled to slots. q,k: [B,H,N,D] → [B, n_slots]."""
    b, _h, n, _d = q.shape
    imp = torch.zeros(b, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)
    qf = q.float()
    for c0 in range(0, n, chunk):
        logits = torch.matmul(qf[:, :, c0 : c0 + chunk], k_t) * scale
        imp += logits.softmax(dim=-1).sum(dim=2).sum(dim=1)
    n_slots = n // gp
    return imp.view(b, n_slots, gp).mean(dim=-1)


def keep_and_drop_from_slots(
    slot_scores: torch.Tensor, keep_slots: int, gp: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """slot_scores [B, n_slots] → keep_tok, drop_tok [B, n_tok]."""
    b, n_slots = slot_scores.shape
    keep_s = slot_scores.topk(keep_slots, dim=1).indices.sort(dim=1).values
    spatial = torch.arange(gp, device=slot_scores.device)
    keep_tok = (keep_s.unsqueeze(-1) * gp + spatial).reshape(b, -1)
    n_tok = n_slots * gp
    mask = torch.ones(b, n_tok, dtype=torch.bool, device=slot_scores.device)
    mask.scatter_(1, keep_tok, False)
    drop_tok = mask.nonzero(as_tuple=False)[:, 1].view(b, n_tok - keep_tok.size(1))
    return keep_tok, drop_tok


def fifo_keep_drop(n_tok: int, drop_tok: int, device, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    keep = torch.arange(drop_tok, n_tok, device=device).unsqueeze(0).expand(batch, -1)
    drop = torch.arange(0, drop_tok, device=device).unsqueeze(0).expand(batch, -1)
    return keep, drop


def full_block_forward(blk, x: torch.Tensor, chunk: int = 256) -> tuple[torch.Tensor, BlockCache]:
    """One probe self-attn block with SDPA, also recording GrayFreeze cache."""
    b, n, c = x.shape
    attn = blk.attn
    xn = blk.norm1(x)
    q, k, v = _qkv(attn, xn)
    o, lse = fused_attn_lse(q, k, v, float(attn.scale))
    y = _proj(attn, o, b, n, c)
    x = x + blk.drop_path(y)
    x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
    return x, BlockCache(q=q, k=k, v=v, o=o, lse=lse)


def grayfreeze_block_forward(
    blk,
    x_old: torch.Tensor,
    x_new: torch.Tensor,
    cache: BlockCache,
    keep_idx: torch.Tensor,
    drop_idx: torch.Tensor,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, BlockCache]:
    """Frozen gray logits for ``x_old``; compute old→new and new→all."""
    b, n_old, c = x_old.shape
    n_new = x_new.size(1)
    attn = blk.attn
    scale = float(attn.scale)

    q_new, k_new, v_new = _qkv(attn, blk.norm1(x_new))

    q_old = _gather_bn(cache.q, keep_idx)
    k_keep = _gather_bn(cache.k, keep_idx)
    v_keep = _gather_bn(cache.v, keep_idx)
    o_prev = _gather_bn(cache.o, keep_idx)
    lse_prev = _gather_bn(cache.lse, keep_idx)
    k_drop = _gather_bn(cache.k, drop_idx)
    v_drop = _gather_bn(cache.v, drop_idx)

    o_keep, lse_keep = softmax_remove_keys(q_old, o_prev, lse_prev, k_drop, v_drop, scale, chunk=chunk)
    o_old, lse_old = softmax_add_keys(q_old, o_keep, lse_keep, k_new, v_new, scale, chunk=chunk)

    k_all = torch.cat([k_keep, k_new], dim=2)
    v_all = torch.cat([v_keep, v_new], dim=2)
    o_new, lse_new = fused_attn_lse(q_new, k_all, v_all, scale)

    o = torch.cat([o_old, o_new], dim=2)
    y = _proj(attn, o, b, n_old + n_new, c)
    x = torch.cat([x_old, x_new], dim=1) + blk.drop_path(y)
    x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
    x_old_out, x_new_out = x[:, :n_old], x[:, n_old:]

    new_cache = BlockCache(
        q=torch.cat([q_old, q_new], dim=2),
        k=k_all,
        v=v_all,
        o=o,
        lse=torch.cat([lse_old, lse_new], dim=2),
    )
    return x_old_out, x_new_out, new_cache


def cross_attn_pool(pooler, x: torch.Tensor) -> torch.Tensor:
    q = pooler.query_tokens.repeat(len(x), 1, 1)
    return pooler.cross_attention_block(q, x)


def full_probe_pool(pooler, tokens: torch.Tensor, chunk: int = 256) -> tuple[torch.Tensor, list[BlockCache]]:
    """Full AttentivePooler forward, filling a GrayFreeze cache for every self-attn block."""
    x = tokens
    caches: list[BlockCache] = []
    if pooler.blocks is not None:
        for blk in pooler.blocks:
            x, cache = full_block_forward(blk, x, chunk=chunk)
            caches.append(cache)
    return cross_attn_pool(pooler, x), caches


def grayfreeze_probe_pool(
    pooler,
    x_old: torch.Tensor,
    x_new: torch.Tensor,
    caches: list[BlockCache],
    keep_idx: torch.Tensor,
    drop_idx: torch.Tensor,
    chunk: int = 256,
) -> tuple[torch.Tensor, list[BlockCache]]:
    """All self-attn blocks GrayFreeze; final cross-attn over the updated 128 tokens."""
    if pooler.blocks is None or len(pooler.blocks) != len(caches):
        raise RuntimeError("GrayFreeze needs AttentivePooler.blocks matching the cache")
    new_caches: list[BlockCache] = []
    for blk, cache in zip(pooler.blocks, caches):
        x_old, x_new, cache = grayfreeze_block_forward(
            blk, x_old, x_new, cache, keep_idx, drop_idx, chunk=chunk
        )
        new_caches.append(cache)
    x = torch.cat([x_old, x_new], dim=1)
    return cross_attn_pool(pooler, x), new_caches


def classify_pooled(mtp_clf, pooled: torch.Tensor) -> dict[float, dict[str, torch.Tensor]]:
    """MTP heads from an already-pooled [B, 3, D] (or [B, 1, D]) feature."""
    if getattr(mtp_clf.base, "action_only", False) or getattr(mtp_clf.base, "num_verb_classes", 1) == 0:
        fa = pooled[:, 0, :]
        fv = fn = fa
    else:
        fv, fn, fa = pooled[:, 0, :], pooled[:, 1, :], pooled[:, 2, :]
    n_h = len(mtp_clf.horizons_sec)
    feat_v = [fv for _ in range(n_h)]
    feat_n = [fn for _ in range(n_h)]
    feat_a = [fa for _ in range(n_h)]
    shared = mtp_clf._communicate(feat_a)
    outputs: dict[float, dict[str, torch.Tensor]] = {}
    action_only = getattr(mtp_clf.base, "action_only", False) or getattr(mtp_clf.base, "num_verb_classes", 1) == 0
    for h, z_a, f_v, f_n, f_a in zip(mtp_clf.horizons_sec, shared, feat_v, feat_n, feat_a):
        delta = z_a - f_a
        if action_only:
            outputs[float(h)] = dict(action=mtp_clf.base.action_classifier(z_a))
        else:
            outputs[float(h)] = dict(
                verb=mtp_clf.base.verb_classifier(f_v + delta),
                noun=mtp_clf.base.noun_classifier(f_n + delta),
                action=mtp_clf.base.action_classifier(z_a),
            )
    return outputs
