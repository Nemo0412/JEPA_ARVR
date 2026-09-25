"""Streaming Gaze+IMU encoder with 128-frame KV cache and attention prune.

No predictor. Protocol:

  * Cache holds 128 frames (64 tubelets). Each streaming step takes 34 new
    frames (17 tubelets).
  * After a 128-frame window is encoded, last-block received-attention is
    mean-pooled to slots. The 17 lowest-score slots (34 frames) are recorded.
  * When the next 34 frames arrive they **replace** those low-score slots:
    kept 94-frame K/V stay in cache (original RoPE), new 34 attend to the
    remaining keys, then the packed 94+34=128 is rescored for the next step.
  * Gaze+pose enter through the 5ch BinaryMapInputAdapter on each chunk.
  * IMU late CA is applied on the packed 64 video slots (aligned by slot id).
  * Probe sees fused encoder tokens only.

FIFO (drop oldest 34) is the same cache path without the attention ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.modules import rotate_queries_or_keys

PruneMode = Literal["attn", "fifo"]

CACHE_FRAMES = 128
NEW_FRAMES = 34


def strip_hier(tokens: torch.Tensor, embed_dim: int) -> torch.Tensor:
    return tokens[:, :, -embed_dim:] if tokens.size(-1) > embed_dim else tokens


def rope_qkv(attn, x: torch.Tensor, pos_ids: torch.Tensor, h_patches: int, w_patches: int):
    """Q/K/V after RoPE. x: [B,N,C], pos_ids: [B,N] global token indices."""
    b, n, _c = x.size()
    qkv = attn.qkv(x).unflatten(-1, (3, attn.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    mask = pos_ids.unsqueeze(1).expand(b, attn.num_heads, n)
    d_mask, h_mask, w_mask = attn.separate_positions(mask, h_patches, w_patches)
    s = 0
    qd = rotate_queries_or_keys(q[..., s : s + attn.d_dim], pos=d_mask)
    kd = rotate_queries_or_keys(k[..., s : s + attn.d_dim], pos=d_mask)
    s += attn.d_dim
    qh = rotate_queries_or_keys(q[..., s : s + attn.h_dim], pos=h_mask)
    kh = rotate_queries_or_keys(k[..., s : s + attn.h_dim], pos=h_mask)
    s += attn.h_dim
    qw = rotate_queries_or_keys(q[..., s : s + attn.w_dim], pos=w_mask)
    kw = rotate_queries_or_keys(k[..., s : s + attn.w_dim], pos=w_mask)
    s += attn.w_dim
    if s < attn.head_dim:
        q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
        k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
    else:
        q = torch.cat([qd, qh, qw], dim=-1)
        k = torch.cat([kd, kh, kw], dim=-1)
    return q, k, v


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


def gather_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    b, _n, d = tokens.shape
    k = idx.size(1)
    return tokens.gather(1, idx.unsqueeze(-1).expand(b, k, d))


def _sincos_1d(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """pos: [B, N] integer positions → [B, N, dim] frozen 1D sincos."""
    half = dim // 2
    omega = torch.arange(half, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / max(half, 1)))
    out = pos.float().unsqueeze(-1) * omega
    return torch.cat([out.sin(), out.cos()], dim=-1)


def dense_time_rank(slot_ids: torch.Tensor) -> torch.Tensor:
    """Re-index current cache slots to {0..S-1} by original time. 0 = oldest survivor.

    After repeated prune the stream slot ids have holes and keep growing. Probe
    PE must stay in a fixed range, so we only encode *order among tokens that
    are still in the cache*, not the absolute stream index.
    """
    return slot_ids.argsort(dim=1).argsort(dim=1)


def probe_positional_encoding(
    *,
    slot_ids: torch.Tensor,
    embed_dim: int,
    gp: int,
    pos_ids: torch.Tensor | None = None,
    grid_size: int = 16,
    mode: str = "rel_rank",
) -> torch.Tensor:
    """3D sincos PE at probe entry for a packed (pruned) KV cache.

    Spatial (h, w) always comes from the original patch inside the slot.
    Temporal axis depends on ``mode``:

      rel_rank     (default) dense 0..S-1 among survivors. Bounded, stable
                   under repeated delete+append. A dropped middle slot shifts
                   later tokens left by 1 — like a compressed sliding window.
      abs_stream   raw stream slot id. Holes + unbounded growth; wrong once
                   the cache has been pruned many times.

    Encoder RoPE is already baked into the cached tokens at encode time; this
    PE is only for AttentivePooler, which has none of its own.
    """
    b, n_slots = slot_ids.shape
    n = n_slots * int(gp)
    if mode == "rel_rank":
        t_slot = dense_time_rank(slot_ids)
    elif mode == "abs_stream":
        t_slot = slot_ids
    else:
        raise ValueError(f"unknown probe PE mode {mode!r}")
    t = t_slot.repeat_interleave(int(gp), dim=1)
    if pos_ids is not None:
        spatial = pos_ids % int(gp)
    else:
        spatial = torch.arange(int(gp), device=slot_ids.device)
        spatial = spatial.view(1, 1, -1).expand(b, n_slots, -1).reshape(b, n)
    h = spatial // int(grid_size)
    w = spatial % int(grid_size)
    d_dim = embed_dim // 2
    hw_dim = embed_dim // 4
    pe = torch.cat(
        [_sincos_1d(t, d_dim), _sincos_1d(h, hw_dim), _sincos_1d(w, hw_dim)],
        dim=-1,
    )
    if pe.size(-1) < embed_dim:
        pe = torch.nn.functional.pad(pe, (0, embed_dim - pe.size(-1)))
    elif pe.size(-1) > embed_dim:
        pe = pe[..., :embed_dim]
    return pe


def add_probe_positional_encoding(
    tokens: torch.Tensor,
    slot_ids: torch.Tensor,
    gp: int,
    *,
    pos_ids: torch.Tensor | None = None,
    grid_size: int = 16,
    mode: str = "rel_rank",
) -> torch.Tensor:
    """tokens [B,N,D] + frozen 3D sincos. Default ``rel_rank`` for pruned caches."""
    pe = probe_positional_encoding(
        slot_ids=slot_ids,
        embed_dim=tokens.size(-1),
        gp=gp,
        pos_ids=pos_ids,
        grid_size=grid_size,
        mode=mode,
    )
    return tokens + pe.to(dtype=tokens.dtype)


def gather_bn(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather token dim of [B,H,N,...] with idx [B,K]."""
    extra = t.shape[3:]
    b, h, _n = t.shape[:3]
    k = idx.size(1)
    idx_e = idx[:, None, :].reshape(b, 1, k, *([1] * len(extra))).expand(b, h, k, *extra)
    return t.gather(2, idx_e)


def gather_kv(cache: list[torch.Tensor], keep_tok: torch.Tensor) -> list[torch.Tensor]:
    return [gather_bn(c, keep_tok) for c in cache]


def token_frame_ids_from_slots(slot_ids: torch.Tensor, gp: int) -> torch.Tensor:
    """slot_ids [B, S] → per-token frame/slot index [B, S*gp] (same id inside a slot)."""
    return slot_ids.repeat_interleave(int(gp), dim=1)


def apply_temporal_rope_qk(
    q: torch.Tensor, k: torch.Tensor, frame_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """1D temporal RoPE: Q'_p = R(p) Q_p, K'_p = R(p) K_p. V unchanged.

    q,k: [B,H,N,D]  frame_ids: [B,N] original Frame/slot Index (abs stream).
    θ_{p,m} = p · ω_m via rotate_queries_or_keys.
    """
    b, h, n, _d = q.shape
    if frame_ids.ndim != 2 or frame_ids.shape != (b, n):
        raise ValueError(f"frame_ids must be [B,N]={b,n}, got {tuple(frame_ids.shape)}")
    pos = frame_ids[:, None, :].expand(b, h, n).to(dtype=q.dtype)
    return rotate_queries_or_keys(q, pos=pos), rotate_queries_or_keys(k, pos=pos)


class ProbeTemporalRoPE:
    """Patch AttentivePooler self-attn (+ optional cross-attn K) with 1D temporal RoPE.

    Position = original Frame/slot Index carried through prune (abs_stream).
    Call ``set_frame_ids`` before each pooler forward. No new parameters.
    """

    def __init__(self, pooler: nn.Module, *, rope_cross_attn_k: bool = True):
        self.pooler = pooler
        self.rope_cross_attn_k = bool(rope_cross_attn_k)
        self._frame_ids: torch.Tensor | None = None
        self._hooks: list[tuple[nn.Module, object]] = []
        self._install()

    def set_frame_ids(self, frame_ids: torch.Tensor | None) -> None:
        """frame_ids [B, N] aligned with pooler input tokens; None disables RoPE."""
        self._frame_ids = frame_ids

    def remove(self) -> None:
        for mod, orig in self._hooks:
            mod.forward = orig  # type: ignore[method-assign]
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def _install(self) -> None:
        pooler = self.pooler
        if getattr(pooler, "blocks", None) is not None:
            for blk in pooler.blocks:
                self._patch_self_attn(blk.attn)
        if self.rope_cross_attn_k and getattr(pooler, "cross_attention_block", None) is not None:
            xblk = pooler.cross_attention_block
            xattn = getattr(xblk, "xattn", xblk)
            if hasattr(xattn, "kv") and hasattr(xattn, "q"):
                self._patch_cross_attn(xattn)

    def _patch_self_attn(self, attn: nn.Module) -> None:
        orig = attn.forward
        rope = self

        def _fwd(x, mask=None, attn_mask=None):
            b, n, c = x.shape
            qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, c // attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            fid = rope._frame_ids
            if fid is not None:
                if fid.shape != (b, n):
                    raise RuntimeError(f"frame_ids {tuple(fid.shape)} != tokens {(b, n)}")
                q, k = apply_temporal_rope_qk(q, k, fid)
            drop_p = float(getattr(attn, "proj_drop_prob", 0.0)) if attn.training else 0.0
            if attn_mask is not None or getattr(attn, "use_sdpa", True):
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=drop_p, is_causal=bool(getattr(attn, "is_causal", False)), attn_mask=attn_mask
                )
            else:
                a = (q @ k.transpose(-2, -1)) * attn.scale
                a = a.softmax(dim=-1)
                a = attn.attn_drop(a)
                y = a @ v
            y = y.transpose(1, 2).reshape(b, n, c)
            y = attn.proj(y)
            y = attn.proj_drop(y)
            return y

        attn.forward = _fwd  # type: ignore[method-assign]
        self._hooks.append((attn, orig))

    def _patch_cross_attn(self, xattn: nn.Module) -> None:
        orig = xattn.forward
        rope = self

        def _fwd(q, x):
            b, nq, c = q.shape
            qq = xattn.q(q).reshape(b, nq, xattn.num_heads, c // xattn.num_heads).permute(0, 2, 1, 3)
            b2, n, _ = x.shape
            kv = xattn.kv(x).reshape(b2, n, 2, xattn.num_heads, c // xattn.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            fid = rope._frame_ids
            if fid is not None:
                if fid.shape != (b2, n):
                    raise RuntimeError(f"frame_ids {tuple(fid.shape)} != cross tokens {(b2, n)}")
                # Queries are learnable (no Frame ID); only rotate K.
                pos = fid[:, None, :].expand(b2, xattn.num_heads, n).to(dtype=k.dtype)
                k = rotate_queries_or_keys(k, pos=pos)
            if getattr(xattn, "use_sdpa", True):
                out = F.scaled_dot_product_attention(qq, k, v)
            else:
                a = (qq @ k.transpose(-2, -1)) * xattn.scale
                a = a.softmax(dim=-1)
                out = a @ v
            return out.transpose(1, 2).reshape(b, nq, c)

        xattn.forward = _fwd  # type: ignore[method-assign]
        self._hooks.append((xattn, orig))


@dataclass
class StreamKVState:
    tokens: torch.Tensor
    cache_k: list[torch.Tensor]
    cache_v: list[torch.Tensor]
    last_q: torch.Tensor
    last_k: torch.Tensor
    pos: torch.Tensor
    slot_ids: torch.Tensor
    slot_scores: torch.Tensor
    next_slot: int


class StreamKVAttnPruneEncoder:
    """Per-layer encoder K/V cache with slot-level attention prune."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        gp: int,
        tubelet_size: int = 2,
        cache_frames: int = CACHE_FRAMES,
        new_frames: int = NEW_FRAMES,
        chunk: int = 256,
    ):
        if cache_frames % tubelet_size or new_frames % tubelet_size:
            raise ValueError("cache_frames and new_frames must be multiples of tubelet_size")
        self.encoder = encoder
        self.gp = int(gp)
        self.tubelet_size = int(tubelet_size)
        self.cache_frames = int(cache_frames)
        self.new_frames = int(new_frames)
        self.n_slots = self.cache_frames // self.tubelet_size
        self.drop_slots = self.new_frames // self.tubelet_size
        self.keep_slots = self.n_slots - self.drop_slots
        self.chunk = int(chunk)
        self.embed_dim = int(encoder.embed_dim)
        self.scale = float(encoder.blocks[-1].attn.scale)

    def _patch_hw(self, clips: torch.Tensor) -> tuple[int, int]:
        _b, _c, _t, h, w = clips.shape
        ps = int(self.encoder.patch_size)
        return h // ps, w // ps

    def _encode_full(self, clips: torch.Tensor, pos: torch.Tensor) -> tuple[
        torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        encoder = self.encoder
        b = clips.size(0)
        gh, gw = self._patch_hw(clips)
        x = encoder.patch_embed(clips)
        n = x.size(1)
        cache_k, cache_v = [], []
        last_q = last_k = None
        for li, blk in enumerate(encoder.blocks):
            q, k, v = rope_qkv(blk.attn, blk.norm1(x), pos, gh, gw)
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            y = y.transpose(1, 2).reshape(b, n, -1)
            y = blk.attn.proj_drop(blk.attn.proj(y))
            x = x + blk.drop_path(y)
            x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
            cache_k.append(k)
            cache_v.append(v)
            if li == len(encoder.blocks) - 1:
                last_q, last_k = q, k
        if encoder.norm is not None:
            x = encoder.norm(x)
        assert last_q is not None and last_k is not None
        return strip_hier(x, self.embed_dim), cache_k, cache_v, last_q, last_k

    def _encode_new(
        self,
        clips: torch.Tensor,
        cache_k: list[torch.Tensor],
        cache_v: list[torch.Tensor],
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        encoder = self.encoder
        b = clips.size(0)
        gh, gw = self._patch_hw(clips)
        x = encoder.patch_embed(clips)
        n_new = x.size(1)
        new_k, new_v = [], []
        last_q = last_k = None
        for li, blk in enumerate(encoder.blocks):
            q, k, v = rope_qkv(blk.attn, blk.norm1(x), pos, gh, gw)
            k_all = torch.cat([cache_k[li], k], dim=2)
            v_all = torch.cat([cache_v[li], v], dim=2)
            y = F.scaled_dot_product_attention(q, k_all, v_all, dropout_p=0.0, is_causal=False)
            y = y.transpose(1, 2).reshape(b, n_new, -1)
            y = blk.attn.proj_drop(blk.attn.proj(y))
            x = x + blk.drop_path(y)
            x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
            new_k.append(k)
            new_v.append(v)
            if li == len(encoder.blocks) - 1:
                last_q, last_k = q, k
        if encoder.norm is not None:
            x = encoder.norm(x)
        assert last_q is not None and last_k is not None
        return strip_hier(x, self.embed_dim), new_k, new_v, last_q, last_k

    def _score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return slot_importance_from_qk(q, k, self.scale, self.gp, chunk=self.chunk)

    def fill(self, clips: torch.Tensor, slot_start: int = 0) -> StreamKVState:
        if clips.size(2) != self.cache_frames:
            raise ValueError(f"fill expects T={self.cache_frames}, got {clips.size(2)}")
        b = clips.size(0)
        n_tok = self.n_slots * self.gp
        device = clips.device
        pos = torch.arange(slot_start * self.gp, slot_start * self.gp + n_tok, device=device)
        pos = pos.unsqueeze(0).expand(b, -1)
        tokens, cache_k, cache_v, last_q, last_k = self._encode_full(clips, pos)
        slot_ids = torch.arange(slot_start, slot_start + self.n_slots, device=device)
        slot_ids = slot_ids.unsqueeze(0).expand(b, -1)
        return StreamKVState(
            tokens=tokens,
            cache_k=cache_k,
            cache_v=cache_v,
            last_q=last_q,
            last_k=last_k,
            pos=pos,
            slot_ids=slot_ids,
            slot_scores=self._score(last_q, last_k),
            next_slot=slot_start + self.n_slots,
        )

    def _select(self, state: StreamKVState, mode: PruneMode) -> tuple[torch.Tensor, torch.Tensor]:
        n_tok = state.tokens.size(1)
        drop_tok_n = self.drop_slots * self.gp
        if mode == "fifo":
            return fifo_keep_drop(n_tok, drop_tok_n, state.tokens.device, batch=state.tokens.size(0))
        return keep_and_drop_from_slots(state.slot_scores, self.keep_slots, self.gp)

    def step(self, state: StreamKVState, clips_new: torch.Tensor, mode: PruneMode = "attn") -> StreamKVState:
        if clips_new.size(2) != self.new_frames:
            raise ValueError(f"step expects T={self.new_frames}, got {clips_new.size(2)}")
        b = clips_new.size(0)
        keep_tok, _drop_tok = self._select(state, mode)
        old_tok = gather_tokens(state.tokens, keep_tok)
        cache_k = gather_kv(state.cache_k, keep_tok)
        cache_v = gather_kv(state.cache_v, keep_tok)
        last_q_keep = gather_bn(state.last_q, keep_tok)
        last_k_keep = gather_bn(state.last_k, keep_tok)
        pos_keep = state.pos.gather(1, keep_tok)
        keep_slots_idx = keep_tok[:, :: self.gp] // self.gp
        slot_keep = state.slot_ids.gather(1, keep_slots_idx)

        n_new = self.drop_slots * self.gp
        pos_new = torch.arange(
            state.next_slot * self.gp,
            state.next_slot * self.gp + n_new,
            device=clips_new.device,
        ).unsqueeze(0).expand(b, -1)
        new_tok, new_k, new_v, last_q_new, last_k_new = self._encode_new(
            clips_new, cache_k, cache_v, pos_new
        )
        slot_new = torch.arange(
            state.next_slot, state.next_slot + self.drop_slots, device=clips_new.device
        ).unsqueeze(0).expand(b, -1)

        tokens = torch.cat([old_tok, new_tok], dim=1)
        cache_k = [torch.cat([ck, nk], dim=2) for ck, nk in zip(cache_k, new_k)]
        cache_v = [torch.cat([cv, nv], dim=2) for cv, nv in zip(cache_v, new_v)]
        last_q = torch.cat([last_q_keep, last_q_new], dim=2)
        last_k = torch.cat([last_k_keep, last_k_new], dim=2)
        pos = torch.cat([pos_keep, pos_new], dim=1)
        slot_ids = torch.cat([slot_keep, slot_new], dim=1)
        return StreamKVState(
            tokens=tokens,
            cache_k=cache_k,
            cache_v=cache_v,
            last_q=last_q,
            last_k=last_k,
            pos=pos,
            slot_ids=slot_ids,
            slot_scores=self._score(last_q, last_k),
            next_slot=state.next_slot + self.drop_slots,
        )


def pack_imu_by_slots(
    imu: torch.Tensor,
    imu_len: torch.Tensor,
    slot_ids: torch.Tensor,
    tubelet_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather IMU frames belonging to ``slot_ids``. imu: [B,T,K,6], slot_ids: [B,S]."""
    b, _t, k, d = imu.shape
    s = slot_ids.size(1)
    frame_idx = torch.stack([slot_ids * tubelet_size, slot_ids * tubelet_size + (tubelet_size - 1)], dim=-1)
    frame_idx = frame_idx.reshape(b, s * tubelet_size).clamp(0, imu.size(1) - 1)
    idx = frame_idx.view(b, s * tubelet_size, 1, 1).expand(b, s * tubelet_size, k, d)
    imu_p = imu.gather(1, idx)
    len_idx = frame_idx
    imu_len_p = imu_len.gather(1, len_idx)
    return imu_p, imu_len_p


class GazeImuStreamKVModel(nn.Module):
    """5ch gaze+pose adapter → streaming encoder KV prune → IMU CA. No predictor."""

    def __init__(
        self,
        concat_ca: nn.Module,
        *,
        cache_frames: int = CACHE_FRAMES,
        new_frames: int = NEW_FRAMES,
        keep_aux: bool = False,
        chunk: int = 256,
    ):
        super().__init__()
        self.concat_ca = concat_ca
        core = concat_ca.base_model
        gp = int(core.grid_size**2)
        self.stream = StreamKVAttnPruneEncoder(
            core.encoder,
            gp=gp,
            tubelet_size=int(core.tubelet_size),
            cache_frames=cache_frames,
            new_frames=new_frames,
            chunk=chunk,
        )
        self.keep_aux = bool(keep_aux)
        self.embed_dim = int(core.encoder.embed_dim)
        self.cache_frames = int(cache_frames)
        self.new_frames = int(new_frames)

    def _adapt(self, clips: torch.Tensor, aux_map: torch.Tensor | None) -> torch.Tensor:
        if aux_map is None:
            return clips
        return self.concat_ca.input_adapter(clips, aux_map)

    def _fuse(self, tokens: torch.Tensor, imu_batch) -> torch.Tensor:
        tri = self.concat_ca.tri
        prev = bool(tri.fusion_cfg.get("keep_aux_tokens_in_predictor", False))
        tri.fusion_cfg["keep_aux_tokens_in_predictor"] = self.keep_aux
        try:
            fused = tri._fuse_for_predictor(tokens, gaze_map=None, imu_batch=imu_batch)
        finally:
            tri.fusion_cfg["keep_aux_tokens_in_predictor"] = prev
        return strip_hier(fused, self.embed_dim)

    def fill(
        self,
        clips: torch.Tensor,
        aux_map: torch.Tensor | None = None,
        slot_start: int = 0,
    ) -> StreamKVState:
        return self.stream.fill(self._adapt(clips, aux_map), slot_start=slot_start)

    def step(
        self,
        state: StreamKVState,
        clips_new: torch.Tensor,
        aux_new: torch.Tensor | None = None,
        mode: PruneMode = "attn",
    ) -> StreamKVState:
        return self.stream.step(state, self._adapt(clips_new, aux_new), mode=mode)

    def tokens_from_state(
        self,
        state: StreamKVState,
        imu: torch.Tensor | None = None,
        imu_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = state.tokens
        if imu is None or imu_len is None:
            return tokens
        imu_p, len_p = pack_imu_by_slots(
            imu, imu_len, state.slot_ids, self.stream.tubelet_size
        )
        return self._fuse(tokens, (imu_p, len_p))

    def forward_window(
        self,
        clips: torch.Tensor,
        aux_map: torch.Tensor | None = None,
        imu_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
        mode: PruneMode = "attn",
    ) -> torch.Tensor:
        """Fill 128, then stream remaining frames in 34-frame steps, fuse IMU.

        ``clips`` is ``[B,3,T,H,W]`` with ``T = 128 + n_steps * 34``.
        """
        t = int(clips.size(2))
        extra = t - self.cache_frames
        if extra < 0 or extra % self.new_frames != 0:
            raise ValueError(
                f"T={t} must be {self.cache_frames} + k*{self.new_frames}"
            )
        n_steps = extra // self.new_frames
        aux0 = None if aux_map is None else aux_map[:, :, : self.cache_frames]
        state = self.fill(clips[:, :, : self.cache_frames], aux0, slot_start=0)
        for i in range(n_steps):
            s = self.cache_frames + i * self.new_frames
            e = s + self.new_frames
            aux_i = None if aux_map is None else aux_map[:, :, s:e]
            state = self.step(state, clips[:, :, s:e], aux_i, mode=mode)
        imu = imu_len = None
        if imu_batch is not None:
            imu, imu_len = imu_batch
        return self.tokens_from_state(state, imu, imu_len)

    def forward(
        self,
        clips: torch.Tensor,
        aux_map: torch.Tensor | None = None,
        imu_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
        mode: PruneMode = "attn",
    ) -> torch.Tensor:
        return self.forward_window(clips, aux_map, imu_batch, mode=mode)

    def forward_oneshot_last128(
        self,
        clips: torch.Tensor,
        aux_map: torch.Tensor | None = None,
        imu_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Upper bound: one-shot encode the newest 128 frames (no streaming)."""
        clips_128 = clips[:, :, -self.cache_frames :]
        aux_128 = None if aux_map is None else aux_map[:, :, -self.cache_frames :]
        adapted = self._adapt(clips_128, aux_128)
        tokens = strip_hier(self.concat_ca.base_model.encoder(adapted), self.embed_dim)
        if imu_batch is None:
            return tokens
        imu, imu_len = imu_batch
        imu_128 = imu[:, -self.cache_frames :]
        len_128 = imu_len[:, -self.cache_frames :]
        return self._fuse(tokens, (imu_128, len_128))


def encode_stream_kv_chunks(
    encoder: nn.Module,
    clips: torch.Tensor,
    *,
    chunk_frames: int = 16,
    train_last_chunk_only: bool = True,
) -> torch.Tensor:
    """Newest-aligned streaming KV encode (matches ``eval_nopred_kvcache_size_sweep``).

    Clips are split into ``chunk_frames`` chunks. Each chunk's Q attends to the
    growing post-RoPE K/V cache (global absolute positions). Returns concatenated
    latents ``[B, N, D]`` (last ``embed_dim`` if hierarchical).

    When ``train_last_chunk_only`` (default), history chunks run under
    ``torch.no_grad()`` and cached K/V are detached before the live chunk — same
    as inference (frozen history) and BPTT-safe for LoRA + probe training.
    """
    b, _c, t, h, w = clips.shape
    if t % chunk_frames != 0:
        raise ValueError(f"T={t} not divisible by chunk_frames={chunk_frames}")
    embed_dim = int(encoder.embed_dim)
    ps = int(encoder.patch_size)
    gh, gw = h // ps, w // ps
    n_chunks = t // chunk_frames
    cache_k: list[torch.Tensor | None] = [None] * len(encoder.blocks)
    cache_v: list[torch.Tensor | None] = [None] * len(encoder.blocks)
    out_chunks: list[torch.Tensor] = []
    n_past = 0

    def _run_chunk(ci: int, with_grad: bool) -> torch.Tensor:
        nonlocal n_past
        chunk = clips[:, :, ci * chunk_frames : (ci + 1) * chunk_frames]
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            x = encoder.patch_embed(chunk)
            n_new = x.size(1)
            pos = torch.arange(n_past, n_past + n_new, device=x.device).unsqueeze(0).expand(b, -1)
            for li, blk in enumerate(encoder.blocks):
                attn = blk.attn
                q, k, v = rope_qkv(attn, blk.norm1(x), pos, gh, gw)
                if cache_k[li] is not None:
                    k_hist, v_hist = cache_k[li], cache_v[li]
                    if with_grad and train_last_chunk_only:
                        k_hist = k_hist.detach()
                        v_hist = v_hist.detach()
                    k_all = torch.cat([k_hist, k], dim=2)
                    v_all = torch.cat([v_hist, v], dim=2)
                else:
                    k_all, v_all = k, v
                y = F.scaled_dot_product_attention(q, k_all, v_all, dropout_p=0.0, is_causal=False)
                y = y.transpose(1, 2).reshape(b, n_new, -1)
                y = attn.proj_drop(attn.proj(y))
                x = x + blk.drop_path(y)
                x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
                k_store, v_store = k, v
                if not with_grad or (train_last_chunk_only and not with_grad):
                    k_store, v_store = k_store.detach(), v_store.detach()
                elif train_last_chunk_only and with_grad:
                    # live chunk: still store detached K/V (not needed for probe)
                    k_store, v_store = k.detach(), v.detach()
                if cache_k[li] is None:
                    cache_k[li], cache_v[li] = k_store, v_store
                else:
                    prev_k, prev_v = cache_k[li], cache_v[li]
                    cache_k[li] = torch.cat([prev_k.detach(), k_store], dim=2)
                    cache_v[li] = torch.cat([prev_v.detach(), v_store], dim=2)
            if encoder.norm is not None:
                x = encoder.norm(x)
            tok = strip_hier(x, embed_dim)
            if not with_grad:
                tok = tok.detach()
        n_past += n_new
        return tok

    for ci in range(n_chunks):
        is_last = ci == n_chunks - 1
        with_grad = (not train_last_chunk_only) or is_last
        out_chunks.append(_run_chunk(ci, with_grad=with_grad))
    return torch.cat(out_chunks, dim=1)


def install_stream_kv_encode_on_wrapper(
    model: nn.Module,
    *,
    cache_frames: int,
    chunk_frames: int = 16,
    train_last_chunk_only: bool = True,
) -> nn.Module:
    """Replace anticipative ``forward`` with newest-aligned stream KV encode.

    ``cache_frames`` = history length in frames (0, 16, …, 112). Input clips are
    assumed newest-aligned; we take the last ``cache_frames + chunk_frames``
    frames (or last ``chunk_frames`` if cache is 0).
    """
    cache_frames = int(cache_frames)
    chunk_frames = int(chunk_frames)
    if cache_frames % chunk_frames != 0:
        raise ValueError(f"cache_frames={cache_frames} must be a multiple of chunk_frames={chunk_frames}")
    need = cache_frames + chunk_frames

    # Unwrap DDP / base_model nesting used by gaze adapters.
    root = model
    if hasattr(root, "module") and not hasattr(root, "encoder"):
        root = root.module
    wrapper = getattr(root, "base_model", root)
    if not hasattr(wrapper, "encoder") or not hasattr(wrapper, "forward"):
        raise RuntimeError(f"install_stream_kv_encode_on_wrapper: no anticipative wrapper on {type(model)}")

    encoder = wrapper.encoder
    orig_forward = wrapper.forward

    def forward_stream_kv(x, anticipation_times):
        if x.ndim != 5:
            raise ValueError(f"expected [B,C,T,H,W], got {tuple(x.shape)}")
        t = int(x.size(2))
        if t < need:
            raise ValueError(f"stream KV needs T>={need}, got T={t} (cache={cache_frames}, chunk={chunk_frames})")
        window = x[:, :, -need:]
        return encode_stream_kv_chunks(
            encoder,
            window,
            chunk_frames=chunk_frames,
            train_last_chunk_only=train_last_chunk_only,
        )

    wrapper.forward = forward_stream_kv
    wrapper._stream_kv_orig_forward = orig_forward
    wrapper._stream_kv_cache_frames = cache_frames
    wrapper._stream_kv_chunk_frames = chunk_frames
    return model
