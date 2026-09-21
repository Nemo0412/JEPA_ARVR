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


def gather_bn(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather token dim of [B,H,N,...] with idx [B,K]."""
    extra = t.shape[3:]
    b, h, _n = t.shape[:3]
    k = idx.size(1)
    idx_e = idx[:, None, :].reshape(b, 1, k, *([1] * len(extra))).expand(b, h, k, *extra)
    return t.gather(2, idx_e)


def gather_kv(cache: list[torch.Tensor], keep_tok: torch.Tensor) -> list[torch.Tensor]:
    return [gather_bn(c, keep_tok) for c in cache]


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
