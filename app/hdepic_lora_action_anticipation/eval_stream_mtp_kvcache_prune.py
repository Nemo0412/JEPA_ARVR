#!/usr/bin/env python3
"""Frozen EGTEA streaming-MTP eval across KV-cache (post-encoder memory) prune
strategies and FPS points (B13 frozen ablation).

Additive: reuses the in-use ``train_stream_mtp`` build/val without editing it.
Strategies (all reduce the encoded memory to a 4 s budget except ``none``):

  * ``attention`` : existing ``TokenPruner`` (final-block received attention top-K)
  * ``recent``    : recency (keep last K tokens) -- ``RecentTokenPruner`` here
  * ``none``      : no prune (full 10 s memory, upper bound)
  * ``loss_aware``: ll native intermediate-layer cascade via a calibrated policy

FPS (method B) is set by ``--fps``/``--max-frames``/``--keep-count`` at launch;
the 4 s budget is expressed in tokens = ``4 * fps / tubelet * grid^2``.

Metric: native per-sample Action Top-5 @ +2/+4/+6 s (frozen, val-only).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation import train_stream_mtp as T


class FpsSubsampledStreamMTPDataset(T.StreamMTPDataset):
    """Method-B FPS subsampling: keep the time window, drop frames.

    The CSVs enumerate ``frame_indices`` at the 8 fps source rate. Lowering FPS
    strides those indices (8->4 fps: stride 2; 8->2 fps: stride 4), anchored to
    the newest frame (t=now) so recency is preserved, then loads only the kept
    frames -- so the encoder itself sees fewer tokens (the FPS compute lever).

    Bucketing by ``n_model_frames`` stays batch-safe: stride is a deterministic
    function of length, so equal-length rows subsample to equal length.
    """

    def __init__(self, csv_path, video_root, img_size: int = 256, *, src_fps: int = 8, fps: int = 8,
                 context_crop_sec: float = 0.0):
        super().__init__(csv_path, video_root, img_size)
        if src_fps % int(fps) != 0:
            raise SystemExit(f"method-B FPS requires fps dividing src_fps={src_fps}, got fps={fps}")
        self.stride = src_fps // int(fps)
        # Newest-anchored context crop: keep only the most recent ``context_crop_sec``
        # seconds of frames (in source-fps units). 0 = no crop. Lets us truncate the
        # long streaming (10 s) cases to a uniform 4 s input without changing the
        # future labels (the prediction task is anchored at t=now = newest frame).
        self.crop_frames = int(round(float(context_crop_sec) * int(src_fps)))

    def __getitem__(self, idx: int):
        if self.stride == 1 and self.crop_frames <= 0:
            return super().__getitem__(idx)
        r = self.rows[idx]
        video_id = str(r["video_id"])
        frame_idx = np.asarray(T._parse_int_list(r["frame_indices"]), dtype=np.int64)
        if self.crop_frames > 0 and frame_idx.shape[0] > self.crop_frames:
            frame_idx = frame_idx[-self.crop_frames:]  # newest ``context_crop_sec`` seconds
        if self.stride > 1:
            # Anchor at the newest frame; take every ``stride``-th going backward.
            frame_idx = frame_idx[::-1][:: self.stride][::-1].copy()
        frame_idx = np.ascontiguousarray(frame_idx)
        pid = video_id.split("_")[0]
        path = self.video_root / pid / f"{video_id}.MP4"
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1, width=self.img_size, height=self.img_size)
        try:
            frame_idx = np.clip(frame_idx, 0, len(vr) - 1)
            frames = vr.get_batch(frame_idx.tolist()).asnumpy()
        finally:
            del vr
        clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).contiguous()
        return {
            "clip": clip,
            "context_sec": float(r["context_sec"]),
            "n_frames": int(clip.shape[1]),
            "mtp_verbs": torch.tensor(T._parse_int_list(r["mtp_verbs"]), dtype=torch.long),
            "mtp_nouns": torch.tensor(T._parse_int_list(r["mtp_nouns"]), dtype=torch.long),
            "mtp_mask": torch.tensor(T._parse_float_list(r["mtp_mask"]), dtype=torch.float32),
        }


class RecentTokenPruner:
    """Recency memory prune: keep the last K tokens (chronological tail).

    Interface-compatible with ``train_stream_mtp.TokenPruner`` so it drops into
    ``PrunedAnticipativeModel`` unchanged. No encoder patch: recency needs no
    importance signal.
    """

    def __init__(self, keep_count: int, gp: int):
        self.gp = int(gp)
        self.keep_count = max(self.gp, (int(keep_count) // self.gp) * self.gp)

    def prune(self, feats: torch.Tensor):
        N = feats.shape[1]
        K = min(self.keep_count, (N // self.gp) * self.gp)
        if K >= N:
            idx = torch.arange(N, device=feats.device).unsqueeze(0).expand(feats.size(0), -1)
            return feats, idx
        idx = torch.arange(N - K, N, device=feats.device).unsqueeze(0).expand(feats.size(0), -1)
        return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])), idx

    def remove(self):  # symmetry with TokenPruner
        return None


class RecentJitterTokenPruner:
    """Step-1 PE-misalignment probe (B13). Keep the recent-K window, then randomly
    drop ``drop_frac`` of tokens PER FRAME and pack. Content stays ~intact (drop is
    tiny, and the survivors are already-encoded recent tokens), but the pack re-bases
    the survivors to ``arange`` -- so each survivor's decoded (t,h,w) drifts from its
    real position (see [[b13-direction-a-coverage-selection]] rebasing note). If Top-5
    falls even with near-intact content, the rebasing PE-scramble is itself harmful,
    and the frozen method gaps sit within that confound. ``drop_frac=0`` reproduces
    ``recent`` exactly (sanity arm)."""

    def __init__(self, keep_count: int, gp: int, *, drop_frac: float, seed: int = 0):
        self.gp = int(gp)
        self.keep_count = max(self.gp, (int(keep_count) // self.gp) * self.gp)
        self.drop_frac = float(drop_frac)
        self.seed = int(seed)
        self._call = 0

    def prune(self, feats: torch.Tensor):
        B, N, D = feats.shape
        gp = self.gp
        K = min(self.keep_count, (N // gp) * gp)
        start = N - K  # recent window start (chronological tail)
        n_frames = K // gp
        keep_per = gp - int(round(gp * self.drop_frac))
        if self.drop_frac <= 0 or keep_per >= gp:
            idx = torch.arange(start, N, device=feats.device).unsqueeze(0).expand(B, -1)
            return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)), idx
        g = torch.Generator(device="cpu").manual_seed(self.seed + self._call)
        self._call += 1
        # per (sample, frame): random subset of keep_per spatial positions, sorted
        rand = torch.rand(B, n_frames, gp, generator=g)
        sel = rand.topk(keep_per, dim=2, largest=False).indices.sort(dim=2).values  # [B, nf, keep_per]
        frame_base = start + torch.arange(n_frames).view(1, n_frames, 1) * gp
        idx = (frame_base + sel).reshape(B, n_frames * keep_per).to(feats.device)  # ascending
        return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)), idx

    def remove(self):
        return None


class RecencyWeightedTokenPruner:
    """Hand recency-weighted schedule (ZERO calibration) -- candidate #5.

    Tests whether loss_aware's frozen edge (39.95 vs attention 39.84) is just a
    recency prior: the keep-pattern diagnostic showed loss_aware is a position-FIXED
    mask that keeps ~0.72 of the budget in the last 4 s. This reproduces that profile
    by hand -- allocate ``recent_frac`` of the K/gp slot budget to the most recent
    ``recent_window_sec`` (contiguous, newest-first) and stride-sample the rest across
    the older history -- with NO calibration set. If it matches loss_aware, that edge
    is a deployable recency prior, not offline calibration. Slot-level selection keeps
    every kept token frame-aligned (multiple of gp), as RoPE decoding needs."""

    def __init__(self, keep_count: int, gp: int, *, recent_frac: float = 0.72,
                 recent_window_sec: float = 4.0, fps: int = 8, tubelet: int = 2):
        self.gp = int(gp)
        self.keep_count = max(self.gp, (int(keep_count) // self.gp) * self.gp)
        self.recent_frac = float(recent_frac)
        self.recent_slots = max(1, int(round(recent_window_sec * fps / tubelet)))

    def prune(self, feats: torch.Tensor):
        B, N, D = feats.shape
        gp = self.gp
        K = min(self.keep_count, (N // gp) * gp)
        if K >= N:
            idx = torch.arange(N, device=feats.device).unsqueeze(0).expand(B, -1)
            return feats, idx
        n_slots = N // gp
        k_slots = K // gp
        recent_region = min(self.recent_slots, n_slots)
        k_recent = min(recent_region, int(round(self.recent_frac * k_slots)))
        k_old = k_slots - k_recent
        old_region = n_slots - recent_region
        if k_old > old_region:  # older region too small; spill into recent
            k_recent = min(recent_region, k_recent + (k_old - old_region))
            k_old = k_slots - k_recent
        # recent: the most-recent k_recent slots (contiguous, newest-first)
        recent_start = n_slots - recent_region
        recent_slots = list(range(n_slots - k_recent, n_slots))
        # old: stride-sample k_old slots evenly across [0, recent_start)
        if k_old > 0 and old_region > 0:
            pos = torch.linspace(0, old_region - 1, k_old).round().long().tolist()
            old_slots = sorted(set(int(p) for p in pos))
            # top up if rounding collided
            i = 0
            while len(old_slots) < k_old and i < old_region:
                if i not in old_slots:
                    old_slots.append(i)
                i += 1
            old_slots = sorted(old_slots)[:k_old]
        else:
            old_slots = []
        keep_slots = sorted(old_slots + recent_slots)
        slot_idx = torch.tensor(keep_slots, device=feats.device)
        tok = (slot_idx.unsqueeze(1) * gp + torch.arange(gp, device=feats.device).unsqueeze(0)).reshape(-1)
        idx = tok.unsqueeze(0).expand(B, -1)
        return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)), idx

    def remove(self):
        return None


class OfflineCalibPruner:
    """[ATTN-CORNER-SINK] B18 · Offline position-indexed prune from a calibrated predictor-blk0
    map. Loads the [slots, gp] mean received-attention map (``calibrate_predictor_blk0_offline``)
    and keeps the FIXED top-K (``mode='high'``) or bottom-K (``mode='low'``) absolute positions
    for every sample -- a deterministic, content-independent decision (the offline twin of the
    online ``pred_attention_*``). Interface-compatible with ``TokenPruner``.
    """

    def __init__(self, calib_path: str, keep_count: int, gp: int, mode: str):
        arr = np.load(calib_path).astype(np.float32).reshape(-1)   # [slots*gp]
        self.score = torch.from_numpy(arr)
        self.gp = int(gp)
        self.keep_count = max(self.gp, (int(keep_count) // self.gp) * self.gp)
        self.mode = str(mode)
        self.calib_len = int(self.score.numel())

    def prune(self, feats: torch.Tensor):
        N = feats.shape[1]
        K = min(self.keep_count, (N // self.gp) * self.gp)
        if K >= N:
            idx = torch.arange(N, device=feats.device).unsqueeze(0).expand(feats.size(0), -1)
            return feats, idx
        score = self.score
        if score.numel() != N:   # align to the newest N positions (calib is oldest..newest per slot)
            score = score[-N:] if score.numel() > N else torch.cat(
                [torch.full((N - score.numel(),), float(score.min())), score])
        _, idx = score.to(feats.device).topk(K, largest=(self.mode == "high"))
        idx = idx.sort().values.unsqueeze(0).expand(feats.size(0), -1)
        return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])), idx

    def remove(self):
        return None


class WindowOffsetPruner:
    """[ATTN-CORNER-SINK] B18 · Keep a CONTIGUOUS block of keep_count/gp temporal slots starting
    at slot ``offset`` (test-(i) per-slot utility curve: accuracy vs window recency). offset is
    clamped so the window stays in-range; ``window@48`` on 64 slots == the ``recent`` 16-slot tail.
    """

    def __init__(self, offset: int, keep_count: int, gp: int):
        self.offset = int(offset)
        self.gp = int(gp)
        self.keep_count = max(self.gp, (int(keep_count) // self.gp) * self.gp)

    def prune(self, feats: torch.Tensor):
        N = feats.shape[1]
        K = min(self.keep_count, (N // self.gp) * self.gp)
        if K >= N:
            idx = torch.arange(N, device=feats.device).unsqueeze(0).expand(feats.size(0), -1)
            return feats, idx
        n_slots = N // self.gp
        k_slots = K // self.gp
        o = min(max(0, self.offset), max(0, n_slots - k_slots))
        start = o * self.gp
        idx = torch.arange(start, start + K, device=feats.device).unsqueeze(0).expand(feats.size(0), -1)
        return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])), idx

    def remove(self):
        return None


class CornerTokenPruner:
    """[ATTN-CORNER-SINK] Drop the four spatial-corner cells (a ring x ring block
    at each corner) at *every* time slot; keep every other token. Tests whether
    the per-head last-block corner attention-sink tokens carry any downstream
    information or are just low-info attention dumps (B18 follow-up).

    ``mode="random"`` drops the same COUNT of cells per slot at fixed random
    (non-corner-biased) positions -- the control that neutralises the token-count
    and position-rebasing confounds, so corner-vs-random isolates the corners.
    Interface-compatible with ``TokenPruner`` (drops into ``PrunedAnticipativeModel``).
    """

    def __init__(self, grid: int, gp: int, ring: int = 1, mode: str = "corner", seed: int = 0):
        self.grid = int(grid)
        self.gp = int(gp)
        self.ring = int(ring)
        self.mode = str(mode)
        m = torch.zeros(self.grid, self.grid, dtype=torch.bool)
        r = self.ring
        m[:r, :r] = m[:r, -r:] = m[-r:, :r] = m[-r:, -r:] = True
        self.n_drop_per_slot = int(m.sum())
        if self.mode == "corner":
            keep = (~m).reshape(-1).nonzero(as_tuple=False).squeeze(1)
        elif self.mode == "random":
            g = torch.Generator().manual_seed(int(seed))
            drop = torch.randperm(self.gp, generator=g)[: self.n_drop_per_slot]
            drop_set = set(drop.tolist())
            keep = torch.tensor([i for i in range(self.gp) if i not in drop_set], dtype=torch.long)
        else:
            raise SystemExit(f"CornerTokenPruner mode must be corner|random, got {mode!r}")
        self.keep_local = keep.sort().values  # ascending within a slot

    def prune(self, feats: torch.Tensor):
        B, N, D = feats.shape
        slots = N // self.gp
        keep_local = self.keep_local.to(feats.device)
        offsets = (torch.arange(slots, device=feats.device) * self.gp).unsqueeze(1)
        idx = (offsets + keep_local.unsqueeze(0)).reshape(-1)  # ascending across slots
        idx = idx.unsqueeze(0).expand(B, -1)
        return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)), idx

    def remove(self):
        return None


class CornerKeyMasker:
    """[ATTN-CORNER-SINK] ll's StreamingLLM-style ablation: forbid EVERY query from
    attending to the corner sink KEY columns (S[:, j] = -inf for j in corners) INSIDE
    the encoder attention, while token count, RoPE positions and hidden-state feed to
    the predictor are otherwise untouched (only the attention *to* the sink columns is
    removed). Contrast with ``CornerTokenPruner`` (removes the sink's OUTPUT
    post-encoder): this tests whether the corner tokens are load-bearing as an
    attention DUMP-POINT (StreamingLLM: evicting the sink is catastrophic).

    Implemented by wrapping the real ``RoPEAttention.forward`` (which already threads
    ``attn_mask`` into SDPA) with an injected boolean key-mask -- no RoPE re-impl.
    """

    def __init__(self, encoder, grid: int, gp: int, ring: int = 1, blocks: str = "all"):
        self.grid = int(grid)
        self.gp = int(gp)
        self.ring = int(ring)
        nb = len(encoder.blocks)
        if str(blocks) == "all":
            ids = list(range(nb))
        elif str(blocks) == "last":
            ids = [nb - 1]
        else:
            ids = [int(b) for b in str(blocks).split(",")]
        self.block_ids = ids
        m = torch.zeros(self.grid, self.grid, dtype=torch.bool)
        r = self.ring
        m[:r, :r] = m[:r, -r:] = m[-r:, :r] = m[-r:, -r:] = True
        self.local_corner = m.reshape(-1)  # (gp,) True at corner cells (to mask)
        self.n_masked_per_slot = int(m.sum())
        self._orig = []
        for i in ids:
            self._install(encoder.blocks[i].attn)

    def _keep_mask(self, N: int, device):
        """Boolean SDPA mask (1,1,1,N): True = attend, False = corner key (masked)."""
        slots = N // self.gp
        col = self.local_corner.to(device).repeat(slots)
        if col.numel() < N:  # any trailing cls/reg cols stay visible
            col = torch.cat([col, torch.zeros(N - col.numel(), dtype=torch.bool, device=device)])
        return (~col).view(1, 1, 1, N)

    def _install(self, m):
        orig = m.forward
        masker = self

        def fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            keep = masker._keep_mask(x.shape[1], x.device)
            am = keep if attn_mask is None else (attn_mask & keep)
            return orig(x, mask=mask, attn_mask=am, T=T, H_patches=H_patches, W_patches=W_patches)

        m.forward = fwd
        self._orig.append((m, orig))

    def prune(self, feats: torch.Tensor):  # no output pruning; corners kept in the feed
        N = feats.shape[1]
        return feats, torch.arange(N, device=feats.device).unsqueeze(0).expand(feats.size(0), -1)

    def remove(self):
        for m, orig in self._orig:
            m.forward = orig


class DesinkTokenPruner:
    """[ATTN-CORNER-SINK] Attention-importance prune with a DE-SINKED score (B18
    finding-13 follow-up). Identical to ``TokenPruner`` except the received-attention
    ranking is computed with the corner sink KEY columns masked (-inf) before softmax:
    sink tokens then score ~0 (pruned) and each query's mass renormalizes over content,
    giving a cleaner content ranking. Only the SELECTION score changes -- the block
    OUTPUT is the normal (unmasked) forward, so kept tokens carry standard features.
    Tests whether discounting the sink before scoring improves prune quality at a fixed
    budget, or whether the sink merely relocates in the score (finding 13) so it cannot.
    """

    def __init__(self, encoder, keep_count: int, gp: int, ring: int = 1, chunk_size: int = 256):
        from src.models.utils.modules import rotate_queries_or_keys
        self.keep_count = max(gp, (keep_count // gp) * gp)
        self.gp = int(gp)
        self.chunk_size = int(chunk_size)
        grid = int(round(gp ** 0.5))
        mm = torch.zeros(grid, grid, dtype=torch.bool)
        r = int(ring)
        mm[:r, :r] = mm[:r, -r:] = mm[-r:, :r] = mm[-r:, -r:] = True
        self.local_corner = mm.reshape(-1)  # (gp,) True at corner cells
        self._importance: torch.Tensor | None = None
        m = encoder.blocks[-1].attn
        self._attn_module = m
        self._orig_forward = m.forward
        pruner = self

        def _fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            out = pruner._orig_forward(x, mask=mask, attn_mask=attn_mask, T=T,
                                       H_patches=H_patches, W_patches=W_patches)
            with torch.no_grad():
                B, N, C = x.size()
                grid_depth = int(N // (m.grid_size * m.grid_size))
                qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv[0], qkv[1], qkv[2]
                if mask is not None:
                    mp = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
                    d_mask, h_mask, w_mask = m.separate_positions(mp, H_patches, W_patches)
                else:
                    mp = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=x.device)
                    d_mask, h_mask, w_mask = m.separate_positions(mp, H_patches, W_patches)
                s = 0
                qd = rotate_queries_or_keys(q[..., s:s + m.d_dim], pos=d_mask)
                kd = rotate_queries_or_keys(k[..., s:s + m.d_dim], pos=d_mask); s += m.d_dim
                qh = rotate_queries_or_keys(q[..., s:s + m.h_dim], pos=h_mask)
                kh = rotate_queries_or_keys(k[..., s:s + m.h_dim], pos=h_mask); s += m.h_dim
                qw = rotate_queries_or_keys(q[..., s:s + m.w_dim], pos=w_mask)
                kw = rotate_queries_or_keys(k[..., s:s + m.w_dim], pos=w_mask); s += m.w_dim
                if s < m.head_dim:
                    q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
                    k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
                else:
                    q = torch.cat([qd, qh, qw], dim=-1)
                    k = torch.cat([kd, kh, kw], dim=-1)
                slots = N // pruner.gp
                col = pruner.local_corner.to(x.device).repeat(slots)
                if col.numel() < N:
                    col = torch.cat([col, torch.zeros(N - col.numel(), dtype=torch.bool, device=x.device)])
                col = col.view(1, 1, 1, N)
                imp = torch.zeros(B, N, device=x.device, dtype=torch.float32)
                for ci in range(0, N, pruner.chunk_size):
                    logits = (q[:, :, ci:ci + pruner.chunk_size, :] @ k.transpose(-2, -1)) * m.scale
                    logits = logits.masked_fill(col, float("-inf"))         # de-sink the score
                    imp += logits.softmax(dim=-1).sum(dim=2).sum(dim=1).float()
                pruner._importance = imp
            return out

        m.forward = _fwd

    def prune(self, feats: torch.Tensor):
        if self._importance is None:
            raise RuntimeError("Run encoder before DesinkTokenPruner.prune()")
        N = feats.shape[1]
        K = min(self.keep_count, (N // self.gp) * self.gp)
        if K >= N:
            idx = torch.arange(N, device=feats.device).unsqueeze(0).expand(feats.size(0), -1)
            return feats, idx
        _, idx = self._importance.topk(K, dim=1)
        idx = idx.sort(dim=1).values
        return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])), idx

    def remove(self):
        if self._orig_forward is not None:
            self._attn_module.forward = self._orig_forward
            self._orig_forward = None


class AttnExcludeSinkPruner:
    """[ATTN-CORNER-SINK] Sink-EXCLUDE prune (the correct operationalisation): the
    received-attention SCORE is the *unmodified* ``TokenPruner`` score (softmax untouched
    -> no relocation). At SELECTION, tokens whose received attention exceeds an
    OFFLINE-CALIBRATED threshold ``tau`` are deemed sinks and dropped from the candidate
    pool; the budget then keeps the top-K of the REMAINING tokens by their genuine score.
    We do not spend budget on sinks and re-rank the rest with their true importance.

    ``tau = thresh_mult * uniform`` where ``uniform = num_heads`` is the fair-share
    received level (sum_j imp[j] = num_heads * N). ``thresh_mult`` is chosen offline from
    the received-attention distribution (see probe_sink_threshold_calib.py).
    """

    def __init__(self, encoder, keep_count: int, gp: int, thresh_mult: float, chunk_size: int = 256):
        self._tp = T.TokenPruner(encoder, keep_count=keep_count, gp=gp, chunk_size=chunk_size)
        self.gp = int(gp)
        self.keep_count = self._tp.keep_count
        self.thresh_mult = float(thresh_mult)
        self.num_heads = int(encoder.blocks[-1].attn.num_heads)
        self.n_excluded_total = 0
        self.n_prune_calls = 0

    def prune(self, feats: torch.Tensor):
        imp = self._tp._importance
        if imp is None:
            raise RuntimeError("Run encoder before AttnExcludeSinkPruner.prune()")
        N = feats.shape[1]
        K = min(self.keep_count, (N // self.gp) * self.gp)
        if K >= N:  # no budget pressure -> keep all (sinks harmless here)
            idx = torch.arange(N, device=feats.device).unsqueeze(0).expand(feats.size(0), -1)
            return feats, idx
        tau = self.thresh_mult * float(self.num_heads)   # uniform received = num_heads
        sink = imp > tau
        self.n_excluded_total += int(sink.sum().item())
        self.n_prune_calls += 1
        imp2 = imp.masked_fill(sink, float("-inf"))       # exclude sinks from selection only
        _, idx = imp2.topk(K, dim=1)
        idx = idx.sort(dim=1).values
        return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])), idx

    def remove(self):
        self._tp.remove()


class _LimitedLoader:
    """Wrap a DataLoader to yield at most ``n`` batches (smoke), keeping __len__."""

    def __init__(self, loader, n: int):
        self._loader = loader
        self._n = int(n)

    def __iter__(self):
        return itertools.islice(iter(self._loader), self._n)

    def __len__(self):
        return min(self._n, len(self._loader))


def enlarge_predictor_budget(base, num_tokens_full: int, gp: int):
    """Raise the predictor's position budget so it can ingest an unpruned context
    that overflows the pretrained ``num_patches`` (= num_frames/tubelet * grid^2
    = 8192). The mask token is a single learned [1,1,D] parameter *repeated*
    ``num_patches`` times, and ``use_rope`` means the sincos ``predictor_pos_embed``
    is unused -- so ``num_patches`` is only a repeat/index count and can be raised
    with NO new parameters and NO buffer resize. True temporal positions are kept
    (no NTK compression); RoPE angles simply extrapolate past the trained depth.

    Sized to cover context (num_tokens_full) + anticipation offset + target block,
    with a generous margin.
    """
    predictor = base.predictor
    needed = int(num_tokens_full) + int(gp) * 64  # ctx + max plausible ant offset + targets
    if int(getattr(predictor, "num_patches", 0)) < needed:
        predictor.num_patches = needed
    return predictor


class PositionAwarePrunedModel(T.PrunedAnticipativeModel):
    """Analysis wrapper (B13): feed the predictor the TRUE positions of surviving
    context tokens instead of the compact ``arange(K)`` rebasing, to isolate which
    axis of the rebasing confound matters (count vs temporal depth vs spatial).

    Modes:
      * ``rebase``        -- identical to ``PrunedAnticipativeModel`` (arange(K)).
      * ``true_temporal`` -- true temporal DEPTH per token, spatial left arange-packed
                             (synthetic): isolates the TEMPORAL axis. Encodes into the
                             1D RoPE index as ``true_frame*gp + (ordinal % gp)`` so
                             ``separate_positions`` decodes (true_frame, synthetic h,w).
      * ``true_full``     -- true (t, h, w) per token (the real 1D index).

    Target positions are re-based off the true newest frame ``F_now`` so the
    anticipation gap (frames ahead of now) matches the rebase baseline exactly.
    True indices come from the post-encoder pruner's returned idx, or (loss_aware,
    encoder-internal prune) from ``encoder_pruner._last_idx``.
    """

    def __init__(self, base, pruner, prune_threshold, *, position_mode="rebase",
                 encoder_pruner=None, gp=None):
        super().__init__(base, pruner, prune_threshold)
        self.position_mode = position_mode
        self.encoder_pruner = encoder_pruner
        self._gp = int(gp) if gp else int(base.grid_size**2)

    def forward(self, x, anticipation_times):
        if self.position_mode == "rebase":
            return super().forward(x, anticipation_times)
        core = self.base
        gp = self._gp
        x_full = core.encoder(x)
        B, N, D_full = x_full.size()
        embed_dim = core.encoder.embed_dim
        if self.pruner is not None and N > self.prune_threshold:
            x_full, kept_idx = self.pruner.prune(x_full)
            B, N, D_full = x_full.size()
            true_idx = kept_idx.to(x.device)
        elif self.encoder_pruner is not None and self.encoder_pruner._last_idx is not None:
            true_idx = self.encoder_pruner._last_idx.to(x.device)
        else:
            true_idx = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
        if true_idx.shape[1] != N:  # defensive: no-prune contexts
            true_idx = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)

        true_frame = true_idx // gp  # [B, N]
        ordinal = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
        if self.position_mode == "true_temporal":
            ctxt_positions = true_frame * gp + (ordinal % gp)  # true depth, synthetic spatial
        elif self.position_mode == "true_full":
            ctxt_positions = true_idx
        else:
            raise SystemExit(f"unknown position_mode {self.position_mode!r}")

        use_hierarchical = D_full > embed_dim
        x = x_full[:, :, -embed_dim:] if use_hierarchical else x_full
        x_accumulate = x.clone()
        anticipation_steps = (anticipation_times * core.frames_per_second / core.tubelet_size).to(torch.int64)
        f_now = true_frame.max(dim=1).values  # [B]
        skip_positions = (f_now + 1 + anticipation_steps) * gp  # [B], target ant_steps+1 frames past now
        N_pred = int(gp * (core.num_output_frames // core.tubelet_size))
        tgt_positions = torch.arange(N_pred, device=x.device).unsqueeze(0).repeat(B, 1)
        tgt_positions = tgt_positions + skip_positions.unsqueeze(1)
        x_pred_input = x_full
        for _ in range(core.num_steps):
            pred_out = core.predictor(x_pred_input, masks_x=ctxt_positions, masks_y=tgt_positions)
            x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
            x_pred = x_pred_full[:, :, -embed_dim:] if x_pred_full.size(-1) != embed_dim else x_pred_full
            x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)
            x_pred_for_input = x_pred_full if x_pred_full.size(-1) == x_pred_input.size(-1) else x_pred
            x_pred_input = torch.cat([x_pred_input[:, N_pred:, :], x_pred_for_input], dim=1)
        return x_accumulate


class PredictorScorePrunedModel(T.PrunedAnticipativeModel):
    """[ATTN-CORNER-SINK] B18 · Two-pass predictor-score pruning of the encoder output.

    The existing ``attention`` strategy scores tokens by the ENCODER last-block received
    attention. Here the score SITE is the PREDICTOR: pass 1 runs the predictor on the FULL
    encoder-output context with a received-attention capture on ``predictor_blocks[score_block]``
    (all queries, all heads summed = the finding-15/15g online score); pass 2 keeps the K
    context tokens with the HIGHEST (``mode='high'``) or LOWEST (``mode='low'``) score, rebases
    their positions to ``arange(K)`` (identical to the attention/recent pruners), and re-runs
    the predictor for the real MTP prediction. Full-context pass-1 at 64 slots is OOD for the
    predictor (RoPE extrapolation past its trained 32-slot depth); ``enlarge_predictor_budget``
    (called in ``build_pruned_model``) raises ``num_patches`` so it fits.
    """

    def __init__(self, base, keep_count: int, gp: int, *, mode: str, score_block: int = 0,
                 chunk_size: int = 256):
        super().__init__(base, pruner=None, prune_threshold=int(keep_count))
        self.gp = int(gp)
        self.keep_count = max(self.gp, (int(keep_count) // self.gp) * self.gp)
        self.mode = str(mode)
        self.score_block = int(score_block)
        self.chunk_size = int(chunk_size)

    def _tgt_positions(self, core, N, B, device, anticipation_times):
        anticipation_steps = (anticipation_times * core.frames_per_second / core.tubelet_size).to(torch.int64)
        skip_positions = N + int(core.grid_size ** 2) * anticipation_steps
        N_pred = int(core.grid_size ** 2 * (core.num_output_frames // core.tubelet_size))
        tgt = torch.arange(N_pred, device=device).unsqueeze(0).repeat(B, 1) + skip_positions.unsqueeze(1)
        return tgt, N_pred

    def _score_select(self, core, x_full, anticipation_times):
        from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import HeadAttnCapture20
        B, N, _ = x_full.size()
        ctxt_positions = torch.arange(N, device=x_full.device).unsqueeze(0).repeat(B, 1)
        tgt_positions, _ = self._tgt_positions(core, N, B, x_full.device, anticipation_times)
        nblk = len(core.predictor.predictor_blocks)
        b = self.score_block if self.score_block >= 0 else nblk + self.score_block
        cap = HeadAttnCapture20(core.predictor.predictor_blocks[b].attn, chunk_size=self.chunk_size)
        with torch.no_grad():
            core.predictor(x_full, masks_x=ctxt_positions, masks_y=tgt_positions)
        cap.remove()
        imp = cap.importance[:, :, :N].sum(dim=1).float()   # (B, N) head-summed received attn over context
        K = min(self.keep_count, (N // self.gp) * self.gp)
        _, idx = imp.topk(K, dim=1, largest=(self.mode == "high"))
        return idx.sort(dim=1).values

    def forward(self, x, anticipation_times):
        core = self.base
        x_full = core.encoder(x)
        B, N, D_full = x_full.size()
        if N > self.prune_threshold:
            idx = self._score_select(core, x_full, anticipation_times)
            x_full = x_full.gather(1, idx.unsqueeze(-1).expand(-1, -1, D_full))
            B, N, D_full = x_full.size()
        # ---- pass 2: mirrors PrunedAnticipativeModel.forward (post-prune) exactly ----
        embed_dim = core.encoder.embed_dim
        use_hierarchical = D_full > embed_dim
        x = x_full[:, :, -embed_dim:] if use_hierarchical else x_full
        x_accumulate = x.clone()
        ctxt_positions = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
        tgt_positions, N_pred = self._tgt_positions(core, N, B, x.device, anticipation_times)
        x_pred_input = x_full
        for _ in range(core.num_steps):
            pred_out = core.predictor(x_pred_input, masks_x=ctxt_positions, masks_y=tgt_positions)
            x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
            x_pred = x_pred_full[:, :, -embed_dim:] if x_pred_full.size(-1) != embed_dim else x_pred_full
            x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)
            x_pred_for_input = x_pred_full if x_pred_full.size(-1) == x_pred_input.size(-1) else x_pred
            x_pred_input = torch.cat([x_pred_input[:, N_pred:, :], x_pred_for_input], dim=1)
        return x_accumulate


def build_pruned_model(base, strategy: str, keep_count: int, gp: int, *,
                       prune_config: str | None, num_tokens_full: int, device,
                       anticipation_sec: float = 2.0,
                       hybrid_weight: float = 0.5, hybrid_prior_layer: int = -1,
                       coverage_seed: str = "recent", coverage_objective: str = "kcenter",
                       jitter_drop_frac: float = 0.05, jitter_seed: int = 0,
                       position_mode: str = "rebase",
                       recency_frac: float = 0.72, recency_window_sec: float = 4.0,
                       corner_ring: int = 1, corner_seed: int = 0, corner_mask_blocks: str = "all",
                       sink_thresh_mult: float = 3.0, pred_score_block: int = 0,
                       pred_calib_path: str | None = None):
    strategy = strategy.lower()
    # Raise the predictor position budget for ALL strategies so any memory budget
    # (keep_count) works -- budgets > ~5888 tokens (predictor targets reach
    # keep+2304 > num_patches=8192) otherwise overflow. Free + result-preserving
    # for small budgets (extra repeated mask-token slots are never indexed).
    enlarge_predictor_budget(base, num_tokens_full, gp)
    if strategy == "attention":
        pruner = T.TokenPruner(base.encoder, keep_count=keep_count, gp=gp)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=keep_count).to(device)
    if strategy == "attention_desink":
        pruner = DesinkTokenPruner(base.encoder, keep_count=keep_count, gp=gp, ring=corner_ring)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=keep_count).to(device)
    if strategy == "attention_exclude_sink":
        pruner = AttnExcludeSinkPruner(base.encoder, keep_count=keep_count, gp=gp, thresh_mult=sink_thresh_mult)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=keep_count).to(device)
    if strategy in ("pred_attention_high", "pred_attention_low"):
        mode = "high" if strategy.endswith("high") else "low"
        return PredictorScorePrunedModel(base, keep_count, gp, mode=mode,
                                         score_block=pred_score_block).to(device)
    if strategy in ("pred_offline_high", "pred_offline_low"):
        if not pred_calib_path:
            raise SystemExit("pred_offline_* requires --pred-calib-path (calibrated .npy map)")
        mode = "high" if strategy.endswith("high") else "low"
        pruner = OfflineCalibPruner(pred_calib_path, keep_count=keep_count, gp=gp, mode=mode)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=keep_count).to(device)
    if strategy == "recent":
        pruner = RecentTokenPruner(keep_count=keep_count, gp=gp)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=keep_count).to(device)
    if strategy == "recency_weighted":
        pruner = RecencyWeightedTokenPruner(keep_count=keep_count, gp=gp,
                                            recent_frac=recency_frac,
                                            recent_window_sec=recency_window_sec)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=keep_count).to(device)
    if strategy in ("corner", "corner_random_ctrl"):
        grid = int(round(gp ** 0.5))
        mode = "corner" if strategy == "corner" else "random"
        pruner = CornerTokenPruner(grid, gp, ring=corner_ring, mode=mode, seed=corner_seed)
        # prune_threshold=0 -> always drop the corners (independent of memory budget)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=0).to(device)
    if strategy == "corner_keymask":
        grid = int(round(gp ** 0.5))
        masker = CornerKeyMasker(base.encoder, grid, gp, ring=corner_ring, blocks=corner_mask_blocks)
        # masker patches the encoder; prune() is a no-op so all tokens still feed the predictor
        return T.PrunedAnticipativeModel(base, masker, prune_threshold=0).to(device)
    if strategy == "none":
        return T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).to(device)
    if strategy == "loss_aware":
        from app.hdepic_lora_action_anticipation.loss_aware_pruning import LossAwarePruningConfig
        from app.hdepic_lora_action_anticipation.vit_encoder_loss_aware_pruning import (
            LossAwareEncoderPruner,
        )
        if not prune_config:
            raise SystemExit("loss_aware strategy requires --prune-config (calibrated JSON)")
        cfg = LossAwarePruningConfig.load(prune_config)
        # Patches base.encoder.forward to cascade-prune internally; wrapper runs
        # the predictor on the already-pruned tokens (pruner=None).
        lae = LossAwareEncoderPruner(base.encoder, cfg, gp=gp, num_tokens_full=num_tokens_full)
        if position_mode != "rebase":
            return PositionAwarePrunedModel(base, None, prune_threshold=10**9,
                                            position_mode=position_mode,
                                            encoder_pruner=lae, gp=gp).to(device)
        return T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).to(device)
    if strategy == "future_attn":
        # Route 1: future-conditioned attention. Post-encoder prune scored by the
        # predictor's future target->context attention (one probe pass).
        from app.hdepic_lora_action_anticipation.future_conditioned_pruning import (
            FutureAttnTokenPruner,
        )
        pruner = FutureAttnTokenPruner(base, keep_count=keep_count, gp=gp,
                                       anticipation_sec=anticipation_sec)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=keep_count).to(device)
    if strategy == "recent_jitter":
        # Step-1 PE-misalignment probe: recent window + per-frame random drop + pack.
        pruner = RecentJitterTokenPruner(keep_count=keep_count, gp=gp,
                                         drop_frac=jitter_drop_frac, seed=jitter_seed)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=0).to(device)
    if strategy == "coverage":
        # Direction A: budget-constrained coverage / de-redundancy. Slot-level
        # farthest-point sampling over encoder features (frozen-safe, no patch).
        from app.hdepic_lora_action_anticipation.coverage_pruning import (
            CoverageTokenPruner,
        )
        pruner = CoverageTokenPruner(keep_count=keep_count, gp=gp, seed=coverage_seed,
                                     objective=coverage_objective)
        if position_mode != "rebase":
            return PositionAwarePrunedModel(base, pruner, prune_threshold=keep_count,
                                            position_mode=position_mode, gp=gp).to(device)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=keep_count).to(device)
    if strategy == "hybrid":
        # Route 3/2: w*loss_aware_prior + (1-w)*attention_content, top-K.
        from app.hdepic_lora_action_anticipation.loss_aware_pruning import (
            LossAwarePruningConfig,
            resolve_prune_layers,
        )
        from app.hdepic_lora_action_anticipation.future_conditioned_pruning import (
            HybridTokenPruner,
        )
        if not prune_config:
            raise SystemExit("hybrid strategy requires --prune-config (calibrated JSON for the prior)")
        cfg = LossAwarePruningConfig.load(prune_config)
        calibrated = cfg.load_calibrated_scores(device)
        prior_layer = hybrid_prior_layer
        if prior_layer < 0:
            prior_layer = min(resolve_prune_layers(cfg))
        pruner = HybridTokenPruner(base.encoder, keep_count=keep_count, gp=gp,
                                   calibrated_scores=calibrated, prior_layer=prior_layer,
                                   weight=hybrid_weight)
        return T.PrunedAnticipativeModel(base, pruner, prune_threshold=keep_count).to(device)
    raise SystemExit(f"unknown --prune-strategy {strategy!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True, help="for vocab maps only")
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="base V-JEPA2 encoder/predictor ckpt")
    ap.add_argument("--init-from-ckpt", type=Path, required=True, help="parent MTP best.pt (model + mtp_classifier)")
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--prune-strategy",
                    choices=["attention", "recent", "none", "loss_aware", "future_attn",
                             "hybrid", "coverage", "recent_jitter", "recency_weighted",
                             "attention_desink", "attention_exclude_sink",
                             "corner", "corner_random_ctrl", "corner_keymask",
                             "pred_attention_high", "pred_attention_low",
                             "pred_offline_high", "pred_offline_low"],
                    required=True)
    ap.add_argument("--prune-config", type=str, default=None,
                    help="calibrated JSON for loss_aware and for the hybrid position prior")
    ap.add_argument("--pred-score-block", type=int, default=0,
                    help="pred_attention_*: predictor block whose received attention scores tokens (0=first)")
    ap.add_argument("--pred-calib-path", type=str, default=None,
                    help="pred_offline_*: calibrated [slots,gp] .npy map from calibrate_predictor_blk0_offline")
    ap.add_argument("--sink-thresh-mult", type=float, default=3.0,
                    help="attention_exclude_sink: exclude tokens with received attn > mult*uniform "
                         "(uniform=num_heads) from selection; offline-calibrated")
    ap.add_argument("--corner-ring", type=int, default=1,
                    help="corner strategy: ring width in patches at each of the 4 corners (per slot)")
    ap.add_argument("--corner-seed", type=int, default=0,
                    help="corner_random_ctrl: RNG seed for the same-count random-cell control")
    ap.add_argument("--corner-mask-blocks", type=str, default="all",
                    help="corner_keymask: which encoder blocks mask the corner key columns "
                         "(all | last | comma-separated indices)")
    ap.add_argument("--hybrid-weight", type=float, default=0.5,
                    help="hybrid: w in w*loss_aware_prior + (1-w)*attention_content")
    ap.add_argument("--hybrid-prior-layer", type=int, default=-1,
                    help="hybrid: calibrated layer whose position table is the prior "
                         "(-1 = smallest calibrated layer, the full-length pre-prune table)")
    ap.add_argument("--coverage-seed", choices=["recent", "centroid"], default="recent",
                    help="coverage(kcenter): farthest-point-sampling seed ('recent' anchors at "
                         "t=now then covers backward; 'centroid' is recency-free pure coverage)")
    ap.add_argument("--coverage-objective", choices=["kcenter", "facility"], default="kcenter",
                    help="coverage objective: 'kcenter'/FPS keeps extremes (spread); 'facility' "
                         "keeps dense-region representatives (one exemplar per redundant cluster)")
    ap.add_argument("--jitter-drop-frac", type=float, default=0.05,
                    help="recent_jitter: fraction of tokens dropped per frame before packing "
                         "(0 reproduces recent; 0.05-0.10 injects PE misalignment, ~intact content)")
    ap.add_argument("--jitter-seed", type=int, default=0, help="recent_jitter: RNG seed")
    ap.add_argument("--recency-frac", type=float, default=0.72,
                    help="recency_weighted: fraction of the slot budget kept in the recent window "
                         "(0.72 mirrors loss_aware's measured recency profile)")
    ap.add_argument("--recency-window-sec", type=float, default=4.0,
                    help="recency_weighted: length of the recent window in seconds")
    ap.add_argument("--position-mode", choices=["rebase", "true_temporal", "true_full"],
                    default="rebase",
                    help="analysis (loss_aware/coverage): feed the predictor TRUE positions "
                         "of survivors instead of arange(K). 'true_temporal' = true depth + "
                         "synthetic spatial (isolate temporal axis); 'true_full' = true (t,h,w)")
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--primary-horizon-sec", type=float, default=2.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=80)   # 10 s @ 8 fps
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8, help="CSV source frame rate (method-B stride base)")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--keep-count", type=int, default=4096)  # 4 s budget @ 8 fps
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-val-batches", type=int, default=0, help="0 = full val; >0 for smoke")
    ap.add_argument("--val-subset-n", type=int, default=0,
                    help="0 = full val; >0 = deterministic random subset of this many rows, "
                         "selected by a stable hash of (video_id, tick_frame) so the SAME ticks "
                         "are chosen across window sizes (paired cross-window comparison)")
    ap.add_argument("--val-subset-seed", type=int, default=0, help="salt for the subset hash")
    ap.add_argument("--only-context-sec", type=float, default=0.0,
                    help="smoke: restrict val to this context length (e.g. 10 to exercise the "
                         "unpruned-budget/NTK path); 0 = all contexts")
    ap.add_argument("--context-crop-sec", type=float, default=0.0,
                    help="crop each clip to its newest N seconds of frames (0 = no crop); "
                         "e.g. truncate the 10 s streaming cases to a uniform 4 s input")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    primary_h = float(args.primary_horizon_sec)
    primary_idx = horizons.index(primary_h) if primary_h in horizons else 0
    weights = [1.0] * len(horizons)  # weights irrelevant for frozen val metrics

    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    val_ds = FpsSubsampledStreamMTPDataset(
        args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps,
        context_crop_sec=args.context_crop_sec,
    )
    if args.only_context_sec > 0:
        want = float(args.only_context_sec)
        val_ds.rows = [r for r in val_ds.rows if abs(float(r["context_sec"]) - want) < 1e-6]
        if not val_ds.rows:
            raise SystemExit(f"no val rows with context_sec == {want}")
        print(f"[smoke] restricted to context_sec={want}: {len(val_ds.rows)} rows", flush=True)
    if args.val_subset_n > 0 and args.val_subset_n < len(val_ds.rows):
        import hashlib
        salt = str(args.val_subset_seed)
        def _rowkey(r):
            s = f"{salt}|{r['video_id']}|{r['tick_frame']}"
            return hashlib.md5(s.encode()).hexdigest()
        order = sorted(range(len(val_ds.rows)), key=lambda i: _rowkey(val_ds.rows[i]))
        keep = set(order[: args.val_subset_n])
        val_ds.rows = [r for i, r in enumerate(val_ds.rows) if i in keep]
        print(f"[subset] deterministic paired subset n={len(val_ds.rows)} "
              f"(seed={args.val_subset_seed}, keyed on video_id+tick_frame)", flush=True)
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

    model = build_pruned_model(
        base, args.prune_strategy, args.keep_count, gp,
        prune_config=args.prune_config, num_tokens_full=num_tokens_full, device=device,
        anticipation_sec=args.anticipation_sec,
        hybrid_weight=args.hybrid_weight, hybrid_prior_layer=args.hybrid_prior_layer,
        coverage_seed=args.coverage_seed, coverage_objective=args.coverage_objective,
        jitter_drop_frac=args.jitter_drop_frac, jitter_seed=args.jitter_seed,
        position_mode=args.position_mode,
        recency_frac=args.recency_frac, recency_window_sec=args.recency_window_sec,
        corner_ring=args.corner_ring, corner_seed=args.corner_seed,
        corner_mask_blocks=args.corner_mask_blocks, sink_thresh_mult=args.sink_thresh_mult,
        pred_score_block=args.pred_score_block, pred_calib_path=args.pred_calib_path,
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
    m_miss, m_unexp = model.load_state_dict(ck["model"], strict=False)
    h_miss, h_unexp = mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
    print(f"[load] parent best={ck.get('best')} model(missing={len(m_miss)} unexpected={len(m_unexp)}) "
          f"mtp(missing={len(h_miss)} unexpected={len(h_unexp)})", flush=True)
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
        "prune_strategy": args.prune_strategy,
        "fps": args.fps, "src_fps": args.src_fps,
        "max_frames": args.max_frames, "keep_count": args.keep_count,
        "prune_config": args.prune_config,
        "hybrid_weight": args.hybrid_weight if args.prune_strategy == "hybrid" else None,
        "hybrid_prior_layer": args.hybrid_prior_layer if args.prune_strategy == "hybrid" else None,
        "coverage_seed": args.coverage_seed if args.prune_strategy == "coverage" else None,
        "coverage_objective": args.coverage_objective if args.prune_strategy == "coverage" else None,
        "jitter_drop_frac": args.jitter_drop_frac if args.prune_strategy == "recent_jitter" else None,
        "jitter_seed": args.jitter_seed if args.prune_strategy == "recent_jitter" else None,
        "recency_frac": args.recency_frac if args.prune_strategy == "recency_weighted" else None,
        "recency_window_sec": args.recency_window_sec if args.prune_strategy == "recency_weighted" else None,
        "position_mode": args.position_mode,
        "val_subset_n": args.val_subset_n if args.val_subset_n > 0 else None,
        "val_subset_seed": args.val_subset_seed if args.val_subset_n > 0 else None,
        "only_context_sec": args.only_context_sec,
        "context_crop_sec": args.context_crop_sec,
        "corner_ring": args.corner_ring if args.prune_strategy in ("corner", "corner_random_ctrl", "corner_keymask") else None,
        "corner_dropped_per_slot": (getattr(model.pruner, "n_drop_per_slot", None)
                                    if args.prune_strategy in ("corner", "corner_random_ctrl") else None),
        "corner_masked_per_slot": (getattr(model.pruner, "n_masked_per_slot", None)
                                   if args.prune_strategy == "corner_keymask" else None),
        "corner_mask_blocks": args.corner_mask_blocks if args.prune_strategy == "corner_keymask" else None,
        "sink_thresh_mult": args.sink_thresh_mult if args.prune_strategy == "attention_exclude_sink" else None,
        "sink_excluded_per_prune": (round(getattr(model.pruner, "n_excluded_total", 0)
                                          / max(1, getattr(model.pruner, "n_prune_calls", 1)), 1)
                                    if args.prune_strategy == "attention_exclude_sink" else None),
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
