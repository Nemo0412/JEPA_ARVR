#!/usr/bin/env python3
"""Offline 64-slot → 16-slot-token prune before predictor (AA accuracy).

Setup
-----
* Encode a **16s / 64-slot** window (128 frames @ 8fps → N=16384 tokens).
* Predictor always sees **K=4096** tokens (= 16 slots × 256 spatial), matching the
  token budget of a recent-16-slot baseline.
* Prune policies (applied on the *same* encoder features, before predictor):

  last16     keep the most recent 16 temporal slots (baseline)
  pred0_topk keep top-K by predictor block-0 attention column-sum
             (space + time free; order preserved after selection)
  enc_topk   keep top-K by encoder last-block attention column-sum
             (existing stream TokenPruner signal; optional reference)

Metrics: stream-MTP action Top-5 @ {2,4,6}s using p01_stream_mtp_2_4_6/best.pt.

Windows are rebuilt from the stream half-split val CSV by extending each tick
backward to 16s (skip if video/half origin cannot supply 128 model frames).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

import app.hdepic_lora_action_anticipation.train_stream_mtp as S  # noqa: E402
from app.hdepic_lora_action_anticipation.mtp import CommunicatingMLPMTPClassifier  # noqa: E402
from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402

logger = logging.getLogger("eval64_pred0")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IMAGENET_MEAN = S.IMAGENET_MEAN
IMAGENET_STD = S.IMAGENET_STD
GP = 256  # 16×16 spatial tokens per tubelet slot


# ── predictor block-0 importance (chunked column-sum) ────────────────────────
def make_pred0_hook(attn_module, chunk_size: int = 256):
    m = attn_module
    orig = m.forward
    store: dict = {"imp": None}

    def _fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
        if mask is not None:
            mask_p = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
        else:
            grid_depth = int(N // (m.grid_size * m.grid_size))
            mask_p = torch.arange(grid_depth * m.grid_size * m.grid_size, device=x.device)
        d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)
        qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
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
            k_t = k.transpose(-2, -1)
            for ci in range(0, N, chunk_size):
                q_c = q[:, :, ci : ci + chunk_size, :]
                logits = (q_c @ k_t) * m.scale
                imp += logits.softmax(dim=-1).sum(dim=2).sum(dim=1).float()
            store["imp"] = imp

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = m.proj(out)
        out = m.proj_drop(out)
        return out

    m.forward = _fwd
    return orig, store


def pred0_importance(predictor, x_full: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
    """Run predictor embed + block0 only; return [B, N] column-sum importance."""
    B, N, _ = x_full.shape
    blk0 = predictor.predictor_blocks[0]
    orig, store = make_pred0_hook(blk0.attn, chunk_size=chunk_size)
    try:
        ctx = torch.arange(N, device=x_full.device).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            x = predictor.predictor_embed(x_full)
            blk0(x, mask=ctx, attn_mask=None)
        if store["imp"] is None:
            raise RuntimeError("predictor block-0 hook did not record importance")
        return store["imp"]
    finally:
        blk0.attn.forward = orig


class EncLastHook:
    """Patch encoder last attn during encode to capture column-sum importance."""

    def __init__(self, encoder: nn.Module, chunk_size: int = 256):
        self.chunk_size = chunk_size
        self._attn = encoder.blocks[-1].attn
        self._orig = self._attn.forward
        self.importance: torch.Tensor | None = None
        pruner = self
        m = self._attn

        def _fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            B, N, C = x.size()
            grid_depth = int(N // (m.grid_size * m.grid_size))
            qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if mask is not None:
                mask_p = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
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
                for ci in range(0, N, pruner.chunk_size):
                    q_c = q[:, :, ci : ci + pruner.chunk_size, :]
                    logits = (q_c @ k.transpose(-2, -1)) * m.scale
                    imp += logits.softmax(dim=-1).sum(dim=2).sum(dim=1).float()
                pruner.importance = imp
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=m.proj_drop_prob, is_causal=m.is_causal, attn_mask=attn_mask
                )
            x = x.transpose(1, 2).reshape(B, N, C)
            x = m.proj(x)
            x = m.proj_drop(x)
            return x

        m.forward = _fwd

    def remove(self):
        self._attn.forward = self._orig

def select_topk(imp: torch.Tensor, k: int) -> torch.Tensor:
    """[B,N] → [B,K] indices sorted ascending (preserve temporal order)."""
    _, idx = imp.topk(k, dim=1)
    return idx.sort(dim=1).values


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def run_predictor(core, x_kept: torch.Tensor, anticipation_sec: float) -> torch.Tensor:
    """AR predictor on pruned tokens with rebased positions 0..N-1."""
    B, N, D_full = x_kept.size()
    embed_dim = core.encoder.embed_dim
    use_hierarchical = D_full > embed_dim
    x = x_kept[:, :, -embed_dim:] if use_hierarchical else x_kept
    x_accumulate = x.clone()
    ctxt_positions = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
    ant = torch.full((B,), float(anticipation_sec), device=x.device)
    anticipation_steps = (ant * core.frames_per_second / core.tubelet_size).to(torch.int64)
    skip_positions = N + int(core.grid_size**2) * anticipation_steps
    N_pred = int(core.grid_size**2 * (core.num_output_frames // core.tubelet_size))
    tgt_positions = torch.arange(N_pred, device=x.device).unsqueeze(0).repeat(B, 1)
    tgt_positions = tgt_positions + skip_positions.unsqueeze(1)
    x_pred_input = x_kept
    for _ in range(core.num_steps):
        pred_out = core.predictor(x_pred_input, masks_x=ctxt_positions, masks_y=tgt_positions)
        x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        x_pred = x_pred_full[:, :, -embed_dim:] if x_pred_full.size(-1) != embed_dim else x_pred_full
        x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)
        x_pred_for_input = x_pred_full if x_pred_full.size(-1) == x_pred_input.size(-1) else x_pred
        x_pred_input = torch.cat([x_pred_input[:, N_pred:, :], x_pred_for_input], dim=1)
    return x_accumulate


# ── dataset: rebuild 16s windows from stream val CSV ─────────────────────────
class Ctx64Dataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        video_root: Path,
        *,
        context_sec: float = 16.0,
        model_fps: float = 8.0,
        img_size: int = 256,
        max_samples: int = 0,
        stride: int = 1,
        require_ctx_sec: float | None = 10.0,
    ):
        self.video_root = Path(video_root)
        self.img_size = int(img_size)
        self.model_fps = float(model_fps)
        self.context_sec = float(context_sec)
        n_model = max(2, int(round(self.context_sec * self.model_fps)))
        if n_model % 2 == 1:
            n_model += 1
        self.n_model = n_model

        rows = []
        with Path(csv_path).open() as f:
            for row in csv.DictReader(f):
                if require_ctx_sec is not None and abs(float(row["context_sec"]) - require_ctx_sec) > 1e-6:
                    continue
                rows.append(row)
        if stride > 1:
            rows = rows[::stride]
        if max_samples > 0:
            rows = rows[:max_samples]

        kept = []
        skipped = 0
        for r in rows:
            tick = int(r["tick_frame"])
            origin = int(r["origin_frame"])
            n_frames = int(r["n_frames"])
            vfps = float(r["vfps"])
            # 16s window ending at tick; stay inside half-split region when possible.
            start = max(origin, tick - int(round(self.context_sec * vfps)))
            # Need enough native span for linspace of n_model frames.
            if tick - start < int(round(0.5 * self.context_sec * vfps)):
                # Fall back: allow earlier than origin (still within video).
                start = max(0, tick - int(round(self.context_sec * vfps)))
            if tick <= start:
                skipped += 1
                continue
            frame_idx = np.linspace(start, max(start, tick - 1), self.n_model)
            frame_idx = np.clip(np.round(frame_idx).astype(np.int64), 0, n_frames - 1)
            if len(np.unique(frame_idx)) < self.n_model // 2:
                skipped += 1
                continue
            kept.append(
                {
                    **r,
                    "frame_indices_64": frame_idx,
                    "start_frame_64": int(start),
                    "context_sec_64": self.context_sec,
                }
            )
        self.rows = kept
        logger.info(
            "Ctx64Dataset: kept=%d skipped=%d n_model=%d context=%.1fs",
            len(self.rows),
            skipped,
            self.n_model,
            self.context_sec,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        video_id = str(r["video_id"])
        pid = video_id.split("_")[0]
        path = self.video_root / pid / f"{video_id}.MP4"
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1, width=self.img_size, height=self.img_size)
        try:
            n_video = len(vr)
            frame_idx = np.clip(r["frame_indices_64"], 0, n_video - 1)
            frames = vr.get_batch(frame_idx.tolist()).asnumpy()
        finally:
            del vr
        clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).contiguous()
        return {
            "clip": clip,
            "mtp_verbs": torch.tensor(S._parse_int_list(r["mtp_verbs"]), dtype=torch.long),
            "mtp_nouns": torch.tensor(S._parse_int_list(r["mtp_nouns"]), dtype=torch.long),
            "mtp_mask": torch.tensor(S._parse_float_list(r["mtp_mask"]), dtype=torch.float32),
            "video_id": video_id,
            "tick_frame": int(r["tick_frame"]),
        }


def collate(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch], dim=0),
        "mtp_verbs": torch.stack([b["mtp_verbs"] for b in batch], dim=0),
        "mtp_nouns": torch.stack([b["mtp_nouns"] for b in batch], dim=0),
        "mtp_mask": torch.stack([b["mtp_mask"] for b in batch], dim=0),
        "video_id": [b["video_id"] for b in batch],
        "tick_frame": [b["tick_frame"] for b in batch],
    }


def update_metrics(totals, counts, outputs, batch, horizons, verb_map, noun_map, action_map, device, prefix: str):
    mtp_verbs = batch["mtp_verbs"]
    mtp_nouns = batch["mtp_nouns"]
    mtp_mask = batch["mtp_mask"]
    for hi, h in enumerate(horizons):
        valid = mtp_mask[:, hi] > 0.5
        if not bool(valid.any()):
            continue
        v_lab, n_lab, a_lab, keep = S.map_labels(
            mtp_verbs[valid, hi], mtp_nouns[valid, hi], verb_map, noun_map, action_map, device
        )
        if not keep:
            continue
        valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
        o = outputs[float(h)]
        acc = S.topk_acc(o["action"][valid_pos].float(), a_lab, k=5) * len(keep)
        key = f"{prefix}/action_top5@{h:g}s"
        totals[key] += acc
        counts[key] += len(keep)


def summarize(totals, counts) -> dict:
    out = {k: totals[k] / max(1, counts[k]) for k in totals}
    out.update({f"n|{k}": int(counts[k]) for k in counts})
    return out


def lift_predictor_capacity(predictor, n_ctx: int, gp: int = GP, n_pred_slots: int = 8):
    """Allow RoPE positions for full 64-slot context during scoring pass."""
    new_cap = ((n_ctx + gp * n_pred_slots) // gp + 8) * gp
    if hasattr(predictor, "num_patches"):
        old = int(predictor.num_patches)
        if new_cap > old:
            predictor.num_patches = new_cap
            logger.info("Lifted predictor.num_patches %d → %d", old, new_cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--train-csv", type=Path, required=True, help="vocab source (same as stream train)")
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--stream-ckpt", type=Path, required=True, help="best.pt with model+mtp_classifier")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--modes", type=str, default="last16,pred0_topk,enc_topk")
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--context-sec", type=float, default=16.0)
    ap.add_argument("--require-ctx-sec", type=float, default=10.0)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    keep_k = int(args.keep_count)
    assert keep_k % GP == 0, "keep_count must be multiple of 256"

    verb_map, noun_map, action_map = S.load_action_maps(args.train_csv)
    logger.info("vocab verbs=%d nouns=%d actions=%d", len(verb_map), len(noun_map), len(action_map))

    n_frames = max(2, int(round(args.context_sec * args.fps)))
    if n_frames % 2:
        n_frames += 1
    n_slots = n_frames // 2
    n_tok = n_slots * GP
    logger.info("encode window: %.1fs → %d frames → %d slots → %d tokens; keep=%d",
                args.context_sec, n_frames, n_slots, n_tok, keep_k)

    ds = Ctx64Dataset(
        args.val_csv,
        args.video_root,
        context_sec=args.context_sec,
        model_fps=args.fps,
        img_size=args.img_size,
        max_samples=args.max_samples,
        stride=args.stride,
        require_ctx_sec=args.require_ctx_sec if args.require_ctx_sec > 0 else None,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=False,
        persistent_workers=False,
    )

    core = S.build_model(device, n_frames, args.fps, args.img_size, str(args.checkpoint))
    for p in core.parameters():
        p.requires_grad = False
    # Inject LoRA modules so stream ckpt keys (*.base.weight / lora_*) match.
    S.load_lora_sidecars(
        core,
        str(args.encoder_lora) if args.encoder_lora else None,
        str(args.predictor_lora) if args.predictor_lora else None,
    )
    for p in core.parameters():
        p.requires_grad = False
    # Match train_stream_mtp checkpoint nesting (keys under base.*).
    wrapped = S.PrunedAnticipativeModel(core, None, prune_threshold=keep_k).to(device)

    classifier = AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=int(core.encoder.embed_dim),
        num_heads=16,
        depth=4,
        use_activation_checkpointing=True,
    ).to(device)
    mtp_clf = CommunicatingMLPMTPClassifier(
        classifier, horizons_sec=horizons, comm_layers=2, comm_heads=4
    ).to(device)

    logger.info("Loading stream ckpt %s ...", args.stream_ckpt)
    try:
        ck = torch.load(args.stream_ckpt, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        ck = torch.load(args.stream_ckpt, map_location="cpu", weights_only=False)
    miss, unexp = wrapped.load_state_dict(ck["model"], strict=False)
    logger.info("Loaded model missing=%d unexpected=%d", len(miss), len(unexp))
    if miss:
        logger.info("missing sample: %s", miss[:8])
    h_miss, h_unexp = mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
    logger.info(
        "Loaded mtp_classifier missing=%d unexpected=%d stream_best=%s",
        len(h_miss),
        len(h_unexp),
        ck.get("best"),
    )
    del ck
    base = wrapped.base  # AnticipativeModule (encoder+predictor)
    lift_predictor_capacity(base.predictor, n_tok, gp=GP)

    base.eval()
    mtp_clf.eval()
    enc_hook = EncLastHook(base.encoder, chunk_size=args.chunk_size) if "enc_topk" in modes else None

    totals = defaultdict(float)
    counts = defaultdict(int)
    slot_hist = {m: np.zeros(n_slots, dtype=np.float64) for m in modes if m != "last16"}
    t0 = time.time()

    with torch.no_grad():
        for it, batch in enumerate(loader):
            clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
            clips = clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))
            batch_dev = {
                "mtp_verbs": batch["mtp_verbs"].to(device),
                "mtp_nouns": batch["mtp_nouns"].to(device),
                "mtp_mask": batch["mtp_mask"].to(device),
            }

            if enc_hook is not None:
                enc_hook.importance = None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                x_full = base.encoder(clips)
            B, N, D = x_full.shape
            assert N == n_tok, f"expected N={n_tok}, got {N}"

            kept_by_mode = {}
            if "last16" in modes:
                kept_by_mode["last16"] = x_full[:, -keep_k:, :]

            if "enc_topk" in modes:
                assert enc_hook is not None and enc_hook.importance is not None
                idx = select_topk(enc_hook.importance, keep_k)
                kept_by_mode["enc_topk"] = gather_tokens(x_full, idx)
                for b in range(B):
                    slots = (idx[b].cpu().numpy() // GP).astype(np.int64)
                    np.add.at(slot_hist["enc_topk"], slots, 1)

            if "pred0_topk" in modes:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    imp = pred0_importance(base.predictor, x_full, chunk_size=args.chunk_size)
                idx = select_topk(imp, keep_k)
                kept_by_mode["pred0_topk"] = gather_tokens(x_full, idx)
                for b in range(B):
                    slots = (idx[b].cpu().numpy() // GP).astype(np.int64)
                    np.add.at(slot_hist["pred0_topk"], slots, 1)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for mode, x_kept in kept_by_mode.items():
                    tokens = run_predictor(base, x_kept, args.anticipation_sec)
                    outputs = mtp_clf(tokens)
                    update_metrics(
                        totals, counts, outputs, batch_dev, horizons,
                        verb_map, noun_map, action_map, device, prefix=mode,
                    )

            if it % args.log_every == 0:
                partial = summarize(totals, counts)
                msg = {k: round(100.0 * v, 2) for k, v in partial.items() if k.startswith(("last16/", "pred0_topk/", "enc_topk/"))}
                logger.info("itr=%d/%d elapsed=%.0fs %s", it, len(loader), time.time() - t0, msg)

    if enc_hook is not None:
        enc_hook.remove()

    metrics = summarize(totals, counts)
    # percent view for readability
    pct = {
        k: round(100.0 * v, 4)
        for k, v in metrics.items()
        if not str(k).startswith("n|") and isinstance(v, float)
    }
    # deltas vs last16
    deltas = {}
    for mode in modes:
        if mode == "last16":
            continue
        for h in horizons:
            a = f"{mode}/action_top5@{h:g}s"
            b = f"last16/action_top5@{h:g}s"
            if a in metrics and b in metrics:
                deltas[f"delta_{mode}_vs_last16@{h:g}s"] = round(100.0 * (metrics[a] - metrics[b]), 4)

    out = {
        "config": {
            "modes": modes,
            "keep_count": keep_k,
            "context_sec": args.context_sec,
            "n_frames": n_frames,
            "n_slots": n_slots,
            "n_tokens": n_tok,
            "max_samples": args.max_samples,
            "stride": args.stride,
            "n_eval": len(ds),
            "stream_ckpt": str(args.stream_ckpt),
            "horizons_sec": horizons,
            "anticipation_sec": args.anticipation_sec,
        },
        "metrics_frac": metrics,
        "metrics_pct": pct,
        "deltas_pp": deltas,
        "seconds": time.time() - t0,
    }
    for mode, hist in slot_hist.items():
        out[f"slot_keep_hist_{mode}"] = hist.tolist()

    out_path = args.out_dir / "metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    logger.info("metrics_pct=%s", json.dumps(pct, indent=2))
    logger.info("deltas_pp=%s", json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
