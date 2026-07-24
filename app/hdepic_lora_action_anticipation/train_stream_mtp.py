#!/usr/bin/env python3
"""Streaming MTP train/val: growing context 4→6→8→10s, predict +2/+4/+6s.

Protocol (see scripts/make_hdepic_stream_half_split.py):
  - Each video split temporally: first half = train, second half = val/test.
  - Stream from half origin; tick every 2s; context grows then slides at 10s.
  - Communicating-MLP MTP heads (no RNN).
  - If encoder tokens exceed ``--keep-count``, attention-importance prune
    *before* the predictor (rebased positions) so long contexts fit.

Optional / separate from the fixed-clip anticipative eval path.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

from decord import VideoReader, cpu  # noqa: E402
from evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar import (  # noqa: E402
    init_module as init_anticipative_module,
)
from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402
from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

from app.hdepic_lora_action_anticipation.mtp import CommunicatingMLPMTPClassifier  # noqa: E402

logger = logging.getLogger("stream_mtp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


# ── prune (post-encoder, before predictor) ───────────────────────────────────
class TokenPruner:
    """Attention-importance prune; keep top-K in original order (multiple of gp)."""

    def __init__(self, encoder: nn.Module, keep_count: int, gp: int, chunk_size: int = 256):
        self.keep_count = max(gp, (keep_count // gp) * gp)
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
                pruner._importance = imp
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=m.proj_drop_prob, is_causal=m.is_causal, attn_mask=attn_mask
                )
            x = x.transpose(1, 2).reshape(B, N, C)
            x = m.proj(x)
            x = m.proj_drop(x)
            return x

        m.forward = _patched_forward

    def prune(self, feats: torch.Tensor):
        if self._importance is None:
            raise RuntimeError("Run encoder before TokenPruner.prune()")
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


class PrunedAnticipativeModel(nn.Module):
    """Encode → optional prune → predictor with rebased context positions."""

    def __init__(self, base: nn.Module, pruner: TokenPruner | None, prune_threshold: int):
        super().__init__()
        self.base = base
        self.pruner = pruner
        self.prune_threshold = int(prune_threshold)
        self.embed_dim = getattr(base, "embed_dim", None)

    def forward(self, x, anticipation_times):
        core = self.base
        x_full = core.encoder(x)
        B, N, D_full = x_full.size()
        embed_dim = core.encoder.embed_dim
        if self.pruner is not None and N > self.prune_threshold:
            x_full, _ = self.pruner.prune(x_full)
            B, N, D_full = x_full.size()
        use_hierarchical = D_full > embed_dim
        x = x_full[:, :, -embed_dim:] if use_hierarchical else x_full
        x_accumulate = x.clone()
        ctxt_positions = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
        anticipation_steps = (anticipation_times * core.frames_per_second / core.tubelet_size).to(torch.int64)
        skip_positions = N + int(core.grid_size**2) * anticipation_steps
        N_pred = int(core.grid_size**2 * (core.num_output_frames // core.tubelet_size))
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


# ── data ─────────────────────────────────────────────────────────────────────
def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if x.strip() != ""]


class StreamMTPDataset(Dataset):
    def __init__(self, csv_path: Path, video_root: Path, img_size: int = 256):
        self.video_root = Path(video_root)
        self.img_size = int(img_size)
        self.rows = []
        with Path(csv_path).open() as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        self._readers: dict[str, VideoReader] = {}

    def __len__(self):
        return len(self.rows)

    def _vr(self, video_id: str) -> VideoReader:
        if video_id not in self._readers:
            pid = video_id.split("_")[0]
            path = self.video_root / pid / f"{video_id}.MP4"
            self._readers[video_id] = VideoReader(
                str(path), ctx=cpu(0), num_threads=1, width=self.img_size, height=self.img_size
            )
        return self._readers[video_id]

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        video_id = str(r["video_id"])
        frame_idx = np.asarray(_parse_int_list(r["frame_indices"]), dtype=np.int64)
        vr = self._vr(video_id)
        frame_idx = np.clip(frame_idx, 0, len(vr) - 1)
        frames = vr.get_batch(frame_idx.tolist()).asnumpy()  # T,H,W,C
        clip = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()  # C,T,H,W uint8
        return {
            "clip": clip,
            "context_sec": float(r["context_sec"]),
            "n_frames": int(clip.shape[1]),
            "mtp_verbs": torch.tensor(_parse_int_list(r["mtp_verbs"]), dtype=torch.long),
            "mtp_nouns": torch.tensor(_parse_int_list(r["mtp_nouns"]), dtype=torch.long),
            "mtp_mask": torch.tensor(_parse_float_list(r["mtp_mask"]), dtype=torch.float32),
        }


class ContextBucketBatchSampler(Sampler[list[int]]):
    """Batches samples that share the same n_model_frames (variable-length safe)."""

    def __init__(self, dataset: StreamMTPDataset, batch_size: int, shuffle: bool, seed: int = 0):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, r in enumerate(dataset.rows):
            buckets[int(r["n_model_frames"])].append(i)
        self.buckets = dict(buckets)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        batches = []
        for _, idxs in self.buckets.items():
            order = list(idxs)
            if self.shuffle:
                rng.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                batches.append(order[i : i + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum(math.ceil(len(v) / self.batch_size) for v in self.buckets.values())


def collate_stream(batch):
    clips = torch.stack([b["clip"] for b in batch], dim=0)  # B,C,T,H,W uint8
    return {
        "clip": clips,
        "mtp_verbs": torch.stack([b["mtp_verbs"] for b in batch], dim=0),
        "mtp_nouns": torch.stack([b["mtp_nouns"] for b in batch], dim=0),
        "mtp_mask": torch.stack([b["mtp_mask"] for b in batch], dim=0),
        "context_sec": torch.tensor([b["context_sec"] for b in batch], dtype=torch.float32),
    }


# ── metrics / vocab ──────────────────────────────────────────────────────────
def load_action_maps(train_csv: Path):
    verbs, nouns, actions = {}, {}, {}
    with Path(train_csv).open() as f:
        for row in csv.DictReader(f):
            vs = _parse_int_list(row["mtp_verbs"])
            ns = _parse_int_list(row["mtp_nouns"])
            ms = _parse_float_list(row["mtp_mask"])
            for v, n, m in zip(vs, ns, ms):
                if m < 0.5 or v < 0 or n < 0:
                    continue
                if v not in verbs:
                    verbs[v] = len(verbs)
                if n not in nouns:
                    nouns[n] = len(nouns)
                if (v, n) not in actions:
                    actions[(v, n)] = len(actions)
    return verbs, nouns, actions


def topk_acc(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    if labels.numel() == 0:
        return float("nan")
    k = min(k, logits.size(-1))
    pred = logits.topk(k, dim=-1).indices
    return float((pred == labels.unsqueeze(-1)).any(dim=-1).float().mean().item())


def map_labels(verbs, nouns, verb_map, noun_map, action_map, device):
    v_out, n_out, a_out, keep = [], [], [], []
    for i, (v, n) in enumerate(zip(verbs.tolist(), nouns.tolist())):
        if v not in verb_map or n not in noun_map or (v, n) not in action_map:
            continue
        v_out.append(verb_map[v])
        n_out.append(noun_map[n])
        a_out.append(action_map[(v, n)])
        keep.append(i)
    if not keep:
        z = torch.zeros(0, device=device, dtype=torch.long)
        return z, z, z, keep
    return (
        torch.tensor(v_out, device=device, dtype=torch.long),
        torch.tensor(n_out, device=device, dtype=torch.long),
        torch.tensor(a_out, device=device, dtype=torch.long),
        keep,
    )


# ── model build / warm start ─────────────────────────────────────────────────
def build_model(device, max_frames: int, fps: int, img_size: int, checkpoint: str):
    model_kwargs = {
        "use_v2_1": False,
        "encoder": {
            "model_name": "vit_large",
            "checkpoint_key": "target_encoder",
            "tubelet_size": 2,
            "patch_size": 16,
            "uniform_power": True,
            "use_rope": True,
        },
        "predictor": {
            "model_name": "vit_predictor",
            "checkpoint_key": "predictor",
            "num_frames": 64,
            "depth": 12,
            "num_heads": 12,
            "predictor_embed_dim": 384,
            "num_mask_tokens": 10,
            "uniform_power": True,
            "use_mask_tokens": True,
            "use_sdpa": True,
            "use_silu": False,
            "wide_silu": False,
            "use_rope": True,
        },
    }
    wrapper_kwargs = {"no_predictor": False, "num_output_frames": 2, "num_steps": 1}
    model = init_anticipative_module(
        frames_per_clip=max_frames,
        frames_per_second=fps,
        resolution=img_size,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    ).to(device)
    return model


def load_lora_sidecars(model, enc_path: str | None, pred_path: str | None):
    try:
        from app.hdepic_lora_action_anticipation.encoder_lora import (
            inject_encoder_lora,
            load_encoder_lora_checkpoint,
            set_encoder_lora_trainable,
        )
        from app.hdepic_lora_action_anticipation.predictor_lora import (
            inject_predictor_lora,
            load_predictor_lora_checkpoint,
            set_predictor_lora_trainable,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LoRA helpers unavailable (%s); continuing without LoRA sidecars", exc)
        return
    if enc_path and Path(enc_path).is_file():
        inject_encoder_lora(model, rank=8, alpha=16.0, dropout=0.05, last_n_blocks=0)
        load_encoder_lora_checkpoint(model, enc_path)
        set_encoder_lora_trainable(model, trainable=False)
        logger.info("Loaded encoder LoRA from %s (frozen)", enc_path)
    if pred_path and Path(pred_path).is_file():
        inject_predictor_lora(model, rank=8, alpha=16.0, dropout=0.05, last_n_blocks=0)
        load_predictor_lora_checkpoint(model, pred_path)
        set_predictor_lora_trainable(model, trainable=True)
        logger.info("Loaded predictor LoRA from %s (trainable)", pred_path)


# ── train / val loops ────────────────────────────────────────────────────────
def run_epoch(
    model,
    classifier,
    loader,
    device,
    horizons,
    weights,
    primary_idx,
    verb_map,
    noun_map,
    action_map,
    optimizer=None,
    scaler=None,
    train: bool = True,
    anticipation_sec: float = 2.0,
    log_every: int = 20,
):
    model.train(mode=train)
    classifier.train(mode=train)
    crit = nn.CrossEntropyLoss()
    totals = defaultdict(float)
    counts = defaultdict(int)
    loss_meter = 0.0
    n_steps = 0
    t0 = time.time()
    for it, batch in enumerate(loader):
        clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
        clips = clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))
        mtp_verbs = batch["mtp_verbs"].to(device, non_blocking=True)
        mtp_nouns = batch["mtp_nouns"].to(device, non_blocking=True)
        mtp_mask = batch["mtp_mask"].to(device, non_blocking=True)
        B = clips.size(0)
        ant = torch.full((B,), float(anticipation_sec), device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            tokens = model(clips, ant)
            outputs = classifier(tokens)
            head_loss = clips.new_zeros(())
            for hi, h in enumerate(horizons):
                valid = mtp_mask[:, hi] > 0.5
                if not bool(valid.any()):
                    continue
                v_lab, n_lab, a_lab, keep = map_labels(
                    mtp_verbs[valid, hi],
                    mtp_nouns[valid, hi],
                    verb_map,
                    noun_map,
                    action_map,
                    device,
                )
                if not keep:
                    continue
                valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                o = outputs[float(h)]
                step_loss = (
                    crit(o["verb"][valid_pos], v_lab)
                    + crit(o["noun"][valid_pos], n_lab)
                    + crit(o["action"][valid_pos], a_lab)
                )
                head_loss = head_loss + float(weights[hi]) * step_loss

                with torch.no_grad():
                    key = f"action_top5@{h:g}s"
                    totals[key] += topk_acc(o["action"][valid_pos].float(), a_lab, k=5) * len(keep)
                    counts[key] += len(keep)

            # primary metric
            h0 = horizons[primary_idx]
            valid = mtp_mask[:, primary_idx] > 0.5
            if bool(valid.any()):
                v_lab, n_lab, a_lab, keep = map_labels(
                    mtp_verbs[valid, primary_idx],
                    mtp_nouns[valid, primary_idx],
                    verb_map,
                    noun_map,
                    action_map,
                    device,
                )
                if keep:
                    valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                    o = outputs[float(h0)]
                    totals["primary_action_top5"] += topk_acc(o["action"][valid_pos].float(), a_lab, k=5) * len(keep)
                    counts["primary_action_top5"] += len(keep)

        if train:
            if not torch.isfinite(head_loss.detach()):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(head_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()

        loss_meter += float(head_loss.detach().item()) if torch.isfinite(head_loss.detach()) else 0.0
        n_steps += 1
        if it % log_every == 0:
            logger.info(
                "%s itr=%d/%d loss=%.4f primary@%gs top5≈%.1f ctx=%.0fs",
                "train" if train else "val",
                it,
                len(loader),
                loss_meter / max(1, n_steps),
                horizons[primary_idx],
                100.0 * totals["primary_action_top5"] / max(1, counts["primary_action_top5"]),
                float(batch["context_sec"][0]),
            )
    metrics = {k: (totals[k] / max(1, counts[k])) for k in totals}
    metrics["loss"] = loss_meter / max(1, n_steps)
    metrics["seconds"] = time.time() - t0
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=None)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--loss-weights", type=str, default="1.0,0.7,0.5")
    ap.add_argument("--primary-horizon-sec", type=float, default=2.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=80)  # 10s @ 8fps
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    weights = [float(x) for x in args.loss_weights.split(",")]
    assert len(horizons) == len(weights)
    primary_h = float(args.primary_horizon_sec)
    primary_idx = horizons.index(primary_h) if primary_h in horizons else 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    verb_map, noun_map, action_map = load_action_maps(args.train_csv)
    logger.info("vocab verbs=%d nouns=%d actions=%d", len(verb_map), len(noun_map), len(action_map))

    train_ds = StreamMTPDataset(args.train_csv, args.video_root, args.img_size)
    val_ds = StreamMTPDataset(args.val_csv, args.video_root, args.img_size)
    train_sampler = ContextBucketBatchSampler(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_sampler = ContextBucketBatchSampler(val_ds, args.batch_size, shuffle=False, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_stream,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_stream,
        pin_memory=True,
    )

    base = build_model(device, args.max_frames, args.fps, args.img_size, str(args.checkpoint))
    for p in base.encoder.parameters():
        p.requires_grad = False
    load_lora_sidecars(
        base,
        str(args.encoder_lora) if args.encoder_lora else None,
        str(args.predictor_lora) if args.predictor_lora else None,
    )
    gp = int(base.grid_size**2)
    pruner = TokenPruner(base.encoder, keep_count=args.keep_count, gp=gp)
    model = PrunedAnticipativeModel(base, pruner, prune_threshold=args.keep_count).to(device)

    classifier = AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=int(base.encoder.embed_dim),
        num_heads=16,
        depth=4,
        use_activation_checkpointing=True,
    ).to(device)
    # Stream-half vocab ≠ clip_split probe vocab → train heads from scratch;
    # backbone LoRA still warms from video joint.
    if args.probe:
        logger.info("Ignoring --probe=%s (vocab mismatch with stream half-split); heads from scratch", args.probe)
    for name, p in classifier.named_parameters():
        p.requires_grad = name.startswith(("verb_classifier.", "noun_classifier.", "action_classifier."))
        if name.startswith("pooler."):
            p.requires_grad = True  # stream protocol: allow light pooler adapt
    mtp_clf = CommunicatingMLPMTPClassifier(
        classifier, horizons_sec=horizons, comm_layers=2, comm_heads=4
    ).to(device)

    params = [p for p in list(model.parameters()) + list(mtp_clf.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best = -1.0
    history = []
    start_epoch = 0
    latest = args.out_dir / "latest.pt"
    if latest.is_file():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best = float(ck.get("best", -1.0))
        logger.info("Resumed from %s epoch=%d best=%.4f", latest, start_epoch, best)

    if args.val_only:
        metrics = run_epoch(
            model, mtp_clf, val_loader, device, horizons, weights, primary_idx,
            verb_map, noun_map, action_map, train=False, anticipation_sec=args.anticipation_sec,
        )
        logger.info("VAL_ONLY metrics: %s", json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}))
        (args.out_dir / "val_only_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        tr = run_epoch(
            model, mtp_clf, train_loader, device, horizons, weights, primary_idx,
            verb_map, noun_map, action_map, optimizer=optimizer, scaler=scaler, train=True,
            anticipation_sec=args.anticipation_sec,
        )
        va = run_epoch(
            model, mtp_clf, val_loader, device, horizons, weights, primary_idx,
            verb_map, noun_map, action_map, train=False, anticipation_sec=args.anticipation_sec,
        )
        primary = float(va.get("primary_action_top5", 0.0))
        logger.info(
            "epoch %d train_loss=%.4f val_primary_top5=%.2f%% %s",
            epoch,
            tr["loss"],
            100.0 * primary,
            {k: round(100.0 * va[k], 2) for k in va if k.startswith("action_top5")},
        )
        history.append({"epoch": epoch, "train": tr, "val": va})
        ck = {
            "epoch": epoch,
            "model": model.state_dict(),
            "mtp_classifier": mtp_clf.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best": best,
            "horizons": horizons,
            "verb_map": verb_map,
            "noun_map": noun_map,
            "action_map": {f"{v},{n}": i for (v, n), i in action_map.items()},
        }
        torch.save(ck, latest)
        if primary > best:
            best = primary
            ck["best"] = best
            torch.save(ck, args.out_dir / "best.pt")
            logger.info("New best primary_action_top5=%.4f", best)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    logger.info("Done. best primary_action_top5=%.4f", best)


if __name__ == "__main__":
    main()
