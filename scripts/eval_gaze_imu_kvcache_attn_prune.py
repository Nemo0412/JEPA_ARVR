#!/usr/bin/env python3
"""Gaze+IMU + streaming KV-cache attention prune. No predictor.

Streaming protocol
------------------
Cache = 128 frames. Each step takes 34 new frames. After encoding the current
128, last-block attention scores mark the lowest 34 frames; the next 34 replace
those slots. Gaze+pose enter via the 5ch concat adapter on every chunk; IMU is
fused on the packed 64 slots. Probe classifies fused encoder tokens.

Arms
  full128     one-shot encode of the newest 128 frames
  fifo        stream: drop oldest 34, encode new 34 vs remaining K/V
  attn_prune  stream: drop lowest-attention 34, encode new 34 vs remaining K/V
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
sys.path.insert(0, os.environ["VJEPA_ROOT"])

from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate  # noqa: E402
from app.hdepic_lora_action_anticipation.mtp import CommunicatingMLPMTPClassifier  # noqa: E402
from app.hdepic_lora_action_anticipation.pose_map_builder import GazePoseInputMapBuilder  # noqa: E402
from app.hdepic_lora_action_anticipation.stream_kvcache_attn_prune import (  # noqa: E402
    CACHE_FRAMES,
    NEW_FRAMES,
    GazeImuStreamKVModel,
)
from app.hdepic_lora_action_anticipation.train_stream_mtp_concat_ca import (  # noqa: E402
    build_concat_ca_model,
    default_gaze_cfg,
)
from app.hdepic_lora_action_anticipation.tri_modal_fusion import ImuTrajectoryLoader  # noqa: E402
from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402

import app.hdepic_lora_action_anticipation.train_stream_mtp as S  # noqa: E402

logger = logging.getLogger("gaze_imu_kvprune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

METHOD_NAME = "Gaze+IMU stream KV-attn-prune"
CA_RUN = Path(
    "/scratch/ll5914/experiments/concat_plus_cross_attn_v2/action_anticipation_frozen/"
    "concat-plus-ca-v2-L3-keepaux-softgate-vitl16-256-12ep-1xh100"
)
NOPRED_CKPT = Path("/scratch/ll5914/experiments/p01_stream_mtp_nopred_vanilla_2_4_6/latest.pt")


def _action_map_from_ckpt(raw) -> dict:
    out = {}
    for k, i in raw.items():
        if isinstance(k, tuple):
            out[k] = int(i)
        else:
            v, n = str(k).split(",")
            out[(int(v), int(n))] = int(i)
    return out


class Ctx64Dataset(Dataset):
    """Rebuild a fixed-length window ending at each stream tick."""

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
        kept, skipped = [], 0
        for r in rows:
            tick = int(r["tick_frame"])
            origin = int(r["origin_frame"])
            n_frames = int(r["n_frames"])
            vfps = float(r["vfps"])
            start = max(origin, tick - int(round(self.context_sec * vfps)))
            if tick - start < int(round(0.5 * self.context_sec * vfps)):
                start = max(0, tick - int(round(self.context_sec * vfps)))
            if tick <= start:
                skipped += 1
                continue
            frame_idx = np.linspace(start, max(start, tick - 1), self.n_model)
            frame_idx = np.clip(np.round(frame_idx).astype(np.int64), 0, n_frames - 1)
            if len(np.unique(frame_idx)) < self.n_model // 2:
                skipped += 1
                continue
            kept.append({**r, "frame_indices_64": frame_idx, "start_frame_64": int(start)})
        self.rows = kept
        logger.info("Ctx64Dataset: kept=%d skipped=%d n_model=%d", len(self.rows), skipped, self.n_model)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        video_id = str(r["video_id"])
        pid = video_id.split("_")[0]
        path = self.video_root / pid / f"{video_id}.MP4"
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1, width=self.img_size, height=self.img_size)
        try:
            frame_idx = np.clip(r["frame_indices_64"], 0, len(vr) - 1)
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


def sample_even_per_video(ds: Ctx64Dataset, clips_per_video: int, max_samples: int) -> Ctx64Dataset:
    by_vid: dict[str, list] = defaultdict(list)
    for r in ds.rows:
        by_vid[str(r["video_id"])].append(r)
    kept = []
    for vid in sorted(by_vid):
        rows = by_vid[vid]
        n_take = min(clips_per_video, len(rows))
        if n_take == 1:
            idxs = [0]
        else:
            idxs = np.linspace(0, len(rows) - 1, n_take).round().astype(int).tolist()
            seen, uniq = set(), []
            for i in idxs:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            idxs = uniq
        kept.extend(rows[i] for i in idxs)
        if max_samples > 0 and len(kept) >= max_samples:
            kept = kept[:max_samples]
            break
    ds.rows = kept
    logger.info("sampled %d clips from %d videos", len(kept), len({str(r["video_id"]) for r in kept}))
    return ds


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


def per_clip_hits(outputs, batch, horizons, verb_map, noun_map, action_map, device):
    out = {}
    mtp_verbs = batch["mtp_verbs"]
    mtp_nouns = batch["mtp_nouns"]
    mtp_mask = batch["mtp_mask"]
    for hi, h in enumerate(horizons):
        valid = mtp_mask[:, hi] > 0.5
        rec = {"valid": bool(valid.any()), "hit_top5": None, "hit_top1": None, "label": None, "pred1": None}
        if rec["valid"]:
            _v, _n, a_lab, keep = S.map_labels(
                mtp_verbs[valid, hi], mtp_nouns[valid, hi], verb_map, noun_map, action_map, device
            )
            if keep:
                valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                logits = outputs[float(h)]["action"][valid_pos].float()
                pred1 = int(logits.argmax(dim=-1)[0].item())
                top5 = logits.topk(min(5, logits.size(-1)), dim=-1).indices[0]
                lab = int(a_lab[0].item())
                rec["label"] = lab
                rec["pred1"] = pred1
                rec["hit_top1"] = pred1 == lab
                rec["hit_top5"] = lab in top5.tolist()
        out[f"{h:g}s"] = rec
    return out


def _stats(xs: list[float]) -> dict:
    xs_sorted = sorted(float(x) for x in xs)
    n = len(xs_sorted)
    return {
        "n": n,
        "mean_ms": float(statistics.fmean(xs_sorted)),
        "median_ms": float(statistics.median(xs_sorted)),
        "p90_ms": float(xs_sorted[max(0, int(0.9 * (n - 1)))]),
        "min_ms": float(xs_sorted[0]),
        "max_ms": float(xs_sorted[-1]),
    }


def _cuda_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


class GazeImuCtxDataset(Ctx64Dataset):
    """16s+ streaming window with gaze/pose maps and IMU trajectories."""

    def __init__(self, *args, gaze_cfg: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaze_cfg = dict(gaze_cfg)
        self._map_builder = None
        self._imu_loader = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_map_builder"] = None
        state["_imu_loader"] = None
        return state

    def _ensure_aux(self):
        if self._map_builder is None or self._imu_loader is None:
            gate = GazeTokenGate({**self.gaze_cfg, "mode": "token_gate", "learnable_gate": False})
            self._map_builder = GazePoseInputMapBuilder(self.gaze_cfg, gate=gate)
            self._imu_loader = ImuTrajectoryLoader(self.gaze_cfg, gate=gate)

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        r = self.rows[idx]
        frame_idx = np.asarray(r["frame_indices_64"], dtype=np.int64)
        video_id = str(r["video_id"])
        meta = {
            "video_id": video_id,
            "frame_indices": frame_idx,
            "vfps": float(r.get("vfps") or 30.0),
            "start_frame": int(r.get("start_frame_64") or frame_idx[0]),
        }
        clip = sample["clip"]
        t, h, w = int(clip.shape[1]), int(clip.shape[2]), int(clip.shape[3])
        self._ensure_aux()
        aux = self._map_builder.build_cpu([meta], t, h, w)[0]
        imu = self._imu_loader.load_batch([meta], torch.device("cpu"))
        sample["aux_map"] = aux
        sample["imu"] = imu[0][0] if imu is not None else torch.zeros(t, 128, 6)
        sample["imu_len"] = imu[1][0] if imu is not None else torch.ones(t, dtype=torch.long)
        return sample


def collate_gaze_imu(batch):
    out = {
        "clip": torch.stack([b["clip"] for b in batch], dim=0),
        "mtp_verbs": torch.stack([b["mtp_verbs"] for b in batch], dim=0),
        "mtp_nouns": torch.stack([b["mtp_nouns"] for b in batch], dim=0),
        "mtp_mask": torch.stack([b["mtp_mask"] for b in batch], dim=0),
        "video_id": [b["video_id"] for b in batch],
        "tick_frame": [b["tick_frame"] for b in batch],
        "aux_map": torch.stack([b["aux_map"] for b in batch], dim=0),
    }
    max_t = max(int(b["imu"].shape[0]) for b in batch)
    k = int(batch[0]["imu"].shape[1])
    imu = torch.zeros(len(batch), max_t, k, 6, dtype=torch.float32)
    imu_len = torch.zeros(len(batch), max_t, dtype=torch.long)
    for i, b in enumerate(batch):
        t = int(b["imu"].shape[0])
        imu[i, :t] = b["imu"]
        imu_len[i, :t] = b["imu_len"][:t]
    out["imu"] = imu
    out["imu_len"] = imu_len
    return out


def freeze_(module: torch.nn.Module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def build_model(device, args) -> GazeImuStreamKVModel:
    kwargs = dict(
        keep_count=10**9,
        freeze_adapter=True,
        freeze_fusion=True,
        freeze_encoder_lora=True,
        prune_mode="postfuse_recency",
    )
    sig = inspect.signature(build_concat_ca_model)
    if "no_predictor" in sig.parameters:
        kwargs["no_predictor"] = True
    if "ca_aux" in sig.parameters:
        kwargs["ca_aux"] = "imu"
    wrapped = build_concat_ca_model(
        device,
        CACHE_FRAMES,
        args.fps,
        args.img_size,
        str(args.checkpoint),
        str(args.encoder_lora) if args.encoder_lora else None,
        None,
        str(args.adapter_ckpt),
        str(args.fusion_ckpt) if args.fusion_ckpt else "",
        **kwargs,
    )
    freeze_(wrapped)
    model = GazeImuStreamKVModel(
        wrapped.concat_ca,
        cache_frames=CACHE_FRAMES,
        new_frames=NEW_FRAMES,
        keep_aux=bool(args.keep_aux),
        chunk=args.chunk,
    ).to(device)
    freeze_(model)
    return model


def load_probe(device, args, embed_dim: int):
    ck = torch.load(args.nopred_ckpt, map_location="cpu", weights_only=False)
    verb_map = ck["verb_map"]
    noun_map = ck["noun_map"]
    action_map = _action_map_from_ckpt(ck["action_map"])
    classifier = AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=int(embed_dim),
        num_heads=16,
        depth=4,
        use_activation_checkpointing=False,
    ).to(device)
    mtp_clf = CommunicatingMLPMTPClassifier(
        classifier, horizons_sec=[2.0, 4.0, 6.0], comm_layers=2, comm_heads=4
    ).to(device)
    miss, unexp = mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
    logger.info(
        "loaded probe %s missing=%d unexpected=%d epoch=%s",
        args.nopred_ckpt,
        len(miss),
        len(unexp),
        ck.get("epoch"),
    )
    freeze_(mtp_clf)
    return mtp_clf, {
        "epoch": ck.get("epoch"),
        "step": ck.get("step"),
        "phase": ck.get("phase"),
        "verb_map": verb_map,
        "noun_map": noun_map,
        "action_map": action_map,
    }


def plot_results(payload: dict, png_path: Path, copy_png: Path | None):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    colors = {"fifo": "#7f7f7f", "attn_prune": "#c45911", "full128": "#1f4e79"}
    labels = {"fifo": "FIFO drop 34", "attn_prune": "attn prune 34", "full128": "full last-128"}
    ax = axes[0]
    lat = payload.get("latency", {}).get("results", {})
    if lat:
        present = [k for k in ("fifo", "attn_prune", "full128") if k in lat]
        xs = np.arange(len(present))
        e2e = [lat[k]["e2e_ms"]["median_ms"] for k in present]
        w = 0.45
        ax.bar(xs, e2e, width=w, color=[colors[k] for k in present])
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[k] for k in present])
        ax.set_ylabel("ms (median)")
        ax.set_title("Steady-state step latency")
        ax.grid(True, axis="y", alpha=0.35)
    else:
        ax.set_title("Latency (not run)")
        ax.axis("off")

    ax = axes[1]
    table = payload.get("accuracy", {}).get("table_action_top5", {})
    horizons = ["2s", "4s", "6s"]
    present = [k for k in ("fifo", "attn_prune", "full128") if k in table]
    if table and present:
        xh = np.arange(len(horizons))
        w = 0.24
        for i, arm in enumerate(present):
            ys = [table[arm].get(f"@{h}", float("nan")) for h in horizons]
            ax.bar(xh + (i - 1) * w, ys, w, color=colors[arm], label=labels[arm])
        ax.set_xticks(xh)
        ax.set_xticklabels([f"+{h}" for h in horizons])
        ax.set_ylabel("Action Top-5 (%)")
        n = payload["accuracy"].get("n_clips", 0)
        ax.set_title(f"Accuracy  ({n} clips)")
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(fontsize=8)
    else:
        ax.set_title("Accuracy (not run)")
        ax.axis("off")

    fig.suptitle("Gaze+IMU  ·  KV cache 128, stream 34, attn prune", fontsize=12)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    if copy_png is not None:
        copy_png.parent.mkdir(parents=True, exist_ok=True)
        copy_png.write_bytes(png_path.read_bytes())


def run_latency(model: GazeImuStreamKVModel, mtp_clf, device, args) -> dict:
    cache_frames = CACHE_FRAMES
    new_frames = NEW_FRAMES
    img = args.img_size
    clips = torch.randn(1, 3, cache_frames + new_frames, img, img, device=device)
    aux = torch.zeros(1, 2, cache_frames + new_frames, img, img, device=device)
    imu = torch.zeros(1, cache_frames + new_frames, 128, 6, device=device)
    imu_len = torch.full((1, cache_frames + new_frames), 128, device=device, dtype=torch.long)

    def run_arm(mode: str):
        if mode == "full128":
            tok = model.forward_oneshot_last128(clips, aux, (imu, imu_len))
        else:
            tok = model.forward_window(clips, aux, (imu, imu_len), mode="attn" if mode == "attn_prune" else "fifo")
        _ = mtp_clf(tok, n_pred_per_horizon=None)
        return int(tok.size(1))

    results = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for mode in ("fifo", "attn_prune", "full128"):
            logger.info("latency warmup %s …", mode)
            for _ in range(args.warmup):
                run_arm(mode)

            def fused(mode=mode):
                run_arm(mode)

            torch.cuda.reset_peak_memory_stats()
            rows = [_cuda_ms(fused) for _ in range(args.repeats)]
            n_tok = run_arm(mode)
            rec = {
                "n_probe": n_tok,
                "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
                "e2e_ms": _stats(rows),
            }
            results[mode] = rec
            logger.info("  %s: e2e %.2f  n=%d", mode, rec["e2e_ms"]["median_ms"], n_tok)

    att = results["attn_prune"]["e2e_ms"]["median_ms"]
    fifo = results["fifo"]["e2e_ms"]["median_ms"]
    full = results["full128"]["e2e_ms"]["median_ms"]
    return {
        "device": torch.cuda.get_device_name(0),
        "cache_frames": cache_frames,
        "new_frames": new_frames,
        "results": results,
        "delta_median_ms": {
            "attn_prune_minus_fifo_e2e": att - fifo,
            "attn_prune_minus_full128_e2e": att - full,
            "fifo_minus_full128_e2e": fifo - full,
        },
    }


def run_accuracy(model, mtp_clf, device, args, ck_meta, gaze_cfg) -> dict:
    horizons = [float(x) for x in args.horizons_sec.split(",") if x.strip()]
    total_frames = CACHE_FRAMES + NEW_FRAMES * int(args.stream_steps)
    context_sec = total_frames / float(args.fps)
    ds = GazeImuCtxDataset(
        args.val_csv,
        args.video_root,
        gaze_cfg=gaze_cfg,
        context_sec=context_sec,
        model_fps=args.fps,
        img_size=args.img_size,
        max_samples=0,
        stride=args.stride,
        require_ctx_sec=args.require_ctx_sec if args.require_ctx_sec > 0 else None,
    )
    ds = sample_even_per_video(ds, args.clips_per_video, args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_gaze_imu,
        pin_memory=False,
    )
    verb_map, noun_map, action_map = ck_meta["verb_map"], ck_meta["noun_map"], ck_meta["action_map"]
    totals = defaultdict(float)
    counts = defaultdict(int)
    per_clip = []
    t0 = time.time()

    with torch.no_grad():
        for it, batch in enumerate(loader):
            clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
            clips = clips.sub_(S.IMAGENET_MEAN.to(device)).div_(S.IMAGENET_STD.to(device))
            aux = batch["aux_map"].to(device, non_blocking=True).float()
            imu = batch["imu"].to(device, non_blocking=True).float()
            imu_len = batch["imu_len"].to(device, non_blocking=True)
            if clips.size(2) != total_frames:
                raise RuntimeError(f"expected T={total_frames}, got {clips.size(2)}")
            batch_dev = {
                "mtp_verbs": batch["mtp_verbs"].to(device),
                "mtp_nouns": batch["mtp_nouns"].to(device),
                "mtp_mask": batch["mtp_mask"].to(device),
            }
            vid = batch["video_id"][0]
            tick = int(batch["tick_frame"][0])
            imu_batch = (imu, imu_len)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok_full = model.forward_oneshot_last128(clips, aux, imu_batch)
                tok_fifo = model.forward_window(clips, aux, imu_batch, mode="fifo")
                tok_attn = model.forward_window(clips, aux, imu_batch, mode="attn")
                out_full = mtp_clf(tok_full, n_pred_per_horizon=None)
                out_fifo = mtp_clf(tok_fifo, n_pred_per_horizon=None)
                out_attn = mtp_clf(tok_attn, n_pred_per_horizon=None)

            for prefix, out in (("full128", out_full), ("fifo", out_fifo), ("attn_prune", out_attn)):
                update_metrics(
                    totals, counts, out, batch_dev, horizons, verb_map, noun_map, action_map, device, prefix=prefix
                )
            rec = {
                "idx": it,
                "video_id": vid,
                "tick_frame": tick,
                "full128": per_clip_hits(out_full, batch_dev, horizons, verb_map, noun_map, action_map, device),
                "fifo": per_clip_hits(out_fifo, batch_dev, horizons, verb_map, noun_map, action_map, device),
                "attn_prune": per_clip_hits(out_attn, batch_dev, horizons, verb_map, noun_map, action_map, device),
            }
            per_clip.append(rec)
            if (it + 1) % args.log_every == 0 or it == 0:
                logger.info(
                    "clip %d/%d %s  full@2s=%s  fifo@2s=%s  attn@2s=%s",
                    it + 1,
                    len(loader),
                    vid,
                    rec["full128"]["2s"]["hit_top5"],
                    rec["fifo"]["2s"]["hit_top5"],
                    rec["attn_prune"]["2s"]["hit_top5"],
                )

    metrics = summarize(totals, counts)
    table = {}
    for prefix in ("fifo", "attn_prune", "full128"):
        table[prefix] = {
            f"@{h:g}s": round(100.0 * metrics.get(f"{prefix}/action_top5@{h:g}s", float("nan")), 4)
            for h in horizons
        }
    delta = {}
    for h in horizons:
        att = metrics.get(f"attn_prune/action_top5@{h:g}s", float("nan"))
        fifo = metrics.get(f"fifo/action_top5@{h:g}s", float("nan"))
        full = metrics.get(f"full128/action_top5@{h:g}s", float("nan"))
        delta[f"attn_prune_minus_fifo@{h:g}s"] = round(100.0 * (att - fifo), 4)
        delta[f"attn_prune_minus_full128@{h:g}s"] = round(100.0 * (att - full), 4)
    return {
        "method": METHOD_NAME,
        "table_action_top5": table,
        "delta_pp": delta,
        "n_clips": len(per_clip),
        "n_videos": len({r["video_id"] for r in per_clip}),
        "per_clip": per_clip,
        "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
        "seconds": time.time() - t0,
        "config": {
            "cache_frames": CACHE_FRAMES,
            "new_frames": NEW_FRAMES,
            "stream_steps": int(args.stream_steps),
            "total_frames": total_frames,
            "context_sec": context_sec,
            "prune": "encoder last-block slot attention, replace lowest 34 frames",
            "gaze": "5ch concat adapter per chunk",
            "imu": "late CA on packed 64 slots",
            "no_predictor": True,
            "keep_aux": bool(args.keep_aux),
            "encoder_lora": str(args.encoder_lora),
            "adapter": str(args.adapter_ckpt),
            "fusion": str(args.fusion_ckpt),
            "probe": str(args.nopred_ckpt),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", type=Path, default=Path(
        "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/HD_EPIC_val_stream_mtp.csv"
    ))
    ap.add_argument("--video-root", type=Path, default=Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"))
    ap.add_argument("--checkpoint", type=Path, default=Path("/scratch/ll5914/models/vjepa2/vitl.pt"))
    ap.add_argument("--nopred-ckpt", type=Path, default=NOPRED_CKPT)
    ap.add_argument("--encoder-lora", type=Path, default=CA_RUN / "encoder_lora_best.pt")
    ap.add_argument("--adapter-ckpt", type=Path, default=CA_RUN / "binary_input_adapter_best.pt")
    ap.add_argument("--fusion-ckpt", type=Path, default=CA_RUN / "tri_modal_fusion_best.pt")
    ap.add_argument("--gaze-root", type=Path, default=Path("/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze"))
    ap.add_argument("--gaze-extract-root", type=Path, default=Path("/scratch/ll5914/datasets/HD-EPIC/_gaze_extract"))
    ap.add_argument("--gaze-sync-root", type=Path, default=Path("/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos"))
    ap.add_argument(
        "--pose-slam-root",
        type=Path,
        default=Path("/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi"),
    )
    ap.add_argument(
        "--pose-mapping-json",
        type=Path,
        default=Path("/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi/vrs_to_multi_slam.json"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("/scratch/ll5914/experiments/gaze_imu_kvcache_attn_prune"))
    ap.add_argument("--copy-json", type=Path, default=Path("/home/ll5914/Jepa_yifan/gaze_imu_kvcache_attn_prune.json"))
    ap.add_argument("--copy-png", type=Path, default=Path("/home/ll5914/Jepa_yifan/gaze_imu_kvcache_attn_prune.png"))
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--stream-steps", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=108)
    ap.add_argument("--clips-per-video", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--require-ctx-sec", type=float, default=10.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--keep-aux", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-accuracy", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_frames = CACHE_FRAMES + NEW_FRAMES * int(args.stream_steps)
    logger.info(
        "loading Gaze+IMU concat-CA + nopred probe  cache=%df new=%df steps=%d T=%d keep_aux=%s",
        CACHE_FRAMES,
        NEW_FRAMES,
        args.stream_steps,
        total_frames,
        args.keep_aux,
    )
    model = build_model(device, args)
    mtp_clf, ck_meta = load_probe(device, args, model.embed_dim)
    gaze_cfg = default_gaze_cfg(
        str(args.gaze_root),
        str(args.gaze_extract_root),
        str(args.gaze_sync_root),
        str(args.pose_slam_root),
        str(args.pose_mapping_json),
        args.img_size,
    )

    payload = {
        "method": METHOD_NAME,
        "note": (
            "Gaze+pose 5ch adapter + IMU CA, no predictor. Streaming 34-frame "
            "chunks into a 128-frame KV cache; lowest-attention 34 frames are "
            "replaced on the next step."
        ),
        "device": torch.cuda.get_device_name(0),
    }
    if not args.skip_latency:
        payload["latency"] = run_latency(model, mtp_clf, device, args)
    if not args.skip_accuracy:
        payload["accuracy"] = run_accuracy(model, mtp_clf, device, args, ck_meta, gaze_cfg)

    png = args.out_dir / "gaze_imu_kvcache_attn_prune.png"
    plot_results(payload, png, args.copy_png)
    payload["png"] = str(png)

    out_path = args.out_dir / "metrics.json"
    text = json.dumps(payload, indent=2) + "\n"
    out_path.write_text(text, encoding="utf-8")
    if args.copy_json is not None:
        args.copy_json.parent.mkdir(parents=True, exist_ok=True)
        args.copy_json.write_text(text, encoding="utf-8")
    logger.info("wrote %s", out_path)
    if "latency" in payload:
        logger.info("latency deltas %s", json.dumps(payload["latency"]["delta_median_ms"]))
    if "accuracy" in payload:
        logger.info("accuracy table %s", json.dumps(payload["accuracy"]["table_action_top5"]))
        logger.info("accuracy delta_pp %s", json.dumps(payload["accuracy"]["delta_pp"]))


if __name__ == "__main__":
    main()
