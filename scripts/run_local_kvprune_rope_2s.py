#!/usr/bin/env python3
"""Local kvprune_rope — 2s-only, from-scratch probe, multi-GPU encode.

Protocol:
  fill 128f → Probe.blocks[0] attn scores → drop 34 / keep 94 → encode new 34
  → packed 128 → probe. RoPE: only_block0 + abs frame/slot ids.

Bottleneck is encode/decode → shard train-cache + val-eval across GPUs.
Probe FT stays on GPU 0 (cached tokens, fast).
"""
from __future__ import annotations

import argparse
import copy
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
import torch.multiprocessing as mp
import torch.nn as nn
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", str(PROJECT_ROOT / "vjepa2")))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

from app.hdepic_lora_action_anticipation.stream_kvcache_attn_prune import (  # noqa: E402
    CACHE_FRAMES,
    NEW_FRAMES,
    ProbeTemporalRoPE,
    StreamKVAttnPruneEncoder,
    probe_blk0_slot_scores,
    token_frame_ids_from_slots,
)
from app.hdepic_lora_action_anticipation.train_stream_mtp import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    map_labels,
    topk_acc,
)
from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402

logger = logging.getLogger("kvprune2s")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GP = 256
VIDEO_FPS = 30.0
HORIZON = 2.0


class ClipAnticipationDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        video_root: Path,
        *,
        horizon_sec: float = 2.0,
        model_fps: float = 8.0,
        video_fps: float = VIDEO_FPS,
        img_size: int = 256,
        n_model: int = CACHE_FRAMES + NEW_FRAMES,
        max_samples: int = 0,
    ):
        self.video_root = Path(video_root)
        self.img_size = int(img_size)
        self.n_model = int(n_model)
        self.horizon_sec = float(horizon_sec)
        self.model_fps = float(model_fps)
        self.video_fps = float(video_fps)
        self.context_sec = self.n_model / self.model_fps

        rows = list(csv.DictReader(Path(csv_path).open()))
        if max_samples > 0:
            rows = rows[:max_samples]

        kept, skipped = [], 0
        need_native = int(round((self.context_sec + self.horizon_sec) * self.video_fps))
        for r in rows:
            start = int(r["start_frame"])
            if start < need_native:
                skipped += 1
                continue
            obs_end = start - int(round(self.horizon_sec * self.video_fps))
            obs_start = obs_end - int(round(self.context_sec * self.video_fps))
            if obs_start < 0:
                skipped += 1
                continue
            frame_idx = np.round(
                np.linspace(obs_start, max(obs_start, obs_end - 1), self.n_model)
            ).astype(np.int64)
            kept.append(
                {
                    "video_id": str(r["video_id"]),
                    "frame_idx": frame_idx,
                    "verb": int(r["verb_class"]),
                    "noun": int(r["noun_class"]),
                }
            )
        self.rows = kept
        logger.info(
            "ClipAnticipationDataset %s: kept=%d skipped=%d T=%d horizon=%.1fs",
            Path(csv_path).name, len(self.rows), skipped, self.n_model, self.horizon_sec,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        video_id = r["video_id"]
        pid = video_id.split("_")[0]
        path = self.video_root / pid / f"{video_id}.MP4"
        if not path.is_file():
            path = self.video_root / pid / f"{video_id}.mp4"
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1, width=self.img_size, height=self.img_size)
        try:
            fi = np.clip(r["frame_idx"], 0, len(vr) - 1)
            frames = vr.get_batch(fi.tolist()).asnumpy()
        finally:
            del vr
        clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).contiguous()
        return {
            "clip": clip,
            "verb": torch.tensor(r["verb"], dtype=torch.long),
            "noun": torch.tensor(r["noun"], dtype=torch.long),
            "video_id": video_id,
            "idx": idx,
        }


def collate(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch], dim=0),
        "verb": torch.stack([b["verb"] for b in batch], dim=0),
        "noun": torch.stack([b["noun"] for b in batch], dim=0),
        "video_id": [b["video_id"] for b in batch],
        "idx": [b["idx"] for b in batch],
    }


def build_class_maps(train_csv: Path):
    verbs, nouns, actions = {}, {}, {}
    with Path(train_csv).open() as f:
        for r in csv.DictReader(f):
            v, n = int(r["verb_class"]), int(r["noun_class"])
            if v not in verbs:
                verbs[v] = len(verbs)
            if n not in nouns:
                nouns[n] = len(nouns)
            if (v, n) not in actions:
                actions[(v, n)] = len(actions)
    return verbs, nouns, actions


def normalize_clip(clip_uint8: torch.Tensor, device) -> torch.Tensor:
    clips = clip_uint8.to(device, non_blocking=True).float().div_(255.0)
    return clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))


def encode_probe_attn_pruned(stream, pooler, clips, embed_dim, *, chunk=256):
    hist = clips[:, :, :CACHE_FRAMES]
    new = clips[:, :, CACHE_FRAMES:]
    state = stream.fill(hist)
    tok0 = state.tokens
    if tok0.size(-1) != embed_dim:
        tok0 = tok0[:, :, -embed_dim:]
    scores = probe_blk0_slot_scores(pooler, tok0, stream.gp, chunk=chunk)
    state = stream.step(state, new, mode="attn", slot_scores=scores)
    tok = state.tokens
    if tok.size(-1) != embed_dim:
        tok = tok[:, :, -embed_dim:]
    frame_ids = token_frame_ids_from_slots(state.slot_ids, stream.gp)
    return tok, frame_ids


def _make_clf(maps, embed_dim, probe_heads, probe_depth, device, state=None):
    verb_map, noun_map, action_map = maps
    clf = AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=embed_dim,
        num_heads=probe_heads,
        depth=probe_depth,
        use_activation_checkpointing=True,
    ).to(device)
    if state is not None:
        clf.load_state_dict(state)
    return clf


def _build_encoder(device, total_frames, fps, img_size, checkpoint):
    model = build_model(device, total_frames, fps, img_size, str(checkpoint), no_predictor=True)
    return model.encoder, int(model.embed_dim), int(model.grid_size) ** 2


def _worker_encode_cache(rank, device_id, indices, args_dict, maps, probe_state, out_path):
    """Encode a shard of train indices → list of cached dicts on disk."""
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [gpu{device_id}] %(message)s")
    log = logging.getLogger(f"cache{device_id}")

    total_frames = CACHE_FRAMES + NEW_FRAMES
    encoder, embed_dim, gp = _build_encoder(
        device, total_frames, args_dict["fps"], args_dict["img_size"], args_dict["checkpoint"]
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    clf = _make_clf(maps, embed_dim, args_dict["probe_heads"], args_dict["probe_depth"], device, probe_state)
    clf.eval()
    stream = StreamKVAttnPruneEncoder(
        encoder, gp=gp, tubelet_size=2, cache_frames=CACHE_FRAMES, new_frames=NEW_FRAMES, chunk=args_dict["chunk"]
    )
    ds = ClipAnticipationDataset(
        Path(args_dict["train_csv"]), Path(args_dict["video_root"]),
        horizon_sec=HORIZON, model_fps=args_dict["fps"], img_size=args_dict["img_size"],
        max_samples=args_dict["max_train"],
    )
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)

    cached = []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            clips = normalize_clip(batch["clip"], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok, frame_ids = encode_probe_attn_pruned(
                    stream, clf.pooler, clips, embed_dim, chunk=args_dict["chunk"]
                )
            cached.append(
                {
                    "tok": tok.float().cpu(),
                    "frame_ids": frame_ids.cpu(),
                    "verb": batch["verb"].clone(),
                    "noun": batch["noun"].clone(),
                    "idx": int(batch["idx"][0]),
                }
            )
            if (i + 1) % 25 == 0:
                log.info("cache %d/%d (%.1fs)", i + 1, len(indices), time.time() - t0)
    torch.save(cached, out_path)
    log.info("wrote %s n=%d in %.1fs", out_path, len(cached), time.time() - t0)


def _worker_eval(rank, device_id, indices, args_dict, maps, probe_state, use_rope, out_path):
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [gpu{device_id}] %(message)s")
    log = logging.getLogger(f"eval{device_id}")

    total_frames = CACHE_FRAMES + NEW_FRAMES
    encoder, embed_dim, gp = _build_encoder(
        device, total_frames, args_dict["fps"], args_dict["img_size"], args_dict["checkpoint"]
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    clf = _make_clf(maps, embed_dim, args_dict["probe_heads"], args_dict["probe_depth"], device, probe_state)
    clf.eval()
    stream = StreamKVAttnPruneEncoder(
        encoder, gp=gp, tubelet_size=2, cache_frames=CACHE_FRAMES, new_frames=NEW_FRAMES, chunk=args_dict["chunk"]
    )
    rope = None
    if use_rope:
        rope = ProbeTemporalRoPE(clf.pooler, rope_cross_attn_k=False, only_block0=True)

    ds = ClipAnticipationDataset(
        Path(args_dict["val_csv"]), Path(args_dict["video_root"]),
        horizon_sec=HORIZON, model_fps=args_dict["fps"], img_size=args_dict["img_size"],
        max_samples=args_dict["max_val"],
    )
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    verb_map, noun_map, action_map = maps
    totals = defaultdict(float)
    counts = defaultdict(int)
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            clips = normalize_clip(batch["clip"], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok, frame_ids = encode_probe_attn_pruned(
                    stream, clf.pooler, clips, embed_dim, chunk=args_dict["chunk"]
                )
                if rope is not None:
                    rope.set_frame_ids(frame_ids)
                try:
                    out = clf(tok)
                finally:
                    if rope is not None:
                        rope.set_frame_ids(None)
            v_lab, n_lab, a_lab, keep = map_labels(
                batch["verb"].to(device), batch["noun"].to(device), verb_map, noun_map, action_map, device
            )
            if not keep:
                continue
            logits = out["action"][keep].float()
            totals["top5"] += topk_acc(logits, a_lab, k=5) * len(keep)
            totals["top1"] += topk_acc(logits, a_lab, k=1) * len(keep)
            counts["n"] += len(keep)
            if (i + 1) % 25 == 0:
                log.info("eval %d/%d (%.1fs)", i + 1, len(indices), time.time() - t0)
    if rope is not None:
        rope.remove()
    payload = {"top5_sum": totals["top5"], "top1_sum": totals["top1"], "n": counts["n"]}
    torch.save(payload, out_path)
    log.info("eval shard done n=%d in %.1fs", counts["n"], time.time() - t0)


def shard_indices(n: int, n_shards: int) -> list[list[int]]:
    idxs = list(range(n))
    shards = [idxs[i::n_shards] for i in range(n_shards)]
    return shards


def run_sharded(worker_fn, device_ids, shards, *extra_args):
    """Spawn one process per non-empty shard."""
    procs = []
    active = []
    for rank, (dev, idxs) in enumerate(zip(device_ids, shards)):
        if not idxs:
            continue
        active.append((rank, dev, idxs))
    ctx = mp.get_context("spawn")
    for rank, dev, idxs in active:
        args = (rank, dev, idxs) + extra_args
        p = ctx.Process(target=worker_fn, args=args)
        p.start()
        procs.append(p)
        logger.info("spawned worker rank=%d gpu=%d n=%d", rank, dev, len(idxs))
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker exited with code {p.exitcode}")


def finetune(clf, cached, *, rope, epochs, lr, maps, device):
    for p in clf.parameters():
        p.requires_grad = True
    clf.train()
    opt = torch.optim.AdamW([p for p in clf.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    crit = nn.CrossEntropyLoss()
    verb_map, noun_map, action_map = maps
    t0 = time.time()
    n_step = 0
    loss_meter = 0.0
    for ep in range(epochs):
        order = np.random.permutation(len(cached))
        for idx in order:
            item = cached[int(idx)]
            tok = item["tok"].to(device, non_blocking=True)
            frame_ids = item["frame_ids"].to(device, non_blocking=True)
            verb = item["verb"].to(device)
            noun = item["noun"].to(device)
            if rope is not None:
                rope.set_frame_ids(frame_ids)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = clf(tok)
                    v_lab, n_lab, a_lab, keep = map_labels(verb, noun, verb_map, noun_map, action_map, device)
                    if not keep:
                        continue
                    loss = crit(out["action"][keep], a_lab)
                    if "verb" in out:
                        loss = loss + crit(out["verb"][keep], v_lab) + crit(out["noun"][keep], n_lab)
            finally:
                if rope is not None:
                    rope.set_frame_ids(None)
            if not torch.isfinite(loss.detach()):
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in clf.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            loss_meter += float(loss.detach())
            n_step += 1
        logger.info(
            "epoch %d/%d steps=%d loss=%.4f rope=%s",
            ep + 1, epochs, n_step, loss_meter / max(1, n_step), rope is not None,
        )
    return time.time() - t0


def merge_eval(shard_paths):
    top5 = top1 = n = 0.0
    for p in shard_paths:
        d = torch.load(p, map_location="cpu")
        top5 += d["top5_sum"]
        top1 += d["top1_sum"]
        n += d["n"]
    n = max(1, int(n))
    return {"top5": round(100.0 * top5 / n, 4), "top1": round(100.0 * top1 / n, 4), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=Path(
        "/mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_train_vjepa.csv"
    ))
    ap.add_argument("--val-csv", type=Path, default=Path(
        "/mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv"
    ))
    ap.add_argument("--video-root", type=Path, default=Path("/mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_videos"))
    ap.add_argument("--checkpoint", type=Path, default=Path("/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_rope_2s"))
    ap.add_argument("--device-ids", type=str, default="0,1,2,3", help="comma-separated GPU ids")
    ap.add_argument("--max-train", type=int, default=800)
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--probe-depth", type=int, default=4)
    ap.add_argument("--probe-heads", type=int, default=16)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(exist_ok=True)

    device_ids = [int(x) for x in args.device_ids.split(",") if x.strip() != ""]
    if not device_ids:
        raise SystemExit("need at least one device id")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    logger.info("using GPUs %s (%d cards)", device_ids, len(device_ids))
    ft_device = torch.device(f"cuda:{device_ids[0]}")
    torch.cuda.set_device(device_ids[0])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_frames = CACHE_FRAMES + NEW_FRAMES
    verb_map, noun_map, action_map = build_class_maps(args.train_csv)
    maps = (verb_map, noun_map, action_map)
    logger.info("classes verb=%d noun=%d action=%d", len(verb_map), len(noun_map), len(action_map))

    # Probe init on FT device (shared weights for all workers via state_dict)
    logger.info("init probe + encoder on gpu %d", device_ids[0])
    encoder, embed_dim, gp = _build_encoder(
        ft_device, total_frames, args.fps, args.img_size, args.checkpoint
    )
    if gp != GP:
        raise RuntimeError(f"expected gp={GP}, got {gp}")
    clf = _make_clf(maps, embed_dim, args.probe_heads, args.probe_depth, ft_device)
    init_state = copy.deepcopy(clf.state_dict())

    train_ds = ClipAnticipationDataset(
        args.train_csv, args.video_root, horizon_sec=HORIZON, model_fps=args.fps,
        img_size=args.img_size, max_samples=args.max_train,
    )
    val_ds = ClipAnticipationDataset(
        args.val_csv, args.video_root, horizon_sec=HORIZON, model_fps=args.fps,
        img_size=args.img_size, max_samples=args.max_val,
    )
    logger.info("train=%d val=%d", len(train_ds), len(val_ds))

    args_dict = {
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "video_root": str(args.video_root),
        "checkpoint": str(args.checkpoint),
        "max_train": args.max_train,
        "max_val": args.max_val,
        "fps": args.fps,
        "img_size": args.img_size,
        "chunk": args.chunk,
        "probe_heads": args.probe_heads,
        "probe_depth": args.probe_depth,
    }

    # Free FT-device model before spawning (workers reload their own copies)
    del encoder, clf
    torch.cuda.empty_cache()

    def multi_eval(probe_state, use_rope: bool, tag: str):
        shards = shard_indices(len(val_ds), len(device_ids))
        paths = [shard_dir / f"eval_{tag}_gpu{d}.pt" for d in device_ids]
        for p in paths:
            if p.exists():
                p.unlink()
        t0 = time.time()
        run_sharded(
            _worker_eval, device_ids, shards, args_dict, maps, probe_state, use_rope, None
        )
        # re-run with correct out paths per shard — fix: pass path per worker
        return None  # placeholder, replaced below

    # --- proper multi_eval ---
    def multi_eval2(probe_state, use_rope: bool, tag: str):
        shards = shard_indices(len(val_ds), len(device_ids))
        paths = []
        ctx = mp.get_context("spawn")
        procs = []
        t0 = time.time()
        for rank, (dev, idxs) in enumerate(zip(device_ids, shards)):
            if not idxs:
                continue
            out_p = shard_dir / f"eval_{tag}_gpu{dev}.pt"
            if out_p.exists():
                out_p.unlink()
            paths.append(out_p)
            p = ctx.Process(
                target=_worker_eval,
                args=(rank, dev, idxs, args_dict, maps, probe_state, use_rope, str(out_p)),
            )
            p.start()
            procs.append(p)
            logger.info("eval-%s spawn gpu=%d n=%d", tag, dev, len(idxs))
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"eval worker failed code={p.exitcode}")
        metrics = merge_eval(paths)
        logger.info("eval-%s %s in %.1fs", tag, metrics, time.time() - t0)
        return metrics

    # zs eval
    logger.info("multi-GPU zs_no_rope eval")
    zs_no = multi_eval2(init_state, False, "zs_no")

    # multi-GPU train cache
    logger.info("multi-GPU cache train tokens")
    train_shards = shard_indices(len(train_ds), len(device_ids))
    cache_paths = []
    ctx = mp.get_context("spawn")
    procs = []
    t_cache0 = time.time()
    for rank, (dev, idxs) in enumerate(zip(device_ids, train_shards)):
        if not idxs:
            continue
        out_p = shard_dir / f"cache_gpu{dev}.pt"
        if out_p.exists():
            out_p.unlink()
        cache_paths.append(out_p)
        p = ctx.Process(
            target=_worker_encode_cache,
            args=(rank, dev, idxs, args_dict, maps, init_state, str(out_p)),
        )
        p.start()
        procs.append(p)
        logger.info("cache spawn gpu=%d n=%d", dev, len(idxs))
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"cache worker failed code={p.exitcode}")
    cached = []
    for pth in cache_paths:
        cached.extend(torch.load(pth, map_location="cpu"))
    cached.sort(key=lambda x: x["idx"])
    cache_sec = time.time() - t_cache0
    logger.info("merged cache n=%d in %.1fs", len(cached), cache_sec)

    # FT on GPU 0
    torch.cuda.set_device(device_ids[0])
    clf = _make_clf(maps, embed_dim, args.probe_heads, args.probe_depth, ft_device, init_state)

    logger.info("FT no RoPE on gpu %d", device_ids[0])
    sec_no = finetune(clf, cached, rope=None, epochs=args.epochs, lr=args.lr, maps=maps, device=ft_device)
    ft_no_state = copy.deepcopy(clf.state_dict())
    torch.save({"probe": ft_no_state, "rope": False, "horizon": HORIZON}, args.out_dir / "probe_ft_no_rope_2s.pt")
    # free before eval workers
    del clf
    torch.cuda.empty_cache()
    ft_no = multi_eval2(ft_no_state, False, "ft_no")

    logger.info("FT + RoPE (only_block0) on gpu %d", device_ids[0])
    clf = _make_clf(maps, embed_dim, args.probe_heads, args.probe_depth, ft_device, init_state)
    rope = ProbeTemporalRoPE(clf.pooler, rope_cross_attn_k=False, only_block0=True)
    sec_rope = finetune(clf, cached, rope=rope, epochs=args.epochs, lr=args.lr, maps=maps, device=ft_device)
    ft_rope_state = copy.deepcopy(clf.state_dict())
    torch.save(
        {"probe": ft_rope_state, "rope": True, "only_block0": True, "horizon": HORIZON},
        args.out_dir / "probe_ft_rope_2s.pt",
    )
    rope.remove()
    del clf
    torch.cuda.empty_cache()
    ft_rope = multi_eval2(ft_rope_state, True, "ft_rope")

    payload = {
        "method": "kvprune_rope_2s_from_scratch_multigpu",
        "horizon_sec": HORIZON,
        "device_ids": device_ids,
        "protocol": {
            "cache_frames": CACHE_FRAMES,
            "new_frames": NEW_FRAMES,
            "keep_frames": CACHE_FRAMES - NEW_FRAMES,
            "prune": "Probe.blocks[0] self-attn received mass",
            "rope": "only_block0 abs frame/slot ids",
            "encoder": "frozen vitl.pt",
            "probe": "random init → FT",
        },
        "accuracy_action": {
            "zs_no_rope": zs_no,
            "ft_no_rope": ft_no,
            "ft_rope": ft_rope,
            "delta_pp_rope_minus_norope_top5": round(ft_rope["top5"] - ft_no["top5"], 4),
        },
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "cache_sec": cache_sec,
        "ft_no_sec": sec_no,
        "ft_rope_sec": sec_rope,
        "device": torch.cuda.get_device_name(device_ids[0]),
    }
    out = args.out_dir / "metrics_2s.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "DONE_2S").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    logger.info("wrote %s", out)
    logger.info("summary %s", json.dumps(payload["accuracy_action"], indent=2))


if __name__ == "__main__":
    # Avoid OpenMP / CUDA fork issues
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    mp.set_start_method("spawn", force=True)
    main()
