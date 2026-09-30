#!/usr/bin/env python3
"""kvprune joint — encoder-LoRA + probe FT (DDP), stream KV + probe-blk0 prune.

Protocol:
  fill 128 → Probe.blocks[0] attn scores (detached) → drop 34 / keep 94
  → encode new 34 → packed 128 → probe CE @ --horizon seconds.

Trainable: encoder LoRA on last 12 blocks (attn qkv/proj) + full probe/heads.
RoPE (optional): temporal Q/K on **all** Probe self-attn blocks by default
(``--only-block0`` restores the older block0-only setting).

Launch examples:
  # 2s ablation (running): only_block0 RoPE vs none
  torchrun --standalone --nproc_per_node=2 scripts/run_local_kvprune_joint_2s.py \\
      --horizon 2 --rope 0 --only-block0 1
  # 6s next: RoPE on all probe self-attn blocks vs none
  torchrun --standalone --nproc_per_node=2 scripts/run_local_kvprune_joint_2s.py \\
      --horizon 6 --rope 0 --out-dir .../kvprune_joint_6s
  torchrun --standalone --nproc_per_node=2 scripts/run_local_kvprune_joint_2s.py \\
      --horizon 6 --rope 1 --out-dir .../kvprune_joint_6s
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from decord import VideoReader, cpu
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler


class _IndicesSampler(Sampler[int]):
    """Fixed index list (used to skip already-finished iters within an epoch)."""

    def __init__(self, indices: list[int]):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _last_csv_iter(path: Path) -> int | None:
    """Best-effort next-iter hint from loss_steps.csv.

    Prefer max(iter)+1 over the final row: a short failed relaunch can append a
    low-iter tail after a long mid-epoch run.
    """
    if not path.is_file():
        return None
    try:
        max_it = -1
        last_it = -1
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "iter" not in reader.fieldnames:
                return None
            for row in reader:
                try:
                    it = int(float(row["iter"]))
                except (TypeError, ValueError, KeyError):
                    continue
                last_it = it
                if it > max_it:
                    max_it = it
        if max_it < 0:
            return None
        # If the file ends in a restarted prefix, use the historical high-water mark.
        return max_it if (last_it >= 0 and last_it + 500 < max_it) else last_it
    except Exception:
        return None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", str(PROJECT_ROOT / "vjepa2")))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

from app.hdepic_lora_action_anticipation.encoder_lora import (  # noqa: E402
    inject_encoder_lora,
    set_encoder_lora_trainable,
)
from app.hdepic_lora_action_anticipation.stream_kvcache_attn_prune import (  # noqa: E402
    CACHE_FRAMES,
    NEW_FRAMES,
    ProbeTemporalRoPE,
    StreamKVAttnPruneEncoder,
    probe_blk0_slot_scores,
    token_frame_ids_from_slots,
    topk_slot_ids_from_scores,
)
from app.hdepic_lora_action_anticipation.train_stream_mtp import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    map_labels,
    topk_acc,
)
from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402

logger = logging.getLogger("kvprune_joint")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GP = 256
VIDEO_FPS = 30.0


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
        if int(os.environ.get("RANK", "0")) == 0:
            logger.info(
                "dataset %s kept=%d skipped=%d T=%d h=%.1fs",
                Path(csv_path).name, len(kept), skipped, self.n_model, self.horizon_sec,
            )

    def __len__(self):
        return len(self.rows)

    def _video_reader(self, path: Path):
        """Reuse demuxers inside one DataLoader worker.

        Opening a new VideoReader every sample is most of the CPU time after the
        clips are on node-local disk. Keep only the last two: holding every file
        open grew until the job was OOM-killed.
        """
        cache = getattr(self, "_vr_cache", None)
        if cache is None:
            cache = {}
            self._vr_cache = cache
            self._vr_order = []
        key = str(path)
        vr = cache.get(key)
        if vr is not None:
            return vr
        threads = max(1, int(os.environ.get("DECORD_THREADS", "2")))
        vr = VideoReader(
            key, ctx=cpu(0), num_threads=threads, width=self.img_size, height=self.img_size,
        )
        cache[key] = vr
        self._vr_order.append(key)
        while len(self._vr_order) > 2:
            cache.pop(self._vr_order.pop(0), None)
        return vr

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        video_id = r["video_id"]
        pid = video_id.split("_")[0]
        path = self.video_root / pid / f"{video_id}.MP4"
        if not path.is_file():
            path = self.video_root / pid / f"{video_id}.mp4"
        vr = self._video_reader(path)
        fi = np.clip(r["frame_idx"], 0, len(vr) - 1)
        frames = vr.get_batch(fi.tolist()).asnumpy()
        clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).contiguous()
        return {
            "clip": clip,
            "verb": torch.tensor(r["verb"], dtype=torch.long),
            "noun": torch.tensor(r["noun"], dtype=torch.long),
        }


def collate(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch], dim=0),
        "verb": torch.stack([b["verb"] for b in batch], dim=0),
        "noun": torch.stack([b["noun"] for b in batch], dim=0),
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
    non_blocking = clip_uint8.is_pinned()
    clips = clip_uint8.to(device, non_blocking=non_blocking).float().div_(255.0)
    return clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))


def loader_kwargs(num_workers: int) -> dict:
    """Keep decoded batches queued so the GPU is not idle between steps."""
    kw = {
        "num_workers": int(num_workers),
        "collate_fn": collate,
        "pin_memory": False,
    }
    if num_workers > 0:
        kw["prefetch_factor"] = 2
    return kw


def encode_clip(encoder: nn.Module, clips: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Full-attention encode. All tokens go to the probe; no stream KV cache."""
    tok = encoder(clips)
    if isinstance(tok, (tuple, list)):
        tok = tok[0]
    if tok.size(-1) != embed_dim:
        tok = tok[:, :, -embed_dim:]
    return tok


def encode_joint(
    stream: StreamKVAttnPruneEncoder,
    pooler: nn.Module,
    clips: torch.Tensor,
    embed_dim: int,
    chunk: int,
    *,
    stream_steps: int = 1,
    protect_hist: int = 0,
    protect_k_frames: int = 34,
    self_only: bool = False,
):
    """Trainable encode with detached probe-blk0 prune scores (discrete topk).

    ``stream_steps`` admits that many ×34-frame chunks after the 128 fill
    (``T = 128 + stream_steps*34``). When ``protect_hist>0``, each prune walks
    scores low→high but skips slots that were Top-K (``protect_k_frames`` /
    tubelet) in any of the previous ``protect_hist`` probe-score passes.

    self_only: the new 34 frames attend only to themselves. Their tokens are
    concatenated with the kept 94 and that pack is what the probe sees.
    """
    expected_t = CACHE_FRAMES + int(stream_steps) * NEW_FRAMES
    if clips.size(2) != expected_t:
        raise ValueError(f"encode_joint expects T={expected_t}, got {clips.size(2)}")
    hist = clips[:, :, :CACHE_FRAMES]
    # Skip encoder QK score refresh — probe scores drive prune; saves a huge softmax.
    state = stream.fill(hist, refresh_scores=False)
    tubelet = int(stream.tubelet_size)
    protect_k_slots = max(0, int(protect_k_frames) // tubelet)
    use_protect = int(protect_hist) > 0
    mode = "attn_protect" if use_protect else "attn"
    topk_hist: list[torch.Tensor] = []

    for si in range(int(stream_steps)):
        tok0 = state.tokens
        if tok0.size(-1) != embed_dim:
            tok0 = tok0[:, :, -embed_dim:]
        with torch.no_grad():
            scores = probe_blk0_slot_scores(pooler, tok0.detach(), stream.gp, chunk=chunk)
            cur_topk = topk_slot_ids_from_scores(scores, state.slot_ids, protect_k_slots)
            if use_protect and topk_hist:
                protected_ids = torch.cat(topk_hist[-int(protect_hist) :], dim=1)
            else:
                protected_ids = None
        s = CACHE_FRAMES + si * NEW_FRAMES
        e = s + NEW_FRAMES
        state = stream.step(
            state,
            clips[:, :, s:e],
            mode=mode,
            slot_scores=scores,
            protected_ids=protected_ids,
            refresh_scores=False,
            self_only=self_only,
        )
        if use_protect:
            topk_hist.append(cur_topk)

    tok = state.tokens
    if tok.size(-1) != embed_dim:
        tok = tok[:, :, -embed_dim:]
    frame_ids = token_frame_ids_from_slots(state.slot_ids, stream.gp)
    return tok, frame_ids


def positions_rel_to_pred(frame_ids: torch.Tensor, horizon_slots: float) -> torch.Tensor:
    """Slots from each token until the labeled action.

    newest observed slot is horizon_slots before the label. Cross-attn queries
    stay at position 0 (unrotated), so their offset to a frame is this value.
    """
    newest = frame_ids.amax(dim=1, keepdim=True)
    return (newest - frame_ids) + float(horizon_slots)


class JointBundle(nn.Module):
    """Wraps anticipative model (w/ encoder LoRA) + probe for DDP."""

    def __init__(
        self,
        backbone: nn.Module,
        clf: AttentiveClassifier,
        chunk: int = 256,
        *,
        stream_steps: int = 1,
        protect_hist: int = 0,
        protect_k_frames: int = 34,
        no_kv: bool = False,
        self_only: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.clf = clf
        self.chunk = int(chunk)
        self.stream_steps = int(stream_steps)
        self.protect_hist = int(protect_hist)
        self.protect_k_frames = int(protect_k_frames)
        self.no_kv = bool(no_kv)
        self.self_only = bool(self_only)
        self.stream = None
        if not self.no_kv:
            self.stream = StreamKVAttnPruneEncoder(
                backbone.encoder,
                gp=int(backbone.grid_size) ** 2,
                tubelet_size=2,
                cache_frames=CACHE_FRAMES,
                new_frames=NEW_FRAMES,
                chunk=self.chunk,
            )
        self._rope: ProbeTemporalRoPE | None = None

    def enable_rope(self, enabled: bool, *, only_block0: bool = False, cross_attn_k: bool = False):
        if self._rope is not None:
            self._rope.remove()
            self._rope = None
        if enabled:
            self._rope = ProbeTemporalRoPE(
                self.clf.pooler,
                rope_cross_attn_k=bool(cross_attn_k),
                only_block0=bool(only_block0),
            )

    def forward(self, clips: torch.Tensor):
        if self.no_kv:
            tok = encode_clip(self.backbone.encoder, clips, int(self.backbone.embed_dim))
            frame_ids = None
        else:
            tok, frame_ids = encode_joint(
                self.stream,
                self.clf.pooler,
                clips,
                int(self.backbone.embed_dim),
                self.chunk,
                stream_steps=self.stream_steps,
                protect_hist=self.protect_hist,
                protect_k_frames=self.protect_k_frames,
                self_only=self.self_only,
            )
        if self._rope is not None and self.rope_time == "relpred":
            if self.stream is None:
                raise RuntimeError("relpred RoPE needs stream slot ids")
            tubelet = float(self.stream.tubelet_size)
            horizon_slots = float(self.horizon_sec) * float(self.model_fps) / tubelet
            frame_ids = positions_rel_to_pred(frame_ids, horizon_slots)
        if self._rope is not None:
            self._rope.set_frame_ids(frame_ids)
        try:
            out = self.clf(tok)
        finally:
            if self._rope is not None:
                self._rope.set_frame_ids(None)
        return out, frame_ids

    @property
    def encoder(self):
        return self.backbone.encoder


def setup_ddp():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local)
    return rank, world, local


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(bundle_mod, loader, device, maps, chunk, rank):
    """bundle_mod: JointBundle (unwrapped)."""
    verb_map, noun_map, action_map = maps
    bundle_mod.eval()
    top5_sum = top1_sum = n = 0.0
    for batch in loader:
        clips = normalize_clip(batch["clip"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = bundle_mod(clips)
        v_lab, n_lab, a_lab, keep = map_labels(
            batch["verb"].to(device), batch["noun"].to(device), verb_map, noun_map, action_map, device
        )
        if not keep:
            continue
        logits = out["action"][keep].float()
        top5_sum += topk_acc(logits, a_lab, k=5) * len(keep)
        top1_sum += topk_acc(logits, a_lab, k=1) * len(keep)
        n += len(keep)
    t = torch.tensor([top5_sum, top1_sum, n], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    n = max(1.0, float(t[2].item()))
    return {
        "top5": round(100.0 * float(t[0].item()) / n, 4),
        "top1": round(100.0 * float(t[1].item()) / n, 4),
        "n": int(t[2].item()),
    }


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
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to .../experiments/kvprune_joint_{horizon}s",
    )
    ap.add_argument("--horizon", type=float, default=2.0, help="Anticipation horizon in seconds")
    ap.add_argument(
        "--no-kv",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = full-attention encode of --frames, tokens go straight to the probe",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Input frames when --no-kv 1 (must be a multiple of tubelet size 2)",
    )
    ap.add_argument(
        "--new-self-only",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = new 34 frames self-attend only, then concat with the kept 94 for the probe",
    )
    ap.add_argument("--rope", type=int, default=1, choices=[0, 1])
    ap.add_argument(
        "--only-block0",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = RoPE on Probe.blocks[0] only; 0 = all Probe self-attn blocks (default)",
    )
    ap.add_argument(
        "--stream-steps",
        type=int,
        default=1,
        help="Number of 34-frame stream ticks after the 128 fill (T=128+steps*34)",
    )
    ap.add_argument(
        "--protect-hist",
        type=int,
        default=0,
        help="If >0: protect slots that were Top-K in any of the last N probe-score passes",
    )
    ap.add_argument(
        "--protect-k",
        type=int,
        default=34,
        help="Top-K size in frames for protect history (tubelet-aligned; default 34)",
    )
    ap.add_argument(
        "--rope-time",
        choices=("abs", "relpred"),
        default="abs",
        help="abs = slot index in the observed stream. "
        "relpred = slots until the labeled action; cross-attn K uses that distance, query stays at 0.",
    )
    ap.add_argument("--max-train", type=int, default=0, help="0 = full train split")
    ap.add_argument("--max-val", type=int, default=0, help="0 = full val split")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-lr-mult", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--val-num-workers",
        type=int,
        default=-1,
        help="Val DataLoader workers; -1 = same as --num-workers (was num_workers//2)",
    )
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--probe-depth", type=int, default=4)
    ap.add_argument("--probe-heads", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=2000,
        help="Save latest.pt every N optimizer steps (0 = epoch-end / signal only)",
    )
    ap.add_argument("--val-every-epochs", type=int, default=1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument(
        "--resume",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 = resume from out_dir/latest.pt when present (weights + epoch)",
    )
    args = ap.parse_args()

    if args.no_kv and args.new_self_only:
        raise SystemExit("--new-self-only is the KV prune path, not --no-kv")
    if args.no_kv:
        if args.frames <= 0 or args.frames % 2 != 0:
            raise SystemExit("--no-kv 1 requires --frames > 0 and a multiple of 2")
        if args.rope:
            raise SystemExit("--no-kv does not combine with --rope 1")
    horizon = float(args.horizon)
    only_block0 = bool(args.only_block0)
    if args.out_dir is None:
        hz = int(horizon) if horizon == int(horizon) else horizon
        args.out_dir = Path(f"/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_{hz}s")

    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}")
    is_main = rank == 0

    stream_steps = max(1, int(args.stream_steps))
    protect_hist = max(0, int(args.protect_hist))
    protect_k = max(0, int(args.protect_k))

    hz_tag = int(horizon) if horizon == int(horizon) else horizon
    if args.no_kv or args.new_self_only:
        # --out-dir is already the run root (nokv clip, or self-only 34+94).
        tag = f"nokv_f{int(args.frames)}" if args.no_kv else "self34"
        out_dir = args.out_dir
    else:
        tag = "rope" if args.rope else "norope"
        if args.rope and only_block0:
            tag = "rope_blk0"
        elif args.rope:
            tag = "rope_all"
        if args.rope and args.rope_time == "relpred":
            tag = f"{tag}_relpred"
        if protect_hist > 0:
            tag = f"{tag}_protectk{protect_k}_h{protect_hist}"
        if stream_steps != 1:
            tag = f"{tag}_s{stream_steps}"
        out_dir = args.out_dir / f"joint_{hz_tag}s_{tag}"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "%s encoder-LoRA+probe  horizon=%.1fs probe_rope=%s only_block0=%s no_kv=%s self_only=%s "
            "frames=%s stream_steps=%d protect_hist=%d protect_k=%d world=%d out=%s",
            "nokv clip" if args.no_kv else ("self34" if args.new_self_only else "joint"),
            horizon, bool(args.rope), only_block0, bool(args.no_kv), bool(args.new_self_only),
            int(args.frames) if args.no_kv else CACHE_FRAMES + stream_steps * NEW_FRAMES,
            stream_steps, protect_hist, protect_k, world, out_dir,
        )
        if args.new_self_only:
            logger.info("new 34 frames: encoder self-attn only; concat kept 94 → probe")
        if protect_hist > 0 and stream_steps < 2:
            logger.warning(
                "protect_hist=%d with stream_steps=1: Top-K history is empty on "
                "the only prune → drops match plain attn (same as rope_all)",
                protect_hist,
            )

    verb_map, noun_map, action_map = build_class_maps(args.train_csv)
    maps = (verb_map, noun_map, action_map)
    if is_main:
        logger.info("classes v=%d n=%d a=%d", len(verb_map), len(noun_map), len(action_map))

    total_frames = int(args.frames) if args.no_kv else CACHE_FRAMES + stream_steps * NEW_FRAMES
    backbone = build_model(
        device, total_frames, args.fps, args.img_size, str(args.checkpoint), no_predictor=True
    )
    # Freeze all, then inject trainable LoRA on last 12 blocks (fits A6000 48GB
    # for 128+34 joint stream encode; early blocks run under no_grad).
    for p in backbone.parameters():
        p.requires_grad = False
    n_lora = inject_encoder_lora(
        backbone,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
        last_n_blocks=12,
        target_suffixes=("attn.qkv", "attn.proj"),
    )
    set_encoder_lora_trainable(backbone, trainable=True)
    if is_main:
        logger.info("injected encoder LoRA modules≈%s", n_lora)

    clf = AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=int(backbone.embed_dim),
        num_heads=args.probe_heads,
        depth=args.probe_depth,
        # Off: RoPE hooks + checkpoint recomputation disagree on saved tensors.
        use_activation_checkpointing=False,
    ).to(device)
    for p in clf.parameters():
        p.requires_grad = True

    bundle = JointBundle(
        backbone,
        clf,
        chunk=args.chunk,
        stream_steps=stream_steps,
        protect_hist=protect_hist,
        protect_k_frames=protect_k,
        no_kv=bool(args.no_kv),
        self_only=bool(args.new_self_only),
    ).to(device)
    bundle.rope_time = str(args.rope_time)
    bundle.horizon_sec = float(horizon)
    bundle.model_fps = float(args.fps)
    if args.rope:
        relpred = args.rope_time == "relpred"
        bundle.enable_rope(True, only_block0=only_block0, cross_attn_k=relpred)
        if is_main:
            logger.info(
                "Probe temporal RoPE ON  scope=%s time=%s",
                "blocks[0] only" if only_block0 else "all self-attn blocks",
                args.rope_time,
            )
    elif is_main:
        logger.info("Probe temporal RoPE OFF")
    bundle = DDP(bundle, device_ids=[local], find_unused_parameters=True)
    raw = bundle.module

    train_ds = ClipAnticipationDataset(
        args.train_csv, args.video_root, horizon_sec=horizon, model_fps=args.fps,
        img_size=args.img_size, n_model=total_frames, max_samples=args.max_train,
    )
    val_ds = ClipAnticipationDataset(
        args.val_csv, args.video_root, horizon_sec=horizon, model_fps=args.fps,
        img_size=args.img_size, n_model=total_frames, max_samples=args.max_val,
    )
    train_samp = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
    val_samp = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_samp,
        **loader_kwargs(args.num_workers),
    )
    val_workers = args.num_workers if args.val_num_workers < 0 else args.val_num_workers
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_samp,
        **loader_kwargs(max(1, val_workers)),
    )
    if is_main:
        logger.info(
            "dataloader workers=%d prefetch=2 pin_memory=0 decord_threads=%s",
            args.num_workers, os.environ.get("DECORD_THREADS", "2"),
        )

    lora_params = [p for n, p in raw.backbone.named_parameters() if p.requires_grad]
    probe_params = [p for p in raw.clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": probe_params, "lr": args.lr},
            {"params": lora_params, "lr": args.lr * args.lora_lr_mult},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    crit = nn.CrossEntropyLoss()

    best_top5 = -1.0
    bad = 0
    history = []
    t_run = time.time()
    global_step = 0
    start_ep = 0
    start_iter = 0
    partial_resume = False
    resume_loss_meter = 0.0
    resume_n_step = 0
    loss_steps_path = out_dir / "loss_steps.csv"
    loss_epoch_path = out_dir / "loss_epoch.csv"
    latest_path = out_dir / "latest.pt"
    loss_steps_f = None
    loss_steps_w = None

    resume_ck = None
    if args.resume:
        for cand in (latest_path, out_dir / "best.pt"):
            if cand.is_file():
                resume_ck = cand
                break
    if resume_ck is not None:
        ck = torch.load(resume_ck, map_location="cpu", weights_only=False)
        raw.clf.load_state_dict(ck["probe"], strict=False)
        raw.backbone.load_state_dict(ck["backbone"], strict=False)
        if ck.get("opt") is not None:
            try:
                opt.load_state_dict(ck["opt"])
            except Exception as exc:
                if is_main:
                    logger.warning("opt state not loaded (%s); fresh AdamW", exc)
        if ck.get("scaler") is not None:
            try:
                scaler.load_state_dict(ck["scaler"])
            except Exception:
                pass
        # epoch = next start_ep (completed epochs if saved at epoch end; in-progress ep if mid-save)
        start_ep = int(ck.get("epoch", 0))
        best_top5 = float(ck.get("best_top5", -1.0))
        global_step = int(ck.get("global_step", 0))
        history = list(ck.get("history", []))
        partial_resume = bool(ck.get("partial", False))
        start_iter = int(ck.get("resume_iter", 0)) if partial_resume else 0
        resume_loss_meter = float(ck.get("loss_meter", 0.0)) if partial_resume else 0.0
        resume_n_step = int(ck.get("n_step", 0)) if partial_resume else 0
        ck_world = int(ck.get("world_size", 0) or 0)
        # Old ckpts lack resume_iter: infer next iter from CSV so we don't rescan from 0.
        if partial_resume and start_iter <= 0:
            inferred = 0
            if is_main:
                last_it = _last_csv_iter(loss_steps_path)
                if last_it is not None:
                    inferred = int(last_it) + 1
                    logger.info("inferred resume_iter=%d from %s", inferred, loss_steps_path.name)
            t_inf = torch.tensor([inferred], device=device, dtype=torch.long)
            dist.broadcast(t_inf, src=0)
            start_iter = int(t_inf.item())
        # DDP shard changes with world size — old resume_iter is not transferable.
        if partial_resume and ck_world and ck_world != world and start_iter > 0:
            if is_main:
                logger.warning(
                    "world_size %d→%d: dropping resume_iter=%d (restart this epoch data; weights kept)",
                    ck_world, world, start_iter,
                )
            start_iter = 0
            resume_loss_meter = 0.0
            resume_n_step = 0
        if is_main:
            logger.info(
                "resumed %s start_ep=%d/%d start_iter=%d best_top5=%.2f step=%d partial=%s world=%d",
                resume_ck, start_ep, args.epochs, start_iter, best_top5, global_step,
                partial_resume, world,
            )

    if is_main:
        # Append whenever we resume so mid-epoch wall hits don't wipe curves.
        steps_mode = "a" if (resume_ck is not None and loss_steps_path.is_file()) else "w"
        epoch_mode = "a" if (resume_ck is not None and loss_epoch_path.is_file()) else "w"
        loss_steps_f = loss_steps_path.open(steps_mode, newline="")
        loss_steps_w = csv.DictWriter(
            loss_steps_f,
            fieldnames=[
                "global_step", "epoch", "iter", "loss", "avg_loss",
                "valid", "wall_sec",
            ],
        )
        if steps_mode == "w":
            loss_steps_w.writeheader()
            loss_steps_f.flush()
        if epoch_mode == "w":
            with loss_epoch_path.open("w", newline="") as ef:
                csv.DictWriter(
                    ef,
                    fieldnames=[
                        "epoch", "train_loss", "val_top5", "val_top1", "val_n", "ep_sec", "wall_sec",
                    ],
                ).writeheader()
        logger.info(
            "loss curves → %s  %s start_ep=%d start_iter=%d ckpt_every=%d",
            loss_steps_path, loss_epoch_path, start_ep, start_iter, args.ckpt_every,
        )

    def _trainable_sd(module: nn.Module) -> dict:
        # Full backbone ~600MB; trainable LoRA+probe is tiny — keeps wall-save cheap.
        return {n: p.detach().cpu() for n, p in module.named_parameters() if p.requires_grad}

    def _save_ckpt(
        resume_epoch: int,
        *,
        is_best: bool = False,
        partial: bool = False,
        resume_iter: int = 0,
        loss_meter_v: float = 0.0,
        n_step_v: int = 0,
    ):
        """Write latest.pt. resume_epoch/iter are where the next launch should continue."""
        if not is_main:
            return
        payload = {
            "epoch": resume_epoch,
            "partial": partial,
            "resume_iter": int(resume_iter) if partial else 0,
            "loss_meter": float(loss_meter_v) if partial else 0.0,
            "n_step": int(n_step_v) if partial else 0,
            "world_size": int(world),
            "probe": _trainable_sd(raw.clf),
            "backbone": _trainable_sd(raw.backbone),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "best_top5": best_top5,
            "global_step": global_step,
            "history": history,
            "rope": bool(args.rope),
            "horizon": horizon,
            "only_block0": only_block0,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        }
        tmp = latest_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, latest_path)
        if is_best:
            tmp_b = out_dir / "best.pt.tmp"
            torch.save(payload, tmp_b)
            os.replace(tmp_b, out_dir / "best.pt")
        logger.info(
            "saved %s ep=%d iter=%d step=%d partial=%s best=%s",
            latest_path.name, resume_epoch, int(resume_iter) if partial else 0,
            global_step, partial, is_best,
        )

    stop_flag = {"stop": False}

    def _on_signal(signum, _frame):
        if not stop_flag["stop"] and is_main:
            logger.warning("signal %s — checkpoint after this step then exit", signum)
        stop_flag["stop"] = True

    # Slurm USR1@120 → bash TERM on torchrun; also catch INT for manual stops.
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    early_stop = False
    try:
        for ep in range(start_ep, args.epochs):
            if stop_flag["stop"]:
                break
            train_samp.set_epoch(ep)
            # Materialize this epoch's shuffle once, then drop already-done prefix.
            ep_indices = list(train_samp)
            skip = int(start_iter) if ep == start_ep else 0
            if skip < 0:
                skip = 0
            if skip > len(ep_indices):
                skip = len(ep_indices)
            if ep == start_ep:
                start_iter = 0  # consume; later epochs start at 0
            if skip and is_main:
                logger.info(
                    "ep%d: skipping first %d/%d iters (resume mid-epoch, no data replay)",
                    ep, skip, len(ep_indices),
                )
            ep_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                sampler=_IndicesSampler(ep_indices[skip:]),
                **loader_kwargs(args.num_workers),
            )
            bundle.train()
            if ep == start_ep and partial_resume:
                loss_meter = float(resume_loss_meter)
                n_step = int(resume_n_step)
                partial_resume = False
            else:
                loss_meter = 0.0
                n_step = 0
            t_ep = time.time()
            for local_i, batch in enumerate(ep_loader):
                it = skip + local_i
                clips = normalize_clip(batch["clip"], device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out, _frame_ids = bundle(clips)
                    v_lab, n_lab, a_lab, keep = map_labels(
                        batch["verb"].to(device), batch["noun"].to(device),
                        verb_map, noun_map, action_map, device,
                    )
                    if not keep:
                        # Keep DDP graph alive even when labels miss the class maps.
                        loss = out["action"].sum() * 0.0
                    else:
                        loss = crit(out["action"][keep], a_lab)
                        if "verb" in out:
                            loss = loss + crit(out["verb"][keep], v_lab) + crit(out["noun"][keep], n_lab)
                if not torch.isfinite(loss.detach()):
                    # Still step a zero path to avoid DDP desync if rare NaN
                    loss = out["action"].sum() * 0.0
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in raw.parameters() if p.requires_grad], 1.0
                )
                scaler.step(opt)
                scaler.update()
                loss_val = float(loss.detach())
                valid = 1 if keep else 0
                if keep:
                    loss_meter += loss_val
                    n_step += 1
                if is_main:
                    global_step += 1
                    avg = loss_meter / max(1, n_step)
                    loss_steps_w.writerow({
                        "global_step": global_step,
                        "epoch": ep,
                        "iter": it,
                        "loss": f"{loss_val:.6f}" if valid else "",
                        "avg_loss": f"{avg:.6f}",
                        "valid": valid,
                        "wall_sec": f"{time.time() - t_run:.1f}",
                    })
                    if global_step % args.log_every == 0:
                        loss_steps_f.flush()
                        logger.info(
                            "[ep%d it%d step%d] loss=%.4f avg=%.4f",
                            ep, it, global_step,
                            loss_val if valid else float("nan"), avg,
                        )

                do_periodic = (
                    is_main
                    and args.ckpt_every > 0
                    and global_step > 0
                    and global_step % args.ckpt_every == 0
                )
                if do_periodic:
                    # Next launch continues at it+1 (same ep shuffle via set_epoch).
                    _save_ckpt(
                        ep, is_best=False, partial=True, resume_iter=it + 1,
                        loss_meter_v=loss_meter, n_step_v=n_step,
                    )
                    loss_steps_f.flush()

                if stop_flag["stop"]:
                    if is_main:
                        _save_ckpt(
                            ep, is_best=False, partial=True, resume_iter=it + 1,
                            loss_meter_v=loss_meter, n_step_v=n_step,
                        )
                        loss_steps_f.flush()
                        logger.warning(
                            "exiting after signal save @ ep=%d iter=%d step=%d",
                            ep, it + 1, global_step,
                        )
                    break

            if stop_flag["stop"]:
                break

            stats = torch.tensor([loss_meter, float(n_step)], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            avg_loss = float(stats[0] / max(1.0, float(stats[1])))

            metrics = None
            if (ep + 1) % args.val_every_epochs == 0:
                metrics = evaluate(raw, val_loader, device, maps, args.chunk, rank)
                if is_main:
                    ep_sec = time.time() - t_ep
                    logger.info(
                        "epoch %d/%d train_loss=%.4f val_top5=%.2f top1=%.2f n=%d ep_sec=%.1f",
                        ep + 1, args.epochs, avg_loss, metrics["top5"], metrics["top1"],
                        metrics["n"], ep_sec,
                    )
                    row = {
                        "epoch": ep + 1,
                        "train_loss": avg_loss,
                        "val_top5": metrics["top5"],
                        "val_top1": metrics["top1"],
                        "val_n": metrics["n"],
                        "ep_sec": round(ep_sec, 1),
                        "wall_sec": round(time.time() - t_run, 1),
                    }
                    history.append(row)
                    with loss_epoch_path.open("a", newline="") as ef:
                        csv.DictWriter(ef, fieldnames=list(row.keys())).writerow(row)
                    is_best = metrics["top5"] > best_top5
                    if is_best:
                        best_top5 = metrics["top5"]
                        bad = 0
                        logger.info("new best top5=%.2f", best_top5)
                    else:
                        bad += 1
                    # Epoch finished → next launch starts at ep+1.
                    _save_ckpt(ep + 1, is_best=is_best, partial=False)
                    if args.patience > 0 and bad >= args.patience:
                        logger.info("early stop patience=%d", args.patience)
                        early_stop = True
                        break
            elif is_main:
                # Still checkpoint even if val is skipped this epoch.
                _save_ckpt(ep + 1, is_best=False, partial=False)

            if early_stop:
                break
    finally:
        if loss_steps_f is not None:
            loss_steps_f.flush()
            loss_steps_f.close()

    if stop_flag["stop"]:
        # Non-zero so slurm wrapper resubmits; weights already in latest.pt.
        if dist.is_initialized():
            dist.barrier()
        raise SystemExit(75)

    if raw._rope is not None:
        raw._rope.remove()
        raw._rope = None

    if is_main:
        payload = {
            "method": (
                f"nokv_f{int(args.frames)}_h{hz_tag}s" if args.no_kv
                else (f"kv_self34_h{hz_tag}s" if args.new_self_only else f"kvprune_joint_{hz_tag}s")
            ),
            "no_kv": bool(args.no_kv),
            "new_self_only": bool(args.new_self_only),
            "frames": int(args.frames) if args.no_kv else CACHE_FRAMES + NEW_FRAMES,
            "rope": bool(args.rope),
            "only_block0": only_block0 if args.rope else None,
            "rope_scope": (
                "probe_blocks[0]" if (args.rope and only_block0)
                else ("probe_all_self_attn" if args.rope else "off")
            ),
            "stream_steps": stream_steps,
            "protect_hist": protect_hist,
            "protect_k_frames": protect_k,
            "prune": (
                f"attn_protect hist={protect_hist} k_frames={protect_k}"
                if protect_hist > 0 else "attn"
            ),
            "rope_time": args.rope_time if args.rope else "off",
            "backbone": "vit_large / ViT-L/16 @256",
            "checkpoint": str(args.checkpoint),
            "trainable": "encoder LoRA last-12 (qkv+proj) + full probe",
            "horizon_sec": horizon,
            "world_size": world,
            "best_top5": best_top5,
            "history": history,
            "loss_steps_csv": str(loss_steps_path),
            "loss_epoch_csv": str(loss_epoch_path),
            "wall_sec": time.time() - t_run,
        }
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        logger.info("done %s", json.dumps(payload, indent=2))

    dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
