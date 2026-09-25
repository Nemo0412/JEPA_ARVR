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
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from decord import VideoReader, cpu
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

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
    clips = clip_uint8.to(device, non_blocking=True).float().div_(255.0)
    return clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))


def encode_joint(stream: StreamKVAttnPruneEncoder, pooler: nn.Module, clips: torch.Tensor, embed_dim: int, chunk: int):
    """Trainable encode with detached probe-blk0 prune scores (discrete topk)."""
    hist = clips[:, :, :CACHE_FRAMES]
    new = clips[:, :, CACHE_FRAMES:]
    # Skip encoder QK score refresh — probe scores drive prune; saves a huge softmax.
    state = stream.fill(hist, refresh_scores=False)
    tok0 = state.tokens
    if tok0.size(-1) != embed_dim:
        tok0 = tok0[:, :, -embed_dim:]
    # Discrete prune: do not backprop through score → index selection.
    with torch.no_grad():
        scores = probe_blk0_slot_scores(pooler, tok0.detach(), stream.gp, chunk=chunk)
    state = stream.step(state, new, mode="attn", slot_scores=scores, refresh_scores=False)
    tok = state.tokens
    if tok.size(-1) != embed_dim:
        tok = tok[:, :, -embed_dim:]
    frame_ids = token_frame_ids_from_slots(state.slot_ids, stream.gp)
    return tok, frame_ids


class JointBundle(nn.Module):
    """Wraps anticipative model (w/ encoder LoRA) + probe for DDP."""

    def __init__(self, backbone: nn.Module, clf: AttentiveClassifier, chunk: int = 256):
        super().__init__()
        self.backbone = backbone
        self.clf = clf
        self.chunk = int(chunk)
        self.stream = StreamKVAttnPruneEncoder(
            backbone.encoder,
            gp=int(backbone.grid_size) ** 2,
            tubelet_size=2,
            cache_frames=CACHE_FRAMES,
            new_frames=NEW_FRAMES,
            chunk=self.chunk,
        )
        self._rope: ProbeTemporalRoPE | None = None

    def enable_rope(self, enabled: bool, *, only_block0: bool = False):
        if self._rope is not None:
            self._rope.remove()
            self._rope = None
        if enabled:
            self._rope = ProbeTemporalRoPE(
                self.clf.pooler,
                rope_cross_attn_k=False,
                only_block0=bool(only_block0),
            )

    def forward(self, clips: torch.Tensor):
        tok, frame_ids = encode_joint(
            self.stream, self.clf.pooler, clips, int(self.backbone.embed_dim), self.chunk
        )
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
    ap.add_argument("--rope", type=int, default=1, choices=[0, 1])
    ap.add_argument(
        "--only-block0",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = RoPE on Probe.blocks[0] only; 0 = all Probe self-attn blocks (default)",
    )
    ap.add_argument("--max-train", type=int, default=0, help="0 = full train split")
    ap.add_argument("--max-val", type=int, default=0, help="0 = full val split")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-lr-mult", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--probe-depth", type=int, default=4)
    ap.add_argument("--probe-heads", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-every-epochs", type=int, default=1)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    horizon = float(args.horizon)
    only_block0 = bool(args.only_block0)
    if args.out_dir is None:
        hz = int(horizon) if horizon == int(horizon) else horizon
        args.out_dir = Path(f"/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_{hz}s")

    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}")
    is_main = rank == 0

    hz_tag = int(horizon) if horizon == int(horizon) else horizon
    tag = "rope" if args.rope else "norope"
    if args.rope and only_block0:
        tag = "rope_blk0"
    elif args.rope:
        tag = "rope_all"
    out_dir = args.out_dir / f"joint_{hz_tag}s_{tag}"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "joint encoder-LoRA+probe  horizon=%.1fs rope=%s only_block0=%s world=%d out=%s",
            horizon, bool(args.rope), only_block0, world, out_dir,
        )

    verb_map, noun_map, action_map = build_class_maps(args.train_csv)
    maps = (verb_map, noun_map, action_map)
    if is_main:
        logger.info("classes v=%d n=%d a=%d", len(verb_map), len(noun_map), len(action_map))

    total_frames = CACHE_FRAMES + NEW_FRAMES
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

    bundle = JointBundle(backbone, clf, chunk=args.chunk).to(device)
    if args.rope:
        bundle.enable_rope(True, only_block0=only_block0)
        if is_main:
            logger.info(
                "Probe temporal RoPE ON  scope=%s",
                "blocks[0] only" if only_block0 else "all self-attn blocks",
            )
    bundle = DDP(bundle, device_ids=[local], find_unused_parameters=True)
    raw = bundle.module

    train_ds = ClipAnticipationDataset(
        args.train_csv, args.video_root, horizon_sec=horizon, model_fps=args.fps,
        img_size=args.img_size, max_samples=args.max_train,
    )
    val_ds = ClipAnticipationDataset(
        args.val_csv, args.video_root, horizon_sec=horizon, model_fps=args.fps,
        img_size=args.img_size, max_samples=args.max_val,
    )
    train_samp = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
    val_samp = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_samp,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_samp,
        num_workers=max(1, args.num_workers // 2), collate_fn=collate, pin_memory=False,
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
    loss_steps_path = out_dir / "loss_steps.csv"
    loss_epoch_path = out_dir / "loss_epoch.csv"
    loss_steps_f = None
    loss_steps_w = None
    loss_epoch_w = None
    if is_main:
        loss_steps_f = loss_steps_path.open("w", newline="")
        loss_steps_w = csv.DictWriter(
            loss_steps_f,
            fieldnames=[
                "global_step", "epoch", "iter", "loss", "avg_loss",
                "valid", "wall_sec",
            ],
        )
        loss_steps_w.writeheader()
        loss_steps_f.flush()
        with loss_epoch_path.open("w", newline="") as ef:
            loss_epoch_w = csv.DictWriter(
                ef,
                fieldnames=[
                    "epoch", "train_loss", "val_top5", "val_top1", "val_n", "ep_sec", "wall_sec",
                ],
            )
            loss_epoch_w.writeheader()
        logger.info("loss curves → %s  %s", loss_steps_path, loss_epoch_path)

    try:
        for ep in range(args.epochs):
            train_samp.set_epoch(ep)
            bundle.train()
            loss_meter = 0.0
            n_step = 0
            t_ep = time.time()
            for it, batch in enumerate(train_loader):
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
                    if metrics["top5"] > best_top5:
                        best_top5 = metrics["top5"]
                        bad = 0
                        ckpt = {
                            "epoch": ep + 1,
                            "probe": raw.clf.state_dict(),
                            "backbone": raw.backbone.state_dict(),
                            "best_top5": best_top5,
                            "rope": bool(args.rope),
                            "horizon": horizon,
                            "only_block0": only_block0,
                            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                        }
                        torch.save(ckpt, out_dir / "best.pt")
                        logger.info("saved best.pt top5=%.2f", best_top5)
                    else:
                        bad += 1
                        if bad >= args.patience:
                            logger.info("early stop patience=%d", args.patience)
                            break
    finally:
        if loss_steps_f is not None:
            loss_steps_f.flush()
            loss_steps_f.close()

    if raw._rope is not None:
        raw._rope.remove()
        raw._rope = None

    if is_main:
        payload = {
            "method": f"kvprune_joint_{hz_tag}s",
            "rope": bool(args.rope),
            "only_block0": only_block0 if args.rope else None,
            "rope_scope": (
                "probe_blocks[0]" if (args.rope and only_block0)
                else ("probe_all_self_attn" if args.rope else "off")
            ),
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
