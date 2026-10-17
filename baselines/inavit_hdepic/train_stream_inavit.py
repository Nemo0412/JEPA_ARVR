#!/usr/bin/env python3
"""Streaming InAViT (MotionFormer + HoIViT) on HD-EPIC P01.

Each stream tick uses the last 4s of RGB (16 frames), and the model predicts
the action 2s ahead. HD-EPIC has no official hand/object tracks, so InAViT's
box tokens are filled with dummy full-frame / quadrant boxes (architecture
unchanged).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

INAVIT_ROOT = Path(__file__).resolve().parents[1] / "InAViT"
sys.path.insert(0, str(INAVIT_ROOT))
RULSTM_HDEPIC = Path(__file__).resolve().parents[1] / "rulstm_hdepic"
sys.path.insert(0, str(RULSTM_HDEPIC))

from train_stream_rulstm import build_vocab, format_metrics, topk_acc  # noqa: E402


IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)


def dummy_boxes(batch: int, num_frames: int, n_obj: int = 4, n_hand: int = 1) -> dict:
    """cxcywh in [0, 1], shape [B, T, O/U, 4]."""
    obj = torch.zeros(batch, num_frames, n_obj, 4)
    quads = [
        (0.25, 0.25, 0.5, 0.5),
        (0.75, 0.25, 0.5, 0.5),
        (0.25, 0.75, 0.5, 0.5),
        (0.75, 0.75, 0.5, 0.5),
    ]
    for i, (cx, cy, w, h) in enumerate(quads[:n_obj]):
        obj[:, :, i, :] = torch.tensor([cx, cy, w, h])
    hand = torch.zeros(batch, num_frames, n_hand, 4)
    hand[:, :, :, :] = torch.tensor([0.5, 0.5, 0.3, 0.3])
    return {"obj": obj, "hand": hand}


class StreamVideoDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        video_root: Path,
        verb_map: dict,
        noun_map: dict,
        action_map: dict,
        obs_sec: float = 4.0,
        num_frames: int = 16,
        crop_size: int = 224,
        horizon_index: int = 0,
    ):
        self.df = pd.read_csv(csv_path)
        self.video_root = Path(video_root)
        self.verb_map = verb_map
        self.noun_map = noun_map
        self.action_map = action_map
        self.obs_sec = float(obs_sec)
        self.num_frames = int(num_frames)
        self.crop_size = int(crop_size)
        self.horizon_index = int(horizon_index)
        self._vr = {}

    def __len__(self):
        return len(self.df)

    def _video_path(self, video_id: str) -> Path:
        for ext in (".MP4", ".mp4"):
            p = self.video_root / f"{video_id}{ext}"
            if p.is_file():
                return p
        pid = video_id.split("_")[0]
        for ext in (".MP4", ".mp4"):
            p = self.video_root / pid / f"{video_id}{ext}"
            if p.is_file():
                return p
        raise FileNotFoundError(video_id)

    def _reader(self, video_id: str):
        if video_id not in self._vr:
            from decord import VideoReader, cpu
            path = self._video_path(video_id)
            self._vr[video_id] = VideoReader(str(path), ctx=cpu(0), num_threads=2)
        return self._vr[video_id]

    def __getitem__(self, index: int):
        r = self.df.iloc[index]
        video_id = str(r.video_id)
        vr = self._reader(video_id)
        vfps = float(r.vfps) if "vfps" in r and not pd.isna(r.vfps) else float(vr.get_avg_fps())
        tick = int(r.tick_frame)
        n = len(vr)
        start = max(0, int(round(tick - self.obs_sec * vfps)))
        end = max(start + 1, min(tick, n - 1))
        idxs = np.linspace(start, end, self.num_frames)
        idxs = np.clip(idxs, 0, n - 1).astype(np.int64)
        frames = vr.get_batch(idxs).asnumpy()  # T,H,W,C uint8
        x = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        x = F.interpolate(
            x.permute(1, 0, 2, 3),
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
        x = (x - mean) / std

        verbs = [int(v) for v in str(r.mtp_verbs).split(",")]
        nouns = [int(v) for v in str(r.mtp_nouns).split(",")]
        masks = [float(v) for v in str(r.mtp_mask).split(",")]
        hi = self.horizon_index
        v, nlab, m = verbs[hi], nouns[hi], masks[hi]
        a = -1
        if m > 0 and v in self.verb_map and nlab in self.noun_map and (v, nlab) in self.action_map:
            a = self.action_map[(v, nlab)]
            v = self.verb_map[v]
            nlab = self.noun_map[nlab]
        else:
            v, nlab, m = -1, -1, 0.0
        return {
            "frames": x,
            "verb": torch.tensor(v, dtype=torch.int64),
            "noun": torch.tensor(nlab, dtype=torch.int64),
            "action": torch.tensor(a, dtype=torch.int64),
            "mask": torch.tensor(m, dtype=torch.float32),
        }


class InAViTStream(nn.Module):
    """Original MotionFormer+HoIViT. Only the classification-layer width
    changes to the HD-EPIC vocab; backbone params stay as InAViT."""

    def __init__(self, backbone: nn.Module, num_verb: int, num_noun: int, num_action: int):
        super().__init__()
        self.backbone = backbone
        dim = int(backbone.embed_dim)
        # Replace EK 97/300 heads with the same Linear head type, HD-EPIC sizes.
        backbone.num_classes = [int(num_verb), int(num_noun), int(num_action)]
        for idx, n_cls in enumerate(backbone.num_classes):
            setattr(backbone, f"head{idx}", nn.Linear(dim, n_cls))
        if hasattr(backbone, "head"):
            backbone.head = nn.Identity()

    def forward(self, frames: torch.Tensor, metadata: dict):
        _verb, extra = self.backbone([frames], metadata)
        return {
            "verb": extra["verb"],
            "noun": extra["noun"],
            "action": extra["action"],
        }


def build_inavit(cfg_file: Path, device: torch.device) -> nn.Module:
    from slowfast.config.defaults import get_cfg
    from slowfast.models.video_model_builder import Motionformer

    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_file))
    cfg.NUM_GPUS = 1
    cfg.TRAIN.ENABLE = True
    # Three Linear heads (verb/noun/action), same as EGTEA InAViT layout.
    cfg.TRAIN.DATASET = "egtea"
    model = Motionformer(cfg)
    return model.to(device)


def run_epoch(model, loader, device, optimizer=None, train=True, n_obj=4, n_hand=1, grad_clip=1.0):
    model.train(mode=train)
    loss_sum = 0.0
    n_steps = 0
    totals = defaultdict(float)
    counts = defaultdict(int)
    horizon = 2.0
    for batch in tqdm(loader, desc="train" if train else "val", leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        v = batch["verb"].to(device)
        n = batch["noun"].to(device)
        a = batch["action"].to(device)
        m = batch["mask"].to(device)
        boxes = dummy_boxes(frames.size(0), frames.size(2), n_obj=n_obj, n_hand=n_hand)
        metadata = {
            "orvit_bboxes": {
                "obj": boxes["obj"].to(device),
                "hand": boxes["hand"].to(device),
            }
        }
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(frames, metadata)
            valid = (m > 0.5) & (v >= 0) & (n >= 0) & (a >= 0)
            if not bool(valid.any()):
                continue
            loss = (
                F.cross_entropy(out["verb"][valid], v[valid])
                + F.cross_entropy(out["noun"][valid], n[valid])
                + F.cross_entropy(out["action"][valid], a[valid])
            )
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        loss_sum += float(loss.detach())
        n_steps += 1
        with torch.no_grad():
            for name, logits, lab in (
                ("verb", out["verb"][valid].float(), v[valid]),
                ("noun", out["noun"][valid].float(), n[valid]),
                ("action", out["action"][valid].float(), a[valid]),
            ):
                for k in (1, 5):
                    key = f"{name}_top{k}@{horizon:g}s"
                    totals[key] += topk_acc(logits, lab, k=k) * lab.numel()
                    counts[key] += lab.numel()
    metrics = {k: 100.0 * totals[k] / max(1, counts[k]) for k in totals}
    metrics["loss"] = loss_sum / max(1, n_steps)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        type=Path,
        default=INAVIT_ROOT / "configs/INAVIT/HDEPIC_STREAM_4s2s.yaml",
    )
    ap.add_argument(
        "--train-csv",
        type=Path,
        default=Path(
            "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_4s2s_all/HD_EPIC_train_stream_mtp.csv"
        ),
    )
    ap.add_argument(
        "--val-csv",
        type=Path,
        default=Path(
            "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_4s2s_all/HD_EPIC_val_stream_mtp.csv"
        ),
    )
    ap.add_argument(
        "--video-root",
        type=Path,
        default=Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/scratch/ll5914/experiments/inavit_hdepic_stream_4s2s"),
    )
    ap.add_argument("--obs-sec", type=float, default=4.0)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verb_map, noun_map, action_map = build_vocab([args.train_csv, args.val_csv])
    print(
        f"device={device} vocab v/n/a={len(verb_map)}/{len(noun_map)}/{len(action_map)}",
        flush=True,
    )
    train_ds = StreamVideoDataset(
        args.train_csv, args.video_root, verb_map, noun_map, action_map,
        obs_sec=args.obs_sec, num_frames=args.num_frames, crop_size=args.crop_size,
    )
    val_ds = StreamVideoDataset(
        args.val_csv, args.video_root, verb_map, noun_map, action_map,
        obs_sec=args.obs_sec, num_frames=args.num_frames, crop_size=args.crop_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    backbone = build_inavit(args.cfg, device)
    model = InAViTStream(backbone, len(verb_map), len(noun_map), len(action_map)).to(device)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    n_bb = sum(p.numel() for n, p in model.named_parameters() if not n.split(".")[-2].startswith("head")) / 1e6
    print(
        f"InAViT MotionFormer+HoIViT {nparam:.1f}M params "
        f"(backbone≈{n_bb:.1f}M; heads resized to HD-EPIC vocab)",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ckpt_best = args.out_dir / "inavit_stream_best.pt"
    ckpt_last = args.out_dir / "inavit_stream_last.pt"
    best = -1.0
    history = []
    horizons = (2.0,)
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer, train=True)
        va = run_epoch(model, val_loader, device, train=False)
        scheduler.step()
        primary = va.get("action_top5@2s", 0.0)
        print(f"\n=== epoch {epoch}/{args.epochs} ===", flush=True)
        print("TRAIN\n" + format_metrics(tr, horizons), flush=True)
        print("VAL\n" + format_metrics(va, horizons), flush=True)
        history.append({"epoch": epoch, "train": tr, "val": va})
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": va}, ckpt_last)
        if primary >= best:
            best = primary
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": va}, ckpt_best)
            print(f"  saved best (action_top5@2s={best:.2f})", flush=True)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nDone. best action_top5@2s={best:.2f}", flush=True)
    state = torch.load(ckpt_best, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    metrics = run_epoch(model, val_loader, device, train=False)
    print("BEST VAL\n" + format_metrics(metrics, horizons), flush=True)
    (args.out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
