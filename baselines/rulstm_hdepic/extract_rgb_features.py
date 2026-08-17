#!/usr/bin/env python3
"""Fast TSN-RGB feature extraction for HD-EPIC P01 (@ ~4 fps).

Bottleneck fix vs v1: batched tensor transforms (no PIL), larger batches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from pretrainedmodels import bninception
from torch import nn
from tqdm import tqdm


def build_model(ckpt: Path, device: torch.device) -> nn.Module:
    model = bninception(pretrained=None)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    state = {k.replace("module.base_model.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.last_linear = nn.Identity()
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.to(device).eval()
    return model


def preprocess_batch(frames_nhwc: np.ndarray, device: torch.device) -> torch.Tensor:
    """frames: uint8 RGB [B,H,W,3] → BGR BNInception tensor."""
    x = torch.from_numpy(frames_nhwc).to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2)  # B,3,H,W
    # Resize short side logic matching [256, 454] fixed size used by RU-LSTM FEATEXT
    x = F.interpolate(x, size=(256, 454), mode="bilinear", align_corners=False)
    x = x[:, [2, 1, 0], ...]  # RGB→BGR
    x = x - torch.tensor([104.0, 117.0, 128.0], device=device).view(1, 3, 1, 1)
    return x


@torch.no_grad()
def extract_video(
    path: Path,
    model: nn.Module,
    device: torch.device,
    alpha: float,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=4)
    n_frames = len(vr)
    vfps = float(vr.get_avg_fps())
    duration = n_frames / max(vfps, 1e-6)
    times = np.arange(0.0, duration, alpha, dtype=np.float64)
    frame_idx = np.clip(np.floor(times * vfps).astype(np.int64), 0, n_frames - 1)
    keep = np.concatenate([[True], frame_idx[1:] != frame_idx[:-1]])
    frame_idx = frame_idx[keep]

    feats = []
    for i in range(0, len(frame_idx), batch_size):
        idx = frame_idx[i : i + batch_size]
        frames = vr.get_batch(idx).asnumpy()
        x = preprocess_batch(frames, device)
        y = model(x).reshape(x.shape[0], -1).float().cpu().numpy().astype(np.float32)
        feats.append(y)
    feat = np.concatenate(feats, 0) if feats else np.zeros((0, 1024), np.float32)
    meta = {
        "video_path": str(path),
        "n_video_frames": int(n_frames),
        "vfps": float(vfps),
        "alpha": float(alpha),
        "feat_fps": float(1.0 / alpha),
        "n_feat_frames": int(feat.shape[0]),
        "frame_indices": frame_idx.tolist(),
        "feat_dim": int(feat.shape[1]) if feat.size else 1024,
    }
    return feat, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video-root",
        type=Path,
        default=Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos/P01"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01"),
    )
    ap.add_argument(
        "--tsn-ckpt",
        type=Path,
        default=Path("/scratch/ll5914/models/rulstm/TSN-rgb.pth.tar"),
    )
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    model = build_model(args.tsn_ckpt, device)

    videos = sorted(args.video_root.glob("*.MP4")) + sorted(args.video_root.glob("*.mp4"))
    print(f"videos={len(videos)} → {args.out}", flush=True)

    for path in tqdm(videos, desc="extract"):
        out_npy = args.out / f"{path.stem}.npy"
        out_meta = args.out / f"{path.stem}.json"
        if out_npy.is_file() and out_meta.is_file() and not args.overwrite:
            continue
        feat, meta = extract_video(path, model, device, args.alpha, args.batch_size)
        np.save(out_npy, feat)
        out_meta.write_text(json.dumps(meta), encoding="utf-8")
        print(f"  {path.stem}: {feat.shape}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
