#!/usr/bin/env python3
"""Extract frozen V-JEPA ViT-L features for Streaming RU-LSTM.

Produces the same on-disk layout as ``extract_rgb_features.py``:
  ``{video_id}.npy``  [T, 1024]
  ``{video_id}.json`` with ``frame_indices`` into the native video.

Sampling matches the JEPA stream encoder:
  - decode @ ``--fps`` (default 8)
  - tubelet_size=2 → one pooled token every 2 frames → effective 4 Hz
    (same rate as TSN ``alpha=0.25``)

Each temporal tubelet is mean-pooled over space → 1024-d (ViT-L embed_dim),
so the original 18M RU-LSTM (``feat_in=1024, hidden=1024``) needs no width change.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

PROJECT_JEPA = Path(os.environ.get("JEPA_ARVR_ROOT", "/home/ll5914/Jepa_yifan/JEPA_ARVR"))
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_JEPA))
sys.path.insert(0, str(VJEPA_ROOT))

from evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar import (  # noqa: E402
    init_module as init_anticipative_module,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def build_encoder(
    checkpoint: Path,
    *,
    device: torch.device,
    img_size: int,
    fps: int,
    chunk_frames: int,
    encoder_lora: Path | None,
):
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
    wrapper_kwargs = {"no_predictor": True, "num_output_frames": 2, "num_steps": 1}
    model = init_anticipative_module(
        frames_per_clip=chunk_frames,
        frames_per_second=fps,
        resolution=img_size,
        checkpoint=str(checkpoint),
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if encoder_lora is not None and encoder_lora.is_file():
        from app.hdepic_lora_action_anticipation.encoder_lora import (
            inject_encoder_lora,
            load_encoder_lora_checkpoint,
            set_encoder_lora_trainable,
        )

        inject_encoder_lora(model, rank=8, alpha=16.0, dropout=0.05, last_n_blocks=0)
        load_encoder_lora_checkpoint(model, str(encoder_lora))
        set_encoder_lora_trainable(model, trainable=False)
        print(f"encoder LoRA loaded (frozen): {encoder_lora}", flush=True)
    else:
        print("encoder LoRA: none (raw vitl.pt)", flush=True)
    return model


@torch.no_grad()
def encode_chunks(
    model,
    frames_bcthw: torch.Tensor,
    *,
    tubelet: int,
    patch: int,
    img_size: int,
) -> torch.Tensor:
    """frames: [B,C,T,H,W] float ImageNet-normalized → [B, T/tubelet, D]."""
    tokens = model.encoder(frames_bcthw)
    b, n, d_full = tokens.shape
    embed_dim = int(model.encoder.embed_dim)
    if d_full > embed_dim:
        tokens = tokens[:, :, -embed_dim:]
    t = frames_bcthw.shape[2]
    t_tok = t // tubelet
    gh = gw = img_size // patch
    expected = t_tok * gh * gw
    if n != expected:
        raise RuntimeError(f"token count mismatch: got N={n}, expected {expected}=({t_tok}*{gh}*{gw})")
    tokens = tokens.view(b, t_tok, gh * gw, embed_dim).mean(dim=2)
    return tokens


@torch.no_grad()
def extract_video(
    path: Path,
    model,
    *,
    device: torch.device,
    fps: float,
    img_size: int,
    chunk_frames: int,
    batch_chunks: int,
    tubelet: int = 2,
    patch: int = 16,
) -> tuple[np.ndarray, dict]:
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=4, width=img_size, height=img_size)
    n_frames = len(vr)
    vfps = float(vr.get_avg_fps())
    duration = n_frames / max(vfps, 1e-6)
    times = np.arange(0.0, duration, 1.0 / fps, dtype=np.float64)
    sample_idx = np.clip(np.floor(times * vfps).astype(np.int64), 0, n_frames - 1)
    # Drop exact duplicate indices while keeping order (variable vfps / rounding).
    keep = np.concatenate([[True], sample_idx[1:] != sample_idx[:-1]])
    sample_idx = sample_idx[keep]
    if len(sample_idx) < tubelet:
        sample_idx = np.pad(sample_idx, (0, tubelet - len(sample_idx)), mode="edge")
    # Truncate to whole tubelets, then pad only for chunked encoder forwards.
    n_real = len(sample_idx) - (len(sample_idx) % tubelet)
    sample_idx = sample_idx[:n_real]
    n_feat_valid = n_real // tubelet
    pad = (-len(sample_idx)) % chunk_frames
    if pad:
        sample_idx = np.concatenate([sample_idx, np.full(pad, sample_idx[-1], dtype=np.int64)])

    n_samp = len(sample_idx)
    n_chunks = n_samp // chunk_frames
    feat_rows = []
    frame_indices: list[int] = []

    for c0 in range(0, n_chunks, batch_chunks):
        c1 = min(n_chunks, c0 + batch_chunks)
        batch_clips = []
        for ci in range(c0, c1):
            sl = sample_idx[ci * chunk_frames : (ci + 1) * chunk_frames]
            frames = vr.get_batch(sl.tolist()).asnumpy()  # T,H,W,C uint8
            clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2)  # C,T,H,W
            batch_clips.append(clip)
            # One feature per tubelet; use the last native frame of each tubelet.
            for t in range(0, chunk_frames, tubelet):
                frame_indices.append(int(sl[t + tubelet - 1]))
        x = torch.stack(batch_clips, dim=0).to(device=device, dtype=torch.float32).div_(255.0)
        x = x.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))
        tok = encode_chunks(
            model, x, tubelet=tubelet, patch=patch, img_size=img_size
        )  # [B, T/tubelet, D]
        feat_rows.append(tok.reshape(-1, tok.shape[-1]).float().cpu().numpy().astype(np.float32))

    feat = np.concatenate(feat_rows, 0) if feat_rows else np.zeros((0, 1024), np.float32)
    feat = feat[:n_feat_valid]
    frame_indices = frame_indices[:n_feat_valid]

    meta = {
        "video_path": str(path),
        "n_video_frames": int(n_frames),
        "vfps": float(vfps),
        "alpha": float(tubelet / fps),
        "feat_fps": float(fps / tubelet),
        "sample_fps": float(fps),
        "tubelet_size": int(tubelet),
        "img_size": int(img_size),
        "chunk_frames": int(chunk_frames),
        "encoder": "vjepa_vit_large",
        "n_feat_frames": int(feat.shape[0]),
        "frame_indices": frame_indices,
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
        default=Path("/scratch/ll5914/datasets/HD-EPIC/rulstm_features/vjepa_vitl_p01"),
    )
    ap.add_argument("--checkpoint", type=Path, default=Path("/scratch/ll5914/models/vjepa2/vitl.pt"))
    ap.add_argument(
        "--encoder-lora",
        type=Path,
        default=Path(
            "/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/"
            "action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/encoder_lora_best.pt"
        ),
    )
    ap.add_argument("--no-encoder-lora", action="store_true")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--chunk-frames", type=int, default=16, help="Must be even; encoder clip length")
    ap.add_argument("--batch-chunks", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.chunk_frames % 2 != 0:
        raise SystemExit("--chunk-frames must be even (tubelet_size=2)")

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    lora = None if args.no_encoder_lora else args.encoder_lora
    model = build_encoder(
        args.checkpoint,
        device=device,
        img_size=args.img_size,
        fps=int(args.fps),
        chunk_frames=args.chunk_frames,
        encoder_lora=lora,
    )
    print(
        f"encoder embed_dim={model.encoder.embed_dim} chunk={args.chunk_frames} "
        f"batch_chunks={args.batch_chunks}",
        flush=True,
    )

    videos = sorted(args.video_root.glob("*.MP4")) + sorted(args.video_root.glob("*.mp4"))
    print(f"videos={len(videos)} → {args.out}", flush=True)

    for path in tqdm(videos, desc="extract_vjepa"):
        out_npy = args.out / f"{path.stem}.npy"
        out_meta = args.out / f"{path.stem}.json"
        if out_npy.is_file() and out_meta.is_file() and not args.overwrite:
            continue
        feat, meta = extract_video(
            path,
            model,
            device=device,
            fps=args.fps,
            img_size=args.img_size,
            chunk_frames=args.chunk_frames,
            batch_chunks=args.batch_chunks,
        )
        if lora is not None:
            meta["encoder_lora"] = str(lora)
        meta["checkpoint"] = str(args.checkpoint)
        np.save(out_npy, feat)
        out_meta.write_text(json.dumps(meta), encoding="utf-8")
        print(f"  {path.stem}: {feat.shape}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
