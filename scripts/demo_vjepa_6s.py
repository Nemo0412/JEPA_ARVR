#!/usr/bin/env python
"""Single-sample demo: V-JEPA2 encoder+predictor action anticipation on 6s of video.

Feeds 6s × 8fps = 48 frames through the FROZEN V-JEPA2 ViT-L encoder + predictor
(same pipeline as the probe training), then runs the trained attentive-probe classifier
to produce top-5 action predictions for 1s-ahead anticipation.

Goal: show the pipeline runs end-to-end at a longer (6s) window without any
architectural or memory issues. The probe was trained on 4s/32frames -- feeding
48 frames is an out-of-distribution length test, so accuracy is not the point here;
the demo succeeds if it runs and returns plausible verb/noun predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

SHARED      = Path("/path/to/VJEPA2-EXP")
DEFAULT_CSV = SHARED / "data/hdepic_vjepa_annotations/p01_fixed/HD_EPIC_test_vjepa.csv"
VIDEO_ROOT  = SHARED / "data/hdepic_vjepa_videos"
CHECKPOINT  = SHARED / "checkpoints/vitl.pt"
PROBE_CKPT  = SHARED / ("outputs/hdepic_lora_action_anticipation/action_anticipation_frozen/"
                         "hdepic-singleprobe-1s-p01fixed-rgbonly-probeonly-vitl-fp32-bs8-noac-10ep-w10/best.pt")
ANN_ROOT    = SHARED / "data/hd-epic-annotations/narrations-and-action-segments"

VIDEO_FPS   = 30.0
TARGET_FPS  = 8.0          # same sampling rate as training
ANTICI_SEC  = 1.0          # anticipation horizon
RESOLUTION  = 256
PATCH_SIZE  = 16
TUBELET     = 2


# ── data helpers ─────────────────────────────────────────────────────────────

def load_sample(csv_path: Path, row_idx: int) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    r = rows[row_idx]
    return {
        "participant_id": r["participant_id"],
        "original_video_id": r["original_video_id"],
        "start_frame": int(r["start_frame"]),
        "narration": r.get("narration", ""),
        "verb_class": int(r.get("verb_class", -1)),
        "noun_class":  int(r.get("noun_class", -1)),
    }


def compute_frame_indices(start_frame: int, num_frames: int) -> np.ndarray:
    aframes = int(ANTICI_SEC * VIDEO_FPS)                  # 30 frames = 1s
    fstp    = max(1, int(VIDEO_FPS / TARGET_FPS))          # step = 3 → eff 10fps
    anchor  = start_frame - aframes
    indices = np.arange(anchor - num_frames * fstp, anchor, fstp)
    return np.clip(indices, 0, None).astype(int)


def find_video(original_video_id: str) -> Path:
    pid  = original_video_id.split("-")[0]
    # HD-EPIC stores as P01_YYYYMMDD-HHMMSS.MP4 (leading dash → underscore)
    stem = original_video_id.replace("-", "_", 1)         # only first dash
    for ext in (".MP4", ".mp4"):
        p = VIDEO_ROOT / pid / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Video not found: {VIDEO_ROOT / pid / stem}.*")


def decode_frames(video_path: Path, indices: np.ndarray) -> np.ndarray:
    from decord import VideoReader, cpu
    vr    = VideoReader(str(video_path), num_threads=2, ctx=cpu(0),
                        width=RESOLUTION, height=RESOLUTION)
    total = len(vr)
    safe  = np.clip(indices, 0, total - 1)
    frames = vr.get_batch(safe.tolist()).asnumpy()         # [T,H,W,3] uint8
    return frames


def load_class_vocab(name: str) -> dict[int, str]:
    """Returns {class_id: key_label}."""
    path = ANN_ROOT / f"HD_EPIC_{name}_classes.csv"
    vocab = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vocab[int(row["id"])] = row["key"].strip()
    return vocab


# ── model helpers ─────────────────────────────────────────────────────────────

def build_backbone(num_frames: int):
    from evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar import init_module
    model_kwargs = {
        "encoder": {
            "model_name": "vit_large",
            "checkpoint_key": "target_encoder",
            "tubelet_size": TUBELET,
            "patch_size": PATCH_SIZE,
            "uniform_power": True,
            "use_rope": True,
        },
        "predictor": {
            "model_name": "vit_predictor",
            "checkpoint_key": "predictor",
            "num_frames": 64,          # sets stock num_patches=8192; fine for ≤54f
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
    backbone = init_module(
        frames_per_clip=num_frames,
        frames_per_second=int(TARGET_FPS),
        resolution=RESOLUTION,
        checkpoint=str(CHECKPOINT),
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    ).cuda()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def build_probe(embed_dim: int = 1024):
    from evals.action_anticipation_frozen.models import AttentiveClassifier
    # class counts matching the p01_fixed split used during training
    probe = AttentiveClassifier(
        verb_classes={i: i for i in range(106)},
        noun_classes={i: i for i in range(303)},
        action_classes={i: i for i in range(1681)},
        embed_dim=embed_dim,
        num_heads=16,
        depth=4,
        use_activation_checkpointing=False,
    ).cuda()
    state = torch.load(PROBE_CKPT, map_location="cpu")
    probe.load_state_dict(state, strict=False)
    probe.eval()
    return probe


# ── main ─────────────────────────────────────────────────────────────────────

def run(row_idx: int, num_frames: int):
    sample  = load_sample(DEFAULT_CSV, row_idx)
    indices = compute_frame_indices(sample["start_frame"], num_frames)
    window_sec = (indices[-1] - indices[0]) / VIDEO_FPS

    print("=" * 62)
    print(f"Participant  : {sample['participant_id']}")
    print(f"Video        : {sample['original_video_id']}")
    print(f"GT narration : {sample['narration']}")
    print(f"GT classes   : verb={sample['verb_class']}  noun={sample['noun_class']}")
    print(f"Start frame  : {sample['start_frame']}")
    print(f"Obs window   : frames {indices[0]}–{indices[-1]}  "
          f"≈{window_sec:.2f}s  ({num_frames} frames @ {TARGET_FPS}fps eff)")
    print(f"Token count  : {(num_frames//TUBELET)*(RESOLUTION//PATCH_SIZE)**2} encoder tokens")

    video_path = find_video(sample["original_video_id"])
    print(f"Video file   : {video_path}")
    frames = decode_frames(video_path, indices)
    print(f"Decoded      : {frames.shape}  dtype={frames.dtype}")

    # [1, C, T, H, W]  float32
    x = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().cuda() / 255.0
    anticipation = torch.tensor([ANTICI_SEC], device="cuda")

    print("\nBuilding V-JEPA2 encoder + predictor ...")
    backbone = build_backbone(num_frames)
    print("Building attentive probe ...")
    probe    = build_probe()

    verb_vocab = load_class_vocab("verb")
    noun_vocab = load_class_vocab("noun")

    print("Running forward pass ...")
    with torch.no_grad():
        feats  = backbone(x, anticipation)            # [1, N_enc+N_pred, 1024]
        logits = probe(feats)                         # dict verb/noun/action

    print(f"Feature shape: {feats.shape}")
    print()

    for head in ("verb", "noun", "action"):
        lg = logits[head][0]                          # [num_classes]
        top5_ids  = lg.topk(5).indices.tolist()
        top5_prob = lg.softmax(0).topk(5).values.tolist()
        gt_id = sample["verb_class"] if head == "verb" else (
                sample["noun_class"] if head == "noun" else -1)
        vocab = verb_vocab if head == "verb" else (noun_vocab if head == "noun" else {})
        print(f"  {head.upper()} top-5:")
        for rank, (cid, prob) in enumerate(zip(top5_ids, top5_prob), 1):
            label   = vocab.get(cid, f"class_{cid}")
            gt_mark = " ← GT" if cid == gt_id else ""
            print(f"    {rank}. [{cid:4d}] {label:<28s}  {prob*100:.1f}%{gt_mark}")
        if head != "action" and gt_id not in top5_ids:
            gt_label = vocab.get(gt_id, f"class_{gt_id}")
            gt_rank  = (lg.argsort(descending=True) == gt_id).nonzero(as_tuple=True)[0].item() + 1
            print(f"       (GT [{gt_id}] {gt_label} is rank {gt_rank})")
    print("=" * 62)
    print("Demo PASSED — V-JEPA2 ran end-to-end on 6s / 48 frames.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row",        type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=48, help="6s × 8fps = 48")
    args = ap.parse_args()
    run(args.row, args.num_frames)


if __name__ == "__main__":
    main()
