#!/usr/bin/env python
"""Single-sample demo: Qwen2.5-VL-3B action anticipation with 6 frames.

Decodes exactly 6 frames from a real HD-EPIC clip (evenly spaced over the
same anticipation window used by the full eval pipeline: 1s before action
start), zero-shot prompts Qwen2.5-VL-3B-Instruct, and prints:
  - clip info (participant, video, narration/ground-truth action)
  - 6 frame indices and the observation window they cover
  - Qwen's raw response

Nothing is trained here -- this is a purely qualitative demo showing the
pipeline runs end-to-end with 6 frames.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ── config ───────────────────────────────────────────────────────────────────
SHARED = Path("/path/to/VJEPA2-EXP")
DEFAULT_CSV   = SHARED / "data/hdepic_vjepa_annotations/p01_fixed/HD_EPIC_test_vjepa.csv"
DEFAULT_VROOT = SHARED / "data/hdepic_vjepa_videos"
MODEL_ID      = "Qwen/Qwen2.5-VL-3B-Instruct"
ANTICIPATION_SEC = 1.0   # seconds before action start (mirrors the full eval)
VIDEO_FPS        = 30.0  # HD-EPIC native fps
TARGET_FPS       = 8.0   # anticipation window sampling rate


def load_sample(csv_path: Path, row_idx: int) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = rows[row_idx]
    return {
        "participant_id": row["participant_id"],
        "video_id":       row["video_id"],
        "original_video_id": row["original_video_id"],
        "start_frame":    int(row["start_frame"]),
        "narration":      row.get("narration", ""),
        "verb_class":     int(row.get("verb_class", -1)),
        "noun_class":     int(row.get("noun_class", -1)),
    }


def compute_frame_indices(start_frame: int, num_frames: int) -> np.ndarray:
    """Mirror the full pipeline: evenly sample `num_frames` in the obs window
    ending `ANTICIPATION_SEC` before `start_frame`."""
    aframes = int(ANTICIPATION_SEC * VIDEO_FPS)
    fstp    = max(1, int(VIDEO_FPS / TARGET_FPS))
    anchor  = start_frame - aframes
    # num_frames evenly from anchor - num_frames*fstp  to  anchor (exclusive)
    indices = np.arange(anchor - num_frames * fstp, anchor, fstp)
    return np.clip(indices, 0, None).astype(int)


def decode_frames(video_path: str, indices: np.ndarray, decode_size: int = 256):
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0),
                     width=decode_size, height=decode_size)
    total = len(vr)
    safe  = np.clip(indices, 0, total - 1)
    frames = vr.get_batch(safe.tolist()).asnumpy()   # [T, H, W, 3] uint8
    return frames


def find_video(video_root: Path, original_video_id: str) -> str:
    pid = original_video_id.split("-")[0]           # e.g. "P01"
    candidate = video_root / pid / f"{original_video_id}.mp4"
    if candidate.exists():
        return str(candidate)
    # fallback: glob
    matches = list((video_root / pid).glob(f"{original_video_id}*"))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"Video not found for {original_video_id} under {video_root}")


def build_messages(frames: np.ndarray) -> list[dict]:
    prompt = (
        "You are watching an egocentric kitchen video. "
        f"You are given {len(frames)} frames sampled from the last moment before an action begins. "
        "Predict the SINGLE next action the person is about to perform. "
        "Reply with exactly one short phrase in the form '<verb> the <noun>', e.g. 'open the fridge'."
    )
    content: list[dict] = []
    for frame in frames:
        pil = Image.fromarray(frame)
        content.append({"type": "image", "image": pil})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_demo(csv_path: Path, video_root: Path, row_idx: int, num_frames: int, decode_size: int):
    sample  = load_sample(csv_path, row_idx)
    indices = compute_frame_indices(sample["start_frame"], num_frames)

    print("=" * 60)
    print(f"Participant : {sample['participant_id']}")
    print(f"Video       : {sample['original_video_id']}")
    print(f"GT narration: {sample['narration']}")
    print(f"Start frame : {sample['start_frame']}  (verb={sample['verb_class']}, noun={sample['noun_class']})")
    print(f"Obs window  : frames {indices[0]}–{indices[-1]}  "
          f"({(indices[-1]-indices[0])/VIDEO_FPS:.2f}s, {num_frames} frames @ {TARGET_FPS}fps eff)")
    print(f"Frame decode size: {decode_size}px")

    video_path = find_video(video_root, sample["original_video_id"])
    print(f"Video file  : {video_path}")
    frames = decode_frames(video_path, indices, decode_size)
    print(f"Decoded     : {frames.shape}  dtype={frames.dtype}")

    print("\nLoading Qwen2.5-VL-3B-Instruct ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()

    messages  = build_messages(frames)
    text_in   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # extract PIL images in content order
    images_in = [m["image"] for m in messages[0]["content"] if m["type"] == "image"]
    inputs    = processor(text=[text_in], images=images_in, return_tensors="pt", padding=True)
    inputs    = {k: v.to("cuda") for k, v in inputs.items()}

    print("Running Qwen forward + generate ...")
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    # strip the prompt tokens
    generated = out_ids[:, inputs["input_ids"].shape[1]:]
    response  = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    print("\n" + "=" * 60)
    print(f"GT narration : {sample['narration']}")
    print(f"Qwen ({num_frames}f) : {response}")
    print("=" * 60)
    return response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",        default=str(DEFAULT_CSV))
    ap.add_argument("--video-root", default=str(DEFAULT_VROOT))
    ap.add_argument("--row",        type=int, default=0, help="row index in test CSV (0-based)")
    ap.add_argument("--num-frames", type=int, default=6)
    ap.add_argument("--decode-size",type=int, default=256, help="square decode resolution")
    args = ap.parse_args()
    run_demo(Path(args.csv), Path(args.video_root), args.row, args.num_frames, args.decode_size)


if __name__ == "__main__":
    main()
