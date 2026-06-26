"""Zero-shot VLM prompting baseline for HD-EPIC 1s action anticipation.

This is intentionally independent of vlm_video_encoder.py / eval.py: it does
not replace the V-JEPA2 encoder inside the attentive-probe training pipeline.
Instead it prompts a chat-capable VLM (Llama-3.2-Vision-Instruct or
LLaVA-OneVision) directly with the observed frames and asks it to name the
next action; no weights are trained. Output is matched against the HD-EPIC
verb/noun class vocabularies to compute action/verb/noun top-1 accuracy.

Replicates the same "1s anticipation, 8 frames over a 4s observation window,
anchored at the action's start frame" windowing used by the encoder-swap VLM
baselines (vlm_video_encoder.py), via decode_videos_to_clips in
vjepa2/evals/action_anticipation_frozen/epickitchens.py with
anticipation_time_sec=1.0, anticipation_point=1.0:

    aframes = int(1.0 * video_fps)
    af = start_frame - aframes
    fstp = int(video_fps / target_fps)
    indices = arange(af - num_frames * fstp, af, fstp)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_class_vocab(csv_path: str) -> list[dict[str, Any]]:
    """Load HD_EPIC_{verb,noun}_classes.csv into id/key/instances records."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                instances = ast.literal_eval(row["instances"])
            except (ValueError, SyntaxError):
                instances = []
            records.append(
                {
                    "id": int(row["id"]),
                    "key": row["key"].strip().lower(),
                    "instances": [str(s).strip().lower() for s in instances],
                }
            )
    return records


def match_class(text: str, vocab: list[dict[str, Any]]) -> int | None:
    """Match free-form predicted text to the closest class id, or None."""
    norm = re.sub(r"[^a-z0-9 :_-]", "", text.strip().lower())
    if not norm:
        return None
    # exact match on key or any synonym instance
    for rec in vocab:
        if norm == rec["key"] or norm in rec["instances"]:
            return rec["id"]
    # substring containment, preferring the longest matched key/instance
    best_id, best_len = None, 0
    for rec in vocab:
        for candidate in [rec["key"], *rec["instances"]]:
            if not candidate:
                continue
            if candidate in norm or norm in candidate:
                if len(candidate) > best_len:
                    best_id, best_len = rec["id"], len(candidate)
    return best_id


def class_names(vocab: list[dict[str, Any]]) -> list[str]:
    return [rec["key"] for rec in vocab]


def class_key_by_id(vocab: list[dict[str, Any]]) -> dict[int, str]:
    return {rec["id"]: rec["key"] for rec in vocab}


def sample_few_shot_rows(csv_path: str, k: int, seed: int) -> list[dict[str, str]]:
    """Deterministically sample k rows, greedily preferring distinct verb classes."""
    import random

    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    rng = random.Random(seed)
    rng.shuffle(rows)
    chosen: list[dict[str, str]] = []
    seen_verbs: set[str] = set()
    for row in rows:
        if len(chosen) >= k:
            break
        if row.get("verb_class") in seen_verbs:
            continue
        chosen.append(row)
        seen_verbs.add(row.get("verb_class"))
    if len(chosen) < k:
        for row in rows:
            if len(chosen) >= k:
                break
            if row not in chosen:
                chosen.append(row)
    return chosen[:k]


def compute_clip_window(start_frame: int, video_fps: float, num_frames: int, target_fps: float) -> np.ndarray:
    """Mirror decode_videos_to_clips with anticipation_time_sec=1.0, anticipation_point=1.0."""
    aframes = int(1.0 * video_fps)
    af = int(start_frame) - aframes
    fstp = max(1, int(video_fps / target_fps))
    nframes = int(num_frames * fstp)
    indices = np.arange(af - nframes, af, fstp).astype(np.int64)
    indices[indices < 0] = 0
    return indices


def decode_frames(video_path: str, indices: np.ndarray, decode_size: int = 0):
    from decord import VideoReader, cpu

    # decode_size > 0 makes decord scale frames to (decode_size, decode_size) *during* decode
    # (FFmpeg scaler), instead of decoding native ~1400px frames and resizing later. HD-EPIC
    # GoPro footage is square, so this preserves aspect ratio and cuts decode cost ~30x at
    # 256px -- the fix for GPU starvation in the 32-frame VLM probe. decode_size=0 keeps native.
    if decode_size > 0:
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0), width=decode_size, height=decode_size)
    else:
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    indices = np.clip(indices, 0, len(vr) - 1)
    buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C] uint8
    return buffer, vr.get_avg_fps()


def build_prompt(verb_names: list[str], noun_names: list[str], num_frames: int) -> str:
    """Full instruction block, shown once on the first conversation turn."""
    return (
        f"You are looking at {num_frames} frames sampled evenly from the last few seconds of a "
        "first-person (egocentric) kitchen video. Predict the SINGLE next action the "
        "camera-wearer is about to perform, starting about one second from now.\n\n"
        "You MUST answer with BOTH a verb AND a noun, separated by a comma, on one line, "
        "and nothing else. Do not answer with only a verb.\n"
        "Format: verb, noun\n"
        "Example answer: pick-up, mug\n\n"
        f"The verb MUST be exactly one of: {', '.join(verb_names)}\n"
        f"The noun MUST be exactly one of: {', '.join(noun_names)}\n\n"
        "Answer (verb, noun):"
    )


# Re-stated right before the final answer cue in few-shot prompts: with several
# labeled examples in between, the model tends to drift back to describing the
# scene instead of emitting the terse 'verb, noun' format from the main
# instructions, which are now many tokens away.
FINAL_ANSWER_REMINDER = (
    "Remember: respond with ONLY 'verb, noun' (no description, no extra words).\n"
    "Answer (verb, noun):"
)


def _strip_answer_marker(line: str) -> str:
    line = line.strip().strip("*").strip()
    line = re.sub(r"(?i)^\**answer\**\s*(\(verb,?\s*noun\))?\s*:?\s*", "", line)
    return line.strip().rstrip(".")


def parse_response(text: str) -> tuple[str, str]:
    """Parse a 'verb, noun' answer out of free-form generation.

    With few-shot context the model sometimes rambles a scene description
    before eventually emitting 'Answer: verb, noun' -- so prefer any line
    containing an explicit answer marker, and require both parts to look like
    short class names (not whole sentences) before accepting a split.
    """
    raw_lines = [line for line in text.strip().splitlines() if line.strip()]
    candidates = [line for line in raw_lines if re.search(r"(?i)answer", line)]
    candidates += raw_lines
    for line in candidates:
        cleaned = _strip_answer_marker(line)
        parts = [p.strip() for p in re.split(r"[,:]", cleaned) if p.strip()]
        if len(parts) >= 2 and len(parts[0].split()) <= 4 and len(parts[1].split()) <= 4:
            return parts[0], parts[1]

    line = _strip_answer_marker(raw_lines[0]) if raw_lines else ""
    parts = [p.strip() for p in re.split(r"[,:]", line) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    words = line.split()
    if len(words) >= 2:
        return words[0], " ".join(words[1:])
    return line, ""


class MllamaBackend:
    model_class = "mllama"

    def __init__(self, model_id: str, torch_dtype, device: str, local_files_only: bool):
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, local_files_only=local_files_only
        ).to(device)
        self.model.eval()

    def generate(
        self,
        frames: np.ndarray,
        prompt_text: str,
        max_new_tokens: int,
        shots: list[tuple[np.ndarray, str]] | None = None,
    ) -> str:
        from PIL import Image

        images_flat: list[Image.Image] = []
        content: list[dict] = []

        if shots:
            # Single-turn, labeled-examples layout (no assistant turns). A multi-turn
            # user/assistant conversation made the model anchor on the FIRST shot's
            # verb verbatim across all queries (e.g. always "press, <varying noun>")
            # instead of treating the shots as exemplars -- this layout fixed it.
            intro_text = prompt_text.rsplit("Answer (verb, noun):", 1)[0].rstrip()
            content.append({"type": "text", "text": intro_text})
            for i, (shot_frames, answer) in enumerate(shots):
                shot_images = [Image.fromarray(shot_frames[j]) for j in range(shot_frames.shape[0])]
                images_flat.extend(shot_images)
                content.append({"type": "text", "text": f"\n\nExample {i + 1}:"})
                content.extend({"type": "image"} for _ in shot_images)
                content.append({"type": "text", "text": f"\nAnswer: {answer}"})
            query_images = [Image.fromarray(frames[j]) for j in range(frames.shape[0])]
            images_flat.extend(query_images)
            content.append({"type": "text", "text": "\n\nNow your turn:"})
            content.extend({"type": "image"} for _ in query_images)
            content.append({"type": "text", "text": "\n" + FINAL_ANSWER_REMINDER})
        else:
            query_images = [Image.fromarray(frames[j]) for j in range(frames.shape[0])]
            images_flat.extend(query_images)
            content.extend({"type": "image"} for _ in query_images)
            content.append({"type": "text", "text": prompt_text})

        conversation = [{"role": "user", "content": content}]
        chat_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images_flat, text=chat_prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, min_new_tokens=4, do_sample=False
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def _resample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    if frames.shape[0] == n:
        return frames
    idx = np.linspace(0, frames.shape[0] - 1, n).round().astype(np.int64)
    return frames[idx]


class LlavaOnevisionBackend:
    model_class = "llava_onevision"

    def __init__(
        self,
        model_id: str,
        torch_dtype,
        device: str,
        local_files_only: bool,
        lora_adapter_path: str | None = None,
    ):
        import torch
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, local_files_only=local_files_only
        ).to(device)
        if lora_adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
        self.model.eval()

    def generate(
        self,
        frames: np.ndarray,
        prompt_text: str,
        max_new_tokens: int,
        shots: list[tuple[np.ndarray, str]] | None = None,
    ) -> str:
        videos_list: list[np.ndarray] = []
        content: list[dict] = []

        if shots:
            intro_text = prompt_text.rsplit("Answer (verb, noun):", 1)[0].rstrip()
            content.append({"type": "text", "text": intro_text})
            for i, (shot_frames, answer) in enumerate(shots):
                content.append({"type": "text", "text": f"\n\nExample {i + 1}:"})
                content.append({"type": "video"})
                content.append({"type": "text", "text": f"\nAnswer: {answer}"})
                # LlavaOnevision's video processor torch.stacks all videos in one call,
                # so every video in videos_list must have the same frame count as the
                # query -- resample shots (sampled at --few-shot-num-frames) up to match.
                videos_list.append(_resample_frames(shot_frames, frames.shape[0]))
            content.append({"type": "text", "text": "\n\nNow your turn:"})
            content.append({"type": "video"})
            content.append({"type": "text", "text": "\n" + FINAL_ANSWER_REMINDER})
            videos_list.append(frames)
        else:
            content.append({"type": "video"})
            content.append({"type": "text", "text": prompt_text})
            videos_list.append(frames)

        conversation = [{"role": "user", "content": content}]
        chat_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(videos=videos_list, text=chat_prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, min_new_tokens=4, do_sample=False
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


BACKENDS = {
    "llama32vision": (MllamaBackend, "meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "llava_onevision": (LlavaOnevisionBackend, "llava-hf/llava-onevision-qwen2-7b-ov-hf"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--verb-classes-csv", required=True)
    parser.add_argument("--noun-classes-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--few-shot-k", type=int, default=0, help="0 = zero-shot (default)")
    parser.add_argument("--few-shot-csv", default=None, help="Source split for few-shot examples (e.g. train split)")
    parser.add_argument("--few-shot-seed", type=int, default=0)
    parser.add_argument(
        "--few-shot-num-frames",
        type=int,
        default=8,
        help="Frames per few-shot example (kept low by default to bound total image/token count "
        "across shots + query -- e.g. for Mllama, every frame becomes a separate full-resolution "
        "tiled image through the full generative model, not just the vision tower)",
    )
    parser.add_argument(
        "--lora-adapter-path",
        default=None,
        help="Path to a peft LoRA adapter (e.g. from train_vlm_lora_sft.py) to load on top of the "
        "base model before evaluation. Only supported for --backend llava_onevision.",
    )
    args = parser.parse_args()
    if args.few_shot_k > 0 and not args.few_shot_csv:
        raise SystemExit("--few-shot-csv is required when --few-shot-k > 0")
    if args.lora_adapter_path and args.backend != "llava_onevision":
        raise SystemExit("--lora-adapter-path is only supported for --backend llava_onevision")

    import torch

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    backend_cls, default_model_id = BACKENDS[args.backend]
    model_id = args.model_id or default_model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading backend=%s model_id=%s device=%s dtype=%s", args.backend, model_id, device, args.torch_dtype)
    if args.backend == "llava_onevision":
        backend = backend_cls(model_id, dtype, device, args.local_files_only, args.lora_adapter_path)
    else:
        backend = backend_cls(model_id, dtype, device, args.local_files_only)

    verb_vocab = load_class_vocab(args.verb_classes_csv)
    noun_vocab = load_class_vocab(args.noun_classes_csv)
    verb_names = class_names(verb_vocab)
    noun_names = class_names(noun_vocab)
    prompt_text = build_prompt(verb_names, noun_names, args.num_frames)
    logger.info("Prompt length (chars): %d, verbs=%d nouns=%d", len(prompt_text), len(verb_names), len(noun_names))

    shots: list[tuple[np.ndarray, str]] = []
    if args.few_shot_k > 0:
        verb_key_by_id = class_key_by_id(verb_vocab)
        noun_key_by_id = class_key_by_id(noun_vocab)
        shot_rows = sample_few_shot_rows(args.few_shot_csv, args.few_shot_k, args.few_shot_seed)
        for row in shot_rows:
            shot_video_path = str(Path(args.video_root) / row["participant_id"] / f"{row['video_id']}.MP4")
            from decord import VideoReader, cpu

            vr_probe = VideoReader(shot_video_path, num_threads=1, ctx=cpu(0))
            vfps = vr_probe.get_avg_fps()
            shot_indices = compute_clip_window(int(row["start_frame"]), vfps, args.few_shot_num_frames, args.target_fps)
            shot_frames, _ = decode_frames(shot_video_path, shot_indices)
            answer = f"{verb_key_by_id[int(row['verb_class'])]}, {noun_key_by_id[int(row['noun_class'])]}"
            shots.append((shot_frames, answer))
        logger.info(
            "Built %d few-shot examples from %s (seed=%d): %s",
            len(shots),
            args.few_shot_csv,
            args.few_shot_seed,
            [a for _, a in shots],
        )

    rows = list(csv.DictReader(open(args.split_csv, newline="", encoding="utf-8")))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    logger.info("Evaluating %d samples", len(rows))

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    n_correct_verb = n_correct_noun = n_correct_action = n_total = 0
    t0 = time.time()
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video_id",
                "start_frame",
                "gt_verb_class",
                "gt_noun_class",
                "raw_response",
                "pred_verb_text",
                "pred_noun_text",
                "pred_verb_class",
                "pred_noun_class",
                "verb_correct",
                "noun_correct",
                "action_correct",
            ]
        )
        for i, row in enumerate(rows):
            video_path = str(Path(args.video_root) / row["participant_id"] / f"{row['video_id']}.MP4")
            try:
                start_frame = int(row["start_frame"])
                gt_verb = int(row["verb_class"])
                gt_noun = int(row["noun_class"])
                # decode at native fps first to discover vfps, then recompute the
                # exact window (mirrors decode_videos_to_clips's two-pass need).
                from decord import VideoReader, cpu

                vr_probe = VideoReader(video_path, num_threads=1, ctx=cpu(0))
                vfps = vr_probe.get_avg_fps()
                indices = compute_clip_window(start_frame, vfps, args.num_frames, args.target_fps)
                frames, _ = decode_frames(video_path, indices)
                raw_response = backend.generate(frames, prompt_text, args.max_new_tokens, shots=shots)
            except Exception as exc:  # noqa: BLE001 - keep evaluating remaining samples
                logger.exception("Sample %d (%s) failed: %s", i, row.get("video_id"), exc)
                raw_response = ""
                gt_verb = int(row.get("verb_class", -1) or -1)
                gt_noun = int(row.get("noun_class", -1) or -1)
                start_frame = row.get("start_frame", "")

            pred_verb_text, pred_noun_text = parse_response(raw_response)
            pred_verb_class = match_class(pred_verb_text, verb_vocab)
            pred_noun_class = match_class(pred_noun_text, noun_vocab)
            verb_correct = int(pred_verb_class == gt_verb)
            noun_correct = int(pred_noun_class == gt_noun)
            action_correct = int(verb_correct and noun_correct)

            n_total += 1
            n_correct_verb += verb_correct
            n_correct_noun += noun_correct
            n_correct_action += action_correct

            writer.writerow(
                [
                    row.get("video_id"),
                    start_frame,
                    gt_verb,
                    gt_noun,
                    raw_response.replace("\n", " ").strip(),
                    pred_verb_text,
                    pred_noun_text,
                    pred_verb_class,
                    pred_noun_class,
                    verb_correct,
                    noun_correct,
                    action_correct,
                ]
            )
            f.flush()

            if (i + 1) % args.log_every == 0 or (i + 1) == len(rows):
                elapsed = time.time() - t0
                logger.info(
                    "[%d/%d] verb-acc=%.2f%% noun-acc=%.2f%% action-acc=%.2f%% (%.1fs/sample)",
                    i + 1,
                    len(rows),
                    100 * n_correct_verb / n_total,
                    100 * n_correct_noun / n_total,
                    100 * n_correct_action / n_total,
                    elapsed / n_total,
                )

    summary = {
        "backend": args.backend,
        "model_id": model_id,
        "n_samples": n_total,
        "verb_top1": 100 * n_correct_verb / max(1, n_total),
        "noun_top1": 100 * n_correct_noun / max(1, n_total),
        "action_top1": 100 * n_correct_action / max(1, n_total),
    }
    summary_path = str(Path(args.output_csv).with_suffix("")) + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary))
    logger.info("Wrote per-sample predictions to %s and summary to %s", args.output_csv, summary_path)


if __name__ == "__main__":
    main()
