"""LoRA SFT fine-tuning of LLaVA-OneVision for HD-EPIC 1s action anticipation.

Independent of vlm_video_encoder.py / eval.py's encoder-swap attentive-probe
pipeline. Reuses zeroshot_vlm_prompting.py's class-vocab loading, clip
windowing, and prompt-building helpers, but instead of greedy-decoding and
scoring, it fine-tunes a LoRA adapter on the language model (vision tower +
projector frozen) to generate the ground-truth "verb, noun" answer for the
same observed-frame video + prompt used by the zero-shot/few-shot baselines.

Only the language model gets a LoRA adapter (target_modules matched by
LORA_TARGET_REGEX, which requires "language_model" in the dotted module
path) -- the vision tower and multi-modal projector stay frozen, consistent
with the user's decision to fine-tune the LM only on top of the existing
zero-shot prompt format.

The trained adapter is evaluated by passing --lora-adapter-path to
zeroshot_vlm_prompting.py's LlavaOnevisionBackend (no separate eval script).
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from pathlib import Path

from app.hdepic_lora_action_anticipation.zeroshot_vlm_prompting import (
    build_prompt,
    class_key_by_id,
    class_names,
    compute_clip_window,
    decode_frames,
    load_class_vocab,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


class HDEpicSFTDataset:
    """Decodes + tokenizes one example per __getitem__, run in DataLoader workers.

    Video decode (decord) and processor tokenization are both CPU-bound and
    serialize with GPU compute if done inline in the training loop -- this
    pushed average GPU utilization below the cluster's fair-use threshold and
    got job 11312820 killed mid-run even though instantaneous nvidia-smi
    snapshots looked fine. Routing through a DataLoader with num_workers>0
    lets the next sample's CPU work happen while the GPU trains on the
    current one.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        video_root: str,
        processor,
        prompt_text: str,
        verb_key_by_id: dict[int, str],
        noun_key_by_id: dict[int, str],
        num_frames: int,
        target_fps: float,
    ):
        self.rows = rows
        self.video_root = video_root
        self.processor = processor
        self.prompt_text = prompt_text
        self.verb_key_by_id = verb_key_by_id
        self.noun_key_by_id = noun_key_by_id
        self.num_frames = num_frames
        self.target_fps = target_fps

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from decord import VideoReader, cpu

        row = self.rows[i]
        video_id = row.get("video_id", "?")
        try:
            video_path = str(Path(self.video_root) / row["participant_id"] / f"{video_id}.MP4")
            vr_probe = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            vfps = vr_probe.get_avg_fps()
            indices = compute_clip_window(int(row["start_frame"]), vfps, self.num_frames, self.target_fps)
            frames, _ = decode_frames(video_path, indices)
            answer = f"{self.verb_key_by_id[int(row['verb_class'])]}, {self.noun_key_by_id[int(row['noun_class'])]}"
            inputs = build_example(self.processor, frames, self.prompt_text, answer)
            return {k: v.squeeze(0) for k, v in inputs.items()}, video_id, i
        except Exception as exc:  # noqa: BLE001 -- surfaced via logging in the training loop
            return None, f"{video_id} ({exc})", i


def _collate_passthrough(batch):
    return batch[0]


def build_example(processor, frames, prompt_text: str, answer_text: str):
    """Tokenize one (frames, prompt, target answer) example, loss-masked to the answer only.

    Relies on add_generation_prompt=True producing a true text prefix of the
    full conversation (prompt + assistant header) -- standard for SFT
    completion-only loss masking with chat templates.
    """
    prompt_conv = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt_text}]}]
    full_conv = prompt_conv + [{"role": "assistant", "content": [{"type": "text", "text": answer_text}]}]

    prompt_chat = processor.apply_chat_template(prompt_conv, add_generation_prompt=True)
    full_chat = processor.apply_chat_template(full_conv, add_generation_prompt=False)

    prompt_inputs = processor(videos=[frames], text=prompt_chat, return_tensors="pt")
    full_inputs = processor(videos=[frames], text=full_chat, return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels
    return full_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--verb-classes-csv", required=True)
    parser.add_argument("--noun-classes-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-adapter-path", default=None, help="Resume LoRA weights from a prior save")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for video decode + tokenization, overlapped with GPU compute "
        "(0 = inline in the main process, no overlap -- can starve average GPU utilization "
        "below the cluster's fair-use threshold)",
    )
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    torch.manual_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model_id=%s device=%s dtype=%s", args.model_id, device, args.torch_dtype)
    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, local_files_only=args.local_files_only
    ).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if args.resume_adapter_path:
        model = PeftModel.from_pretrained(model, args.resume_adapter_path, is_trainable=True)
        logger.info("Resumed LoRA adapter from %s", args.resume_adapter_path)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGET_REGEX,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    verb_vocab = load_class_vocab(args.verb_classes_csv)
    noun_vocab = load_class_vocab(args.noun_classes_csv)
    verb_names = class_names(verb_vocab)
    noun_names = class_names(noun_vocab)
    verb_key_by_id = class_key_by_id(verb_vocab)
    noun_key_by_id = class_key_by_id(noun_vocab)
    prompt_text = build_prompt(verb_names, noun_names, args.num_frames)

    rows = list(csv.DictReader(open(args.train_csv, newline="", encoding="utf-8")))
    random.Random(args.seed).shuffle(rows)
    if args.max_train_samples > 0:
        rows = rows[: args.max_train_samples]
    logger.info("Training on %d samples for %d epoch(s)", len(rows), args.num_epochs)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    optimizer.zero_grad()
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    import torch.utils.data

    dataset = HDEpicSFTDataset(
        rows, args.video_root, processor, prompt_text, verb_key_by_id, noun_key_by_id, args.num_frames, args.target_fps
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_passthrough,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    for epoch in range(args.num_epochs):
        for i, (inputs, video_id, row_idx) in enumerate(loader):
            if inputs is None:
                logger.warning("Sample %d (epoch %d, %s) failed in dataset, skipping", row_idx, epoch, video_id)
                continue
            try:
                inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                loss = outputs.loss / args.grad_accum_steps
                loss.backward()

                running_loss += outputs.loss.item()
                running_count += 1
            except Exception:
                logger.exception("Sample %d (epoch %d, %s) failed, skipping", row_idx, epoch, video_id)
                continue

            if (i + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / max(1, running_count)
                    logger.info(
                        "epoch=%d step=%d sample=%d/%d avg_loss=%.4f (%.1fs/sample)",
                        epoch,
                        global_step,
                        i + 1,
                        len(rows),
                        avg_loss,
                        elapsed / max(1, i + 1),
                    )
                    running_loss = 0.0
                    running_count = 0

                if global_step % args.save_every_steps == 0:
                    ckpt_dir = str(Path(args.output_dir) / f"step_{global_step}")
                    model.save_pretrained(ckpt_dir)
                    logger.info("Saved checkpoint to %s", ckpt_dir)

    final_dir = str(Path(args.output_dir) / "final")
    model.save_pretrained(final_dir)
    logger.info("Training done. Final adapter saved to %s", final_dir)


if __name__ == "__main__":
    main()
