"""LoRA + classification-head probe fine-tuning for HD-EPIC 1s action anticipation.

Per user decision (2026-06-22), this supersedes train_vlm_lora_sft.py's text-generation
SFT approach (kept for reference, not deleted) -- training methodology now mirrors the
PhD's Qwen2.5-VL-3B probe exactly (refer_repo/JEPA_ARVR/qwen/train_hdepic_qwen_probe.py),
applied to both LLaVA-OneVision and Llama-3.2-Vision-Instruct so all three backbones are
directly comparable: LoRA is injected only into the language model's q/k/v/o attention
projections (vision tower + multi-modal projector left frozen, identified by skipping any
named-module path containing "vision"/"visual"/"projector"); three linear classification
heads (verb/noun/action) sit on the LLM's last-token hidden state; trained with
CrossEntropyLoss. Metrics are Top-3 accuracy + class-mean Recall@5, matching the PhD's
evaluate() exactly, instead of this baseline family's usual top-1.

Reuses zeroshot_vlm_prompting.py's class-vocab loading and clip windowing so the observed
window (32 frames @ 8fps decoded, anticipation_sec=1.0) matches the V-JEPA2 baseline; frames
are then subsampled down to --probe-num-frames (default 8, matching the PhD's SAMPLE_FRAMES)
before being sent to the VLM, since both backends turn every frame into a full image/token
block through the generative pipeline (unlike a ViT encoder that just patchifies once).

The action class space (--action-map) is *not* the full 106x303 cartesian product; like the
PhD's script, it's built from the (verb, noun) pairs actually observed across train+val+test,
enumerated in sorted order for a deterministic mapping.
"""

from __future__ import annotations

import argparse
import csv
import os
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from app.hdepic_lora_action_anticipation.zeroshot_vlm_prompting import (
    compute_clip_window,
    decode_frames,
    load_class_vocab,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TASK_PROMPT = "Based on this video, predict what action the person will perform next."
_VISION_SUBSTRINGS = ("vision", "visual", "projector")


def _resample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    if frames.shape[0] <= n:
        # Already at or below target: return as-is (no upsampling / frame duplication).
        # For variable-length windows (e.g. 1-min clips near video start that have less than
        # n frames of history), the honest choice is to pass only what's available.
        return frames
    idx = np.linspace(0, frames.shape[0] - 1, n).round().astype(np.int64)
    return frames[idx]


class LoRALinear(nn.Module):
    """Hand-rolled LoRA wrapper -- matches the PhD's qwen/train_hdepic_qwen_probe.py exactly
    (not peft) for architectural parity when comparing against his Qwen2.5-VL-3B probe."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank
        d_in, d_out = linear.in_features, linear.out_features
        dev, dtype = linear.weight.device, linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(rank, d_in, device=dev, dtype=dtype) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=dev, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora_to_llm(model: nn.Module, rank: int, alpha: float) -> int:
    """Inject LoRA into q/k/v/o projections outside any vision-tower/projector submodule."""
    n_injected = 0
    for mod_name, module in model.named_modules():
        if any(s in mod_name.lower() for s in _VISION_SUBSTRINGS):
            continue
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            orig = getattr(module, proj_name, None)
            if orig is None or not isinstance(orig, nn.Linear):
                continue
            for p in orig.parameters():
                p.requires_grad = False
            setattr(module, proj_name, LoRALinear(orig, rank=rank, alpha=alpha))
            n_injected += 1
    if n_injected == 0:
        raise RuntimeError("No q/k/v/o projections found outside vision tower/projector.")
    for name, p in model.named_parameters():
        p.requires_grad = "lora_A" in name or "lora_B" in name
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA injected into %d LLM projections (rank=%d, alpha=%.1f); trainable %.2fM / %.0fM (%.2f%%)",
        n_injected,
        rank,
        alpha,
        n_lora / 1e6,
        n_total / 1e6,
        100 * n_lora / n_total,
    )
    return n_injected


class VLMProbe(nn.Module):
    """<backbone> (LoRA on LLM) + classification heads on last-token hidden state."""

    def __init__(self, backbone, hidden_size: int, num_verbs: int, num_nouns: int, num_actions: int):
        super().__init__()
        self.backbone = backbone
        dev = next(backbone.parameters()).device
        self.verb_head = nn.Linear(hidden_size, num_verbs).to(device=dev)
        self.noun_head = nn.Linear(hidden_size, num_nouns).to(device=dev)
        self.action_head = nn.Linear(hidden_size, num_actions).to(device=dev)

    def forward(self, **model_inputs):
        outputs = self.backbone(**model_inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :].float()
        return self.verb_head(last_hidden), self.noun_head(last_hidden), self.action_head(last_hidden)


def build_llava_inputs_batch(processor, frames_list: list[np.ndarray]):
    conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": TASK_PROMPT}]}]
    chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
    texts = [chat_prompt] * len(frames_list)
    return processor(videos=list(frames_list), text=texts, return_tensors="pt", padding=True)


def build_mllama_inputs_batch(processor, frames_list: list[np.ndarray]):
    from PIL import Image

    n_frames = frames_list[0].shape[0]
    content = [{"type": "image"} for _ in range(n_frames)] + [{"type": "text", "text": TASK_PROMPT}]
    conversation = [{"role": "user", "content": content}]
    chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
    texts = [chat_prompt] * len(frames_list)
    images = [[Image.fromarray(frames[j]) for j in range(frames.shape[0])] for frames in frames_list]
    return processor(images=images, text=texts, return_tensors="pt", padding=True)


# Optional square pixel size each Qwen video frame is resized to before the AutoProcessor
# (0 = keep native resolution). HD-EPIC frames are decoded at native ~1400x1400; at native
# resolution Qwen is ~259x V-JEPA2's compute (B12), so the equal-compute study matches
# V-JEPA2's 256 here. Set once in main() from --qwen-frame-size; the processor's smart_resize
# then rounds to the nearest multiple of 28 (256 -> 252 -> an 18x18 patch grid per frame).
_QWEN_FRAME_SIZE = 0


def build_qwen_inputs_batch(processor, frames_list: list[np.ndarray]):
    """Matches the PhD's prepare_batch_cpu() in qwen/train_hdepic_qwen_probe.py exactly:
    one {"type": "video"} content block per sample, frames passed as PIL images."""
    from PIL import Image

    texts, video_lists = [], []
    for frames in frames_list:
        pil_frames = [Image.fromarray(frames[t]) for t in range(frames.shape[0])]
        if _QWEN_FRAME_SIZE > 0:
            pil_frames = [f.resize((_QWEN_FRAME_SIZE, _QWEN_FRAME_SIZE)) for f in pil_frames]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames},
                    {"type": "text", "text": TASK_PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        video_lists.append(pil_frames)
    return processor(text=texts, videos=video_lists, return_tensors="pt", padding=True)


BACKEND_BATCH_BUILDERS = {
    "llava_onevision": build_llava_inputs_batch,
    "llama32vision": build_mllama_inputs_batch,
    "qwen25vl": build_qwen_inputs_batch,
}
DEFAULT_MODEL_IDS = {
    "llava_onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llama32vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "qwen25vl": "Qwen/Qwen2.5-VL-3B-Instruct",
}


def load_action_map(rows_by_split: dict[str, list[dict]]) -> dict[tuple[int, int], int]:
    pairs = set()
    for rows in rows_by_split.values():
        for row in rows:
            pairs.add((int(row["verb_class"]), int(row["noun_class"])))
    return {pair: i for i, pair in enumerate(sorted(pairs))}


class HDEpicProbeDataset(torch.utils.data.Dataset):
    """Returns raw decoded frames + labels; tokenization happens batched in BatchCollator
    (one processor() call per training batch instead of per sample), so the LLM forward/
    backward sees a real batch_size>1 -- a batch_size=1 loop was measured to leave the GPU
    bursty/idle between tiny per-sample kernel launches (avg utilization ~9% over 5 min,
    well under the cluster's 60%-over-2h fair-use threshold), even with CPU/GPU overlap via
    DataLoader workers."""

    def __init__(self, rows, video_root, action_map, num_frames, probe_num_frames, target_fps,
                 decode_size=0, processor=None, backend=None, cache_dir=None):
        self.rows = rows
        self.video_root = video_root
        self.action_map = action_map
        self.num_frames = num_frames
        self.probe_num_frames = probe_num_frames
        self.target_fps = target_fps
        # decode_size > 0 -> decode frames directly at this square size (decord scaler) instead
        # of native ~1400px, the fix for GPU starvation in the 32-frame probe. Set from
        # --qwen-frame-size so it matches the resolution the model sees anyway.
        self.decode_size = decode_size
        # Preprocessing cache (cache_dir set): since there is no augmentation and the clip
        # window is deterministic, each sample tokenizes to identical tensors every epoch. We
        # tokenize once per sample, store the processor output to disk, and reload it on later
        # epochs -- turning a CPU-bound (decode+tokenize 32 frames) pipeline into a cheap
        # tensor load so the GPU stops starving. processor/backend are needed to build the
        # tensors on a cache miss; shared by every keep_ratio run (pruning is in-model, so the
        # preprocessing is identical across keep_ratio).
        self.processor = processor
        self.backend = backend
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.rows)

    def _cache_path(self, row):
        return os.path.join(
            self.cache_dir,
            f"{row['participant_id']}__{row.get('video_id', 'x')}__{int(row['start_frame'])}.pt",
        )

    def _decode_and_tokenize(self, row):
        from decord import VideoReader, cpu

        video_id = row.get("video_id", "?")
        video_path = str(Path(self.video_root) / row["participant_id"] / f"{video_id}.MP4")
        vr_probe = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        vfps = vr_probe.get_avg_fps()
        indices = compute_clip_window(int(row["start_frame"]), vfps, self.num_frames, self.target_fps)
        frames, _ = decode_frames(video_path, indices, decode_size=self.decode_size)
        frames = _resample_frames(frames, self.probe_num_frames)
        # One-sample processor call; leading dim is the batch/video axis (BatchCollator cats it).
        inputs = BACKEND_BATCH_BUILDERS[self.backend](self.processor, [frames])
        # float16 for the heavy pixel tensor halves cache size; the model recasts to its dtype.
        return {
            k: (v.half() if (k == "pixel_values_videos" and torch.is_floating_point(v)) else v)
            for k, v in dict(inputs).items()
        }

    def __getitem__(self, i: int):
        row = self.rows[i]
        video_id = row.get("video_id", "?")
        try:
            verb_id = int(row["verb_class"])
            noun_id = int(row["noun_class"])
            action_id = self.action_map.get((verb_id, noun_id), -1)

            if self.cache_dir:
                path = self._cache_path(row)
                inputs = None
                if os.path.exists(path):
                    try:
                        inputs = torch.load(path, map_location="cpu", weights_only=False)
                    except Exception:  # noqa: BLE001 -- corrupt/partial cache file: recompute
                        inputs = None
                if inputs is None:
                    inputs = self._decode_and_tokenize(row)
                    tmp = f"{path}.tmp.{os.getpid()}.{i}"
                    torch.save(inputs, tmp)
                    os.replace(tmp, path)  # atomic on same FS; safe under concurrent workers
                return inputs, verb_id, noun_id, action_id, video_id, i

            # No cache: return raw frames; BatchCollator tokenizes the whole batch at once.
            from decord import VideoReader, cpu

            video_path = str(Path(self.video_root) / row["participant_id"] / f"{video_id}.MP4")
            vr_probe = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            vfps = vr_probe.get_avg_fps()
            indices = compute_clip_window(int(row["start_frame"]), vfps, self.num_frames, self.target_fps)
            frames, _ = decode_frames(video_path, indices, decode_size=self.decode_size)
            frames = _resample_frames(frames, self.probe_num_frames)
            return frames, verb_id, noun_id, action_id, video_id, i
        except Exception as exc:  # noqa: BLE001 -- surfaced via logging in the training/eval loop
            return None, -1, -1, -1, f"{video_id} ({exc})", i


class CachedEmbedsDataset(torch.utils.data.Dataset):
    """Loads pre-pruned post-vision-tower LLM inputs (B12 1-min→4s-token Qwen study).

    Built by ``scripts/build_qwen_prune_cache.py``: each sample's frozen-vision-tower output,
    pruned to the 4s-level token budget, is saved as ``inputs_embeds`` (+ M-RoPE
    ``position_ids``). Training loads these and runs ONLY the LLM (LoRA) + heads — the vision
    tower (61% of step time) never runs during training. Mirrors the V-JEPA2 1-min side, which
    caches its frozen-encoder pruned tokens. ``action_id`` is derived here from ``action_map``
    so the cache stays independent of the action vocabulary.

    Each item is ``({"inputs_embeds", "position_ids"}, verb_id, noun_id, action_id, video_id, i)``;
    with batch_size=1 the stock ``BatchCollator`` dict path passes the tensors through unchanged.
    """

    def __init__(self, rows, cache_dir: str, action_map):
        self.rows = rows
        self.cache_dir = cache_dir
        self.action_map = action_map

    def __len__(self) -> int:
        return len(self.rows)

    def _cache_path(self, row):
        return os.path.join(
            self.cache_dir,
            f"{row['participant_id']}__{row.get('video_id', 'x')}__{int(row['start_frame'])}.pt",
        )

    def __getitem__(self, i: int):
        row = self.rows[i]
        video_id = row.get("video_id", "?")
        try:
            verb_id = int(row["verb_class"])
            noun_id = int(row["noun_class"])
            action_id = self.action_map.get((verb_id, noun_id), -1)
            payload = torch.load(self._cache_path(row), map_location="cpu", weights_only=False)
            inputs = {"inputs_embeds": payload["inputs_embeds"], "position_ids": payload["position_ids"]}
            return inputs, verb_id, noun_id, action_id, video_id, i
        except Exception as exc:  # noqa: BLE001 -- surfaced via logging in the training/eval loop
            return None, -1, -1, -1, f"{video_id} ({exc})", i


class BatchCollator:
    """Tokenizes a whole batch at once (runs in the DataLoader worker process)."""

    def __init__(self, processor, backend: str):
        self.processor = processor
        self.builder = BACKEND_BATCH_BUILDERS[backend]

    def __call__(self, batch):
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            return None, None, None, None, [], []
        items, verb_ids, noun_ids, action_ids, video_ids, row_idxs = zip(*batch)
        if isinstance(items[0], dict):
            # Cache mode: each item is a per-sample processor output. Every key's leading dim
            # is the batch/video axis (pixel_values_videos is patches, input_ids/grid_thw are
            # [1, ...]), so cat along dim 0 reproduces the processor's native batched layout.
            inputs = {k: torch.cat([it[k] for it in items], dim=0) for k in items[0]}
        else:
            inputs = self.builder(self.processor, list(items))
        return (
            inputs,
            torch.tensor(verb_ids, dtype=torch.long),
            torch.tensor(noun_ids, dtype=torch.long),
            torch.tensor(action_ids, dtype=torch.long),
            list(video_ids),
            list(row_idxs),
        )


def cached_embeds_collate(batch):
    """Left-pad a batch of pre-pruned LLM inputs to the batch's max sequence length.

    Pruned clips vary in length (edge clips keep fewer tokens), so we pad on the LEFT and
    build an attention_mask: left-padding keeps each sample's real last token at index -1, so
    the probe's ``hidden_states[-1][:, -1, :]`` read stays valid for every row. Pad position_ids
    are dummy (0) and excluded by the mask. Lets the cheap LLM-only step run at batch_size>1 so
    the GPU is well-fed (avoids the low-util cancellation that bs=1 micro-steps would risk).
    """
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None, None, None, [], []
    items, verb_ids, noun_ids, action_ids, video_ids, row_idxs = zip(*batch)
    lengths = [it["inputs_embeds"].shape[1] for it in items]
    max_l = max(lengths)
    B = len(items)
    D = items[0]["inputs_embeds"].shape[-1]
    emb = items[0]["inputs_embeds"].new_zeros(B, max_l, D)
    pos = items[0]["position_ids"].new_zeros(3, B, max_l)
    attn = torch.zeros(B, max_l, dtype=torch.long)
    for i, it in enumerate(items):
        l = lengths[i]
        emb[i, max_l - l:, :] = it["inputs_embeds"][0]
        pos[:, i, max_l - l:] = it["position_ids"][:, 0, :]
        attn[i, max_l - l:] = 1
    inputs = {"inputs_embeds": emb, "position_ids": pos, "attention_mask": attn}
    return (
        inputs,
        torch.tensor(verb_ids, dtype=torch.long),
        torch.tensor(noun_ids, dtype=torch.long),
        torch.tensor(action_ids, dtype=torch.long),
        list(video_ids),
        list(row_idxs),
    )


def _make_loader(dataset, processor, backend: str, batch_size: int, num_workers: int, cached: bool = False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cached_embeds_collate if cached else BatchCollator(processor, backend),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    v_c, v_t = defaultdict(int), defaultdict(int)
    n_c, n_t = defaultdict(int), defaultdict(int)
    a_c, a_t = defaultdict(int), defaultdict(int)
    v_top3 = n_top3 = a_top3 = total = 0
    v_top5 = n_top5 = a_top5 = 0

    for inputs, v_ids, n_ids, a_ids, video_ids, row_idxs in loader:
        if inputs is None:
            logger.warning("Eval batch %s failed entirely in dataset, skipping", row_idxs)
            continue
        inputs = {k: v.to(device) for k, v in inputs.items()}
        v_logits, n_logits, a_logits = model(**inputs)
        for i in range(v_logits.shape[0]):
            vi, ni, ai = int(v_ids[i]), int(n_ids[i]), int(a_ids[i])
            v_t[vi] += 1
            n_t[ni] += 1
            v_in_top5 = vi in v_logits[i].topk(5).indices.tolist()
            n_in_top5 = ni in n_logits[i].topk(5).indices.tolist()
            if v_in_top5:
                v_c[vi] += 1
                v_top5 += 1
            if n_in_top5:
                n_c[ni] += 1
                n_top5 += 1
            if vi in v_logits[i].topk(3).indices.tolist():
                v_top3 += 1
            if ni in n_logits[i].topk(3).indices.tolist():
                n_top3 += 1
            if ai != -1:
                a_t[ai] += 1
                if ai in a_logits[i].topk(5).indices.tolist():
                    a_c[ai] += 1
                    a_top5 += 1
                if ai in a_logits[i].topk(3).indices.tolist():
                    a_top3 += 1
            total += 1

    def cmr(c, t):
        r = [c.get(k, 0) / v for k, v in t.items()]
        return float(np.mean(r) * 100) if r else 0.0

    n_act = sum(a_t.values())
    return {
        "n_samples": total,
        "verb_top3": 100 * v_top3 / max(total, 1),
        "noun_top3": 100 * n_top3 / max(total, 1),
        "action_top3": 100 * a_top3 / max(n_act, 1),
        "verb_top5": 100 * v_top5 / max(total, 1),
        "noun_top5": 100 * n_top5 / max(total, 1),
        "action_top5": 100 * a_top5 / max(n_act, 1),
        "verb_r5": cmr(v_c, v_t),
        "noun_r5": cmr(n_c, n_t),
        "action_r5": cmr(a_c, a_t),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(BACKEND_BATCH_BUILDERS), required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--verb-classes-csv", required=True)
    parser.add_argument("--noun-classes-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--probe-num-frames", type=int, default=8)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size (matches PhD's BATCH_SIZE)")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Effective batch = batch_size * grad_accum_steps")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume-from",
        default="",
        help="Optional probe-last.pt/probe-best.pt checkpoint to continue training from. "
        "--num-epochs remains the total target epoch count.",
    )
    parser.add_argument(
        "--prune-keep-ratio",
        type=float,
        default=1.0,
        help="qwen25vl only: fraction of merged video tokens kept (attention-importance) "
        "before the LLM. 1.0 = no pruning. See qwen_token_pruning.py (B12 equal-compute study).",
    )
    parser.add_argument(
        "--prune-chunk-size",
        type=int,
        default=0,
        help="qwen25vl only: query-axis chunk for the importance softmax (0 = whole per-frame segment).",
    )
    parser.add_argument(
        "--qwen-frame-size",
        type=int,
        default=0,
        help="qwen25vl only: resize each video frame to this square size before the processor "
        "(0 = native). Use 256 to match V-JEPA2's resolution for the equal-compute study.",
    )
    parser.add_argument(
        "--preproc-cache-dir",
        default="",
        help="If set, cache per-sample processor tensors under here (keyed by clip config) and "
        "reload them on later epochs instead of re-decoding+re-tokenizing -- removes the CPU "
        "dataloader bottleneck so the GPU stops starving. Shared across keep_ratio runs.",
    )
    parser.add_argument(
        "--cached-embeds-dir",
        default="",
        help="qwen25vl only: if set, load pre-pruned post-vision-tower LLM inputs from here "
        "(built by scripts/build_qwen_prune_cache.py) and train ONLY the LLM (LoRA)+heads, "
        "skipping the vision tower entirely. Forces batch_size=1; pruning must already be baked "
        "into the cache (use --prune-keep-ratio 1.0). The B12 1-min->4s-token Qwen experiment.",
    )
    args = parser.parse_args()

    global _QWEN_FRAME_SIZE
    _QWEN_FRAME_SIZE = args.qwen_frame_size

    torch.manual_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_id or DEFAULT_MODEL_IDS[args.backend]

    logger.info("Loading backend=%s model_id=%s device=%s dtype=%s", args.backend, model_id, device, args.torch_dtype)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=args.local_files_only)
    if hasattr(processor, "tokenizer"):
        # left-padding so the batch's last token position ([:, -1, :]) is always real
        # content, never a pad token, regardless of each sample's actual sequence length
        processor.tokenizer.padding_side = "left"
    if args.backend == "llava_onevision":
        from transformers import LlavaOnevisionForConditionalGeneration

        backbone = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=args.local_files_only
        ).to(device)
        hidden_size = backbone.config.text_config.hidden_size
    elif args.backend == "qwen25vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=args.local_files_only
        ).to(device)
        # PhD's script reads qwen_raw.config.hidden_size directly; newer transformers
        # versions nest this under config.text_config instead, so fall back to that.
        hidden_size = getattr(backbone.config, "hidden_size", None) or backbone.config.text_config.hidden_size
    else:
        from transformers import MllamaForConditionalGeneration

        backbone = MllamaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=args.local_files_only
        ).to(device)
        hidden_size = backbone.config.text_config.hidden_size

    for p in backbone.parameters():
        p.requires_grad = False
    apply_lora_to_llm(backbone, rank=args.lora_rank, alpha=args.lora_alpha)
    if args.backend == "qwen25vl":
        # Vision tower is frozen -> wrap its forward in no_grad to save activation memory,
        # matching the PhD's script (qwen_raw.visual.forward wrapped before grad-ckpt enable).
        # Newer transformers versions nest it under backbone.model.visual instead of
        # backbone.visual directly, so locate it by name rather than assuming the attribute.
        visual_module = dict(backbone.named_modules()).get("visual") or dict(backbone.named_modules()).get("model.visual")
        if visual_module is None:
            raise RuntimeError("Could not locate the Qwen2.5-VL vision tower submodule (expected 'visual' or 'model.visual').")
        _orig_visual_fwd = visual_module.forward

        @torch.no_grad()
        def _visual_no_grad(*a, **kw):
            return _orig_visual_fwd(*a, **kw)

        visual_module.forward = _visual_no_grad
    backbone.gradient_checkpointing_enable()
    backbone.config.use_cache = False

    prune_state = None
    if args.prune_keep_ratio < 1.0:
        if args.backend != "qwen25vl":
            raise ValueError("--prune-keep-ratio < 1.0 is only supported for backend=qwen25vl")
        from app.hdepic_lora_action_anticipation.qwen_token_pruning import install_qwen_video_token_pruner

        prune_state = install_qwen_video_token_pruner(
            backbone, keep_ratio=args.prune_keep_ratio, chunk_size=args.prune_chunk_size
        )
        logger.info(
            "Qwen video-token pruning ENABLED: keep_ratio=%.3f chunk_size=%d (importance = "
            "vision-tower last-block attention received; pruned before the LLM)",
            args.prune_keep_ratio,
            args.prune_chunk_size,
        )

    if args.cached_embeds_dir:
        if args.backend != "qwen25vl":
            raise ValueError("--cached-embeds-dir is only supported for backend=qwen25vl")
        if args.prune_keep_ratio < 1.0:
            raise ValueError(
                "--cached-embeds-dir already holds pruned tokens; set --prune-keep-ratio 1.0 "
                "(do not also install the in-model pruner)"
            )
        logger.info(
            "Cached pruned-embeds training from: %s (vision tower skipped; batch_size=%d, "
            "left-padded variable-length collation)", args.cached_embeds_dir, args.batch_size,
        )

    verb_vocab = load_class_vocab(args.verb_classes_csv)
    noun_vocab = load_class_vocab(args.noun_classes_csv)
    num_verbs, num_nouns = len(verb_vocab), len(noun_vocab)

    def read_rows(path):
        return list(csv.DictReader(open(path, newline="", encoding="utf-8")))

    train_rows = read_rows(args.train_csv)
    val_rows = read_rows(args.val_csv)
    test_rows = read_rows(args.test_csv) if args.test_csv else []
    action_map = load_action_map({"train": train_rows, "val": val_rows, "test": test_rows})
    num_actions = len(action_map)
    logger.info("verbs=%d nouns=%d actions=%d (observed verb,noun pairs)", num_verbs, num_nouns, num_actions)

    random.Random(args.seed).shuffle(train_rows)
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    logger.info("train=%d val=%d test=%d", len(train_rows), len(val_rows), len(test_rows))

    model = VLMProbe(backbone, hidden_size, num_verbs, num_nouns, num_actions)
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info("Total trainable: %.2fM (LoRA + 3 heads)", sum(p.numel() for p in trainable) / 1e6)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # Preprocessing cache: a config-specific subdir so different clip settings never collide.
    # Identical across keep_ratio (pruning is in-model), so all runs with the same clip config
    # share one cache. Built lazily on first epoch; reused (as cheap tensor loads) thereafter.
    cache_dir = ""
    if args.preproc_cache_dir:
        cache_dir = os.path.join(
            args.preproc_cache_dir,
            f"{args.backend}_nf{args.num_frames}_pnf{args.probe_num_frames}"
            f"_fps{args.target_fps}_px{args.qwen_frame_size}",
        )
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Preprocessing cache: %s", cache_dir)

    def make_ds(rows):
        if args.cached_embeds_dir:
            return CachedEmbedsDataset(rows, args.cached_embeds_dir, action_map)
        return HDEpicProbeDataset(
            rows, args.video_root, action_map, args.num_frames, args.probe_num_frames,
            args.target_fps, decode_size=args.qwen_frame_size,
            processor=processor, backend=args.backend, cache_dir=cache_dir or None,
        )

    cached = bool(args.cached_embeds_dir)
    train_ds = make_ds(train_rows)
    val_ds = make_ds(val_rows)
    train_loader = _make_loader(train_ds, processor, args.backend, args.batch_size, args.num_workers, cached)
    val_loader = _make_loader(val_ds, processor, args.backend, args.batch_size, args.num_workers, cached)
    test_loader = None
    if test_rows:
        test_ds = make_ds(test_rows)
        test_loader = _make_loader(test_ds, processor, args.backend, args.batch_size, args.num_workers, cached)

    batches_per_epoch = len(train_loader)
    steps_per_epoch = max(1, batches_per_epoch // args.grad_accum_steps)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_verb_r5 = 0.0
    global_step = 0
    start_epoch = 0

    def save_ckpt(path, epoch, metrics):
        sd = model.state_dict()
        lora_state = {
            k: v
            for k, v in sd.items()
            if "lora_A" in k or "lora_B" in k or k.startswith(("verb_head", "noun_head", "action_head"))
        }
        torch.save(
            {
                "epoch": epoch,
                "model_lora": lora_state,
                "metrics": metrics,
                "action_map": action_map,
                "lora_rank": args.lora_rank,
                "backend": args.backend,
                "model_id": model_id,
            },
            path,
        )

    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device, weights_only=False)
        if ck.get("backend") != args.backend:
            raise ValueError(f"Resume backend mismatch: checkpoint={ck.get('backend')} args={args.backend}")
        if len(ck.get("action_map", {})) != len(action_map):
            raise ValueError(
                f"Resume action_map size mismatch: checkpoint={len(ck.get('action_map', {}))} current={len(action_map)}"
            )
        missing, unexpected = model.load_state_dict(ck["model_lora"], strict=False)
        start_epoch = int(ck.get("epoch", 0))
        global_step = start_epoch * steps_per_epoch
        best_verb_r5 = float((ck.get("metrics") or {}).get("verb_r5", 0.0))
        logger.info(
            "Resumed from %s at epoch=%d global_step=%d best_verb_r5=%.2f "
            "(missing=%d unexpected=%d)",
            args.resume_from,
            start_epoch,
            global_step,
            best_verb_r5,
            len(missing),
            len(unexpected),
        )
        for _ in range(global_step):
            scheduler.step()

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_loss = 0
        optimizer.zero_grad()

        for step, (inputs, v_ids, n_ids, a_ids, video_ids, row_idxs) in enumerate(train_loader):
            if inputs is None:
                logger.warning("Batch %s (epoch %d) failed entirely in dataset, skipping", row_idxs, epoch)
                continue
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                v_ids_t = v_ids.to(device)
                n_ids_t = n_ids.to(device)
                a_ids_t = a_ids.to(device)
                v_logits, n_logits, a_logits = model(**inputs)
                loss = criterion(v_logits, v_ids_t) + criterion(n_logits, n_ids_t)
                valid_a = a_ids_t >= 0
                if valid_a.any():
                    loss = loss + criterion(a_logits[valid_a], a_ids_t[valid_a])
                (loss / args.grad_accum_steps).backward()
                epoch_loss += loss.item()
                n_loss += 1
                if prune_state is not None and prune_state.last_stats is not None and global_step == 0 and step == 0:
                    s = prune_state.last_stats
                    logger.info(
                        "Pruning realized: video tokens %d -> %d (keep_ratio=%.3f); LLM seq_len %d -> %d",
                        s["n_tokens"], s["n_kept"], args.prune_keep_ratio, s["seq_len"], s["pruned_seq_len"],
                    )
            except Exception:
                logger.exception("Batch %s (epoch %d) failed, skipping", row_idxs, epoch)
                continue

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "epoch=%d step=%d batch=%d/%d avg_loss=%.4f lr=%.2e (%.1fs/batch)",
                        epoch,
                        global_step,
                        step + 1,
                        batches_per_epoch,
                        epoch_loss / max(1, n_loss),
                        scheduler.get_last_lr()[0],
                        elapsed / max(1, step + 1),
                    )

        logger.info("epoch=%d done avg_loss=%.4f elapsed=%.0fs", epoch, epoch_loss / max(1, n_loss), time.time() - t0)

        metrics = evaluate(model, val_loader, device)
        logger.info("epoch=%d val: %s", epoch, metrics)
        save_ckpt(str(Path(args.output_dir) / "probe-last.pt"), epoch + 1, metrics)
        if metrics["verb_r5"] > best_verb_r5:
            best_verb_r5 = metrics["verb_r5"]
            save_ckpt(str(Path(args.output_dir) / "probe-best.pt"), epoch + 1, metrics)
            logger.info("Saved best (verb R@5=%.1f%%)", best_verb_r5)
        model.train()

    logger.info("Training complete. Best val verb R@5=%.1f%%", best_verb_r5)

    if test_loader is not None:
        best_ck = torch.load(str(Path(args.output_dir) / "probe-best.pt"), map_location=device, weights_only=False)
        model.load_state_dict(best_ck["model_lora"], strict=False)
        test_metrics = evaluate(model, test_loader, device)
        logger.info("Test set results: %s", test_metrics)
        with open(str(Path(args.output_dir) / "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
