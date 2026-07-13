#!/usr/bin/env python
"""Build the pruned-video-embedding cache for Qwen2.5-VL 1-min→4s-token training.

The V-JEPA2 analogue (B12 1-min side) caches the frozen encoder's pruned tokens so training
skips the encoder every epoch. Here we do the same for Qwen: the vision tower is frozen and
the attention-importance pruning is computed from the (frozen) vision tower's last-block
attention, so the kept-token set — and the resulting pruned ``inputs_embeds`` / ``position_ids``
fed to the LLM — are deterministic per sample across epochs. We compute them once and save them.

Training (``train_vlm_probe_lora.py --cached-embeds-dir``) then loads these and runs ONLY the
LLM (with LoRA) + probe heads, turning a 3.6 s/sample (vision-tower-bound) step into a
~0.16 s/sample LLM step on ~1296 surviving tokens (4s-level budget at keep_ratio≈0.0667).

Per-sample artifact ``<participant>__<video>__<startframe>.pt``::
    {"inputs_embeds": (1, L', D) bf16,   # text + kept video tokens, post-vision-tower
     "position_ids":  (3, 1, L') long,   # M-RoPE (t,y,x) for the survivors
     "verb_id", "noun_id": int,          # action_id is derived at train time from action_map
     "video_id": str, "stats": dict}

Usage (inside the singularity overlay, GPU):
  python scripts/build_qwen_prune_cache.py \
      --train-csv ... --val-csv ... --test-csv ... \
      --video-root ... --verb-classes-csv ... --noun-classes-csv ... \
      --out-dir data/qwen_prune_embed_cache/kr0p0667_nf480_px256 \
      --keep-ratio 0.0667 --num-frames 480 --qwen-frame-size 256 --num-workers 10
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time

import torch

from app.hdepic_lora_action_anticipation.train_vlm_probe_lora import (
    BatchCollator,
    HDEpicProbeDataset,
    load_action_map,
)
from app.hdepic_lora_action_anticipation.zeroshot_vlm_prompting import load_class_vocab
import app.hdepic_lora_action_anticipation.train_vlm_probe_lora as train_mod
from app.hdepic_lora_action_anticipation.qwen_token_pruning import (
    install_qwen_video_token_pruner,
    compute_pruned_llm_inputs,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _cache_path(out_dir: str, row: dict) -> str:
    return os.path.join(
        out_dir,
        f"{row['participant_id']}__{row.get('video_id', 'x')}__{int(row['start_frame'])}.pt",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--test-csv", default="")
    p.add_argument("--video-root", required=True)
    p.add_argument("--verb-classes-csv", required=True)
    p.add_argument("--noun-classes-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--keep-ratio", type=float, required=True)
    p.add_argument("--prune-chunk-size", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=480)
    p.add_argument("--probe-num-frames", type=int, default=480)
    p.add_argument("--target-fps", type=float, default=8.0)
    p.add_argument("--qwen-frame-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--splits", default="train,val,test", help="comma list of splits to build")
    p.add_argument("--max-samples-per-split", type=int, default=0,
                   help="smoke: cap each split to the first N rows (0 = all)")
    args = p.parse_args()

    if not (0.0 < args.keep_ratio < 1.0):
        raise ValueError(f"--keep-ratio must be in (0,1), got {args.keep_ratio}")

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.torch_dtype)

    # --- rows + maps (action_map identical to training; we only store verb/noun ids) ---
    def read_rows(path):
        return list(csv.DictReader(open(path, newline="", encoding="utf-8")))

    train_rows = read_rows(args.train_csv)
    val_rows = read_rows(args.val_csv)
    test_rows = read_rows(args.test_csv) if args.test_csv else []
    # action_map from the FULL splits (must match training's vocabulary regardless of any cap)
    action_map = load_action_map({"train": train_rows, "val": val_rows, "test": test_rows})
    if args.max_samples_per_split > 0:
        n = args.max_samples_per_split
        train_rows, val_rows, test_rows = train_rows[:n], val_rows[:n], test_rows[:n]
        logger.info("max_samples_per_split=%d (CSV order)", n)
    splits = {"train": train_rows, "val": val_rows, "test": test_rows}

    # --- model: frozen Qwen + pruner (no LoRA: embeds are LoRA-independent) ---
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    train_mod._QWEN_FRAME_SIZE = args.qwen_frame_size
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, local_files_only=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    logger.info("Loading Qwen %s on %s dtype=%s", QWEN_MODEL_ID, device, dtype)
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=dtype, local_files_only=True
    ).to(device)
    backbone.eval()
    for p_ in backbone.parameters():
        p_.requires_grad = False
    state = install_qwen_video_token_pruner(
        backbone, keep_ratio=args.keep_ratio, chunk_size=args.prune_chunk_size
    )
    logger.info("Pruner installed: keep_ratio=%.4f chunk_size=%d", args.keep_ratio, args.prune_chunk_size)

    for split_name in args.splits.split(","):
        split_name = split_name.strip()
        rows = splits.get(split_name)
        if not rows:
            logger.info("Split %s empty/absent, skipping", split_name)
            continue

        existing = sum(1 for r in rows if os.path.exists(_cache_path(args.out_dir, r)))
        # Pre-filter to UNCACHED rows so a resumed build never re-decodes (and never sits
        # GPU-idle skipping) already-built samples -- that idle skip phase both wastes ~2.4s
        # decode per done row and trips the low-util watchdog before reaching new work.
        todo = [r for r in rows if not os.path.exists(_cache_path(args.out_dir, r))]
        logger.info("[%s] %d/%d already cached; building %d", split_name, existing, len(rows), len(todo))
        if not todo:
            continue

        ds = HDEpicProbeDataset(
            todo, args.video_root, action_map, args.num_frames, args.probe_num_frames,
            args.target_fps, decode_size=args.qwen_frame_size,
            processor=processor, backend="qwen25vl", cache_dir=None,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
            collate_fn=BatchCollator(processor, "qwen25vl"),
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )

        t0 = time.time()
        done = errors = skipped = 0
        for inputs, v_ids, n_ids, a_ids, video_ids, row_idxs in loader:
            ri = int(row_idxs[0]) if row_idxs else -1
            if inputs is None or ri < 0:
                errors += 1
                logger.warning("[%s] batch %s failed in dataset", split_name, video_ids)
                continue
            row = todo[ri]
            out_path = _cache_path(args.out_dir, row)
            if os.path.exists(out_path):  # concurrent builder won the race; fine
                skipped += 1
                continue
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    emb, pos, _ = compute_pruned_llm_inputs(state, inputs)
                payload = {
                    # keep model dtype (bf16) so training feeds it straight to the LLM
                    "inputs_embeds": emb.to(torch.bfloat16).cpu(),
                    "position_ids": pos.cpu(),
                    "verb_id": int(v_ids[0]),
                    "noun_id": int(n_ids[0]),
                    "video_id": video_ids[0],
                    "stats": dict(state.last_stats or {}),
                }
                tmp = f"{out_path}.tmp.{os.getpid()}"
                torch.save(payload, tmp)
                os.replace(tmp, out_path)
                done += 1
            except Exception:
                errors += 1
                logger.exception("[%s] sample %s failed", split_name, video_ids)
                continue

            if done % 50 == 0 and done:
                el = time.time() - t0
                rate = (done + skipped) / max(el, 1e-6)
                eta = (len(todo) - done - skipped) / max(rate, 1e-6)
                logger.info(
                    "[%s] %d built (+%d skip, %d err) | %.2fs/sample ETA %.0fmin | last stats=%s",
                    split_name, done, skipped, errors, el / max(done, 1), eta / 60, state.last_stats,
                )
        logger.info(
            "[%s] DONE: %d built, %d skipped, %d errors in %.1f min",
            split_name, done, skipped, errors, (time.time() - t0) / 60,
        )

    logger.info("All splits done. Cache dir: %s", args.out_dir)


if __name__ == "__main__":
    main()
