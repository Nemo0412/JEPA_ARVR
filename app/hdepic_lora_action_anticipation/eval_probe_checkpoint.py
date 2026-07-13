"""Re-evaluate a saved train_vlm_probe_lora.py checkpoint with the current evaluate()
(adds instance-level top-5 accuracy alongside the existing top-3/class-mean-R@5 metrics),
without retraining. Loads the LoRA+head state dict saved in probe-{best,last}.pt.
"""

from __future__ import annotations

import argparse
import csv
import json

import torch

import app.hdepic_lora_action_anticipation.train_vlm_probe_lora as train_mod
from app.hdepic_lora_action_anticipation.train_vlm_probe_lora import (
    DEFAULT_MODEL_IDS,
    VLMProbe,
    apply_lora_to_llm,
    evaluate,
    load_action_map,
    _make_loader,
    HDEpicProbeDataset,
)
from app.hdepic_lora_action_anticipation.zeroshot_vlm_prompting import load_class_vocab


def read_rows(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--train-csv", required=True, help="used only to rebuild the action_map's class space")
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--verb-classes-csv", required=True)
    parser.add_argument("--noun-classes-csv", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--probe-num-frames", type=int, default=8)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prune-keep-ratio", type=float, default=1.0)
    parser.add_argument("--prune-chunk-size", type=int, default=0)
    parser.add_argument("--qwen-frame-size", type=int, default=0)
    parser.add_argument("--preproc-cache-dir", default="")
    args = parser.parse_args()

    train_mod._QWEN_FRAME_SIZE = args.qwen_frame_size
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backend = ck["backend"]
    model_id = ck.get("model_id") or DEFAULT_MODEL_IDS[backend]
    lora_rank = ck["lora_rank"]
    print(f"[eval] checkpoint={args.checkpoint} backend={backend} model_id={model_id} epoch={ck.get('epoch')}")
    print(f"[eval] checkpoint's own recorded metrics: {ck.get('metrics')}")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=args.local_files_only)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    if backend == "llava_onevision":
        from transformers import LlavaOnevisionForConditionalGeneration

        backbone = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=args.local_files_only
        ).to(device)
        hidden_size = backbone.config.text_config.hidden_size
    elif backend == "qwen25vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=args.local_files_only
        ).to(device)
        hidden_size = getattr(backbone.config, "hidden_size", None) or backbone.config.text_config.hidden_size
    else:
        from transformers import MllamaForConditionalGeneration

        backbone = MllamaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=args.local_files_only
        ).to(device)
        hidden_size = backbone.config.text_config.hidden_size

    for p in backbone.parameters():
        p.requires_grad = False
    apply_lora_to_llm(backbone, rank=lora_rank, alpha=32.0)
    if backend == "qwen25vl":
        visual_module = dict(backbone.named_modules()).get("visual") or dict(backbone.named_modules()).get("model.visual")
        if visual_module is None:
            raise RuntimeError("Could not locate the Qwen2.5-VL vision tower submodule (expected 'visual' or 'model.visual').")
        _orig_visual_fwd = visual_module.forward

        @torch.no_grad()
        def _visual_no_grad(*a, **kw):
            return _orig_visual_fwd(*a, **kw)

        visual_module.forward = _visual_no_grad

    if args.prune_keep_ratio < 1.0:
        if backend != "qwen25vl":
            raise ValueError("--prune-keep-ratio < 1.0 is only supported for backend=qwen25vl")
        from app.hdepic_lora_action_anticipation.qwen_token_pruning import install_qwen_video_token_pruner

        install_qwen_video_token_pruner(backbone, keep_ratio=args.prune_keep_ratio, chunk_size=args.prune_chunk_size)
        print(f"[eval] qwen pruning enabled: keep_ratio={args.prune_keep_ratio} chunk_size={args.prune_chunk_size}")

    verb_vocab = load_class_vocab(args.verb_classes_csv)
    noun_vocab = load_class_vocab(args.noun_classes_csv)
    num_verbs, num_nouns = len(verb_vocab), len(noun_vocab)

    train_rows = read_rows(args.train_csv)
    val_rows = read_rows(args.val_csv)
    test_rows = read_rows(args.test_csv) if args.test_csv else []
    action_map = load_action_map({"train": train_rows, "val": val_rows, "test": test_rows})
    num_actions = len(action_map)
    assert num_actions == len(ck["action_map"]), (
        f"action_map size mismatch: rebuilt={num_actions} vs checkpoint={len(ck['action_map'])}"
    )

    model = VLMProbe(backbone, hidden_size, num_verbs, num_nouns, num_actions)
    missing, unexpected = model.load_state_dict(ck["model_lora"], strict=False)
    print(f"[eval] load_state_dict missing(non-lora ok)={len(missing)} unexpected={len(unexpected)}")

    eval_rows = read_rows(args.eval_csv)
    cache_dir = ""
    if args.preproc_cache_dir:
        cache_dir = (
            f"{args.preproc_cache_dir}/{backend}_nf{args.num_frames}_pnf{args.probe_num_frames}"
            f"_fps{args.target_fps}_px{args.qwen_frame_size}"
        )
        print(f"[eval] preprocessing cache: {cache_dir}")

    eval_ds = HDEpicProbeDataset(
        eval_rows,
        args.video_root,
        action_map,
        args.num_frames,
        args.probe_num_frames,
        args.target_fps,
        decode_size=args.qwen_frame_size,
        processor=processor,
        backend=backend,
        cache_dir=cache_dir or None,
    )
    eval_loader = _make_loader(eval_ds, processor, backend, args.batch_size, args.num_workers)

    metrics = evaluate(model, eval_loader, device)
    print(f"[eval] recomputed metrics on {args.eval_csv}: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
