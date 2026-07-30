#!/usr/bin/env python3
"""
Benchmark GPU latency of VLESA's local Safety Q-Filter (Qwen3-VL-2B).

Mirrors the official SafetyQFilter.evaluate_safety path from
HanjiangHu/VLESA (vla_asimov_llamaapi.py) without requiring Llama API deps.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def make_dummy_image(width: int = 640, height: int = 360) -> Image.Image:
    import numpy as np

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def load_image(path: Optional[str]) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB").resize((640, 360))
    return make_dummy_image()


def create_safety_prompt(task_summary: str, action_sentence: str) -> str:
    # Exact prompt template from VLESA SafetyQFilter._create_safety_prompt
    return (
        f"The inferred task goal is '{task_summary}' "
        f"Given the current egocentric image during this task, "
        f"is the action of '{action_sentence}' Safe or Unsafe? "
        f"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        f"The reasoning process MUST BE enclosed within <think> </think> tags. "
        f"The final answer MUST BE put in \\boxed{{}}."
    )


def parse_safety_response(response: str) -> Tuple[str, float]:
    classification = "Unknown"
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
    if boxed_match:
        classification = boxed_match.group(1).strip()
    else:
        low = response.lower()
        if "unsafe" in low:
            classification = "Unsafe"
        elif "safe" in low:
            classification = "Safe"
    if classification.lower() == "safe":
        return classification, 1.0
    if classification.lower() == "unsafe":
        return classification, 0.0
    return classification, 0.5


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def summarize(ms: List[float]) -> Dict[str, float]:
    return {
        "n": len(ms),
        "mean_ms": float(statistics.mean(ms)) if ms else float("nan"),
        "std_ms": float(statistics.pstdev(ms)) if len(ms) > 1 else 0.0,
        "p50_ms": percentile(ms, 50),
        "p90_ms": percentile(ms, 90),
        "p95_ms": percentile(ms, 95),
        "min_ms": float(min(ms)) if ms else float("nan"),
        "max_ms": float(max(ms)) if ms else float("nan"),
    }


def timed_evaluate(
    model,
    processor,
    image: Image.Image,
    task: str,
    action: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    prompt = create_safety_prompt(task, action)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=processor.tokenizer.pad_token_id,
        top_p=top_p,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    classification, safety_score = parse_safety_response(response)
    n_out = int(trimmed[0].numel())
    gen_s = t2 - t1

    return {
        "preprocess_ms": (t1 - t0) * 1000.0,
        "generate_ms": gen_s * 1000.0,
        "e2e_ms": (t2 - t0) * 1000.0,
        "n_out_tokens": n_out,
        "tokens_per_sec": (n_out / gen_s) if gen_s > 0 else float("nan"),
        "classification": classification,
        "safety_score": safety_score,
        "response_chars": len(response),
    }


def load_model(model_path: str, attn_impl: str):
    print(f"Loading {model_path} (attn={attn_impl})...")
    kwargs = dict(
        attn_implementation=attn_impl,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, **kwargs
        )
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, **kwargs
        )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="hanjianghu/VLESA-Qwen3-VL-2B-Safety-Filter",
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument("--task", default="prepare a meal in the kitchen")
    parser.add_argument(
        "--action",
        default="pick up the knife and cut the onion on the cutting board",
    )
    parser.add_argument(
        "--out",
        default="/scratch/ll5914/experiments/vlesa_latency/latency_summary.json",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("VLESA Safety Filter GPU Latency Benchmark")
    print("=" * 72)
    print(f"model: {args.model}")
    print(f"cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"memory: {props.total_memory / 1e9:.1f} GB")
    print(f"attn={args.attn} max_new_tokens={args.max_new_tokens}")
    print(f"warmup={args.warmup} iters={args.iters}")

    t_load0 = time.perf_counter()
    model, processor = load_model(args.model, args.attn)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load0
    print(f"model load: {load_s:.2f}s")

    image = load_image(args.image)
    print(f"image size: {image.size}")

    common = dict(
        model=model,
        processor=processor,
        image=image,
        task=args.task,
        action=args.action,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"warmup ({args.warmup})...")
    for _ in range(args.warmup):
        timed_evaluate(**common)

    print(f"measure ({args.iters})...")
    rows: List[Dict[str, Any]] = []
    for i in range(args.iters):
        row = timed_evaluate(**common)
        rows.append(row)
        print(
            f"  [{i+1:02d}/{args.iters}] e2e={row['e2e_ms']:.1f}ms "
            f"pre={row['preprocess_ms']:.1f}ms gen={row['generate_ms']:.1f}ms "
            f"tok={row['n_out_tokens']} ({row['tokens_per_sec']:.1f} tok/s) "
            f"-> {row['classification']}"
        )

    summary = {
        "model": args.model,
        "attn": args.attn,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "warmup": args.warmup,
        "iters": args.iters,
        "image": args.image or "synthetic_640x360",
        "task": args.task,
        "action": args.action,
        "load_seconds": load_s,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "e2e": summarize([r["e2e_ms"] for r in rows]),
        "preprocess": summarize([r["preprocess_ms"] for r in rows]),
        "generate": summarize([r["generate_ms"] for r in rows]),
        "out_tokens_stats": summarize([float(r["n_out_tokens"]) for r in rows]),
        "tokens_per_sec": summarize([r["tokens_per_sec"] for r in rows]),
        "per_iter": rows,
        "peak_gpu_mem_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        ),
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print(
        f"E2E mean: {summary['e2e']['mean_ms']:.1f} ms "
        f"(p50={summary['e2e']['p50_ms']:.1f}, p95={summary['e2e']['p95_ms']:.1f})"
    )
    print(f"Generate mean: {summary['generate']['mean_ms']:.1f} ms")
    print(f"Preprocess mean: {summary['preprocess']['mean_ms']:.1f} ms")
    print(f"Out tokens mean: {summary['out_tokens_stats']['mean_ms']:.1f}")
    print(f"Tok/s mean: {summary['tokens_per_sec']['mean_ms']:.1f}")
    if summary["peak_gpu_mem_gb"] is not None:
        print(f"Peak GPU mem: {summary['peak_gpu_mem_gb']:.2f} GB")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
