#!/usr/bin/env python
"""Profile Qwen2.5-VL-3B per-sample FLOPs + wall-time breakdown for 1-min window.

Measures the same code path as train_vlm_probe_lora.py but for 480 frames (60s @ 8fps,
256px) — the 1-min window the advisor requested — and compares against the prior 4s / 32-frame
configuration so we have an apples-to-apples FLOPs ratio.

Breakdown reported per configuration:
  1. Vision tower (32-layer ViT) forward — fixed cost pre-prune, scales with N_frames
  2. LLM forward (36 layers on merged video tokens) — scales with N_tokens, attention is O(N²)
  3. LLM backward (gradient through LoRA params + heads)
  4. Total train step (= 1+2+3, with grad-checkpoint recompute counted)

FLOPs via FlopCounterMode (analytic, counts SDPA fwd+bwd without the head-equality bug for GQA).
Wall-time via CUDA Events (measures actual GPU execution, not dispatch overhead).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


# ---------------------------------------------------------------------------
# GQA SDPA FLOPs patch (same as profile_b12_flops.py — required for Qwen GQA)
# ---------------------------------------------------------------------------

def _patch_sdpa_flops_for_gqa() -> None:
    import torch.utils.flop_counter as fc
    bmm_flop = fc.bmm_flop

    def sdpa_flop_count(query_shape, key_shape, value_shape):
        b, h, s_q, d_q = query_shape
        _b2, _h2, s_k, _d2 = key_shape
        _b3, _h3, _s3, d_v = value_shape
        total = bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
        total += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_v))
        return total

    def sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape):
        b, h, s_q, d_q = query_shape
        _b2, _h2, s_k, _d2 = key_shape
        _b3, _h3, _s3, d_v = value_shape
        total = bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
        total += bmm_flop((b * h, s_q, d_v), (b * h, d_v, s_k))
        total += bmm_flop((b * h, s_k, s_q), (b * h, s_q, d_v))
        total += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_q))
        total += bmm_flop((b * h, d_q, s_q), (b * h, s_q, s_k))
        return total

    fc.sdpa_flop_count = sdpa_flop_count
    fc.sdpa_backward_flop_count = sdpa_backward_flop_count


_patch_sdpa_flops_for_gqa()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cuda_event_pair():
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _count_flops(step_fn):
    counter = FlopCounterMode(display=False)
    with counter:
        step_fn()
    torch.cuda.synchronize()
    total = int(counter.get_total_flops())
    global_counts = counter.get_flop_counts().get("Global", {})
    breakdown = {str(op): int(f) for op, f in sorted(global_counts.items(), key=lambda kv: -kv[1])}
    return total, breakdown


def _write_result(out_dir: Path, name: str, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"name": name, **payload}, indent=2, sort_keys=True), flush=True)
    print(f"[profile] wrote {path}", flush=True)


# ---------------------------------------------------------------------------
# Model builder (shared between 4s and 1min configs)
# ---------------------------------------------------------------------------

def _build_qwen_model(dtype=torch.bfloat16):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import app.hdepic_lora_action_anticipation.train_vlm_probe_lora as train_mod
    from app.hdepic_lora_action_anticipation.train_vlm_probe_lora import (
        DEFAULT_MODEL_IDS,
        VLMProbe,
        apply_lora_to_llm,
    )

    model_id = DEFAULT_MODEL_IDS["qwen25vl"]
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    train_mod._QWEN_FRAME_SIZE = 256

    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, local_files_only=True,
    ).to("cuda")
    hidden_size = (
        getattr(backbone.config, "hidden_size", None)
        or backbone.config.text_config.hidden_size
    )
    for p in backbone.parameters():
        p.requires_grad = False
    apply_lora_to_llm(backbone, rank=16, alpha=32.0)

    # Freeze vision tower (same as training: no_grad wrapper)
    visual_module = (
        dict(backbone.named_modules()).get("visual")
        or dict(backbone.named_modules()).get("model.visual")
    )
    orig_visual_fwd = visual_module.forward

    @torch.no_grad()
    def visual_no_grad(*a, **kw):
        return orig_visual_fwd(*a, **kw)

    visual_module.forward = visual_no_grad

    backbone.gradient_checkpointing_enable()
    backbone.config.use_cache = False

    model = VLMProbe(backbone, hidden_size, 106, 303, 1680).to("cuda")
    return processor, model, visual_module, orig_visual_fwd


# ---------------------------------------------------------------------------
# Per-section wall-time timing via CUDA Events
# ---------------------------------------------------------------------------

def _measure_section_times(model, inputs_gpu, labels, opt, criterion, n_runs=3):
    """Returns dict of avg wall-time (ms) for each section, measured with CUDA Events.

    Sections:
      vision_tower   — frozen ViT forward on all frames
      llm_forward    — 36-layer transformer forward on merged tokens
      llm_backward   — backward through LoRA + heads
      total_step     — end-to-end (including opt.step)
    """
    # Install hooks to record CUDA events at vision-tower in/out and LLM in/out.
    # We time by wrapping the backbone forward and capturing events around key sub-calls.
    vis_start, vis_end = _cuda_event_pair()
    llm_fwd_start, llm_fwd_end = _cuda_event_pair()
    bwd_start, bwd_end = _cuda_event_pair()
    total_start, total_end = _cuda_event_pair()

    # Locate sub-modules
    backbone = model.backbone
    vis_mod = (
        dict(backbone.named_modules()).get("visual")
        or dict(backbone.named_modules()).get("model.visual")
    )
    llm_mod = (
        dict(backbone.named_modules()).get("model")
        or dict(backbone.named_modules()).get("language_model")
    )

    vis_times, llm_fwd_times, bwd_times, total_times = [], [], [], []

    def fwd_hook_pre_vis(mod, inp):  vis_start.record()
    def fwd_hook_post_vis(mod, inp, out): vis_end.record()
    def fwd_hook_pre_llm(mod, inp, kw): llm_fwd_start.record()
    def fwd_hook_post_llm(mod, inp, out): llm_fwd_end.record()

    h1 = vis_mod.register_forward_pre_hook(fwd_hook_pre_vis)
    h2 = vis_mod.register_forward_hook(fwd_hook_post_vis)
    h3 = llm_mod.register_forward_pre_hook(fwd_hook_pre_llm, with_kwargs=True)
    h4 = llm_mod.register_forward_hook(fwd_hook_post_llm)

    try:
        for _ in range(n_runs):
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            total_start.record()

            v_logits, n_logits, a_logits = model(**inputs_gpu)
            loss = (
                criterion(v_logits, labels["verb"])
                + criterion(n_logits, labels["noun"])
                + criterion(a_logits, labels["action"])
            )

            llm_fwd_end.record()  # LLM fwd done (after loss, before bwd)
            bwd_start.record()
            loss.backward()
            bwd_end.record()
            opt.step()
            total_end.record()
            torch.cuda.synchronize()

            vis_times.append(vis_start.elapsed_time(vis_end))
            llm_fwd_times.append(llm_fwd_start.elapsed_time(llm_fwd_end))
            bwd_times.append(bwd_start.elapsed_time(bwd_end))
            total_times.append(total_start.elapsed_time(total_end))
    finally:
        h1.remove(); h2.remove(); h3.remove(); h4.remove()

    def avg(lst): return sum(lst) / len(lst)
    return {
        "vision_tower_fwd_ms": avg(vis_times),
        "llm_fwd_ms": avg(llm_fwd_times),
        "llm_bwd_ms": avg(bwd_times),
        "total_step_ms": avg(total_times),
    }


# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------

def profile_qwen_window(out_dir: Path, n_frames: int, warmup: int, n_runs: int, keep_ratio: float = 1.0) -> None:
    from app.hdepic_lora_action_anticipation.train_vlm_probe_lora import build_qwen_inputs_batch

    tag = f"{n_frames}f" if keep_ratio >= 1.0 else f"{n_frames}f_kr{str(keep_ratio).replace('.', 'p')}"
    label = tag
    print(f"\n{'='*60}", flush=True)
    print(f"[profile] Qwen2.5-VL-3B  window={n_frames} frames  ({n_frames/8:.0f}s @ 8fps)  keep_ratio={keep_ratio}", flush=True)
    print(f"{'='*60}\n", flush=True)

    dtype = torch.bfloat16
    processor, model, _, _ = _build_qwen_model(dtype)
    # In-model pruning: single forward = ViT(n_frames) → prune → LLM(kept). Same path as the
    # 4s kr0p15 profile, just at this window — directly comparable to the no-prune forward FLOPs.
    if keep_ratio < 1.0:
        from app.hdepic_lora_action_anticipation.qwen_token_pruning import install_qwen_video_token_pruner
        install_qwen_video_token_pruner(model.backbone, keep_ratio=keep_ratio, chunk_size=0)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    frames = np.zeros((n_frames, 256, 256, 3), dtype=np.uint8)
    inputs_cpu = build_qwen_inputs_batch(processor, [frames])
    inputs_gpu = {k: v.to("cuda") for k, v in inputs_cpu.items()}

    n_video_tokens = inputs_cpu.get("pixel_values_videos", inputs_cpu.get("pixel_values", None))
    if n_video_tokens is not None:
        # grid_thw shape: (n_clips, 3) with [t, h, w]; merged tokens = prod
        grid_thw = inputs_cpu.get("video_grid_thw", inputs_cpu.get("image_grid_thw", None))
        if grid_thw is not None:
            t, h, w = grid_thw[0].tolist()
            merged_tokens = int(t * h * w)
            print(f"  grid_thw = ({t}, {h}, {w})  → {merged_tokens} merged video tokens", flush=True)

    labels = {
        "verb": torch.tensor([0], device="cuda"),
        "noun": torch.tensor([0], device="cuda"),
        "action": torch.tensor([0], device="cuda"),
    }

    def step():
        opt.zero_grad(set_to_none=True)
        v_logits, n_logits, a_logits = model(**inputs_gpu)
        loss = (
            criterion(v_logits, labels["verb"])
            + criterion(n_logits, labels["noun"])
            + criterion(a_logits, labels["action"])
        )
        loss.backward()
        opt.step()

    def forward_only():
        with torch.no_grad():
            v_logits, n_logits, a_logits = model(**inputs_gpu)
        return v_logits

    # Warmup
    print(f"  warmup ({warmup} run(s))…", flush=True)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    # FLOPs
    print("  counting FLOPs (forward only)…", flush=True)
    torch.cuda.empty_cache()
    fwd_flops, fwd_breakdown = _count_flops(forward_only)

    print("  counting FLOPs (full train step)…", flush=True)
    torch.cuda.empty_cache()
    full_flops, full_breakdown = _count_flops(step)

    # Wall-time breakdown
    print(f"  measuring wall-time sections ({n_runs} run(s))…", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timing = _measure_section_times(model, inputs_gpu, labels, opt, criterion, n_runs=n_runs)

    peak_mem = int(torch.cuda.max_memory_allocated())

    result = {
        "n_frames": n_frames,
        "window_sec": n_frames / 8.0,
        "keep_ratio": keep_ratio,
        "forward_flops_per_sample": fwd_flops,
        "forward_tflops_per_sample": fwd_flops / 1e12,
        "train_step_flops_per_sample": full_flops,
        "train_step_tflops_per_sample": full_flops / 1e12,
        "forward_flops_by_op": fwd_breakdown,
        "train_step_flops_by_op": full_breakdown,
        "wall_time_breakdown_ms": timing,
        "peak_cuda_memory_bytes": peak_mem,
        "peak_cuda_memory_gb": peak_mem / 1e9,
    }

    _write_result(out_dir, f"qwen25vl_{label}_train_step", result)

    # Pretty summary
    print(f"\n  === SUMMARY ({label}) ===", flush=True)
    print(f"  FLOPs fwd:       {fwd_flops/1e12:.2f} TF", flush=True)
    print(f"  FLOPs train step:{full_flops/1e12:.2f} TF", flush=True)
    print(f"  Vision tower fwd:{timing['vision_tower_fwd_ms']:.0f} ms", flush=True)
    print(f"  LLM fwd:         {timing['llm_fwd_ms']:.0f} ms", flush=True)
    print(f"  LLM bwd:         {timing['llm_bwd_ms']:.0f} ms", flush=True)
    print(f"  Total step:      {timing['total_step_ms']:.0f} ms", flush=True)
    print(f"  Peak VRAM:       {peak_mem/1e9:.1f} GB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/path/to/VJEPA2-EXP/outputs/b12_flops_profile")
    parser.add_argument(
        "--windows", default="32,480",
        help="Comma-separated list of frame counts to profile (default: '32,480' = 4s and 1min)",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--keep-ratio", type=float, default=1.0,
                        help="<1.0 installs in-model pruning so the single forward is "
                             "ViT(window)→prune→LLM(kept), comparable to the no-prune forward FLOPs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    torch.manual_seed(0)
    for n in [int(x) for x in args.windows.split(",")]:
        profile_qwen_window(out_dir, n_frames=n, warmup=args.warmup, n_runs=args.n_runs, keep_ratio=args.keep_ratio)


if __name__ == "__main__":
    main()
