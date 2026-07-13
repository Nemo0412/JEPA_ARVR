#!/usr/bin/env python
"""Profile B12 per-sample TRUE FLOPs on GPU.

This measures the actual code paths used in the B12 comparison:
- V-JEPA2 frozen encoder+predictor forward, then AttentiveClassifier backward.
- Qwen2.5-VL frozen vision tower, LoRA LLM + heads backward, with keep_ratio 1.0/0.15.

FLOPs are counted with ``torch.utils.flop_counter.FlopCounterMode``, which has correct
analytic formulas for ``scaled_dot_product_attention`` (forward AND backward) plus conv/mm/
bmm/addmm. Unlike ``torch.profiler(with_flops=True)``, it does NOT undercount fused SDPA,
which is the dominant attention cost here -- so these are the "real" per-sample FLOPs.

For each train step we report:
- forward-only FLOPs (a single forward pass with grad disabled), and
- full train-step FLOPs (forward + backward; with activation/gradient checkpointing this
  includes the recomputed forward, i.e. the true training compute).
Wall-time and peak memory are measured in a separate, un-instrumented pass.
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


def _patch_sdpa_flops_for_gqa() -> None:
    """Relax torch's SDPA flop-count head-equality assertion for grouped-query attention.

    Qwen2.5-VL uses GQA (fewer KV heads than query heads); the KV is broadcast to the query
    head count inside SDPA, so the true flops already use ``b*h_query`` -- which is exactly what
    torch's formulas compute. Only the ``h_q == h_k`` sanity assertion is wrong for GQA, so we
    reinstall identical math without that cross-head check.
    """
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


def _count_flops(step_fn) -> tuple[int, dict]:
    """Run step_fn under FlopCounterMode and return (total_flops, per_op_breakdown)."""
    counter = FlopCounterMode(display=False)
    with counter:
        step_fn()
    torch.cuda.synchronize()
    total = int(counter.get_total_flops())
    # Global per-op breakdown (op packet -> flops), stringified for JSON.
    global_counts = counter.get_flop_counts().get("Global", {})
    breakdown = {str(op): int(f) for op, f in sorted(global_counts.items(), key=lambda kv: -kv[1])}
    return total, breakdown


def _write_result(out_dir: Path, name: str, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"name": name, **payload}, indent=2, sort_keys=True), flush=True)
    print(f"[profile] wrote {path}", flush=True)


def _profile_step(name: str, full_step_fn, out_dir: Path, warmup: int, forward_fn=None) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        full_step_fn()
    torch.cuda.synchronize()

    fwd_flops, fwd_breakdown = (None, None)
    if forward_fn is not None:
        with torch.no_grad():
            fwd_flops, fwd_breakdown = _count_flops(forward_fn)

    full_flops, full_breakdown = _count_flops(full_step_fn)

    # Wall-time / peak memory measured WITHOUT the flop-counter dispatch overhead.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    full_step_fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    _write_result(
        out_dir,
        name,
        {
            "forward_flops_per_sample": fwd_flops,
            "forward_tflops_per_sample": (fwd_flops / 1e12) if fwd_flops is not None else None,
            "train_step_flops_per_sample": full_flops,
            "train_step_tflops_per_sample": full_flops / 1e12,
            "forward_flops_by_op": fwd_breakdown,
            "train_step_flops_by_op": full_breakdown,
            "wall_time_sec": elapsed,
            "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "note": "FlopCounterMode true FLOPs (SDPA fwd+bwd counted); full step = forward+backward incl. checkpoint recompute",
        },
    )


def profile_vjepa(out_dir: Path, warmup: int) -> None:
    from evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar import init_module
    from evals.action_anticipation_frozen.models import AttentiveClassifier

    device = torch.device("cuda")
    model_kwargs = {
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
    wrapper_kwargs = {"no_predictor": False, "num_output_frames": 2, "num_steps": 1}
    backbone = init_module(
        frames_per_clip=32,
        frames_per_second=8,
        resolution=256,
        checkpoint="/scratch/yh6416/VJEPA2-EXP/checkpoints/vitl.pt",
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    ).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    classifier = AttentiveClassifier(
        verb_classes={i: i for i in range(106)},
        noun_classes={i: i for i in range(303)},
        action_classes={i: i for i in range(1681)},
        embed_dim=1024,
        num_heads=16,
        depth=4,
        use_activation_checkpointing=True,
    ).to(device)
    classifier.train()
    opt = torch.optim.AdamW(classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(1, 3, 32, 256, 256, device=device)
    anticipation = torch.ones(1, device=device)
    labels = {
        "verb": torch.tensor([0], device=device),
        "noun": torch.tensor([0], device=device),
        "action": torch.tensor([0], device=device),
    }

    def forward_only():
        feats = backbone(x, anticipation)
        logits = classifier(feats)
        return sum(criterion(logits[k], labels[k]) for k in labels)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            feats = backbone(x, anticipation)
        logits = classifier(feats)
        loss = sum(criterion(logits[k], labels[k]) for k in labels)
        loss.backward()
        opt.step()

    _profile_step(
        "vjepa2_encoder_predictor_probe_train_step", step, out_dir, warmup, forward_fn=forward_only
    )


def _build_qwen_model(keep_ratio: float, dtype: torch.dtype):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    import app.hdepic_lora_action_anticipation.train_vlm_probe_lora as train_mod
    from app.hdepic_lora_action_anticipation.qwen_token_pruning import install_qwen_video_token_pruner
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
        model_id, torch_dtype=dtype, local_files_only=True
    ).to("cuda")
    hidden_size = getattr(backbone.config, "hidden_size", None) or backbone.config.text_config.hidden_size
    for p in backbone.parameters():
        p.requires_grad = False
    apply_lora_to_llm(backbone, rank=16, alpha=32.0)

    visual_module = dict(backbone.named_modules()).get("visual") or dict(backbone.named_modules()).get("model.visual")
    if visual_module is None:
        raise RuntimeError("Could not locate Qwen visual module")
    orig_visual_fwd = visual_module.forward

    @torch.no_grad()
    def visual_no_grad(*a, **kw):
        return orig_visual_fwd(*a, **kw)

    visual_module.forward = visual_no_grad
    backbone.gradient_checkpointing_enable()
    backbone.config.use_cache = False

    if keep_ratio < 1.0:
        install_qwen_video_token_pruner(backbone, keep_ratio=keep_ratio, chunk_size=0)

    model = VLMProbe(backbone, hidden_size, 106, 303, 1680).to("cuda")
    return processor, model


def profile_qwen(out_dir: Path, warmup: int, keep_ratio: float) -> None:
    from app.hdepic_lora_action_anticipation.train_vlm_probe_lora import build_qwen_inputs_batch

    dtype = torch.bfloat16
    processor, model = _build_qwen_model(keep_ratio, dtype)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    frames = np.zeros((32, 256, 256, 3), dtype=np.uint8)
    inputs_cpu = build_qwen_inputs_batch(processor, [frames])
    labels = {
        "verb": torch.tensor([0], device="cuda"),
        "noun": torch.tensor([0], device="cuda"),
        "action": torch.tensor([0], device="cuda"),
    }

    def forward_only():
        inputs = {k: v.to("cuda") for k, v in inputs_cpu.items()}
        v_logits, n_logits, a_logits = model(**inputs)
        return (
            criterion(v_logits, labels["verb"])
            + criterion(n_logits, labels["noun"])
            + criterion(a_logits, labels["action"])
        )

    def step():
        opt.zero_grad(set_to_none=True)
        inputs = {k: v.to("cuda") for k, v in inputs_cpu.items()}
        v_logits, n_logits, a_logits = model(**inputs)
        loss = (
            criterion(v_logits, labels["verb"])
            + criterion(n_logits, labels["noun"])
            + criterion(a_logits, labels["action"])
        )
        loss.backward()
        opt.step()

    tag = str(keep_ratio).replace(".", "p")
    _profile_step(
        f"qwen25vl_kr{tag}_probe_lora_train_step", step, out_dir, warmup, forward_fn=forward_only
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/scratch/yh6416/VJEPA2-EXP/outputs/b12_flops_profile")
    parser.add_argument("--target", choices=["all", "vjepa", "qwen"], default="all")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    torch.manual_seed(0)
    if args.target in ("all", "vjepa"):
        profile_vjepa(out_dir, args.warmup)
    if args.target in ("all", "qwen"):
        profile_qwen(out_dir, args.warmup, keep_ratio=1.0)
        profile_qwen(out_dir, args.warmup, keep_ratio=0.15)


if __name__ == "__main__":
    main()
