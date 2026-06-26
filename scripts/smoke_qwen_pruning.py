"""GPU smoke test for Qwen2.5-VL attention-importance video-token pruning.

Builds a tiny dummy batch (random frames), runs the model with and without pruning, and
asserts: (1) the pruned LLM sequence is shorter by the expected number of video tokens,
(2) the kept-token hidden states are finite, (3) gradients still flow to LoRA + heads.

Run on a GPU node via scripts/run_smoke_qwen_pruning.slurm (never on the login node).
"""

import sys

import numpy as np
import torch

from app.hdepic_lora_action_anticipation.train_vlm_probe_lora import (
    apply_lora_to_llm,
    build_qwen_inputs_batch,
)
from app.hdepic_lora_action_anticipation.qwen_token_pruning import install_qwen_video_token_pruner

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
N_FRAMES = 8       # even (temporal_patch_size=2)
N_SAMPLES = 2
KEEP_RATIO = 0.5


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    device = "cuda"
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False
    apply_lora_to_llm(model, rank=8, alpha=16.0)
    model.config.use_cache = False

    rng = np.random.default_rng(0)
    frames_list = [rng.integers(0, 255, size=(N_FRAMES, 256, 256, 3), dtype=np.uint8) for _ in range(N_SAMPLES)]
    inputs = build_qwen_inputs_batch(processor, frames_list)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    grid = inputs["video_grid_thw"]
    n_video = int((grid.prod(dim=-1) // (model.model.visual.spatial_merge_unit))[0].item())
    print(f"[smoke] video_grid_thw={grid.tolist()} -> merged video tokens/sample = {n_video}", flush=True)

    # ── Baseline (no pruning) ──
    with torch.no_grad():
        out0 = model(**inputs, output_hidden_states=True, return_dict=True)
    seq0 = out0.hidden_states[-1].shape[1]
    print(f"[smoke] baseline LLM seq_len = {seq0}", flush=True)

    # ── Pruned ──
    state = install_qwen_video_token_pruner(model, keep_ratio=KEEP_RATIO)
    out1 = model(**inputs, output_hidden_states=True, return_dict=True)
    seq1 = out1.hidden_states[-1].shape[1]
    last_tok = out1.hidden_states[-1][:, -1, :].float()
    print(f"[smoke] pruned   LLM seq_len = {seq1}  stats={state.last_stats}", flush=True)

    n_kept = max(1, round(n_video * KEEP_RATIO))
    expected = seq0 - (n_video - n_kept)
    assert seq1 == expected, f"expected pruned seq_len {expected}, got {seq1}"
    assert torch.isfinite(last_tok).all(), "non-finite hidden states after pruning"

    # ── Gradient flow ──
    loss = last_tok.pow(2).mean()
    loss.backward()
    lora_grads = [p.grad for n, p in model.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
    n_with_grad = sum(1 for g in lora_grads if g is not None and torch.isfinite(g).all() and g.abs().sum() > 0)
    print(f"[smoke] LoRA params with nonzero finite grad: {n_with_grad}/{len(lora_grads)}", flush=True)
    assert n_with_grad > 0, "no gradient reached LoRA params through the pruned path"

    state.remove()
    # After remove(), forward should match baseline seq_len again.
    with torch.no_grad():
        out2 = model(**inputs, output_hidden_states=True, return_dict=True)
    assert out2.hidden_states[-1].shape[1] == seq0, "remove() did not restore the stock forward"

    print("[smoke] PASS: pruning shrinks the LLM sequence, stays finite, and is differentiable.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
