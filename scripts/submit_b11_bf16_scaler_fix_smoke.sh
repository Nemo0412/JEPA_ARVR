#!/bin/bash
set -euo pipefail

# Validate the bf16 GradScaler fix (make_grad_scaler -> disabled scaler under bf16).
#
# Root cause: GradScaler() was enabled under bf16 (eval.py). bf16 needs no loss
# scaling; the default x65536 scaling overflowed the deep-encoder bf16 backward
# to inf, which became NaN in zero-initialised LoRA-A grads (0 * inf). That drove
# both failure modes: discard -> 40% iters drop encoder grads -> no learning;
# keep -> NaN grads stepped into LoRA -> encoder NaN output -> exit(1).
#
# This smoke runs the FIXED code (PROJECT_ROOT = debug worktree) in normal
# discard mode (EVAL_KEEP_NONFINITE_GRADS=0). Pass criteria:
#   - ~0 "Discarding encoder-LoRA grads" lines (vs 60/150 before)
#   - loss falls and train acc rises
#   - no "Nan detected at output of encoder" / exit(1)
#
# To reproduce the OLD broken behaviour for A/B, add EVAL_BF16_GRAD_SCALER=1.

# Code (the edited worktree) — drives PYTHONPATH, cd, and the config template.
CODE_ROOT="${CODE_ROOT:-/path/to/VJEPA2-EXP-debug-keep-nonfinite}"
# Data / checkpoints / overlay live only in the parent checkout.
DATA_ROOT="${DATA_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${CODE_ROOT}/scripts/run_hdepic_lora_probe.slurm"

mkdir -p "${CODE_ROOT}/logs"

submit_one() {
    local tag="$1"
    local bf16="$2"
    local iters="$3"
    local keep_nf="$4"
    local export_csv="ALL,PROJECT_ROOT=${CODE_ROOT},OVERLAY=${DATA_ROOT}/overlay-15GB-500K.ext3,CHECKPOINT=${DATA_ROOT}/checkpoints/vitg-384.pt,ADAPTED_ANNOTATIONS_DIR=${DATA_ROOT}/data/hdepic_vjepa_annotations,ADAPTED_VIDEO_ROOT=${DATA_ROOT}/data/hdepic_vjepa_videos,GAZE_MODE=none,ENCODER_LORA_ENABLED=1,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv,attn.proj,ENCODER_LORA_ACTIVATION_CHECKPOINTING=1,LORA_PROBE_TRAIN_MODE=full,EVAL_SINGLE_PROBE=1,EVAL_LR=0.0001,EVAL_BATCH_SIZE=4,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,EVAL_MAX_TRAIN_ITERS=${iters},EVAL_NUM_EPOCHS=1,EVAL_NUM_WORKERS=2,EVAL_GRAD_DIAG_INTERVAL=10,EVAL_KEEP_NONFINITE_GRADS=${keep_nf},EVAL_USE_BFLOAT16=${bf16},LORA_PRETRAINED_PROBE=,LORA_TAG=${tag},CONFIG_PATH=${CODE_ROOT}/configs/generated/hdepic_bf16_scaler_fix_smoke.yaml,RESUME_CHECKPOINT=0,OUTPUT_DIR=${CODE_ROOT}/outputs/hdepic_lora_action_anticipation"

    echo "[submit-bf16-scaler-fix] tag=${tag} bf16=${bf16} iters=${iters} keep_nf=${keep_nf} (scaler disabled via make_grad_scaler)"
    sbatch \
        --partition=h100_tandon \
        --account=your_slurm_account \
        --gres=gpu:h100:1 \
        --cpus-per-task=12 \
        --mem=768GB \
        --time=8:00:00 \
        --job-name="VJEPA2-EXP__bf16_scaler_fix" \
        --output="${CODE_ROOT}/logs/hdepic_bf16_scaler_fix_%j.out" \
        --error="${CODE_ROOT}/logs/hdepic_bf16_scaler_fix_%j.err" \
        --export="${export_csv}" \
        "${RUN_SCRIPT}"
}

# bf16, normal discard mode, 400 iters: trains cleanly with the scaler fix.
submit_one "hdepic-bf16-scaler-fix-i400" 1 400 0
