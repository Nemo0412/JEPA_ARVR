#!/bin/bash
set -euo pipefail

# Full-P01 short smoke for B11 encoder-LoRA + gaze/pose matrix.
# This intentionally leaves DEBUG_SUBSET_PATH empty so the run sees the real
# train distribution, dataloader pressure, gaze/pose coverage, and 20-head
# shared-encoder gradient behavior.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_encoder_lora_gaze_pose_matrix_20head_lowlr_h100.slurm"

export LORA_TAG="${LORA_TAG:-hdepic-20head-lora-enclora-gaze-pose-lrscale002-r4-last4-bs2-fullsmoke-ep1-i600}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_encoder_lora_gaze_pose_matrix_20head_lowlr_fullsmoke.yaml}"
export ENCODER_LORA_TARGET_SUFFIXES="${ENCODER_LORA_TARGET_SUFFIXES:-attn.qkv,attn.proj}"

echo "[submit-b11-fullsmoke] project=${PROJECT_ROOT}"
echo "[submit-b11-fullsmoke] tag=${LORA_TAG}"
echo "[submit-b11-fullsmoke] script=${RUN_SCRIPT}"

sbatch \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",LORA_TAG="${LORA_TAG}",CONFIG_PATH="${CONFIG_PATH}",DEBUG_SUBSET_PATH=,EVAL_MAX_TRAIN_ITERS=600,EVAL_NUM_EPOCHS=1,RESUME_CHECKPOINT=0,EVAL_LR_SCALE=0.02,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,ENCODER_LORA_RANK=4,ENCODER_LORA_ALPHA=8.0,ENCODER_LORA_LAST_N_BLOCKS=4,ENCODER_LORA_LR_MULT=0.1 \
  "${RUN_SCRIPT}"
