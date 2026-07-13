#!/bin/bash
set -euo pipefail

# Full B11 encoder-LoRA + gaze/pose matrix training run.
# Launch this only after the full-P01 short smoke shows sane early training:
# action train should lift off, healthy heads should not collapse, and val loss
# should stay in a reasonable scale.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_encoder_lora_gaze_pose_matrix_20head_lowlr_h100.slurm"

export LORA_TAG="${LORA_TAG:-hdepic-20head-lora-enclora-gaze-pose-lrscale002-r4-last4-bs2-10ep}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_encoder_lora_gaze_pose_matrix_20head_lowlr_fulltrain.yaml}"
export ENCODER_LORA_TARGET_SUFFIXES="${ENCODER_LORA_TARGET_SUFFIXES:-attn.qkv,attn.proj}"

echo "[submit-b11-fulltrain] project=${PROJECT_ROOT}"
echo "[submit-b11-fulltrain] tag=${LORA_TAG}"
echo "[submit-b11-fulltrain] script=${RUN_SCRIPT}"

sbatch \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",LORA_TAG="${LORA_TAG}",CONFIG_PATH="${CONFIG_PATH}",DEBUG_SUBSET_PATH=,EVAL_MAX_TRAIN_ITERS=0,EVAL_NUM_EPOCHS=10,RESUME_CHECKPOINT=0,EVAL_LR_SCALE=0.02,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,ENCODER_LORA_RANK=4,ENCODER_LORA_ALPHA=8.0,ENCODER_LORA_LAST_N_BLOCKS=4,ENCODER_LORA_LR_MULT=0.1 \
  "${RUN_SCRIPT}"
