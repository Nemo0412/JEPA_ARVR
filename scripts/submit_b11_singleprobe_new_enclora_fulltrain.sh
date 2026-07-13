#!/bin/bash
set -euo pipefail

# Full B11 JEPA_ARVR-style single new probe + encoder LoRA training run.
# Launch this after the smoke confirms nonzero pose coverage, finite optimizer
# steps, and sane early train/val loss scale.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm"

export LORA_TAG="${LORA_TAG:-hdepic-singleprobe-new-full-enclora-gaze-pose-h100-r8-allblocks-bs4-10ep}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_single_probe_encoder_lora_gaze_pose_matrix_fulltrain.yaml}"
export LORA_PRETRAINED_PROBE="${LORA_PRETRAINED_PROBE-}"
export ENCODER_LORA_TARGET_SUFFIXES="${ENCODER_LORA_TARGET_SUFFIXES:-attn.qkv,attn.proj}"

echo "[submit-b11-single-fulltrain] project=${PROJECT_ROOT}"
echo "[submit-b11-single-fulltrain] tag=${LORA_TAG}"
echo "[submit-b11-single-fulltrain] script=${RUN_SCRIPT}"

sbatch \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",LORA_TAG="${LORA_TAG}",CONFIG_PATH="${CONFIG_PATH}",DEBUG_SUBSET_PATH=,EVAL_MAX_TRAIN_ITERS=0,EVAL_NUM_EPOCHS=10,RESUME_CHECKPOINT=0,EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,EVAL_LR=0.0001,EVAL_BATCH_SIZE=4,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5 \
  "${RUN_SCRIPT}"
