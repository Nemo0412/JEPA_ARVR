#!/bin/bash
set -euo pipefail

# Minimal fp32 smoke to verify latest.pt save after scaler fix (2 train iters + val).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm"
CONFIG_PATH="${PROJECT_ROOT}/configs/generated/hdepic_single_probe_encoder_lora_gaze_pose_matrix_smoke.yaml"
TAG="${LORA_TAG:-hdepic-singleprobe-fp32-savecheck-i2}"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},CONFIG_PATH=${CONFIG_PATH},DEBUG_SUBSET_PATH=,EVAL_MAX_TRAIN_ITERS=2,EVAL_NUM_EPOCHS=1,RESUME_CHECKPOINT=0,EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,EVAL_LR=0.0001,EVAL_BATCH_SIZE=4,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv,attn.proj,LORA_PRETRAINED_PROBE=,LORA_TAG=${TAG},EVAL_USE_BFLOAT16=0,EVAL_NUM_WORKERS=2"

echo "[submit-fp32-savecheck] tag=${TAG} EVAL_MAX_TRAIN_ITERS=2"
sbatch --export="${export_csv}" "${RUN_SCRIPT}"
