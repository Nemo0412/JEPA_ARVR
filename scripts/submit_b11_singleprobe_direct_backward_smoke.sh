#!/bin/bash
set -euo pipefail

# 5ch gaze/pose adapter + JEPA_ARVR-style direct backward (no tokens_proxy detach).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm"
TAG="${LORA_TAG:-hdepic-singleprobe-enclora-direct-bwd-graddiag-smoke-bf16-i150}"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},LORA_TAG=${TAG},CONFIG_PATH=${PROJECT_ROOT}/configs/generated/hdepic_single_probe_encoder_lora_gaze_pose_matrix_smoke.yaml,DEBUG_SUBSET_PATH=,EVAL_MAX_TRAIN_ITERS=150,EVAL_NUM_EPOCHS=1,RESUME_CHECKPOINT=0,EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,EVAL_LR=0.0001,EVAL_BATCH_SIZE=4,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv,attn.proj,LORA_PRETRAINED_PROBE=,EVAL_USE_BFLOAT16=1,EVAL_NUM_WORKERS=2,EVAL_GRAD_DIAG_INTERVAL=10,BINARY_INPUT_ADAPTER_DIRECT_BACKWARD=1"

echo "[submit-direct-bwd-smoke] tag=${TAG} BINARY_INPUT_ADAPTER_DIRECT_BACKWARD=1"
sbatch --export="${export_csv}" "${RUN_SCRIPT}"
