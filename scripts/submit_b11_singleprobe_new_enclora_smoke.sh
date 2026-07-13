#!/bin/bash
set -euo pipefail

# Short smoke to compare bf16 vs fp32 numerics after LoRA-A randn*0.02 init
# and [grad-diag] logging. Uses 150 train iters for fast turnaround.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm"
CONFIG_PATH="${PROJECT_ROOT}/configs/generated/hdepic_single_probe_encoder_lora_gaze_pose_matrix_smoke.yaml"

COMMON_EXPORT=(
  "ALL"
  "PROJECT_ROOT=${PROJECT_ROOT}"
  "CONFIG_PATH=${CONFIG_PATH}"
  "DEBUG_SUBSET_PATH="
  "EVAL_MAX_TRAIN_ITERS=150"
  "EVAL_NUM_EPOCHS=1"
  "RESUME_CHECKPOINT=0"
  "EVAL_SINGLE_PROBE=1"
  "LORA_PROBE_TRAIN_MODE=full"
  "EVAL_LR=0.0001"
  "EVAL_BATCH_SIZE=4"
  "EVAL_GRAD_CLIP=1.0"
  "EVAL_WARMUP_EPOCHS=2"
  "ENCODER_LORA_RANK=8"
  "ENCODER_LORA_ALPHA=16.0"
  "ENCODER_LORA_LAST_N_BLOCKS=0"
  "ENCODER_LORA_LR_MULT=0.5"
  "ENCODER_LORA_TARGET_SUFFIXES=attn.qkv,attn.proj"
  "LORA_PRETRAINED_PROBE="
)

submit_one() {
  local tag="$1"
  local use_bf16="$2"
  local export_csv
  export_csv=$(IFS=,; echo "${COMMON_EXPORT[*]},LORA_TAG=${tag},EVAL_USE_BFLOAT16=${use_bf16}")
  echo "[submit-graddiag-smoke] tag=${tag} EVAL_USE_BFLOAT16=${use_bf16}"
  sbatch --export="${export_csv}" "${RUN_SCRIPT}"
}

submit_one "hdepic-singleprobe-enclora-graddiag-smoke-bf16-i150" "1"
submit_one "hdepic-singleprobe-enclora-graddiag-smoke-fp32-i150" "0"
