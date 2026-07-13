#!/bin/bash
set -euo pipefail

# Val-only from 1s fulltrain checkpoint; logs action Top-5 micro acc.
# Uses --export=NONE (not ALL) so stale shell env cannot flip AR10s/direct_rope mode.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm"

# Hardcode tag/config — do NOT inherit LORA_TAG/CONFIG_PATH from the submitting shell.
TAG="hdepic-singleprobe-vitl-fp32-bs8-noac-10ep-w4"
CONFIG_PATH="${PROJECT_ROOT}/configs/generated/hdepic_single_probe_encoder_lora_gaze_pose_matrix_1s_val_top5.yaml"
CHECKPOINT="${PROJECT_ROOT}/checkpoints/vitl.pt"

# Explicit 1s / concat_ar mode: clear every env knob that selects AR10s or direct_rope.
export_csv="NONE"
export_csv+=",PROJECT_ROOT=${PROJECT_ROOT}"
export_csv+=",LORA_TAG=${TAG}"
export_csv+=",CONFIG_PATH=${CONFIG_PATH}"
export_csv+=",CHECKPOINT=${CHECKPOINT}"
export_csv+=",BACKBONE=vitl"
export_csv+=",EVAL_RESOLUTION=256"
export_csv+=",DEBUG_SUBSET_PATH="
export_csv+=",EVAL_MAX_TRAIN_ITERS=0"
export_csv+=",EVAL_NUM_EPOCHS=1"
export_csv+=",EVAL_VAL_EVERY=1"
export_csv+=",RESUME_CHECKPOINT=1"
export_csv+=",VAL_ONLY=1"
export_csv+=",EVAL_SINGLE_PROBE=1"
export_csv+=",LORA_PROBE_TRAIN_MODE=full"
export_csv+=",EVAL_LR=0.0001"
export_csv+=",EVAL_BATCH_SIZE=8"
export_csv+=",EVAL_GRAD_CLIP=1.0"
export_csv+=",EVAL_WARMUP_EPOCHS=0"
export_csv+=",ENCODER_LORA_RANK=8"
export_csv+=",ENCODER_LORA_ALPHA=16.0"
export_csv+=",ENCODER_LORA_LAST_N_BLOCKS=0"
export_csv+=",ENCODER_LORA_LR_MULT=0.5"
export_csv+=",ENCODER_LORA_TARGET_SUFFIXES=attn.qkv|attn.proj"
export_csv+=",ENCODER_LORA_ACTIVATION_CHECKPOINTING=0"
export_csv+=",BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=0"
export_csv+=",LORA_PRETRAINED_PROBE="
export_csv+=",EVAL_USE_BFLOAT16=0"
export_csv+=",EVAL_NUM_WORKERS=2"
export_csv+=",EVAL_VAL_NUM_WORKERS=2"
export_csv+=",GAZE_MODE=binary_input_adapter_gaze_pose_matrix"
# --- mode guards (must stay off / empty for 1s concat_ar) ---
export_csv+=",HDEPIC_AR10S_SLIDING_GAZE_POSE=0"
export_csv+=",AR10S_SLIDING_PROBE=0"
export_csv+=",HDEPIC_DIRECT_ROPE_10S=0"
export_csv+=",EVAL_ANTICIPATION_SEC="
export_csv+=",EVAL_TRAIN_ANTICIPATION_SEC_MIN="
export_csv+=",EVAL_TRAIN_ANTICIPATION_SEC_MAX="
export_csv+=",MODEL_RETURN_MODE="
export_csv+=",MODEL_MAX_ROLLOUT_STEPS="
export_csv+=",HDEPIC_AR_STEPS="
export_csv+=",MODEL_NUM_STEPS="
export_csv+=",MODEL_ROPE_SCALE_MODE="

echo "[submit-1s-val-top5] tag=${TAG}"
echo "[submit-1s-val-top5] config=${CONFIG_PATH}"
echo "[submit-1s-val-top5] mode=1s concat_ar (export=NONE, AR10s/direct_rope cleared)"
sbatch --time=02:00:00 --export="${export_csv}" "${RUN_SCRIPT}"
