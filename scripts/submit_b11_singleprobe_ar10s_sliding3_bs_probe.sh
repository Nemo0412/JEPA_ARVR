#!/bin/bash
set -euo pipefail

# Probe max batch size for 3-step JEPA_ARVR-style sliding AR @10s (NOT 41-step fine rollout).
# Does not affect direct_rope job. Quick fail: EVAL_MAX_TRAIN_ITERS (default 15).
#
# Usage:
#   EVAL_BATCH_SIZE=8 bash scripts/submit_b11_singleprobe_ar10s_sliding3_bs_probe.sh
#   for bs in 8 6 4 2; do EVAL_BATCH_SIZE=$bs bash scripts/submit_b11_singleprobe_ar10s_sliding3_bs_probe.sh; done

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm"
BS="${EVAL_BATCH_SIZE:-8}"
ITERS="${EVAL_MAX_TRAIN_ITERS:-15}"
AR_STEPS="${HDEPIC_AR_STEPS:-3}"
WORKERS="${EVAL_NUM_WORKERS:-2}"
TAG="${LORA_TAG:-hdepic-ar10s-sliding${AR_STEPS}-bs${BS}-probe-i${ITERS}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_singleprobe_ar10s_sliding_gaze_pose_enclora_fulltrain.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitl.pt}"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},LORA_TAG=${TAG},CONFIG_PATH=${CONFIG_PATH},CHECKPOINT=${CHECKPOINT},BACKBONE=vitl,EVAL_RESOLUTION=256,HDEPIC_AR10S_SLIDING_GAZE_POSE=1,HDEPIC_AR_STEPS=${AR_STEPS},MODEL_NUM_STEPS=${AR_STEPS},EVAL_ANTICIPATION_SEC=10,EVAL_TRAIN_ANTICIPATION_SEC_MIN=10,EVAL_TRAIN_ANTICIPATION_SEC_MAX=10,MODEL_RETURN_MODE=final_window,MODEL_MAX_ROLLOUT_STEPS=512,EVAL_GPUS=1,EVAL_MAX_TRAIN_ITERS=${ITERS},EVAL_NUM_EPOCHS=1,EVAL_VAL_EVERY=99,RESUME_CHECKPOINT=0,EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,EVAL_LR=0.0001,EVAL_BATCH_SIZE=${BS},EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=0,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv|attn.proj,ENCODER_LORA_ACTIVATION_CHECKPOINTING=0,BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=0,LORA_PRETRAINED_PROBE=,EVAL_USE_BFLOAT16=0,EVAL_NUM_WORKERS=${WORKERS},EVAL_VAL_NUM_WORKERS=2"

echo "[submit-ar10s-sliding3-probe] bs=${BS} ar_steps=${AR_STEPS} iters=${ITERS} tag=${TAG}"
echo "[submit-ar10s-sliding3-probe] ~3 predictor fwd/sample (NOT 41-step fine rollout)"
sbatch --time=02:00:00 --export="${export_csv}" "${RUN_SCRIPT}"
