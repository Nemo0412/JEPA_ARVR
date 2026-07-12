#!/bin/bash
set -euo pipefail

# Smoke test for tri-modal projected cross-attention fusion (video + gaze + IMU proxy).
# Uses legacy split with train_only class space (same constraint as other legacy runs).

PROJECT_ROOT="${PROJECT_ROOT:-/home/ll5914/Jepa_yifan/JEPA_ARVR}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_tri_modal_fusion_train.slurm"
TAG="${LORA_TAG:-hdepic-tri-modal-fusion-1s-smoke-vitl-fp32-bs4}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_tri_modal_fusion_1s_smoke.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitl.pt}"

export_csv="ALL"
export_csv+=",PROJECT_ROOT=${PROJECT_ROOT}"
export_csv+=",LORA_TAG=${TAG}"
export_csv+=",CONFIG_PATH=${CONFIG_PATH}"
export_csv+=",CHECKPOINT=${CHECKPOINT}"
export_csv+=",BACKBONE=vitl"
export_csv+=",EVAL_RESOLUTION=256"
export_csv+=",EVAL_NUM_EPOCHS=1"
export_csv+=",EVAL_MAX_TRAIN_ITERS=30"
export_csv+=",EVAL_BATCH_SIZE=4"
export_csv+=",EVAL_USE_BFLOAT16=0"
export_csv+=",EVAL_SINGLE_PROBE=1"
export_csv+=",LORA_PROBE_TRAIN_MODE=full"
export_csv+=",EVAL_LR=0.0001"
export_csv+=",EVAL_GRAD_CLIP=1.0"
export_csv+=",EVAL_WARMUP_EPOCHS=2"
export_csv+=",LORA_CLASS_SPACE=train_only"
export_csv+=",LORA_TEMPORAL_SAMPLING=legacy"
export_csv+=",ENCODER_LORA_ENABLED=1"
export_csv+=",ENCODER_LORA_RANK=8"
export_csv+=",ENCODER_LORA_ALPHA=16.0"
export_csv+=",ENCODER_LORA_LAST_N_BLOCKS=0"
export_csv+=",ENCODER_LORA_LR_MULT=0.5"
export_csv+=",ENCODER_LORA_TARGET_SUFFIXES=attn.qkv|attn.proj"
export_csv+=",ENCODER_LORA_ACTIVATION_CHECKPOINTING=0"
export_csv+=",TRI_MODAL_USE_GAZE_BRANCH=1"
export_csv+=",TRI_MODAL_USE_IMU_BRANCH=1"
export_csv+=",TRI_MODAL_FUSION_NUM_HEADS=4"
export_csv+=",TRI_MODAL_FUSION_NUM_LAYERS=1"
export_csv+=",TRI_MODAL_USE_GATED_RESIDUAL=1"
export_csv+=",RESUME_CHECKPOINT=0"

echo "[submit-tri-modal-smoke] tag=${TAG}"
echo "[submit-tri-modal-smoke] GAZE_MODE=projected_tri_modal_cross_attention"

if command -v sbatch >/dev/null 2>&1; then
  sbatch --time=4:00:00 --export="${export_csv}" "${RUN_SCRIPT}"
else
  echo "[submit-tri-modal-smoke] sbatch not found; running locally"
  eval "export ${export_csv#ALL,}"
  bash "${RUN_SCRIPT}"
fi
