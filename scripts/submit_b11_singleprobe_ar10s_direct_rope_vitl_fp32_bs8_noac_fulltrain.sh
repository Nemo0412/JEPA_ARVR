#!/bin/bash
set -euo pipefail

# ViT-L @256 fp32 fulltrain: 10s direct_rope (single predictor fwd) + gaze/pose + encoder LoRA.
# H100 bs8 w4 no-AC. Verify .out: vit_encoder_predictor_direct_rope + rope_scale_mode=ntk_temporal.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm"
WORKERS="${EVAL_NUM_WORKERS:-4}"
VAL_WORKERS="${EVAL_VAL_NUM_WORKERS:-${WORKERS}}"
TAG="${LORA_TAG:-hdepic-singleprobe-ar10s-direct-rope-vitl-fp32-bs8-noac-10ep-w${WORKERS}-tr8-10}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_singleprobe_ar10s_direct_rope_gaze_pose_enclora_fulltrain.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitl.pt}"
SLURM_TIME="${SLURM_TIME:-96:00:00}"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},LORA_TAG=${TAG},CONFIG_PATH=${CONFIG_PATH},CHECKPOINT=${CHECKPOINT},BACKBONE=vitl,EVAL_RESOLUTION=256,HDEPIC_DIRECT_ROPE_10S=1,EVAL_ANTICIPATION_SEC=10,EVAL_TRAIN_ANTICIPATION_SEC_MIN=8,EVAL_TRAIN_ANTICIPATION_SEC_MAX=10,MODEL_ROPE_SCALE_MODE=ntk_temporal,DEBUG_SUBSET_PATH=,EVAL_MAX_TRAIN_ITERS=0,EVAL_NUM_EPOCHS=10,EVAL_VAL_EVERY=2,RESUME_CHECKPOINT=0,EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,EVAL_LR=0.0001,EVAL_BATCH_SIZE=8,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv|attn.proj,ENCODER_LORA_ACTIVATION_CHECKPOINTING=0,BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=0,LORA_PRETRAINED_PROBE=,EVAL_USE_BFLOAT16=0,EVAL_NUM_WORKERS=${WORKERS},EVAL_VAL_NUM_WORKERS=${VAL_WORKERS}"

echo "[submit-vitl-ar10s-direct-rope-fulltrain] tag=${TAG}"
echo "[submit-vitl-ar10s-direct-rope-fulltrain] direct_rope ntk @10s train=[8,10]s fp32 bs8 w${WORKERS}"
sbatch --time="${SLURM_TIME}" --export="${export_csv}" "${RUN_SCRIPT}"
