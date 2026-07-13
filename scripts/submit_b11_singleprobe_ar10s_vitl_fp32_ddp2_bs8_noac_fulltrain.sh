#!/bin/bash
set -euo pipefail

# ViT-L @256 fp32 2xH100 DDP fulltrain: 10s AR sliding + gaze/pose + encoder LoRA.
# Per-GPU bs8 (global bs16); workers=2/rank to avoid cgroup OOM (w4 peaked 768GB in smoke).
# Verify .out: vit_encoder_predictor_rollout + train_anticipation=[8,10] + cuda:0 cuda:1.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_singleprobe_ar10s_ddp_fulltrain_h100.slurm"
TAG="${LORA_TAG:-hdepic-singleprobe-ar10s-vitl-ddp2-bs16-noac-10ep-w2-tr8-10}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_singleprobe_ar10s_sliding_gaze_pose_enclora_fulltrain.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/vitl.pt}"
SLURM_TIME="${SLURM_TIME:-96:00:00}"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},LORA_TAG=${TAG},CONFIG_PATH=${CONFIG_PATH},CHECKPOINT=${CHECKPOINT},BACKBONE=vitl,EVAL_RESOLUTION=256,HDEPIC_AR10S_SLIDING_GAZE_POSE=1,EVAL_ANTICIPATION_SEC=10,EVAL_TRAIN_ANTICIPATION_SEC_MIN=8,EVAL_TRAIN_ANTICIPATION_SEC_MAX=10,MODEL_RETURN_MODE=final_window,MODEL_MAX_ROLLOUT_STEPS=512,EVAL_GPUS=2,EVAL_CPUS_PER_TASK=24,DEBUG_SUBSET_PATH=,EVAL_MAX_TRAIN_ITERS=0,EVAL_NUM_EPOCHS=10,EVAL_VAL_EVERY=2,RESUME_CHECKPOINT=0,EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,EVAL_LR=0.0002,EVAL_BATCH_SIZE=8,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv|attn.proj,ENCODER_LORA_ACTIVATION_CHECKPOINTING=0,BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING=0,LORA_PRETRAINED_PROBE=,EVAL_USE_BFLOAT16=0,EVAL_NUM_WORKERS=2,EVAL_VAL_NUM_WORKERS=2"

echo "[submit-vitl-ar10s-ddp-fulltrain] tag=${TAG}"
echo "[submit-vitl-ar10s-ddp-fulltrain] 2xH100 DDP per_gpu_bs=8 global_bs=16 w2/rank fp32 no-AC"
sbatch --time="${SLURM_TIME}" --export="${export_csv}" "${RUN_SCRIPT}"
