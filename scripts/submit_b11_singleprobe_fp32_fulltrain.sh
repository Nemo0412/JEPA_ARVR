#!/bin/bash
set -euo pipefail

# Full B11 single-probe + encoder LoRA training in fp32 (stable grads on H100).
# Smoke 10742531: workers=2, GPU ~23GB, zero discard. Job 10745336 OOM at workers=10
# (cgroup host RAM hit 768GB during itr=0 encoder backward, not GPU).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm"

export LORA_TAG="${LORA_TAG:-hdepic-singleprobe-fp32-full-enclora-gaze-pose-h100-r8-allblocks-bs4-10ep-w2}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_single_probe_encoder_lora_gaze_pose_matrix_fulltrain.yaml}"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},LORA_TAG=${LORA_TAG},CONFIG_PATH=${CONFIG_PATH},DEBUG_SUBSET_PATH=,EVAL_MAX_TRAIN_ITERS=0,EVAL_NUM_EPOCHS=10,EVAL_VAL_EVERY=2,RESUME_CHECKPOINT=0,EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,EVAL_LR=0.0001,EVAL_BATCH_SIZE=4,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv,attn.proj,LORA_PRETRAINED_PROBE=,EVAL_USE_BFLOAT16=0,EVAL_NUM_WORKERS=2,EVAL_VAL_NUM_WORKERS=2"

echo "[submit-b11-single-fp32-fulltrain] tag=${LORA_TAG} EVAL_USE_BFLOAT16=0 workers=2 val_workers=2 mem=768GB"
sbatch --export="${export_csv}" "${RUN_SCRIPT}"
