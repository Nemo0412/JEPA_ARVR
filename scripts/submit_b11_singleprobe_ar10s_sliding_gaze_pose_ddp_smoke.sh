#!/bin/bash
set -euo pipefail

# 2-GPU DDP smoke: true 10s AR sliding + 5ch gaze/pose + encoder LoRA.
# Per-GPU batch=8 -> effective global batch=16; workers=4/rank for GPU utilization.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_singleprobe_ar10s_ddp_smoke_h100.slurm"
TAG="${LORA_TAG:-hdepic-ar10s-vitl-ddp2-bs16-w4-smoke-i150}"
CONFIG_PATH="${PROJECT_ROOT}/configs/generated/hdepic_singleprobe_ar10s_vitl_ddp_smoke.yaml"
CHECKPOINT="${PROJECT_ROOT}/checkpoints/vitl.pt"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},LORA_TAG=${TAG},CONFIG_PATH=${CONFIG_PATH},CHECKPOINT=${CHECKPOINT},BACKBONE=vitl,EVAL_RESOLUTION=256,HDEPIC_AR10S_SLIDING_GAZE_POSE=1,EVAL_ANTICIPATION_SEC=10,MODEL_RETURN_MODE=final_window,EVAL_GPUS=2,EVAL_CPUS_PER_TASK=24,EVAL_MAX_TRAIN_ITERS=150,EVAL_NUM_EPOCHS=1,EVAL_VAL_EVERY=1,RESUME_CHECKPOINT=0,EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,EVAL_LR=0.0004,EVAL_BATCH_SIZE=8,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv,attn.proj,LORA_PRETRAINED_PROBE=,EVAL_USE_BFLOAT16=0,EVAL_NUM_WORKERS=4,EVAL_VAL_NUM_WORKERS=4,PERF_MONITOR=1,PERF_INTERVAL=15"

echo "[submit-ar10s-ddp-smoke] tag=${TAG}"
echo "[submit-ar10s-ddp-smoke] ViT-L@256 (PhD-aligned) | 2xH100 DDP | per_gpu_bs=8 global_bs=16 | workers=4/rank | 150 iters | fp32"
sbatch --export="${export_csv}" "${RUN_SCRIPT}"
