#!/bin/bash
set -euo pipefail

# Full train: current 5ch gaze/pose + encoder LoRA pipeline @ 10s AR sliding window.
# NOT the probe-only AR10s line (see submit_b11_ar10s_sliding_probe_fulltrain.sh).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm"
TAG="${LORA_TAG:-hdepic-singleprobe-ar10s-sliding-fp32-enclora-gaze-pose-h100-bs4-10ep-w2}"
CONFIG_PATH="${PROJECT_ROOT}/configs/generated/hdepic_singleprobe_ar10s_sliding_gaze_pose_enclora_fulltrain.yaml"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},LORA_TAG=${TAG},CONFIG_PATH=${CONFIG_PATH},HDEPIC_AR10S_SLIDING_GAZE_POSE=1,EVAL_ANTICIPATION_SEC=10,EVAL_TRAIN_ANTICIPATION_SEC_MIN=8,EVAL_TRAIN_ANTICIPATION_SEC_MAX=10,MODEL_RETURN_MODE=final_window,MODEL_MAX_ROLLOUT_STEPS=512,DEBUG_SUBSET_PATH=,EVAL_MAX_TRAIN_ITERS=0,EVAL_NUM_EPOCHS=10,EVAL_VAL_EVERY=2,RESUME_CHECKPOINT=0,EVAL_SINGLE_PROBE=1,LORA_PROBE_TRAIN_MODE=full,EVAL_LR=0.0001,EVAL_BATCH_SIZE=4,EVAL_GRAD_CLIP=1.0,EVAL_WARMUP_EPOCHS=2,ENCODER_LORA_RANK=8,ENCODER_LORA_ALPHA=16.0,ENCODER_LORA_LAST_N_BLOCKS=0,ENCODER_LORA_LR_MULT=0.5,ENCODER_LORA_TARGET_SUFFIXES=attn.qkv|attn.proj,LORA_PRETRAINED_PROBE=,EVAL_USE_BFLOAT16=0,EVAL_NUM_WORKERS=2,EVAL_VAL_NUM_WORKERS=2"

echo "[submit-ar10s-gaze-pose-fulltrain] tag=${TAG}"
echo "[submit-ar10s-gaze-pose-fulltrain] val=10s train=[8,10]s rollout/final_window workers=2"
sbatch --export="${export_csv}" "${RUN_SCRIPT}"
