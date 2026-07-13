#!/bin/bash
set -euo pipefail

# Full train: JEPA_ARVR-style 10s sliding AR, **probe-only** (no gaze/pose, no enc LoRA).
# For 5ch gaze/pose + enc_lora @ 10s, use submit_b11_singleprobe_ar10s_sliding_gaze_pose_fulltrain.sh

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_ar10s_sliding_probe.slurm"
TAG="${LORA_TAG:-hdepic-ar10s-sliding-probe-full-10ep-w2}"
CONFIG_PATH="${PROJECT_ROOT}/configs/generated/hdepic_ar10s_sliding_probe_fulltrain.yaml"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},LORA_TAG=${TAG},CONFIG_PATH=${CONFIG_PATH},EVAL_MAX_TRAIN_ITERS=0,EVAL_NUM_EPOCHS=10,EVAL_VAL_EVERY=2,RESUME_CHECKPOINT=0,EVAL_USE_BFLOAT16=0,EVAL_NUM_WORKERS=2,EVAL_VAL_NUM_WORKERS=2"

echo "[submit-ar10s-sliding-fulltrain] tag=${TAG} val_every=2 workers=2"
sbatch --export="${export_csv}" "${RUN_SCRIPT}"
