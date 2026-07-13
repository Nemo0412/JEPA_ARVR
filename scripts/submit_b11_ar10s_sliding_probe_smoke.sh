#!/bin/bash
set -euo pipefail

# Short smoke: JEPA_ARVR-style 10s sliding-window AR probe (frozen enc+pred).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_hdepic_ar10s_sliding_probe.slurm"
TAG="${LORA_TAG:-hdepic-ar10s-sliding-probe-smoke-i150}"
CONFIG_PATH="${PROJECT_ROOT}/configs/generated/hdepic_ar10s_sliding_probe_smoke.yaml"

export_csv="ALL,PROJECT_ROOT=${PROJECT_ROOT},LORA_TAG=${TAG},CONFIG_PATH=${CONFIG_PATH},EVAL_MAX_TRAIN_ITERS=150,EVAL_NUM_EPOCHS=1,EVAL_VAL_EVERY=1,RESUME_CHECKPOINT=0,EVAL_USE_BFLOAT16=0,EVAL_NUM_WORKERS=2,EVAL_VAL_NUM_WORKERS=2"

echo "[submit-ar10s-sliding-smoke] tag=${TAG} workers=2"
sbatch --export="${export_csv}" "${RUN_SCRIPT}"
