#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__vlm_probe_eval
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/vlm_probe_eval_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/vlm_probe_eval_%j.err

# Re-evaluates a saved train_vlm_probe_lora.py checkpoint with eval_probe_checkpoint.py
# (adds instance-level top-5 accuracy on top of the original top-3/class-mean-R@5 metrics).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
EVAL_CSV="${EVAL_CSV:?EVAL_CSV is required}"
TRAIN_CSV="${TRAIN_CSV:?TRAIN_CSV is required}"
VAL_CSV="${VAL_CSV:?VAL_CSV is required}"
TEST_CSV="${TEST_CSV:-}"
VIDEO_ROOT="${VIDEO_ROOT:?VIDEO_ROOT is required}"
VERB_CLASSES_CSV="${VERB_CLASSES_CSV:?VERB_CLASSES_CSV is required}"
NOUN_CLASSES_CSV="${NOUN_CLASSES_CSV:?NOUN_CLASSES_CSV is required}"
NUM_FRAMES="${NUM_FRAMES:-32}"
PROBE_NUM_FRAMES="${PROBE_NUM_FRAMES:-8}"
TARGET_FPS="${TARGET_FPS:-8.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
PRUNE_KEEP_RATIO="${PRUNE_KEEP_RATIO:-1.0}"
PRUNE_CHUNK_SIZE="${PRUNE_CHUNK_SIZE:-0}"
QWEN_FRAME_SIZE="${QWEN_FRAME_SIZE:-0}"
PREPROC_CACHE_DIR="${PREPROC_CACHE_DIR:-}"
LOCAL_FILES_ONLY_FLAG=""
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_FILES_ONLY_FLAG="--local-files-only"
fi

export HF_HOME="${HF_HOME:-/path/to/.huggingface}"
export PROJECT_ROOT ENV_NAME CHECKPOINT EVAL_CSV TRAIN_CSV VAL_CSV TEST_CSV VIDEO_ROOT
export VERB_CLASSES_CSV NOUN_CLASSES_CSV NUM_FRAMES PROBE_NUM_FRAMES TARGET_FPS
export BATCH_SIZE NUM_WORKERS TORCH_DTYPE PRUNE_KEEP_RATIO PRUNE_CHUNK_SIZE QWEN_FRAME_SIZE
export PREPROC_CACHE_DIR LOCAL_FILES_ONLY_FLAG

echo "[probe_eval] checkpoint=${CHECKPOINT} eval_csv=${EVAL_CSV}"

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

python -m app.hdepic_lora_action_anticipation.eval_probe_checkpoint \
  --checkpoint "${CHECKPOINT}" \
  --eval-csv "${EVAL_CSV}" \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  ${TEST_CSV:+--test-csv "${TEST_CSV}"} \
  --video-root "${VIDEO_ROOT}" \
  --verb-classes-csv "${VERB_CLASSES_CSV}" \
  --noun-classes-csv "${NOUN_CLASSES_CSV}" \
  --num-frames "${NUM_FRAMES}" \
  --probe-num-frames "${PROBE_NUM_FRAMES}" \
  --target-fps "${TARGET_FPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --prune-keep-ratio "${PRUNE_KEEP_RATIO}" \
  --prune-chunk-size "${PRUNE_CHUNK_SIZE}" \
  --qwen-frame-size "${QWEN_FRAME_SIZE}" \
  ${PREPROC_CACHE_DIR:+--preproc-cache-dir "${PREPROC_CACHE_DIR}"} \
  ${LOCAL_FILES_ONLY_FLAG}
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
