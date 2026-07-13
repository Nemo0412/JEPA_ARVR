#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__dataloader_stress_debug
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240GB
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_dataloader_stress_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_dataloader_stress_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# D2 stress probe: intentionally exercises the risky online binary-map
# DataLoader shape on the fixed P01 debug subset. Results are diagnostic only.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export DEBUG_SUBSET_PATH="${DEBUG_SUBSET_PATH:-${PROJECT_ROOT}/configs/debug_subset_p01.json}"
export GAZE_MODE="${GAZE_MODE:-binary_input_adapter}"
export PREDICTION_HORIZONS="${PREDICTION_HORIZONS:-1}"
export LABEL_HORIZONS="${LABEL_HORIZONS:-10}"
export PAST_WINDOW_CURRICULUM="${PAST_WINDOW_CURRICULUM:-1}"

# Default is deliberately closer to the old risky path than to production-safe
# settings. Override these env vars to bisect worker/cache/thread effects.
export EVAL_GPUS="${EVAL_GPUS:-1}"
export EVAL_DEBUGMODE="${EVAL_DEBUGMODE:-auto}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-4}"
export EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-1200}"
export EVAL_MAX_LR="${EVAL_MAX_LR:-0.001}"
export EVAL_CLIP_BALANCED="${EVAL_CLIP_BALANCED:-1}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"

export BINARY_INPUT_ADAPTER_AUG_AWARE="${BINARY_INPUT_ADAPTER_AUG_AWARE:-1}"
export BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG="${BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG:-0}"
export BINARY_INPUT_ADAPTER_LR_MULT="${BINARY_INPUT_ADAPTER_LR_MULT:-0.01}"
export BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP="${BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP:-1.0}"

export GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:--1}"

stress_tag_parts=(
  "hdepic-lora-binary-map-10s-DEBUG_SUBSET-D2stress"
  "g${EVAL_GPUS}"
  "bs${EVAL_BATCH_SIZE}"
  "w${EVAL_NUM_WORKERS}"
  "vw${EVAL_VAL_NUM_WORKERS}"
  "cache${GAZE_DECODER_CACHE_READER}"
  "th${GAZE_DECODER_NUM_THREADS}"
  "aug${BINARY_INPUT_ADAPTER_AUG_AWARE}"
)
export LORA_TAG_PREFIX="${LORA_TAG_PREFIX:-$(IFS=-; echo "${stress_tag_parts[*]}")}"
export GAZE_DATALOADER_TRACE_DIR="${GAZE_DATALOADER_TRACE_DIR:-${PROJECT_ROOT}/outputs/dataloader_traces/${LORA_TAG_PREFIX}-p1s-l10s-train-curriculum}"
export GAZE_DATALOADER_TRACE_EVENTS="${GAZE_DATALOADER_TRACE_EVENTS:-1}"

export PERF_MONITOR="${PERF_MONITOR:-1}"
export PERF_INTERVAL="${PERF_INTERVAL:-15}"

echo "[d2-stress] DEBUG_SUBSET_PATH=${DEBUG_SUBSET_PATH}"
echo "[d2-stress] LORA_TAG_PREFIX=${LORA_TAG_PREFIX}"
echo "[d2-stress] EVAL_GPUS=${EVAL_GPUS} batch=${EVAL_BATCH_SIZE} train_workers=${EVAL_NUM_WORKERS} val_workers=${EVAL_VAL_NUM_WORKERS}"
echo "[d2-stress] cache=${GAZE_DECODER_CACHE_READER} decoder_threads=${GAZE_DECODER_NUM_THREADS} aug_aware=${BINARY_INPUT_ADAPTER_AUG_AWARE}"
echo "[d2-stress] trace=${GAZE_DATALOADER_TRACE_DIR}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_past_window_train.slurm"
