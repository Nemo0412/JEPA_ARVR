#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__dataloader_ddp_pressure_h100
#SBATCH --account=your_slurm_account
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=640GB
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_dataloader_full_h100_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_dataloader_full_h100_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Full-data D2 pressure probe for reproducing DataLoader worker failures on
# H100 under the same two-rank DDP shape as the clean conservative run.
# It intentionally disables DEBUG_SUBSET and raises worker/thread pressure.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export DEBUG_SUBSET_PATH=""
export GAZE_MODE="${GAZE_MODE:-binary_input_adapter}"
export PREDICTION_HORIZONS="${PREDICTION_HORIZONS:-1}"
export LABEL_HORIZONS="${LABEL_HORIZONS:-10}"
export PAST_WINDOW_CURRICULUM="${PAST_WINDOW_CURRICULUM:-1}"

export EVAL_GPUS="${EVAL_GPUS:-2}"
export EVAL_DEBUGMODE="${EVAL_DEBUGMODE:-auto}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-4}"
export EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-0}"
export EVAL_MAX_LR="${EVAL_MAX_LR:-0.001}"
export EVAL_CLIP_BALANCED="${EVAL_CLIP_BALANCED:-1}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"

export BINARY_INPUT_ADAPTER_AUG_AWARE="${BINARY_INPUT_ADAPTER_AUG_AWARE:-1}"
export BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG="${BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG:-0}"
export BINARY_INPUT_ADAPTER_LR_MULT="${BINARY_INPUT_ADAPTER_LR_MULT:-0.01}"
export BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP="${BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP:-1.0}"

export GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:--1}"

tag_parts=(
  "hdepic-lora-binary-map-10s-FULL-D2ddp-pressure-h100"
  "g${EVAL_GPUS}"
  "bs${EVAL_BATCH_SIZE}"
  "w${EVAL_NUM_WORKERS}"
  "vw${EVAL_VAL_NUM_WORKERS}"
  "cache${GAZE_DECODER_CACHE_READER}"
  "th${GAZE_DECODER_NUM_THREADS}"
  "aug${BINARY_INPUT_ADAPTER_AUG_AWARE}"
)
export LORA_TAG_PREFIX="${LORA_TAG_PREFIX:-$(IFS=-; echo "${tag_parts[*]}")}"
export GAZE_DATALOADER_TRACE_DIR="${GAZE_DATALOADER_TRACE_DIR:-${PROJECT_ROOT}/outputs/dataloader_traces/${LORA_TAG_PREFIX}-p1s-l10s-train-curriculum}"
export GAZE_DATALOADER_TRACE_EVENTS="${GAZE_DATALOADER_TRACE_EVENTS:-1}"

export PERF_MONITOR="${PERF_MONITOR:-1}"
export PERF_INTERVAL="${PERF_INTERVAL:-15}"

echo "[d2-full-h100] DEBUG_SUBSET_PATH=${DEBUG_SUBSET_PATH:-}"
echo "[d2-full-h100] LORA_TAG_PREFIX=${LORA_TAG_PREFIX}"
echo "[d2-full-h100] EVAL_GPUS=${EVAL_GPUS} batch=${EVAL_BATCH_SIZE} train_workers=${EVAL_NUM_WORKERS} val_workers=${EVAL_VAL_NUM_WORKERS}"
echo "[d2-full-h100] cache=${GAZE_DECODER_CACHE_READER} decoder_threads=${GAZE_DECODER_NUM_THREADS} aug_aware=${BINARY_INPUT_ADAPTER_AUG_AWARE}"
echo "[d2-full-h100] trace=${GAZE_DATALOADER_TRACE_DIR}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_past_window_train.slurm"
