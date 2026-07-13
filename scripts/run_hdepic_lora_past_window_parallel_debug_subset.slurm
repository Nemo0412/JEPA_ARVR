#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__pastwin_2gpu_debug
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=640GB
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_pastwin_2gpu_debug_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_pastwin_2gpu_debug_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Two-GPU resource probe for the P0 multi-GPU TODO. This intentionally uses the
# fixed P01 debug subset and a DEBUG_SUBSET tag; never report these metrics as
# teacher-facing results.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export EVAL_GPUS="${EVAL_GPUS:-2}"
export EVAL_DEBUGMODE="${EVAL_DEBUGMODE:-0}"
export EVAL_CPUS_PER_TASK="${EVAL_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-24}}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-12}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-1}"
export EVAL_MAX_LR="${EVAL_MAX_LR:-0.002}"
export EVAL_CLIP_BALANCED="${EVAL_CLIP_BALANCED:-1}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"

export DEBUG_SUBSET_PATH="${DEBUG_SUBSET_PATH:-${PROJECT_ROOT}/configs/debug_subset_p01.json}"
export PREDICTION_HORIZONS="${PREDICTION_HORIZONS:-1}"
export LABEL_HORIZONS="${LABEL_HORIZONS:-10}"
export PAST_WINDOW_CURRICULUM="${PAST_WINDOW_CURRICULUM:-1}"
export GAZE_MODE="${GAZE_MODE:-binary_input_adapter}"
export LORA_TAG_PREFIX="${LORA_TAG_PREFIX:-hdepic-lora-binary-map-10s-DEBUG_SUBSET-2gpu-w4}"

export BINARY_INPUT_ADAPTER_AUG_AWARE="${BINARY_INPUT_ADAPTER_AUG_AWARE:-0}"
export BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG="${BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG:-1}"
export BINARY_INPUT_ADAPTER_LR_MULT="${BINARY_INPUT_ADAPTER_LR_MULT:-0.05}"
export BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP="${BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP:-1.0}"
export GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:-1}"
export GAZE_DATALOADER_TRACE_DIR="${GAZE_DATALOADER_TRACE_DIR:-${PROJECT_ROOT}/outputs/dataloader_traces/${LORA_TAG_PREFIX}-p1s-l10s-train-curriculum}"
export GAZE_DATALOADER_TRACE_EVENTS="${GAZE_DATALOADER_TRACE_EVENTS:-1}"

export PERF_MONITOR="${PERF_MONITOR:-1}"
export PERF_INTERVAL="${PERF_INTERVAL:-30}"

echo "[parallel-debug] 2-GPU fixed-subset probe"
echo "[parallel-debug] DEBUG_SUBSET_PATH=${DEBUG_SUBSET_PATH}"
echo "[parallel-debug] LORA_TAG_PREFIX=${LORA_TAG_PREFIX}"
echo "[parallel-debug] EVAL_GPUS=${EVAL_GPUS} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} EVAL_NUM_EPOCHS=${EVAL_NUM_EPOCHS}"
echo "[parallel-debug] EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS} EVAL_VAL_NUM_WORKERS=${EVAL_VAL_NUM_WORKERS}"
echo "[parallel-debug] GAZE_DATALOADER_TRACE_DIR=${GAZE_DATALOADER_TRACE_DIR}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_past_window_train.slurm"
