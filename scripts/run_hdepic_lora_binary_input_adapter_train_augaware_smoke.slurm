#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_biadapter_augaware_smoke
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240GB
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_biadapter_augaware_smoke_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_biadapter_augaware_smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Smoke test for the aug-aware binary input adapter path on an A100. Runs at
# most ~200 train iters and one short val pass to verify:
#   - Joint RGB+binary aug pipeline produces aligned outputs in dataloader
#   - Forward and backward pass complete without shape mismatch or NaN
#   - The adapter optimizer/LR split applies
# After smoke passes, the full 5-epoch run moves back to H100 with a new tag.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=binary_input_adapter
export LORA_TAG="${LORA_TAG:-hdepic-lora-binary-input-adapter-augaware-smoke-a100}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_input_adapter_augaware_smoke.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
export RESUME_CHECKPOINT=0
export VAL_ONLY=0
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-1}"
export EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-200}"
export EVAL_MAX_LR="${EVAL_MAX_LR:-0.002}"
export EVAL_LR_SCALE="${EVAL_LR_SCALE:-1}"

# Aug-aware binary map: joint train aug on RGB + binary, no disable_train_aug.
export BINARY_INPUT_ADAPTER_AUG_AWARE=1

export GAZE_BINARY_RADIUS_PX="${GAZE_BINARY_RADIUS_PX:-64}"
export BINARY_INPUT_ADAPTER_HIDDEN_DIM="${BINARY_INPUT_ADAPTER_HIDDEN_DIM:-8}"
export BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL="${BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL:-1}"
export BINARY_INPUT_ADAPTER_SCALE="${BINARY_INPUT_ADAPTER_SCALE:-1.0}"
export BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP="${BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP:-1.0}"
export BINARY_INPUT_ADAPTER_LR_MULT="${BINARY_INPUT_ADAPTER_LR_MULT:-0.05}"
export BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING="${BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING:-0}"
export BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP=0
export BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT=""
export GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:-1}"
export GAZE_DATALOADER_TRACE_DIR="${GAZE_DATALOADER_TRACE_DIR:-${PROJECT_ROOT}/outputs/dataloader_traces/${LORA_TAG}}"
export GAZE_DATALOADER_TRACE_EVENTS="${GAZE_DATALOADER_TRACE_EVENTS:-1}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
