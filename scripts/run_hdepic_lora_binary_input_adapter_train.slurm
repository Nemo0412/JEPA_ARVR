#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_biadapter
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=640GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_input_adapter_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_input_adapter_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=binary_input_adapter
export LORA_TAG="${LORA_TAG:-hdepic-lora-binary-input-adapter-r64-h8-dlmap-multihead-bs2-w12-gazefixed}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_input_adapter_radius64_gazefixed.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
export RESUME_CHECKPOINT=0
export VAL_ONLY=0
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-5}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-4}"
export EVAL_MAX_TRAIN_ITERS=0
# Safety default after the NaN diagnostics. EVAL_LR_SCALE first rescales the
# full LR grid; EVAL_MAX_LR is then applied as a final cap.
export EVAL_MAX_LR="${EVAL_MAX_LR:-0.002}"
export EVAL_LR_SCALE="${EVAL_LR_SCALE:-1}"

export GAZE_BINARY_RADIUS_PX="${GAZE_BINARY_RADIUS_PX:-64}"
export GAZE_BINARY_MAP_TYPE="${GAZE_BINARY_MAP_TYPE:-binary}"
export BINARY_INPUT_ADAPTER_HIDDEN_DIM="${BINARY_INPUT_ADAPTER_HIDDEN_DIM:-8}"
export BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL="${BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL:-1}"
export BINARY_INPUT_ADAPTER_SCALE="${BINARY_INPUT_ADAPTER_SCALE:-1.0}"
export BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP="${BINARY_INPUT_ADAPTER_RESIDUAL_CLAMP:-1.0}"
export BINARY_INPUT_ADAPTER_LR_MULT="${BINARY_INPUT_ADAPTER_LR_MULT:-0.05}"
export BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING="${BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING:-0}"
export BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG="${BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG:-1}"
export BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP=0
export BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT=""

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
