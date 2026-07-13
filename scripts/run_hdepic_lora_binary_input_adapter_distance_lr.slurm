#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_biadapter_distlr
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=640GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_input_adapter_distlr_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_input_adapter_distlr_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export LORA_TAG="${LORA_TAG:-hdepic-lora-binary-input-adapter-distance-lrmult05-r64-h8-bs2-w12}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_input_adapter_distance_lrmult05.yaml}"
export GAZE_BINARY_MAP_TYPE="${GAZE_BINARY_MAP_TYPE:-distance}"
export BINARY_INPUT_ADAPTER_LR_MULT="${BINARY_INPUT_ADAPTER_LR_MULT:-0.5}"
export BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG="${BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG:-1}"
export BINARY_INPUT_ADAPTER_AUG_AWARE="${BINARY_INPUT_ADAPTER_AUG_AWARE:-0}"
export BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP=0
export RESUME_CHECKPOINT=0
export VAL_ONLY=0

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_binary_input_adapter_train.slurm"
