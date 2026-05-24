#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_biadapter
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=640GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_binary_input_adapter_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_lora_binary_input_adapter_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=binary_input_adapter
export LORA_TAG="${LORA_TAG:-hdepic-lora-binary-input-adapter-r64-h8-dlmap-multihead-bs2-w12}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_input_adapter_radius64.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"
export RESUME_CHECKPOINT=0
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-5}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"

export GAZE_BINARY_RADIUS_PX="${GAZE_BINARY_RADIUS_PX:-64}"
export BINARY_INPUT_ADAPTER_HIDDEN_DIM="${BINARY_INPUT_ADAPTER_HIDDEN_DIM:-8}"
export BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL="${BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL:-1}"
export BINARY_INPUT_ADAPTER_SCALE="${BINARY_INPUT_ADAPTER_SCALE:-1.0}"
export BINARY_INPUT_ADAPTER_LR_MULT="${BINARY_INPUT_ADAPTER_LR_MULT:-0.05}"
export BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING="${BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING:-0}"
export BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG="${BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG:-1}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
