#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b11_valonly_filtered
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240GB
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_b11_valonly_filtered_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_b11_valonly_filtered_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Val-only rerun for B11 with legacy filtered+action-head metrics (1s horizon).

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
TAG="hdepic-lora-binary-gaze-pose-matrix-5ep-bs2-w10-m768-p01"

export SOURCE_YAML="${SOURCE_YAML:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_gaze_pose_matrix_train.yaml}"
export VAL_TAG="${VAL_TAG:-valdump-b11-matrix-filtered-actionhead-1s}"
export LORA_VAL_METRIC_SCOPE="${LORA_VAL_METRIC_SCOPE:-filtered}"
export ANTICIPATION_SEC="${ANTICIPATION_SEC:-1.0}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-2}"
export BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT="${BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/action_anticipation_frozen/${TAG}/binary_input_adapter_latest.pt}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_valonly_dump.slurm"
