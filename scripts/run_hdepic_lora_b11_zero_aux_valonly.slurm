#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b11_zero_aux_val
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240GB
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_b11_zero_aux_val_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_b11_zero_aux_val_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Val-only controls on the trained B11 checkpoint (no full re-train).
# Compares normal aux maps vs zero gaze / zero pose / zero both at validation time.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
TAG="hdepic-lora-binary-gaze-pose-matrix-5ep-bs2-w10-m768-p01"
CKPT="${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation/action_anticipation_frozen/${TAG}/binary_input_adapter_latest.pt"

export SOURCE_YAML="${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_gaze_pose_matrix_train.yaml"
export LORA_VAL_METRIC_SCOPE="${LORA_VAL_METRIC_SCOPE:-filtered}"
export ANTICIPATION_SEC="${ANTICIPATION_SEC:-1.0}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-2}"
export VALDUMP_PREDICTION_DUMP="${VALDUMP_PREDICTION_DUMP:-0}"
export BINARY_INPUT_ADAPTER_LOAD_CHECKPOINT="${CKPT}"

run_val() {
  local label="$1"
  local zero_gaze="$2"
  local zero_pose="$3"
  export VAL_TAG="valdump-b11-matrix-${label}-filtered-1s"
  export BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP="${zero_gaze}"
  export POSE_MAP_FORCE_ZERO="${zero_pose}"
  echo ""
  echo "========== B11 val-only: ${label} (zero_gaze=${zero_gaze} zero_pose=${zero_pose}) =========="
  bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_valonly_dump.slurm"
}

run_val "normal" 0 0
run_val "zero-both" 1 1
run_val "zero-gaze" 1 0
run_val "zero-pose" 0 1
