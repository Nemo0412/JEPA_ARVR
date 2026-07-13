#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_rnn_long_gaze
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=640GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_rnn_long_gaze_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_rnn_long_gaze_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Long-history gaze-only RNN branch.
# Video input remains the normal observed clip; gaze history extends earlier
# than the clip and ends at the clip's final frame. This branch intentionally
# disables video-token conditioning so only the final gaze segment overlaps with
# video, while earlier gaze is modeled as gaze-only context.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_RNN_HISTORY_SEC="${GAZE_RNN_HISTORY_SEC:-20}"
export GAZE_RNN_USE_VIDEO_TOKEN=0
export GAZE_RNN_VIDEO_FUSION=nearest_concat
export GAZE_HIDDEN_DUMP="${GAZE_HIDDEN_DUMP:-0}"
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-5}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-}"
export EVAL_MAX_LR="${EVAL_MAX_LR:-0.002}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_rnn_long_gaze.yaml}"
export LORA_TAG="${LORA_TAG:-hdepic-lora-rnn-long-gaze-${GAZE_RNN_HISTORY_SEC}s}"

# Optional branch-local passthrough for rare one-off overrides, e.g.
#   LONG_GAZE_EXTRA_EXPORTS="GAZE_RNN_TRAJ_LEN=2048 GAZE_LR_MULT=0.2"
# Keep stable branch parameters above; use this only for temporary probes.
if [[ -n "${LONG_GAZE_EXTRA_EXPORTS:-}" ]]; then
  for kv in ${LONG_GAZE_EXTRA_EXPORTS}; do
    export "${kv}"
  done
fi

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_rnn_gaze_train.slurm"
