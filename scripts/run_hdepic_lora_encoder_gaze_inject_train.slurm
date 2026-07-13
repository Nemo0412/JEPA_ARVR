#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_encoder_gaze_inject
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=640GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_encoder_gaze_inject_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_encoder_gaze_inject_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# B8: encoder-output gaze inject (zero-init adapter).
# RNN-encoded gaze tokens are mixed into the frozen V-JEPA token stream
# between encoder output and predictor input via a zero-initialized
# cross-attention residual. Observed-window classifier path (x_accumulate)
# stays unmodified, so only the predicted future tokens see gaze influence.
# Same-window gaze by default (history_sec=0) to isolate the architectural
# axis from B7's long-history experiments.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=encoder_output_inject
export GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
export GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
export GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/Videos}"

# Gaze RNN encoder shape (matches B2 same-window RNN default).
export GAZE_RNN_HIDDEN_DIM="${GAZE_RNN_HIDDEN_DIM:-256}"
export GAZE_RNN_NUM_LAYERS="${GAZE_RNN_NUM_LAYERS:-2}"
export GAZE_RNN_NUM_TOKENS="${GAZE_RNN_NUM_TOKENS:-64}"
export GAZE_RNN_DROPOUT="${GAZE_RNN_DROPOUT:-0.1}"
export GAZE_RNN_BIDIR="${GAZE_RNN_BIDIR:-1}"
export GAZE_RNN_TRAJ_LEN="${GAZE_RNN_TRAJ_LEN:-1024}"
# B8 is gaze-only: no video-token conditioning at the gaze encoder, since the
# gaze influence is applied later via the encoder-output cross-attn adapter.
export GAZE_RNN_USE_VIDEO_TOKEN=0
export GAZE_RNN_VIDEO_FUSION=nearest_concat
export GAZE_RNN_HISTORY_SEC="${GAZE_RNN_HISTORY_SEC:-0}"

# Encoder-output cross-attention adapter (B8-specific).
export ENCODER_GAZE_ADAPTER_NUM_HEADS="${ENCODER_GAZE_ADAPTER_NUM_HEADS:-4}"
export ENCODER_GAZE_ADAPTER_DROPOUT="${ENCODER_GAZE_ADAPTER_DROPOUT:-0.0}"
export ENCODER_GAZE_ADAPTER_LR_MULT="${ENCODER_GAZE_ADAPTER_LR_MULT:-0.05}"
export ENCODER_GAZE_ADAPTER_WEIGHT_DECAY="${ENCODER_GAZE_ADAPTER_WEIGHT_DECAY:-0.0001}"

# Match B2/B5 head LR cap; safe default after 2026-05-24 short diagnostics.
export EVAL_MAX_LR="${EVAL_MAX_LR:-0.002}"
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-5}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
export LORA_TAG="${LORA_TAG:-hdepic-lora-encoder-gaze-inject}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_encoder_gaze_inject.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_lora_action_anticipation}"

# Optional branch-local passthrough for rare one-off overrides, e.g.
#   ENCODER_GAZE_EXTRA_EXPORTS="ENCODER_GAZE_ADAPTER_LR_MULT=0.1"
# Keep stable branch parameters above; use this only for temporary probes.
if [[ -n "${ENCODER_GAZE_EXTRA_EXPORTS:-}" ]]; then
  for kv in ${ENCODER_GAZE_EXTRA_EXPORTS}; do
    export "${kv}"
  done
fi

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
