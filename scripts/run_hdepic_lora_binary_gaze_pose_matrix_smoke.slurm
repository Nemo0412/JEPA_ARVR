#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_binary_gaze_pose_matrix_smoke
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256GB
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_gaze_pose_matrix_smoke_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_gaze_pose_matrix_smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# B11: binary gaze map + inter-frame pose matrix -> 5ch input adapter (no probe GRU).
# Smoke only: ~300 train iters, 1 epoch, debug subset.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=binary_input_adapter_gaze_pose_matrix
export GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
export GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
export GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/Videos}"

export POSE_SLAM_ROOT="${POSE_SLAM_ROOT:-${GAZE_ROOT}/P01/SLAM/multi}"
export POSE_MAPPING_JSON="${POSE_MAPPING_JSON:-${POSE_SLAM_ROOT}/vrs_to_multi_slam.json}"
export POSE_FEATURE_SET="${POSE_FEATURE_SET:-pose_6d}"
export POSE_QUALITY_MIN="${POSE_QUALITY_MIN:-0.0}"
export POSE_HISTORY_SEC="${POSE_HISTORY_SEC:-0}"
export POSE_INTERFRAME_K_MAX="${POSE_INTERFRAME_K_MAX:-128}"
export POSE_MAP_PATCH_HEIGHT="${POSE_MAP_PATCH_HEIGHT:-128}"
export POSE_MAP_PATCH_WIDTH="${POSE_MAP_PATCH_WIDTH:-9}"

export BINARY_INPUT_ADAPTER_IN_CHANNELS="${BINARY_INPUT_ADAPTER_IN_CHANNELS:-5}"
export BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL="${BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL:-3}"
export BINARY_INPUT_ADAPTER_HIDDEN_DIM="${BINARY_INPUT_ADAPTER_HIDDEN_DIM:-8}"

export EVAL_MAX_LR="${EVAL_MAX_LR:-0.002}"
export EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-300}"
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-0}"
export GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:-1}"

export LORA_TAG="${LORA_TAG:-hdepic-lora-binary-gaze-pose-matrix-smoke-DEBUG_SUBSET}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_gaze_pose_matrix_smoke.yaml}"
export DEBUG_SUBSET_PATH="${DEBUG_SUBSET_PATH:-${PROJECT_ROOT}/configs/debug_subset_p01.json}"

# Optional ablations:
# export BINARY_INPUT_ADAPTER_FORCE_ZERO_MAP=1   # gaze-off / pose-only
# export POSE_MAP_FORCE_ZERO=1                   # zero-pose control

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
