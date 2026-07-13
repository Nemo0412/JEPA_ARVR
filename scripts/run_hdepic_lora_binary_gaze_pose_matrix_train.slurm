#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_lora_binary_gaze_pose_matrix_train
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=768GB
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_gaze_pose_matrix_train_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_lora_binary_gaze_pose_matrix_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# B11 full P01: binary gaze map + inter-frame pose matrix -> 5ch input adapter (no probe GRU).

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
export BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG=1

export EVAL_MAX_LR="${EVAL_MAX_LR:-0.0015}"
export EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-0}"
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-5}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-10}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-2}"
export GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:-2}"

export PERF_MONITOR="${PERF_MONITOR:-1}"
export PERF_INTERVAL="${PERF_INTERVAL:-30}"

export DEBUG_SUBSET_PATH=""
export LORA_TAG="${LORA_TAG:-hdepic-lora-binary-gaze-pose-matrix-5ep-bs2-w10-m768-p01}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_lora_binary_gaze_pose_matrix_train.yaml}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
