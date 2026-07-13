#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_singleprobe_enclora_pose
#SBATCH --account=your_slurm_account
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=768GB
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_singleprobe_enclora_pose_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_singleprobe_enclora_pose_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# Single new full AttentiveClassifier probe + JEPA_ARVR-style Encoder-LoRA +
# binary gaze/pose-map input adapter. No pretrained EK100/HD-EPIC probe is
# loaded; the probe and encoder LoRA are trained together.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=binary_input_adapter_gaze_pose_matrix
export GAZE_ROOT="${GAZE_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze}"
export GAZE_EXTRACT_ROOT="${GAZE_EXTRACT_ROOT:-${PROJECT_ROOT}/outputs/hdepic_gaze_token_gate/extracted}"
export GAZE_SYNC_ROOT="${GAZE_SYNC_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC/Videos}"

# JEPA_ARVR-style probe training hyperparameters.
export LORA_PROBE_TRAIN_MODE="${LORA_PROBE_TRAIN_MODE:-full}"
export EVAL_SINGLE_PROBE="${EVAL_SINGLE_PROBE:-1}"
export EVAL_LR="${EVAL_LR:-0.0001}"
export EVAL_START_LR="${EVAL_START_LR:-0.0}"
export EVAL_FINAL_LR="${EVAL_FINAL_LR:-0.0}"
export EVAL_WARMUP_EPOCHS="${EVAL_WARMUP_EPOCHS:-2}"
export EVAL_WEIGHT_DECAY="${EVAL_WEIGHT_DECAY:-0.0001}"
export EVAL_FINAL_WEIGHT_DECAY="${EVAL_FINAL_WEIGHT_DECAY:-0.0001}"
export EVAL_USE_FOCAL_LOSS="${EVAL_USE_FOCAL_LOSS:-0}"
export EVAL_GRAD_CLIP="${EVAL_GRAD_CLIP:-1.0}"
export EVAL_NUM_EPOCHS="${EVAL_NUM_EPOCHS:-10}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export LORA_PRETRAINED_PROBE="${LORA_PRETRAINED_PROBE-}"

# Full-probe mode ignores these pooler-LoRA knobs, but keep the generated config
# explicit and backwards-compatible.
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16.0}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# JEPA_ARVR encoder adaptation: all encoder blocks, qkv/proj only, LoRA LR=5e-5
# when the probe LR is 1e-4.
export ENCODER_LORA_ENABLED=1
export ENCODER_LORA_RANK="${ENCODER_LORA_RANK:-8}"
export ENCODER_LORA_ALPHA="${ENCODER_LORA_ALPHA:-16.0}"
export ENCODER_LORA_DROPOUT="${ENCODER_LORA_DROPOUT:-0.05}"
export ENCODER_LORA_LAST_N_BLOCKS="${ENCODER_LORA_LAST_N_BLOCKS:-0}"
export ENCODER_LORA_LR_MULT="${ENCODER_LORA_LR_MULT:-0.5}"
export ENCODER_LORA_WEIGHT_DECAY="${ENCODER_LORA_WEIGHT_DECAY:-0.0001}"
export ENCODER_LORA_ACTIVATION_CHECKPOINTING="${ENCODER_LORA_ACTIVATION_CHECKPOINTING:-1}"
export ENCODER_LORA_TARGET_SUFFIXES="${ENCODER_LORA_TARGET_SUFFIXES:-attn.qkv,attn.proj}"

export POSE_SLAM_ROOT="${POSE_SLAM_ROOT:-${GAZE_ROOT}/P01/SLAM/multi}"
export POSE_MAPPING_JSON="${POSE_MAPPING_JSON:-${POSE_SLAM_ROOT}/vrs_to_multi_slam.json}"
export POSE_FEATURE_SET="${POSE_FEATURE_SET:-pose_6d}"
export POSE_QUALITY_MIN="${POSE_QUALITY_MIN:-0.0}"
export POSE_HISTORY_SEC="${POSE_HISTORY_SEC:-0}"
export POSE_INTERFRAME_K_MAX="${POSE_INTERFRAME_K_MAX:-128}"
export POSE_MAP_PATCH_HEIGHT="${POSE_MAP_PATCH_HEIGHT:-128}"
export POSE_MAP_PATCH_WIDTH="${POSE_MAP_PATCH_WIDTH:-9}"
export POSE_MAP_LAYOUT="${POSE_MAP_LAYOUT:-topleft}"
export POSE_MAP_NORMALIZE="${POSE_MAP_NORMALIZE:-none}"

export BINARY_INPUT_ADAPTER_IN_CHANNELS="${BINARY_INPUT_ADAPTER_IN_CHANNELS:-5}"
export BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL="${BINARY_INPUT_ADAPTER_TEMPORAL_KERNEL:-3}"
export BINARY_INPUT_ADAPTER_HIDDEN_DIM="${BINARY_INPUT_ADAPTER_HIDDEN_DIM:-8}"
export BINARY_INPUT_ADAPTER_LR_MULT="${BINARY_INPUT_ADAPTER_LR_MULT:-1.0}"
export BINARY_INPUT_ADAPTER_WEIGHT_DECAY="${BINARY_INPUT_ADAPTER_WEIGHT_DECAY:-0.0001}"
export BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING="${BINARY_INPUT_ADAPTER_ACTIVATION_CHECKPOINTING:-1}"
export BINARY_INPUT_ADAPTER_AUG_AWARE="${BINARY_INPUT_ADAPTER_AUG_AWARE:-0}"
export BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG="${BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG:-1}"

export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-2}"
export EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-0}"
export EVAL_GPUS="${EVAL_GPUS:-1}"
export EVAL_DEBUGMODE="${EVAL_DEBUGMODE:-auto}"
export GAZE_DECODER_CACHE_READER="${GAZE_DECODER_CACHE_READER:-1}"
export GAZE_DECODER_NUM_THREADS="${GAZE_DECODER_NUM_THREADS:-2}"

export PERF_MONITOR="${PERF_MONITOR:-1}"
export PERF_INTERVAL="${PERF_INTERVAL:-30}"

export DEBUG_SUBSET_PATH="${DEBUG_SUBSET_PATH:-}"
export LORA_TAG="${LORA_TAG:-hdepic-singleprobe-new-full-enclora-gaze-pose-h100-r${ENCODER_LORA_RANK}-allblocks-bs4-10ep}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_single_probe_encoder_lora_gaze_pose_matrix.yaml}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"

if [[ "${HDEPIC_DIRECT_ROPE_10S:-0}" == "1" ]]; then
  cat <<'EOF'
[singleprobe-direct-rope] === MODE: 10s direct_rope + gaze/pose + encoder LoRA ===
[singleprobe-direct-rope] model: vit_encoder_predictor_direct_rope (NTK rope, 1x predictor fwd)
[singleprobe-direct-rope] NOT AR rollout (~41 steps) | NOT concat_ar @1s
[singleprobe-direct-rope] + 5ch binary_input_adapter_gaze_pose_matrix + trainable encoder LoRA
[singleprobe-direct-rope] ==========================================================
EOF
elif [[ "${HDEPIC_AR10S_SLIDING_GAZE_POSE:-0}" == "1" ]]; then
  export AR10S_SLIDING_PROBE=1
  export EVAL_ANTICIPATION_SEC="${EVAL_ANTICIPATION_SEC:-10}"
  export EVAL_TRAIN_ANTICIPATION_SEC_MIN="${EVAL_TRAIN_ANTICIPATION_SEC_MIN:-8}"
  export EVAL_TRAIN_ANTICIPATION_SEC_MAX="${EVAL_TRAIN_ANTICIPATION_SEC_MAX:-10}"
  export MODEL_RETURN_MODE="${MODEL_RETURN_MODE:-final_window}"
  export MODEL_MAX_ROLLOUT_STEPS="${MODEL_MAX_ROLLOUT_STEPS:-512}"
  cat <<'EOF'
[singleprobe-ar10s] === MODE: 10s AR sliding + gaze/pose + encoder LoRA ===
[singleprobe-ar10s] model: vit_encoder_predictor_rollout return_mode=final_window
[singleprobe-ar10s] probe input = last-N sliding window after 10s AR (JEPA_ARVR-style)
[singleprobe-ar10s] + 5ch binary_input_adapter_gaze_pose_matrix + trainable encoder LoRA
[singleprobe-ar10s] NOT concat_ar (~1s) | NOT probe-only AR10s line
[singleprobe-ar10s] expect ~41 predictor steps/sample; fp32 proxy backward -> use workers=2
[singleprobe-ar10s] ==========================================================
EOF
fi

echo "[singleprobe] H100 run: batch=${EVAL_BATCH_SIZE} probe_mode=${LORA_PROBE_TRAIN_MODE} lr=${EVAL_LR} warmup=${EVAL_WARMUP_EPOCHS} grad_clip=${EVAL_GRAD_CLIP}"
echo "[singleprobe] pretrained_probe=${LORA_PRETRAINED_PROBE:-<none>}"
echo "[singleprobe] Encoder-LoRA: rank=${ENCODER_LORA_RANK} alpha=${ENCODER_LORA_ALPHA} last=${ENCODER_LORA_LAST_N_BLOCKS} lr_mult=${ENCODER_LORA_LR_MULT} targets=${ENCODER_LORA_TARGET_SUFFIXES}"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
