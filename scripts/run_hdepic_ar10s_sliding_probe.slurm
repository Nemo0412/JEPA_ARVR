#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_ar10s_sliding_probe
#SBATCH --account=your_slurm_account
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_ar10s_sliding_probe_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_ar10s_sliding_probe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# JEPA_ARVR train_hdepic_anticipation10s-aligned path:
#   frozen encoder + predictor, sliding-window AR (return_mode=final_window),
#   probe-only training on 10s-fixed anticipation. No gaze map / encoder LoRA.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export PROJECT_ROOT
export GAZE_MODE=none
export ENCODER_LORA_ENABLED=0
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
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
export EVAL_VAL_NUM_WORKERS="${EVAL_VAL_NUM_WORKERS:-2}"
export EVAL_MAX_TRAIN_ITERS="${EVAL_MAX_TRAIN_ITERS:-0}"
export EVAL_VAL_EVERY="${EVAL_VAL_EVERY:-2}"
export EVAL_USE_BFLOAT16="${EVAL_USE_BFLOAT16:-0}"
export LORA_PRETRAINED_PROBE="${LORA_PRETRAINED_PROBE-}"
export LORA_TAG="${LORA_TAG:-hdepic-ar10s-sliding-probe-full-10ep-w2}"
export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/generated/hdepic_ar10s_sliding_probe_fulltrain.yaml}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-0}"
export AR10S_SLIDING_PROBE=1

cat <<'EOF'
[ar10s-sliding] === PROBE INPUT MODE (read this first) ===
[ar10s-sliding] return_mode=final_window
[ar10s-sliding]   probe input = last N-token sliding window AFTER 10s AR rollout
[ar10s-sliding]   (mostly predicted latents; original encoder tokens slid out)
[ar10s-sliding] NOT legacy observed_plus_target (encoder full + final target chunk)
[ar10s-sliding] NOT observed_plus_rollout (encoder full + all rollout chunks)
[ar10s-sliding] NOT upstream concat_ar (encoder recent + single predicted concat)
[ar10s-sliding] Aligned with JEPA_ARVR anticipate_features(), not our prior ar_valonly yaml
[ar10s-sliding] ============================================
EOF
echo "[ar10s-sliding] tag=${LORA_TAG} anticipation=10s probe_only=1 enc_lora=0 fp32=${EVAL_USE_BFLOAT16}"
echo "[ar10s-sliding] batch=${EVAL_BATCH_SIZE} workers=${EVAL_NUM_WORKERS} val_workers=${EVAL_VAL_NUM_WORKERS} lr=${EVAL_LR} epochs=${EVAL_NUM_EPOCHS} val_every=${EVAL_VAL_EVERY} mem=128GB time=48h"
echo "[ar10s-sliding] expect ~41 predictor steps/sample @10s (horizon_chunks=40 + 1)"

bash "${PROJECT_ROOT}/scripts/run_hdepic_lora_probe.slurm"
