#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__vlm_lora_sft
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/vlm_lora_sft_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/vlm_lora_sft_%j.err

# LoRA SFT fine-tuning of LLaVA-OneVision's language model for HD-EPIC 1s
# action anticipation: runs app.hdepic_lora_action_anticipation.train_vlm_lora_sft.
# Independent of run_vlm_zeroshot_prompting.slurm (eval-only) and of
# run_hdepic_lora_probe.slurm / vlm_video_encoder.py (encoder-swap baselines).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

MODEL_ID="${MODEL_ID:-}"
TRAIN_CSV="${TRAIN_CSV:?TRAIN_CSV is required}"
VIDEO_ROOT="${VIDEO_ROOT:?VIDEO_ROOT is required}"
VERB_CLASSES_CSV="${VERB_CLASSES_CSV:?VERB_CLASSES_CSV is required}"
NOUN_CLASSES_CSV="${NOUN_CLASSES_CSV:?NOUN_CLASSES_CSV is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
NUM_FRAMES="${NUM_FRAMES:-32}"
TARGET_FPS="${TARGET_FPS:-8.0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-200}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RESUME_ADAPTER_PATH="${RESUME_ADAPTER_PATH:-}"
SEED="${SEED:-0}"
LOCAL_FILES_ONLY_FLAG=""
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_FILES_ONLY_FLAG="--local-files-only"
fi

export HF_HOME="${HF_HOME:-/scratch/yh6416/.huggingface}"
export PROJECT_ROOT ENV_NAME MODEL_ID TRAIN_CSV VIDEO_ROOT VERB_CLASSES_CSV NOUN_CLASSES_CSV OUTPUT_DIR
export NUM_FRAMES TARGET_FPS MAX_TRAIN_SAMPLES NUM_EPOCHS LR LORA_R LORA_ALPHA LORA_DROPOUT
export GRAD_ACCUM_STEPS LOG_EVERY SAVE_EVERY_STEPS TORCH_DTYPE NUM_WORKERS RESUME_ADAPTER_PATH SEED LOCAL_FILES_ONLY_FLAG

echo "[lora_sft] model_id=${MODEL_ID:-<default>} train_csv=${TRAIN_CSV}"
echo "[lora_sft] num_frames=${NUM_FRAMES} target_fps=${TARGET_FPS} max_train_samples=${MAX_TRAIN_SAMPLES} epochs=${NUM_EPOCHS} lr=${LR}"
echo "[lora_sft] lora_r=${LORA_R} lora_alpha=${LORA_ALPHA} grad_accum_steps=${GRAD_ACCUM_STEPS} dtype=${TORCH_DTYPE} num_workers=${NUM_WORKERS}"
echo "[lora_sft] output_dir=${OUTPUT_DIR}"

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

mkdir -p "${OUTPUT_DIR}"

python -m app.hdepic_lora_action_anticipation.train_vlm_lora_sft \
  ${MODEL_ID:+--model-id "${MODEL_ID}"} \
  --train-csv "${TRAIN_CSV}" \
  --video-root "${VIDEO_ROOT}" \
  --verb-classes-csv "${VERB_CLASSES_CSV}" \
  --noun-classes-csv "${NOUN_CLASSES_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-frames "${NUM_FRAMES}" \
  --target-fps "${TARGET_FPS}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --num-epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --log-every "${LOG_EVERY}" \
  --save-every-steps "${SAVE_EVERY_STEPS}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  ${LOCAL_FILES_ONLY_FLAG} \
  ${RESUME_ADAPTER_PATH:+--resume-adapter-path "${RESUME_ADAPTER_PATH}"}
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
