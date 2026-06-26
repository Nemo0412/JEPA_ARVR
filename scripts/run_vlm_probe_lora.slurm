#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__vlm_probe_lora
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/vlm_probe_lora_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/vlm_probe_lora_%j.err

# LoRA + classification-head probe fine-tuning of LLaVA-OneVision / Llama-3.2-Vision
# for HD-EPIC 1s action anticipation: runs app.hdepic_lora_action_anticipation.
# train_vlm_probe_lora. Mirrors the PhD's Qwen2.5-VL-3B probe
# (refer_repo/JEPA_ARVR/qwen/train_hdepic_qwen_probe.py); supersedes
# run_vlm_lora_sft.slurm's text-generation SFT approach as the standard
# training-based baseline (per 2026-06-22 decision), kept for reference, not deleted.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

BACKEND="${BACKEND:?BACKEND is required (llama32vision, llava_onevision, or qwen25vl)}"
MODEL_ID="${MODEL_ID:-}"
TRAIN_CSV="${TRAIN_CSV:?TRAIN_CSV is required}"
VAL_CSV="${VAL_CSV:?VAL_CSV is required}"
TEST_CSV="${TEST_CSV:-}"
VIDEO_ROOT="${VIDEO_ROOT:?VIDEO_ROOT is required}"
VERB_CLASSES_CSV="${VERB_CLASSES_CSV:?VERB_CLASSES_CSV is required}"
NOUN_CLASSES_CSV="${NOUN_CLASSES_CSV:?NOUN_CLASSES_CSV is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
NUM_FRAMES="${NUM_FRAMES:-32}"
PROBE_NUM_FRAMES="${PROBE_NUM_FRAMES:-8}"
TARGET_FPS="${TARGET_FPS:-8.0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
LOG_EVERY="${LOG_EVERY:-20}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
PRUNE_KEEP_RATIO="${PRUNE_KEEP_RATIO:-1.0}"
PRUNE_CHUNK_SIZE="${PRUNE_CHUNK_SIZE:-0}"
QWEN_FRAME_SIZE="${QWEN_FRAME_SIZE:-0}"
PREPROC_CACHE_DIR="${PREPROC_CACHE_DIR:-}"
RESUME_FROM="${RESUME_FROM:-}"
LOCAL_FILES_ONLY_FLAG=""
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_FILES_ONLY_FLAG="--local-files-only"
fi

export HF_HOME="${HF_HOME:-/path/to/.huggingface}"
export PROJECT_ROOT ENV_NAME BACKEND MODEL_ID TRAIN_CSV VAL_CSV TEST_CSV VIDEO_ROOT
export VERB_CLASSES_CSV NOUN_CLASSES_CSV OUTPUT_DIR NUM_FRAMES PROBE_NUM_FRAMES TARGET_FPS
export MAX_TRAIN_SAMPLES NUM_EPOCHS LR WEIGHT_DECAY WARMUP_EPOCHS LORA_RANK LORA_ALPHA
export BATCH_SIZE GRAD_ACCUM_STEPS LOG_EVERY TORCH_DTYPE NUM_WORKERS SEED LOCAL_FILES_ONLY_FLAG
export PRUNE_KEEP_RATIO PRUNE_CHUNK_SIZE QWEN_FRAME_SIZE PREPROC_CACHE_DIR RESUME_FROM

echo "[probe_lora] backend=${BACKEND} model_id=${MODEL_ID:-<default>} train_csv=${TRAIN_CSV}"
echo "[probe_lora] num_frames=${NUM_FRAMES} probe_num_frames=${PROBE_NUM_FRAMES} target_fps=${TARGET_FPS}"
echo "[probe_lora] epochs=${NUM_EPOCHS} lr=${LR} lora_rank=${LORA_RANK} lora_alpha=${LORA_ALPHA} batch_size=${BATCH_SIZE} grad_accum_steps=${GRAD_ACCUM_STEPS}"
echo "[probe_lora] output_dir=${OUTPUT_DIR}"

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

mkdir -p "${OUTPUT_DIR}"

python -m app.hdepic_lora_action_anticipation.train_vlm_probe_lora \
  --backend "${BACKEND}" \
  ${MODEL_ID:+--model-id "${MODEL_ID}"} \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  ${TEST_CSV:+--test-csv "${TEST_CSV}"} \
  --video-root "${VIDEO_ROOT}" \
  --verb-classes-csv "${VERB_CLASSES_CSV}" \
  --noun-classes-csv "${NOUN_CLASSES_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-frames "${NUM_FRAMES}" \
  --probe-num-frames "${PROBE_NUM_FRAMES}" \
  --target-fps "${TARGET_FPS}" \
  --prune-keep-ratio "${PRUNE_KEEP_RATIO}" \
  --prune-chunk-size "${PRUNE_CHUNK_SIZE}" \
  --qwen-frame-size "${QWEN_FRAME_SIZE}" \
  ${PREPROC_CACHE_DIR:+--preproc-cache-dir "${PREPROC_CACHE_DIR}"} \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --num-epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --warmup-epochs "${WARMUP_EPOCHS}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --log-every "${LOG_EVERY}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  ${RESUME_FROM:+--resume-from "${RESUME_FROM}"} \
  ${LOCAL_FILES_ONLY_FLAG}
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
