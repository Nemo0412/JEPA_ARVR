#!/bin/bash
set -euo pipefail

# Submit wrapper for LoRA SFT fine-tuning of LLaVA-OneVision's language model
# (run_vlm_lora_sft.slurm / train_vlm_lora_sft.py).
#
# Usage:
#   bash scripts/submit_vlm_lora_sft.sh
#
# Smoke test (few samples, short time limit):
#   MAX_TRAIN_SAMPLES=20 SAVE_EVERY_STEPS=1 SLURM_TIME=00:30:00 \
#     bash scripts/submit_vlm_lora_sft.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_vlm_lora_sft.slurm"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"

MODEL_ID="${MODEL_ID:-}"
TRAIN_CSV="${TRAIN_CSV:-${SHARED_PROJECT_ROOT}/data/hdepic_vjepa_annotations/p01_fixed/HD_EPIC_train_vjepa.csv}"
VIDEO_ROOT="${VIDEO_ROOT:-${SHARED_PROJECT_ROOT}/data/hdepic_vjepa_videos}"
HDEPIC_ANN_ROOT="${HDEPIC_ANN_ROOT:-${SHARED_PROJECT_ROOT}/data/hd-epic-annotations/narrations-and-action-segments}"
VERB_CLASSES_CSV="${VERB_CLASSES_CSV:-${HDEPIC_ANN_ROOT}/HD_EPIC_verb_classes.csv}"
NOUN_CLASSES_CSV="${NOUN_CLASSES_CSV:-${HDEPIC_ANN_ROOT}/HD_EPIC_noun_classes.csv}"
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
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
RUN_TAG="${RUN_TAG:-llava_onevision_lora_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${SHARED_PROJECT_ROOT}/outputs/vlm_lora_sft/${RUN_TAG}}"

export_csv="ALL"
export_csv+=",PROJECT_ROOT=${PROJECT_ROOT}"
export_csv+=",SHARED_PROJECT_ROOT=${SHARED_PROJECT_ROOT}"
export_csv+=",OVERLAY=${OVERLAY}"
export_csv+=",MODEL_ID=${MODEL_ID}"
export_csv+=",TRAIN_CSV=${TRAIN_CSV}"
export_csv+=",VIDEO_ROOT=${VIDEO_ROOT}"
export_csv+=",VERB_CLASSES_CSV=${VERB_CLASSES_CSV}"
export_csv+=",NOUN_CLASSES_CSV=${NOUN_CLASSES_CSV}"
export_csv+=",OUTPUT_DIR=${OUTPUT_DIR}"
export_csv+=",NUM_FRAMES=${NUM_FRAMES}"
export_csv+=",TARGET_FPS=${TARGET_FPS}"
export_csv+=",MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}"
export_csv+=",NUM_EPOCHS=${NUM_EPOCHS}"
export_csv+=",LR=${LR}"
export_csv+=",LORA_R=${LORA_R}"
export_csv+=",LORA_ALPHA=${LORA_ALPHA}"
export_csv+=",LORA_DROPOUT=${LORA_DROPOUT}"
export_csv+=",GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
export_csv+=",LOG_EVERY=${LOG_EVERY}"
export_csv+=",SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS}"
export_csv+=",TORCH_DTYPE=${TORCH_DTYPE}"
export_csv+=",NUM_WORKERS=${NUM_WORKERS}"
export_csv+=",RESUME_ADAPTER_PATH=${RESUME_ADAPTER_PATH}"
export_csv+=",SEED=${SEED}"
export_csv+=",LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}"
export_csv+=",HF_HOME=${HF_HOME:-/scratch/yh6416/.huggingface}"

PARTITION="${SLURM_PARTITION:-h100_tandon}"
GRES="${SLURM_GRES:-gpu:h100:1}"
MEM="${SLURM_MEM:-128GB}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

echo "[submit-vlm-lora-sft] train_csv=${TRAIN_CSV} max_train_samples=${MAX_TRAIN_SAMPLES} epochs=${NUM_EPOCHS}"
echo "[submit-vlm-lora-sft] output_dir=${OUTPUT_DIR}"
echo "[submit-vlm-lora-sft] partition=${PARTITION} gres=${GRES} mem=${MEM} cpus-per-task=${CPUS_PER_TASK} time=${SLURM_TIME}"
sbatch --time="${SLURM_TIME}" --partition="${PARTITION}" --gres="${GRES}" --mem="${MEM}" --cpus-per-task="${CPUS_PER_TASK}" --export="${export_csv}" "${RUN_SCRIPT}"
