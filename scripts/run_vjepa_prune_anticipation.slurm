#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__vjepa_prune_ant
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/vjepa_prune_ant_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/vjepa_prune_ant_%j.err

# B12 1-min V-JEPA2 side: frozen ViT-L/256 encoder -> attention-importance token pruning
# -> re-based predictor (LoRA) AR rollout -> AttentivePooler probe.
# Runs app.hdepic_lora_action_anticipation.train_vjepa_prune_anticipation (worktree
# vlm-pruning-compare). Pruning ported from refer_repo/JEPA_ARVR (PhD code).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

OUT_DIR="${OUT_DIR:-${SHARED_PROJECT_ROOT}/outputs/vjepa_prune_anticipation/b12_1min}"
NUM_FRAMES="${NUM_FRAMES:-480}"
TARGET_FPS="${TARGET_FPS:-8.0}"
KEEP_COUNT="${KEEP_COUNT:-4096}"
PRUNE_CHUNK_SIZE="${PRUNE_CHUNK_SIZE:-256}"
ENCODER_PRUNE_SCHEDULE="${ENCODER_PRUNE_SCHEDULE:-}"
ENCODER_PRUNE_METRIC="${ENCODER_PRUNE_METRIC:-next_attn}"
ANTICIPATION_SEC="${ANTICIPATION_SEC:-1.0}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
LORA_LR="${LORA_LR:-5e-5}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
NUM_WORKERS="${NUM_WORKERS:-10}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
CACHE_DIR="${CACHE_DIR:-${SHARED_PROJECT_ROOT}/data/preproc_cache_vjepa}"
CACHE_BUILD_BATCH="${CACHE_BUILD_BATCH:-4}"
POSITION_MODE="${POSITION_MODE:-rebased}"
RUN_TAG="${RUN_TAG:-b12_vjepa_1min}"
BEST_METRIC="${BEST_METRIC:-action_top5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Full-context (no-prune) + sharded-cache-build knobs. CACHE_SHARD defaults to the Slurm array
# task id so a job array parallelizes the one-time encoder cache build (each task = one shard).
NO_PRUNE="${NO_PRUNE:-0}"
BUILD_CACHE_ONLY="${BUILD_CACHE_ONLY:-0}"
CACHE_NUM_SHARDS="${CACHE_NUM_SHARDS:-1}"
CACHE_SHARD="${CACHE_SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"
[ "${NO_PRUNE}" = "1" ] && EXTRA_ARGS="${EXTRA_ARGS} --no-prune"
[ "${BUILD_CACHE_ONLY}" = "1" ] && EXTRA_ARGS="${EXTRA_ARGS} --build-cache-only"
[ -n "${ENCODER_PRUNE_SCHEDULE}" ] && EXTRA_ARGS="${EXTRA_ARGS} --encoder-prune-schedule ${ENCODER_PRUNE_SCHEDULE} --encoder-prune-metric ${ENCODER_PRUNE_METRIC}"
EXTRA_ARGS="${EXTRA_ARGS} --cache-shard ${CACHE_SHARD} --cache-num-shards ${CACHE_NUM_SHARDS}"

export PROJECT_ROOT SHARED_PROJECT_ROOT ENV_NAME OUT_DIR NUM_FRAMES TARGET_FPS KEEP_COUNT
export PRUNE_CHUNK_SIZE ENCODER_PRUNE_SCHEDULE ENCODER_PRUNE_METRIC ANTICIPATION_SEC LORA_RANK LORA_ALPHA LORA_LR LR BATCH_SIZE GRAD_ACCUM
export NUM_EPOCHS WARMUP_EPOCHS NUM_WORKERS MAX_TRAIN_SAMPLES MAX_EVAL_SAMPLES RUN_TAG BEST_METRIC EXTRA_ARGS
export CACHE_DIR CACHE_BUILD_BATCH POSITION_MODE

echo "[vjepa_prune_ant] tag=${RUN_TAG} frames=${NUM_FRAMES} keep_count=${KEEP_COUNT} lora_rank=${LORA_RANK}"
echo "[vjepa_prune_ant] out_dir=${OUT_DIR} epochs=${NUM_EPOCHS} bs=${BATCH_SIZE} ga=${GRAD_ACCUM}"
nvidia-smi --query-gpu=name --format=csv,noheader || true

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
mkdir -p "${OUT_DIR}"

python -m app.hdepic_lora_action_anticipation.train_vjepa_prune_anticipation \
  --out-dir "${OUT_DIR}" \
  --num-frames "${NUM_FRAMES}" \
  --target-fps "${TARGET_FPS}" \
  --keep-count "${KEEP_COUNT}" \
  --position-mode "${POSITION_MODE}" \
  --cache-dir "${CACHE_DIR}" \
  --cache-build-batch "${CACHE_BUILD_BATCH}" \
  --prune-chunk-size "${PRUNE_CHUNK_SIZE}" \
  --anticipation-sec "${ANTICIPATION_SEC}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-lr "${LORA_LR}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --num-epochs "${NUM_EPOCHS}" \
  --warmup-epochs "${WARMUP_EPOCHS}" \
  --num-workers "${NUM_WORKERS}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --max-eval-samples "${MAX_EVAL_SAMPLES}" \
  --run-tag "${RUN_TAG}" \
  --best-metric "${BEST_METRIC}" \
  ${EXTRA_ARGS}
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
