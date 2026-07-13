#!/bin/bash
set -euo pipefail

# B12 7s V-JEPA2 pruned training using the best 1min pruning method:
# mid-encoder next_attn, schedule 9:0.5,17:4096, true positions.
#
# 7s full encoder tokens = 7168; final keep_count=4096 matches the 4s V-JEPA token budget.
# Cache is enabled through train_vjepa_prune_anticipation.py's frozen-encoder cache.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export NUM_FRAMES="56"
export TARGET_FPS="8.0"
export KEEP_COUNT="4096"
export POSITION_MODE="true"
export ENCODER_PRUNE_SCHEDULE="9:0.5,17:4096"
export ENCODER_PRUNE_METRIC="next_attn"
export ANTICIPATION_SEC="1.0"
export LORA_RANK="8"
export LORA_ALPHA="16.0"
export LORA_LR="5e-5"
export LR="1e-4"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export CACHE_BUILD_BATCH="${CACHE_BUILD_BATCH:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
export NUM_WORKERS="${NUM_WORKERS:-10}"
export BEST_METRIC="${BEST_METRIC:-action_top5}"
export RUN_TAG="b12_vjepa_7s_mid_nextattn_truepos"
export OUT_DIR="${SHARED_PROJECT_ROOT}/outputs/vjepa_prune_anticipation/b12_7s_mid_nextattn_truepos"

SBATCH_EXTRA=()
if [ -n "${SBATCH_DEPENDENCY:-}" ]; then
  SBATCH_EXTRA+=(--dependency="${SBATCH_DEPENDENCY}")
fi

sbatch \
  "${SBATCH_EXTRA[@]}" \
  --export=ALL \
  --partition="${SLURM_PARTITION:-h100_tandon}" \
  --gres="${SLURM_GRES:-gpu:h100:1}" \
  --mem="${SLURM_MEM:-128G}" \
  --time="${SLURM_TIME:-18:00:00}" \
  --job-name=VJEPA2-EXP__vjepa_7s_midnext \
  --output="${SHARED_PROJECT_ROOT}/logs/vjepa_7s_midnext_%j.out" \
  --error="${SHARED_PROJECT_ROOT}/logs/vjepa_7s_midnext_%j.err" \
  "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
