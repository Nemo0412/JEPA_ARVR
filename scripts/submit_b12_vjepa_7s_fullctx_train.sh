#!/bin/bash
set -euo pipefail

# B12 7s V-JEPA2 full-context/no-prune training.
# Reads the frozen encoder cache built by submit_b12_vjepa_7s_fullctx_cache.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export NUM_FRAMES="56"
export TARGET_FPS="8.0"
export NO_PRUNE="1"
export ANTICIPATION_SEC="1.0"
export LORA_RANK="8"
export LORA_ALPHA="16.0"
export LORA_LR="5e-5"
export LR="1e-4"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export CACHE_BUILD_BATCH="${CACHE_BUILD_BATCH:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export BEST_METRIC="${BEST_METRIC:-action_top5}"
export RUN_TAG="b12_vjepa_7s_fullctx"
export OUT_DIR="${SHARED_PROJECT_ROOT}/outputs/vjepa_prune_anticipation/b12_7s_fullctx"

SBATCH_EXTRA=()
if [ -n "${SBATCH_DEPENDENCY:-}" ]; then
  SBATCH_EXTRA+=(--dependency="${SBATCH_DEPENDENCY}")
fi

sbatch \
  "${SBATCH_EXTRA[@]}" \
  --export=ALL \
  --partition="${SLURM_PARTITION:-h100_tandon}" \
  --gres="${SLURM_GRES:-gpu:h100:1}" \
  --mem="${SLURM_MEM:-160G}" \
  --time="${SLURM_TIME:-24:00:00}" \
  --job-name=VJEPA2-EXP__vjepa_7s_fullctx \
  --output="${SHARED_PROJECT_ROOT}/logs/vjepa_7s_fullctx_%j.out" \
  --error="${SHARED_PROJECT_ROOT}/logs/vjepa_7s_fullctx_%j.err" \
  "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
