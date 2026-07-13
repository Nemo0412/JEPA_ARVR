#!/bin/bash
set -euo pipefail

# B12 7s V-JEPA2 full-context cache build.
#
# 56 frames @ 8 fps with tubelet_size=2 => 28 × 16 × 16 = 7168 encoder tokens.
# This builds the frozen encoder cache for the no-prune/full-context 7s control.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export NUM_FRAMES="56"
export TARGET_FPS="8.0"
export NO_PRUNE="1"
export BUILD_CACHE_ONLY="1"
export CACHE_NUM_SHARDS="${CACHE_NUM_SHARDS:-4}"
export ANTICIPATION_SEC="1.0"
export CACHE_BUILD_BATCH="${CACHE_BUILD_BATCH:-4}"
export NUM_WORKERS="${NUM_WORKERS:-10}"
export RUN_TAG="b12_vjepa_7s_fullctx_cache"
export OUT_DIR="${SHARED_PROJECT_ROOT}/outputs/vjepa_prune_anticipation/b12_7s_fullctx_cache"

sbatch \
  --export=ALL \
  --array="0-$((CACHE_NUM_SHARDS - 1))" \
  --partition="${SLURM_PARTITION:-h100_tandon}" \
  --gres="${SLURM_GRES:-gpu:h100:1}" \
  --mem="${SLURM_MEM:-160G}" \
  --time="${SLURM_TIME:-08:00:00}" \
  --job-name=VJEPA2-EXP__vjepa_7s_fullctx_cache \
  --output="${SHARED_PROJECT_ROOT}/logs/vjepa_7s_fullctx_cache_%A_%a.out" \
  --error="${SHARED_PROJECT_ROOT}/logs/vjepa_7s_fullctx_cache_%A_%a.err" \
  "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
