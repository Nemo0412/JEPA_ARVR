#!/bin/bash
# B12 1-min V-JEPA2 — FULL-CONTEXT (no-prune) cache build, PARALLELIZED as a Slurm array.
# Each array task = one shard: it decodes its slice of train/val/test, runs the frozen ViT-L/256
# encoder on the 480-frame window, and writes the FULL ~61440-token output (fp16, ~126 MB/sample)
# to the cache key `nf480_fps8.0_px256_keep61440_idx`, then exits (--build-cache-only).
#
# 4 shards × ~1850 samples × ~5 s ≈ 2.6 h/shard. Total cache ≈ 928 GB (fits the ~2.09 TB free quota).
# After all shards finish, run submit_b12_vjepa_1min_fullctx_train.sh (reads this cache).
#
# Usage:  bash scripts/submit_b12_vjepa_1min_fullctx_cache.sh        # default 4 shards (array 0-3)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

NUM_SHARDS="${NUM_SHARDS:-4}"

export NUM_FRAMES="480"
export TARGET_FPS="8.0"
export NO_PRUNE="1"             # keep ALL tokens -> keep_count auto = 61440, key keep61440
export BUILD_CACHE_ONLY="1"     # build cache + exit, no training
export CACHE_NUM_SHARDS="${NUM_SHARDS}"
export CACHE_BUILD_BATCH="6"
export NUM_WORKERS="10"
export RUN_TAG="b12_vjepa_1min_fullctx_cache"
# OUT_DIR is unused for cache-only but required by the runner; point at the train out dir.
export OUT_DIR="/path/to/VJEPA2-EXP/outputs/vjepa_prune_anticipation/b12_1min_fullctx"

sbatch \
  --export=ALL \
  --array=0-$((NUM_SHARDS - 1)) \
  --partition=h100_tandon \
  --gres=gpu:h100:1 \
  --mem=200G \
  --time=06:00:00 \
  --job-name=VJEPA2-EXP__vjepa_1min_fullctx_cache \
  --output=/path/to/VJEPA2-EXP/logs/vjepa_1min_fullctx_cache_%A_%a.out \
  --error=/path/to/VJEPA2-EXP/logs/vjepa_1min_fullctx_cache_%A_%a.err \
  "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
