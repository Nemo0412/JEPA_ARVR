#!/bin/bash
# B12 1-min V-JEPA2 — TRUE-POSITION variant: pruned 4096 tokens keep their REAL positions across
# the 60s (0..61439); predictor predicts the 1s after the real last frame (positions 61440..),
# num_patches lifted. Tests whether the 1-min weakness is the RE-BASING specifically (vs the long
# window). Builds a fresh {tok,idx} cache (key suffix _idx); train 10 epochs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export NUM_FRAMES="480"
export TARGET_FPS="8.0"
export KEEP_COUNT="4096"
export POSITION_MODE="true"
export PRUNE_CHUNK_SIZE="256"
export ANTICIPATION_SEC="1.0"
export LORA_RANK="8"
export LORA_ALPHA="16.0"
export LORA_LR="5e-5"
export LR="1e-4"
export BATCH_SIZE="8"
export GRAD_ACCUM="1"
export CACHE_BUILD_BATCH="8"
export NUM_EPOCHS="10"
export WARMUP_EPOCHS="2"
export NUM_WORKERS="10"
export RUN_TAG="b12_vjepa_1min_truepos"
export OUT_DIR="/path/to/VJEPA2-EXP/outputs/vjepa_prune_anticipation/b12_1min_truepos"

sbatch \
  --export=ALL \
  --partition=h100_tandon \
  --gres=gpu:h100:1 \
  --mem=200G \
  --time=24:00:00 \
  --job-name=VJEPA2-EXP__vjepa_1min_truepos \
  "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
