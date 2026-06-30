#!/bin/bash
# B12 1-min V-JEPA2 — FULL-CONTEXT (no-prune) SMOKE: validates the new no-prune path end-to-end on a
# real H100 before the ~928 GB build. Builds the full ~61440-token cache for a handful of samples
# (real cache key keep61440 -> reused by the real run), trains 1 epoch, evals. Watch peak GPU mem +
# per-sample time to confirm the ~3 h/epoch estimate and that batch-size=1 fits.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export NUM_FRAMES="480"
export TARGET_FPS="8.0"
export NO_PRUNE="1"
export ANTICIPATION_SEC="1.0"
export BATCH_SIZE="1"
export GRAD_ACCUM="2"
export CACHE_BUILD_BATCH="4"
export NUM_EPOCHS="1"
export WARMUP_EPOCHS="0"
export NUM_WORKERS="8"
export MAX_TRAIN_SAMPLES="8"
export MAX_EVAL_SAMPLES="4"
export LOG_EVERY="1"
export RUN_TAG="b12_vjepa_1min_fullctx_smoke"
export OUT_DIR="/path/to/VJEPA2-EXP/outputs/vjepa_prune_anticipation/b12_1min_fullctx_smoke"

sbatch \
  --export=ALL \
  --partition=h100_tandon \
  --gres=gpu:h100:1 \
  --mem=200G \
  --time=01:30:00 \
  --job-name=VJEPA2-EXP__vjepa_1min_fullctx_smoke \
  --output=/path/to/VJEPA2-EXP/logs/vjepa_1min_fullctx_smoke_%j.out \
  --error=/path/to/VJEPA2-EXP/logs/vjepa_1min_fullctx_smoke_%j.err \
  "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
