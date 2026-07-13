#!/bin/bash
# Build predictor-block-0-guided 1min→4096 token cache.
# Submits a 4-shard job array; each shard takes ~75 min on a standard GPU.
# Run BEFORE submit_b12_vjepa_1min_pred0_train.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
export NUM_SHARDS="4"

sbatch \
    --array=0-3 \
    --export=ALL \
    --job-name=VJEPA2-EXP__pred0_cache \
    "${WORKTREE_ROOT}/scripts/run_build_pred0_1min_cache.slurm"
echo "Cache-build array (4 shards) submitted."
