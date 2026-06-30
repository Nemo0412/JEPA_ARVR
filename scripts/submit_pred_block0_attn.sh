#!/bin/bash
# Submit predictor block-0 attention visualization job.
# Usage: bash scripts/submit_pred_block0_attn.sh [N_SAMPLE] [CHUNK_SIZE]
set -euo pipefail
SHARED="/path/to/VJEPA2-EXP"
WORKTREE="${SHARED}/.worktrees/vlm-pruning-compare"

N_SAMPLE="${1:-100}"
CHUNK_SIZE="${2:-256}"

sbatch \
  --export=ALL,\
PROJECT_ROOT="${WORKTREE}",\
SHARED_PROJECT_ROOT="${SHARED}",\
N_SAMPLE="${N_SAMPLE}",\
CHUNK_SIZE="${CHUNK_SIZE}" \
  "${WORKTREE}/scripts/run_pred_block0_attn.slurm"
