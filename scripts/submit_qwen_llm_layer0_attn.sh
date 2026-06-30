#!/bin/bash
# Submit Qwen LLM layer-0 attention visualization job.
# Usage: bash scripts/submit_qwen_llm_layer0_attn.sh [N_SAMPLE]
set -euo pipefail
SHARED="/path/to/VJEPA2-EXP"
WORKTREE="${SHARED}/.worktrees/vlm-pruning-compare"

N_SAMPLE="${1:-50}"

sbatch \
  --export=ALL,\
PROJECT_ROOT="${WORKTREE}",\
SHARED_PROJECT_ROOT="${SHARED}",\
N_SAMPLE="${N_SAMPLE}" \
  "${WORKTREE}/scripts/run_qwen_llm_layer0_attn.slurm"
