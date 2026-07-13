#!/bin/bash
# B12 1-min V-JEPA2 side — FULL training: frozen ViT-L/256 encoder, 480 frames (60s @ 8fps),
# attention-importance pruning to 4096 tokens, predictor LoRA, 10 epochs on phd_split.
# Launch only after the smoke job is green.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

# Code lives in the worktree; data/ckpt/outputs live in the main project folder.
export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export NUM_FRAMES="480"
export TARGET_FPS="8.0"
export KEEP_COUNT="4096"
export PRUNE_CHUNK_SIZE="256"
export ANTICIPATION_SEC="1.0"
export LORA_RANK="8"
export LORA_ALPHA="16.0"
export LORA_LR="5e-5"
export LR="1e-4"
export BATCH_SIZE="8"             # training reads tiny cached tokens -> large batch is cheap
export GRAD_ACCUM="1"            # effective batch 8 (matches Qwen 1-min side)
export CACHE_BUILD_BATCH="8"     # longer encoder bursts during the one-time build -> better util
export NUM_EPOCHS="10"
export WARMUP_EPOCHS="2"
export NUM_WORKERS="10"
export NUM_FRAMES TARGET_FPS
export RUN_TAG="b12_vjepa_1min_keep4096"
export OUT_DIR="/path/to/VJEPA2-EXP/outputs/vjepa_prune_anticipation/b12_1min_keep4096"

sbatch \
  --export=ALL \
  --partition=h100_tandon \
  --gres=gpu:h100:1 \
  --mem=200G \
  --time=24:00:00 \
  --job-name=VJEPA2-EXP__vjepa_1min_full \
  "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
