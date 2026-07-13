#!/bin/bash
# B12 1-min V-JEPA2 side — SMOKE: validate prune->predictor(LoRA)->probe fwd+bwd is finite and
# shapes are correct on a few samples, 1 epoch. Frozen ViT-L/256 encoder, 480 frames, keep 4096.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

# Code lives in the worktree; data/ckpt/outputs live in the main project folder.
export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export NUM_FRAMES="480"
export TARGET_FPS="8.0"
export KEEP_COUNT="4096"          # ~6.7% of 480f's 61440 tokens -> W=16-slot re-based context
export PRUNE_CHUNK_SIZE="256"
export ANTICIPATION_SEC="1.0"
export LORA_RANK="8"
export LORA_ALPHA="16.0"
export BATCH_SIZE="2"
export GRAD_ACCUM="1"
export NUM_EPOCHS="1"
export WARMUP_EPOCHS="0"
export NUM_WORKERS="8"
export MAX_TRAIN_SAMPLES="16"
export MAX_EVAL_SAMPLES="16"
export NUM_FRAMES TARGET_FPS
export RUN_TAG="b12_vjepa_1min_smoke"
export OUT_DIR="/path/to/VJEPA2-EXP/outputs/vjepa_prune_anticipation/b12_1min_smoke"

sbatch \
  --export=ALL \
  --partition=h100_tandon \
  --gres=gpu:h100:1 \
  --time=02:00:00 \
  --job-name=VJEPA2-EXP__vjepa_1min_smoke \
  "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
