#!/bin/bash
# B12 V-JEPA2 4s BASELINE / CONTROL for the 1-min run, same pipeline (predictor-LoRA + probe +
# pruned-token cache, no aug). 4s = 32 frames @ 8fps -> 4096 encoder tokens; keep_count=4096 keeps
# ALL tokens (no pruning) at TRUE positions (W=16 = predictor's trained regime), so there is NO
# re-basing distortion. Comparing this to the 1-min run isolates the 60s-window + re-basing effect.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export NUM_FRAMES="32"            # 4s @ 8fps
export TARGET_FPS="8.0"
export KEEP_COUNT="4096"          # = all 4096 tokens at 32f -> no pruning, true positions
export PRUNE_CHUNK_SIZE="256"
export ANTICIPATION_SEC="1.0"
export LORA_RANK="8"
export LORA_ALPHA="16.0"
export LORA_LR="5e-5"
export LR="1e-4"
export BATCH_SIZE="8"
export GRAD_ACCUM="1"
export CACHE_BUILD_BATCH="16"     # 32-frame clips are light -> larger build batch
export NUM_EPOCHS="10"
export WARMUP_EPOCHS="2"
export NUM_WORKERS="10"
export RUN_TAG="b12_vjepa_4s_keep4096"
export OUT_DIR="/path/to/VJEPA2-EXP/outputs/vjepa_prune_anticipation/b12_4s_keep4096"

sbatch \
  --export=ALL \
  --partition=h100_tandon \
  --gres=gpu:h100:1 \
  --mem=200G \
  --time=12:00:00 \
  --job-name=VJEPA2-EXP__vjepa_4s \
  "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
