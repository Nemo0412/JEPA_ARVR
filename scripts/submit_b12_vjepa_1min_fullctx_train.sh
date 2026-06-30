#!/bin/bash
# B12 1-min V-JEPA2 — FULL-CONTEXT (no-prune) training. Reads the ~928 GB full-token cache built by
# submit_b12_vjepa_1min_fullctx_cache.sh. The predictor (LoRA) takes the WHOLE 60s context at its
# REAL positions (0..61439, num_patches lifted) and predicts the 1s future at 61440.. ; probe over
# the slid window. This is the no-prune control for TODO #2: does keeping the central foreground
# tokens (that attention-importance pruning dropped) fix the noun collapse?
#
# Per-sample predictor fwd+bwd ≈ 2-3 s -> ~3 h/epoch; ~6-7 epochs fit in 24 h. batch-size=1
# (61440-token attention + probe over 61440 tokens). Any missing cache files are skipped, so this
# may also build the cache itself if launched before/without the array — but that is ~10 h serial.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

export NUM_FRAMES="480"
export TARGET_FPS="8.0"
export NO_PRUNE="1"             # full context, position_mode forced to 'true', key keep61440
export ANTICIPATION_SEC="1.0"
export LORA_RANK="8"
export LORA_ALPHA="16.0"
export LORA_LR="5e-5"
export LR="1e-4"
export BATCH_SIZE="1"
export GRAD_ACCUM="4"
export CACHE_BUILD_BATCH="4"    # only used if some cache files are still missing
export NUM_EPOCHS="8"
export WARMUP_EPOCHS="1"
export NUM_WORKERS="8"
export RUN_TAG="b12_vjepa_1min_fullctx"
export OUT_DIR="/path/to/VJEPA2-EXP/outputs/vjepa_prune_anticipation/b12_1min_fullctx"

sbatch \
  --export=ALL \
  --partition=h100_tandon \
  --gres=gpu:h100:1 \
  --mem=200G \
  --time=24:00:00 \
  --job-name=VJEPA2-EXP__vjepa_1min_fullctx \
  --output=/path/to/VJEPA2-EXP/logs/vjepa_1min_fullctx_%j.out \
  --error=/path/to/VJEPA2-EXP/logs/vjepa_1min_fullctx_%j.err \
  "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
