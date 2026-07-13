#!/bin/bash
# B12 1-min V-JEPA2 — predictor-block-0-guided pruning, 1min→4096 tokens.
# Reads the pre-built pred0 cache (nf480_fps8.0_px256_pred0_keep4096_idx).
# Must run submit_b12_vjepa_1min_pred0_cache_build.sh first.
#
# Comparison against the encoder-col-sum baseline (b12_1min_keep4096):
#   - Same keep_count=4096, same predictor LoRA + rebased positions
#   - Difference: token SELECTION uses predictor block-0 attention instead of encoder block-24
#
# Usage: bash scripts/submit_b12_vjepa_1min_pred0_train.sh [--smoke]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKTREE_ROOT}"

export PROJECT_ROOT="${WORKTREE_ROOT}"
export SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

SMOKE=0
for arg in "$@"; do [[ "${arg}" == "--smoke" ]] && SMOKE=1; done

export NUM_FRAMES="480"
export TARGET_FPS="8.0"
export KEEP_COUNT="4096"         # matches the pred0 cache
export POSITION_MODE="rebased"   # PhD-style re-based positions, same as b12_1min_keep4096
export ANTICIPATION_SEC="1.0"
export LORA_RANK="8"
export LORA_ALPHA="16.0"
export LORA_LR="5e-5"
export LR="1e-4"
export GRAD_ACCUM="4"
export WARMUP_EPOCHS="2"
export NUM_WORKERS="10"
export CACHE_BUILD_BATCH="4"
export CACHE_DIR="${SHARED_PROJECT_ROOT}/data/preproc_cache_vjepa"

# Tell the training script to use the pre-built pred0 cache directly
export EXTRA_ARGS="--cache-key-override nf480_fps8.0_px256_pred0_keep4096_idx"

if [[ "${SMOKE}" == "1" ]]; then
    export MAX_TRAIN_SAMPLES="64"
    export MAX_EVAL_SAMPLES="32"
    export NUM_EPOCHS="2"
    export BATCH_SIZE="4"
    export RUN_TAG="b12_vjepa_1min_pred0_smoke"
    export OUT_DIR="${SHARED_PROJECT_ROOT}/outputs/vjepa_prune_anticipation/b12_mid_pred0_smoke"
    sbatch \
        --export=ALL \
        --partition=h100_tandon \
        --gres=gpu:h100:1 \
        --mem=64G \
        --time=00:30:00 \
        --job-name=VJEPA2-EXP__vjepa_pred0_smoke \
        --output="${SHARED_PROJECT_ROOT}/logs/vjepa_pred0_smoke_%j.out" \
        --error="${SHARED_PROJECT_ROOT}/logs/vjepa_pred0_smoke_%j.err" \
        "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
    echo "Smoke job submitted."
else
    export MAX_TRAIN_SAMPLES="0"
    export MAX_EVAL_SAMPLES="0"
    export NUM_EPOCHS="10"
    export BATCH_SIZE="4"
    export RUN_TAG="b12_vjepa_1min_pred0"
    export OUT_DIR="${SHARED_PROJECT_ROOT}/outputs/vjepa_prune_anticipation/b12_mid_pred0_truepos_fulltrain"
    sbatch \
        --export=ALL \
        --partition=h100_tandon \
        --gres=gpu:h100:1 \
        --mem=128G \
        --time=24:00:00 \
        --job-name=VJEPA2-EXP__vjepa_1min_pred0 \
        --output="${SHARED_PROJECT_ROOT}/logs/vjepa_1min_pred0_%j.out" \
        --error="${SHARED_PROJECT_ROOT}/logs/vjepa_1min_pred0_%j.err" \
        "${WORKTREE_ROOT}/scripts/run_vjepa_prune_anticipation.slurm"
    echo "Full-train job submitted."
fi
