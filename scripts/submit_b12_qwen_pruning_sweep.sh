#!/bin/bash
set -euo pipefail

# B12 equal-compute study: Qwen2.5-VL-3B video-token pruning sweep.
#
# Submits one probe-LoRA training job per keep_ratio (attention-importance pruning of the
# merged video tokens before the LLM; see app/.../qwen_token_pruning.py). All jobs share the
# same input clip and 256px frame resolution (matched to V-JEPA2) so that keep_ratio alone
# traces the accuracy-vs-compute curve. keep_ratio=1.0 is the un-pruned anchor.
#
# DEFAULT WINDOW = 4s, the PhD's actual observation window (verified: FRAMES_PER_CLIP=32,
# FPS=8 across the whole JEPA_ARVR repo and its git history -> 32/8 = 4s; the PhD's reported
# "6s" does not match their code). ANTICIPATION 1s before action onset is hardcoded in
# compute_clip_window. For a longer window override NUM_FRAMES/TARGET_FPS (e.g. the planned
# 30s run), keeping PROBE_NUM_FRAMES == NUM_FRAMES.
#
# NO TEMPORAL DOWNSAMPLING (advisor requirement, 2026-06-25): every decoded frame is fed to
# the model (PROBE_NUM_FRAMES == NUM_FRAMES), so frames are never sampled-then-dropped. The
# compute lever is token PRUNING, not frame count -- this is exactly why pruning lets us feed
# all frames yet still hit a fair compute budget. (The PhD's own Qwen probe instead
# downsampled 32->8 frames via SAMPLE_FRAMES=8; we do not -- we feed all 32.)
#
#   clip      : NUM_FRAMES=32 @ TARGET_FPS=8 = 4s window, all 32 frames fed (PhD V-JEPA2 config)
#   resolution: QWEN_FRAME_SIZE=256  -> 18x18 patch grid/frame (matched to V-JEPA2's 256)
#   tokens    : grid_t=16 -> 16 x 81 = 1296 merged video tokens at keep=1.0; pruning cuts this.
#
# Note: the vision tower (32-layer ViT) still processes all frames pre-prune, a fixed cost like
# V-JEPA2's encoder; pruning controls the dominant 36-layer LLM term. Compute parity with
# V-JEPA2 (256px) is ~keep_ratio 0.35 (LLM is ~2.8x V-JEPA2 at matched resolution; B12 doc).
#
# Usage:
#   bash scripts/submit_b12_qwen_pruning_sweep.sh                    # full 4s sweep, full training
#   KEEP_RATIOS="1.0 0.5" MAX_TRAIN_SAMPLES=32 NUM_EPOCHS=1 \
#     SLURM_TIME=01:00:00 RUN_PREFIX=smoke \
#     bash scripts/submit_b12_qwen_pruning_sweep.sh                  # quick real-data smoke

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_vlm_probe_lora.sh"

KEEP_RATIOS="${KEEP_RATIOS:-1.0 0.5 0.35 0.25 0.125}"
RUN_PREFIX="${RUN_PREFIX:-b12_qwen_prune_4s}"

export BACKEND="qwen25vl"
export NUM_FRAMES="${NUM_FRAMES:-32}"
export TARGET_FPS="${TARGET_FPS:-8.0}"
export PROBE_NUM_FRAMES="${PROBE_NUM_FRAMES:-32}"   # == NUM_FRAMES: feed every frame, no downsampling
export QWEN_FRAME_SIZE="${QWEN_FRAME_SIZE:-256}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
export SLURM_TIME="${SLURM_TIME:-24:00:00}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
# Throughput / GPU-utilisation (decode-at-256 + these keep the H100 fed; avoids the 8.8%-util
# starvation that risks HPC's low-utilisation cancellation). Effective batch stays 8 (=PhD's
# 4x2) via batch_size=8 / grad_accum=1, so training dynamics are unchanged.
# Preprocessing cache (shared across keep_ratio): turns the CPU-bound decode+tokenize pipeline
# into cheap tensor loads after epoch 0, so the GPU stops starving even on the heavily-pruned
# (low-compute) keep=0.15 job. With the cache, few CPUs are needed -> short queue + high util.
export PREPROC_CACHE_DIR="${PREPROC_CACHE_DIR:-/path/to/VJEPA2-EXP/data/preproc_cache_qwen}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-10}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-12}"
# GPU choice: H100. Measured 2026-06-25 with decode-at-256 + 14 workers: H100 steady-state
# util ~75-80% (1.7 s/batch); H200 was *lower* (~60%) because faster compute waits more on
# data (partly data-bound). So H100 is the safer pick for the 60%/2h fair-use threshold.
export SLURM_PARTITION="${SLURM_PARTITION:-h100_tandon}"
export SLURM_GRES="${SLURM_GRES:-gpu:h100:1}"

for kr in ${KEEP_RATIOS}; do
  tag="${kr/./p}"   # 0.35 -> 0p35 for a filesystem-safe run tag
  RUN_TAG="${RUN_PREFIX}_kr${tag}" PRUNE_KEEP_RATIO="${kr}" bash "${SUBMIT}"
done
