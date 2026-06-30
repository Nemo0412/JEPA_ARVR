#!/bin/bash
#SBATCH --job-name=b12_qwen_1min_perf_smoke
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --partition=h100_tandon
#SBATCH --output=/path/to/VJEPA2-EXP/logs/b12_qwen_1min_perf_smoke_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/b12_qwen_1min_perf_smoke_%j.err

# GPU utilization + memory profiling smoke run for Qwen2.5-VL-3B 1-min window training.
# Samples nvidia-smi every 5 seconds into logs/perf_<jobid>_nvidia_smi.csv.
# 400 train samples × ~3.6 s/sample ≈ 24 min training, then val on 682 cached samples
# ≈ 5 min → ~30 min total. Provides a full train+val GPU util curve for reporting.

set -euo pipefail

SHARED="/path/to/VJEPA2-EXP"
WORKTREE="${SHARED}//path/to/VJEPA2-EXP"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="${SHARED}/overlay-15GB-500K.ext3"
ENV_NAME="EgoLifeExp"
ANN_ROOT="${SHARED}/data/hdepic_vjepa_annotations/phd_split"

PERF_INTERVAL=5   # seconds between nvidia-smi samples
PERF_PREFIX="${SHARED}/logs/perf_${SLURM_JOB_ID}"

# --- Start nvidia-smi background sampler ---
echo "[perf] Sampling every ${PERF_INTERVAL}s → ${PERF_PREFIX}_nvidia_smi.csv"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l "${PERF_INTERVAL}" \
  > "${PERF_PREFIX}_nvidia_smi.csv" \
  2> "${PERF_PREFIX}_nvidia_smi.err" &
PERF_PID=$!
echo "[perf] nvidia-smi PID=${PERF_PID}"

# --- Run training ---
payload=""
read -r -d '' payload <<BASH || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${WORKTREE}:${SHARED}:${SHARED}/vjepa2:\${PYTHONPATH:-}"
cd "${WORKTREE}"

python -m app.hdepic_lora_action_anticipation.train_vlm_probe_lora \\
  --backend qwen25vl \\
  --train-csv "${ANN_ROOT}/HD_EPIC_train_vjepa.csv" \\
  --val-csv   "${ANN_ROOT}/HD_EPIC_val_vjepa.csv" \\
  --video-root "${SHARED}/data/hdepic_vjepa_videos" \\
  --verb-classes-csv "${SHARED}/data/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_verb_classes.csv" \\
  --noun-classes-csv "${SHARED}/data/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_noun_classes.csv" \\
  --output-dir "${SHARED}/outputs/vlm_probe_lora/b12_qwen_1min_perf_smoke" \\
  --num-frames 480 \\
  --probe-num-frames 480 \\
  --target-fps 8.0 \\
  --qwen-frame-size 256 \\
  --prune-keep-ratio 1.0 \\
  --prune-chunk-size 0 \\
  --preproc-cache-dir "${SHARED}/data/preproc_cache_qwen" \\
  --max-train-samples 400 \\
  --num-epochs 1 \\
  --lr 2e-4 \\
  --weight-decay 1e-4 \\
  --warmup-epochs 0 \\
  --lora-rank 16 \\
  --lora-alpha 32.0 \\
  --batch-size 1 \\
  --grad-accum-steps 8 \\
  --log-every 20 \\
  --torch-dtype bfloat16 \\
  --num-workers 10 \\
  --seed 0 \\
  --local-files-only
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
TRAIN_EXIT=$?

# --- Stop nvidia-smi ---
kill "${PERF_PID}" 2>/dev/null || true
wait "${PERF_PID}" 2>/dev/null || true

echo "[perf] Done. CSV lines: $(wc -l < "${PERF_PREFIX}_nvidia_smi.csv" 2>/dev/null || echo 0)"
echo "[perf] File: ${PERF_PREFIX}_nvidia_smi.csv"

exit ${TRAIN_EXIT}
