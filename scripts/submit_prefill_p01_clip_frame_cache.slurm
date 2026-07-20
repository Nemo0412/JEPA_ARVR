#!/bin/bash
#SBATCH --job-name=prefill_clip_cache
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
# No GPU: this is CPU decode-only. Keep under 2h in case the scheduler
# still tracks the node; chain with auto-resubmit until cache is complete.
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/ll5914/logs/prefill_clip_cache_%j.out
#SBATCH --error=/scratch/ll5914/logs/prefill_clip_cache_%j.err

set -euo pipefail

# CPU-only prefill of decoded uint8 clips for tri-modal util ≥60%.
# Prefer: let the GPU train job fill-on-miss (same cache path). Use this
# script to warm the cache ahead of time on a CPU allocation if available.
# If the partition requires a GPU, prefer NOT to run this on a GPU node
# (0% util → kill); run on login/CPU instead:
#   TRI_MODAL_FRAME_CACHE=... PYTHONPATH=... python -m app.hdepic_lora_action_anticipation.prefill_clip_frame_cache ...

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
VJEPA_ROOT="/home/ll5914/ARVR_Video/vjepa2"
ANN_DIR="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split"
VIDEO_ROOT="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
CACHE_ROOT="${TRI_MODAL_FRAME_CACHE:-/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1}"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_prefill_p01_clip_frame_cache.slurm"
DONE_FLAG="${CACHE_ROOT}/PREFILL_DONE"

mkdir -p /scratch/ll5914/logs "${CACHE_ROOT}"
export TRI_MODAL_FRAME_CACHE="${CACHE_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Prefill already done (${DONE_FLAG})"
  exit 0
fi

_RESUBMITTED=0
request_resubmit() {
  [[ -f "${DONE_FLAG}" ]] && return 0
  [[ "${_RESUBMITTED}" == "1" ]] && return 0
  _RESUBMITTED=1
  sbatch "${SCRIPT_PATH}" || true
}
trap 'request_resubmit' USR1 TERM

echo "Prefill clip cache -> ${CACHE_ROOT} @ $(date) on $(hostname)"
set +e
"${PYTHON}" -m app.hdepic_lora_action_anticipation.prefill_clip_frame_cache \
  --video-root "${VIDEO_ROOT}" \
  --train-csv "${ANN_DIR}/HD_EPIC_train_vjepa.csv" \
  --val-csv "${ANN_DIR}/HD_EPIC_val_vjepa.csv" \
  --frames-per-clip 32 \
  --fps 8 \
  --anticipation-sec 1.0 \
  --anticipation-point 1.0
rc=$?
set -e

# Mark done when train+val clips are present (heuristic: ≥7000 npy files).
n_files=$(find "${CACHE_ROOT}" -type f -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
echo "cache npy files=${n_files} exit=${rc}"
if [[ "${n_files}" -ge 7000 ]]; then
  echo "complete n_files=${n_files}" > "${DONE_FLAG}"
  echo "PREFILL_DONE"
else
  request_resubmit
fi

du -sh "${CACHE_ROOT}" || true
exit 0
