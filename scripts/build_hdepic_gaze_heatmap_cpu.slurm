#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_gaze_heat
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_gaze_heatmap_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_gaze_heatmap_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# CPU-only first-pass gaze heatmap/overlay generation.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
VIDEO_ID="${VIDEO_ID:-P01-20240202-110250}"
GAZE_ZIP="${GAZE_ZIP:-${PROJECT_ROOT}/data/raw/HD-EPIC/SLAM-and-Gaze/P01/mps_${VIDEO_ID}_vrs.zip}"
VIDEO_PATH="${VIDEO_PATH:-${PROJECT_ROOT}/data/raw/HD-EPIC/Videos/P01/${VIDEO_ID}.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_gaze_heatmaps}"
MAX_OVERLAY_FRAMES="${MAX_OVERLAY_FRAMES:-300}"
SIGMA="${SIGMA:-18}"
LIST_ONLY="${LIST_ONLY:-0}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

export PROJECT_ROOT SIF OVERLAY ENV_NAME VIDEO_ID GAZE_ZIP VIDEO_PATH OUTPUT_DIR MAX_OVERLAY_FRAMES SIGMA LIST_ONLY

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

if [[ ! -f "${GAZE_ZIP}" ]]; then
  found_zip="$(find "${PROJECT_ROOT}/data/raw/HD-EPIC" -name "mps_${VIDEO_ID}_vrs.zip" -print -quit)"
  if [[ -n "${found_zip}" ]]; then
    echo "[gaze-heatmap] Auto-detected gaze zip: ${found_zip}"
    GAZE_ZIP="${found_zip}"
  else
    echo "[gaze-heatmap] Missing gaze zip: ${GAZE_ZIP}" >&2
    echo "[gaze-heatmap] Tried auto-detecting mps_${VIDEO_ID}_vrs.zip under ${PROJECT_ROOT}/data/raw/HD-EPIC" >&2
    exit 1
  fi
fi

args=(
  "${PROJECT_ROOT}/scripts/build_hdepic_gaze_heatmap_video.py"
  --gaze-zip "${GAZE_ZIP}"
  --output-dir "${OUTPUT_DIR}"
  --video-id "${VIDEO_ID}"
  --sigma "${SIGMA}"
  --max-overlay-frames "${MAX_OVERLAY_FRAMES}"
)

if [[ -f "${VIDEO_PATH}" ]]; then
  args+=(--video "${VIDEO_PATH}")
else
  echo "[gaze-heatmap] Video not found, producing aggregate heatmap only: ${VIDEO_PATH}" >&2
fi

if [[ "${LIST_ONLY}" == "1" ]]; then
  args+=(--list-only)
fi

python "${args[@]}"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
