#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_video_health
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_video_health_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_video_health_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# CPU-only video health check for HD-EPIC videos referenced by the current
# V-JEPA/EK100-style train/val CSVs. This does not use a GPU.
#
# Examples:
#   sbatch scripts/check_hdepic_video_health_cpu.slurm
#   sbatch --export=ALL,HORIZONS="1 10",ANTICIPATION_POINTS="0.0 0.25 0.75" scripts/check_hdepic_video_health_cpu.slurm
#   sbatch --export=ALL,MAX_ROWS_PER_VIDEO=20 scripts/check_hdepic_video_health_cpu.slurm

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations}"
TRAIN_CSV="${TRAIN_CSV:-${ANNOTATIONS_DIR}/HD_EPIC_train_vjepa.csv}"
VAL_CSV="${VAL_CSV:-${ANNOTATIONS_DIR}/HD_EPIC_val_vjepa.csv}"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/hdepic_video_health}"

FILE_FORMAT="${FILE_FORMAT:-1}"
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-16}"
DECODER_FPS="${DECODER_FPS:-5}"
HORIZONS="${HORIZONS:-1}"
ANTICIPATION_POINTS="${ANTICIPATION_POINTS:-0.0}"
MAX_ROWS_PER_VIDEO="${MAX_ROWS_PER_VIDEO:-0}"
SKIP_FRAME_CHECK="${SKIP_FRAME_CHECK:-0}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

export PROJECT_ROOT ENV_NAME TRAIN_CSV VAL_CSV ADAPTED_VIDEO_ROOT OUTPUT_DIR
export FILE_FORMAT FRAMES_PER_CLIP DECODER_FPS HORIZONS ANTICIPATION_POINTS MAX_ROWS_PER_VIDEO SKIP_FRAME_CHECK

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

cd "${PROJECT_ROOT}"

if [[ ! -f "${TRAIN_CSV}" ]]; then
    echo "[health] Missing TRAIN_CSV=${TRAIN_CSV}" >&2
    exit 1
fi
if [[ ! -f "${VAL_CSV}" ]]; then
    echo "[health] Missing VAL_CSV=${VAL_CSV}" >&2
    exit 1
fi

args=(
    --csv "${TRAIN_CSV}" "${VAL_CSV}"
    --base-path "${ADAPTED_VIDEO_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --file-format "${FILE_FORMAT}"
    --frames-per-clip "${FRAMES_PER_CLIP}"
    --decoder-fps "${DECODER_FPS}"
    --anticipation-times ${HORIZONS}
    --anticipation-points ${ANTICIPATION_POINTS}
    --max-rows-per-video "${MAX_ROWS_PER_VIDEO}"
)

if [[ "${SKIP_FRAME_CHECK}" == "1" ]]; then
    args+=(--skip-frame-check)
fi

echo "========================================"
echo "Mode       : HD-EPIC video health"
echo "Job ID     : ${SLURM_JOB_ID:-unknown}"
echo "Node       : $(hostname)"
echo "Start      : $(date)"
echo "Project    : ${PROJECT_ROOT}"
echo "Train CSV  : ${TRAIN_CSV}"
echo "Val CSV    : ${VAL_CSV}"
echo "Video root : ${ADAPTED_VIDEO_ROOT}"
echo "Output dir : ${OUTPUT_DIR}"
echo "Horizons   : ${HORIZONS}"
echo "Points     : ${ANTICIPATION_POINTS}"
echo "========================================"

set +e
python "${PROJECT_ROOT}/scripts/check_hdepic_video_health.py" "${args[@]}"
status=$?
set -e

echo "[health] Summary preview:"
python - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["OUTPUT_DIR"]) / "video_health_summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    for key in [
        "num_rows",
        "num_videos",
        "ok_videos",
        "bad_videos",
        "missing_videos",
        "open_failed_videos",
        "frame_failed_videos",
        "bad_row_checks",
    ]:
        print(f"{key}: {summary.get(key)}")
else:
    print(f"missing summary: {summary_path}")
PY

echo "[health] Bad videos:"
bad_video_file="${OUTPUT_DIR}/video_health_bad_videos.txt"
if [[ -s "${bad_video_file}" ]]; then
    cat "${bad_video_file}"
else
    echo "<none>"
fi

echo "========================================"
echo "End        : $(date)"
echo "Exit code  : ${status}"
echo "========================================"
exit "${status}"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
