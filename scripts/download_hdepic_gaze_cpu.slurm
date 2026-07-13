#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_gaze_dl
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_gaze_download_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_gaze_download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# CPU-only, controlled HD-EPIC SLAM-and-Gaze downloader.
#
# Default: download the P01 video used by the reference gaze prototype.
# This avoids accidentally downloading the full SLAM-and-Gaze component
# (~349GB). Use GAZE_VIDEO_IDS for precise files, or clear it and set
# GAZE_PARTICIPANTS to download participant-level SLAM-and-Gaze data.
#
# Examples:
#   sbatch scripts/download_hdepic_gaze_cpu.slurm
#   sbatch --export=ALL,GAZE_VIDEO_IDS=P01-20240202-110250 scripts/download_hdepic_gaze_cpu.slurm
#   sbatch --export=ALL,GAZE_VIDEO_IDS=P01-20240202-110250,P01-20240203-184214 scripts/download_hdepic_gaze_cpu.slurm
#   sbatch --export=ALL,GAZE_VIDEO_IDS=,GAZE_PARTICIPANTS=1 scripts/download_hdepic_gaze_cpu.slurm
#   sbatch --export=ALL,DRY_RUN=1,GAZE_VIDEO_IDS=,GAZE_PARTICIPANTS=1 scripts/download_hdepic_gaze_cpu.slurm

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RAW_DATA_PARENT="${RAW_DATA_PARENT:-${DATA_ROOT}/raw}"
RAW_ROOT="${RAW_ROOT:-${RAW_DATA_PARENT}/HD-EPIC}"
DOWNLOADER_REPO="${DOWNLOADER_REPO:-${DATA_ROOT}/hd-epic-downloader}"
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-${DATA_ROOT}/hd-epic-annotations}"
INSPECT_OUTPUT_DIR="${INSPECT_OUTPUT_DIR:-${DATA_ROOT}/hdepic_gaze_inspection}"

# Use "-" rather than ":-" so callers can intentionally pass
# GAZE_VIDEO_IDS= to switch from single-video mode to participant mode.
GAZE_VIDEO_IDS="${GAZE_VIDEO_IDS-P01-20240202-110250}"
GAZE_PARTICIPANTS="${GAZE_PARTICIPANTS-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${PROJECT_ROOT}/logs" "${DATA_ROOT}" "${RAW_DATA_PARENT}" "${INSPECT_OUTPUT_DIR}"

echo "========================================"
echo "Mode       : HD-EPIC SLAM-and-Gaze download"
echo "Job ID     : ${SLURM_JOB_ID:-not-in-slurm}"
echo "Node       : $(hostname)"
echo "Start      : $(date)"
echo "Project    : ${PROJECT_ROOT}"
echo "Output root: ${RAW_DATA_PARENT}"
echo "Video IDs  : ${GAZE_VIDEO_IDS:-<none>}"
echo "Participants: ${GAZE_PARTICIPANTS:-<none>}"
echo "Dry run    : ${DRY_RUN}"
echo "========================================"

if [[ ! -f "${SIF}" ]]; then
    echo "Missing Singularity image: ${SIF}" >&2
    exit 1
fi
if [[ ! -f "${OVERLAY}" ]]; then
    echo "Missing overlay: ${OVERLAY}" >&2
    exit 1
fi

export PROJECT_ROOT DATA_ROOT RAW_DATA_PARENT RAW_ROOT DOWNLOADER_REPO ANNOTATIONS_ROOT INSPECT_OUTPUT_DIR
export GAZE_VIDEO_IDS GAZE_PARTICIPANTS DRY_RUN ENV_NAME

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

mkdir -p "${DATA_ROOT}" "${RAW_DATA_PARENT}" "${INSPECT_OUTPUT_DIR}"

if [[ ! -d "${DOWNLOADER_REPO}/.git" ]]; then
    echo "[gaze-download] Cloning HD-EPIC downloader"
    git clone --depth 1 https://github.com/hd-epic/hd-epic-downloader.git "${DOWNLOADER_REPO}"
else
    echo "[gaze-download] Updating HD-EPIC downloader"
    git -C "${DOWNLOADER_REPO}" pull --ff-only
fi

if [[ ! -d "${ANNOTATIONS_ROOT}/.git" ]]; then
    echo "[gaze-download] Cloning HD-EPIC annotations"
    git clone --depth 1 https://github.com/hd-epic/hd-epic-annotations.git "${ANNOTATIONS_ROOT}"
else
    echo "[gaze-download] Updating HD-EPIC annotations"
    git -C "${ANNOTATIONS_ROOT}" pull --ff-only
fi

downloader_args=("${RAW_DATA_PARENT}" "--slam-gaze")
if [[ -n "${GAZE_VIDEO_IDS}" ]]; then
    downloader_args+=("--video-id" "${GAZE_VIDEO_IDS}")
else
    downloader_args+=("--participants" "${GAZE_PARTICIPANTS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    downloader_args+=("--dry-run")
fi

echo "[gaze-download] Running official downloader:"
printf '  %q' python hd-epic-downloader.py "${downloader_args[@]}"
printf '\n'

(
    cd "${DOWNLOADER_REPO}"
    python hd-epic-downloader.py "${downloader_args[@]}"
)

echo "[gaze-download] Download step complete; scanning local gaze files"
participants=()
if [[ -n "${GAZE_VIDEO_IDS}" ]]; then
    IFS=',' read -ra vids <<< "${GAZE_VIDEO_IDS}"
    for vid in "${vids[@]}"; do
        participants+=("${vid:0:3}")
    done
else
    IFS=',' read -ra nums <<< "${GAZE_PARTICIPANTS}"
    for n in "${nums[@]}"; do
        if [[ "${n}" == P* ]]; then
            participants+=("${n}")
        else
            participants+=("P$(printf '%02d' "${n}")")
        fi
    done
fi

python "${PROJECT_ROOT}/scripts/inspect_hdepic_gaze_data.py" \
    --raw-root "${RAW_ROOT}" \
    --annotations-root "${ANNOTATIONS_ROOT}" \
    --downloader-md5 "${DOWNLOADER_REPO}/data/md5.txt" \
    --participants "${participants[@]}" \
    --output-dir "${INSPECT_OUTPUT_DIR}"

echo "[gaze-download] Report: ${INSPECT_OUTPUT_DIR}/gaze_data_inspection.md"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"

echo "========================================"
echo "End        : $(date)"
echo "========================================"
