#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_gaze_scan
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_gaze_scan_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_gaze_scan_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# CPU-only gaze data discovery. This does not download the 349GB SLAM-and-Gaze
# payload. It updates the official downloader repo, reads its md5 manifest, and
# scans local files/annotations for gaze or gaze-priming artifacts.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RAW_ROOT="${RAW_ROOT:-${DATA_ROOT}/raw/HD-EPIC}"
DOWNLOADER_REPO="${DOWNLOADER_REPO:-${DATA_ROOT}/hd-epic-downloader}"
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-${DATA_ROOT}/hd-epic-annotations}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/hdepic_gaze_inspection}"
GAZE_PARTICIPANTS="${GAZE_PARTICIPANTS:-P01}"

mkdir -p "${PROJECT_ROOT}/logs" "${DATA_ROOT}" "${OUTPUT_DIR}"

export PROJECT_ROOT ENV_NAME DATA_ROOT RAW_ROOT DOWNLOADER_REPO ANNOTATIONS_ROOT OUTPUT_DIR GAZE_PARTICIPANTS

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

mkdir -p "${DATA_ROOT}" "${OUTPUT_DIR}"

if [[ ! -d "${DOWNLOADER_REPO}/.git" ]]; then
    echo "[gaze] Cloning HD-EPIC downloader"
    git clone --depth 1 https://github.com/hd-epic/hd-epic-downloader.git "${DOWNLOADER_REPO}"
else
    echo "[gaze] Updating HD-EPIC downloader"
    git -C "${DOWNLOADER_REPO}" pull --ff-only
fi

if [[ ! -d "${ANNOTATIONS_ROOT}/.git" ]]; then
    echo "[gaze] Cloning HD-EPIC annotations"
    git clone --depth 1 https://github.com/hd-epic/hd-epic-annotations.git "${ANNOTATIONS_ROOT}"
else
    echo "[gaze] Updating HD-EPIC annotations"
    git -C "${ANNOTATIONS_ROOT}" pull --ff-only
fi

read -ra participants <<< "${GAZE_PARTICIPANTS}"

python "${PROJECT_ROOT}/scripts/inspect_hdepic_gaze_data.py" \
    --raw-root "${RAW_ROOT}" \
    --annotations-root "${ANNOTATIONS_ROOT}" \
    --downloader-md5 "${DOWNLOADER_REPO}/data/md5.txt" \
    --participants "${participants[@]}" \
    --output-dir "${OUTPUT_DIR}"

echo "[gaze] Report: ${OUTPUT_DIR}/gaze_data_inspection.md"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
