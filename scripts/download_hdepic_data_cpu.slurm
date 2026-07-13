#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_download
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_download_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# CPU-only, resumable HD-EPIC downloader.
#
# Submit from the HPC login node:
#   sbatch scripts/download_hdepic_data_cpu.slurm
#
# Optional subset:
#   sbatch --export=ALL,DOWNLOAD_PARTICIPANTS=1,2,3 scripts/download_hdepic_data_cpu.slurm

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RAW_DATA_PARENT="${RAW_DATA_PARENT:-${DATA_ROOT}/raw}"
DOWNLOADER_REPO="${DOWNLOADER_REPO:-${DATA_ROOT}/hd-epic-downloader}"
DOWNLOAD_PARTICIPANTS="${DOWNLOAD_PARTICIPANTS:-}"

mkdir -p "${PROJECT_ROOT}/logs" "${DATA_ROOT}" "${RAW_DATA_PARENT}"

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID:-not-in-slurm}"
echo "Node       : $(hostname)"
echo "Start      : $(date)"
echo "Project    : ${PROJECT_ROOT}"
echo "Output root: ${RAW_DATA_PARENT}"
echo "Participants: ${DOWNLOAD_PARTICIPANTS:-all}"
echo "========================================"

if [[ ! -f "${SIF}" ]]; then
    echo "Missing Singularity image: ${SIF}" >&2
    exit 1
fi
if [[ ! -f "${OVERLAY}" ]]; then
    echo "Missing overlay: ${OVERLAY}" >&2
    exit 1
fi

export PROJECT_ROOT DATA_ROOT RAW_DATA_PARENT DOWNLOADER_REPO DOWNLOAD_PARTICIPANTS ENV_NAME

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

mkdir -p "${DATA_ROOT}" "${RAW_DATA_PARENT}"

if [[ ! -d "${DOWNLOADER_REPO}/.git" ]]; then
    echo "[download] Cloning HD-EPIC downloader"
    git clone --depth 1 https://github.com/hd-epic/hd-epic-downloader.git "${DOWNLOADER_REPO}"
else
    echo "[download] Updating HD-EPIC downloader"
    git -C "${DOWNLOADER_REPO}" pull --ff-only
fi

downloader_args=("${RAW_DATA_PARENT}" "--videos")
if [[ -n "${DOWNLOAD_PARTICIPANTS}" ]]; then
    downloader_args+=("--participants" "${DOWNLOAD_PARTICIPANTS}")
fi

echo "[download] Running official HD-EPIC downloader"
(
    cd "${DOWNLOADER_REPO}"
    python hd-epic-downloader.py "${downloader_args[@]}"
)

echo "[download] Done"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"

echo "========================================"
echo "End        : $(date)"
echo "========================================"
