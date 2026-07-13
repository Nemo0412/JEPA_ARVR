#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_slam_pose_scan
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_slam_pose_scan_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_slam_pose_scan_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RAW_ROOT="${RAW_ROOT:-${DATA_ROOT}/raw/HD-EPIC}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-${DATA_ROOT}/hdepic_vjepa_annotations}"
SLAM_ROOT="${SLAM_ROOT:-${RAW_ROOT}/SLAM-and-Gaze/P01/SLAM/multi}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/hdepic_slam_inspection}"
SAMPLE_ROWS="${SAMPLE_ROWS:-200}"
INTERFRAME_K_MAX="${INTERFRAME_K_MAX:-128}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

export PROJECT_ROOT ENV_NAME DATA_ROOT RAW_ROOT ANNOTATIONS_DIR SLAM_ROOT OUTPUT_DIR SAMPLE_ROWS INTERFRAME_K_MAX

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/scripts/inspect_hdepic_slam_pose_data.py" \
    --raw-root "${RAW_ROOT}" \
    --annotations-dir "${ANNOTATIONS_DIR}" \
    --slam-root "${SLAM_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --sample-rows "${SAMPLE_ROWS}" \
    --interframe-k-max "${INTERFRAME_K_MAX}"

echo "[slam-pose] Report: ${OUTPUT_DIR}/slam_pose_inspection.md"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
