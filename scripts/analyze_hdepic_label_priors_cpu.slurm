#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_priors
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_label_priors_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/hdepic_label_priors_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/outputs/hdepic_label_priors/current_label_priors.json}"

mkdir -p "${PROJECT_ROOT}/logs" "$(dirname "${OUTPUT}")"

export PROJECT_ROOT SIF OVERLAY ENV_NAME ANNOTATIONS_DIR OUTPUT

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

python "${PROJECT_ROOT}/scripts/analyze_hdepic_label_priors.py" \
  --train-csv "${ANNOTATIONS_DIR}/HD_EPIC_train_vjepa.csv" \
  --val-csv "${ANNOTATIONS_DIR}/HD_EPIC_val_vjepa.csv" \
  --output "${OUTPUT}"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
