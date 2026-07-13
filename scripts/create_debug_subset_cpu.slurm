#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__create_debug_subset
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/create_debug_subset_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/create_debug_subset_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

# One-time CPU job: sample a fixed deterministic P01 debug subset from the
# HD-EPIC annotation CSVs and write configs/debug_subset_p01.json.
#
# After this job completes, rsync/scp the JSON back and commit it:
#   scp greene:$PROJECT_ROOT/configs/debug_subset_p01.json configs/
#   git add configs/debug_subset_p01.json
#   git commit -m "fix: populate P01 debug subset (seed=42)"
#
# Usage (from login node, no Python needed):
#   sbatch scripts/create_debug_subset_cpu.slurm

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/configs/debug_subset_p01.json}"
NUM_VIDEOS="${NUM_VIDEOS:-6}"
SEED="${SEED:-42}"

mkdir -p "${PROJECT_ROOT}/logs" "$(dirname "${OUTPUT}")"

export PROJECT_ROOT SIF OVERLAY ENV_NAME ANNOTATIONS_DIR OUTPUT NUM_VIDEOS SEED

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/scripts/create_debug_subset.py" \
    --train-csv  "${ANNOTATIONS_DIR}/HD_EPIC_train_vjepa.csv" \
    --val-csv    "${ANNOTATIONS_DIR}/HD_EPIC_val_vjepa.csv" \
    --output     "${OUTPUT}" \
    --num-videos "${NUM_VIDEOS}" \
    --seed       "${SEED}"

echo "[debug_subset] Written to ${OUTPUT}"
echo "[debug_subset] Next step: scp this file back and commit it into configs/"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
