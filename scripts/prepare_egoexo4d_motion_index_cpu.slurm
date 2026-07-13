#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__ref_egoexo_index
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/ref_egoexo_index_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/ref_egoexo_index_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
EGOEXO_ROOT="${EGOEXO_ROOT:-/path/to/egoexo4d_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/egoexo4d_motion_index}"
MOTION_FPS="${MOTION_FPS:-30}"
SELECTED_TAKES="${SELECTED_TAKES:-${EGOEXO_ROOT}/selected_take_uids.txt}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

python3 -m app.ref_egoexo4d_motion_prediction.prepare_index \
  --egoexo-root "${EGOEXO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --motion-fps "${MOTION_FPS}" \
  --selected-takes "${SELECTED_TAKES}"
