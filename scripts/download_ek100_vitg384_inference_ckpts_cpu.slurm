#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__ek100_probe_ckpt
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/download_ek100_probe_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/download_ek100_probe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"
bash scripts/download_ek100_vitg384_inference_ckpts.sh
