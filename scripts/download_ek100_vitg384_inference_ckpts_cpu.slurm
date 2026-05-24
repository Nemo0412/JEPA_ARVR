#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__ek100_probe_ckpt
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/download_ek100_probe_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/download_ek100_probe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"
bash scripts/download_ek100_vitg384_inference_ckpts.sh
