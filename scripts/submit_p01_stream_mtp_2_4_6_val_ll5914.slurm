#!/bin/bash
#SBATCH --job-name=p01_stream_val
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=384GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ll5914/logs/p01_stream_mtp_val_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stream_mtp_val_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VAL_ONLY=1
exec bash "${SCRIPT_DIR}/submit_p01_stream_mtp_2_4_6_ll5914.slurm"
