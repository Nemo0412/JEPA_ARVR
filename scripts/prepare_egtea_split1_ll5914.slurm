#!/bin/bash
#SBATCH --job-name=egtea_prep
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ll5914/logs/egtea_prep_%j.out
#SBATCH --error=/scratch/ll5914/logs/egtea_prep_%j.err

set -euo pipefail

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
SCRIPT="/home/ll5914/Jepa_yifan/JEPA_ARVR/scripts/prepare_egtea_split1.py"

echo "Preparing EGTEA split1: $(date)"
"${PYTHON}" "${SCRIPT}" --egtea-root /scratch/ll5914/datasets/EGTEA
echo "Done: $(date)"
