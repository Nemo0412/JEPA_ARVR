#!/bin/bash
#SBATCH --job-name=ocl_trimodal
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/ll5914/logs/ocl_system_%j.out
#SBATCH --error=/scratch/ll5914/logs/ocl_system_%j.err

set -euo pipefail
PY=/scratch/ll5914/conda_envs/jepa_opencl/bin/python
D=/home/ll5914/Jepa_yifan/JEPA_ARVR/scripts/opencl_kernels
echo "=== $(hostname) ==="; nvidia-smi -L || true
echo "=== tri-modal smoke Nv=64 Ng=16 Ni=8 ==="
$PY $D/profile_system_pipeline.py --Nv 64 --Ng 16 --Ni 8 --D 64 --H 4 --layers 1 \
  --out $D/profile_system_gpu_N64.json
echo "=== tri-modal ViT-L token budgets Nv=256 Ng=100 Ni=26 ==="
$PY $D/profile_system_pipeline.py --vitl-tokens --D 64 --H 4 --layers 1 \
  --out $D/profile_system_gpu_N256.json
