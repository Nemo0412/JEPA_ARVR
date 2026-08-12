#!/bin/bash
#SBATCH --job-name=ocl_breakd
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/ll5914/logs/ocl_breakdown_%j.out
#SBATCH --error=/scratch/ll5914/logs/ocl_breakdown_%j.err

set -euo pipefail
PYTHON=/scratch/ll5914/conda_envs/jepa_opencl/bin/python
ROOT=/home/ll5914/Jepa_yifan/JEPA_ARVR
OUTDIR=$ROOT/scripts/opencl_kernels

echo "=== host $(hostname) ==="
nvidia-smi -L || true
echo "=== latency breakdown N=64 D=64 (GPU) ==="
$PYTHON $OUTDIR/profile_latency_breakdown.py --N 64 --D 64 --H 4 \
  --out $OUTDIR/profile_breakdown_gpu_N64.json
echo "=== latency breakdown N=256 D=256 (GPU) ==="
$PYTHON $OUTDIR/profile_latency_breakdown.py --N 256 --D 256 --H 8 --Nk 64 --K 64 --chunk 32 \
  --out $OUTDIR/profile_breakdown_gpu_N256.json
