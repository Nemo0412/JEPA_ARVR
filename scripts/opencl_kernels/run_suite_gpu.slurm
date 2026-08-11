#!/bin/bash
#SBATCH --job-name=ocl_suite_gpu
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ll5914/logs/ocl_suite_gpu_%j.out
#SBATCH --error=/scratch/ll5914/logs/ocl_suite_gpu_%j.err

set -euo pipefail
PYTHON=/scratch/ll5914/conda_envs/jepa_opencl/bin/python
ROOT=/home/ll5914/Jepa_yifan/JEPA_ARVR

echo "=== host $(hostname) ==="
nvidia-smi -L || true
echo "=== OpenCL devices ==="
$PYTHON "$ROOT/scripts/opencl_kernels/suite_bench.py" --list-devices
echo "=== suite_bench (default shapes, prefer GPU) ==="
$PYTHON "$ROOT/scripts/opencl_kernels/suite_bench.py"
echo "=== suite_bench larger N=256 D=256 H=8 ==="
$PYTHON "$ROOT/scripts/opencl_kernels/suite_bench.py" --N 256 --D 256 --H 8 --Nk 64 --K 64 --chunk 32
