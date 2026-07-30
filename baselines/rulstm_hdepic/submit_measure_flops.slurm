#!/bin/bash
#SBATCH --job-name=rulstm_flops
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5914/logs/rulstm_flops_%j.out
#SBATCH --error=/scratch/ll5914/logs/rulstm_flops_%j.err
export TMPDIR=/scratch/ll5914/tmp
mkdir -p /scratch/ll5914/tmp /scratch/ll5914/logs
/scratch/ll5914/conda_envs/SVD/bin/python -u /home/ll5914/Jepa_baseline/rulstm_hdepic/measure_flops.py
