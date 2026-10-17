#!/bin/bash
#SBATCH --job-name=rulstm_hdepic_4s2s
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5914/logs/rulstm_hdepic_stream_4s2s_%j.out
#SBATCH --error=/scratch/ll5914/logs/rulstm_hdepic_stream_4s2s_%j.err

# Original RU-LSTM (not a from-scratch LSTM). Delegates to the 4s/+2s job.
set -euo pipefail
bash /home/ll5914/Jepa_baseline/rulstm_hdepic/submit_rulstm_hdepic_stream_4s2s.slurm
