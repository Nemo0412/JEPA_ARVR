#!/bin/bash
#SBATCH --job-name=p01_stream_vid_val
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/ll5914/logs/p01_stream_video_val_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stream_video_val_%j.err

set -euo pipefail

# Val / inference for video-only streaming 2/4/6 (no MTP).
export VAL_ONLY=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/submit_p01_stream_video_2_4_6_ll5914.slurm"
