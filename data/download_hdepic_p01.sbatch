#!/bin/bash
#SBATCH -J hdepic_p01_download
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH --mem=16G
#SBATCH --partition=cs
#SBATCH --account=torch_pr_111_general
#SBATCH -o /scratch/ll5914/logs/hdepic_p01_download_%j.out
#SBATCH -e /scratch/ll5914/logs/hdepic_p01_download_%j.err

echo "[INFO] Job started on $(hostname) at $(date)"
echo "[INFO] Downloading HD-EPIC P01 videos..."

SCRIPT="/scratch/ll5914/datasets/HD-EPIC/hd-epic-downloader/hd-epic-downloader.py"
OUTPUT="/scratch/ll5914/datasets/HD-EPIC"

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

# Download only MP4 videos for P01 (participant 1)
cd /scratch/ll5914/datasets/HD-EPIC/hd-epic-downloader
python hd-epic-downloader.py "$OUTPUT" --videos --participants 1

echo "[INFO] Download finished at $(date)"
echo "[INFO] Disk usage:"
du -sh "$OUTPUT/HD-EPIC/P01/" 2>/dev/null || du -sh "$OUTPUT/" 2>/dev/null
