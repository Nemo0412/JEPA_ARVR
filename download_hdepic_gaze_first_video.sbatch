#!/bin/bash
#SBATCH --job-name=hdepic_gaze1
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ll5914/logs/hdepic_gaze_first_video_%j.out

set -euo pipefail

# P01 第一个视频（按文件名日期）：20240202-110250
VID="P01-20240202-110250"
ROOT="/scratch/ll5914/datasets"
SCRIPT_DIR="/scratch/ll5914/datasets/HD-EPIC/hd-epic-downloader"

echo "HD-EPIC slam-gaze（单视频 gaze 包）: ${VID}"
echo "输出根目�? ${ROOT}/HD-EPIC/"
echo "Start: $(date)"
echo ""

cd "${SCRIPT_DIR}"

# �?slam-gaze�?-video-id �?manifest 里与�?id 匹配的文件（�?1 �?zip�?
python3 hd-epic-downloader.py "${ROOT}" --slam-gaze --video-id "${VID}"

echo ""
echo "Done: $(date)"
ls -lh "${ROOT}/HD-EPIC/SLAM-and-Gaze/P01/GAZE_HAND/"*${VID}* 2>/dev/null || ls -lR "${ROOT}/HD-EPIC/SLAM-and-Gaze/" 2>/dev/null | head -40
