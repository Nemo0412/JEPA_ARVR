#!/bin/bash
#SBATCH --job-name=hdepic_gaze1
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ll5914/logs/hdepic_gaze_first_video_%j.out

set -euo pipefail

# First P01 video (by filename date): 20240202-110250
VID="P01-20240202-110250"
ROOT="/scratch/ll5914/datasets"
SCRIPT_DIR="/scratch/ll5914/datasets/HD-EPIC/hd-epic-downloader"

echo "HD-EPIC slam-gaze (single video gaze package): ${VID}"
echo "Output root: ${ROOT}/HD-EPIC/"
echo "Start: $(date)"
echo ""

cd "${SCRIPT_DIR}"

# slam-gaze only; with --video-id, downloads the matching zip from the manifest (~1 file)
python3 hd-epic-downloader.py "${ROOT}" --slam-gaze --video-id "${VID}"

echo ""
echo "Done: $(date)"
ls -lh "${ROOT}/HD-EPIC/SLAM-and-Gaze/P01/GAZE_HAND/"*${VID}* 2>/dev/null || ls -lR "${ROOT}/HD-EPIC/SLAM-and-Gaze/" 2>/dev/null | head -40
