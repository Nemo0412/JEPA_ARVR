#!/bin/bash
#SBATCH --job-name=egtea_download
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5914/logs/egtea_download_%j.out
#SBATCH --error=/scratch/ll5914/logs/egtea_download_%j.err

set -euo pipefail

ROOT="/scratch/ll5914/datasets/EGTEA"
mkdir -p "${ROOT}"/{Videos,Gaze_Data,Action_Annotations}

download() {
    local url="$1"
    local out="$2"
    if [[ -f "${out}" ]] && [[ $(stat -c%s "${out}") -gt 1000000 ]]; then
        echo "[skip] ${out} already exists"
        return 0
    fi
    echo "[download] ${out} ..."
    curl -L --retry 5 --retry-delay 10 -o "${out}" "${url}"
}

# Packaged videos + gaze (from https://github.com/amitsou/EGTEA_Gaze_Plus_Downloader)
download "https://www.dropbox.com/s/xzdw0mloohtxjn5/gtea_videos.zip?dl=1" "${ROOT}/Videos/gtea_videos.zip"
download "https://www.dropbox.com/s/2aryvztw044w9ih/gaze_data.zip?dl=1" "${ROOT}/Gaze_Data/gaze_data.zip"
download "https://www.dropbox.com/s/ksro6eqa6v59859/action_annotation.zip?dl=1" "${ROOT}/Action_Annotations/action_annotation.zip"

echo "[extract] videos ..."
if [[ ! -d "${ROOT}/Videos/extracted" ]]; then
    mkdir -p "${ROOT}/Videos/extracted"
    unzip -q "${ROOT}/Videos/gtea_videos.zip" -d "${ROOT}/Videos/extracted"
fi

echo "[extract] gaze ..."
if [[ ! -d "${ROOT}/gaze_raw/extracted" ]]; then
    mkdir -p "${ROOT}/gaze_raw/extracted"
    unzip -q "${ROOT}/Gaze_Data/gaze_data.zip" -d "${ROOT}/gaze_raw/extracted"
fi

echo "[extract] action annotations ..."
if [[ ! -d "${ROOT}/Action_Annotations/extracted" ]]; then
    unzip -q "${ROOT}/Action_Annotations/action_annotation.zip" -d "${ROOT}/Action_Annotations/extracted"
fi

echo "Download complete: $(date)"
find "${ROOT}/Videos/extracted" -name "*.mp4" 2>/dev/null | wc -l
find "${ROOT}/gaze_raw/extracted" -name "*.txt" 2>/dev/null | wc -l
