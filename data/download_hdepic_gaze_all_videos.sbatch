#!/bin/bash
#SBATCH --job-name=hdepic_gaze_all
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ll5914/logs/hdepic_gaze_all_%j.out

set -euo pipefail

ROOT="/scratch/ll5914/datasets"
SCRIPT_DIR="${ROOT}/HD-EPIC/hd-epic-downloader"
GAZE_DIR="${ROOT}/HD-EPIC/SLAM-and-Gaze/P01/GAZE_HAND"
EXTRACT_DIR="${ROOT}/HD-EPIC/_gaze_extract"

# All 27 P01 video IDs (downloader skips already-downloaded files)
VIDEO_IDS=(
    P01-20240202-110250
    P01-20240202-161354
    P01-20240202-161948
    P01-20240202-171220
    P01-20240202-175627
    P01-20240202-195538
    P01-20240203-093333
    P01-20240203-121517
    P01-20240203-123350
    P01-20240203-130505
    P01-20240203-132119
    P01-20240203-135502
    P01-20240203-150506
    P01-20240203-152323
    P01-20240203-152956
    P01-20240203-161757
    P01-20240203-184045
    P01-20240203-184214
    P01-20240204-095114
    P01-20240204-120411
    P01-20240204-121042
    P01-20240204-124504
    P01-20240204-130448
    P01-20240204-142301
    P01-20240204-145458
    P01-20240204-152537
    P01-20240204-160230
)

# Build comma-separated list for the downloader
IDS_CSV=$(IFS=,; echo "${VIDEO_IDS[*]}")

echo "Downloading gaze packages for ${#VIDEO_IDS[@]} videos..."
echo "Start: $(date)"
echo ""

cd "${SCRIPT_DIR}"
python3 hd-epic-downloader.py "${ROOT}" --slam-gaze --video-id "${IDS_CSV}"

echo ""
echo "Download done: $(date)"
echo "Zip files present:"
ls -lh "${GAZE_DIR}/" | grep -v "^total"

# ── Extract each zip to _gaze_extract/<name>/ ─────────────────────────
mkdir -p "${EXTRACT_DIR}"

echo ""
echo "Extracting zips..."
for VID in "${VIDEO_IDS[@]}"; do
    ZIP="${GAZE_DIR}/mps_${VID}_vrs.zip"
    DEST="${EXTRACT_DIR}/mps_${VID}_vrs"
    if [[ -f "${ZIP}" ]]; then
        if [[ -d "${DEST}" ]]; then
            echo "  [skip] already extracted: ${VID}"
        else
            echo "  Extracting ${VID}..."
            unzip -q "${ZIP}" -d "${EXTRACT_DIR}"
        fi
    else
        echo "  [WARN] zip not found: ${ZIP}"
    fi
done

echo ""
echo "Extraction done: $(date)"
echo "Extracted directories:"
ls "${EXTRACT_DIR}/"
echo ""
echo "Verifying general_eye_gaze.csv presence:"
MISSING=0
for VID in "${VIDEO_IDS[@]}"; do
    CSV="${EXTRACT_DIR}/mps_${VID}_vrs/eye_gaze/general_eye_gaze.csv"
    if [[ -f "${CSV}" ]]; then
        echo "  OK  ${VID}"
    else
        echo "  MISSING  ${VID}"
        MISSING=$((MISSING + 1))
    fi
done
echo ""
echo "Summary: ${#VIDEO_IDS[@]} videos, ${MISSING} missing gaze CSV files."
echo "All done: $(date)"
