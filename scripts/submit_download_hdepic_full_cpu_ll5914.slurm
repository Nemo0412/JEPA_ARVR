#!/bin/bash
#SBATCH --job-name=hdepic_full_dl
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=cpu_short
#SBATCH --qos=cpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/scratch/ll5914/logs/hdepic_full_dl_%j.out
#SBATCH --error=/scratch/ll5914/logs/hdepic_full_dl_%j.err

set -euo pipefail

# Official HD-EPIC dump for all 9 participants, then clip_split_all + 4 train jobs.
# Skips VRS (1.9 TB raw Aria). Videos + SLAM/gaze + audio/hands/digital-twin.
# Already-downloaded P01 files are skipped by md5.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
PROJECT_ROOT="/home/ll5914/Jepa_yifan/JEPA_ARVR"
DOWNLOADER="/scratch/ll5914/datasets/HD-EPIC/hd-epic-downloader"
DATA_PARENT="/scratch/ll5914/datasets/HD-EPIC"
HD_EPIC_ROOT="${DATA_PARENT}/HD-EPIC"
ANN_PKL="${DATA_PARENT}/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
POOL_DIR="${DATA_PARENT}/hdepic_vjepa_annotations/full_pool"
SPLIT_DIR="${DATA_PARENT}/hdepic_vjepa_annotations/clip_split_all"
LINK_ROOT="${DATA_PARENT}/hdepic_vjepa_videos"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/submit_clip_nopred_probe_128_full_ll5914.slurm"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_download_hdepic_full_cpu_ll5914.slurm"
DONE_FLAG="${DATA_PARENT}/FULL_DOWNLOAD_DONE"
mkdir -p /scratch/ll5914/logs "${POOL_DIR}" "${LINK_ROOT}"

export PYTHONUNBUFFERED=1
_RESUBMITTED=0
DL_PID=""

request_resubmit() {
  if [[ -f "${DONE_FLAG}" || "${_RESUBMITTED}" == "1" ]]; then
    return 0
  fi
  _RESUBMITTED=1
  echo "Auto-resubmitting HD-EPIC download after job ${SLURM_JOB_ID:-}..."
  dep=()
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    dep=(--dependency="afterany:${SLURM_JOB_ID}")
  fi
  sbatch "${dep[@]}" "${SCRIPT_PATH}" || echo "WARN: resubmit failed" >&2
}

on_signal() {
  echo "Caught signal — resubmit download @ $(date)"
  request_resubmit
  if [[ -n "${DL_PID}" ]] && kill -0 "${DL_PID}" 2>/dev/null; then
    kill -TERM "${DL_PID}" 2>/dev/null || true
  fi
}
trap on_signal USR1 TERM

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Already done (${DONE_FLAG}); exiting."
  exit 0
fi

echo "========================================================"
echo "HD-EPIC full download  Job=${SLURM_JOB_ID:-manual}"
echo "Start: $(date)  host=$(hostname)"
echo "Out: ${HD_EPIC_ROOT}"
echo "Skip: VRS (1.9TB). Resume via md5."
echo "========================================================"

# Drop 0-byte leftovers from previous failed urllib opens (do not touch real P01 videos).
find "${HD_EPIC_ROOT}/Videos" -iname '*.mp4' -size 0 -delete 2>/dev/null || true
find "${HD_EPIC_ROOT}" -type f -size 0 -delete 2>/dev/null || true

cd "${DOWNLOADER}"
set +e
"${PYTHON}" hd-epic-downloader.py "${DATA_PARENT}" \
  --videos --slam-gaze --digital-twin --audio --hands \
  --consent-form --acquisition-guidelines &
DL_PID=$!
wait "${DL_PID}"
dl_rc=$?
DL_PID=""
set -e

echo "Downloader exit_code=${dl_rc} at $(date)"
# Failed urllib opens can leave 0-byte placeholders; only count real videos.
n_mp4=$(find "${HD_EPIC_ROOT}/Videos" -iname '*.mp4' -size +10M | wc -l)
echo "Official mp4 count (>10MB): ${n_mp4}"
find "${HD_EPIC_ROOT}/Videos" -iname '*.mp4' -size +10M | awk -F/ '{print $(NF-1)}' | sort | uniq -c || true
echo "Zero-byte mp4 leftovers: $(find "${HD_EPIC_ROOT}/Videos" -iname '*.mp4' -size 0 | wc -l)"

complete=$("${PYTHON}" - <<'PY'
from pathlib import Path
root = Path("/scratch/ll5914/datasets/HD-EPIC/HD-EPIC")
md5p = Path("/scratch/ll5914/datasets/HD-EPIC/hd-epic-downloader/data/md5.txt")
missing = []
for line in md5p.read_text().splitlines():
    parts = line.split()
    if len(parts) < 2:
        continue
    rel = " ".join(parts[1:]).lstrip("./")
    if not rel.startswith("Videos/") or not rel.lower().endswith(".mp4"):
        continue
    p = root / rel
    if not p.exists() or p.stat().st_size < 1_000_000:
        missing.append(rel)
trunc = root / "Videos/P03/P03-20240216-185832.mp4"
if trunc.exists() and trunc.stat().st_size < 3_000_000_000:
    missing.append("Videos/P03/P03-20240216-185832.mp4 (truncated)")
if missing:
    print("INCOMPLETE")
    for m in missing:
        print(m)
else:
    print("COMPLETE")
PY
)
echo "${complete}"
if [[ "${complete}" != COMPLETE* ]]; then
  echo "Download incomplete vs official 156-video list. Resubmit to continue."
  request_resubmit
  exit 0
fi

echo "[prep] Convert all participants -> ${POOL_DIR}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/convert_hdepic_to_vjepa_csv.py" \
  --annotations-pkl "${ANN_PKL}" \
  --video-root "${HD_EPIC_ROOT}" \
  --output-dir "${POOL_DIR}" \
  --link-root "${LINK_ROOT}" \
  --link-method symlink \
  --skip-missing-videos \
  --no-video-probe --fps 30 \
  --split-preset legacy --val-ratio 0.2

echo "[prep] Clip-level 80/20 over P01-P09 -> ${SPLIT_DIR}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/make_hdepic_clip_split_all.py" --pool "${POOL_DIR}" --out "${SPLIT_DIR}" --force

n_vid=$(find "${LINK_ROOT}" -iname '*.mp4' -o -iname '*.MP4' | wc -l)
echo "Linked V-JEPA videos: ${n_vid}"
if [[ "${n_vid}" -lt 140 ]]; then
  echo "WARN: expected ~156 videos, got ${n_vid}. Will resubmit download."
  request_resubmit
  exit 0
fi

echo "full download+split complete" > "${DONE_FLAG}"
date >> "${DONE_FLAG}"
cat "${SPLIT_DIR}/split_stats.json"

echo "[train] Submit 4 full-HD-EPIC nopred probe jobs"
sbatch --job-name=np128f_v2 --export=HORIZON=2,AUX=vanilla "${TRAIN_SCRIPT}"
sbatch --job-name=np128f_v6 --export=HORIZON=6,AUX=vanilla "${TRAIN_SCRIPT}"
sbatch --job-name=np128f_i2 --export=HORIZON=2,AUX=imu "${TRAIN_SCRIPT}"
sbatch --job-name=np128f_i6 --export=HORIZON=6,AUX=imu "${TRAIN_SCRIPT}"
squeue -u ll5914 -o '%.18i %.12P %.16j %.2t %.10M %.20R'
echo "End: $(date)"
