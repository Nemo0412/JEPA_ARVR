#!/bin/bash
#SBATCH --job-name=hdepic_p03_dl
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=cpu_short
#SBATCH --qos=cpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/scratch/ll5914/logs/hdepic_p03_dl_%j.out
#SBATCH --error=/scratch/ll5914/logs/hdepic_p03_dl_%j.err

set -euo pipefail

# Patch the 4 incomplete P03 videos, rebuild clip_split_all, do not re-submit
# the four np128f jobs (they are already queued). Release held full-train jobs
# and resubmit the failed RGB+6s job once the dump is complete.

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
SCRIPT_PATH="${PROJECT_ROOT}/scripts/submit_download_hdepic_missing_cpu_ll5914.slurm"
DONE_FLAG="${DATA_PARENT}/FULL_DOWNLOAD_DONE"
HELD_FULL_JOBS="${HELD_FULL_JOBS:-18190550,18187173,18187174}"
VIDEO_IDS="P03-20240216-205923,P03-20240218-190556,P03-20240218-202148,P03-20240216-185832"
mkdir -p /scratch/ll5914/logs "${POOL_DIR}" "${LINK_ROOT}"

export PYTHONUNBUFFERED=1
_RESUBMITTED=0
DL_PID=""

request_resubmit() {
  if [[ "${_RESUBMITTED}" == "1" ]]; then
    return 0
  fi
  _RESUBMITTED=1
  echo "Auto-resubmitting P03 patch download after job ${SLURM_JOB_ID:-}..."
  dep=()
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    dep=(--dependency="afterany:${SLURM_JOB_ID}")
  fi
  sbatch --export=HELD_FULL_JOBS="${HELD_FULL_JOBS}" "${dep[@]}" "${SCRIPT_PATH}" \
    || echo "WARN: resubmit failed" >&2
}

on_signal() {
  echo "Caught signal — resubmit P03 patch download @ $(date)"
  request_resubmit
  if [[ -n "${DL_PID}" ]] && kill -0 "${DL_PID}" 2>/dev/null; then
    kill -TERM "${DL_PID}" 2>/dev/null || true
  fi
}
trap on_signal USR1 TERM

echo "========================================================"
echo "HD-EPIC P03 patch download  Job=${SLURM_JOB_ID:-manual}"
echo "Start: $(date)  host=$(hostname)"
echo "Videos: ${VIDEO_IDS}"
echo "========================================================"

# Truncated 185832 fails md5; delete so we do not hash 2.5GB then rewrite it.
trunc="${HD_EPIC_ROOT}/Videos/P03/P03-20240216-185832.mp4"
if [[ -f "${trunc}" ]]; then
  sz=$(stat -c%s "${trunc}")
  if [[ "${sz}" -lt 3000000000 ]]; then
    echo "Removing truncated ${trunc} (${sz} bytes)"
    rm -f "${trunc}"
  fi
fi
find "${HD_EPIC_ROOT}/Videos/P03" -iname '*.mp4' -size 0 -delete 2>/dev/null || true

cd "${DOWNLOADER}"
set +e
"${PYTHON}" hd-epic-downloader.py "${DATA_PARENT}" \
  --videos --video-id "${VIDEO_IDS}" &
DL_PID=$!
wait "${DL_PID}"
dl_rc=$?
DL_PID=""
set -e
echo "Downloader exit_code=${dl_rc} at $(date)"

complete=$("${PYTHON}" - <<'PY'
from pathlib import Path
root = Path("/scratch/ll5914/datasets/HD-EPIC/HD-EPIC")
md5p = Path("/scratch/ll5914/datasets/HD-EPIC/hd-epic-downloader/data/md5.txt")
missing = []
n = 0
for line in md5p.read_text().splitlines():
    parts = line.split()
    if len(parts) < 2:
        continue
    rel = " ".join(parts[1:]).lstrip("./")
    if not rel.startswith("Videos/") or not rel.lower().endswith(".mp4"):
        continue
    n += 1
    p = root / rel
    if not p.exists() or p.stat().st_size < 1_000_000:
        missing.append(f"{rel} size={p.stat().st_size if p.exists() else 0}")
trunc = root / "Videos/P03/P03-20240216-185832.mp4"
if trunc.exists() and trunc.stat().st_size < 3_000_000_000:
    missing.append(f"Videos/P03/P03-20240216-185832.mp4 truncated={trunc.stat().st_size}")
print(f"official={n} missing={len(missing)}")
for m in missing:
    print(m)
print("COMPLETE" if not missing else "INCOMPLETE")
PY
)
echo "${complete}"
if ! grep -q '^COMPLETE$' <<<"${complete}"; then
  echo "Still incomplete. Resubmit to continue."
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
n_ok=0
for f in "${LINK_ROOT}"/P0*/*.[mM][pP]4; do
  [[ -e "${f}" ]] || continue
  n_ok=$((n_ok + 1))
done
echo "Linked V-JEPA videos: ${n_vid} resolvable=${n_ok}"
if [[ "${n_ok}" -lt 156 ]]; then
  echo "WARN: expected 156 resolvable videos, got ${n_ok}. Resubmit."
  request_resubmit
  exit 0
fi

echo "full download+split complete" > "${DONE_FLAG}"
date >> "${DONE_FLAG}"
echo "patched P03 missing/truncated videos" >> "${DONE_FLAG}"
cat "${SPLIT_DIR}/split_stats.json"

echo "[train] Submit 4 full-HD-EPIC nopred probe jobs if not already queued"
queued="$(squeue -u ll5914 -h -n np128f_v2,np128f_v6,np128f_i2,np128f_i6 -o '%j' | tr '\n' ' ')"
echo "already queued: ${queued:-none}"
if [[ " ${queued} " != *" np128f_v2 "* ]]; then
  sbatch --job-name=np128f_v2 --export=HORIZON=2,AUX=vanilla "${TRAIN_SCRIPT}"
fi
if [[ " ${queued} " != *" np128f_v6 "* ]]; then
  sbatch --job-name=np128f_v6 --export=HORIZON=6,AUX=vanilla "${TRAIN_SCRIPT}"
fi
if [[ " ${queued} " != *" np128f_i2 "* ]]; then
  sbatch --job-name=np128f_i2 --export=HORIZON=2,AUX=imu "${TRAIN_SCRIPT}"
fi
if [[ " ${queued} " != *" np128f_i6 "* ]]; then
  sbatch --job-name=np128f_i6 --export=HORIZON=6,AUX=imu "${TRAIN_SCRIPT}"
fi
squeue -u ll5914 -o '%.18i %.12P %.16j %.2t %.10M %.20R'
echo "End: $(date)"
