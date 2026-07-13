#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_gaze_val
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_gaze_valtree_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_gaze_valtree_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# CPU-only prep for gaze-enhanced val-only inference.
# Downloads SLAM/Gaze files for validation videos, builds gaze overlays, then
# creates a V-JEPA-compatible video tree that swaps in available overlays.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RAW_DATA_PARENT="${RAW_DATA_PARENT:-${DATA_ROOT}/raw}"
RAW_ROOT="${RAW_ROOT:-${RAW_DATA_PARENT}/HD-EPIC}"
DOWNLOADER_REPO="${DOWNLOADER_REPO:-${DATA_ROOT}/hd-epic-downloader}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-${DATA_ROOT}/hdepic_vjepa_annotations}"
VAL_CSV="${VAL_CSV:-${ANNOTATIONS_DIR}/HD_EPIC_val_vjepa.csv}"
SOURCE_VIDEO_ROOT="${SOURCE_VIDEO_ROOT:-${DATA_ROOT}/hdepic_vjepa_videos}"

SIGMA="${SIGMA:-96}"
GAZE_MAP_TYPE="${GAZE_MAP_TYPE:-gaussian}"
BINARY_RADIUS="${BINARY_RADIUS:-${SIGMA}}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.28}"
VAL_LIMIT="${VAL_LIMIT:-0}"
if [[ "${GAZE_MAP_TYPE}" == "binary" ]]; then
  RADIUS_TAG="${BINARY_RADIUS//./p}"
  OVERLAY_OUTPUT_DIR="${OVERLAY_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_gaze_heatmaps/val_binary_radius_${RADIUS_TAG}}"
  GAZE_VIDEO_ROOT="${GAZE_VIDEO_ROOT:-${DATA_ROOT}/hdepic_vjepa_videos_gaze_binary_radius${RADIUS_TAG}}"
else
  OVERLAY_OUTPUT_DIR="${OVERLAY_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_gaze_heatmaps/val_sigma_${SIGMA}}"
  GAZE_VIDEO_ROOT="${GAZE_VIDEO_ROOT:-${DATA_ROOT}/hdepic_vjepa_videos_gaze_sigma${SIGMA}}"
fi
REPORT_DIR="${REPORT_DIR:-${PROJECT_ROOT}/outputs/hdepic_gaze_val_prep}"
MAX_OVERLAY_FRAMES="${MAX_OVERLAY_FRAMES:-0}"
FALLBACK_FOV_DEG="${FALLBACK_FOV_DEG:-90}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${PROJECT_ROOT}/logs" "${REPORT_DIR}" "${OVERLAY_OUTPUT_DIR}"

export PROJECT_ROOT SIF OVERLAY ENV_NAME DATA_ROOT RAW_DATA_PARENT RAW_ROOT DOWNLOADER_REPO
export VAL_CSV SOURCE_VIDEO_ROOT SIGMA GAZE_MAP_TYPE BINARY_RADIUS OVERLAY_ALPHA VAL_LIMIT OVERLAY_OUTPUT_DIR GAZE_VIDEO_ROOT REPORT_DIR
export MAX_OVERLAY_FRAMES FALLBACK_FOV_DEG DRY_RUN

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

if [[ ! -d "${DOWNLOADER_REPO}/.git" ]]; then
  echo "[gaze-val] Cloning HD-EPIC downloader"
  git clone --depth 1 https://github.com/hd-epic/hd-epic-downloader.git "${DOWNLOADER_REPO}"
else
  echo "[gaze-val] Updating HD-EPIC downloader"
  git -C "${DOWNLOADER_REPO}" pull --ff-only
fi

ids_file="${REPORT_DIR}/val_video_ids.txt"
python - <<'PY'
import os
from pathlib import Path
import pandas as pd

val_csv = Path(os.environ["VAL_CSV"])
limit = int(os.environ["VAL_LIMIT"])
df = pd.read_csv(val_csv)
col = "original_video_id" if "original_video_id" in df.columns else "video_id"
ids = []
for raw in df[col].dropna().astype(str):
    vid = raw.replace("_", "-")
    if vid not in ids:
        ids.append(vid)
if limit > 0:
    ids = ids[:limit]
out = Path(os.environ["REPORT_DIR"]) / "val_video_ids.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(ids) + "\n", encoding="utf-8")
print(f"[gaze-val] selected {len(ids)} val video id(s)")
for vid in ids:
    print(f"[gaze-val]   {vid}")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[gaze-val] DRY_RUN=1; skipping download/overlay/tree build"
  exit 0
fi

while IFS= read -r video_id; do
  [[ -z "${video_id}" ]] && continue
  participant="${video_id%%-*}"
  echo "[gaze-val] Downloading gaze for ${video_id}"
  (
    cd "${DOWNLOADER_REPO}"
    python hd-epic-downloader.py "${RAW_DATA_PARENT}" --slam-gaze --video-id "${video_id}"
  )

  gaze_zip="$(find "${RAW_ROOT}/SLAM-and-Gaze" -name "mps_${video_id}_vrs.zip" -print -quit)"
  video_path="${RAW_ROOT}/Videos/${participant}/${video_id}.mp4"
  if [[ -z "${gaze_zip}" ]]; then
    echo "[gaze-val] skip overlay, missing gaze zip for ${video_id}" >&2
    continue
  fi
  if [[ ! -f "${video_path}" ]]; then
    echo "[gaze-val] skip overlay, missing video for ${video_id}: ${video_path}" >&2
    continue
  fi

  echo "[gaze-val] Building overlay for ${video_id} map=${GAZE_MAP_TYPE} sigma=${SIGMA} binary_radius=${BINARY_RADIUS}"
  overlay_args=(
    "${PROJECT_ROOT}/scripts/build_hdepic_gaze_heatmap_video.py"
    --gaze-zip "${gaze_zip}"
    --video "${video_path}"
    --output-dir "${OVERLAY_OUTPUT_DIR}"
    --video-id "${video_id}"
    --fallback-fov-deg "${FALLBACK_FOV_DEG}"
    --map-type "${GAZE_MAP_TYPE}"
    --sigma "${SIGMA}"
    --overlay-alpha "${OVERLAY_ALPHA}"
  )
  if [[ "${GAZE_MAP_TYPE}" == "binary" ]]; then
    overlay_args+=(--binary-radius "${BINARY_RADIUS}")
  fi
  overlay_args+=(--max-overlay-frames "${MAX_OVERLAY_FRAMES}")
  python "${overlay_args[@]}"
done < "${ids_file}"

python "${PROJECT_ROOT}/scripts/create_hdepic_gaze_video_tree.py" \
  --source-video-root "${SOURCE_VIDEO_ROOT}" \
  --overlay-dir "${OVERLAY_OUTPUT_DIR}" \
  --output-video-root "${GAZE_VIDEO_ROOT}" \
  --report "${REPORT_DIR}/gaze_video_tree_${GAZE_MAP_TYPE}_${SIGMA}.json" \
  --replace-existing

echo "[gaze-val] Gaze video root: ${GAZE_VIDEO_ROOT}"
BASH
)

singularity exec \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
