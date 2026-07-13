#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_refresh_csv
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_refresh_csv_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_refresh_csv_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# CPU-only refresh of HD-EPIC -> V-JEPA/EK100-compatible annotations and link tree.
# Does not download data and does not submit eval.
#
# Examples:
#   sbatch --export=ALL,HDEPIC_SPLIT_PRESET=legacy,INCLUDE_PARTICIPANTS=P01,VAL_PARTICIPANTS=,VAL_RATIO=0.2 scripts/refresh_hdepic_vjepa_annotations.slurm
#   sbatch --export=ALL,INCLUDE_PARTICIPANTS="P01 P02",VAL_PARTICIPANTS=P02 scripts/refresh_hdepic_vjepa_annotations.slurm
#   sbatch --export=ALL,HDEPIC_SPLIT_PRESET=p01_fixed,INCLUDE_PARTICIPANTS=P01,VAL_PARTICIPANTS= scripts/refresh_hdepic_vjepa_annotations.slurm

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RAW_DATA_PARENT="${RAW_DATA_PARENT:-${DATA_ROOT}/raw}"
HD_EPIC_ROOT="${HD_EPIC_ROOT:-${RAW_DATA_PARENT}/HD-EPIC}"
ANNOTATIONS_REPO="${ANNOTATIONS_REPO:-${DATA_ROOT}/hd-epic-annotations}"
ADAPTED_ANNOTATIONS_DIR="${ADAPTED_ANNOTATIONS_DIR:-${DATA_ROOT}/hdepic_vjepa_annotations}"
ADAPTED_VIDEO_ROOT="${ADAPTED_VIDEO_ROOT:-${DATA_ROOT}/hdepic_vjepa_videos}"

INCLUDE_PARTICIPANTS="${INCLUDE_PARTICIPANTS:-P01}"
VAL_PARTICIPANTS="${VAL_PARTICIPANTS-P02}"
VAL_RATIO="${VAL_RATIO:-0.2}"
HDEPIC_SPLIT_PRESET="${HDEPIC_SPLIT_PRESET:-p01_fixed}"
LINK_METHOD="${LINK_METHOD:-symlink}"
CONVERT_FPS="${CONVERT_FPS:-30}"

mkdir -p "${PROJECT_ROOT}/logs" "${ADAPTED_ANNOTATIONS_DIR}" "${ADAPTED_VIDEO_ROOT}"

export PROJECT_ROOT SIF OVERLAY ENV_NAME
export HD_EPIC_ROOT ANNOTATIONS_REPO ADAPTED_ANNOTATIONS_DIR ADAPTED_VIDEO_ROOT
export INCLUDE_PARTICIPANTS VAL_PARTICIPANTS VAL_RATIO HDEPIC_SPLIT_PRESET LINK_METHOD CONVERT_FPS

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

cd "${PROJECT_ROOT}"

ANNOTATIONS_PKL="${ANNOTATIONS_REPO}/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
if [[ ! -f "${ANNOTATIONS_PKL}" ]]; then
    ANNOTATIONS_PKL="$(find "${ANNOTATIONS_REPO}" -name 'HD_EPIC_Narrations.pkl' -print -quit)"
fi
if [[ -z "${ANNOTATIONS_PKL}" || ! -f "${ANNOTATIONS_PKL}" ]]; then
    echo "[refresh] Could not find HD_EPIC_Narrations.pkl under ${ANNOTATIONS_REPO}" >&2
    exit 1
fi

convert_args=(
    --annotations-pkl "${ANNOTATIONS_PKL}"
    --video-root "${HD_EPIC_ROOT}"
    --output-dir "${ADAPTED_ANNOTATIONS_DIR}"
    --link-root "${ADAPTED_VIDEO_ROOT}"
    --link-method "${LINK_METHOD}"
    --skip-missing-videos
    --no-video-probe
    --fps "${CONVERT_FPS}"
    --split-preset "${HDEPIC_SPLIT_PRESET}"
)

if [[ "${HDEPIC_SPLIT_PRESET}" == "legacy" && -n "${VAL_PARTICIPANTS}" ]]; then
    convert_args+=(--val-participants ${VAL_PARTICIPANTS})
elif [[ "${HDEPIC_SPLIT_PRESET}" == "legacy" ]]; then
    convert_args+=(--val-ratio "${VAL_RATIO}")
fi
if [[ -n "${INCLUDE_PARTICIPANTS}" ]]; then
    convert_args+=(--include-participants ${INCLUDE_PARTICIPANTS})
fi

echo "[refresh] INCLUDE_PARTICIPANTS=${INCLUDE_PARTICIPANTS}"
echo "[refresh] HDEPIC_SPLIT_PRESET=${HDEPIC_SPLIT_PRESET}"
if [[ "${HDEPIC_SPLIT_PRESET}" == "legacy" ]]; then
    echo "[refresh] VAL_PARTICIPANTS=${VAL_PARTICIPANTS:-<ratio split>}"
else
    echo "[refresh] fixed split preset active; VAL_PARTICIPANTS/VAL_RATIO ignored"
fi
python "${PROJECT_ROOT}/scripts/convert_hdepic_to_vjepa_csv.py" "${convert_args[@]}"

echo "[refresh] Train rows:"
wc -l "${ADAPTED_ANNOTATIONS_DIR}/HD_EPIC_train_vjepa.csv"
echo "[refresh] Val rows:"
wc -l "${ADAPTED_ANNOTATIONS_DIR}/HD_EPIC_val_vjepa.csv"
if [[ -f "${ADAPTED_ANNOTATIONS_DIR}/HD_EPIC_test_vjepa.csv" ]]; then
    echo "[refresh] Test rows:"
    wc -l "${ADAPTED_ANNOTATIONS_DIR}/HD_EPIC_test_vjepa.csv"
fi
echo "[refresh] Stats:"
python - <<'PY'
import json
import os
from pathlib import Path

stats_path = Path(os.environ["ADAPTED_ANNOTATIONS_DIR"]) / "conversion_stats.json"
stats = json.loads(stats_path.read_text())
for key in [
    "split_preset",
    "split_policy",
    "annotation_rows",
    "converted_rows",
    "missing_videos",
    "unique_videos",
    "unique_verbs",
    "unique_nouns",
    "unique_actions",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_videos",
    "val_videos",
    "test_videos",
    "unmatched_rows",
    "missing_split_video_ids",
]:
    value = stats.get(key)
    if isinstance(value, list):
        value = len(value)
    print(f"{key}: {value}")
PY
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
