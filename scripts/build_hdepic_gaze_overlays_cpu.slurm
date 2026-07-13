#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_gaze_batch
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_gaze_overlays_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_gaze_overlays_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# CPU-only batch builder for the simplest gaze-enhanced video prototype:
# overlay available gaze heat maps on their matching HD-EPIC videos.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"
RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/data/raw/HD-EPIC}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hdepic_gaze_heatmaps}"
SOURCE_VIDEO_ROOT="${SOURCE_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos}"
CREATE_VIDEO_TREE="${CREATE_VIDEO_TREE:-0}"
GAZE_VIDEO_ROOT="${GAZE_VIDEO_ROOT:-}"
REPORT_DIR="${REPORT_DIR:-${PROJECT_ROOT}/outputs/hdepic_gaze_tree_prep}"
MAX_OVERLAY_FRAMES="${MAX_OVERLAY_FRAMES:-300}"
FALLBACK_FOV_DEG="${FALLBACK_FOV_DEG:-90}"
GAZE_MAP_TYPE="${GAZE_MAP_TYPE:-gaussian}"
BINARY_RADIUS="${BINARY_RADIUS:-}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.28}"
# Use whitespace, not commas, when overriding through sbatch --export.
# Slurm treats commas as --export separators, so SIGMAS=18,48,96 arrives as
# only SIGMAS=18. The parser below still accepts commas if SIGMAS is set inside
# an already-running shell, but HPC launch examples should use spaces.
SIGMAS="${SIGMAS:-18 48 96}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}" "${REPORT_DIR}"

export PROJECT_ROOT SIF OVERLAY ENV_NAME RAW_ROOT OUTPUT_DIR SOURCE_VIDEO_ROOT CREATE_VIDEO_TREE GAZE_VIDEO_ROOT REPORT_DIR
export MAX_OVERLAY_FRAMES FALLBACK_FOV_DEG GAZE_MAP_TYPE BINARY_RADIUS OVERLAY_ALPHA SIGMAS LIMIT DRY_RUN

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}

mapfile -t zips < <(find "${RAW_ROOT}/SLAM-and-Gaze" -name "mps_*_vrs.zip" | sort)
echo "[gaze-batch] Found ${#zips[@]} gaze zip(s)"

video_count=0
variant_count=0
for gaze_zip in "${zips[@]}"; do
  base="$(basename "${gaze_zip}")"
  video_id="${base#mps_}"
  video_id="${video_id%_vrs.zip}"
  participant="${video_id%%-*}"
  video_path="${RAW_ROOT}/Videos/${participant}/${video_id}.mp4"

  echo "[gaze-batch] ${video_id}"
  echo "[gaze-batch]   zip: ${gaze_zip}"
  echo "[gaze-batch]   video: ${video_path}"

  if [[ ! -f "${video_path}" ]]; then
    echo "[gaze-batch]   skip: missing video" >&2
    continue
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  sigmas_text="${SIGMAS//,/ }"
  read -ra sigmas <<< "${sigmas_text}"
  for sigma in "${sigmas[@]}"; do
    sigma_tag="${sigma//./p}"
    if [[ "${GAZE_MAP_TYPE}" == "binary" ]]; then
      radius="${BINARY_RADIUS:-${sigma}}"
      radius_tag="${radius//./p}"
      variant_output_dir="${OUTPUT_DIR}/binary_radius_${radius_tag}"
      echo "[gaze-batch]   binary radius: ${radius}"
    else
      radius=""
      variant_output_dir="${OUTPUT_DIR}/sigma_${sigma_tag}"
      echo "[gaze-batch]   sigma: ${sigma}"
    fi
    overlay_args=(
      "${PROJECT_ROOT}/scripts/build_hdepic_gaze_heatmap_video.py"
      --gaze-zip "${gaze_zip}" \
      --video "${video_path}" \
      --output-dir "${variant_output_dir}" \
      --video-id "${video_id}" \
      --fallback-fov-deg "${FALLBACK_FOV_DEG}" \
      --map-type "${GAZE_MAP_TYPE}" \
      --sigma "${sigma}" \
      --overlay-alpha "${OVERLAY_ALPHA}" \
      --max-overlay-frames "${MAX_OVERLAY_FRAMES}"
    )
    if [[ -n "${radius}" ]]; then
      overlay_args+=(--binary-radius "${radius}")
    fi
    python "${overlay_args[@]}"
    variant_count=$((variant_count + 1))
  done

  video_count=$((video_count + 1))
  if [[ "${LIMIT}" != "0" && "${video_count}" -ge "${LIMIT}" ]]; then
    echo "[gaze-batch] Reached LIMIT=${LIMIT}"
    break
  fi
done

echo "[gaze-batch] Built ${variant_count} overlay variant(s) for ${video_count} video(s)"

if [[ "${CREATE_VIDEO_TREE}" == "1" && "${DRY_RUN}" != "1" ]]; then
  sigmas_text="${SIGMAS//,/ }"
  read -ra sigmas <<< "${sigmas_text}"
  if [[ "${GAZE_MAP_TYPE}" == "binary" ]]; then
    radius="${BINARY_RADIUS:-${sigmas[0]}}"
    radius_tag="${radius//./p}"
    tree_overlay_dir="${OUTPUT_DIR}/binary_radius_${radius_tag}"
    tree_root="${GAZE_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos_gaze_binary_radius${radius_tag}}"
    report_path="${REPORT_DIR}/gaze_video_tree_binary_radius${radius_tag}.json"
  else
    sigma="${sigmas[0]}"
    sigma_tag="${sigma//./p}"
    tree_overlay_dir="${OUTPUT_DIR}/sigma_${sigma_tag}"
    tree_root="${GAZE_VIDEO_ROOT:-${PROJECT_ROOT}/data/hdepic_vjepa_videos_gaze_sigma${sigma}}"
    report_path="${REPORT_DIR}/gaze_video_tree_sigma${sigma_tag}.json"
  fi
  echo "[gaze-batch] Creating V-JEPA video tree: ${tree_root}"
  python "${PROJECT_ROOT}/scripts/create_hdepic_gaze_video_tree.py" \
    --source-video-root "${SOURCE_VIDEO_ROOT}" \
    --overlay-dir "${tree_overlay_dir}" \
    --output-video-root "${tree_root}" \
    --report "${report_path}" \
    --replace-existing
fi
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
