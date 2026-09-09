#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_single_frame_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}" VJEPA_ROOT="${PROJECT_ROOT}/vjepa2"
export SF_MODE="${SF_MODE:-prepare}" SF_INPUTS="${SF_INPUTS:-}" SF_PARTIAL="${SF_PARTIAL:-0}"
export SF_OUTPUT="${SF_OUTPUT:-${DATA_ROOT}/outputs/attn_corner_sink/q3_single_frame_source/analysis-${SLURM_JOB_ID}}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT" PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
if [[ "$SF_MODE" == prepare ]]; then
 python -m py_compile app/hdepic_lora_action_anticipation/probe_b18_q3_single_frame_source.py
 python -m app.hdepic_lora_action_anticipation.probe_b18_q3_single_frame_source prepare \
  --source-manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_boundaries/prepared/manifest.json" \
  --hdepic-train "${DATA_ROOT}/data/hdepic_vjepa_annotations/splits/p01_fixed_train_videos.txt" \
  --hdepic-root ${HDEPIC_VIDEO_ROOT:?Set HDEPIC_VIDEO_ROOT}/P01 \
  --out "${DATA_ROOT}/outputs/attn_corner_sink/q3_single_frame_source/prepared" --workers 4
else
 read -r -a inputs <<< "$SF_INPUTS"
 extra=()
 [[ "$SF_PARTIAL" == 1 ]] && extra+=(--allow-partial)
 python -m py_compile scripts/egtea/analyze_b18_q3_single_frame_source.py
 python scripts/egtea/analyze_b18_q3_single_frame_source.py \
  --manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_single_frame_source/prepared/manifest.json" \
  --runs "${inputs[@]}" --out "$SF_OUTPUT" "${extra[@]}"
fi
EOF
