#!/bin/bash
# Full post-hoc CPU analysis of fixed B17 v2 +2s correctness versus causal gaze quality.
#SBATCH --job-name=VJEPA2-EXP__b17_gaze_quality_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
EXPECTED_CODE_ROOT="${PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-${EXPECTED_CODE_ROOT}}"
export VJEPA_ROOT
[[ "$(realpath "${CODE_ROOT}")" == "${EXPECTED_CODE_ROOT}" ]] \
  || { echo "ERROR: analysis must use shared code root" >&2; exit 2; }
[[ "$(realpath "$CODE_ROOT")" == "$(realpath "$PROJECT_ROOT")" ]] \
  || { echo "ERROR: wrong shared code root" >&2; exit 2; }

VAL_CSV="${VAL_CSV:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1/EGTEA_val_stream_mtp.csv}"
GAZE_ROOT="${GAZE_ROOT:-${DATA_ROOT}/data/egtea/gaze_data/gaze_data/gaze_data}"
PAIRED_NPZ="${PAIRED_NPZ:-${DATA_ROOT}/outputs/b17_token_pruning/gaze_spatial/b17v2-la3840-uniform-calib-attention-gaze-gaze_shift-true_full-full-16268443.paired.npz}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/outputs/b17_token_pruning/gaze_quality_analysis}"
for path in "${VAL_CSV}" "${PAIRED_NPZ}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing required file ${path}" >&2; exit 2; }
done
[[ -d "${GAZE_ROOT}" ]] || { echo "ERROR: missing gaze root ${GAZE_ROOT}" >&2; exit 2; }
mkdir -p "${OUT_ROOT}"

RUN_TAG="b17-gaze-quality-disagreement-posthoc-v1-${SLURM_JOB_ID}"
OUT_JSON="${OUT_ROOT}/${RUN_TAG}.json"
[[ ! -e "${OUT_JSON}" && ! -e "${OUT_JSON%.json}.features.npz" ]] \
  || { echo "ERROR: refusing to overwrite ${RUN_TAG}" >&2; exit 2; }
CODE_COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)"
CODE_SNAPSHOT_SHA256="$({
  sha256sum \
    "${CODE_ROOT}/scripts/analyze_b17_gaze_quality_disagreement.py" \
    "${CODE_ROOT}/scripts/egtea/run_b17_gaze_quality_analysis.slurm"
  git -C "${CODE_ROOT}" status --porcelain=v1 --untracked-files=all
} | sha256sum | awk '{print $1}')"

export CODE_ROOT VJEPA_ROOT VAL_CSV GAZE_ROOT PAIRED_NPZ OUT_JSON CODE_COMMIT CODE_SNAPSHOT_SHA256
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python scripts/analyze_b17_gaze_quality_disagreement.py \
  --val-csv "$VAL_CSV" --gaze-root "$GAZE_ROOT" --paired-npz "$PAIRED_NPZ" \
  --out-json "$OUT_JSON" --job-id "$SLURM_JOB_ID" --code-root "$CODE_ROOT" \
  --code-commit "$CODE_COMMIT" --code-snapshot-sha256 "$CODE_SNAPSHOT_SHA256" \
  --bootstrap-samples 4000 --seed 17
EOF

echo "B17 gaze-quality analysis completed: ${OUT_JSON}"
