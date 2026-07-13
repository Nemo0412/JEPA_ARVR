#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__hdepic_rescore_window
#SBATCH --account=your_slurm_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/VJEPA2-EXP/logs/hdepic_rescore_window_%j.out
#SBATCH --error=/path/to/VJEPA2-EXP/logs/hdepic_rescore_window_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

set -euo pipefail

# CPU-only wrapper for scripts/rescore_window.py. This keeps Python off the
# login node and allows a dependent rescore job after val-only dump jobs.

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/VJEPA2-EXP}"
VJEPA_ROOT="${VJEPA_ROOT:-${PROJECT_ROOT}/vjepa2}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

ANNOTATIONS="${ANNOTATIONS:-${PROJECT_ROOT}/data/hdepic_vjepa_annotations/HD_EPIC_val_vjepa.csv}"
WINDOW_SEC="${WINDOW_SEC:-5.0}"
OUT_DIR="${OUT_DIR:?must set OUT_DIR}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUT_DIR}"

export PROJECT_ROOT VJEPA_ROOT SIF OVERLAY ENV_NAME
export ANNOTATIONS WINDOW_SEC OUT_DIR
for idx in 1 2 3 4 5 6 7 8; do
    pred_var="PREDICTION_${idx}"
    label_var="LABEL_${idx}"
    export "${pred_var}=${!pred_var:-}"
    export "${label_var}=${!label_var:-}"
done

payload=$(cat <<'BASH'
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
args=()
for idx in 1 2 3 4 5 6 7 8; do
    pred_var="PREDICTION_${idx}"
    label_var="LABEL_${idx}"
    pred="${!pred_var:-}"
    label="${!label_var:-}"
    if [[ -n "${pred}" ]]; then
        args+=(--predictions "${pred}")
        if [[ -n "${label}" ]]; then
            args+=(--label "${label}")
        fi
    fi
done
if [[ "${#args[@]}" -eq 0 ]]; then
    echo "Set at least PREDICTION_1" >&2
    exit 2
fi

python scripts/rescore_window.py \
    "${args[@]}" \
    --annotations "${ANNOTATIONS}" \
    --window-sec "${WINDOW_SEC}" \
    --out-dir "${OUT_DIR}"
BASH
)

singularity exec \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "${payload}"
