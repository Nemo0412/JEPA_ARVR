#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_random_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export OUTROOT="${DATA_ROOT}/outputs/attn_corner_sink/q3_random_drop_accuracy"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUTROOT"
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m py_compile app/hdepic_lora_action_anticipation/eval_b18_q3_random_drop_accuracy.py scripts/egtea/analyze_b18_q3_random_drop_accuracy.py scripts/egtea/archive_b18_q3_token_self_offslot.py
python -m app.hdepic_lora_action_anticipation.eval_b18_q3_random_drop_accuracy prepare --out "${OUTROOT}/prepared-${SLURM_JOB_ID}"
EOF
