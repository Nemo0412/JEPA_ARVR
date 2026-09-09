#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_mechanism_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
export MECH_MODE="${MECH_MODE:-prepare}"
export MECH_INPUTS="${MECH_INPUTS:-}" MECH_PARTIAL="${MECH_PARTIAL:-0}"
export MECH_OUTPUT="${MECH_OUTPUT:-${DATA_ROOT}/outputs/attn_corner_sink/q3_oldest_mechanism/analysis-${SLURM_JOB_ID}}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
if [[ "$MECH_MODE" == prepare ]]; then
 python -m py_compile app/hdepic_lora_action_anticipation/probe_b18_q3_oldest_mechanism.py
 python -m app.hdepic_lora_action_anticipation.probe_b18_q3_oldest_mechanism prepare \
  --source-manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_boundaries/prepared/manifest.json" \
  --out "${DATA_ROOT}/outputs/attn_corner_sink/q3_oldest_mechanism/prepared"
else
 read -r -a mech_inputs <<< "$MECH_INPUTS"
 extra=()
 [[ "$MECH_PARTIAL" == 1 ]] && extra+=(--allow-partial)
 python -m py_compile scripts/egtea/analyze_b18_q3_oldest_mechanism.py
 python scripts/egtea/analyze_b18_q3_oldest_mechanism.py \
  --manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_oldest_mechanism/prepared/manifest.json" \
  --runs "${mech_inputs[@]}" --out "$MECH_OUTPUT" "${extra[@]}"
fi
EOF
