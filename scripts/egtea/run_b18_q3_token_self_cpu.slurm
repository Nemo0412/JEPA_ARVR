#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_token_self_cpu
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
export TS_MODE="${TS_MODE:-prepare}" TS_INPUTS="${TS_INPUTS:-}" TS_PARTIAL="${TS_PARTIAL:-0}"
export TS_OUTPUT="${TS_OUTPUT:-${DATA_ROOT}/outputs/attn_corner_sink/q3_token_self_return/analysis-${SLURM_JOB_ID}}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT" PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
if [[ "$TS_MODE" == prepare ]]; then
 python -m py_compile app/hdepic_lora_action_anticipation/probe_b18_q3_token_self_return.py
 python -m app.hdepic_lora_action_anticipation.probe_b18_q3_token_self_return prepare \
  --source-manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_boundaries/prepared/manifest.json" \
  --out "${DATA_ROOT}/outputs/attn_corner_sink/q3_token_self_return/prepared-v2"
else
 read -r -a inputs <<< "$TS_INPUTS"
 extra=()
 [[ "$TS_PARTIAL" == 1 ]] && extra+=(--allow-partial)
 python -m py_compile scripts/egtea/analyze_b18_q3_token_self_return.py
 python scripts/egtea/analyze_b18_q3_token_self_return.py \
  --manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_token_self_return/prepared-v2/manifest.json" \
  --runs "${inputs[@]}" --out "$TS_OUTPUT" "${extra[@]}"
fi
EOF
