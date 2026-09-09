#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b18_q3_onset_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
export DATA_ROOT=${DATA_ROOT}
export CODE_ROOT="${PROJECT_ROOT}"
export ONSET_ROOT="${DATA_ROOT}/outputs/attn_corner_sink/q3_layer_attention_onset"
export ONSET_MANIFEST="${ONSET_MANIFEST:-${ONSET_ROOT}/prepared-v2/manifest.json}"
export TMPDIR="${ONSET_ROOT}/tmp-${SLURM_JOB_ID}" MPLCONFIGDIR="${ONSET_ROOT}/mpl-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
export ONSET_MODE="${ONSET_MODE:-prepare}" ONSET_INPUTS="${ONSET_INPUTS:-}" ONSET_PARTIAL="${ONSET_PARTIAL:-0}"
export ONSET_OUTPUT="${ONSET_OUTPUT:-${ONSET_ROOT}/analysis-${SLURM_JOB_ID}}"
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PROJECT_ROOT="$CODE_ROOT" SHARED_PROJECT_ROOT="$DATA_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python -m py_compile app/hdepic_lora_action_anticipation/probe_b18_q3_layer_attention_onset.py scripts/egtea/analyze_b18_q3_layer_attention_onset.py
if [[ "$ONSET_MODE" == prepare ]]; then
 python -m app.hdepic_lora_action_anticipation.probe_b18_q3_layer_attention_onset prepare \
  --source-manifest "${DATA_ROOT}/outputs/attn_corner_sink/q3_oldest_mechanism/prepared/manifest.json" \
  --out "$(dirname "$ONSET_MANIFEST")"
else
 read -r -a onset_inputs <<< "$ONSET_INPUTS"
 extra=(); [[ "$ONSET_PARTIAL" == 1 ]] && extra+=(--allow-partial)
 python scripts/egtea/analyze_b18_q3_layer_attention_onset.py --manifest "$ONSET_MANIFEST" \
  --runs "${onset_inputs[@]}" --out "$ONSET_OUTPUT" "${extra[@]}"
fi
EOF
