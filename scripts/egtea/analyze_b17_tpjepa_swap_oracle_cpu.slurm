#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b17_tpjepa_swap_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${PROJECT_ROOT}"
export VJEPA_ROOT
MANIFEST_CSV="${B17_MANIFEST_DIR}/oracle_heldout_2048.csv"
MANIFEST_SHA256=937f1e140fc0eeb3942804cebdece5ecef43e39202d92be18f3a5d59fcdc9d42
ARTIFACT="${ARTIFACT:?set ARTIFACT to the completed 128-row v2 artifact}"
ARTIFACT_SHA256="${ARTIFACT_SHA256:?set ARTIFACT_SHA256 to its registered SHA-256}"
OUT_ROOT="$DATA_ROOT/outputs/b17_token_pruning/task_protected_jepa/attention_jepa_swap_v2/audit"
mkdir -p "$OUT_ROOT"
OUTPUT="$OUT_ROOT/b17-tpjepa-swap-v2-audit-${SLURM_JOB_ID}.json"
[[ -f "$ARTIFACT" ]] || { echo "ERROR: missing artifact $ARTIFACT" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "ERROR: refusing to overwrite $OUTPUT" >&2; exit 2; }

export CODE_ROOT VJEPA_ROOT ARTIFACT ARTIFACT_SHA256 MANIFEST_CSV MANIFEST_SHA256 OUTPUT
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python scripts/analyze_b17_tpjepa_swap_oracle.py \
  --artifact "$ARTIFACT" \
  --expected-artifact-sha256 "$ARTIFACT_SHA256" \
  --manifest-csv "$MANIFEST_CSV" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --output "$OUTPUT"
EOF

echo "result=$OUTPUT"
