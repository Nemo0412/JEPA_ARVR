#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b17_completion_e0_audit
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
AUDIT_MODE="${AUDIT_MODE:?set AUDIT_MODE=teacher or final}"
TEACHER_ARTIFACT="${TEACHER_ARTIFACT:?set TEACHER_ARTIFACT}"
TEACHER_SHA256="${TEACHER_SHA256:?set TEACHER_SHA256}"
JEPA_ARTIFACT="${JEPA_ARTIFACT:-}"
JEPA_SHA256="${JEPA_SHA256:-}"
case "$AUDIT_MODE" in teacher|final) ;; *) echo "ERROR: AUDIT_MODE must be teacher or final" >&2; exit 2;; esac
if [[ "$AUDIT_MODE" == "final" ]]; then
  [[ -n "$JEPA_ARTIFACT" && -n "$JEPA_SHA256" ]] \
    || { echo "ERROR: final audit requires JEPA_ARTIFACT and JEPA_SHA256" >&2; exit 2; }
fi
for path in "$MANIFEST_CSV" "$TEACHER_ARTIFACT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 2; }
done
if [[ -n "$JEPA_ARTIFACT" ]]; then
  [[ -f "$JEPA_ARTIFACT" ]] || { echo "ERROR: missing $JEPA_ARTIFACT" >&2; exit 2; }
fi

OUT_ROOT="$DATA_ROOT/outputs/b17_token_pruning/predictor_completion_e0/audit"
mkdir -p "$OUT_ROOT"
OUT_JSON="$OUT_ROOT/b17-completion-e0-${AUDIT_MODE}-audit-${SLURM_JOB_ID}.json"
[[ ! -e "$OUT_JSON" ]] || { echo "ERROR: refusing to overwrite $OUT_JSON" >&2; exit 2; }

export CODE_ROOT VJEPA_ROOT AUDIT_MODE TEACHER_ARTIFACT TEACHER_SHA256
export JEPA_ARTIFACT JEPA_SHA256 MANIFEST_CSV MANIFEST_SHA256 OUT_JSON PYTHONNOUSERSITE=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
args=(
  --mode "$AUDIT_MODE"
  --teacher-artifact "$TEACHER_ARTIFACT"
  --expected-teacher-sha256 "$TEACHER_SHA256"
  --manifest-csv "$MANIFEST_CSV"
  --expected-manifest-sha256 "$MANIFEST_SHA256"
  --output "$OUT_JSON"
)
if [[ "$AUDIT_MODE" == "final" ]]; then
  args+=(--jepa-artifact "$JEPA_ARTIFACT" --expected-jepa-sha256 "$JEPA_SHA256")
fi
python scripts/analyze_b17_predictor_completion_e0.py "${args[@]}"
EOF

sha256sum "$OUT_JSON"
echo "audit=$OUT_JSON"
