#!/bin/bash
#SBATCH --job-name=b17_tpjepa_cal_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${PROJECT_ROOT}"
EXPECTED_CODE_ROOT=${PROJECT_ROOT}
MANIFEST="${B17_MANIFEST_DIR}/oracle_calibration_512.csv"
MANIFEST_SHA256=e09fa32c7351f4ffe0c387b70c708e76c0cec125b9df0959e0d58abb2111b18b
SHARD_0="$DATA_ROOT/outputs/b17_token_pruning/task_protected_jepa/oracle_targets/calibration/b17-tpjepa-coarse-calibration-0-256-16474330.pt"
SHARD_0_SHA256=ca596cade25b86590be6ba553ee0c546b98fb23f730cd49cdbb72a6dc9b0efd8
SHARD_1="$DATA_ROOT/outputs/b17_token_pruning/task_protected_jepa/oracle_targets/calibration/b17-tpjepa-coarse-calibration-256-512-16474329.pt"
SHARD_1_SHA256=146bdf8dde16b8bf8f92f561ed9c22f917ce0ef68fb30dea08d0398cf3b200f1
OUT_ROOT="$DATA_ROOT/outputs/b17_token_pruning/task_protected_jepa/calibration_audit"
OUT_JSON="$OUT_ROOT/b17-tpjepa-schema-v2-calibration-audit-${SLURM_JOB_ID}.json"

[[ "$(realpath "$CODE_ROOT")" == "$EXPECTED_CODE_ROOT" ]] \
  || { echo "ERROR: wrong isolated code root" >&2; exit 2; }
[[ "$(realpath "$CODE_ROOT")" == "$(realpath "$PROJECT_ROOT")" ]] \
  || { echo "ERROR: wrong shared code root" >&2; exit 2; }
for path in "$MANIFEST" "$SHARD_0" "$SHARD_1"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file $path" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT"
[[ ! -e "$OUT_JSON" ]] || { echo "ERROR: refusing to overwrite $OUT_JSON" >&2; exit 2; }

export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export PYTHONPATH="$CODE_ROOT:\${PYTHONPATH:-}"
cd "$CODE_ROOT"
python scripts/analyze_b17_tpjepa_calibration_targets.py \
  --target-shard "$SHARD_0" \
  --expected-target-sha256 "$SHARD_0_SHA256" \
  --target-shard "$SHARD_1" \
  --expected-target-sha256 "$SHARD_1_SHA256" \
  --manifest-csv "$MANIFEST" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --output-json "$OUT_JSON"
EOF

sha256sum "$OUT_JSON"
echo "audit=$OUT_JSON"
