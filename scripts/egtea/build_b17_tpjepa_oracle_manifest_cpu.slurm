#!/bin/bash
#SBATCH --job-name=b17_tpjepa_manifest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

PROJECT_ROOT=${PROJECT_ROOT}
SOURCE_CSV=${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1/EGTEA_train_stream_mtp.csv
OUTPUT_DIR="${B17_MANIFEST_DIR}"

cd "$PROJECT_ROOT"
test "$(git rev-parse --show-toplevel)" = "$PROJECT_ROOT"

export PROJECT_ROOT SOURCE_CSV OUTPUT_DIR
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
python scripts/build_b17_tpjepa_oracle_manifest.py \
  --train-csv "$SOURCE_CSV" \
  --output-dir "$OUTPUT_DIR"
sha256sum \
  "$OUTPUT_DIR/oracle_calibration_512.csv" \
  "$OUTPUT_DIR/oracle_heldout_2048.csv" \
  "$OUTPUT_DIR/oracle_manifest.json"
EOF
