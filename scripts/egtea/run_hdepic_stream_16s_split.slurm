#!/bin/bash
# [ATTN-CORNER-SINK] B18 · Build HD-EPIC P01 16s streaming-MTP index (genuine 64-slot samples)
# for the predictor-score pruning replication. Source stage2 + videos from ll's dataset (P01);
# output to our data dir. CPU only.
#SBATCH --job-name=VJEPA2-EXP__hdepic_stream_16s_split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT

STAGE2="${STAGE2:-${HDEPIC_STAGE2:?Set HDEPIC_STAGE2}}"
VIDEO_ROOT="${VIDEO_ROOT:-${HDEPIC_VIDEO_ROOT:?Set HDEPIC_VIDEO_ROOT}}"
OUT="${OUT:-${DATA_ROOT}/data/hdepic_vjepa_annotations/stream_16s_split}"
MAX_CONTEXT="${MAX_CONTEXT:-16}"; SCHEDULE="${SCHEDULE:-4,6,8,10,12,14,16}"; PARTICIPANT="${PARTICIPANT:-P01}"

mkdir -p "${OUT}" ${DATA_ROOT}/outputs/attn_corner_sink/hdepic_stream_16s_split
[[ -d "${STAGE2}" ]] || { echo "ERROR missing stage2 ${STAGE2}" >&2; exit 2; }
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR missing video root ${VIDEO_ROOT}" >&2; exit 2; }
export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 STAGE2 VIDEO_ROOT OUT MAX_CONTEXT SCHEDULE PARTICIPANT

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python "${CODE_ROOT}/scripts/make_hdepic_stream_half_split.py" \
  --stage2 "$STAGE2" --video-root "$VIDEO_ROOT" --out "$OUT" \
  --max-context-sec "$MAX_CONTEXT" --context-schedule "$SCHEDULE" \
  --tick-sec 2.0 --min-context-sec 4.0 --horizons-sec 2,4,6 --model-fps 8.0 --participant "$PARTICIPANT"
echo "=== val context_sec distribution ==="
python - <<'PY'
import csv, os
from collections import Counter
p=os.path.join(os.environ["OUT"],"HD_EPIC_val_stream_mtp.csv")
c=Counter()
with open(p) as f:
    for r in csv.DictReader(f): c[r["context_sec"]]+=1
print("val context_sec:", dict(sorted(c.items(), key=lambda kv: float(kv[0]))), "total:", sum(c.values()))
PY
EOF
echo "hdepic 16s split done: ${OUT}"
