#!/bin/bash
# [ATTN-CORNER-SINK] B18 · Regenerate the EGTEA streaming-MTP index with a 16 s window
# (context schedule 4..16) so there are genuine 64-slot samples for the predictor-score
# pruning experiment. Writes to a NEW dir (stream_16s_split/split1); does NOT touch the
# standard 10 s stream_half_split CSVs. CPU only (decord metadata read).
#SBATCH --job-name=VJEPA2-EXP__egtea_stream_16s_split
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

SOURCE="${SOURCE:-${DATA_ROOT}/data/egtea/vjepa_annotations/v1/split1}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/data/egtea/session_videos}"
OUT="${OUT:-${DATA_ROOT}/data/egtea/vjepa_annotations/stream_16s_split/split1}"
MAX_CONTEXT="${MAX_CONTEXT:-16}"
SCHEDULE="${SCHEDULE:-4,6,8,10,12,14,16}"

mkdir -p "$(dirname "${OUT}")" ${DATA_ROOT}/outputs/attn_corner_sink/stream_16s_split
[[ -d "${SOURCE}" ]] || { echo "ERROR: missing source ${SOURCE}" >&2; exit 2; }
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing video root ${VIDEO_ROOT}" >&2; exit 2; }

export DATA_ROOT CODE_ROOT VJEPA_ROOT PYTHONNOUSERSITE=1 SOURCE VIDEO_ROOT OUT MAX_CONTEXT SCHEDULE
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
cd "$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
python "$PROJECT_ROOT/scripts/egtea/make_b13_egtea_stream_half_split.py" \
  --source "$SOURCE" --video-root "$VIDEO_ROOT" --out "$OUT" \
  --max-context-sec "$MAX_CONTEXT" --context-schedule "$SCHEDULE" \
  --tick-sec 2.0 --min-context-sec 4.0 --horizons-sec 2,4,6 --model-fps 8.0
echo "=== context_sec distribution (val) ==="
python - <<'PY'
import csv, os
from collections import Counter
p=os.path.join(os.environ["OUT"],"EGTEA_val_stream_mtp.csv")
c=Counter()
with open(p) as f:
    for r in csv.DictReader(f): c[r["context_sec"]]+=1
print("val context_sec:", dict(sorted(c.items(), key=lambda kv: float(kv[0]))))
print("val total rows:", sum(c.values()))
PY
EOF
echo "16s split regeneration done: ${OUT}"
