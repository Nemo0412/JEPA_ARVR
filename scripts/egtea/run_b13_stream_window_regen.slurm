#!/bin/bash
# B13 window-size ablation (Axis 4): regenerate the streaming-MTP split CSVs at
# different sliding-window sizes via the authored generator. CPU-only (opens
# session videos to read frame counts). Tick set is fixed by min-context +
# horizons, so val n stays comparable (same samples, different history length).
#SBATCH --job-name=VJEPA2-EXP__b13_window_regen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
SHARE_GPU="${SHARE_GPU:-0}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"

DATA_ROOT="${DATA_ROOT:-${DATA_ROOT}}"
CODE_ROOT="${CODE_ROOT:-${DATA_ROOT}}"
export VJEPA_ROOT
OUT_BASE="${OUT_BASE:-${DATA_ROOT}/data/egtea/vjepa_annotations}"
mkdir -p "${DATA_ROOT}/outputs/egtea_stream_window_ablation"

# context-schedule per window is auto-generated (4,6,...,W) inside the loop.
WINDOWS="${WINDOWS:-20 30 40 60}"

export DATA_ROOT CODE_ROOT VJEPA_ROOT OUT_BASE WINDOWS PYTHONNOUSERSITE=1

share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
# The generator is untracked (absent from the worktree) and resolves its default
# data paths from its own location (parents[2]); run the shared-checkout copy
# read-only so DEFAULT_SOURCE / DEFAULT_VIDEO_ROOT point at the real data.
GEN="${DATA_ROOT}/scripts/egtea/make_b13_egtea_stream_half_split.py"
[[ -f "$GEN" ]] || { echo "ERROR: generator missing: $GEN" >&2; exit 2; }
cd "$DATA_ROOT"
export PYTHONPATH="${DATA_ROOT}:${VJEPA_ROOT}:${PYTHONPATH:-}"
for W in $WINDOWS; do
  # context schedule = 4,6,8,...,W (2 s steps); auto-generated for any window.
  SCH=$(seq -s, 4 2 "$W")
  OUT="${OUT_BASE}/stream_half_split_w${W}/split1"
  echo "=== regen window=${W}s schedule=${SCH} -> ${OUT} ==="
  python "$GEN" \
    --out "$OUT" --max-context-sec "$W" --context-schedule "$SCH" \
    --tick-sec 2.0 --min-context-sec 4.0 --horizons-sec 2,4,6 --model-fps 8.0
  N=$(($(wc -l < "${OUT}/EGTEA_val_stream_mtp.csv") - 1))
  echo "window=${W}s val_rows=${N}"
done
EOF

echo "B13 window regen completed for: ${WINDOWS}"
