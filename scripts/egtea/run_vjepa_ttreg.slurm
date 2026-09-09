#!/bin/bash
# [ATTN-CORNER-SINK] B18 · V-JEPA 2.1 test-time-register port (no training). Slurm only.
#SBATCH --job-name=VJEPA2-EXP__vjepa_ttreg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=00:40:00
set -euo pipefail
SHARE_GPU="${SHARE_GPU:-1}"
: "${PROJECT_ROOT:?Export the absolute JEPA_ARVR root before sbatch}"
source "$PROJECT_ROOT/scripts/egtea/share_runtime.sh"
DATA_ROOT=${DATA_ROOT}
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
export VJEPA_ROOT
CKPT="${DATA_ROOT}/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
ANN="${DATA_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1"
VAL="${ANN}/EGTEA_val_stream_mtp.csv"
VID="${DATA_ROOT}/data/egtea/stream_session_videos"
SCALE="${SCALE:-1.0}"; N_REG="${N_REG:-4}"
OUT="${DATA_ROOT}/outputs/attn_corner_sink/vjepa_ttreg/port_scale${SCALE}_reg${N_REG}.json"
export DATA_ROOT CODE_ROOT VJEPA_ROOT CKPT VAL VID OUT SCALE N_REG
export LAYERS="${LAYERS:-14,15,16,17,18,19,20}" NEVAL="${NEVAL:-20}" NCALIB="${NCALIB:-20}"
share_exec bash -s <<'EOF'
python() { "$PYTHON" "$@"; }
set -euo pipefail
export SHARED_PROJECT_ROOT="$DATA_ROOT" PROJECT_ROOT="$CODE_ROOT"
export PYTHONPATH="${CODE_ROOT}:${VJEPA_ROOT}:\${PYTHONPATH:-}"
cd "$CODE_ROOT"
python app/hdepic_lora_action_anticipation/vjepa_testtime_register.py \
  --checkpoint "$CKPT" --val-csv "$VAL" --video-root "$VID" --out-json "$OUT" \
  --img-size 384 --max-frames 32 --context-sec 4 --scale "$SCALE" \
  --n-registers "$N_REG" --layers "$LAYERS" --n-calib "$NCALIB" --n-eval "$NEVAL"
EOF
