#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_full_heads_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
CODE_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PROJECT_ROOT=${PROJECT_ROOT:-${CODE_ROOT}}
export PROJECT_ROOT CODE_ROOT
source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
IMAGE=${IMAGE:-/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif}
OVERLAY=${OVERLAY:-${PROJECT_ROOT}/overlay-15GB-500K.ext3}
DEPS=${DEPS:-${PROJECT_ROOT}/outputs/b19_scalant/deps}
RULSTM_DEPS=${RULSTM_DEPS:-${PROJECT_ROOT}/rulstm_deps}
export PROJECT_ROOT CODE_ROOT DEPS RULSTM_DEPS

scalant_exec cpu \
  /bin/bash -c '
    set -euo pipefail
    export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    cd "${CODE_ROOT}"
    export PYTHONPATH="${RULSTM_DEPS}:${DEPS}:${CODE_ROOT}:${PYTHONPATH:-}"
    python -m py_compile app/scalant_baseline/official_stream_full_eval_gpu.py app/scalant_baseline/official_stream_full_eval_cpu.py
    python - <<'\''PY'\''
from app.scalant_baseline.official_stream_data import build_official_split1_ticks
from app.scalant_baseline.official_stream_full_eval_cpu import score
from app.scalant_baseline.official_stream_full_eval_gpu import PROTOCOL
import torch
ticks = build_official_split1_ticks("val")
counts = tuple(sum(t.target_actions[i] >= 0 for t in ticks) for i in (0, 1))
assert len(ticks) == 16227 and counts == (4678, 4655), (len(ticks), counts)
probe = score(torch.eye(5), torch.arange(5))
assert probe["count"] == 5 and probe["mean_recall_at5_present_classes"] == 1.0
print({"protocol": PROTOCOL, "ticks": len(ticks), "scored": counts}, flush=True)
PY
  '
