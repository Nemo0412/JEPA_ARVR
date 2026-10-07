#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_scalant_s23_train_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:20:00
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
    export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
    source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
    scalant_activate
    cd "${CODE_ROOT}"
    export PYTHONPATH="${RULSTM_DEPS}:${DEPS}:${CODE_ROOT}:${PYTHONPATH:-}"
    python -m py_compile app/scalant_baseline/official_stream_train.py app/scalant_baseline/official_stream_data.py app/scalant_baseline/official_three_split_audit.py
    python - <<'\''PY'\''
import json
from pathlib import Path
from app.scalant_baseline.egtea_cache import MAIN_ROOT
from app.scalant_baseline.official_stream_data import build_official_split1_ticks
from app.scalant_baseline.official_stream_train import _assert_population, protocol_for_split, source_identity
for split in (2, 3):
    path = MAIN_ROOT / f"outputs/b19_scalant/implementation/official_split{split}_stream_audit.json"
    audit = json.loads(path.read_text())
    identity = source_identity(split)
    for population in ("train", "val"):
        rows = build_official_split1_ticks(population, split)
        record = audit["population"][population]
        _assert_population(rows, population, record["processed_ticks"],
                           (record["scored_h2"], record["scored_h6"]))
        assert not record["missing_queries"] and not record["malformed_queries"]
        prefix = "training" if population == "train" else "validation"
        assert identity[f"{prefix}{split}_sha256"] == audit["annotation_sha256"][population]
    print({"split": split, "protocol": protocol_for_split(split),
           "val_scored": (audit["population"]["val"]["scored_h2"],
                          audit["population"]["val"]["scored_h6"])}, flush=True)
PY
  '
