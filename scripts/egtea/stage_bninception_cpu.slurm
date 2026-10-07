#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__b19_tsn_source
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
CODE_ROOT=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
PROJECT_ROOT=${PROJECT_ROOT:-${CODE_ROOT}}
export PROJECT_ROOT CODE_ROOT
source "${CODE_ROOT}/scripts/egtea/scalant_runtime.sh"
DEST=${PROJECT_ROOT}/outputs/b19_scalant/deps_tsn
mkdir -p "$DEST"
curl --fail --location --silent --show-error --retry 3 \
  https://raw.githubusercontent.com/Cadene/pretrained-models.pytorch/master/pretrainedmodels/models/bninception.py \
  -o "$DEST/bninception.py.tmp"
test -s "$DEST/bninception.py.tmp"
# Keep the architecture source identical to the completed parameter audit.
printf '%s  %s\n' b262cc6308f17fbd6bf8f7dfd2b199efbd2acbcf34e46d26a7dea894b3c7966d \
  "$DEST/bninception.py.tmp" | sha256sum --check --status
mv "$DEST/bninception.py.tmp" "$DEST/bninception.py"
sha256sum "$DEST/bninception.py"
