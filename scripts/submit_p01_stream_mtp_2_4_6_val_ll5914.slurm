#!/bin/bash
#SBATCH --job-name=p01_stream_val
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/ll5914/logs/p01_stream_mtp_val_%j.out
#SBATCH --error=/scratch/ll5914/logs/p01_stream_mtp_val_%j.err

set -euo pipefail

# Val-only wrapper for streaming MTP. Prefers OUT_DIR/best.pt via resume of latest
# after copying best→latest when available (handled inside train script resume path
# by loading whatever latest exists; here we promote best if present).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="/scratch/ll5914/experiments/p01_stream_mtp_2_4_6"
if [[ -f "${OUT_DIR}/best.pt" ]]; then
  cp -f "${OUT_DIR}/best.pt" "${OUT_DIR}/latest.pt"
  echo "Promoted best.pt -> latest.pt for val-only"
elif [[ ! -f "${OUT_DIR}/latest.pt" ]]; then
  echo "ERROR: no best/latest under ${OUT_DIR}" >&2
  exit 1
fi
export VAL_ONLY=1
exec bash "${SCRIPT_DIR}/submit_p01_stream_mtp_2_4_6_ll5914.slurm"
