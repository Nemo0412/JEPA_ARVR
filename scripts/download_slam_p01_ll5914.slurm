#!/bin/bash
#SBATCH --job-name=hdepic_slam_dl
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5914/logs/hdepic_slam_dl_%j.out
#SBATCH --error=/scratch/ll5914/logs/hdepic_slam_dl_%j.err

set -euo pipefail

# Download HD-EPIC P01 SLAM/multi data (closed-loop trajectories) needed for
# binary_input_adapter_gaze_pose_matrix mode.
# Running on GPU node because that's our available partition;
# GPU won't be used but the job needs to run somewhere.

PYTHON="/scratch/ll5914/conda_envs/SVD/bin/python"
DOWNLOADER_DIR="/scratch/ll5914/datasets/HD-EPIC/hd-epic-downloader"
OUTPUT_ROOT="/scratch/ll5914/datasets/HD-EPIC"
SLAM_DEST="${OUTPUT_ROOT}/SLAM-and-Gaze/P01/SLAM"

echo "========================================"
echo "Job : HD-EPIC P01 SLAM download"
echo "ID  : ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Dest: ${SLAM_DEST}"
echo "========================================"

mkdir -p "${SLAM_DEST}"

cd "${DOWNLOADER_DIR}"
echo "[slam-dl] Downloading P01 SLAM/multi data..."
"${PYTHON}" hd-epic-downloader.py \
    "${OUTPUT_ROOT}" \
    --slam-gaze \
    --participants 1

echo "[slam-dl] Download complete. Verifying..."

if [[ ! -f "${SLAM_DEST}/multi/vrs_to_multi_slam.json" ]]; then
    echo "[slam-dl] ERROR: vrs_to_multi_slam.json not found in ${SLAM_DEST}/multi/" >&2
    exit 1
fi

N_ZIPS=$(ls "${SLAM_DEST}/multi/"*.zip 2>/dev/null | wc -l)
echo "[slam-dl] Found ${N_ZIPS} SLAM zip files"

echo "[slam-dl] Checking trajectory content..."
"${PYTHON}" - <<'PY'
import zipfile, pathlib, sys
slam_root = pathlib.Path("/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi")
zips = sorted(slam_root.glob("*.zip"))
if not zips:
    print("ERROR: no SLAM zips found", file=sys.stderr)
    sys.exit(1)
z = zipfile.ZipFile(zips[0])
names = z.namelist()
has_traj = any("closed_loop_trajectory.csv" in n for n in names)
print(f"First zip: {zips[0].name}, files: {names[:5]}")
print(f"Has closed_loop_trajectory.csv: {has_traj}")
if not has_traj:
    print("ERROR: closed_loop_trajectory.csv missing", file=sys.stderr)
    sys.exit(1)
PY

echo "========================================"
echo "SLAM data ready at: ${SLAM_DEST}/multi/"
echo "End: $(date)"
echo "========================================"
