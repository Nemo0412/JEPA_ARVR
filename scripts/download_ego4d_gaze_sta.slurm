#!/bin/bash
#SBATCH --job-name=ego4d_dl
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5914/logs/ego4d_download_%j.out
#SBATCH --error=/scratch/ll5914/logs/ego4d_download_%j.err

set -euo pipefail

# Ego4D: annotations + gaze + FHO clips (STA/LTA; not full multi-TB full_scale).
# Requires AWS profile "ego4d" (credentials from https://ego4d.dev / ego4ddataset.com).
# License keys expire ~14 days — re-sign with the same email if 403 Forbidden.

PYTHON="${PYTHON:-/scratch/ll5914/conda_envs/SVD/bin/python}"
OUT_DIR="${OUT_DIR:-/scratch/ll5914/datasets/Ego4D}"
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-ego4d}"
VERSION="${EGO4D_VERSION:-v2}"

export AWS_PROFILE="${AWS_PROFILE_NAME}"
mkdir -p "${OUT_DIR}" /scratch/ll5914/logs

echo "=== $(hostname) $(date) ==="
echo "OUT_DIR=${OUT_DIR}"
echo "AWS_PROFILE=${AWS_PROFILE_NAME} VERSION=${VERSION}"

if ! "${PYTHON}" -c "import ego4d" 2>/dev/null; then
  echo "Installing ego4d into env..."
  "${PYTHON}" -m pip install -q ego4d boto3
fi

# Sanity: credentials must exist before S3 access.
if [[ ! -f "${HOME}/.aws/credentials" ]]; then
  echo "ERROR: missing ~/.aws/credentials" >&2
  echo "After ego4d.dev approval, put keys under profile [${AWS_PROFILE_NAME}], e.g.:" >&2
  cat >&2 <<'EOF'
[~/.aws/credentials]
[ego4d]
aws_access_key_id = YOUR_KEY
aws_secret_access_key = YOUR_SECRET
aws_session_token = YOUR_TOKEN   # if temporary

[~/.aws/config]
[profile ego4d]
region = us-west-2
EOF
  exit 1
fi

if ! grep -q "^\[${AWS_PROFILE_NAME}\]" "${HOME}/.aws/credentials"; then
  echo "ERROR: profile [${AWS_PROFILE_NAME}] not found in ~/.aws/credentials" >&2
  exit 1
fi

echo "=== 1/2 metadata + annotations + gaze ==="
"${PYTHON}" -m ego4d.cli.cli \
  --output_directory "${OUT_DIR}" \
  --datasets annotations gaze \
  --aws_profile_name "${AWS_PROFILE_NAME}" \
  --version "${VERSION}" \
  --yes

echo "=== 2/2 FHO clips (STA/LTA; not full 5TB) ==="
"${PYTHON}" -m ego4d.cli.cli \
  --output_directory "${OUT_DIR}" \
  --datasets clips \
  --benchmarks FHO \
  --aws_profile_name "${AWS_PROFILE_NAME}" \
  --version "${VERSION}" \
  --yes

echo "Done: $(date)"
du -sh "${OUT_DIR}"/* 2>/dev/null || du -sh "${OUT_DIR}"
ls -la "${OUT_DIR}/${VERSION}" 2>/dev/null || ls -la "${OUT_DIR}"
