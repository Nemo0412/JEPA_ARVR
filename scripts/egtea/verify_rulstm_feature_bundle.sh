#!/bin/bash
# Verify that a RU-LSTM feature bundle has one feature/metadata pair for every
# frozen EGTEA streaming session. This checks identity by stem, not only count.

set -euo pipefail

if [[ "$#" != "2" ]]; then
  echo "usage: $0 <feature-dir> <annotation-dir>" >&2
  exit 2
fi

FEAT_DIR="$1"
ANN_DIR="$2"
TRAIN_CSV="${ANN_DIR}/EGTEA_train_stream_mtp.csv"
VAL_CSV="${ANN_DIR}/EGTEA_val_stream_mtp.csv"

for path in "${TRAIN_CSV}" "${VAL_CSV}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
[[ -d "${FEAT_DIR}" ]] || { echo "ERROR: missing ${FEAT_DIR}" >&2; exit 2; }

TMP_ROOT="$(mktemp -d)"
trap 'rm -f "${TMP_ROOT}/expected" "${TMP_ROOT}/npy" "${TMP_ROOT}/json"; rmdir "${TMP_ROOT}"' EXIT

awk -F, 'FNR > 1 {print $2}' "${TRAIN_CSV}" "${VAL_CSV}" | sort -u > "${TMP_ROOT}/expected"
find "${FEAT_DIR}" -maxdepth 1 -type f -name '*.npy' -printf '%f\n' | sed 's/\.npy$//' | sort -u > "${TMP_ROOT}/npy"
find "${FEAT_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%f\n' | sed 's/\.json$//' | sort -u > "${TMP_ROOT}/json"

for list in expected npy json; do
  [[ "$(wc -l < "${TMP_ROOT}/${list}")" == "86" ]] || {
    echo "ERROR: ${list} session count is not 86" >&2
    exit 2
  }
done
cmp "${TMP_ROOT}/expected" "${TMP_ROOT}/npy" || {
  echo "ERROR: .npy stems do not match frozen CSV sessions" >&2; exit 2;
}
cmp "${TMP_ROOT}/expected" "${TMP_ROOT}/json" || {
  echo "ERROR: .json stems do not match frozen CSV sessions" >&2; exit 2;
}

echo "RU-LSTM EGTEA feature bundle stems: PASS (86 .npy + 86 .json)"
