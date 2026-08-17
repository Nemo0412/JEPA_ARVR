#!/bin/bash
# Verify the nested <session>/<session>.MP4 streaming view against the frozen
# train/val session union. The builder creates symlinks, so find must follow
# them while checking the resolved video files.

set -euo pipefail

if [[ "$#" != "2" ]]; then
  echo "usage: $0 <video-root> <annotation-dir>" >&2
  exit 2
fi

VIDEO_ROOT="$1"
ANN_DIR="$2"
TRAIN_CSV="${ANN_DIR}/EGTEA_train_stream_mtp.csv"
VAL_CSV="${ANN_DIR}/EGTEA_val_stream_mtp.csv"

for path in "${TRAIN_CSV}" "${VAL_CSV}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
[[ -d "${VIDEO_ROOT}" ]] || { echo "ERROR: missing ${VIDEO_ROOT}" >&2; exit 2; }

TMP_ROOT="$(mktemp -d)"
trap 'rm -f "${TMP_ROOT}/expected" "${TMP_ROOT}/actual"; rmdir "${TMP_ROOT}"' EXIT

awk -F, 'FNR > 1 {print $2}' "${TRAIN_CSV}" "${VAL_CSV}" | sort -u > "${TMP_ROOT}/expected"
find -L "${VIDEO_ROOT}" -mindepth 2 -maxdepth 2 -type f -name '*.MP4' -printf '%f\n' \
  | sed 's/\.MP4$//' | sort -u > "${TMP_ROOT}/actual"

for list in expected actual; do
  [[ "$(wc -l < "${TMP_ROOT}/${list}")" == "86" ]] || {
    echo "ERROR: ${list} stream session count is not 86" >&2
    exit 2
  }
done
cmp "${TMP_ROOT}/expected" "${TMP_ROOT}/actual" || {
  echo "ERROR: nested stream video stems do not match frozen CSV sessions" >&2
  exit 2
}

echo "EGTEA nested stream video tree: PASS (86 sessions)"
