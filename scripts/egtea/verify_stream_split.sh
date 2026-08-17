#!/bin/bash
# Fast, dependency-free identity checks for the shared EGTEA stream split.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ANN_DIR="${ANN_DIR:-${PROJECT_ROOT}/data/egtea/vjepa_annotations/stream_half_split/split1}"

check_csv() {
  local name="$1"
  local expected_rows="$2"
  local expected_md5="$3"
  local path="${ANN_DIR}/${name}"
  [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
  [[ "$(md5sum "${path}" | awk '{print $1}')" == "${expected_md5}" ]] || {
    echo "ERROR: MD5 mismatch: ${path}" >&2; exit 2;
  }
  [[ "$(($(wc -l < "${path}") - 1))" == "${expected_rows}" ]] || {
    echo "ERROR: row-count mismatch: ${path}" >&2; exit 2;
  }
}

check_csv EGTEA_train_stream_mtp.csv 25256 dcdc98c9993ae028dbb4a2877deff09a
check_csv EGTEA_val_stream_mtp.csv 22741 cb04d6eddc2e65910899fde8b3c08659
check_csv EGTEA_test_stream_mtp.csv 22741 cb04d6eddc2e65910899fde8b3c08659
cmp "${ANN_DIR}/EGTEA_val_stream_mtp.csv" "${ANN_DIR}/EGTEA_test_stream_mtp.csv"
grep -Fq '"rows": 171' "${ANN_DIR}/frozen_gaze_coverage.json"

SOURCE_DIR="${PROJECT_ROOT}/data/egtea/vjepa_annotations/v1/split1"
[[ "$(md5sum "${SOURCE_DIR}/EGTEA_train_vjepa.csv" | awk '{print $1}')" == "2df58b9ebf12d8e71d70265206e35529" ]]
[[ "$(md5sum "${SOURCE_DIR}/EGTEA_val_vjepa.csv" | awk '{print $1}')" == "4cccf4ec90ae18878005b894cbd15cf1" ]]
[[ "$(md5sum "${SOURCE_DIR}/EGTEA_test_vjepa.csv" | awk '{print $1}')" == "cab47f194aa734be86c4c9412cb8e54b" ]]

echo "EGTEA temporal-half stream split: PASS"
