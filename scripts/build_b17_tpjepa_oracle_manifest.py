#!/usr/bin/env python3
"""Build the fixed train-only B17 oracle calibration/held-out manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.hdepic_lora_action_anticipation.task_protected_jepa_pruning import (
    EXPECTED_TRAIN_CSV_SHA256,
    write_oracle_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-source-sha256",
        default=EXPECTED_TRAIN_CSV_SHA256,
        help="Fail closed if the train CSV bytes do not match this digest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_oracle_manifests(
        source_csv=args.train_csv,
        output_dir=args.output_dir,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
