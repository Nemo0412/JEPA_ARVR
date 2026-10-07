"""Verify all existing official EGTEA split/horizon feature candidates."""

import argparse
import gc
import json
from pathlib import Path

from .egtea_cache import EGTEAInstanceCache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"kind": "b19_scalant_data_candidate_audit", "candidates": []}
    for split in (1, 2, 3):
        for horizon in (2, 6):
            for population in ("training", "validation"):
                dataset = EGTEAInstanceCache(split, horizon, population,
                                             verify_file_hash=True)
                item = {"split": split, "horizon": horizon,
                        "population": population, "rows": len(dataset),
                        "input_shape": list(dataset.features.shape),
                        "first_uid": int(dataset.ids[0]),
                        "last_uid": int(dataset.ids[-1]),
                        "file_sha256_verified": True}
                report["candidates"].append(item)
                print(json.dumps(item), flush=True)
                del dataset
                gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
