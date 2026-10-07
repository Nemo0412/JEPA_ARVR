"""Stage one full fixed-window horizon for an end-to-end budget smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .egtea_cache import MAIN_ROOT
from .official_instance_data import HISTORY_STEPS, build_instance_rows
from .official_stream_train import source_identity
from .stream_data import StreamingTSNReader


BASE = MAIN_ROOT / "outputs/b19_scalant/instance32s_cache"


def ids_hash(rows):
    return hashlib.sha256(",".join(str(row.uid) for row in rows).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=(2, 6), required=True)
    args = parser.parse_args()
    output = BASE / f"h{args.horizon}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("Refusing to replace an existing fixed-window cache")
    report = {"kind": "fixed32s_instance_feature_cache_for_budget_smoke",
              "horizon_seconds": args.horizon,
              "source_identity": source_identity(), "population": {}}
    reader = StreamingTSNReader()
    try:
        for population in ("train", "val"):
            rows, early = build_instance_rows(population, args.horizon)
            if early:
                raise ValueError("Unexpected early sample exclusion")
            path = output / f"{population}.npy"
            temporary = Path(str(path) + ".tmp")
            array = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float32,
                shape=(len(rows), HISTORY_STEPS, 1024))
            start = time.perf_counter()
            for index, row in enumerate(rows):
                array[index] = reader.read(row).numpy()
                if (index + 1) % 1000 == 0:
                    print(json.dumps({"population": population,
                                      "finished": index + 1,
                                      "total": len(rows),
                                      "seconds": time.perf_counter() - start}), flush=True)
            array.flush()
            del array
            os.replace(temporary, path)
            report["population"][population] = {
                "samples": len(rows), "uids_sha256": ids_hash(rows),
                "path": str(path), "bytes": path.stat().st_size,
                "read_and_write_seconds": time.perf_counter() - start}
    finally:
        reader.close()
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output / "manifest.json")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
