"""Audit source feature/label coverage for fixed-window EGTEA instances."""

from __future__ import annotations

import json
from collections import Counter

import lmdb

from .egtea_cache import MAIN_ROOT
from .official_instance_data import build_instance_rows
from .stream_source_audit import LMDB


OUTPUT = MAIN_ROOT / "outputs/b19_scalant/implementation/official_instance32s_audit.json"


def main():
    env = lmdb.open(str(LMDB), readonly=True, lock=False, readahead=False)
    report = {"kind": "official_split1_fixed32s_instance_feasibility_not_training",
              "source": str(LMDB), "history_seconds": 32,
              "long_memory_seconds": 28, "working_window_seconds": 4,
              "feature_fps": 4, "population": {}}
    try:
        with env.begin() as txn:
            for population in ("train", "val"):
                for horizon in (2, 6):
                    rows, early = build_instance_rows(population, horizon)
                    counts = Counter()
                    missing = []
                    bad = []
                    missing_uids = set()
                    for row in rows:
                        counts["samples"] += 1
                        counts["known_past_positions"] += sum(a >= 0 for a in row.past_actions)
                        counts["known_work_positions"] += sum(a >= 0 for a in row.past_actions[-16:])
                        for key in row.keys:
                            counts["feature_queries"] += 1
                            payload = txn.get(key.encode())
                            if payload is None:
                                counts["missing_queries"] += 1
                                missing_uids.add(row.uid)
                                if len(missing) < 15:
                                    missing.append({"uid": row.uid, "key": key})
                            elif len(payload) != 4096:
                                counts["bad_payloads"] += 1
                                if len(bad) < 15:
                                    bad.append({"uid": row.uid, "key": key,
                                                "bytes": len(payload)})
                    report["population"][f"{population}_h{horizon}"] = {
                        **counts, "early_discarded": len(early),
                        "early_uids": early[:20], "missing_uids": len(missing_uids),
                        "missing_examples": missing, "bad_examples": bad,
                    }
    finally:
        env.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if any(data.get("missing_queries", 0) or data.get("bad_payloads", 0)
           for data in report["population"].values()):
        raise ValueError("Fixed-window source feature coverage failed")


if __name__ == "__main__":
    main()
