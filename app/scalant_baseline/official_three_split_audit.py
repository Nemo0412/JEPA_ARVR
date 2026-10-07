"""Audit official EGTEA split-2/3 streaming ticks and required TSN keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import lmdb

from .egtea_cache import ANNOTATIONS, MAIN_ROOT
from .official_stream_audit import official_actions, target
from .official_stream_data import build_official_split1_ticks
from .official_stream_train import file_sha256
from .stream_source_audit import LMDB, STATS


EXPECTED_ROWS = {2: (8299, 2022), 3: (8300, 2021)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    split = args.split
    by_video, memberships = official_actions(split)
    counts = Counter(memberships.values())
    if (counts["train"], counts["val"]) != EXPECTED_ROWS[split]:
        raise ValueError(f"Official split-{split} population changed: {counts}")
    metadata = json.loads(STATS.read_text())["video_metadata"]
    if set(metadata) != set(by_video):
        raise ValueError("Official label/video metadata coverage changed")
    report = {"kind": "official_stream_split_source_audit_not_training",
              "split": split,
              "annotation_sha256": {
                  "train": file_sha256(ANNOTATIONS / f"training{split}.csv"),
                  "val": file_sha256(ANNOTATIONS / f"validation{split}.csv"),
                  "actions": file_sha256(ANNOTATIONS / "actions.csv"),
              },
              "feature_source": str(LMDB),
              "metadata_sha256": file_sha256(STATS),
              "official_action_rows": dict(counts),
              "population": {}}
    env = lmdb.open(str(LMDB), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as transaction:
            for population in ("train", "val"):
                ticks = build_official_split1_ticks(population, split)
                scored = [0, 0]
                uids = [set(), set()]
                seen_videos = [set(), set()]
                missing = bad = 0
                examples = []
                identity = hashlib.sha256()
                for tick in ticks:
                    identity.update(f"{tick.video}|{tick.tick_frame}|{tick.target_actions}\n".encode())
                    for i, horizon in enumerate((2, 6)):
                        if tick.target_actions[i] < 0:
                            continue
                        scored[i] += 1
                        frame_30hz = round(tick.tick_frame * 30 / tick.fps)
                        chosen = target(by_video[tick.video], frame_30hz + 30 * horizon)
                        if (chosen is None or chosen[3] != population
                                or chosen[2] != tick.target_actions[i]):
                            raise ValueError(f"Official target mismatch at {tick.video}/{tick.tick_frame}")
                        uids[i].add(chosen[4])
                        seen_videos[i].add(tick.video)
                    for key in tick.keys:
                        payload = transaction.get(key.encode())
                        if payload is None:
                            missing += 1
                            if len(examples) < 10:
                                examples.append({"kind": "missing", "key": key})
                        elif len(payload) != 4096:
                            bad += 1
                            if len(examples) < 10:
                                examples.append({"kind": "malformed", "key": key,
                                                 "bytes": len(payload)})
                report["population"][population] = {
                    "processed_ticks": len(ticks),
                    "scored_h2": scored[0], "scored_h6": scored[1],
                    "unique_uids_h2": len(uids[0]), "unique_uids_h6": len(uids[1]),
                    "videos_scored_h2": len(seen_videos[0]),
                    "videos_scored_h6": len(seen_videos[1]),
                    "h2_h6_uid_sets_equal": uids[0] == uids[1],
                    "feature_queries": 16 * len(ticks),
                    "missing_queries": missing, "malformed_queries": bad,
                    "feature_error_examples": examples,
                    "tick_identity_sha256": identity.hexdigest(),
                }
    finally:
        env.close()
    output = MAIN_ROOT / f"outputs/b19_scalant/implementation/official_split{split}_stream_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "population": report["population"]}), flush=True)
    if any(item["missing_queries"] or item["malformed_queries"]
           for item in report["population"].values()):
        raise ValueError("Required split-specific TSN feature keys are absent or malformed")


if __name__ == "__main__":
    main()
