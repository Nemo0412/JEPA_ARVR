"""Audit split-1 RU-LSTM TSN coverage for a 4s EGTEA stream grid.

This is a data-source diagnostic. It does not define or execute training.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import numpy as np

from .egtea_cache import MAIN_ROOT, load_actions


SOURCE = Path(os.environ.get("SCALANT_LEGACY_ANNOTATIONS", str(
    MAIN_ROOT / "data/egtea/vjepa_annotations/v1/split1"
))).expanduser().resolve()
# Formal runs read only video_metadata (native FPS/frame count) from STATS.
# The historical filename does not select the temporal-half label protocol.
STATS = Path(os.environ.get("SCALANT_VIDEO_METADATA", str(
    MAIN_ROOT / "data/egtea/scalant_video_metadata.json"
))).expanduser().resolve()
# Preserve the completed experiments: all three official label splits used
# this same s1 RGB TSN bundle. Changing its producer is a new experiment.
LMDB = Path(os.environ.get("SCALANT_TSN_FEATURES", str(
    MAIN_ROOT / "data/egtea/rulstm_features/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd"
))).expanduser().resolve()
OUTPUT = MAIN_ROOT / "outputs/b19_scalant/implementation/split1_stream_source_audit.json"


def stream_keys(video: str, fps: float, origin: int, tick: int) -> list[str]:
    """Sixteen causal 4 Hz positions in the current 4s block, TSN's 30 fps keys."""
    start_seconds = (tick - origin) / fps - 4.0 + origin / fps
    frames = [max(1, math.floor((start_seconds + step * .25) * 30))
              for step in range(16)]
    return [f"{video}_frame_{frame:010d}.jpg" for frame in frames]


def main() -> None:
    metadata = json.loads(STATS.read_text())["video_metadata"]
    mapping = load_actions()
    intervals = defaultdict(list)
    seen = set()
    for population in ("train", "val", "test"):
        with (SOURCE / f"EGTEA_{population}_vjepa.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                item = (row["participant_id"], int(row["start_frame"]),
                        int(row["stop_frame"]), int(row["verb_class"]),
                        int(row["noun_class"]), int(row["action_class"]))
                if item in seen:
                    continue
                seen.add(item)
                intervals[item[0]].append(item[1:])
    for actions in intervals.values():
        actions.sort()

    env = lmdb.open(str(LMDB), readonly=True, lock=False, readahead=False)
    report = {"kind": "data_source_audit_not_training", "split": 1,
              "feature_source": str(LMDB), "grid_seconds": 4,
              "chunk_seconds": 4, "feature_fps": 4, "horizons_seconds": [2, 6],
              "n_sessions": len(metadata), "populations": {}}
    with env.begin() as txn:
        for population in ("train", "val"):
            counters = Counter()
            missing = []
            mismatched_verbs = Counter()
            remapped_nouns = Counter()
            unknown_actions = Counter()
            session_missing = Counter()
            for video, meta in sorted(metadata.items()):
                fps, n_frames = float(meta["fps"]), int(meta["frames"])
                origin = 0 if population == "train" else n_frames // 2
                region_end = n_frames // 2 if population == "train" else n_frames
                tick = origin + round(4 * fps)
                step = max(1, round(4 * fps))
                while tick + round(6 * fps) < n_frames and tick < region_end:
                    counters["candidate_ticks"] += 1
                    targets = []
                    for horizon in (2, 6):
                        target = tick + round(horizon * fps)
                        selected = next(((v, n, a) for sf, ef, v, n, a in intervals[video]
                                         if sf <= target <= ef), None)
                        if selected is None:
                            selected = next(((v, n, a) for sf, _, v, n, a in intervals[video]
                                             if sf >= target), None)
                        targets.append(selected)
                    # The model must process every tick so its memory keeps
                    # growing, even when there is no supervised target.
                    if any(target is not None for target in targets):
                        counters["supervised_ticks"] += 1
                    for target in targets:
                        if target is not None:
                            counters["valid_target_pairs"] += 1
                            verb, noun, action = target
                            if not 0 <= action < len(mapping):
                                unknown_actions[str(action)] += 1
                            else:
                                official_verb, official_noun = mapping[action]
                                if verb != official_verb:
                                    mismatched_verbs[str((action, verb, official_verb))] += 1
                                if noun != official_noun:
                                    remapped_nouns[str((action, noun, official_noun))] += 1
                    for key in stream_keys(video, fps, origin, tick):
                        counters["feature_queries"] += 1
                        payload = txn.get(key.encode())
                        if payload is None:
                            counters["missing_feature_queries"] += 1
                            session_missing[video] += 1
                            if len(missing) < 20:
                                missing.append(key)
                        elif len(payload) != 1024 * np.dtype("float32").itemsize:
                            counters["bad_payload_bytes"] += 1
                    tick += step
            report["populations"][population] = {
                **counters, "missing_examples": missing,
                "sessions_with_missing": len(session_missing),
                "missing_by_session_top10": session_missing.most_common(10),
                "unknown_action_ids": dict(unknown_actions),
                "mismatched_verbs": dict(mismatched_verbs),
                "noun_id_remaps": dict(remapped_nouns),
            }
    env.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
