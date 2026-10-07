"""Read-only audit of the RU-LSTM official split-1 action stream.

This does not define a training/evaluation protocol. In particular, it keeps
the official action-instance memberships while evaluating what a shared 4s
video clock would expose for the independently trained +2s/+6s horizons.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import lmdb
import numpy as np

from .egtea_cache import ANNOTATIONS, MAIN_ROOT, load_actions
from .stream_source_audit import LMDB, STATS, stream_keys


OUTPUT = MAIN_ROOT / "outputs/b19_scalant/implementation/official_split1_stream_audit.json"


def official_actions(split: int = 1):
    if split not in (1, 2, 3):
        raise ValueError("Expected official EGTEA split 1, 2 or 3")
    mapping = load_actions()
    by_video = defaultdict(list)
    ids = {}
    for population, name in (("train", f"training{split}.csv"),
                             ("val", f"validation{split}.csv")):
        with (ANNOTATIONS / name).open(newline="") as stream:
            for row in csv.reader(stream):
                uid, video, start, stop, verb, noun, action = row
                uid, start, stop, verb, noun, action = map(
                    int, (uid, start, stop, verb, noun, action))
                if uid in ids:
                    raise ValueError(f"Duplicate official action ID {uid}")
                if not 0 <= action < len(mapping) or mapping[action] != (verb, noun):
                    raise ValueError(f"Bad official label {uid}")
                ids[uid] = population
                by_video[video].append((start, stop, action, population, uid))
    for actions in by_video.values():
        actions.sort()
    return by_video, ids


def target(actions, frame):
    covering = [action for action in actions if action[0] <= frame <= action[1]]
    if covering:
        return covering[0]
    return next((action for action in actions if action[0] >= frame), None)


def main():
    metadata = json.loads(STATS.read_text())["video_metadata"]
    actions, ids = official_actions()
    if len(ids) != 10321 or set(metadata) != set(actions):
        raise ValueError("Official split-1 row count or video metadata changed")
    report = {
        "kind": "official_split1_stream_source_audit_not_training",
        "annotation_source": str(ANNOTATIONS),
        "feature_source": str(LMDB),
        "annotation_and_feature_key_clock_fps": 30,
        "video_metadata_clock": "native source-video FPS",
        "grid_seconds": 4,
        "horizons_seconds": [2, 6],
        "official_action_instances": dict(Counter(ids.values())),
        "videos": len(actions),
        "population": {},
    }
    env = lmdb.open(str(LMDB), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            for population in ("train", "val"):
                counts = Counter()
                selected = set()
                missing = []
                bad = []
                missing_videos = Counter()
                selected_videos = set()
                max_tokens = 0
                for video, meta in sorted(metadata.items()):
                    fps, n_frames = float(meta["fps"]), int(meta["frames"])
                    step = max(1, round(4 * fps))
                    video_ticks = []
                    tick = step
                    while tick < n_frames:
                        # A target may have no action left. Such a tick can be
                        # omitted only after the final scored prediction.
                        # RU-LSTM's annotation and LMDB key clocks are 30 Hz;
                        # video_metadata uses each native source video's FPS.
                        target_frame_30hz = round(tick * 30 / fps)
                        labels = [target(actions[video], target_frame_30hz + 30 * h)
                                  if tick + round(h * fps) < n_frames else None
                                  for h in (2, 6)]
                        own = [item for item in labels if item and item[3] == population]
                        if own:
                            selected_videos.add(video)
                            for item in own:
                                selected.add(item[4])
                            counts["scored_h2"] += int(labels[0] is not None and labels[0][3] == population)
                            counts["scored_h6"] += int(labels[1] is not None and labels[1][3] == population)
                        video_ticks.append((tick, bool(own)))
                        tick += step
                    last_scored = max((i for i, (_, own) in enumerate(video_ticks) if own), default=-1)
                    if last_scored < 0:
                        continue
                    counts["processed_ticks"] += last_scored + 1
                    counts["unscored_prefix_or_gaps"] += sum(
                        not own for _, own in video_ticks[:last_scored + 1])
                    counts["terminal_ticks_omitted"] += len(video_ticks) - last_scored - 1
                    max_tokens = max(max_tokens, (last_scored + 1) * 16)
                    for tick, own in video_ticks[:last_scored + 1]:
                        counts["feature_queries"] += 16
                        for key in stream_keys(video, fps, 0, tick):
                            payload = txn.get(key.encode())
                            if payload is None:
                                counts["missing_feature_queries"] += 1
                                missing_videos[video] += 1
                                if len(missing) < 20:
                                    missing.append({"key": key, "tick_frame": tick,
                                                    "scored_tick": own})
                            elif len(payload) != 1024 * np.dtype("float32").itemsize:
                                counts["bad_feature_payloads"] += 1
                                if len(bad) < 20:
                                    bad.append(key)
                report["population"][population] = {
                    **counts,
                    "videos_with_scored_ticks": len(selected_videos),
                    "unique_official_actions_selected": len(selected),
                    "max_history_tokens": max_tokens,
                    "missing_videos": len(missing_videos),
                    "missing_examples": missing,
                    "bad_examples": bad,
                }
    finally:
        env.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
