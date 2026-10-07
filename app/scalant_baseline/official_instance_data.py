"""Fixed 32s EGTEA action-instance inputs for a ScalAnt feasibility study.

The source EK100 loader forms one sample per annotated action and pads early
history. Here the last 32s are 128 TSN positions, with the final 16 positions
forming the user's 4s working window and the preceding 112 the 28s memory.
The source loader includes the feature at the anticipation boundary; unlike
the separate periodic-tick protocol, samples are anchored to action starts.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass

from .egtea_cache import ANNOTATIONS, load_actions
from .official_stream_audit import official_actions


FEATURE_FPS = 4
ANNOTATION_FPS = 30
HISTORY_STEPS = 128
WORK_STEPS = 16


@dataclass(frozen=True)
class InstanceRow:
    uid: int
    video: str
    action: int
    verb: int
    noun: int
    horizon: int
    anchor_frame_30hz: int
    keys: tuple[str, ...]
    past_actions: tuple[int, ...]
    past_verbs: tuple[int, ...]
    past_nouns: tuple[int, ...]


def _known_past(actions, frame: int, population: str):
    # EGTEA's sparse labels do not define a background class: only a single
    # action owned by this official split may supervise an auxiliary past slot.
    # Unknown, overlapping, and other-split slots retain their features but
    # carry -1 and must be ignored by the loss.
    covering = [a for a in actions if a[0] <= frame <= a[1]]
    return covering[0] if len(covering) == 1 and covering[0][3] == population else None


def build_instance_rows(population: str, horizon: int):
    if population not in ("train", "val") or horizon not in (2, 6):
        raise ValueError("Expected official split-1 train/val and +2s/+6s")
    by_video, ids = official_actions()
    mapping = load_actions()
    if len(ids) != 10321:
        raise ValueError("Official split-1 membership changed")
    name = "training1.csv" if population == "train" else "validation1.csv"
    rows, early = [], []
    with (ANNOTATIONS / name).open(newline="") as source:
        for item in csv.reader(source):
            uid, video, start, stop, verb, noun, action = item
            uid, start, verb, noun, action = map(int, (uid, start, verb, noun, action))
            anchor_frame = start - horizon * ANNOTATION_FPS
            # The original convert_to_anticipation discards samples whose
            # anticipation boundary precedes its first 30 Hz source frame.
            if anchor_frame < 1:
                early.append(uid)
                continue
            if ids[uid] != population or mapping[action] != (verb, noun):
                raise ValueError(f"Official label/membership mismatch at {uid}")
            # Match the source loader's inclusive 4 Hz anticipation boundary.
            last_index = math.floor(start * FEATURE_FPS / ANNOTATION_FPS) - horizon * FEATURE_FPS
            indices = [max(0, last_index - HISTORY_STEPS + 1 + i)
                       for i in range(HISTORY_STEPS)]
            frames = [max(1, math.floor(index * ANNOTATION_FPS / FEATURE_FPS))
                      for index in indices]
            keys = tuple(f"{video}_frame_{frame:010d}.jpg" for frame in frames)
            past = [_known_past(by_video[video], frame, population)
                    for frame in frames]
            labels = tuple(a[2] if a is not None else -1 for a in past)
            verbs = tuple(mapping[a][0] if a >= 0 else -1 for a in labels)
            nouns = tuple(mapping[a][1] if a >= 0 else -1 for a in labels)
            rows.append(InstanceRow(uid, video, action, verb, noun, horizon,
                                    anchor_frame, keys, labels, verbs, nouns))
    expected = 8299 if population == "train" else 2022
    if len(rows) + len(early) != expected or len({r.uid for r in rows}) != len(rows):
        raise ValueError("Official action-instance coverage changed")
    return rows, early
