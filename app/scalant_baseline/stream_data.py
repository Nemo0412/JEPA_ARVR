"""Shared stream tick types, TSN reader and a historical split-1 index.

Formal official-label runs use official_stream_data.build_official_split1_ticks;
they do not call the temporal-half index retained in this module.

The index keeps every 4s tick through the last tick with a valid +2s or +6s
target. Later ticks with neither target are terminal video tails and are
excluded: they cannot affect any scored prediction, and the released TSN
LMDB lacks some of their features. A -1 target for one horizon is still
retained if the other horizon is valid. The validation population is the
second temporal half of each session.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lmdb
import numpy as np
import torch

from .egtea_cache import MAIN_ROOT, load_actions
from .stream_source_audit import LMDB, SOURCE, STATS, stream_keys


@dataclass(frozen=True)
class StreamTick:
    video: str
    population: str
    tick_frame: int
    fps: float
    target_actions: tuple[int, int]
    target_verbs: tuple[int, int]
    target_nouns: tuple[int, int]
    keys: tuple[str, ...]
    past_actions: tuple[int, ...]
    past_verbs: tuple[int, ...]
    past_nouns: tuple[int, ...]


def _read_intervals(source: Path = SOURCE):
    intervals = defaultdict(list)
    seen = set()
    for population in ("train", "val", "test"):
        with (source / f"EGTEA_{population}_vjepa.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                item = (row["participant_id"], int(row["start_frame"]),
                        int(row["stop_frame"]), int(row["verb_class"]),
                        int(row["noun_class"]), int(row["action_class"]))
                if item not in seen:
                    seen.add(item)
                    intervals[item[0]].append(item[1:])
    for actions in intervals.values():
        actions.sort()
    return intervals


def _target(actions, frame):
    covering = [(v, n, a) for start, stop, v, n, a in actions if start <= frame <= stop]
    if covering:
        # Match the checked-in streaming generator's first covering action.
        return covering[0]
    return next(((v, n, a) for start, _, v, n, a in actions if start >= frame), None)


def _past(actions, frame):
    pairs = {(v, n, a) for start, stop, v, n, a in actions if start <= frame <= stop}
    return next(iter(pairs)) if len(pairs) == 1 else None


def build_split1_ticks(population: str, horizon_seconds: tuple[int, int] = (2, 6),
                       stats: Path = STATS) -> list[StreamTick]:
    if population not in ("train", "val") or horizon_seconds != (2, 6):
        raise ValueError("Expected split-1 train/val and independent +2/+6 targets")
    metadata = json.loads(stats.read_text())["video_metadata"]
    intervals = _read_intervals()
    official_mapping = load_actions()

    def official(pair):
        if pair is None:
            return (-1, -1, -1)
        verb, _, action = pair
        if not 0 <= action < len(official_mapping):
            raise ValueError(f"Unmapped source action ID {action}")
        official_verb, official_noun = official_mapping[action]
        if verb != official_verb:
            raise ValueError(f"Verb mismatch at action {action}: {verb} != {official_verb}")
        return (action, official_verb, official_noun)
    ticks = []
    for video, meta in sorted(metadata.items()):
        fps, n_frames = float(meta["fps"]), int(meta["frames"])
        origin = 0 if population == "train" else n_frames // 2
        region_end = n_frames // 2 if population == "train" else n_frames
        step = max(1, round(4 * fps))
        tick = origin + step
        terminal_tail_started = False
        while tick < region_end and tick + round(6 * fps) < n_frames:
            target_pairs = [_target(intervals[video], tick + round(h * fps))
                            for h in horizon_seconds]
            # Deliberate terminal-tail exception: the RU-LSTM TSN LMDB lacks
            # some features after the final scorable action (split-1 val:
            # 1,132 such ticks). No later scored tick can use this history;
            # skipping it preserves scored predictions but reduces processed
            # ticks, final memory length and runtime versus literal 4s-to-end
            # processing. Reject any non-terminal gap below.
            if all(pair is None for pair in target_pairs):
                terminal_tail_started = True
                tick += step
                continue
            if terminal_tail_started:
                raise ValueError(f"Supervised tick after skipped tail in {video}")
            keys = tuple(stream_keys(video, fps, origin, tick))
            target_labels = [official(pair) for pair in target_pairs]
            past_pairs = [_past(intervals[video], round(int(key.rsplit('_frame_', 1)[1][:-4])
                                                       * fps / 30)) for key in keys]
            past_labels = [official(pair) for pair in past_pairs]
            ticks.append(StreamTick(
                video, population, tick, fps,
                tuple(label[0] for label in target_labels),
                tuple(label[1] for label in target_labels),
                tuple(label[2] for label in target_labels),
                keys,
                tuple(label[0] for label in past_labels),
                tuple(label[1] for label in past_labels),
                tuple(label[2] for label in past_labels),
            ))
            tick += step
    return ticks


class StreamingTSNReader:
    """Read exact 1024-D source vectors; missing positions fail closed."""

    def __init__(self, path: Path = LMDB):
        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)

    def read(self, tick: StreamTick) -> torch.Tensor:
        with self.env.begin() as txn:
            values = []
            for key in tick.keys:
                payload = txn.get(key.encode())
                if payload is None:
                    raise KeyError(f"TSN feature missing: {key}")
                array = np.frombuffer(payload, dtype=np.float32)
                if array.shape != (1024,) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid TSN vector: {key}")
                values.append(array)
        return torch.from_numpy(np.stack(values))

    def close(self) -> None:
        self.env.close()
