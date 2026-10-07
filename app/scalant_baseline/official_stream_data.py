"""Official RU-LSTM EGTEA split labels on a causal 4s video clock.

Training and validation use the released 8,299/2,022 action-instance
memberships, even though both populations contain the same source videos.
Every population starts memory at video time zero. Ticks with no label for
that population still advance memory until its last scored tick. Only ticks
after that point are dropped: released TSN features may be absent in this
terminal suffix, which changes final memory/runtime but no scored prediction.
"""

from __future__ import annotations

import json

from .official_stream_audit import official_actions, target
from .stream_data import StreamTick
from .stream_source_audit import STATS, stream_keys
from .egtea_cache import load_actions


def _past(actions, frame, population):
    # Past auxiliary targets obey the official membership boundary. Features
    # are still read for every slot, regardless of which split owns its action.
    covering = [a for a in actions if a[0] <= frame <= a[1]]
    return covering[0] if len(covering) == 1 and covering[0][3] == population else None


def build_official_split1_ticks(population: str, split: int = 1) -> list[StreamTick]:
    if population not in ("train", "val"):
        raise ValueError("Expected official train or val")
    metadata = json.loads(STATS.read_text())["video_metadata"]
    by_video, ids = official_actions(split)
    if len(ids) != 10321 or set(metadata) != set(by_video):
        raise ValueError("Official split-1 annotations or video metadata changed")
    mapping = load_actions()
    ticks = []
    for video, meta in sorted(metadata.items()):
        fps, n_frames = float(meta["fps"]), int(meta["frames"])
        step = max(1, round(4 * fps))
        video_ticks = []
        tick = step
        while tick < n_frames:
            # Official RU-LSTM action intervals use 30 FPS even when this
            # session video's native frame rate differs from 30 FPS.
            target_frame_30hz = round(tick * 30 / fps)
            target_items = [target(by_video[video], target_frame_30hz + 30 * h)
                            if tick + round(h * fps) < n_frames else None
                            for h in (2, 6)]
            owned = [item if item and item[3] == population else None
                     for item in target_items]
            keys = tuple(stream_keys(video, fps, 0, tick))
            past = [_past(by_video[video],
                          int(key.rsplit("_frame_", 1)[1][:-4]),
                          population) for key in keys]

            def labels(items):
                action = tuple(item[2] if item else -1 for item in items)
                verb = tuple(mapping[a][0] if a >= 0 else -1 for a in action)
                noun = tuple(mapping[a][1] if a >= 0 else -1 for a in action)
                return action, verb, noun

            future_a, future_v, future_n = labels(owned)
            past_a, past_v, past_n = labels(past)
            video_ticks.append(StreamTick(video, population, tick, fps,
                                          future_a, future_v, future_n, keys,
                                          past_a, past_v, past_n))
            tick += step
        # Skip only the terminal suffix after this population's final scored
        # prediction. Some released TSN keys are absent there; no later
        # prediction can consume it, but final memory/runtime become shorter.
        last_scored = max((i for i, t in enumerate(video_ticks)
                           if any(a >= 0 for a in t.target_actions)), default=-1)
        if last_scored >= 0:
            ticks.extend(video_ticks[:last_scored + 1])
    return ticks
