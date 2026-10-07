"""Validation populations for the fixed-history action-instance adapter.

The covered population uses the existing 4s video clock only to choose
official action IDs. The model still makes one action-anchored prediction per
validation row; repeated streaming ticks never reweight an action here.
"""

from __future__ import annotations

import json

from .official_instance_data import InstanceRow
from .official_stream_audit import official_actions, target
from .stream_source_audit import STATS


def covered_validation_uids(horizon: int) -> frozenset[int]:
    if horizon not in (2, 6):
        raise ValueError("Expected +2s or +6s")
    metadata = json.loads(STATS.read_text())["video_metadata"]
    by_video, memberships = official_actions()
    if len(memberships) != 10321 or set(metadata) != set(by_video):
        raise ValueError("Official split-1 annotations or video metadata changed")
    covered = set()
    for video, meta in sorted(metadata.items()):
        fps, n_frames = float(meta["fps"]), int(meta["frames"])
        step = max(1, round(4 * fps))
        tick = step
        while tick < n_frames:
            if tick + round(horizon * fps) < n_frames:
                # Match official_stream_data's 30 Hz target clock and its
                # next-action fallback, including sparse unlabeled ticks.
                frame_30hz = round(tick * 30 / fps)
                item = target(by_video[video], frame_30hz + 30 * horizon)
                if item is not None and item[3] == "val":
                    covered.add(item[4])
            tick += step
    return frozenset(covered)


def validation_scopes(rows: list[InstanceRow], horizon: int) -> dict[str, tuple[int, ...]]:
    """Return ordered official and tick-covered UIDs for separate metrics."""
    if len(rows) != 2022 or len({row.uid for row in rows}) != len(rows):
        raise ValueError("Expected all 2,022 official validation instances")
    if any(row.horizon != horizon for row in rows):
        raise ValueError("Validation rows have a different anticipation horizon")
    covered = covered_validation_uids(horizon)
    all_uids = tuple(row.uid for row in rows)
    if not covered <= set(all_uids) or len(covered) != 1626:
        raise ValueError("4s tick-covered official validation population changed")
    return {
        "official_all": all_uids,
        "official_tick_covered": tuple(uid for uid in all_uids if uid in covered),
    }
