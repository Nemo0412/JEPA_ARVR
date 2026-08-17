"""EGTEA Gaze+ reader used by the opencl@29063fe dataset adapter.

This is deliberately isolated from the immutable upstream tree.  It maps
EGTEA's per-session BeGaze exports onto the upstream ``GazeRecord`` contract;
it does not change gaze rasterization, CA, pruning, predictor, or FFN logic.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FPS = 24.0
GAZE_W, GAZE_H = 1280.0, 960.0


def _str2frame(frame_str, fps=FPS):
    h, m, s, f = frame_str.split(":")
    return int((3600 * int(h) + 60 * int(m) + int(s)) * fps + int(f))


def parse_gtea_gaze(filename):
    """Return [session_frame, (x_norm, y_norm, type)] from BeGaze text."""
    lines = [ln.rstrip("\n") for ln in open(filename, encoding="latin-1")]
    hdr = next((i for i, ln in enumerate(lines) if ln.startswith("Time\t") or ln.split()[:1] == ["Time"]), None)
    if hdr is None:
        raise ValueError(f"no data header in {filename}")
    records, max_frame = [], 0
    for ln in lines[hdr + 1:]:
        value = ln.split()
        if len(value) == 7:
            px, py, frame, gtype = float(value[3]), float(value[4]), int(value[5]), value[6]
        elif len(value) == 26:
            px, py, frame, gtype = float(value[5]), float(value[6]), _str2frame(value[-2]), value[-1]
        else:
            continue
        if frame < 0:
            continue
        records.append((frame, px, py, gtype))
        max_frame = max(max_frame, frame)
    arr = np.zeros((max_frame + 1, 3), dtype=np.float32)
    for frame, px, py, gtype in records:
        if arr[frame, 2] > 0:
            arr[frame, 0] = (arr[frame, 0] + px) / 2.0
            arr[frame, 1] = (arr[frame, 1] + py) / 2.0
        else:
            arr[frame, 0], arr[frame, 1] = px, py
        arr[frame, 2] = 1 if gtype == "Fixation" else 2 if gtype == "Saccade" else 3
    px, py = arr[:, 0], arr[:, 1]
    arr[(px < 0) | (px > GAZE_W - 1) | (py < 0) | (py > GAZE_H - 1), 2] = 4
    arr[:, 0] = np.clip(px, 0, GAZE_W - 1) / GAZE_W
    arr[:, 1] = np.clip(py, 0, GAZE_H - 1) / GAZE_H
    return arr


def parse_clip_name(clip_name):
    """``<session>-...-F<start>-F<end>`` -> (session, start, end)."""
    parts = clip_name.rsplit("-", 4)
    if len(parts) != 5:
        raise ValueError(f"unexpected clip name: {clip_name}")
    return parts[0], int(parts[3].lstrip("F")), int(parts[4].lstrip("F"))


class EgteaGazeSource:
    """Lazy per-session BeGaze cache with a frozen OOB fallback allowlist."""

    def __init__(self, gaze_dir, record_cls, approved_center_rows=None):
        self.gaze_dir = Path(gaze_dir)
        self.record_cls = record_cls
        self._sessions: dict[str, np.ndarray | None] = {}
        self.approved_center_rows = {
            str(session): {(int(row["start_frame"]), int(row["last_frame"])) for row in rows}
            for session, rows in (approved_center_rows or {}).items()
        }
        self._logged_center_rows: set[tuple[str, int, int]] = set()

    def _session_gaze(self, session):
        if session not in self._sessions:
            try:
                self._sessions[session] = parse_gtea_gaze(self.gaze_dir / f"{session}.txt")
            except Exception as exc:
                logger.warning("EGTEA gaze parse failed for %s: %s", session, exc)
                self._sessions[session] = None
        return self._sessions[session]

    @staticmethod
    def _clean_xy(gaze):
        xy = gaze[:, :2].astype(np.float64, copy=True)
        bad = ~np.isfinite(xy).all(axis=1) | ((xy == 0).all(axis=1))
        xy[bad] = 0.5
        return np.clip(xy, 0.0, 1.0)

    def _record(self, xy):
        ts_us = (np.arange(len(xy), dtype=np.float64) / FPS) * 1e6
        return self.record_cls(timestamps_us=ts_us, xy_norm=xy, yaw=None, pitch=None, sync=None)

    def record_for_clip(self, clip_name):
        session, start, end = parse_clip_name(clip_name)
        gaze = self._session_gaze(session)
        clip_len = max(1, end - start + 1)
        if gaze is None or end >= len(gaze):
            xy = np.full((clip_len, 2), 0.5, dtype=np.float64)
        else:
            xy = self._clean_xy(gaze[start:end + 1])
        # Fixed clips retain their absolute session timeline.
        return self.record_cls(
            timestamps_us=(np.arange(start, start + len(xy), dtype=np.float64) / FPS) * 1e6,
            xy_norm=xy,
            yaw=None,
            pitch=None,
            sync=None,
        )

    def record_for_session(self, session):
        gaze = self._session_gaze(session)
        if gaze is None:
            raise RuntimeError(f"EGTEA gaze unavailable for session {session}")
        xy = self._clean_xy(gaze)
        approved = self.approved_center_rows.get(str(session), set())
        if approved:
            max_last = max(last for _, last in approved)
            if max_last >= len(xy):
                xy = np.concatenate([xy, np.full((max_last + 1 - len(xy), 2), 0.5, dtype=np.float64)])
        return self._record(xy)

    def assert_session_frame_indices(self, session, frame_indices):
        gaze = self._session_gaze(session)
        if gaze is None:
            raise RuntimeError(f"EGTEA gaze unavailable for session {session}")
        frames = np.asarray(frame_indices, dtype=np.int64)
        if frames.size == 0 or int(frames.max()) < len(gaze):
            return
        key = (int(frames.min()), int(frames.max()))
        if key not in self.approved_center_rows.get(str(session), set()):
            raise RuntimeError(
                "EGTEA gaze OOB is not in the approved B13 center-fallback manifest: "
                f"session={session} sampled=[{key[0]},{key[1]}] gaze_last={len(gaze) - 1}"
            )
        log_key = (str(session), *key)
        if log_key not in self._logged_center_rows:
            logger.warning(
                "B13 approved center-gaze fallback session=%s sampled=[%d,%d] real_gaze_last=%d",
                session, key[0], key[1], len(gaze) - 1,
            )
            self._logged_center_rows.add(log_key)
