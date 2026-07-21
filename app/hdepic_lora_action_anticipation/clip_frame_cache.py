"""Persistent decoded-clip cache to cut video-decode stalls (GPU util).

Bottleneck for tri-modal fusion-only on H100: ``decord`` ``get_batch`` ≈10–20s per
batch of 32, while the GPU step is ≈4s → AveUtil stays ~20–30% even with perfect
prefetch. Caching uint8 RGB clips on scratch replaces decode with a single
``np.load`` (~tens of ms) after warmup, so GPU can stay busy (≥60%).

Cache key = video_id + frame index list (deterministic when anticipation is fixed).
Enable with env ``TRI_MODAL_FRAME_CACHE=/path/to/dir``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_STATS = {"hit": 0, "miss": 0, "logged": 0}


def frame_cache_root() -> Path | None:
    raw = os.environ.get("TRI_MODAL_FRAME_CACHE", "").strip()
    if not raw:
        return None
    return Path(raw)


def _clip_path(root: Path, video_id: str, indices: np.ndarray) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(video_id))
    digest = hashlib.sha1(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[:20]
    return root / safe_id / f"{digest}.npy"


def load_or_decode_clip(
    *,
    video_id: str,
    indices: np.ndarray,
    decode_fn,
) -> np.ndarray:
    """Return uint8 THWC clip; ``decode_fn`` called only on cache miss.

    ``decode_fn`` should return a numpy array shaped (T, H, W, C), uint8.
    """
    root = frame_cache_root()
    if root is None:
        return decode_fn()

    path = _clip_path(root, video_id, indices)
    if path.is_file():
        try:
            buf = np.load(path, mmap_mode="r")
            out = np.asarray(buf)
            if out.ndim == 4:
                _STATS["hit"] += 1
                _maybe_log()
                return out.copy() if not out.flags.writeable else out
        except Exception:
            logger.debug("frame-cache load failed for %s; re-decoding", path, exc_info=True)

    buf = decode_fn()
    _STATS["miss"] += 1
    _maybe_log()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so concurrent workers never read a partial file.
        # np.save always appends ".npy" if the path does not already end with it.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".npy", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with open(tmp_path, "wb") as f:
                np.save(f, np.asarray(buf, dtype=np.uint8))
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    except Exception:
        logger.debug("frame-cache store failed for %s", path, exc_info=True)
    return buf


def _maybe_log() -> None:
    total = _STATS["hit"] + _STATS["miss"]
    if total == 0 or total % 64 != 0:
        return
    if _STATS["logged"] == total:
        return
    _STATS["logged"] = total
    hit = _STATS["hit"]
    miss = _STATS["miss"]
    rate = 100.0 * hit / max(1, total)
    logger.info(
        "TRI_MODAL_FRAME_CACHE stats: hit=%d miss=%d hit_rate=%.1f%% root=%s",
        hit,
        miss,
        rate,
        frame_cache_root(),
    )


def cache_path_for(video_id: str, indices: np.ndarray) -> Path | None:
    root = frame_cache_root()
    if root is None:
        return None
    return _clip_path(root, video_id, indices)
