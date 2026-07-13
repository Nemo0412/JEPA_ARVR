"""SLAM head/device pose loading for HD-EPIC action anticipation.

Reads ``closed_loop_trajectory.csv`` from per-session SLAM zip archives. IMU
signals are already fused into the SLAM output (6DoF pose, angular/linear
velocity, gravity); there is no separate raw accelerometer/gyro CSV in HD-EPIC.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate, _as_path, _clean_video_id, _find_first

logger = logging.getLogger(__name__)

TRAjectory_COLUMNS = {
    "timestamp": "tracking_timestamp_us",
    "tx": "tx_world_device",
    "ty": "ty_world_device",
    "tz": "tz_world_device",
    "qx": "qx_world_device",
    "qy": "qy_world_device",
    "qz": "qz_world_device",
    "qw": "qw_world_device",
    "avx": "angular_velocity_x_device",
    "avy": "angular_velocity_y_device",
    "avz": "angular_velocity_z_device",
    "lvx": "device_linear_velocity_x_device",
    "lvy": "device_linear_velocity_y_device",
    "lvz": "device_linear_velocity_z_device",
    "quality": "quality_score",
}

USECOLS = list(dict.fromkeys(TRAjectory_COLUMNS.values()))
CHUNK_ROWS = 50_000


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return q / norms


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """First two columns of rotation matrix (Zhou et al. 6D rep)."""
    q = _quat_normalize(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    return np.stack([r00, r01, r02, r10, r11, r12], axis=-1)


def feature_dim_for_set(feature_set: str) -> int:
    fs = str(feature_set).lower()
    if fs == "pose_6d":
        return 9  # delta translation (3) + rot6d (6)
    if fs == "pose_vel":
        return 12  # pose_6d + angular velocity (3)
    if fs == "pose_full":
        return 15  # pose_vel + linear velocity (3)
    raise ValueError(f"Unknown pose feature_set={feature_set!r}")


def window_smooth_pose_matrix(feats: np.ndarray, k_max: int) -> np.ndarray:
    """Fit a pose segment ``[N, D]`` to ``[K_max, D]`` using contiguous-window means.

    When SLAM has a higher sample rate than the target matrix, this reduces the
    pose rate by averaging over temporal windows instead of picking or truncating
    samples. Short segments are copied then zero-padded.
    """
    d = int(feats.shape[1]) if feats.ndim == 2 and feats.size else 0
    if d == 0:
        raise ValueError("window_smooth_pose_matrix expects feats with shape [N, D] and D > 0")
    out = np.zeros((int(k_max), d), dtype=np.float32)
    if feats.size == 0:
        return out
    feats = feats.astype(np.float32, copy=False)
    n = int(feats.shape[0])
    if n <= int(k_max):
        out[:n] = feats
        return out
    edges = np.linspace(0, n, int(k_max) + 1)
    for idx in range(int(k_max)):
        lo = int(np.floor(edges[idx]))
        hi = int(np.floor(edges[idx + 1]))
        if hi <= lo:
            hi = min(n, lo + 1)
        out[idx] = feats[lo:hi].mean(axis=0, dtype=np.float32)
    return out


def pad_or_truncate_pose_matrix(feats: np.ndarray, k_max: int) -> np.ndarray:
    """Backward-compatible alias for the smoothed fixed-size pose matrix."""
    return window_smooth_pose_matrix(feats, k_max)


def build_pose_features(
    translation: np.ndarray,
    quaternion: np.ndarray,
    angular_vel: np.ndarray | None,
    linear_vel: np.ndarray | None,
    feature_set: str,
) -> np.ndarray:
    """Build clip-relative pose feature matrix ``[N, D]``."""
    n = translation.shape[0]
    if n == 0:
        return np.zeros((0, feature_dim_for_set(feature_set)), dtype=np.float32)

    t0 = translation[0:1]
    q0 = quaternion[0:1]
    delta_t = (translation - t0).astype(np.float32)
    q_rel = _quat_mul(_quat_conj(q0), quaternion)
    rot6d = _quat_to_rot6d(q_rel).astype(np.float32)

    parts = [delta_t, rot6d]
    fs = str(feature_set).lower()
    if fs in {"pose_vel", "pose_full"}:
        if angular_vel is None:
            av = np.zeros((n, 3), dtype=np.float32)
        else:
            av = angular_vel.astype(np.float32)
        parts.append(av)
    if fs == "pose_full":
        if linear_vel is None:
            lv = np.zeros((n, 3), dtype=np.float32)
        else:
            lv = linear_vel.astype(np.float32)
        parts.append(lv)
    feats = np.concatenate(parts, axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class PoseRecord:
    timestamps_us: np.ndarray
    translation: np.ndarray
    quaternion: np.ndarray
    angular_vel: np.ndarray | None
    linear_vel: np.ndarray | None
    quality: np.ndarray | None


def _dataframe_to_pose_record(df: pd.DataFrame, quality_min: float) -> PoseRecord | None:
    if df.empty:
        return None
    lower = {c.lower(): c for c in df.columns}
    t_col = lower.get(TRAjectory_COLUMNS["timestamp"]) or next(
        (c for c in df.columns if "timestamp" in c.lower()), None
    )
    if t_col is None:
        return None

    def col(key: str) -> np.ndarray | None:
        name = lower.get(TRAjectory_COLUMNS[key])
        if name is None:
            return None
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)

    ts = col("timestamp")
    if ts is None:
        return None
    tx, ty, tz = col("tx"), col("ty"), col("tz")
    qx, qy, qz, qw = col("qx"), col("qy"), col("qz"), col("qw")
    if any(x is None for x in (tx, ty, tz, qx, qy, qz, qw)):
        return None
    translation = np.stack([tx, ty, tz], axis=1)
    quaternion = np.stack([qx, qy, qz, qw], axis=1)
    avx, avy, avz = col("avx"), col("avy"), col("avz")
    angular_vel = np.stack([avx, avy, avz], axis=1) if all(x is not None for x in (avx, avy, avz)) else None
    lvx, lvy, lvz = col("lvx"), col("lvy"), col("lvz")
    linear_vel = np.stack([lvx, lvy, lvz], axis=1) if all(x is not None for x in (lvx, lvy, lvz)) else None
    quality = col("quality")

    valid = np.isfinite(ts) & np.isfinite(translation).all(axis=1) & np.isfinite(quaternion).all(axis=1)
    if quality is not None:
        valid = valid & np.isfinite(quality) & (quality >= quality_min)
    if not valid.any():
        return None
    order = np.argsort(ts[valid], kind="stable")
    idx = np.where(valid)[0][order]
    return PoseRecord(
        timestamps_us=ts[idx],
        translation=translation[idx],
        quaternion=quaternion[idx],
        angular_vel=angular_vel[idx] if angular_vel is not None else None,
        linear_vel=linear_vel[idx] if linear_vel is not None else None,
        quality=quality[idx] if quality is not None else None,
    )


def _concat_records(records: list[PoseRecord]) -> PoseRecord | None:
    if not records:
        return None
    if len(records) == 1:
        return records[0]
    ts = np.concatenate([r.timestamps_us for r in records])
    translation = np.concatenate([r.translation for r in records])
    quaternion = np.concatenate([r.quaternion for r in records])
    angular_vel = records[0].angular_vel
    if angular_vel is not None:
        angular_vel = np.concatenate([r.angular_vel for r in records if r.angular_vel is not None])
    linear_vel = records[0].linear_vel
    if linear_vel is not None:
        linear_vel = np.concatenate([r.linear_vel for r in records if r.linear_vel is not None])
    quality = records[0].quality
    if quality is not None:
        quality = np.concatenate([r.quality for r in records if r.quality is not None])
    order = np.argsort(ts, kind="stable")
    return PoseRecord(
        timestamps_us=ts[order],
        translation=translation[order],
        quaternion=quaternion[order],
        angular_vel=angular_vel[order] if angular_vel is not None else None,
        linear_vel=linear_vel[order] if linear_vel is not None else None,
        quality=quality[order] if quality is not None else None,
    )


class SlamPoseLoader:
    """Lazy loader for SLAM closed-loop trajectories keyed by video_id."""

    def __init__(self, cfg: dict[str, Any], gate: GazeTokenGate | None = None):
        self.slam_root = _as_path(cfg.get("slam_root"))
        self.mapping_json = _as_path(cfg.get("mapping_json"))
        self.trajectory_file = str(cfg.get("trajectory_file", "closed_loop_trajectory.csv"))
        self.feature_set = str(cfg.get("feature_set", "pose_6d"))
        self.quality_min = float(cfg.get("quality_min", 0.0))
        self.history_sec = float(cfg.get("history_sec", 0.0))
        self.gate = gate or GazeTokenGate({"mode": "none", "gaze_root": cfg.get("gaze_root")})
        self._session_map: dict[str, str] | None = None
        self._inner_path_cache: dict[str, str] = {}
        self._session_record_cache: dict[str, PoseRecord | None] = {}
        self._sync_cache: dict[str, pd.DataFrame | None] = {}
        self.cache_sessions = bool(cfg.get("cache_sessions", True))

    @property
    def input_dim(self) -> int:
        return feature_dim_for_set(self.feature_set)

    def _load_session_map(self) -> dict[str, str]:
        if self._session_map is not None:
            return self._session_map
        if self.mapping_json is None or not self.mapping_json.exists():
            self._session_map = {}
            return self._session_map
        raw = json.loads(self.mapping_json.read_text(encoding="utf-8"))
        out: dict[str, str] = {}
        for vrs_path, session_id in raw.items():
            name = Path(vrs_path).stem
            if "-" in name:
                participant, rest = name.split("-", 1)
                out[f"{participant}_{rest}"] = str(session_id)
        self._session_map = out
        return out

    def resolve_session_id(self, video_id: str) -> str | None:
        clean = _vjepa_video_id(video_id)
        return self._load_session_map().get(clean)

    def _zip_path_for_session(self, session_id: str) -> Path | None:
        if self.slam_root is None:
            return None
        candidate = self.slam_root / f"{session_id}.zip"
        if candidate.exists():
            return candidate
        return _find_first(self.slam_root, [f"**/{session_id}.zip"])

    def _inner_csv_path(self, zip_path: Path, session_id: str) -> str | None:
        key = str(zip_path)
        if key in self._inner_path_cache:
            return self._inner_path_cache[key]
        inner = f"{session_id}/slam/{self.trajectory_file}"
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                if inner not in names:
                    inner = next((n for n in names if n.endswith(self.trajectory_file)), None)
                if inner is None:
                    return None
                self._inner_path_cache[key] = inner
                return inner
        except Exception as exc:
            logger.warning("Failed resolving SLAM trajectory path in %s: %s", zip_path, exc)
            return None

    def has_pose_source(self, video_id: str) -> bool:
        session_id = self.resolve_session_id(video_id)
        if session_id is None:
            return False
        zip_path = self._zip_path_for_session(session_id)
        if zip_path is None:
            return False
        return self._inner_csv_path(zip_path, session_id) is not None

    def _stream_pose_window(
        self,
        zip_path: Path,
        inner: str,
        q0_us: float,
        q1_us: float,
    ) -> PoseRecord | None:
        """Stream only rows whose timestamps fall inside ``[q0_us, q1_us]``."""
        t_col = TRAjectory_COLUMNS["timestamp"]
        parts: list[PoseRecord] = []
        seen_past_window = False
        try:
            with zipfile.ZipFile(zip_path) as zf:
                with zf.open(inner) as fh:
                    reader = pd.read_csv(
                        io.TextIOWrapper(fh, encoding="utf-8"),
                        chunksize=CHUNK_ROWS,
                        usecols=lambda c: c in USECOLS,
                    )
                    for chunk in reader:
                        ts = pd.to_numeric(chunk[t_col], errors="coerce").to_numpy(dtype=np.float64)
                        if ts.size == 0:
                            continue
                        if np.nanmax(ts) < q0_us:
                            continue
                        if np.nanmin(ts) > q1_us:
                            seen_past_window = True
                            break
                        mask = (ts >= q0_us) & (ts <= q1_us)
                        if mask.any():
                            sub = chunk.loc[mask].copy()
                            rec = _dataframe_to_pose_record(sub, self.quality_min)
                            if rec is not None:
                                parts.append(rec)
                        if np.nanmin(ts) > q1_us:
                            seen_past_window = True
                            break
        except Exception as exc:
            logger.warning("Failed streaming SLAM trajectory from %s: %s", zip_path, exc)
            return None
        if not parts and not seen_past_window:
            return None
        return _concat_records(parts)

    def _load_full_session_record(self, zip_path: Path, inner: str, session_id: str) -> PoseRecord | None:
        """Load and cache the full SLAM trajectory for a session (amortize zip reads)."""
        if session_id in self._session_record_cache:
            return self._session_record_cache[session_id]
        rec: PoseRecord | None = None
        try:
            with zipfile.ZipFile(zip_path) as zf:
                with zf.open(inner) as fh:
                    df = pd.read_csv(
                        io.TextIOWrapper(fh, encoding="utf-8"),
                        usecols=lambda c: c in USECOLS,
                    )
            rec = _dataframe_to_pose_record(df, self.quality_min)
        except Exception as exc:
            logger.warning("Failed loading SLAM trajectory from %s: %s", zip_path, exc)
        self._session_record_cache[session_id] = rec
        return rec

    @staticmethod
    def _slice_record(record: PoseRecord, q0_us: float, q1_us: float) -> PoseRecord | None:
        ts = record.timestamps_us
        mask = (ts >= q0_us) & (ts <= q1_us)
        if not mask.any():
            return None
        idx = np.where(mask)[0]
        return PoseRecord(
            timestamps_us=ts[idx],
            translation=record.translation[idx],
            quaternion=record.quaternion[idx],
            angular_vel=record.angular_vel[idx] if record.angular_vel is not None else None,
            linear_vel=record.linear_vel[idx] if record.linear_vel is not None else None,
            quality=record.quality[idx] if record.quality is not None else None,
        )

    def _sync_for_video(self, video_id: str) -> pd.DataFrame | None:
        clean_id = _clean_video_id(video_id)
        if clean_id in self._sync_cache:
            return self._sync_cache[clean_id]
        sync = None
        if self.gate.sync_root is not None:
            sync_path = _find_first(
                self.gate.sync_root,
                [f"{clean_id}_mp4_to_vrs_time_ns.csv", f"*{clean_id}*mp4_to_vrs_time_ns.csv"],
            )
            if sync_path is not None:
                sync = pd.read_csv(sync_path)
        self._sync_cache[clean_id] = sync
        return sync

    def _clip_time_us(self, meta) -> tuple[float, float] | None:
        frame_indices = meta.get("frame_indices")
        if frame_indices is None:
            return None
        if hasattr(frame_indices, "detach"):
            frame_indices = frame_indices.detach().cpu().numpy()
        frame_indices = np.asarray(frame_indices, dtype=np.float64)
        vfps = meta.get("vfps", 30.0)
        if hasattr(vfps, "detach"):
            vfps = float(vfps.detach().cpu())
        vfps = float(vfps)
        if frame_indices.size < 2 or vfps <= 0:
            return None

        mp4_t1_ns = float(frame_indices.max()) / vfps * 1e9
        if self.history_sec > 0.0:
            mp4_t0_ns = max(0.0, mp4_t1_ns - self.history_sec * 1e9)
        else:
            mp4_t0_ns = float(frame_indices.min()) / vfps * 1e9

        video_id = str(meta.get("video_id"))
        sync = self._sync_for_video(video_id)

        if sync is not None and {"mp4_time_ns", "vrs_device_time_ns"}.issubset(sync.columns):
            sync_mp4 = sync["mp4_time_ns"].to_numpy(dtype=np.float64)
            sync_vrs = sync["vrs_device_time_ns"].to_numpy(dtype=np.float64)
            vrs = np.interp([mp4_t0_ns, mp4_t1_ns], sync_mp4, sync_vrs)
            q_us = vrs / 1000.0
        else:
            q_us = np.array([mp4_t0_ns, mp4_t1_ns]) / 1000.0
        return float(q_us[0]), float(q_us[1])

    def _load_clip_record(self, meta) -> PoseRecord | None:
        """Load SLAM pose samples for the clip observed-context window."""
        video_id = str(meta.get("video_id"))
        session_id = self.resolve_session_id(video_id)
        if session_id is None:
            return None
        zip_path = self._zip_path_for_session(session_id)
        if zip_path is None:
            return None
        inner = self._inner_csv_path(zip_path, session_id)
        if inner is None:
            return None
        window = self._clip_time_us(meta)
        if window is None:
            return None
        q0, q1 = window
        if self.cache_sessions:
            full = self._load_full_session_record(zip_path, inner, session_id)
            record = None if full is None else self._slice_record(full, q0, q1)
        else:
            record = self._stream_pose_window(zip_path, inner, q0, q1)
        if record is None or record.timestamps_us.size < 2:
            return None
        return record

    def frame_timestamps_us(self, meta) -> np.ndarray | None:
        """Map each clip ``frame_indices`` entry to VRS/device time in microseconds."""
        frame_indices = meta.get("frame_indices")
        if frame_indices is None:
            return None
        if hasattr(frame_indices, "detach"):
            frame_indices = frame_indices.detach().cpu().numpy()
        frame_indices = np.asarray(frame_indices, dtype=np.float64)
        vfps = meta.get("vfps", 30.0)
        if hasattr(vfps, "detach"):
            vfps = float(vfps.detach().cpu())
        vfps = float(vfps)
        if frame_indices.size < 2 or vfps <= 0:
            return None

        mp4_ns = frame_indices / vfps * 1e9
        video_id = str(meta.get("video_id"))
        sync = self._sync_for_video(video_id)
        if sync is not None and {"mp4_time_ns", "vrs_device_time_ns"}.issubset(sync.columns):
            vrs_ns = np.interp(
                mp4_ns,
                sync["mp4_time_ns"].to_numpy(dtype=np.float64),
                sync["vrs_device_time_ns"].to_numpy(dtype=np.float64),
            )
            return (vrs_ns / 1000.0).astype(np.float64)
        return (mp4_ns / 1000.0).astype(np.float64)

    @staticmethod
    def _slice_record_interval(record: PoseRecord, t0_us: float, t1_us: float) -> PoseRecord | None:
        ts = record.timestamps_us
        mask = (ts >= t0_us) & (ts < t1_us)
        if not mask.any():
            return None
        idx = np.where(mask)[0]
        return PoseRecord(
            timestamps_us=ts[idx],
            translation=record.translation[idx],
            quaternion=record.quaternion[idx],
            angular_vel=record.angular_vel[idx] if record.angular_vel is not None else None,
            linear_vel=record.linear_vel[idx] if record.linear_vel is not None else None,
            quality=record.quality[idx] if record.quality is not None else None,
        )

    def query_interframe_matrices(self, meta, k_max: int) -> np.ndarray | None:
        """Return inter-frame pose matrices ``[T_vid, K_max, D]`` aligned to video frames.

        For frame ``i`` in ``0..T-2``, matrix ``i`` contains all SLAM samples in
        ``[t(frame_i), t(frame_{i+1}))`` with interval-relative ``pose_*`` features.
        Frame ``T-1`` is zero-padded.
        """
        record = self._load_clip_record(meta)
        frame_ts = self.frame_timestamps_us(meta)
        if record is None or frame_ts is None:
            return None
        t_vid = int(frame_ts.shape[0])
        if t_vid < 2:
            return None
        k_max = int(k_max)
        if k_max <= 0:
            raise ValueError(f"interframe k_max must be positive, got {k_max}")
        d = self.input_dim
        out = np.zeros((t_vid, k_max, d), dtype=np.float32)
        for i in range(t_vid - 1):
            seg_record = self._slice_record_interval(record, float(frame_ts[i]), float(frame_ts[i + 1]))
            if seg_record is None or seg_record.timestamps_us.size < 1:
                continue
            feats = build_pose_features(
                seg_record.translation,
                seg_record.quaternion,
                seg_record.angular_vel,
                seg_record.linear_vel,
                self.feature_set,
            )
            if feats.size == 0:
                continue
            out[i] = window_smooth_pose_matrix(feats, k_max)
        return out

    def query_clip_features(self, meta) -> np.ndarray | None:
        """Return pose feature trajectory ``[N, D]`` for one clip metadata dict."""
        record = self._load_clip_record(meta)
        if record is None:
            return None
        feats = build_pose_features(
            record.translation,
            record.quaternion,
            record.angular_vel,
            record.linear_vel,
            self.feature_set,
        )
        if feats.shape[0] < 2:
            return None
        return feats


def _vjepa_video_id(video_id: str) -> str:
    text = str(video_id)
    if "-" in text and "_" not in text:
        participant, rest = text.split("-", 1)
        return f"{participant}_{rest}"
    return text
