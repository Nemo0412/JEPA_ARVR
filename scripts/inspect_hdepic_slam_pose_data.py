#!/usr/bin/env python3
"""Inspect HD-EPIC SLAM pose/IMU-derived trajectory data and clip alignment."""

from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from collections import Counter
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import pandas as pd

from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate, _clean_video_id, _vjepa_video_id
from app.hdepic_lora_action_anticipation.pose_slam import SlamPoseLoader


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/HD-EPIC"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/hdepic_vjepa_annotations"))
    parser.add_argument("--slam-root", type=Path, default=None)
    parser.add_argument("--mapping-json", type=Path, default=None)
    parser.add_argument("--sync-root", type=Path, default=None)
    parser.add_argument("--gaze-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/hdepic_slam_inspection"))
    parser.add_argument("--sample-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visualize",
        nargs="*",
        default=None,
        help="Optional video_ids for sanity plots (default: 3 reference videos if matplotlib available)",
    )
    parser.add_argument("--interframe-k-max", type=int, default=128)
    parser.add_argument("--skip-clip-audit", action="store_true")
    return parser.parse_args()


def _default_slam_root(raw_root: Path) -> Path:
    return raw_root / "SLAM-and-Gaze" / "P01" / "SLAM" / "multi"


def _read_trajectory_header_stats(zip_path: Path, session_id: str) -> dict:
    inner = f"{session_id}/slam/closed_loop_trajectory.csv"
    with zipfile.ZipFile(zip_path) as zf:
        name = inner if inner in zf.namelist() else next(
            (n for n in zf.namelist() if n.endswith("closed_loop_trajectory.csv")), None
        )
        if name is None:
            return {"error": "missing_trajectory_csv"}
        with zf.open(name) as fh:
            df_head = pd.read_csv(TextIOWrapper(fh, encoding="utf-8"), nrows=1000)
        row_count = 0
        with zf.open(name) as fh:
            wrapper = io.TextIOWrapper(fh, encoding="utf-8")
            for row_count, _ in enumerate(wrapper, start=0):
                pass
        row_count = max(0, row_count - 1)

    ts_col = next((c for c in df_head.columns if "timestamp" in c.lower()), None)
    dt_us = None
    hz = None
    if ts_col and len(df_head) > 2:
        ts = pd.to_numeric(df_head[ts_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if ts.size > 2:
            dts = np.diff(np.sort(ts))
            dts = dts[dts > 0]
            if dts.size:
                dt_us = float(np.median(dts))
                hz = 1e6 / dt_us if dt_us > 0 else None

    q_col = next((c for c in df_head.columns if c.lower() == "quality_score"), None)
    quality = None
    if q_col:
        q = pd.to_numeric(df_head[q_col], errors="coerce").dropna()
        if len(q):
            quality = {
                "min": float(q.min()),
                "median": float(q.median()),
                "max": float(q.max()),
            }

    return {
        "row_count": int(row_count),
        "columns": list(df_head.columns),
        "median_dt_us": dt_us,
        "approx_hz": hz,
        "quality_head_stats": quality,
        "zip_size_bytes": zip_path.stat().st_size,
    }


def _load_csv_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _simulate_clip_window(row: pd.Series, frames_per_clip: int = 32, fps: float = 8.0) -> dict:
    """Build metadata similar to ClipBalancedDecodeVideosToClips for audit."""
    start = float(row["start_frame"])
    stop = float(row["stop_frame"])
    anchor = random.uniform(start, stop)
    anticipation = random.uniform(0.25, 1.75)
    anchor_frame = max(start, anchor - anticipation * fps)
    indices = np.linspace(
        max(0, anchor_frame - (frames_per_clip - 1)),
        anchor_frame,
        frames_per_clip,
    )
    return {
        "video_id": str(row["video_id"]),
        "start_frame": int(row["start_frame"]),
        "frame_indices": indices.tolist(),
        "vfps": fps,
    }


def audit_sessions(slam_root: Path, mapping: dict) -> dict:
    sessions = sorted(set(str(v) for v in mapping.values()))
    per_session = {}
    for sid in sessions:
        zpath = slam_root / f"{sid}.zip"
        if not zpath.exists():
            per_session[sid] = {"error": "zip_missing", "path": str(zpath)}
            continue
        per_session[sid] = _read_trajectory_header_stats(zpath, sid)
    return per_session


def audit_interframe_interval_counts(
    loader: SlamPoseLoader,
    rows: pd.DataFrame,
    sample_n: int,
    seed: int,
    k_max: int = 128,
) -> dict:
    """Audit per inter-frame interval SLAM sample counts for matrix packing."""
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    indices = indices[: min(sample_n, len(indices))]

    interval_counts: list[int] = []
    clips_ok = 0
    clips_with_pose = 0
    truncated_intervals = 0
    empty_intervals = 0

    for idx in indices:
        row = rows.iloc[idx]
        meta = _simulate_clip_window(row)
        frame_ts = loader.frame_timestamps_us(meta)
        record = loader._load_clip_record(meta)  # noqa: SLF001
        if frame_ts is None or record is None:
            continue
        clips_with_pose += 1
        ts = record.timestamps_us
        valid_clip = True
        for i in range(len(frame_ts) - 1):
            t0, t1 = float(frame_ts[i]), float(frame_ts[i + 1])
            count = int(np.sum((ts >= t0) & (ts < t1)))
            interval_counts.append(count)
            if count == 0:
                empty_intervals += 1
            if count > k_max:
                truncated_intervals += 1
        mats = loader.query_interframe_matrices(meta, k_max)
        if mats is not None:
            clips_ok += 1

    counts_arr = np.asarray(interval_counts, dtype=np.int64)
    n_intervals = int(counts_arr.size)
    return {
        "sample_count": len(indices),
        "clips_with_pose_window": clips_with_pose,
        "clips_query_ok": clips_ok,
        "k_max": int(k_max),
        "interval_count_total": n_intervals,
        "interval_count_median": float(np.median(counts_arr)) if n_intervals else None,
        "interval_count_p95": float(np.percentile(counts_arr, 95)) if n_intervals else None,
        "interval_count_p99": float(np.percentile(counts_arr, 99)) if n_intervals else None,
        "interval_count_min": int(counts_arr.min()) if n_intervals else None,
        "interval_count_max": int(counts_arr.max()) if n_intervals else None,
        "empty_interval_frac": empty_intervals / max(1, n_intervals),
        "truncated_interval_frac": truncated_intervals / max(1, n_intervals),
    }


def audit_clip_alignment(
    loader: SlamPoseLoader,
    gate: GazeTokenGate,
    rows: pd.DataFrame,
    sample_n: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    indices = indices[: min(sample_n, len(indices))]

    pose_found = 0
    pose_query_ok = 0
    gaze_found = 0
    gaze_query_ok = 0
    pose_points: list[int] = []
    max_gap_us: list[float] = []
    per_video: Counter = Counter()

    for idx in indices:
        row = rows.iloc[idx]
        meta = _simulate_clip_window(row)
        vid = meta["video_id"]
        per_video[vid] += 1

        record = loader.has_pose_source(vid)
        if record:
            pose_found += 1
        feats = loader.query_clip_features(meta)
        if feats is not None:
            pose_query_ok += 1
            pose_points.append(int(feats.shape[0]))
            window = loader._clip_time_us(meta)  # noqa: SLF001
            if window is not None and feats.shape[0] > 1:
                q0, q1 = window
                # Approximate max gap from uniform spacing inside returned window.
                span = max(q1 - q0, 1.0)
                max_gap_us.append(float(span / max(1, feats.shape[0] - 1)))

        gaze_record = gate._load_record(vid)  # noqa: SLF001
        if gaze_record is not None:
            gaze_found += 1
        if gaze_record is not None and gaze_record.timestamps_us.size:
            pick = gate._query_indices(gaze_record, meta["frame_indices"], meta["vfps"])  # noqa: SLF001
            if pick is not None:
                gaze_query_ok += 1

    n = len(indices)
    return {
        "sample_count": n,
        "pose_record_found": pose_found,
        "pose_query_ok": pose_query_ok,
        "gaze_record_found": gaze_found,
        "gaze_query_ok": gaze_query_ok,
        "pose_record_found_frac": pose_found / max(1, n),
        "pose_query_ok_frac": pose_query_ok / max(1, n),
        "gaze_record_found_frac": gaze_found / max(1, n),
        "gaze_query_ok_frac": gaze_query_ok / max(1, n),
        "pose_points_median": float(np.median(pose_points)) if pose_points else None,
        "pose_points_min": int(min(pose_points)) if pose_points else None,
        "pose_points_max": int(max(pose_points)) if pose_points else None,
        "max_in_window_gap_us_median": float(np.median(max_gap_us)) if max_gap_us else None,
        "unique_videos_in_sample": len(per_video),
    }


def write_markdown(path: Path, payload: dict):
    m = payload["mapping"]
    sessions = payload["session_stats"]
    clip = payload.get("clip_audit", {})
    interframe = payload.get("interframe_audit", {})
    train = payload.get("train_videos", {})
    val = payload.get("val_videos", {})

    lines = [
        "# HD-EPIC SLAM Pose Inspection",
        "",
        "## Inputs",
        "",
        f"- slam_root: `{payload['slam_root']}`",
        f"- mapping_json: `{payload['mapping_json']}`",
        f"- annotations_dir: `{payload['annotations_dir']}`",
        "",
        "## Video to SLAM Session Mapping",
        "",
        f"- mapped videos: {len(m)}",
        f"- unique sessions: {len(set(m.values()))}",
        "",
        "## Session Trajectory Stats",
        "",
    ]
    for sid, stats in sorted(sessions.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        if "error" in stats:
            lines.append(f"- session `{sid}`: **{stats['error']}**")
            continue
        hz = stats.get("approx_hz")
        hz_text = f"{hz:.1f} Hz" if hz else "n/a"
        lines.append(
            f"- session `{sid}`: rows={stats['row_count']}, median_dt_us={stats.get('median_dt_us')}, "
            f"rate≈{hz_text}, zip={stats['zip_size_bytes'] / (1024**2):.1f} MiB"
        )

    if clip:
        for split_name, split_clip in clip.items():
            if not isinstance(split_clip, dict):
                continue
            lines += [
                "",
                f"## Clip Alignment Audit ({split_name}, simulated sampling)",
                "",
                f"- sample_count: {split_clip.get('sample_count')}",
                f"- pose record_found: {split_clip.get('pose_record_found')} ({split_clip.get('pose_record_found_frac', 0):.3f})",
                f"- pose query_ok: {split_clip.get('pose_query_ok')} ({split_clip.get('pose_query_ok_frac', 0):.3f})",
                f"- gaze record_found: {split_clip.get('gaze_record_found')} ({split_clip.get('gaze_record_found_frac', 0):.3f})",
                f"- gaze query_ok: {split_clip.get('gaze_query_ok')} ({split_clip.get('gaze_query_ok_frac', 0):.3f})",
                f"- pose points median/min/max: {split_clip.get('pose_points_median')} / {split_clip.get('pose_points_min')} / {split_clip.get('pose_points_max')}",
                f"- max in-window timestamp gap (median us): {split_clip.get('max_in_window_gap_us_median')}",
            ]

    if interframe:
        for split_name, split_if in interframe.items():
            if not isinstance(split_if, dict):
                continue
            lines += [
                "",
                f"## Inter-frame Pose Interval Audit ({split_name}, k_max={split_if.get('k_max')})",
                "",
                f"- sample_count: {split_if.get('sample_count')}",
                f"- clips_with_pose_window: {split_if.get('clips_with_pose_window')}",
                f"- clips_query_ok: {split_if.get('clips_query_ok')}",
                f"- interval_count median/p95/p99/min/max: "
                f"{split_if.get('interval_count_median')} / {split_if.get('interval_count_p95')} / "
                f"{split_if.get('interval_count_p99')} / {split_if.get('interval_count_min')} / "
                f"{split_if.get('interval_count_max')}",
                f"- empty_interval_frac: {split_if.get('empty_interval_frac')}",
                f"- truncated_interval_frac (count > k_max): {split_if.get('truncated_interval_frac')}",
            ]

    lines += [
        "",
        "## Split Video Coverage",
        "",
        f"- train CSV videos: {train.get('total_videos')} mapped={train.get('mapped')} missing={train.get('missing')}",
        f"- val CSV videos: {val.get('total_videos')} mapped={val.get('mapped')} missing={val.get('missing')}",
        "",
        "## Feature Sets (for training)",
        "",
        "- `pose_6d`: delta translation (3) + relative rot6d (6) → D=9",
        "- `pose_vel`: pose_6d + angular velocity (3) → D=12",
        "- `pose_full`: pose_vel + linear velocity (3) → D=15",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _split_video_coverage(mapping: dict, csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"error": "missing_csv", "path": str(csv_path)}
    df = _load_csv_rows(csv_path)
    vids = sorted(set(str(v) for v in df["video_id"].unique()))
    mapped = [v for v in vids if v in mapping]
    missing = [v for v in vids if v not in mapping]
    return {
        "total_videos": len(vids),
        "mapped": len(mapped),
        "missing": len(missing),
        "missing_ids": missing[:20],
    }


def visualize_reference_videos(
    loader: SlamPoseLoader,
    gate: GazeTokenGate,
    video_ids: list[str],
    output_dir: Path,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[slam-inspect] matplotlib not available; skipping visualization")
        return

    out = output_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    for video_id in video_ids:
        clean = _clean_video_id(video_id)
        meta = {
            "video_id": _vjepa_video_id(video_id) if "-" in video_id else video_id,
            "frame_indices": list(np.linspace(0, 32 * 30, 32)),
            "vfps": 30.0,
        }
        window = loader._clip_time_us(meta)  # noqa: SLF001
        if window is None:
            continue
        q0, q1 = window
        feats = loader.query_clip_features(meta)
        if feats is None or feats.shape[0] < 2:
            continue
        t_rel = np.linspace(0.0, (q1 - q0) / 1e6, feats.shape[0])

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t_rel, feats[:, :3])
        axes[0].set_ylabel("delta translation")
        axes[0].legend(["tx", "ty", "tz"], loc="upper right")
        if feats.shape[1] >= 9:
            axes[1].plot(t_rel, feats[:, 3:6])
            axes[1].set_ylabel("rot6d (first 3)")
        axes[1].set_xlabel("time (s) within clip window")
        fig.suptitle(f"SLAM pose features: {meta['video_id']}")
        fig.tight_layout()
        fig.savefig(out / f"{clean}_pose_features.png", dpi=120)
        plt.close(fig)

        gaze_record = gate._load_record(meta["video_id"])  # noqa: SLF001
        if gaze_record is not None and gaze_record.yaw is not None:
            gmask = (gaze_record.timestamps_us >= q0) & (gaze_record.timestamps_us <= q1)
            if gmask.any():
                gt = (gaze_record.timestamps_us[gmask] - gaze_record.timestamps_us[gmask][0]) / 1e6
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(gt, np.degrees(gaze_record.yaw[gmask]), label="gaze yaw (deg)")
                ax2.plot(gt, np.degrees(gaze_record.pitch[gmask]), label="gaze pitch (deg)")
                ax2.set_xlabel("time (s)")
                ax2.set_ylabel("angle (deg)")
                ax2.legend()
                ax2.set_title(f"Gaze vs time (same window): {meta['video_id']}")
                fig2.tight_layout()
                fig2.savefig(out / f"{clean}_gaze_overlay.png", dpi=120)
                plt.close(fig2)

    print(f"[slam-inspect] Wrote figures under {out}")


def main():
    args = parse_args()
    slam_root = args.slam_root or _default_slam_root(args.raw_root)
    mapping_json = args.mapping_json or (slam_root / "vrs_to_multi_slam.json")
    sync_root = args.sync_root or (args.raw_root / "HD-EPIC" / "Videos")
    gaze_root = args.gaze_root or (args.raw_root / "SLAM-and-Gaze")

    mapping_raw = json.loads(mapping_json.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for vrs_path, session_id in mapping_raw.items():
        name = Path(vrs_path).stem
        if "-" in name:
            participant, rest = name.split("-", 1)
            mapping[f"{participant}_{rest}"] = str(session_id)

    session_stats = audit_sessions(slam_root, mapping_raw)

    pose_cfg = {
        "slam_root": str(slam_root),
        "mapping_json": str(mapping_json),
        "feature_set": "pose_6d",
        "gaze_root": str(gaze_root),
        "sync_root": str(sync_root),
    }
    gate = GazeTokenGate({"mode": "none", "gaze_root": str(gaze_root), "sync_root": str(sync_root)})
    loader = SlamPoseLoader(pose_cfg, gate=gate)

    train_csv = args.annotations_dir / "HD_EPIC_train_vjepa.csv"
    val_csv = args.annotations_dir / "HD_EPIC_val_vjepa.csv"
    train_cov = _split_video_coverage(mapping, train_csv)
    val_cov = _split_video_coverage(mapping, val_csv)

    clip_audit = {}
    interframe_audit = {}
    if not args.skip_clip_audit and train_csv.exists():
        train_rows = _load_csv_rows(train_csv)
        clip_audit["train"] = audit_clip_alignment(loader, gate, train_rows, args.sample_rows, args.seed)
        interframe_audit["train"] = audit_interframe_interval_counts(
            loader, train_rows, args.sample_rows, args.seed, k_max=args.interframe_k_max
        )
        if val_csv.exists():
            val_rows = _load_csv_rows(val_csv)
            clip_audit["val"] = audit_clip_alignment(
                loader, gate, val_rows, min(args.sample_rows, len(val_rows)), args.seed + 1
            )
            interframe_audit["val"] = audit_interframe_interval_counts(
                loader, val_rows, min(args.sample_rows, len(val_rows)), args.seed + 1, k_max=args.interframe_k_max
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "slam_root": str(slam_root),
        "mapping_json": str(mapping_json),
        "annotations_dir": str(args.annotations_dir),
        "mapping": mapping,
        "session_stats": session_stats,
        "train_videos": train_cov,
        "val_videos": val_cov,
        "clip_audit": clip_audit,
        "interframe_audit": interframe_audit,
    }
    (args.output_dir / "slam_pose_inspection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "slam_pose_inspection.md", payload)

    ref_videos = args.visualize
    if ref_videos is None:
        ref_videos = [
            "P01_20240202-110250",
            "P01_20240203-123350",
            "P01_20240204-135502",
        ]
    if ref_videos:
        visualize_reference_videos(loader, gate, ref_videos, args.output_dir)

    print(f"mapped_videos: {len(mapping)}")
    print(f"sessions: {len(session_stats)}")
    if clip_audit.get("train"):
        t = clip_audit["train"]
        print(f"train_pose_query_ok_frac: {t.get('pose_query_ok_frac')}")
        print(f"train_gaze_query_ok_frac: {t.get('gaze_query_ok_frac')}")
    if interframe_audit.get("train"):
        t = interframe_audit["train"]
        print(
            "train_interframe_interval_median/p99/trunc_frac: "
            f"{t.get('interval_count_median')}/{t.get('interval_count_p99')}/{t.get('truncated_interval_frac')}"
        )
    print(f"wrote: {args.output_dir / 'slam_pose_inspection.md'}")


if __name__ == "__main__":
    main()
