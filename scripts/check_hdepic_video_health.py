#!/usr/bin/env python
"""Check HD-EPIC videos referenced by V-JEPA/EK100-style CSVs.

This is intentionally project-local and does not modify the upstream vjepa2/
tree. It mimics the action_anticipation_frozen EK100 decoder path resolution
and decord frame access closely enough to catch common partial/corrupt mp4
failures before spending GPU time.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("check_hdepic_video_health")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        type=Path,
        help="One or more EK100-style CSV files to check.",
    )
    parser.add_argument(
        "--base-path",
        required=True,
        type=Path,
        help="V-JEPA data base_path, e.g. data/hdepic_vjepa_videos.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for health summary and bad-row reports.",
    )
    parser.add_argument(
        "--file-format",
        type=int,
        default=1,
        choices=[0, 1],
        help="EK100 path format. 1 means base_path/PID/video_id.MP4.",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=16,
        help="Decoder frames_per_clip.",
    )
    parser.add_argument(
        "--decoder-fps",
        type=float,
        default=5.0,
        help="Decoder sampling FPS used by action_anticipation_frozen.",
    )
    parser.add_argument(
        "--anticipation-times",
        nargs="+",
        type=float,
        default=[1.0],
        help="Anticipation horizons in seconds to validate, e.g. 1 10 60.",
    )
    parser.add_argument(
        "--anticipation-points",
        nargs="+",
        type=float,
        default=[0.0],
        help=(
            "Action-relative points to validate. 0.0 matches current validation "
            "config; use 0.25 0.75 to approximate training range."
        ),
    )
    parser.add_argument(
        "--max-rows-per-video",
        type=int,
        default=0,
        help="Optional cap for faster smoke checks. 0 checks every row.",
    )
    parser.add_argument(
        "--skip-frame-check",
        action="store_true",
        help="Only test VideoReader open/fps/length, not get_batch(indices).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def import_decord():
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise RuntimeError("decord is required inside the V-JEPA/EgoLifeExp environment") from exc
    return VideoReader, cpu


def load_annotations(csv_paths):
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df["source_csv"] = str(csv_path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    required = {"participant_id", "video_id", "start_frame", "stop_frame"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    return df


def resolve_path(base_path, video_id, participant_id, file_format):
    if file_format == 0:
        return base_path / participant_id / "videos" / f"{video_id}.MP4"
    return base_path / participant_id / f"{video_id}.MP4"


def sample_rows(df, max_rows):
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sort_values(["start_frame", "stop_frame"]).iloc[:max_rows]


def build_indices(row, video_fps, decoder_fps, frames_per_clip, anticipation_time, anticipation_point):
    frame_step = max(1, int(video_fps / decoder_fps))
    needed_frames = int(frames_per_clip * frame_step)
    start_frame = int(row["start_frame"])
    stop_frame = int(row["stop_frame"])
    anticipation_frames = int(anticipation_time * video_fps)
    anchor = int(start_frame * anticipation_point + (1 - anticipation_point) * stop_frame - anticipation_frames)
    indices = np.arange(anchor - needed_frames, anchor, frame_step).astype(np.int64)
    indices[indices < 0] = 0
    return indices


def check_video(video_id, video_df, base_path, file_format, args, VideoReader, cpu):
    participant_id = str(video_df["participant_id"].iloc[0])
    path = resolve_path(base_path, video_id, participant_id, file_format)
    result = {
        "video_id": video_id,
        "participant_id": participant_id,
        "path": str(path),
        "rows": int(len(video_df)),
        "checked_rows": 0,
        "ok": False,
        "open_ok": False,
        "frame_ok": False,
        "fps": None,
        "num_frames": None,
        "errors": [],
    }
    bad_rows = []

    if not path.exists():
        result["errors"].append({"stage": "exists", "error": "missing file"})
        return result, bad_rows

    try:
        vr = VideoReader(str(path), num_threads=-1, ctx=cpu(0))
        vr.seek(0)
        result["open_ok"] = True
        result["fps"] = float(vr.get_avg_fps())
        result["num_frames"] = int(len(vr))
    except Exception as exc:
        result["errors"].append({"stage": "open", "error": repr(exc)})
        return result, bad_rows

    if args.skip_frame_check:
        result["frame_ok"] = True
        result["ok"] = True
        result["checked_rows"] = int(len(sample_rows(video_df, args.max_rows_per_video)))
        return result, bad_rows

    rows_to_check = sample_rows(video_df, args.max_rows_per_video)
    result["checked_rows"] = int(len(rows_to_check))
    frame_errors = 0

    for row_index, row in rows_to_check.iterrows():
        for anticipation_time in args.anticipation_times:
            for anticipation_point in args.anticipation_points:
                indices = build_indices(
                    row=row,
                    video_fps=result["fps"],
                    decoder_fps=args.decoder_fps,
                    frames_per_clip=args.frames_per_clip,
                    anticipation_time=anticipation_time,
                    anticipation_point=anticipation_point,
                )
                max_index = int(indices.max()) if len(indices) else -1
                if max_index >= result["num_frames"]:
                    error = f"indices exceed video length: max_index={max_index}, num_frames={result['num_frames']}"
                    frame_errors += 1
                    bad_rows.append(
                        {
                            "video_id": video_id,
                            "participant_id": participant_id,
                            "path": str(path),
                            "row_index": int(row_index),
                            "source_csv": row.get("source_csv", ""),
                            "start_frame": int(row["start_frame"]),
                            "stop_frame": int(row["stop_frame"]),
                            "anticipation_time": anticipation_time,
                            "anticipation_point": anticipation_point,
                            "error": error,
                        }
                    )
                    continue
                try:
                    vr.get_batch(indices).asnumpy()
                except Exception as exc:
                    frame_errors += 1
                    bad_rows.append(
                        {
                            "video_id": video_id,
                            "participant_id": participant_id,
                            "path": str(path),
                            "row_index": int(row_index),
                            "source_csv": row.get("source_csv", ""),
                            "start_frame": int(row["start_frame"]),
                            "stop_frame": int(row["stop_frame"]),
                            "anticipation_time": anticipation_time,
                            "anticipation_point": anticipation_point,
                            "error": repr(exc),
                        }
                    )

    result["frame_ok"] = frame_errors == 0
    result["ok"] = result["open_ok"] and result["frame_ok"]
    if frame_errors:
        result["errors"].append({"stage": "get_batch", "error": f"{frame_errors} failed row/horizon checks"})
    return result, bad_rows


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    VideoReader, cpu = import_decord()
    df = load_annotations(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.base_path.expanduser().resolve()

    results = []
    bad_rows = []
    for video_id, video_df in df.groupby("video_id", sort=True):
        result, video_bad_rows = check_video(
            video_id=str(video_id),
            video_df=video_df,
            base_path=base_path,
            file_format=args.file_format,
            args=args,
            VideoReader=VideoReader,
            cpu=cpu,
        )
        results.append(result)
        bad_rows.extend(video_bad_rows)
        status = "OK" if result["ok"] else "BAD"
        LOGGER.info(
            "%s %s rows=%s checked=%s fps=%s frames=%s",
            status,
            result["path"],
            result["rows"],
            result["checked_rows"],
            result["fps"],
            result["num_frames"],
        )

    summary = {
        "csv": [str(p) for p in args.csv],
        "base_path": str(base_path),
        "file_format": args.file_format,
        "frames_per_clip": args.frames_per_clip,
        "decoder_fps": args.decoder_fps,
        "anticipation_times": args.anticipation_times,
        "anticipation_points": args.anticipation_points,
        "num_rows": int(len(df)),
        "num_videos": int(len(results)),
        "ok_videos": int(sum(1 for r in results if r["ok"])),
        "bad_videos": int(sum(1 for r in results if not r["ok"])),
        "missing_videos": int(sum(1 for r in results if any(e["stage"] == "exists" for e in r["errors"]))),
        "open_failed_videos": int(sum(1 for r in results if any(e["stage"] == "open" for e in r["errors"]))),
        "frame_failed_videos": int(sum(1 for r in results if any(e["stage"] == "get_batch" for e in r["errors"]))),
        "bad_row_checks": int(len(bad_rows)),
        "videos": results,
    }

    summary_path = args.output_dir / "video_health_summary.json"
    bad_video_path = args.output_dir / "video_health_bad_videos.txt"
    bad_row_path = args.output_dir / "video_health_bad_rows.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bad_video_path.write_text(
        "\n".join(r["path"] for r in results if not r["ok"]) + ("\n" if summary["bad_videos"] else ""),
        encoding="utf-8",
    )
    pd.DataFrame(bad_rows).to_csv(bad_row_path, index=False)

    LOGGER.info("Wrote %s", summary_path)
    LOGGER.info("Wrote %s", bad_video_path)
    LOGGER.info("Wrote %s", bad_row_path)
    LOGGER.info(
        "Summary: %s/%s videos OK, bad_videos=%s, bad_row_checks=%s",
        summary["ok_videos"],
        summary["num_videos"],
        summary["bad_videos"],
        summary["bad_row_checks"],
    )

    if summary["bad_videos"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
