#!/usr/bin/env python3
"""Build a first-pass gaze heatmap or gaze-overlay video from HD-EPIC MPS data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaze-zip", type=Path, required=True, help="Downloaded mps_*_vrs.zip file")
    parser.add_argument("--video", type=Path, default=None, help="Optional source video for overlay output")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--map-type", choices=["gaussian", "binary"], default="gaussian")
    parser.add_argument("--sigma", type=float, default=18.0)
    parser.add_argument("--binary-radius", type=float, default=None, help="Pixel radius for binary gaze disks; defaults to --sigma")
    parser.add_argument("--overlay-alpha", type=float, default=0.28)
    parser.add_argument("--fallback-fov-deg", type=float, default=90.0)
    parser.add_argument("--max-overlay-frames", type=int, default=300, help="0 means write the full video")
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def extract_zip(zip_path: Path, extract_dir: Path):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
        return zf.namelist()


def numeric(value):
    try:
        if value is None or value == "":
            return None
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def score_xy_columns(fieldnames):
    names = list(fieldnames or [])
    lower = {name: name.lower() for name in names}
    candidates = [
        ("gaze_center_x", "gaze_center_y"),
        ("gaze x [px]", "gaze y [px]"),
        ("gaze_x", "gaze_y"),
        ("x", "y"),
        ("u", "v"),
    ]
    for x_name, y_name in candidates:
        for actual_x in names:
            if lower[actual_x] == x_name:
                for actual_y in names:
                    if lower[actual_y] == y_name:
                        return actual_x, actual_y

    x_ranked = []
    y_ranked = []
    for name in names:
        text = lower[name]
        if "gaze" not in text and "projection" not in text and "projected" not in text and text not in {"x", "y", "u", "v"}:
            continue
        if text.endswith("_x") or text.endswith(".x") or " x " in text or text in {"x", "u"}:
            x_ranked.append(name)
        if text.endswith("_y") or text.endswith(".y") or " y " in text or text in {"y", "v"}:
            y_ranked.append(name)
    if x_ranked and y_ranked:
        return x_ranked[0], y_ranked[0]
    return None, None


def score_yaw_pitch_columns(fieldnames):
    names = list(fieldnames or [])
    lower = {name: name.lower() for name in names}
    yaw_candidates = []
    pitch_candidates = []
    for name in names:
        text = lower[name]
        if "yaw" in text:
            yaw_candidates.append(name)
        if "pitch" in text:
            pitch_candidates.append(name)

    def rank(name):
        text = lower[name]
        score = 0
        if "rads" in text or "rad" in text:
            score -= 4
        if "cpf" in text:
            score -= 2
        if "low" in text or "high" in text:
            score += 5
        return score

    if yaw_candidates and pitch_candidates:
        return sorted(yaw_candidates, key=rank)[0], sorted(pitch_candidates, key=rank)[0]
    return None, None


def find_timestamp_column(fieldnames):
    for name in fieldnames or []:
        text = name.lower()
        if "timestamp" in text or text in {"time", "t", "tracking_timestamp_us"}:
            return name
    return None


def clean_fieldname(name):
    return str(name).strip().lstrip("#").strip()


def normalized_gaze_from_yaw_pitch(yaw, pitch, fov_deg):
    fov = math.radians(max(1e-3, min(179.0, fov_deg)))
    scale = 2.0 * math.tan(fov / 2.0)
    x = 0.5 + math.tan(yaw) / scale
    y = 0.5 - math.tan(pitch) / scale
    return x, y


def load_gaze_points(root: Path, fallback_fov_deg: float):
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".txt", ".json", ".jsonl"}:
            continue
        if "gaze" not in path.name.lower() and "eye" not in path.name.lower():
            continue
        candidates.append(path)

    tried = []
    for path in candidates:
        if path.suffix.lower() in {".csv", ".txt"}:
            points = load_csv_points(path, fallback_fov_deg=fallback_fov_deg)
        else:
            points = load_json_points(path)
        tried.append({"path": str(path), "points": len(points)})
        if points:
            return path, points, tried
    return None, [], tried


def load_csv_points(path: Path, fallback_fov_deg: float):
    for delimiter in [",", "\t", " "]:
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                content = [line for line in f if line.strip()]
                if not content:
                    continue
                header_idx = 0
                for idx, line in enumerate(content):
                    lower = line.lower()
                    if delimiter in line and any(token in lower for token in ("gaze", "yaw", "pitch", "timestamp")):
                        header_idx = idx
                        break
                content = content[header_idx:]
                reader = csv.DictReader(content, delimiter=delimiter, skipinitialspace=True)
                if not reader.fieldnames or len(reader.fieldnames) < 2:
                    continue
                raw_fieldnames = reader.fieldnames
                clean_names = [clean_fieldname(name) for name in raw_fieldnames]
                reader.fieldnames = clean_names

                x_col, y_col = score_xy_columns(reader.fieldnames)
                yaw_col, pitch_col = score_yaw_pitch_columns(reader.fieldnames)
                if (not x_col or not y_col) and (not yaw_col or not pitch_col):
                    continue
                t_col = find_timestamp_column(reader.fieldnames)
                rows = []
                for row in reader:
                    row = {clean_fieldname(k): v for k, v in row.items()}
                    if x_col and y_col:
                        x = numeric(row.get(x_col))
                        y = numeric(row.get(y_col))
                    else:
                        yaw = numeric(row.get(yaw_col))
                        pitch = numeric(row.get(pitch_col))
                        if yaw is None or pitch is None:
                            continue
                        x, y = normalized_gaze_from_yaw_pitch(yaw, pitch, fallback_fov_deg)
                    if x is None or y is None:
                        continue
                    t = numeric(row.get(t_col)) if t_col else None
                    rows.append((t, x, y))
                if rows:
                    return rows
        except Exception:
            continue
    return []


def walk_json_records(obj):
    if isinstance(obj, list):
        for item in obj:
            yield from walk_json_records(item)
    elif isinstance(obj, dict):
        yield obj
        for value in obj.values():
            if isinstance(value, (list, dict)):
                yield from walk_json_records(value)


def load_json_points(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records = list(walk_json_records(json.loads(text)))
    except Exception:
        return []

    if not records:
        return []
    keys = set().union(*(r.keys() for r in records if isinstance(r, dict)))
    x_col, y_col = score_xy_columns(keys)
    t_col = find_timestamp_column(keys)
    points = []
    for row in records:
        if not isinstance(row, dict):
            continue
        x = numeric(row.get(x_col)) if x_col else None
        y = numeric(row.get(y_col)) if y_col else None
        if x is None or y is None:
            continue
        t = numeric(row.get(t_col)) if t_col else None
        points.append((t, x, y))
    return points


def normalize_points(points, width, height):
    arr = np.array([[p[1], p[2]] for p in points], dtype=np.float32)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    if arr.size == 0:
        return arr
    max_abs = np.nanmax(np.abs(arr), axis=0)
    if max_abs[0] <= 2.0 and max_abs[1] <= 2.0:
        arr[:, 0] *= width
        arr[:, 1] *= height
    return arr[(arr[:, 0] >= 0) & (arr[:, 0] < width) & (arr[:, 1] >= 0) & (arr[:, 1] < height)]


def build_heatmap(points_xy, width, height, sigma, map_type="gaussian", binary_radius=None):
    heat = np.zeros((height, width), dtype=np.float32)
    disk_radius = float(binary_radius if binary_radius is not None else sigma)
    radius = max(2, int(math.ceil(disk_radius if map_type == "binary" else 3 * sigma)))
    for x, y in points_xy:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        if map_type == "binary":
            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], (dist2 <= disk_radius**2).astype(np.float32))
        else:
            heat[y0:y1, x0:x1] += np.exp(-dist2 / (2 * sigma**2))
    if map_type != "binary" and heat.max() > 0:
        heat /= heat.max()
    return heat


def write_heatmap_png(heat, path: Path, map_type="gaussian"):
    import cv2

    img = np.uint8(np.clip(heat * 255, 0, 255))
    if map_type == "binary":
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), color)


def write_overlay(
    video_path: Path,
    points_xy,
    output_path: Path,
    max_frames: int,
    sigma: float,
    map_type="gaussian",
    binary_radius=None,
    overlay_alpha=0.28,
):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    if max_frames and max_frames > 0:
        frame_count = min(frame_count, max_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    if len(points_xy) == 0:
        point_indices = [[] for _ in range(frame_count)]
    else:
        bins = np.linspace(0, len(points_xy), frame_count + 1, dtype=int)
        point_indices = [points_xy[bins[i] : bins[i + 1]] for i in range(frame_count)]

    for idx in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        local = point_indices[idx]
        if len(local):
            heat = build_heatmap(local, width, height, sigma, map_type=map_type, binary_radius=binary_radius)
            alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
            if map_type == "binary":
                mask = heat > 0
                color = np.zeros_like(frame)
                color[:, :, 2] = 255
                blended = cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0)
                frame[mask] = blended[mask]
            else:
                color = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)
                frame = cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0)
        writer.write(frame)

    writer.release()
    cap.release()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = args.extract_dir or args.output_dir / "extracted" / args.gaze_zip.stem
    members = extract_zip(args.gaze_zip, extract_dir)
    manifest_path = args.output_dir / f"{args.gaze_zip.stem}_manifest.json"
    manifest_path.write_text(json.dumps(members, indent=2), encoding="utf-8")
    if args.list_only:
        print(f"extracted: {extract_dir}")
        print(f"members: {len(members)}")
        print(f"manifest: {manifest_path}")
        return

    gaze_file, points, tried = load_gaze_points(extract_dir, fallback_fov_deg=args.fallback_fov_deg)
    width = args.width
    height = args.height
    if args.video:
        try:
            import cv2

            cap = cv2.VideoCapture(str(args.video))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            cap.release()
        except Exception:
            pass

    points_xy = normalize_points(points, width, height)
    binary_radius = args.binary_radius if args.binary_radius is not None else args.sigma
    heat = build_heatmap(
        points_xy,
        width,
        height,
        args.sigma,
        map_type=args.map_type,
        binary_radius=binary_radius,
    )
    video_id = args.video_id or args.gaze_zip.stem
    heatmap_png = args.output_dir / f"{video_id}_gaze_heatmap.png"
    write_heatmap_png(heat, heatmap_png, map_type=args.map_type)

    overlay_path = None
    if args.video:
        overlay_path = args.output_dir / f"{video_id}_gaze_overlay.mp4"
        write_overlay(
            args.video,
            points_xy,
            overlay_path,
            args.max_overlay_frames,
            args.sigma,
            map_type=args.map_type,
            binary_radius=binary_radius,
            overlay_alpha=args.overlay_alpha,
        )

    report = {
        "gaze_zip": str(args.gaze_zip),
        "video": str(args.video) if args.video else None,
        "extract_dir": str(extract_dir),
        "members": len(members),
        "gaze_file": str(gaze_file) if gaze_file else None,
        "candidate_files": tried,
        "raw_points": len(points),
        "usable_points": int(len(points_xy)),
        "width": width,
        "height": height,
        "map_type": args.map_type,
        "sigma": args.sigma,
        "binary_radius": binary_radius,
        "overlay_alpha": args.overlay_alpha,
        "fallback_fov_deg": args.fallback_fov_deg,
        "heatmap_png": str(heatmap_png),
        "overlay_video": str(overlay_path) if overlay_path else None,
    }
    report_path = args.output_dir / f"{video_id}_gaze_heatmap_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
