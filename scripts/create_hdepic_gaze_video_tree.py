#!/usr/bin/env python3
"""Create a V-JEPA video tree that swaps available videos with gaze overlays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video-root", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--output-video-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    return parser.parse_args()


def video_id_from_vjepa_name(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        participant, rest = stem.split("_", 1)
        return f"{participant}-{rest}"
    return stem


def find_overlay(overlay_dir: Path, video_id: str) -> Path | None:
    candidates = [
        overlay_dir / f"{video_id}_gaze_overlay.mp4",
        overlay_dir / f"{video_id}_gaze_overlay.MP4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(overlay_dir.rglob(f"{video_id}_gaze_overlay.mp4"))
    if matches:
        return matches[0].resolve()
    return None


def link_or_replace(src: Path, dst: Path, replace_existing: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not replace_existing:
            return
        dst.unlink()
    os.symlink(str(src), str(dst))


def main():
    args = parse_args()
    source_root = args.source_video_root.resolve()
    overlay_dir = args.overlay_dir.resolve()
    output_root = args.output_video_root.resolve()

    rows = []
    replaced = 0
    linked_original = 0
    for src in sorted(source_root.rglob("*")):
        if not src.is_file() and not src.is_symlink():
            continue
        if src.suffix not in VIDEO_EXTS:
            continue
        rel = src.relative_to(source_root)
        dst = output_root / rel
        video_id = video_id_from_vjepa_name(src)
        overlay = find_overlay(overlay_dir, video_id)
        if overlay is not None:
            link_or_replace(overlay, dst, args.replace_existing)
            target = overlay
            used_overlay = True
            replaced += 1
        else:
            link_or_replace(src.resolve(), dst, args.replace_existing)
            target = src.resolve()
            used_overlay = False
            linked_original += 1
        rows.append(
            {
                "relative_path": str(rel),
                "video_id": video_id,
                "target": str(target),
                "used_overlay": used_overlay,
            }
        )

    report = {
        "source_video_root": str(source_root),
        "overlay_dir": str(overlay_dir),
        "output_video_root": str(output_root),
        "total_videos": len(rows),
        "replaced_with_overlay": replaced,
        "linked_original": linked_original,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote: {args.report}")
    print(f"total_videos: {len(rows)}")
    print(f"replaced_with_overlay: {replaced}")
    print(f"linked_original: {linked_original}")


if __name__ == "__main__":
    main()
