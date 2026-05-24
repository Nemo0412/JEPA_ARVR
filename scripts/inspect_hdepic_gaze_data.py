#!/usr/bin/env python3
"""Inspect HD-EPIC gaze-related files from local data and downloader manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/HD-EPIC"))
    parser.add_argument("--annotations-root", type=Path, default=Path("data/hd-epic-annotations"))
    parser.add_argument("--downloader-md5", type=Path, default=None)
    parser.add_argument("--participants", nargs="*", default=["P01"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/hdepic_gaze_inspection"))
    return parser.parse_args()


def wanted_participant(path: Path, participants: set[str]) -> bool:
    if not participants:
        return True
    return any(part in path.parts or part in str(path) for part in participants)


def classify(path: Path) -> str:
    lower = str(path).lower()
    if "eye_gaze" in lower or "gaze" in lower:
        return "eye_gaze"
    if "hand_tracking" in lower:
        return "hand_tracking"
    if "slam" in lower:
        return "slam"
    return "other"


def read_manifest(md5_path: Path | None, participants: set[str]):
    rows = []
    if md5_path is None or not md5_path.exists():
        return rows
    for line in md5_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        rel = Path(" ".join(parts[1:]))
        rel_str = str(rel)
        if "SLAM-and-Gaze" not in rel_str:
            continue
        if not wanted_participant(rel, participants):
            continue
        rows.append(
            {
                "md5": parts[0],
                "relative_path": rel_str,
                "participant": next((p for p in participants if p in rel_str), None),
                "kind": classify(rel),
            }
        )
    return rows


def scan_local(raw_root: Path, participants: set[str]):
    roots = [
        raw_root / "SLAM-and-Gaze",
        raw_root / "SLAM-and-Gaze".lower(),
        raw_root,
    ]
    seen = set()
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path in seen:
                continue
            seen.add(path)
            rel = path.relative_to(raw_root) if raw_root in path.parents else path
            rel_str = str(rel)
            lower = rel_str.lower()
            if "gaze" not in lower and "slam" not in lower and "hand_tracking" not in lower:
                continue
            if not wanted_participant(rel, participants):
                continue
            rows.append(
                {
                    "path": str(path),
                    "relative_path": rel_str,
                    "participant": next((p for p in participants if p in rel_str), None),
                    "kind": classify(rel),
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def scan_annotations(annotations_root: Path):
    if not annotations_root.exists():
        return []
    rows = []
    for path in annotations_root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if "gaze" not in lower and "prim" not in lower:
            continue
        rows.append({"path": str(path), "size_bytes": path.stat().st_size})
    return rows


def write_markdown(output_path: Path, payload: dict):
    manifest = payload["manifest"]
    local = payload["local"]
    annotations = payload["annotations"]

    manifest_counts = Counter(row["kind"] for row in manifest)
    local_counts = Counter(row["kind"] for row in local)

    lines = [
        "# HD-EPIC Gaze Data Inspection",
        "",
        "## Inputs",
        "",
        f"- raw_root: `{payload['raw_root']}`",
        f"- annotations_root: `{payload['annotations_root']}`",
        f"- downloader_md5: `{payload['downloader_md5']}`",
        f"- participants: `{', '.join(payload['participants']) or 'all'}`",
        "",
        "## Manifest Summary",
        "",
        f"- slam/gaze files listed: {len(manifest)}",
    ]
    for key, value in sorted(manifest_counts.items()):
        lines.append(f"- {key}: {value}")

    lines += [
        "",
        "## Local Summary",
        "",
        f"- local gaze/slam-related files found: {len(local)}",
    ]
    for key, value in sorted(local_counts.items()):
        lines.append(f"- {key}: {value}")

    lines += [
        "",
        "## Annotation Summary",
        "",
        f"- gaze/priming annotation-like files found: {len(annotations)}",
        "",
        "## First Manifest Entries",
        "",
    ]
    for row in manifest[:30]:
        lines.append(f"- `{row['kind']}` `{row['relative_path']}`")

    lines += ["", "## First Local Entries", ""]
    for row in local[:30]:
        size_mb = row["size_bytes"] / (1024 * 1024)
        lines.append(f"- `{row['kind']}` `{row['relative_path']}` ({size_mb:.2f} MiB)")

    lines += ["", "## Annotation-Like Entries", ""]
    for row in annotations[:50]:
        lines.append(f"- `{row['path']}` ({row['size_bytes']} bytes)")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    participants = set(args.participants)
    manifest = read_manifest(args.downloader_md5, participants)
    local = scan_local(args.raw_root, participants)
    annotations = scan_annotations(args.annotations_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_root": str(args.raw_root),
        "annotations_root": str(args.annotations_root),
        "downloader_md5": str(args.downloader_md5) if args.downloader_md5 else None,
        "participants": sorted(participants),
        "manifest": manifest,
        "local": local,
        "annotations": annotations,
    }
    (args.output_dir / "gaze_data_inspection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "gaze_data_inspection.md", payload)

    print(f"manifest_slam_gaze_files: {len(manifest)}")
    print(f"local_gaze_related_files: {len(local)}")
    print(f"annotation_like_files: {len(annotations)}")
    print(f"wrote: {args.output_dir / 'gaze_data_inspection.md'}")


if __name__ == "__main__":
    main()
