#!/usr/bin/env python3
"""Create the video layout expected by ll's streaming datasets.

The frozen EGTEA stream CSV uses ``video_id=<session>``.  ll's loader resolves
``<video_root>/<video_id.split('_')[0]>/<video_id>.MP4``.  EGTEA full-session
videos are stored flat as ``<session_videos>/<session>.mp4``; this utility
creates the required nested symlink tree without copying video bytes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--session-video-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.csv.open(newline="") as handle:
        sessions = sorted({str(row["video_id"]) for row in csv.DictReader(handle)})

    made = reused = 0
    for session in sessions:
        source = None
        for extension in (".mp4", ".MP4", ".mkv"):
            candidate = args.session_video_root / f"{session}{extension}"
            if candidate.is_file():
                source = candidate.resolve()
                break
        if source is None:
            raise FileNotFoundError(f"missing full-session video for {session}")

        target_dir = args.out / session
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{session}.MP4"
        if target.is_symlink() and target.resolve() == source:
            reused += 1
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace existing target: {target}")
        target.symlink_to(source)
        made += 1

    print(f"stream video tree: sessions={len(sessions)} made={made} reused={reused} out={args.out}")


if __name__ == "__main__":
    main()
