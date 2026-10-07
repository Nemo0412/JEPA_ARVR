"""Build native video FPS/frame metadata without making a label split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path

from .egtea_cache import ANNOTATIONS
from .stream_source_audit import STATS


def probe(path: Path) -> dict:
    def run(count_frames: bool = False):
        command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
        if count_frames:
            command.append("-count_frames")
        command += ["-show_entries", "stream=avg_frame_rate,nb_frames,nb_read_frames",
                    "-of", "json", str(path)]
        return json.loads(subprocess.check_output(command, text=True))["streams"][0]

    info = run()
    fps = float(Fraction(info["avg_frame_rate"]))
    count = info.get("nb_frames", "N/A")
    if count in ("N/A", None):
        count = run(True).get("nb_read_frames")
    frames = int(count)
    if not math.isfinite(fps) or fps <= 0 or frames <= 0:
        raise ValueError(f"Invalid native video clock: {path}")
    return {"fps": fps, "frames": frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=STATS)
    args = parser.parse_args()
    videos = set()
    for name in ("training1.csv", "validation1.csv"):
        with (args.annotations / name).open(newline="") as stream:
            videos.update(row[1] for row in csv.reader(stream))
    if len(videos) != 86:
        raise ValueError("Expected the 86 official EGTEA session videos")
    metadata = {}
    for video in sorted(videos):
        candidates = [args.video_root / f"{video}.mp4",
                      args.video_root / f"{video}.MP4",
                      args.video_root / video / f"{video}.MP4",
                      args.video_root / video / f"{video}.mp4"]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise ValueError(f"Expected one source video for {video}, found {matches}")
        metadata[video] = probe(matches[0])
    # The official label memberships come from the RU-LSTM CSVs. This file
    # supplies only native clocks, never a midpoint or a new train/val split.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps({"video_metadata": metadata}, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "videos": len(metadata)}), flush=True)


if __name__ == "__main__":
    main()
