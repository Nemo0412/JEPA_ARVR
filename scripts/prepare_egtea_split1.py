#!/usr/bin/env python3
"""Prepare EGTEA Gaze+ split1 for V-JEPA video+gaze training."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd

OUTPUT_COLUMNS = [
    "participant_id",
    "video_id",
    "original_video_id",
    "start_frame",
    "stop_frame",
    "verb_class",
    "noun_class",
    "start_timestamp",
    "end_timestamp",
    "narration",
]

# RULSTM CSV column order (see rulstm README): id, video, start, stop, VERB, NOUN, ACTION.
# EGTEA Gaze+ has 19 verbs / 51 nouns / 106 actions, matching cols 5/6/7 respectively.
RULSTM_COLUMNS = [
    "clip_id",
    "video_name",
    "start_frame",
    "stop_frame",
    "verb_class",
    "noun_class",
    "action_class",
]


# RULSTM's EGTEA CSVs express start/stop frames assuming a 30 fps timebase
# (same convention they use for EPIC), but the raw EGTEA videos decode at 24 fps.
# The V-JEPA EK100 loader uses start_frame/stop_frame DIRECTLY as decord frame
# indices on the native-fps decode, so we must rescale 30fps->24fps (factor 0.8),
# otherwise every action is sought ~25% too late in the video (verified: RULSTM
# frame == original EGTEA 24fps frame * 30/24).
RULSTM_FPS = 30.0
VIDEO_FPS = 24.0


def parse_args():
    p = argparse.ArgumentParser(description="Prepare EGTEA split1 for V-JEPA training.")
    p.add_argument("--egtea-root", type=Path, default=Path("/scratch/ll5914/datasets/EGTEA"))
    p.add_argument("--rulstm-fps", type=float, default=RULSTM_FPS,
                   help="Frame-rate assumed by the RULSTM annotation CSVs.")
    p.add_argument("--video-fps", type=float, default=VIDEO_FPS,
                   help="Native frame-rate of the raw EGTEA videos.")
    p.add_argument("--video-ext", default=".mp4")
    return p.parse_args()


def to_vjepa_video_id(name: str) -> tuple[str, str]:
    name = str(name).strip()
    if name.startswith("OP"):
        participant = name.split("-", 1)[0]
    else:
        participant = name.split("-", 1)[0]
    if "-" in name:
        rest = name[len(participant) + 1 :]
        video_id = f"{participant}_{rest}"
    else:
        video_id = name.replace("-", "_", 1)
    return participant, video_id


def convert_rulstm_csv(src: Path, dst: Path, rulstm_fps: float, video_fps: float):
    scale = video_fps / rulstm_fps  # 24/30 = 0.8
    df = pd.read_csv(src, header=None, names=RULSTM_COLUMNS)
    rows = []
    for _, row in df.iterrows():
        participant, video_id = to_vjepa_video_id(row["video_name"])
        sf_raw = int(row["start_frame"])
        ef_raw = int(row["stop_frame"])
        # Rescale RULSTM (30fps) frame indices to the native video (24fps) timebase.
        sf = int(round(sf_raw * scale))
        ef = int(round(ef_raw * scale))
        rows.append(
            {
                "participant_id": participant,
                "video_id": video_id,
                "original_video_id": str(row["video_name"]).strip(),
                "start_frame": sf,
                "stop_frame": ef,
                "verb_class": int(row["verb_class"]),
                "noun_class": int(row["noun_class"]),
                "start_timestamp": sf_raw / rulstm_fps,
                "end_timestamp": ef_raw / rulstm_fps,
                "narration": f"action_{int(row['action_class'])}",
            }
        )
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return out


def find_videos(video_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in video_root.rglob(f"*{args.video_ext}"):
        stem = path.stem
        mapping[stem] = path
        participant, vid = to_vjepa_video_id(stem)
        mapping[vid] = path
        mapping[f"{participant}-{stem.split('-', 1)[-1]}" if "-" in stem else stem] = path
    return mapping


def link_videos(df: pd.DataFrame, video_index: dict[str, Path], out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    stats = {"linked": 0, "missing": []}
    seen = set()
    for _, row in df.iterrows():
        vid = row["video_id"]
        if vid in seen:
            continue
        seen.add(vid)
        orig = row["original_video_id"]
        src = video_index.get(orig) or video_index.get(vid) or video_index.get(orig.replace("_", "-", 1))
        if src is None:
            stats["missing"].append(orig)
            continue
        participant = row["participant_id"]
        dst_dir = out_root / participant
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{vid}.MP4"
        if dst.exists() or dst.is_symlink():
            stats["linked"] += 1
            continue
        dst.symlink_to(src.resolve())
        stats["linked"] += 1
    return stats


def parse_gaze_txt(txt_path: Path, fps: float) -> pd.DataFrame:
    rows = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,;]+", line)
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if len(nums) < 3:
            continue
        frame = nums[0]
        x, y = nums[1], nums[2]
        if x > 2.0 or y > 2.0:
            x = x / 1280.0
            y = y / 960.0
        ts_us = (frame / fps) * 1_000_000.0
        rows.append((ts_us, x, y))
    if not rows:
        return pd.DataFrame(columns=["tracking_timestamp_us", "gaze_x", "gaze_y"])
    df = pd.DataFrame(rows, columns=["tracking_timestamp_us", "gaze_x", "gaze_y"])
    return df.sort_values("tracking_timestamp_us").drop_duplicates("tracking_timestamp_us")


def convert_gaze(gaze_root: Path, extract_root: Path, fps: float) -> dict:
    extract_root.mkdir(parents=True, exist_ok=True)
    stats = {"converted": 0, "missing": 0}
    txt_files = list(gaze_root.rglob("*.txt"))
    index = {p.stem: p for p in txt_files}
    for stem, txt in index.items():
        participant, vid = to_vjepa_video_id(stem)
        out_dir = extract_root / vid
        out_csv = out_dir / "general_eye_gaze.csv"
        df = parse_gaze_txt(txt, fps=fps)
        if df.empty:
            stats["missing"] += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        stats["converted"] += 1
    return stats


def main():
    global args
    args = parse_args()
    root = args.egtea_root
    ann_dir = root / "hdepic_vjepa_annotations" / "split1"
    ann_dir.mkdir(parents=True, exist_ok=True)

    train_df = convert_rulstm_csv(root / "splits/training1.csv", ann_dir / "EGTEA_train_vjepa.csv", args.rulstm_fps, args.video_fps)
    val_df = convert_rulstm_csv(root / "splits/validation1.csv", ann_dir / "EGTEA_val_vjepa.csv", args.rulstm_fps, args.video_fps)
    test_df = val_df.copy()
    test_df.to_csv(ann_dir / "EGTEA_test_vjepa.csv", index=False)

    video_roots = [root / "Raw_Videos", root / "Videos/extracted", root / "Videos"]
    video_index: dict[str, Path] = {}
    for video_src in video_roots:
        if video_src.exists():
            video_index.update(find_videos(video_src))
    if not video_index:
        raise FileNotFoundError(
            f"No EGTEA videos found under {[str(p) for p in video_roots]}. "
            "Run /scratch/ll5914/logs/download_egtea_raw_videos.sh first."
        )
    video_out = root / "egtea_vjepa_videos"
    link_stats = link_videos(pd.concat([train_df, val_df], ignore_index=True), video_index, video_out)

    gaze_src = root / "gaze_raw/extracted"
    if not gaze_src.exists():
        gaze_src = root / "Gaze_Data"
    gaze_stats = convert_gaze(gaze_src, root / "gaze_extract", fps=args.video_fps)

    summary = {
        "train_clips": len(train_df),
        "val_clips": len(val_df),
        "videos_linked": link_stats["linked"],
        "videos_missing": sorted(set(link_stats["missing"]))[:20],
        "videos_missing_count": len(set(link_stats["missing"])),
        "gaze_converted": gaze_stats["converted"],
        "gaze_missing": gaze_stats["missing"],
        "annotation_dir": str(ann_dir),
        "video_root": str(video_out),
        "gaze_extract_root": str(root / "gaze_extract"),
    }
    (ann_dir / "prep_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
