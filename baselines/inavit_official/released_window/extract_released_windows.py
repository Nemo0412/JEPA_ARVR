"""Frames for the released-InAViT validation windows vs an honest 1s control (EK100 val, CPU only).

Window definitions (nominal clock F = 60 fps for EK55 ids, 50 for EK100 ids; released
EpicKitchensVideoRecord: start = round(start_ts * F), num = round(stop_ts * F) - start):
  R<k>  released anticipation window, view k of 10 (`pack_frames_to_video_clip` +
        `get_start_end_idx` + `temporal_sampling`, TARGET_FPS 30, 16 x 4 frames). Offsets are clamped
        to [0, num-1] relative to the action start, exactly as released.
  A1    16 frames uniform over the same 128-nominal-frame span (16*4*F/30), ending 1 s before the
        action start; no clamping into the action (rows ending before the video start are misses).
Nominal frame m maps to native decoded index round((m-1)/F * vfps); the h5 box key is m.
Pixels: MD5-verified original MP4 -> 456x256 PIL bilinear -> JPEG q90 (the frames_mp4_native recipe),
written to <out>/frames/<video>/frame_{i+1:010d}.jpg; plans to <out>/plans/<video>.json.
Usage: extract_released_windows.py <out> <video> [<video> ...]
"""
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import av
import torch
from PIL import Image

ROOT = Path("/scratch/yh6416/VJEPA2-EXP")
COLLAB = ROOT / "outputs/collab_inavit_ek100"
ANN = ROOT / "outputs/inavit_official_ek100/released_window/EPIC_100_validation.csv"
ARCHIVE = "/archive/yh6416/VJEPA2-EXP/datasets/ek100_original_videos"
VIEWS = (0, 5)
SSH = ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
       "-o", "HostKeyAlias=dtn.torch.hpc.nyu.edu", "-o", f"UserKnownHostsFile={COLLAB / 'meta/dtn_known_hosts'}"]
RECIPE = {"source": "official-md5-verified-original-mp4", "frame_clock": "native-decoded-index",
          "resize": [456, 256], "resize_filter": "PIL-bilinear", "jpeg_quality": 90, "jpeg_subsampling": 2}


def timestamp_to_sec(ts):  # released epickitchens_record.timestamp_to_sec
    x = time.strptime(ts, "%H:%M:%S.%f")
    return float(timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()) + \
        float(ts.split(".")[-1]) / 100


def nominal_fps(video):
    return 50 if len(video.split("_")[1]) == 3 else 60


def released_window(start, num, fps, view, num_views=10, num_samples=16, sampling_rate=4, target_fps=30):
    """Nominal 1-based frame numbers of the released anticipation clip (validation pkl branch)."""
    clip_size = num_samples * sampling_rate * fps / target_fps
    delta = max(num - clip_size, 0)
    start_idx = delta * view / num_views
    end_idx = start_idx - fps / target_fps
    start_idx = end_idx - num_samples * sampling_rate * fps / target_fps
    if start_idx < 0:
        start_idx = 0
    start_idx, end_idx = start_idx + 1, end_idx + 1
    index = torch.clamp(torch.linspace(start_idx, end_idx, num_samples), 0, num - 1).long()
    return (start + index).tolist()


def honest_window(start, fps, gap_s=1.0, num_samples=16, sampling_rate=4, target_fps=30):
    clip_size = num_samples * sampling_rate * fps / target_fps
    end = start - gap_s * fps
    if end < 1:
        return None
    return torch.clamp(torch.linspace(end - clip_size, end, num_samples), 1, None).long().tolist()


def main():
    out, videos = Path(sys.argv[1]), sys.argv[2:]
    rows = defaultdict(list)
    with ANN.open() as f:
        for r in csv.DictReader(f):
            rows[r["video_id"]].append(r)
    info = {r["video_id"]: (float(r["duration"]), float(r["fps"])) for r in csv.DictReader((COLLAB / "meta/EPIC_100_video_info.csv").open())}
    staging = out / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    (out / "plans").mkdir(exist_ok=True)
    for video in videos:
        target = out / "frames" / video
        if (target / "_meta.json").exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        receipt = json.loads((COLLAB / "hf_staging/.receipts" / f"{video}.json").read_text())
        assert receipt["status"] == "verified", video
        source = staging / f"{video}.MP4"
        probes = list((COLLAB / "hf_probe").glob(f"*/{video}.MP4"))
        if probes:
            source = probes[0]
        else:
            subprocess.run(["rsync", "-t", "--partial", "--timeout=120", "-e", shlex.join(SSH),
                            f"dtn011.hpc.nyu.edu:{ARCHIVE}/videos/{video}.MP4", str(source)], check=True, timeout=1800)
        md5 = hashlib.md5()
        with source.open("rb") as f:
            while block := f.read(8 << 20):
                md5.update(block)
        assert md5.hexdigest() == receipt["md5"], video
        container = av.open(str(source))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        n, vfps = stream.frames, float(stream.average_rate)
        assert abs(vfps - info[video][1]) < 0.001, video
        F = nominal_fps(video)
        plan = {}
        for r in rows[video]:
            s = int(round(timestamp_to_sec(r["start_timestamp"]) * F))
            e = int(round(timestamp_to_sec(r["stop_timestamp"]) * F))
            entry = {"start_nominal": s, "num": e - s}
            for k in VIEWS:
                entry[f"R{k}"] = released_window(s, e - s, F, k)
                assert min(entry[f"R{k}"]) >= s, "released window must not precede the action (clamp)"
            entry["A1"] = honest_window(s, F)
            for key in [f"R{k}" for k in VIEWS] + ["A1"]:
                if entry[key] is not None:
                    entry[key + "_native"] = [min(n - 1, max(0, int(round((m - 1) / F * vfps)))) for m in entry[key]]
            plan[r["narration_id"]] = entry
        wanted = sorted({i for e in plan.values() for key, v in e.items() if key.endswith("_native") for i in v})
        wset, saved = set(wanted), 0
        first = None
        for index, frame in enumerate(container.decode(video=0)):
            if first is None:
                first = float(frame.time)
            if index in wset:
                frame.to_image().resize((456, 256), Image.Resampling.BILINEAR).save(
                    target / f"frame_{index + 1:010d}.jpg", format="JPEG", quality=90, subsampling=2)
                saved += 1
            if index >= wanted[-1]:
                break
        container.close()
        assert saved == len(wanted), (video, saved, len(wanted))
        (out / "plans" / f"{video}.json").write_text(json.dumps(plan))
        meta = {"video_id": video, "n_frames": n, "vfps": vfps, "nominal_fps": F, "frames": saved,
                "source_md5": md5.hexdigest(), "pixel_recipe": RECIPE}
        (target / "_meta.tmp").write_text(json.dumps(meta))
        (target / "_meta.tmp").replace(target / "_meta.json")
        if source.parent == staging:
            source.unlink()
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
