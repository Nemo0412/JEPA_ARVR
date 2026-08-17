#!/usr/bin/env python3
"""Create the strict B13 EGTEA gaze-coverage and approved-fallback manifest.

The only permitted synthetic gaze is the user-approved (0.5, 0.5) center
value for exact frozen validation rows whose BeGaze export ends early.  This
tool derives that row allowlist from the immutable CSV and rejects any drift.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.hdepic_lora_action_anticipation.egtea_gaze import parse_gtea_gaze


EXPECTED = {
    "train": (25_256, "dcdc98c9993ae028dbb4a2877deff09a"),
    "val": (22_741, "cb04d6eddc2e65910899fde8b3c08659"),
}
APPROVED_OOB = {
    "OP01-R06-GreekSalad": (54, 16_917, 19_507),
    "OP03-R01-PastaSalad": (92, 18_369, 22_760),
    "OP03-R07-Pizza": (25, 15_438, 16_591),
}


def md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 -- data identity check, not security
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_indices(raw: str) -> np.ndarray:
    return np.fromiter((int(value) for value in raw.split(",") if value), dtype=np.int64)


def audit_split(split: str, path: Path, gaze_root: Path, cache: dict[str, np.ndarray]) -> tuple[dict, dict[str, list[dict]]]:
    expected_rows, expected_md5 = EXPECTED[split]
    if md5(path) != expected_md5:
        raise RuntimeError(f"{split}: frozen CSV MD5 drift: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise RuntimeError(f"{split}: row-count drift {len(rows)} != {expected_rows}")

    contexts = Counter()
    oob: dict[str, list[dict]] = defaultdict(list)
    for line, row in enumerate(rows, start=2):
        session = str(row["video_id"])
        start, tick = int(row["start_frame"]), int(row["tick_frame"])
        context, model_frames = int(float(row["context_sec"])), int(row["n_model_frames"])
        indices = parse_indices(row["frame_indices"])
        if context not in (4, 6, 8, 10) or model_frames != context * 8:
            raise RuntimeError(f"{split}:{line}: invalid context/model-frame contract")
        if len(indices) != model_frames or indices[0] != start or indices[-1] != tick - 1 or np.any(indices >= tick):
            raise RuntimeError(f"{split}:{line}: non-causal sampling")
        contexts[context] += 1
        if session not in cache:
            gaze_path = gaze_root / f"{session}.txt"
            if not gaze_path.is_file():
                raise RuntimeError(f"{split}:{line}: missing gaze file {gaze_path}")
            cache[session] = parse_gtea_gaze(gaze_path)
        if tick > len(cache[session]):
            oob[session].append(
                {"line": line, "start_frame": start, "last_frame": int(indices[-1]), "tick_frame": tick, "context_sec": context}
            )
    expected_contexts = {4: 86, 6: 86, 8: 86, 10: expected_rows - 258}
    if dict(contexts) != expected_contexts:
        raise RuntimeError(f"{split}: context drift {dict(contexts)} != {expected_contexts}")
    return ({"csv": path.name, "md5": expected_md5, "rows": expected_rows,
             "context_rows": {str(k): contexts[k] for k in (4, 6, 8, 10)},
             "rows_gaze_timeline_oob": sum(len(v) for v in oob.values())}, oob)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--gaze-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache: dict[str, np.ndarray] = {}
    train, train_oob = audit_split("train", args.annotation_dir / "EGTEA_train_stream_mtp.csv", args.gaze_root, cache)
    val, val_oob = audit_split("val", args.annotation_dir / "EGTEA_val_stream_mtp.csv", args.gaze_root, cache)
    if train_oob:
        raise RuntimeError(f"train contains unapproved gaze OOB rows: {sorted(train_oob)}")
    if set(val_oob) != set(APPROVED_OOB):
        raise RuntimeError(f"validation OOB session drift: {sorted(val_oob)}")
    for session, (count, first_oob, last_oob) in APPROVED_OOB.items():
        rows = val_oob[session]
        gaze_len = len(cache[session])
        actual = (len(rows), gaze_len, max(row["last_frame"] for row in rows))
        if actual != (count, first_oob, last_oob) or any(row["context_sec"] != 10 for row in rows):
            raise RuntimeError(f"approved fallback row drift for {session}: {actual}")
    payload = {
        "schema_version": 1,
        "protocol": "egtea_split1_temporal_half_streaming_mtp",
        "context_sec": [4, 6, 8, 10], "model_fps": 8, "horizons_sec": [2, 4, 6],
        "gaze_causality": {"allowed_interval": "[start_frame, tick_frame)", "future_gaze_read": False},
        "splits": {"train": train, "val": val},
        "center_fallback": {
            "policy": "user_approved_existing_egtea_fixed_clip_center_0.5",
            "xy_norm": [0.5, 0.5], "split": "val", "rows": 171,
            "sessions": {session: rows for session, rows in sorted(val_oob.items())},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("B13_EGTEA_GAZE_APPROVED_FALLBACK_AUDIT_PASS")


if __name__ == "__main__":
    main()
