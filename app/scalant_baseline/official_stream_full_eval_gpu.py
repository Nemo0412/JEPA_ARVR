"""Dump all selected-checkpoint split-1 future heads on official 4s ticks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .official_stream_audit import official_actions, target
from .official_stream_data import build_official_split1_ticks
from .official_stream_train import (
    PROTOCOL as TRAIN_PROTOCOL,
    file_sha256,
    group_by_video,
    load_checkpoint,
    make_model_and_optimizer,
    source_identity,
)
from .stream_data import StreamingTSNReader
from .stream_model import FullPrefixStreamingScalAnt


PROTOCOL = "egtea-action-anticipation/official-split1-stream4s-selected-full-heads-v1"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=(2, 6), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    latest = torch.load(args.checkpoint_dir / "latest.pt", map_location="cpu", weights_only=True)
    if (latest["protocol"] != TRAIN_PROTOCOL
            or latest["horizon_seconds"] != args.horizon
            or latest["completed_epoch"] != 49
            or latest["source_identity"] != source_identity()):
        raise ValueError("Final split-1 checkpoint identity or cursor mismatch")
    selected_epoch = latest["best_epoch"]
    selected_recall = latest["best_recall"]
    del latest
    saved = json.loads((args.checkpoint_dir / f"epoch_{selected_epoch:02d}.json").read_text())
    if (saved["selected_best_epoch"] != selected_epoch
            or saved["validation"]["action_mean_recall_at5_present_classes"] != selected_recall):
        raise ValueError("Saved epoch does not agree with selected checkpoint")

    model, optimizer, scheduler = make_model_and_optimizer(args.horizon)
    selected = load_checkpoint(args.checkpoint_dir / "best.pt", model, optimizer,
                               scheduler, args.horizon)
    if (selected["completed_epoch"] != selected_epoch
            or selected["best_epoch"] != selected_epoch
            or selected["best_recall"] != selected_recall):
        raise ValueError("Best checkpoint does not agree with final selection")
    del selected, optimizer, scheduler
    model.eval()

    ticks = build_official_split1_ticks("val")
    if len(ticks) != 16227:
        raise ValueError("Official split-1 validation tick count changed")
    by_video, ids = official_actions()
    if len(ids) != 10321:
        raise ValueError("Official split-1 action membership changed")
    stream = FullPrefixStreamingScalAnt(model)
    reader = StreamingTSNReader()
    rows = {key: [] for key in ("video", "tick_frame", "uid", "action", "verb", "noun",
                                "action_logits", "verb_logits", "noun_logits")}
    processed = 0
    horizon_index = 0 if args.horizon == 2 else 1
    try:
        for video, video_ticks in sorted(group_by_video(ticks).items()):
            for tick in video_ticks:
                chunk = reader.read(tick).unsqueeze(0).cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = stream.process(video, chunk)
                processed += 1
                action = tick.target_actions[horizon_index]
                if action < 0:
                    continue
                frame_30hz = round(tick.tick_frame * 30 / tick.fps)
                selected_action = target(by_video[video], frame_30hz + 30 * args.horizon)
                if (selected_action is None or selected_action[3] != "val"
                        or selected_action[2] != action):
                    raise ValueError(f"Target UID mismatch at {video}/{tick.tick_frame}")
                logits = (output.future_actions[0, -1].float(),
                          output.future_verbs[0, -1].float(),
                          output.future_nouns[0, -1].float())
                if (tuple(x.numel() for x in logits) != (106, 19, 53)
                        or not all(torch.isfinite(x).all() for x in logits)):
                    raise ValueError(f"Missing/nonfinite head at {video}/{tick.tick_frame}")
                rows["video"].append(video)
                rows["tick_frame"].append(tick.tick_frame)
                rows["uid"].append(selected_action[4])
                rows["action"].append(action)
                rows["verb"].append(tick.target_verbs[horizon_index])
                rows["noun"].append(tick.target_nouns[horizon_index])
                for name, value in zip(("action_logits", "verb_logits", "noun_logits"), logits):
                    rows[name].append(value.cpu().numpy())
    finally:
        reader.close()

    expected = 4678 if args.horizon == 2 else 4655
    if processed != 16227 or len(rows["uid"]) != expected:
        raise ValueError(f"Incomplete split-1 validation: {processed}/{len(rows['uid'])}")
    arrays = {
        "video": np.asarray(rows["video"]),
        **{name: np.asarray(rows[name], dtype=np.int32)
           for name in ("tick_frame", "uid", "action", "verb", "noun")},
        **{name: np.stack(rows[name]).astype(np.float32)
           for name in ("action_logits", "verb_logits", "noun_logits")},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / f"s1_h{args.horizon}_selected_full_heads.npz"
    temporary = archive.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream_file:
        np.savez_compressed(stream_file, **arrays)
    os.replace(temporary, archive)
    report = {
        "protocol": PROTOCOL,
        "training_protocol": TRAIN_PROTOCOL,
        "horizon_seconds": args.horizon,
        "selected_epoch": selected_epoch,
        "selected_action_recall5": selected_recall,
        "checkpoint": str(args.checkpoint_dir / "best.pt"),
        "checkpoint_sha256": file_sha256(args.checkpoint_dir / "best.pt"),
        "source_identity": source_identity(),
        "processed_ticks": processed,
        "scored_ticks": expected,
        "archive": str(archive),
        "archive_sha256": file_sha256(archive),
    }
    metadata = args.output_dir / f"s1_h{args.horizon}_selected_full_heads.json"
    temporary = metadata.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, metadata)
    print(json.dumps({"metadata": str(metadata), "scored_ticks": expected,
                      "selected_epoch": selected_epoch}), flush=True)


if __name__ == "__main__":
    main()
