"""Replay official split-1 validation from the selected ScalAnt checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .official_stream_data import build_official_split1_ticks
from .official_stream_train import (
    PROTOCOL,
    group_by_video,
    load_checkpoint,
    make_model_and_optimizer,
    source_identity,
    validate,
)
from .stream_data import StreamingTSNReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=(2, 6), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected = args.output_dir / "best.pt"
    latest = args.output_dir / "latest.pt"
    expected_json = args.output_dir / "replay_best.json"
    latest_payload = torch.load(latest, map_location="cpu", weights_only=True)
    if (latest_payload["protocol"] != PROTOCOL
            or latest_payload["horizon_seconds"] != args.horizon
            or latest_payload["completed_epoch"] != 49
            or latest_payload["source_identity"] != source_identity()):
        raise ValueError("Final latest checkpoint identity or cursor mismatch")
    selected_epoch = latest_payload["best_epoch"]
    selected_recall = latest_payload["best_recall"]
    del latest_payload

    expected = json.loads((args.output_dir / f"epoch_{selected_epoch:02d}.json").read_text())
    if expected["selected_best_epoch"] != selected_epoch:
        raise ValueError("Selected epoch record does not agree with final checkpoint")
    if expected["validation"]["action_mean_recall_at5_present_classes"] != selected_recall:
        raise ValueError("Selected recall does not agree with final checkpoint")

    model, optimizer, scheduler = make_model_and_optimizer(args.horizon)
    selected_payload = load_checkpoint(selected, model, optimizer, scheduler,
                                       args.horizon)
    if (selected_payload["completed_epoch"] != selected_epoch
            or selected_payload["best_epoch"] != selected_epoch
            or selected_payload["best_recall"] != selected_recall):
        raise ValueError("Best checkpoint does not agree with final selection")
    del selected_payload

    ticks = build_official_split1_ticks("val")
    if len(ticks) != 16227:
        raise ValueError("Validation tick population changed")
    reader = StreamingTSNReader()
    try:
        replay = validate(model, reader, group_by_video(ticks), args.horizon)
    finally:
        reader.close()
    replay = json.loads(json.dumps(replay))

    prior = expected["validation"]
    fields = ("processed_ticks", "scored_tick_count", "classes_with_scored_ticks",
              "action_top5_micro", "action_mean_recall_at5_present_classes",
              "by_class_total")
    mismatches = {field: [prior[field], replay[field]] for field in fields
                  if prior[field] != replay[field]}
    result = {"protocol": PROTOCOL, "horizon_seconds": args.horizon,
              "final_epoch": 49, "selected_epoch": selected_epoch,
              "checkpoint": str(selected), "metric_replay_exact": not mismatches,
              "mismatches": mismatches, "validation": replay}
    temporary = expected_json.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(expected_json)
    print(json.dumps({"horizon": args.horizon, "selected_epoch": selected_epoch,
                      "metric_replay_exact": not mismatches,
                      "action_recall5": replay["action_mean_recall_at5_present_classes"],
                      "action_top5": replay["action_top5_micro"]}), flush=True)
    if mismatches:
        raise ValueError(f"Selected-checkpoint replay differs in {list(mismatches)}")


if __name__ == "__main__":
    main()
