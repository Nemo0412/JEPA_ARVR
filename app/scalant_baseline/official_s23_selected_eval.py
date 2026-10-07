"""Replay selected split-2/3 checkpoints on all official streaming ticks."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .egtea_cache import MAIN_ROOT, load_actions
from .official_stream_audit import official_actions, target
from .official_stream_data import build_official_split1_ticks
from .official_stream_train import (
    file_sha256, group_by_video, load_checkpoint, make_model_and_optimizer,
    protocol_for_split, source_identity,
)
from .stream_data import StreamingTSNReader
from .stream_model import FullPrefixStreamingScalAnt


HEADS = ("action", "verb_direct", "noun_direct",
         "verb_from_action_source", "noun_from_action_source")


def _score(hits: list[int], labels: list[int], class_count: int) -> dict:
    totals = Counter(labels)
    correct = Counter(label for label, hit in zip(labels, hits) if hit)
    return {
        "scored_tick_count": len(labels),
        "classes_with_scored_ticks": len(totals),
        "class_count": class_count,
        "top5_micro": sum(hits) / len(labels),
        "mean_recall_at5_present_classes":
            sum(correct[c] / n for c, n in totals.items()) / len(totals),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=(2, 3), required=True)
    parser.add_argument("--horizon", type=int, choices=(2, 6), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = (f"egtea-action-anticipation/official-split{args.split}"
                "-stream4s-selected-full-heads-v1")
    train_protocol = protocol_for_split(args.split)
    expected_dir = MAIN_ROOT / f"outputs/b19_scalant/train/official_s{args.split}_h{args.horizon}"
    if args.checkpoint_dir.resolve() != expected_dir.resolve():
        raise ValueError("Checkpoint directory does not match split and horizon")
    audit = json.loads(args.audit.read_text())
    if (audit["split"] != args.split
            or audit["kind"] != "official_stream_split_source_audit_not_training"):
        raise ValueError("Official source audit identity mismatch")
    expected = audit["population"]["val"]
    expected_ticks = expected["processed_ticks"]
    expected_scored = expected[f"scored_h{args.horizon}"]
    expected_uids = expected[f"unique_uids_h{args.horizon}"]

    latest = torch.load(args.checkpoint_dir / "latest.pt", map_location="cpu", weights_only=True)
    if (latest["protocol"] != train_protocol or latest.get("split") != args.split
            or latest["horizon_seconds"] != args.horizon
            or latest["source_identity"] != source_identity(args.split)
            or latest["completed_epoch"] < 40):
        raise ValueError("Latest checkpoint identity or terminal cursor mismatch")
    if args.split == 3 and latest["completed_epoch"] != 49:
        raise ValueError("Split-3 training did not complete epoch 49")
    selected_epoch, selected_recall = latest["best_epoch"], latest["best_recall"]
    completed_epoch = latest["completed_epoch"]
    del latest
    selected_record = json.loads(
        (args.checkpoint_dir / f"epoch_{selected_epoch:02d}.json").read_text())
    saved_validation = selected_record["validation"]
    if (selected_record["split"] != args.split
            or selected_record["horizon_seconds"] != args.horizon
            or selected_record["selected_best_epoch"] != selected_epoch
            or saved_validation["action_mean_recall_at5_present_classes"] != selected_recall
            or saved_validation["scored_tick_count"] != expected_scored):
        raise ValueError("Selected epoch record does not match checkpoint selection")

    model, optimizer, scheduler = make_model_and_optimizer(args.horizon)
    selected = load_checkpoint(args.checkpoint_dir / "best.pt", model, optimizer,
                               scheduler, args.horizon, split=args.split)
    if (selected["completed_epoch"] != selected_epoch
            or selected["best_epoch"] != selected_epoch
            or selected["best_recall"] != selected_recall):
        raise ValueError("Best checkpoint differs from selected epoch")
    del selected, optimizer, scheduler
    model.eval()

    ticks = build_official_split1_ticks("val", split=args.split)
    by_video, memberships = official_actions(split=args.split)
    mapping = load_actions()
    if len(ticks) != expected_ticks or len(memberships) != 10321 or len(mapping) != 106:
        raise ValueError("Official tick or class population changed")
    verb_map = torch.zeros(106, 19, device="cuda")
    noun_map = torch.zeros(106, 53, device="cuda")
    for action, (verb, noun) in enumerate(mapping):
        verb_map[action, verb] = 1
        noun_map[action, noun] = 1

    stream = FullPrefixStreamingScalAnt(model)
    reader = StreamingTSNReader()
    rows: dict[str, list] = {name: [] for name in
                             ("video", "tick_frame", "uid", "action", "verb", "noun")}
    hits = {name: [] for name in HEADS}
    ties = {name: [] for name in HEADS}
    processed = 0
    index = 0 if args.horizon == 2 else 1
    try:
        for video, video_ticks in sorted(group_by_video(ticks).items()):
            for tick in video_ticks:
                chunk = reader.read(tick).unsqueeze(0).cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = stream.process(video, chunk)
                processed += 1
                action = tick.target_actions[index]
                if action < 0:
                    continue
                frame_30hz = round(tick.tick_frame * 30 / tick.fps)
                chosen = target(by_video[video], frame_30hz + 30 * args.horizon)
                if (chosen is None or chosen[3] != "val" or chosen[2] != action
                        or mapping[action] != (tick.target_verbs[index],
                                               tick.target_nouns[index])):
                    raise ValueError(f"Official UID/class mismatch at {video}/{tick.tick_frame}")
                action_logits = output.future_actions[0, -1].float()
                verb_logits = output.future_verbs[0, -1].float()
                noun_logits = output.future_nouns[0, -1].float()
                vectors = {
                    "action": (action_logits, action),
                    "verb_direct": (verb_logits, tick.target_verbs[index]),
                    "noun_direct": (noun_logits, tick.target_nouns[index]),
                    # ScalAnt's released class mapping sums raw Action scores.
                    "verb_from_action_source": (action_logits @ verb_map, tick.target_verbs[index]),
                    "noun_from_action_source": (action_logits @ noun_map, tick.target_nouns[index]),
                }
                for name, (vector, label) in vectors.items():
                    if (vector.ndim != 1 or not 0 <= label < vector.numel()
                            or not torch.isfinite(vector).all()):
                        raise ValueError(f"Invalid {name} logits at {video}/{tick.tick_frame}")
                    # Match the training scorer's one-vector GPU topk(5).
                    hits[name].append(int(label in vector.topk(5).indices.tolist()))
                    top_six = vector.topk(6).values
                    ties[name].append(int(bool(top_six[4] == top_six[5])))
                for name, value in (("video", video), ("tick_frame", tick.tick_frame),
                                    ("uid", chosen[4]), ("action", action),
                                    ("verb", tick.target_verbs[index]),
                                    ("noun", tick.target_nouns[index])):
                    rows[name].append(value)
    finally:
        reader.close()
    if (processed != expected_ticks or len(rows["uid"]) != expected_scored
            or len(set(rows["uid"])) != expected_uids):
        raise ValueError("Selected-checkpoint replay missed audited validation ticks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / f"s{args.split}_h{args.horizon}_selected_hits.npz"
    arrays = {"video": np.asarray(rows["video"]),
              **{name: np.asarray(rows[name], dtype=np.int32)
                 for name in ("tick_frame", "uid", "action", "verb", "noun")},
              **{f"{name}_{suffix}": np.asarray(values[name], dtype=np.uint8)
                 for suffix, values in (("hits", hits), ("ties", ties)) for name in HEADS}}
    temporary = archive.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream_file:
        np.savez_compressed(stream_file, **arrays)
    os.replace(temporary, archive)
    with np.load(archive, allow_pickle=False) as check:
        if any(not np.array_equal(check[key], value) for key, value in arrays.items()):
            raise ValueError("Saved hit archive round-trip mismatch")

    classes = {"action": 106, "verb_direct": 19, "noun_direct": 53,
               "verb_from_action_source": 19, "noun_from_action_source": 53}
    scores = {name: {**_score(hits[name], rows["action" if name == "action" else
                                         "verb" if name.startswith("verb") else "noun"], classes[name]),
                     "boundary_tie_rows": sum(ties[name])}
              for name in HEADS}
    action_delta_hits = round((scores["action"]["top5_micro"]
                               - saved_validation["action_top5_micro"]) * expected_scored)
    if abs(action_delta_hits) > scores["action"]["boundary_tie_rows"]:
        raise ValueError("Action replay differs beyond rank-five tie bound")
    report = {
        "kind": "official_selected_checkpoint_full_head_replay",
        "protocol": protocol,
        "training_protocol": train_protocol,
        "split": args.split,
        "horizon_seconds": args.horizon,
        "completed_epoch": completed_epoch,
        "selected_epoch": selected_epoch,
        "selected_action_recall5_training_time": selected_recall,
        "selected_action_top5_training_time": saved_validation["action_top5_micro"],
        "action_replay_top5_hit_delta": action_delta_hits,
        "processed_ticks": processed,
        "scored_ticks": expected_scored,
        "unique_action_uids": expected_uids,
        "metrics_gpu_single_vector_topk": scores,
        "checkpoint": str(args.checkpoint_dir / "best.pt"),
        "checkpoint_sha256": file_sha256(args.checkpoint_dir / "best.pt"),
        "source_identity": source_identity(args.split),
        "audit": str(args.audit),
        "audit_sha256": file_sha256(args.audit),
        "hit_archive": str(archive),
        "hit_archive_sha256": file_sha256(archive),
    }
    output = args.output_dir / f"s{args.split}_h{args.horizon}_selected_full_heads.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "selected_epoch": selected_epoch,
                      "scored_ticks": expected_scored, "action_replay_hit_delta": action_delta_hits,
                      "scores": scores}), flush=True)


if __name__ == "__main__":
    main()
