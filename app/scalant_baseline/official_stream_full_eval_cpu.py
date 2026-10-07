"""Audit split-1 selected-checkpoint logits and score all future heads."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .egtea_cache import load_actions
from .official_stream_audit import official_actions, target
from .official_stream_data import build_official_split1_ticks
from .official_stream_full_eval_gpu import PROTOCOL
from .official_stream_train import file_sha256, group_by_video, source_identity


def score(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    if (logits.ndim != 2 or labels.shape != (len(logits),)
            or not torch.isfinite(logits).all()
            or ((labels < 0) | (labels >= logits.shape[1])).any()):
        raise ValueError("Invalid scored logits or labels")
    hits = (logits.topk(5, dim=1).indices == labels[:, None]).any(dim=1)
    total = Counter(labels.tolist())
    correct = Counter(labels[hits].tolist())
    return {
        "count": len(labels),
        "supported_classes": len(total),
        "class_count": logits.shape[1],
        "top5_micro": float(hits.float().mean()),
        "mean_recall_at5_present_classes": sum(correct[c] / n for c, n in total.items()) / len(total),
        "by_class_total": dict(sorted(total.items())),
        "by_class_top5": dict(sorted(correct.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    mapping = load_actions()
    by_video, memberships = official_actions()
    if len(mapping) != 106 or len(memberships) != 10321:
        raise ValueError("Official split-1 action mapping changed")
    ticks = build_official_split1_ticks("val")
    if len(ticks) != 16227:
        raise ValueError("Official split-1 validation tick grid changed")
    reports = {}
    for horizon in (2, 6):
        metadata_path = args.input_dir / f"s1_h{horizon}_selected_full_heads.json"
        metadata = json.loads(metadata_path.read_text())
        archive = args.input_dir / f"s1_h{horizon}_selected_full_heads.npz"
        expected_n = 4678 if horizon == 2 else 4655
        if (metadata["protocol"] != PROTOCOL
                or metadata["horizon_seconds"] != horizon
                or metadata["processed_ticks"] != 16227
                or metadata["scored_ticks"] != expected_n
                or metadata["source_identity"] != source_identity()
                or metadata["archive"] != str(archive)
                or metadata["archive_sha256"] != file_sha256(archive)):
            raise ValueError(f"Full-head artifact provenance mismatch for h{horizon}")
        with np.load(archive, allow_pickle=False) as saved:
            values = {name: saved[name] for name in saved.files}
        dimensions = {"action_logits": 106, "verb_logits": 19, "noun_logits": 53}
        if (set(values) != {"video", "tick_frame", "uid", "action", "verb", "noun", *dimensions}
                or any(values[name].shape != (expected_n, size) or values[name].dtype != np.float32
                       for name, size in dimensions.items())
                or any(values[name].shape != (expected_n,)
                       for name in ("video", "tick_frame", "uid", "action", "verb", "noun"))):
            raise ValueError(f"Full-head archive shape mismatch for h{horizon}")

        expected = {name: [] for name in ("video", "tick_frame", "uid", "action", "verb", "noun")}
        index = 0 if horizon == 2 else 1
        for video, video_ticks in sorted(group_by_video(ticks).items()):
            for tick in video_ticks:
                action = tick.target_actions[index]
                if action < 0:
                    continue
                frame_30hz = round(tick.tick_frame * 30 / tick.fps)
                chosen = target(by_video[video], frame_30hz + 30 * horizon)
                if chosen is None or chosen[3] != "val" or chosen[2] != action:
                    raise ValueError(f"Unexpected official target at {video}/{tick.tick_frame}")
                for name, value in (("video", video), ("tick_frame", tick.tick_frame),
                                    ("uid", chosen[4]), ("action", action),
                                    ("verb", tick.target_verbs[index]),
                                    ("noun", tick.target_nouns[index])):
                    expected[name].append(value)
        for name, sequence in expected.items():
            if not np.array_equal(values[name], np.asarray(sequence)):
                raise ValueError(f"Scored {name} sequence differs from official ticks for h{horizon}")
        if len(set(values["uid"].tolist())) != 1626:
            raise ValueError(f"Covered official validation action count changed for h{horizon}")
        for action, verb, noun in zip(values["action"], values["verb"], values["noun"]):
            if mapping[int(action)] != (int(verb), int(noun)):
                raise ValueError(f"Verb/noun label mapping differs at action {action}")

        action_logits = torch.from_numpy(values["action_logits"])
        action_labels = torch.from_numpy(values["action"].astype(np.int64))
        verb_labels = torch.from_numpy(values["verb"].astype(np.int64))
        noun_labels = torch.from_numpy(values["noun"].astype(np.int64))
        verb_map = torch.zeros(106, 19)
        noun_map = torch.zeros(106, 53)
        for action, (verb, noun) in enumerate(mapping):
            verb_map[action, verb] = 1
            noun_map[action, noun] = 1
        metrics = {
            "action": score(action_logits, action_labels),
            "verb_direct": score(torch.from_numpy(values["verb_logits"]), verb_labels),
            "noun_direct": score(torch.from_numpy(values["noun_logits"]), noun_labels),
            # The pinned ScalAnt source sums raw action scores through these
            # class mappings; this differs from RU-LSTM's probability sum.
            "verb_from_action_source": score(action_logits @ verb_map, verb_labels),
            "noun_from_action_source": score(action_logits @ noun_map, noun_labels),
        }
        reports[str(horizon)] = {
            "metadata": str(metadata_path),
            "selected_epoch": metadata["selected_epoch"],
            "selected_action_recall5_training_time": metadata["selected_action_recall5"],
            "action_recall5_replay_delta": (metrics["action"]["mean_recall_at5_present_classes"]
                                            - metadata["selected_action_recall5"]),
            "scored_ticks": expected_n,
            "unique_action_uids": 1626,
            "metrics": metrics,
        }
    report = {"protocol": PROTOCOL, "kind": "selected_checkpoint_full_head_validation",
              "scope": "official_split1_streaming_ticks", "horizons": reports}
    output = args.input_dir / "split1_selected_full_heads_report.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "horizons": {
        h: {head: {"mean_recall5": metrics["mean_recall_at5_present_classes"],
                   "top5": metrics["top5_micro"]}
            for head, metrics in item["metrics"].items()}
        for h, item in reports.items()}}), flush=True)


if __name__ == "__main__":
    main()
