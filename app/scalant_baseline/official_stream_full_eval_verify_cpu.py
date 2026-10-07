"""Cross-check saved GPU top-k hits against official labels and CPU ranking."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .egtea_cache import load_actions
from .official_stream_full_eval_gpu import PROTOCOL
from .official_stream_train import file_sha256


def _metrics(hits: np.ndarray, labels: np.ndarray, classes: int) -> dict:
    totals = Counter(int(label) for label in labels)
    correct = Counter(int(label) for label, hit in zip(labels, hits) if hit)
    return {"count": len(labels), "supported_classes": len(totals),
            "class_count": classes, "top5_micro": int(hits.sum()) / len(labels),
            "mean_recall_at5_present_classes": sum(correct[c] / n for c, n in totals.items()) / len(totals)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    gpu_path = args.input_dir / "split1_selected_full_heads_gpu_topk.json"
    gpu = json.loads(gpu_path.read_text())
    if gpu["protocol"] != PROTOCOL or gpu["kind"] != "saved_logit_gpu_vector_topk":
        raise ValueError("GPU top-k protocol mismatch")
    hit_path = args.input_dir / "split1_selected_full_heads_gpu_topk.npz"
    if gpu["hit_archive"] != str(hit_path) or gpu["hit_archive_sha256"] != file_sha256(hit_path):
        raise ValueError("GPU hit archive identity mismatch")
    with np.load(hit_path, allow_pickle=False) as archive:
        gpu_bits = {key: archive[key] for key in archive.files}
    mapping = load_actions()
    verb_map, noun_map = torch.zeros(106, 19), torch.zeros(106, 53)
    for action, (verb, noun) in enumerate(mapping):
        verb_map[action, verb] = 1
        noun_map[action, noun] = 1
    result = {"protocol": PROTOCOL, "kind": "verified_selected_checkpoint_full_heads",
              "gpu_topk_report": str(gpu_path), "horizons": {}}
    for horizon in (2, 6):
        path = args.input_dir / f"s1_h{horizon}_selected_full_heads.npz"
        metadata = json.loads((args.input_dir / f"s1_h{horizon}_selected_full_heads.json").read_text())
        recorded = gpu["horizons"][str(horizon)]
        if (metadata["protocol"] != PROTOCOL
                or recorded["source_archive"] != str(path)
                or recorded["source_archive_sha256"] != file_sha256(path)
                or metadata["selected_epoch"] != recorded["selected_epoch"]):
            raise ValueError(f"Source archive identity changed for h{horizon}")
        with np.load(path, allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        action = torch.from_numpy(saved["action_logits"])
        heads = {
            "action": (action, saved["action"]),
            "verb_direct": (torch.from_numpy(saved["verb_logits"]), saved["verb"]),
            "noun_direct": (torch.from_numpy(saved["noun_logits"]), saved["noun"]),
            "verb_from_action_source": (action @ verb_map, saved["verb"]),
            "noun_from_action_source": (action @ noun_map, saved["noun"]),
        }
        checked = {}
        for name, (logits, labels) in heads.items():
            hits = gpu_bits[f"h{horizon}_{name}_hits"]
            ties = gpu_bits[f"h{horizon}_{name}_ties"]
            if (hits.shape != labels.shape or ties.shape != labels.shape
                    or not np.isin(hits, [0, 1]).all()
                    or not np.isin(ties, [0, 1]).all()):
                raise ValueError(f"Invalid GPU hit/tie bits for h{horizon}/{name}")
            recomputed = _metrics(hits, labels, logits.shape[1])
            claimed = recorded["scores"][name]
            for field, value in recomputed.items():
                if not math.isclose(value, claimed[field], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"GPU hit bits do not reproduce h{horizon}/{name}/{field}")
            if int(ties.sum()) != claimed["boundary_tie_rows"]:
                raise ValueError(f"GPU boundary tie count differs for h{horizon}/{name}")
            cpu_hits = ((logits.topk(5, dim=1).indices
                         == torch.from_numpy(labels.astype(np.int64))[:, None])
                        .any(dim=1).numpy().astype(np.uint8))
            differs = np.flatnonzero(cpu_hits != hits)
            outside_ties = differs[ties[differs] == 0]
            if len(outside_ties):
                raise ValueError(f"CPU/GPU h{horizon}/{name} differs outside rank-five ties: "
                                 f"{outside_ties[:10].tolist()}")
            checked[name] = {**claimed, "cpu_gpu_hit_differences": len(differs),
                             "all_cpu_gpu_differences_on_boundary_ties": True}
        action_score = checked["action"]
        selected = recorded["selected_action_recall5_training_time"]
        if horizon == 6 and action_score["mean_recall_at5_present_classes"] != selected:
            raise ValueError("+6s Action replay no longer matches its selected score")
        if horizon == 2:
            saved_top5 = 0.4918768704574605
            hit_change = round((action_score["top5_micro"] - saved_top5) * len(saved["action"]))
            if hit_change != 1:
                raise ValueError(f"Unexpected +2s Action replay hit change: {hit_change}")
        result["horizons"][str(horizon)] = {
            "selected_epoch": recorded["selected_epoch"],
            "selected_action_recall5_training_time": selected,
            "scored_ticks": len(saved["action"]),
            "metrics_gpu_source_topk": checked,
        }
    output = args.input_dir / "split1_selected_full_heads_verified.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "cpu_gpu_hit_differences": {
        h: {name: metric["cpu_gpu_hit_differences"] for name, metric in value["metrics_gpu_source_topk"].items()}
        for h, value in result["horizons"].items()}}), flush=True)


if __name__ == "__main__":
    main()
