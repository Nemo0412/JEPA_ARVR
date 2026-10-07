"""Score saved selected-checkpoint logits with the original GPU topk route.

The original tick scorer calls topk on one float32 logit vector at a time.
CPU topk can choose a different class when several logits tie at rank five,
so score the saved vectors in the same device/shape pattern and retain hits.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .egtea_cache import load_actions
from .official_stream_full_eval_gpu import PROTOCOL
from .official_stream_train import file_sha256


def _score(logits: torch.Tensor, labels: np.ndarray):
    hits, ties = [], []
    for row, label in zip(logits, labels):
        # The saved split-1 scorer uses topk(5) on each single GPU vector.
        hits.append(int(int(label) in row.topk(5).indices.tolist()))
        top_six = row.topk(6)
        ties.append(int(bool(top_six.values[4] == top_six.values[5])))
    totals = Counter(int(x) for x in labels)
    correct = Counter(int(x) for x, hit in zip(labels, hits) if hit)
    metric = {"count": len(labels), "supported_classes": len(totals),
              "class_count": logits.shape[1],
              "top5_micro": sum(hits) / len(hits),
              "mean_recall_at5_present_classes": sum(correct[c] / n for c, n in totals.items()) / len(totals),
              "boundary_tie_rows": sum(ties)}
    return np.asarray(hits, dtype=np.uint8), np.asarray(ties, dtype=np.uint8), metric


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    mapping = load_actions()
    if len(mapping) != 106:
        raise ValueError("Official action mapping changed")
    verb_map = torch.zeros(106, 19, device="cuda")
    noun_map = torch.zeros(106, 53, device="cuda")
    for action, (verb, noun) in enumerate(mapping):
        verb_map[action, verb] = 1
        noun_map[action, noun] = 1
    result = {"protocol": PROTOCOL, "kind": "saved_logit_gpu_vector_topk", "horizons": {}}
    arrays = {}
    for horizon in (2, 6):
        metadata = json.loads((args.input_dir / f"s1_h{horizon}_selected_full_heads.json").read_text())
        path = args.input_dir / f"s1_h{horizon}_selected_full_heads.npz"
        if (metadata["protocol"] != PROTOCOL or metadata["horizon_seconds"] != horizon
                or metadata["archive_sha256"] != file_sha256(path)):
            raise ValueError(f"Saved logit identity mismatch for h{horizon}")
        with np.load(path, allow_pickle=False) as source:
            saved = {key: source[key] for key in source.files}
        action = torch.from_numpy(saved["action_logits"]).cuda()
        heads = {
            "action": (action, saved["action"]),
            "verb_direct": (torch.from_numpy(saved["verb_logits"]).cuda(), saved["verb"]),
            "noun_direct": (torch.from_numpy(saved["noun_logits"]).cuda(), saved["noun"]),
            "verb_from_action_source": (action @ verb_map, saved["verb"]),
            "noun_from_action_source": (action @ noun_map, saved["noun"]),
        }
        scores = {}
        for name, (logits, labels) in heads.items():
            hits, ties, metric = _score(logits, labels)
            arrays[f"h{horizon}_{name}_hits"] = hits
            arrays[f"h{horizon}_{name}_ties"] = ties
            scores[name] = metric
        result["horizons"][str(horizon)] = {
            "source_archive": str(path),
            "source_archive_sha256": metadata["archive_sha256"],
            "selected_epoch": metadata["selected_epoch"],
            "selected_action_recall5_training_time": metadata["selected_action_recall5"],
            "scores": scores,
        }
    archive = args.input_dir / "split1_selected_full_heads_gpu_topk.npz"
    temporary = archive.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, archive)
    result["hit_archive"] = str(archive)
    result["hit_archive_sha256"] = file_sha256(archive)
    output = args.input_dir / "split1_selected_full_heads_gpu_topk.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "action": {
        h: item["scores"]["action"] for h, item in result["horizons"].items()}}), flush=True)


if __name__ == "__main__":
    main()
