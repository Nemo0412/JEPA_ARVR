"""Read audited EGTEA RGB features without changing the ScalAnt submodule.

This module supplies an action-instance input contract. It does not choose a
training loss or claim that RU-LSTM's TSN producer matches TeSTra's EK100 one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
# The shared baseline is a standalone checkout; data/output assets can live
# elsewhere. Keep path configuration separate from label/tick semantics.
MAIN_ROOT = Path(os.environ.get("SCALANT_WORKSPACE_ROOT", str(ROOT))).expanduser().resolve()
ANNOTATIONS = Path(os.environ.get(
    "SCALANT_ANNOTATIONS", str(ROOT / "refer_repo/rulstm/RULSTM/data/egtea")
)).expanduser().resolve()
CACHE_ROOT = Path(os.environ.get(
    "SCALANT_INSTANCE_CACHE", str(MAIN_ROOT / "outputs/b19_rulstm/cache")
)).expanduser().resolve()
EXPECTED_ROWS = {1: (8299, 2022), 2: (8299, 2022), 3: (8300, 2021)}
OBSERVED_STEPS = 16
FEATURE_DIM = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_actions(path: Path = ANNOTATIONS) -> tuple[tuple[int, int], ...]:
    mapping = []
    with (path / "actions.csv").open(newline="") as stream:
        for row in csv.reader(stream):
            action, pair = int(row[0]), row[1].strip()
            if action != len(mapping):
                raise ValueError(f"Non-contiguous action ID {action}")
            verb, noun = map(int, pair.split("_"))
            mapping.append((verb, noun))
    if len(mapping) != 106:
        raise ValueError(f"Expected 106 actions, got {len(mapping)}")
    return tuple(mapping)


class EGTEAInstanceCache(Dataset):
    """One official action annotation and 16 causal 4 Hz RGB features per row.

    The source cache has additional future features for RU-LSTM sequence
    completion; this reader deliberately exposes only its first 16 positions.
    """

    def __init__(self, split: int, horizon: int, population: str,
                 cache_root: Path = CACHE_ROOT, annotations: Path = ANNOTATIONS,
                 verify_file_hash: bool = False):
        if split not in EXPECTED_ROWS or horizon not in (2, 6):
            raise ValueError("Expected official split 1/2/3 and horizon 2/6")
        if population not in ("training", "validation"):
            raise ValueError("Expected training or validation")
        self.split, self.horizon, self.population = split, horizon, population
        source = Path(cache_root) / f"s{split}_h{horizon}"
        audit = json.loads((source / "audit.json").read_text())
        if not audit.get("passed") or audit["config"]["protocol"] != "egtea-rulstm/native-three-split-rgb-4s-independent-recall5-v1":
            raise ValueError("Source cache lacks a passed, recognized audit")
        expected = audit["config"]
        if (expected["split"], expected["horizon_sec"], expected["observed_steps"],
            expected["feature_dim"], expected["alpha"]) != (split, horizon, 16, 1024, 0.25):
            raise ValueError("Source cache timing or representation changed")
        csv_path = Path(annotations) / f"{population}{split}.csv"
        if sha256(csv_path) != audit["populations"][population]["csv_sha256"]:
            raise ValueError("Annotation SHA-256 differs from source audit")
        with csv_path.open(newline="") as stream:
            rows = [tuple(map(int, (row[0], row[2], row[3], row[4], row[5], row[6])))
                    for row in csv.reader(stream)]
        expected_n = EXPECTED_ROWS[split][population == "validation"]
        if len(rows) != expected_n or len({row[0] for row in rows}) != len(rows):
            raise ValueError("Official population size or IDs changed")
        mapping = load_actions(Path(annotations))
        for uid, start, end, verb, noun, action in rows:
            if not 0 <= action < 106 or mapping[action] != (verb, noun) or end <= start:
                raise ValueError(f"Bad label or interval for UID {uid}")
        file_path = source / f"{population}.pt"
        if verify_file_hash and sha256(file_path) != audit["populations"][population]["cache_sha256"]:
            raise ValueError("Cached feature file SHA-256 differs from source audit")
        cache = torch.load(file_path, map_location="cpu", weights_only=True)
        features, labels, ids = cache["features"], cache["labels"], cache["ids"]
        if features.shape != (expected_n, expected["sequence_length"], FEATURE_DIM):
            raise ValueError(f"Unexpected feature shape {tuple(features.shape)}")
        if labels.tolist() != [row[5] for row in rows] or ids.tolist() != [row[0] for row in rows]:
            raise ValueError("Cached ID/label order differs from official CSV")
        if ids.tolist() != audit["populations"][population]["ids"]:
            raise ValueError("Cached ID order differs from source audit")
        if not torch.isfinite(features[:, :OBSERVED_STEPS]).all():
            raise ValueError("Nonfinite observed feature")
        self.features = features[:, :OBSERVED_STEPS].contiguous()
        self.labels, self.ids, self.rows = labels, ids, rows
        self.action_to_verb_noun = mapping

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        action = int(self.labels[index])
        verb, noun = self.action_to_verb_noun[action]
        return {"past_feats": self.features[index], "action": self.labels[index],
                "verb": torch.tensor(verb), "noun": torch.tensor(noun),
                "uid": self.ids[index]}
