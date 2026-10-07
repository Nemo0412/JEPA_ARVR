"""One complete cached fixed-window epoch for walltime budgeting only."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .egtea_cache import MAIN_ROOT
from .official_instance_budget import make_model, masked_mixup_loss
from .official_instance_cache import BASE, ids_hash
from .official_instance_data import build_instance_rows
from .official_stream_train import source_identity


OUTPUT = MAIN_ROOT / "outputs/b19_scalant/implementation/official_instance32s_full_epoch_budget.json"
BATCH = 128


def labels(rows):
    return {name: torch.tensor(values, dtype=torch.long)
            for name, values in {
                "action": [r.action for r in rows],
                "verb": [r.verb for r in rows],
                "noun": [r.noun for r in rows],
                "past_action": [r.past_actions for r in rows],
                "past_verb": [r.past_verbs for r in rows],
                "past_noun": [r.past_nouns for r in rows],
            }.items()}


def main():
    horizon = 2
    cache = BASE / f"h{horizon}"
    manifest = json.loads((cache / "manifest.json").read_text())
    if (manifest["kind"] != "fixed32s_instance_feature_cache_for_budget_smoke"
            or manifest["horizon_seconds"] != horizon
            or manifest["source_identity"] != source_identity()):
        raise ValueError("Feature cache identity mismatch")
    rows = {}
    arrays = {}
    targets = {}
    for population, expected in (("train", 8299), ("val", 2022)):
        items, early = build_instance_rows(population, horizon)
        record = manifest["population"][population]
        if early or len(items) != expected or record["uids_sha256"] != ids_hash(items):
            raise ValueError(f"Cache/official row mismatch for {population}")
        array = np.load(record["path"], mmap_mode="r")
        if array.shape != (expected, 128, 1024) or array.dtype != np.float32:
            raise ValueError(f"Cache shape/dtype mismatch for {population}")
        rows[population] = items
        arrays[population] = array
        targets[population] = labels(items)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    model, optimizer = make_model()
    model.train()
    permutation = torch.randperm(len(rows["train"]),
                                 generator=torch.Generator().manual_seed(42)).tolist()
    torch.cuda.synchronize()
    start = time.perf_counter()
    completed = 0
    for begin in range(0, len(permutation), BATCH):
        indices = permutation[begin:begin + BATCH]
        features = torch.from_numpy(np.asarray(arrays["train"][indices])).cuda()
        target = {name: value[indices].cuda() for name, value in targets["train"].items()}
        lam = 0.6 if len(indices) > 1 else 1.0
        mixed = features * lam + features.flip(0) * (1 - lam)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(mixed)
            loss = masked_mixup_loss(output, target, lam, 0.1)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss after {completed} instances")
        loss.backward()
        optimizer.step()
        completed += len(indices)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    if completed != 8299:
        raise ValueError("Incomplete official training population")

    model.eval()
    seen = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(rows["val"]), BATCH):
            stop = min(begin + BATCH, len(rows["val"]))
            features = torch.from_numpy(np.array(arrays["val"][begin:stop], copy=True)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(features)
            if not torch.isfinite(output.future_actions).all():
                raise ValueError("Nonfinite validation logits")
            seen += stop - begin
    torch.cuda.synchronize()
    val_seconds = time.perf_counter() - start
    if seen != 2022:
        raise ValueError("Incomplete official validation population")

    temporary_ckpt = OUTPUT.with_suffix(".diagnostic_checkpoint.pt")
    start = time.perf_counter()
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, temporary_ckpt)
    checkpoint_seconds = time.perf_counter() - start
    checkpoint_bytes = temporary_ckpt.stat().st_size
    temporary_ckpt.unlink()
    report = {"kind": "fixed32s_instance_full_epoch_budget_not_accuracy",
              "gpu": torch.cuda.get_device_name(),
              "train_instances": completed, "val_instances": seen,
              "batch": BATCH, "optimizer_steps": (completed + BATCH - 1) // BATCH,
              "train_seconds": train_seconds, "validation_seconds": val_seconds,
              "checkpoint_write_seconds": checkpoint_seconds,
              "checkpoint_bytes": checkpoint_bytes,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "cache_manifest": str(cache / "manifest.json")}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(OUTPUT) + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
