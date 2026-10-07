"""Real-feature fixed-32s ScalAnt batch feasibility and throughput diagnostic.

This is not a formal training run. It probes the released six-head model with
official EGTEA instances, 28s long memory, 4s working memory and a masked
mixup loss suitable for unknown past labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .egtea_cache import MAIN_ROOT, ROOT
from .legacy_autograd_fix import install_clone_per_layer_fix
from .official_instance_data import build_instance_rows
from .stream_data import StreamingTSNReader


UPSTREAM = ROOT / "refer_repo/scalable-anticipation"
SOURCE_CONFIG = UPSTREAM / "configs/ek100/default.yaml"
AUDIT = MAIN_ROOT / "outputs/b19_scalant/implementation/official_instance32s_audit.json"
OUTPUT = MAIN_ROOT / "outputs/b19_scalant/implementation/official_instance32s_budget.json"


def make_model():
    sys.path.insert(0, str(UPSTREAM))
    from scalant.config import load_config
    from scalant.models import QueryPredictor

    install_clone_per_layer_fix()
    cfg = load_config(str(SOURCE_CONFIG))
    cfg.DATA.TAU_O = 32.0
    cfg.DATA.LONG_MEMORY_LENGTH = 28.0
    cfg.DATA.TAU_A = 2.0
    cfg.MODEL.INPUT_DIM = 1024
    cfg.MODEL.IGNORE_INDEX = -1
    cfg.MODEL.N_QUERIES = 1
    model = QueryPredictor(cfg, {"action": 106, "verb": 19, "noun": 53}, None).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    return model, optimizer


def _ce(logits, labels, smoothing: float):
    valid = labels >= 0
    if not valid.any():
        return None
    return F.cross_entropy(logits[valid].float(), labels[valid],
                           label_smoothing=smoothing, reduction="sum"), valid.sum()


def masked_mixup_loss(output, labels: dict, lam: float, smoothing: float):
    """Mix known labels while keeping unknown past positions out of loss.

    Upstream MixUp blends the background one-hot bit, then its criterion only
    ignores exactly-one background targets. Sparse EGTEA unknown labels would
    therefore leak into the loss. Weighted masked CE keeps the intended
    feature/label mixture without inventing background supervision.
    """
    terms = []
    pairs = ((output.future_actions[:, -1], "action"),
             (output.future_verbs[:, -1], "verb"),
             (output.future_nouns[:, -1], "noun"),
             (output.past_actions, "past_action"),
             (output.past_verbs, "past_verb"),
             (output.past_nouns, "past_noun"))
    for logits, name in pairs:
        total = logits.new_zeros(())
        count = logits.new_zeros(())
        for weight, target in ((lam, labels[name]),
                               (1.0 - lam, labels[name].flip(0))):
            result = _ce(logits, target, smoothing)
            if result is not None:
                value, n = result
                total = total + weight * value
                count = count + weight * n
        if count.item() > 0:
            terms.append(total / count)
    if len(terms) != 6:
        raise ValueError("Diagnostic batch did not exercise all six released heads")
    return sum(terms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steady-steps", type=int, default=0)
    args = parser.parse_args()
    audit = json.loads(AUDIT.read_text())
    if audit["kind"] != "official_split1_fixed32s_instance_feasibility_not_training":
        raise ValueError("Missing recognized fixed-window source audit")
    if any(x.get("missing_queries", 0) or x.get("bad_payloads", 0)
           for x in audit["population"].values()):
        raise ValueError("Fixed-window source audit has missing or malformed vectors")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    rows, early = build_instance_rows("train", 2)
    indices = torch.randperm(len(rows), generator=torch.Generator().manual_seed(42))[:128].tolist()
    selected = [rows[i] for i in indices]
    first_known = next((i for i, row in enumerate(selected)
                        if any(label >= 0 for label in row.past_actions)), None)
    if first_known is None:
        raise ValueError("No known past action in representative 128-instance batch")
    selected[0], selected[first_known] = selected[first_known], selected[0]
    reader = StreamingTSNReader()
    start = time.perf_counter()
    try:
        x_cpu = torch.stack([reader.read(row) for row in selected])
    finally:
        reader.close()
    read_seconds = time.perf_counter() - start
    labels_cpu = {name: torch.tensor(values, dtype=torch.long)
                  for name, values in {
                      "action": [r.action for r in selected],
                      "verb": [r.verb for r in selected],
                      "noun": [r.noun for r in selected],
                      "past_action": [r.past_actions for r in selected],
                      "past_verb": [r.past_verbs for r in selected],
                      "past_noun": [r.past_nouns for r in selected],
                  }.items()}
    model, optimizer = make_model()
    report = {"kind": "fixed32s_instance_real_feature_budget_not_accuracy",
              "gpu": torch.cuda.get_device_name(), "history_tokens": 128,
              "working_tokens": 16, "source_train_instances": len(rows),
              "source_early_discarded": len(early),
              "preload_128_seconds": read_seconds, "batch": []}
    for batch in (1, 8, 16, 32, 64, 128):
        x = x_cpu[:batch].cuda()
        labels = {name: value[:batch].cuda() for name, value in labels_cpu.items()}
        lam = 0.6 if batch > 1 else 1.0
        mixed = x * lam + x.flip(0) * (1.0 - lam)
        timings = []
        try:
            model.train()
            for repetition in range(3):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(mixed)
                    loss = masked_mixup_loss(output, labels, lam, 0.1)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite diagnostic loss at batch {batch}")
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - t0)
            record = {"batch": batch, "status": "passed",
                      "seconds": timings,
                      "steady_samples_per_second": batch / (sum(timings[1:]) / 2),
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "last_loss": float(loss.detach())}
        except torch.cuda.OutOfMemoryError as error:
            record = {"batch": batch, "status": "out_of_memory",
                      "error": str(error).splitlines()[0],
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
            report["batch"].append(record)
            break
        report["batch"].append(record)
        print(json.dumps(record), flush=True)
        del x, labels, mixed, output, loss
        torch.cuda.empty_cache()
    if args.steady_steps:
        x = x_cpu.cuda()
        labels = {name: value.cuda() for name, value in labels_cpu.items()}
        lam = 0.6
        mixed = x * lam + x.flip(0) * (1.0 - lam)
        model.train()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(args.steady_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(mixed)
                loss = masked_mixup_loss(output, labels, lam, 0.1)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite sustained diagnostic loss at step {step}")
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        report["steady_loop"] = {"steps": args.steady_steps,
                                 "batch": 128, "seconds": seconds,
                                 "samples_per_second": args.steady_steps * 128 / seconds,
                                 "last_loss": float(loss.detach())}
        print(json.dumps(report["steady_loop"]), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(OUTPUT) + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({"saved": str(OUTPUT), "gpu": report["gpu"],
                      "source_train_instances": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
