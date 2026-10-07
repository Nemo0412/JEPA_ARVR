"""Measure full-prefix ScalAnt training-step cost at representative stream ages.

Synthetic TSN-shaped input isolates model compute. This is a budget diagnostic,
not an accuracy run and not a claim of efficient incremental inference.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .egtea_cache import MAIN_ROOT, ROOT
from .legacy_autograd_fix import install_clone_per_layer_fix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="16,160,800,1600")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-seconds", type=float, default=0.0,
                        help="Run each length long enough to sample steady GPU utilization")
    parser.add_argument("--autocast-bfloat16", action="store_true",
                        help="Exercise the released EK100 config's bfloat16 compute")
    parser.add_argument("--output", type=Path, default=MAIN_ROOT /
                        "outputs/b19_scalant/implementation/stream_full_prefix_profile.json")
    args = parser.parse_args()
    torch.manual_seed(1979)
    upstream = ROOT / "refer_repo/scalable-anticipation"
    sys.path.insert(0, str(upstream))
    from scalant.config import load_config
    from scalant.models import QueryPredictor
    install_clone_per_layer_fix()
    cfg = load_config(str(upstream / "configs/ek100/default.yaml"))
    cfg.DATA.TAU_O = 4.0
    cfg.DATA.LONG_MEMORY_LENGTH = 0.0
    cfg.MODEL.INPUT_DIM = 1024
    cfg.MODEL.IGNORE_INDEX = -1
    cfg.MODEL.N_QUERIES = 1
    model = QueryPredictor(cfg, {"action": 106, "verb": 19, "noun": 53}, None)
    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    report = {"kind": "synthetic_full_prefix_training_step_profile_not_accuracy",
              "source_config": str(upstream / "configs/ek100/default.yaml"),
              "autocast_bfloat16": args.autocast_bfloat16,
              "work_tokens": 16, "feature_dim": 1024,
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "repeats": args.repeats, "lengths": []}
    for length in [int(s) for s in args.lengths.split(",")]:
        if length < 16 or length % 16:
            raise ValueError("Lengths must be whole 4s chunks")
        model.long_mem_len = length - 16
        features = torch.randn(1, length, 1024, device="cuda")
        times = []
        torch.cuda.reset_peak_memory_stats()
        length_started = time.monotonic()
        while len(times) < args.repeats or time.monotonic() - length_started < args.minimum_seconds:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.autocast_bfloat16):
                output = model(features)
            # Six source heads, with the past supervised on the current 4s
            # chunk.  The target labels are arbitrary, so this is compute only.
            losses = [F.cross_entropy(output.future_actions[:, -1],
                                      torch.tensor([0], device="cuda")),
                      F.cross_entropy(output.future_verbs[:, -1],
                                      torch.tensor([0], device="cuda")),
                      F.cross_entropy(output.future_nouns[:, -1],
                                      torch.tensor([0], device="cuda")),
                      F.cross_entropy(output.past_actions[:, -16:].reshape(-1, 106),
                                      torch.zeros(16, dtype=torch.long, device="cuda")),
                      F.cross_entropy(output.past_verbs[:, -16:].reshape(-1, 19),
                                      torch.zeros(16, dtype=torch.long, device="cuda")),
                      F.cross_entropy(output.past_nouns[:, -16:].reshape(-1, 53),
                                      torch.zeros(16, dtype=torch.long, device="cuda"))]
            sum(losses).backward()
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.monotonic() - started)
        steady = times[1:]
        record = {"tokens": length, "steps": len(times),
                  "steady_step_mean_seconds": statistics.mean(steady) if steady else None,
                  "steady_step_median_seconds": statistics.median(steady) if steady else None,
                  "seconds_after_warmup_first10": steady[:10],
                  "first_step_seconds_including_compile": times[0],
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        report["lengths"].append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
