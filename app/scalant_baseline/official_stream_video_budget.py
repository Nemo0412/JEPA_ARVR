"""One-video real-feature streaming compute budget, with no saved model or metric.

Runs the longest official training prefix so feature IO, bfloat16 autocast,
masked six-head backward, full-prefix recomputation and long-history memory
all occur. Optimizer steps are diagnostic; their weights are discarded.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter

import torch

from .egtea_cache import MAIN_ROOT, ROOT
from .legacy_autograd_fix import install_clone_per_layer_fix
from .official_stream_data import build_official_split1_ticks
from .stream_data import StreamingTSNReader
from .stream_model import FullPrefixStreamingScalAnt, masked_stream_loss


OUTPUT = MAIN_ROOT / "outputs/b19_scalant/implementation/official_split1_longest_video_budget.json"


def main():
    torch.manual_seed(42)
    upstream = ROOT / "refer_repo/scalable-anticipation"
    sys.path.insert(0, str(upstream))
    from scalant.config import load_config
    from scalant.models import QueryPredictor

    install_clone_per_layer_fix()
    ticks = build_official_split1_ticks("train")
    counts = Counter(t.video for t in ticks)
    video, expected_ticks = max(counts.items(), key=lambda item: (item[1], item[0]))
    selected = [t for t in ticks if t.video == video]
    cfg = load_config(str(upstream / "configs/ek100/default.yaml"))
    cfg.DATA.TAU_O = 4.0
    cfg.DATA.LONG_MEMORY_LENGTH = 0.0
    cfg.MODEL.INPUT_DIM = 1024
    cfg.MODEL.IGNORE_INDEX = -1
    cfg.MODEL.N_QUERIES = 1
    model = QueryPredictor(cfg, {"action": 106, "verb": 19, "noun": 53}, None).cuda().train()
    stream = FullPrefixStreamingScalAnt(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    reader = StreamingTSNReader()
    processed = backward = future = past = 0
    read_seconds = 0.0
    train_seconds = 0.0
    checkpoints = []
    started = time.monotonic()
    try:
        for tick in selected:
            read_started = time.monotonic()
            chunk = reader.read(tick).unsqueeze(0).cuda()
            read_seconds += time.monotonic() - read_started
            compute_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = stream.process(tick.video, chunk)
                loss = masked_stream_loss(output, tick, horizon=2)
            if loss.value is not None:
                if not torch.isfinite(loss.value):
                    raise ValueError(f"Nonfinite masked loss at {tick.video}/{tick.tick_frame}")
                loss.value.backward()
                optimizer.step()
                backward += 1
                future += loss.valid_future
                past += loss.valid_past
            processed += 1
            if processed % 100 == 0 or processed == expected_ticks:
                torch.cuda.synchronize()
                train_seconds += time.monotonic() - compute_started
                checkpoints.append({"ticks": processed,
                                    "history_tokens": stream.history.shape[1],
                                    "elapsed_seconds": time.monotonic() - started,
                                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
                print(json.dumps(checkpoints[-1]), flush=True)
            else:
                train_seconds += time.monotonic() - compute_started
    finally:
        reader.close()
    report = {
        "kind": "official_split1_real_video_budget_not_training_or_accuracy",
        "video": video,
        "ticks": processed,
        "history_tokens": stream.history.shape[1],
        "horizon_seconds": 2,
        "autocast": "bfloat16",
        "diagnostic_optimizer_step_per_nonempty_loss": True,
        "backward_ticks": backward,
        "future_labels": future,
        "known_past_positions": past,
        "read_seconds_unsynchronized": read_seconds,
        "compute_seconds_with_periodic_sync": train_seconds,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "checkpoints": checkpoints,
    }
    if processed != expected_ticks or stream.history.shape[1] != 16 * expected_ticks:
        raise ValueError("Incomplete full video or history growth")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
