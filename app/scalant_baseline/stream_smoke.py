"""CPU smoke of the split-1 stream index, source reader and action scorer."""

from __future__ import annotations

import json

import torch

from .egtea_cache import MAIN_ROOT
from .stream_data import StreamingTSNReader, build_split1_ticks
from .stream_eval import ActionTickScorer


def main() -> None:
    reader = StreamingTSNReader()
    report = {"kind": "stream_adapter_cpu_smoke_not_training", "populations": {}}
    try:
        for population in ("train", "val"):
            ticks = build_split1_ticks(population)
            expected = 12650 if population == "train" else 11390
            if len(ticks) != expected:
                raise ValueError(f"Unexpected {population} processing grid: {len(ticks)}")
            if len({tick.video for tick in ticks}) != 86:
                raise ValueError("Split-1 streaming session coverage changed")
            sample_indices = (0, len(ticks) // 2, len(ticks) - 1)
            for index in sample_indices:
                values = reader.read(ticks[index])
                if values.shape != (16, 1024) or not torch.isfinite(values).all():
                    raise ValueError(f"Bad feature chunk at {population} {index}")
            horizon_counts = {}
            for horizon in (2, 6):
                scorer = ActionTickScorer(horizon)
                for tick in ticks:
                    scorer.update(tick, torch.zeros(106))
                horizon_counts[horizon] = scorer.report()["scored_tick_count"]
            expected_pairs = 25300 if population == "train" else 22694
            if sum(horizon_counts.values()) != expected_pairs:
                raise ValueError(f"Scored {population} horizon denominator changed")
            report["populations"][population] = {
                "processing_ticks": len(ticks),
                "terminal_unlabelled_ticks_skipped": (12650 if population == "train" else 12522) - len(ticks),
                "masked_ticks_by_horizon": {h: len(ticks) - n for h, n in horizon_counts.items()},
                "scored_ticks_by_horizon": horizon_counts,
                "feature_samples_read": len(sample_indices),
            }
    finally:
        reader.close()
    path = MAIN_ROOT / "outputs/b19_scalant/implementation/split1_stream_adapter_smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
