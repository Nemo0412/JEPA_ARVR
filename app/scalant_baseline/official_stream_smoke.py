"""CPU index/feature smoke against the independent official-source audit."""

from __future__ import annotations

import json

import torch

from .egtea_cache import MAIN_ROOT
from .official_stream_audit import OUTPUT as AUDIT
from .official_stream_data import build_official_split1_ticks
from .stream_data import StreamingTSNReader
from .stream_eval import ActionTickScorer


OUTPUT = MAIN_ROOT / "outputs/b19_scalant/implementation/official_split1_stream_smoke.json"


def main():
    audit = json.loads(AUDIT.read_text())
    if audit["kind"] != "official_split1_stream_source_audit_not_training":
        raise ValueError("Unrecognized official source audit")
    report = {"kind": "official_split1_stream_adapter_smoke_not_training",
              "population": {}}
    reader = StreamingTSNReader()
    try:
        for population in ("train", "val"):
            ticks = build_official_split1_ticks(population)
            observed = audit["population"][population]
            if len(ticks) != observed["processed_ticks"]:
                raise ValueError(f"{population} tick count differs from independent audit")
            if observed.get("missing_feature_queries") or observed.get("bad_feature_payloads"):
                raise ValueError(f"{population} source features incomplete")
            scored = {}
            for horizon in (2, 6):
                scorer = ActionTickScorer(horizon)
                for tick in ticks:
                    scorer.update(tick, torch.zeros(106))
                scored[horizon] = scorer.report()["scored_tick_count"]
                if scored[horizon] != observed[f"scored_h{horizon}"]:
                    raise ValueError(f"{population} h{horizon} scorer denominator differs")
            sample_indices = sorted({0, len(ticks) // 2, len(ticks) - 1})
            for index in sample_indices:
                features = reader.read(ticks[index])
                if features.shape != (16, 1024) or not torch.isfinite(features).all():
                    raise ValueError(f"Bad exact TSN chunk at {population} index {index}")
            report["population"][population] = {
                "processed_ticks": len(ticks),
                "scored_ticks": scored,
                "sample_feature_reads": len(sample_indices),
                "videos": len({t.video for t in ticks}),
                "ticks_without_either_future_label": sum(
                    all(a < 0 for a in t.target_actions) for t in ticks),
                "ticks_with_known_past": sum(any(a >= 0 for a in t.past_actions) for t in ticks),
            }
    finally:
        reader.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
