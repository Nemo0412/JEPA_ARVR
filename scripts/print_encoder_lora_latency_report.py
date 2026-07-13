#!/usr/bin/env python3
"""Pretty-print encoder-LoRA fp32 latency breakdown JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to latency JSON from EVAL_LATENCY_REPORT")
    parser.add_argument("--use-wall", action="store_true", help="Use wall_ms_avg instead of cuda_ms_avg")
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    sections = payload.get("sections", {})
    key = "wall_ms_avg" if args.use_wall else "cuda_ms_avg"
    total = sum(row.get(key, 0.0) for row in sections.values()) or 1.0

    print(f"Report: {args.report}")
    print(f"iter_wall_ms_avg: {payload.get('iter_wall_ms_avg', 0.0):.1f}")
    print(f"{'section':24s}  {'ms_avg':>10s}  {'pct':>6s}")
    for name, row in sections.items():
        ms = row.get(key, 0.0)
        pct = 100.0 * ms / total
        print(f"{name:24s}  {ms:10.1f}  {pct:5.1f}%")
    print(f"{'TOTAL (sections)':24s}  {total:10.1f}  {'100.0':>5s}%")


if __name__ == "__main__":
    main()
