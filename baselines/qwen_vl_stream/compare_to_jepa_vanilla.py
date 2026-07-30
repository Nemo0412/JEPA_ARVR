#!/usr/bin/env python3
"""Compare Qwen stream probe vs V-JEPA video-only vanilla (p01_stream_mtp_2_4_6)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    if x is None:
        return None
    x = float(x)
    return x * 100.0 if x <= 1.5 else x


def best_val(history: list) -> dict | None:
    best, score = None, -1.0
    for row in history:
        va = row.get("val") or {}
        s = float(va.get("primary_action_top5", va.get("action_top5@2s", -1)))
        if s > score:
            score, best = s, va
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-dir", type=Path, required=True)
    ap.add_argument(
        "--jepa-history",
        type=Path,
        default=Path("/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/history.json"),
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.qwen_dir / "compare_vs_jepa_vanilla.json")

    jepa_h = json.loads(args.jepa_history.read_text())
    jepa = best_val(jepa_h) or {}
    qwen = None
    hist = args.qwen_dir / "history.json"
    if hist.is_file():
        qwen = best_val(json.loads(hist.read_text()))
    if qwen is None and (args.qwen_dir / "val_only_metrics.json").is_file():
        qwen = json.loads((args.qwen_dir / "val_only_metrics.json").read_text())

    params = {}
    if (args.qwen_dir / "param_count.json").is_file():
        params = json.loads((args.qwen_dir / "param_count.json").read_text())

    table = []
    for h in (2.0, 4.0, 6.0):
        row = {"horizon_s": h}
        for name in ("verb", "noun", "action"):
            for k in (1, 5):
                key = f"{name}_top{k}@{h:g}s"
                # JEPA vanilla history may only have action_top5
                row[f"jepa_{name}_top{k}"] = pct(jepa.get(key))
                row[f"qwen_{name}_top{k}"] = pct(qwen.get(key)) if qwen else None
        table.append(row)

    report = {
        "jepa_vanilla": {
            "path": str(args.jepa_history),
            "note": "ViT-L frozen + stream MTP; ~400M total / ~89M trainable",
        },
        "qwen2vl2b": {
            "params": params,
            "dir": str(args.qwen_dir),
            "note": "Official smallest Qwen-VL (~2B); still ~5x JEPA total params",
        },
        "table": table,
    }
    out.write_text(json.dumps(report, indent=2))

    print("=" * 72)
    print("V-JEPA video-only vanilla vs Qwen2-VL-2B stream probe")
    print("=" * 72)
    print(f"{'H':>4}  {'JEPA act@5':>12} {'Qwen act@5':>12}  {'JEPA act@1':>12} {'Qwen act@1':>12}")
    for row in table:
        def fmt(x):
            return f"{x:12.2f}" if x is not None else f"{'n/a':>12}"
        print(
            f"{row['horizon_s']:4.0f}  {fmt(row['jepa_action_top5'])} {fmt(row['qwen_action_top5'])}  "
            f"{fmt(row['jepa_action_top1'])} {fmt(row['qwen_action_top1'])}"
        )
    if params:
        print(f"\nQwen total={params.get('total_m'):.1f}M trainable={params.get('trainable_m'):.2f}M")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
