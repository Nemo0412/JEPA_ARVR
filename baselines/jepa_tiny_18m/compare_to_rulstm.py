#!/usr/bin/env python3
"""Compare tiny≈18M V-JEPA stream metrics vs original RU-LSTM (~18M)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    if x is None:
        return None
    # JEPA history stores fractions; RU-LSTM stores percent already for some runs
    return float(x) * 100.0 if float(x) <= 1.5 else float(x)


def best_from_history(history: list) -> dict | None:
    best = None
    best_score = -1.0
    for row in history:
        va = row.get("val") or {}
        score = float(va.get("primary_action_top5", va.get("action_top5@2s", -1)))
        if score > best_score:
            best_score = score
            best = va
    return best


def pick(metrics: dict, name: str, h: float):
    # try JEPA fraction keys and RU-LSTM percent keys
    for k in (f"{name}_top1@{h:g}s", f"{name}_top1@{int(h)}s"):
        if k in metrics:
            return pct(metrics[k])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa-dir", type=Path, required=True)
    ap.add_argument(
        "--rulstm-metrics",
        type=Path,
        default=Path("/scratch/ll5914/experiments/rulstm_hdepic_p01_stream/val_metrics.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = ap.parse_args()
    out = args.out or (args.jepa_dir / "compare_vs_rulstm.json")

    rulstm = json.loads(args.rulstm_metrics.read_text())
    jepa = None
    hist_path = args.jepa_dir / "history.json"
    val_only = args.jepa_dir / "val_only_metrics.json"
    if hist_path.is_file():
        history = json.loads(hist_path.read_text())
        jepa = best_from_history(history) if history else None
    if jepa is None and val_only.is_file():
        jepa = json.loads(val_only.read_text())

    param_path = args.jepa_dir / "param_count.json"
    params = json.loads(param_path.read_text()) if param_path.is_file() else {}

    table = []
    for h in (2.0, 4.0, 6.0):
        row = {"horizon_s": h}
        for name in ("verb", "noun", "action"):
            for k in (1, 5):
                rk = f"{name}_top{k}@{int(h)}s"
                row[f"rulstm_{name}_top{k}"] = rulstm.get(rk)
                # JEPA
                jk = f"{name}_top{k}@{h:g}s"
                row[f"jepa_{name}_top{k}"] = pct(jepa.get(jk)) if jepa and jk in jepa else None
        table.append(row)

    report = {
        "rulstm": {
            "params_m": 18.1,
            "metrics_path": str(args.rulstm_metrics),
        },
        "jepa_tiny18m": {
            "params": params,
            "metrics_source": str(hist_path if hist_path.is_file() else val_only),
        },
        "table": table,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("RU-LSTM original (~18M) vs V-JEPA tiny≈18M  | action Top-1 / Top-5 (%)")
    print("=" * 72)
    print(f"{'H':>4}  {'RU act@1':>10} {'JEPA act@1':>10}  {'RU act@5':>10} {'JEPA act@5':>10}")
    for row in table:
        h = row["horizon_s"]
        def fmt(x):
            return f"{x:10.2f}" if x is not None else f"{'n/a':>10}"
        print(
            f"{h:4.0f}  {fmt(row['rulstm_action_top1'])} {fmt(row['jepa_action_top1'])}  "
            f"{fmt(row['rulstm_action_top5'])} {fmt(row['jepa_action_top5'])}"
        )
    if params:
        print(f"\nJEPA params: total={params.get('total_m'):.1f}M trainable={params.get('trainable_m'):.1f}M")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
