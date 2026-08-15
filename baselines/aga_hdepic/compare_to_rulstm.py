#!/usr/bin/env python3
"""Compare Streaming AGA vs original RU-LSTM on HD-EPIC P01 (+2/+4/+6s)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    if x is None:
        return None
    return float(x) * 100.0 if float(x) <= 1.5 else float(x)


def best_from_history(history: list) -> dict | None:
    best = None
    best_score = -1.0
    for row in history:
        va = row.get("val") or {}
        score = float(va.get("action_top5@2s", -1))
        if score > best_score:
            best_score = score
            best = va
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aga-dir", type=Path, required=True)
    ap.add_argument(
        "--rulstm-metrics",
        type=Path,
        default=Path("/scratch/ll5914/experiments/rulstm_hdepic_p01_stream/val_metrics.json"),
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.aga_dir / "compare_vs_rulstm.json")

    rulstm = json.loads(args.rulstm_metrics.read_text()) if args.rulstm_metrics.is_file() else {}
    aga = None
    hist_path = args.aga_dir / "history.json"
    val_path = args.aga_dir / "val_metrics.json"
    if hist_path.is_file():
        history = json.loads(hist_path.read_text())
        aga = best_from_history(history) if history else None
    if aga is None and val_path.is_file():
        aga = json.loads(val_path.read_text())

    param_path = args.aga_dir / "param_count.json"
    params = json.loads(param_path.read_text()) if param_path.is_file() else {}

    table = []
    for h in (2.0, 4.0, 6.0):
        row = {"horizon_s": h}
        for name in ("verb", "noun", "action"):
            for k in (1, 5):
                rk = f"{name}_top{k}@{int(h)}s"
                row[f"rulstm_{name}_top{k}"] = rulstm.get(rk, rulstm.get(f"{name}_top{k}@{h:g}s"))
                jk = f"{name}_top{k}@{h:g}s"
                row[f"aga_{name}_top{k}"] = pct(aga.get(jk)) if aga and jk in aga else None
        table.append(row)

    report = {
        "rulstm": {"metrics_path": str(args.rulstm_metrics)},
        "aga": {
            "dir": str(args.aga_dir),
            "params_m": params.get("total_m"),
            "best_action_top5@2s": aga.get("action_top5@2s") if aga else None,
        },
        "table": table,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
