#!/usr/bin/env python
"""Parse keep-nonfinite baseline Slurm log into structured grad/loss forensics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GRAD_SNAPSHOT_RE = re.compile(
    r"\[grad-snapshot\] itr=(\d+) reason=(\S+) use_bfloat16=(\w+)"
)
GRAD_LINE_RE = re.compile(
    r"^\s{2}(head\[0\]|encoder_lora|tokens\.grad|input_adapter|opt\[[^\]]+\]):\s*(.+)$"
)
FIRST_BAD_RE = re.compile(r"encoder_lora\.first_bad=(\S+)\s+bad_elems=(\S+)")
LOSS_RE = re.compile(
    r"\[\s*(\d+)\] loss=([\d.]+).*enc_lora_ok=(\w+)"
)
KEEP_RE = re.compile(r"Keeping non-finite encoder-LoRA grads at itr=(\d+)")
CLIP_NAN_RE = re.compile(r"Proceeding at itr=(\d+) despite non-finite clipped grad norm")
NAN_ENCODER_RE = re.compile(r"Nan detected at output of encoder")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    snapshots: list[dict] = []
    losses: list[dict] = []
    keep_iters: list[int] = []
    clip_nan_iters: list[int] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        m = GRAD_SNAPSHOT_RE.search(line)
        if m:
            snap = {
                "itr": int(m.group(1)),
                "reason": m.group(2),
                "use_bfloat16": m.group(3),
                "lines": {},
                "first_bad": None,
                "bad_elems": None,
            }
            i += 1
            while i < len(lines):
                sub = lines[i]
                if GRAD_SNAPSHOT_RE.search(sub) or sub.startswith("[INFO") or sub.startswith("[WARNING"):
                    if not sub.startswith("  "):
                        break
                gm = GRAD_LINE_RE.match(sub)
                if gm:
                    snap["lines"][gm.group(1)] = gm.group(2).strip()
                fb = FIRST_BAD_RE.search(sub)
                if fb:
                    snap["first_bad"] = fb.group(1)
                    snap["bad_elems"] = fb.group(2)
                i += 1
            snapshots.append(snap)
            continue

        lm = LOSS_RE.search(line)
        if lm:
            losses.append(
                {
                    "itr": int(lm.group(1)),
                    "loss": float(lm.group(2)),
                    "enc_lora_ok": lm.group(3) == "True",
                }
            )

        km = KEEP_RE.search(line)
        if km:
            keep_iters.append(int(km.group(1)))

        cm = CLIP_NAN_RE.search(line)
        if cm:
            clip_nan_iters.append(int(cm.group(1)))

        i += 1

    nan_line = next((idx + 1 for idx, ln in enumerate(lines) if NAN_ENCODER_RE.search(ln)), None)

    def enc_status(s: dict) -> str:
        raw = s["lines"].get("encoder_lora", "")
        if raw.startswith("nonfinite"):
            return "nonfinite"
        if raw.startswith("0.0000e+00"):
            return "zero"
        return "finite"

    timeline = []
    for s in snapshots:
        timeline.append(
            {
                "itr": s["itr"],
                "head": s["lines"].get("head[0]"),
                "encoder_lora": s["lines"].get("encoder_lora"),
                "encoder_status": enc_status(s),
                "first_bad": s["first_bad"],
            }
        )

    # Phase detection
    phases = []
    for s in snapshots:
        itr = s["itr"]
        st = enc_status(s)
        if itr == 0:
            phases.append({"itr": itr, "phase": "init_explosion", "encoder_lora": s["lines"].get("encoder_lora")})
        elif itr == 100 and st == "finite":
            phases.append({"itr": itr, "phase": "recovered_finite", "encoder_lora": s["lines"].get("encoder_lora")})
        elif itr == 340 and st == "nonfinite":
            phases.append({"itr": itr, "phase": "nonfinite_grad_returns", "encoder_lora": s["lines"].get("encoder_lora")})
        elif itr == 380 and st == "nonfinite":
            phases.append({"itr": itr, "phase": "pre_crash_nonfinite_grad", "encoder_lora": s["lines"].get("encoder_lora")})

    focus = [t for t in timeline if t["itr"] in {0, 10, 30, 60, 100, 150, 200, 300, 310, 330, 340, 350, 360, 370, 380}]

    return {
        "log_path": str(path),
        "nan_encoder_line": nan_line,
        "crash_after_itr": 381,
        "num_grad_snapshots": len(snapshots),
        "num_keep_nonfinite_events": len(keep_iters),
        "num_clip_nan_proceed_events": len(clip_nan_iters),
        "keep_iters_near_crash": [k for k in keep_iters if 365 <= k <= 385],
        "clip_nan_near_crash": [k for k in clip_nan_iters if 365 <= k <= 385],
        "loss_near_crash": [l for l in losses if l["itr"] >= 360],
        "focus_snapshots": focus,
        "full_snapshot_timeline": timeline,
        "phases": phases,
        "pollution_chain": [
            "itr=0: encoder_lora grad all nonfinite (lora_A block0); head grad 7.2e7; optimizer.step() still runs (KEEP_NONFINITE)",
            "itr=1..339: intermittent nonfinite grads kept+stepped; loss falls 720->~6",
            "itr=100..200: grad snapshots show finite tiny encoder_lora norms — forward still healthy",
            "itr=340,350,380: encoder_lora grad nonfinite again; itr=380 enc_lora_ok=False",
            "itr=381: classifier receives NaN encoder tokens -> hard exit (models.py Classifier.forward)",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# keep-nonfinite grad forensics",
        "",
        f"Source: `{report['log_path']}`",
        f"Encoder NaN exit: line {report['nan_encoder_line']} (after itr {report['crash_after_itr']})",
        "",
        "## Pollution chain (from logs only)",
        "",
    ]
    for step in report["pollution_chain"]:
        lines.append(f"- {step}")
    lines.extend(["", "## Focus grad snapshots", "", "| itr | head | encoder_lora | status | first_bad |", "|-----|------|--------------|--------|-----------|"])
    for row in report["focus_snapshots"]:
        lines.append(
            f"| {row['itr']} | {row.get('head','')} | {row.get('encoder_lora','')} | {row['encoder_status']} | {row.get('first_bad') or ''} |"
        )
    lines.extend(["", "## Loss near crash", ""])
    for row in report["loss_near_crash"]:
        lines.append(f"- itr={row['itr']}: loss={row['loss']} enc_lora_ok={row['enc_lora_ok']}")
    lines.extend(
        [
            "",
            "## keep-nonfinite / clip-nan near crash",
            f"- keep_iters: {report['keep_iters_near_crash']}",
            f"- clip_nan_proceed: {report['clip_nan_near_crash']}",
            "",
            "## Note on missing tensor dumps",
            "Training did not save encoder token tensors or LoRA weights per-iter. "
            "Only `[grad-snapshot]` every 10 iters + loss lines exist. "
            "For per-forward encoder output stats, rerun short job with forward diagnostics enabled.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    report = parse_log(args.log)
    out_json = args.out_json or args.log.with_suffix(".forensics.json")
    out_md = args.out_md or args.log.with_suffix(".forensics.md")

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
