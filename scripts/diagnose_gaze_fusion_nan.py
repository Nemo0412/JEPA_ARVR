#!/usr/bin/env python3
"""Diagnose NaNs in video-token-conditioned gaze RNN fusion.

This script has two cheap modes:

1. Parse pulled Slurm logs and CSV files to find the first fusion-monitor NaN,
   validation rows, stderr failures, and effective gaze learning rates.
2. Run a finite-input smoke test through ``GazeTrajectoryEncoder`` for the
   fusion variants. This does not replace a real batch diagnostic, but it tells
   us whether the module is finite at initialization and whether a high-LR
   synthetic update can corrupt it.

For exact cluster-side localization during training, launch the normal run with
``GAZE_RNN_DIAG=1``. The runtime hook in ``gaze_rnn.py`` will log the first
non-finite tensor/parameter stats and identify whether the first bad value
appears in input tokens, video projection, gate/attention, GRU output, or pooler
tokens. Add ``GAZE_RNN_DIAG_VERBOSE=1`` only for a very short run when you also
want finite range stats before the failure point.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable

try:
    import yaml
except Exception:  # pragma: no cover - cluster env may still have yaml.
    yaml = None

torch = None
GazeTrajectoryEncoder = None


FUSION_PATTERNS = {
    "gated_nearest": re.compile(r"gate=([+-]?(?:nan|inf|\d+(?:\.\d+)?))", re.IGNORECASE),
    "residual_conditioned": re.compile(r"alpha=([+-]?(?:nan|inf|\d+(?:\.\d+)?))", re.IGNORECASE),
    "local_attention": re.compile(r"attn_max=([+-]?(?:nan|inf|\d+(?:\.\d+)?))", re.IGNORECASE),
}


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def summarize_log(path: Path, pattern: re.Pattern[str]) -> dict:
    total = 0
    bad = 0
    first_bad = None
    last_line = None
    first_values = []
    last_values = []
    for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        match = pattern.search(line)
        if not match:
            continue
        total += 1
        value_text = match.group(1)
        value = parse_float(value_text)
        if len(first_values) < 3:
            first_values.append(value_text)
        last_values = (last_values + [value_text])[-3:]
        last_line = (line_no, line.strip())
        if not math.isfinite(value):
            bad += 1
            if first_bad is None:
                first_bad = (line_no, line.strip())
    return {
        "path": str(path),
        "monitor_count": total,
        "nonfinite_count": bad,
        "first_nonfinite": first_bad,
        "last_monitor": last_line,
        "first_values": first_values,
        "last_values": last_values,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_err(path: Path) -> list[str]:
    if not path.exists():
        return []
    keep = []
    needles = ("error", "traceback", "runtimeerror", "segmentation", "aborted", "killed", "exception")
    for line in path.read_text(errors="replace").splitlines():
        if any(n in line.lower() for n in needles):
            keep.append(line.strip())
    return keep


def load_effective_lrs(config_path: Path) -> list[tuple[int, float, float]]:
    if not config_path.exists():
        return []
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        cfg = yaml.safe_load(text)
        lora = cfg.get("experiment", {}).get("optimization", {}).get("multihead_kwargs", [])
        gaze_lr_mult = (
            cfg.get("experiment", {})
            .get("lora", {})
            .get("gaze", {})
            .get("rnn", {})
            .get("gaze_lr_mult", 1.0)
        )
        out = []
        for idx, item in enumerate(lora):
            base_lr = float(item.get("start_lr", item.get("lr", 0.0)) or 0.0)
            out.append((idx, base_lr, base_lr * float(gaze_lr_mult)))
        return out

    # Dependency-free fallback for local machines without PyYAML. It is only
    # intended for this generated config layout.
    gaze_lr_match = re.search(r"^\s*gaze_lr_mult:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
    gaze_lr_mult = float(gaze_lr_match.group(1)) if gaze_lr_match else 1.0
    lora = [{"start_lr": float(m.group(1))} for m in re.finditer(r"^\s*start_lr:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)]
    out = []
    for idx, item in enumerate(lora):
        base_lr = float(item.get("start_lr", item.get("lr", 0.0)) or 0.0)
        out.append((idx, base_lr, base_lr * float(gaze_lr_mult)))
    return out


def _load_torch_deps():
    global torch, GazeTrajectoryEncoder
    if torch is None:
        import torch as torch_mod

        from app.hdepic_lora_action_anticipation.gaze_rnn import GazeTrajectoryEncoder as encoder_cls

        torch = torch_mod
        GazeTrajectoryEncoder = encoder_cls


def tensor_stats(name: str, x) -> tuple[str, bool]:
    finite = torch.isfinite(x)
    ok = bool(finite.all().item())
    xf = x.detach().float()
    if finite.any():
        vals = xf[finite]
        msg = (
            f"{name}: shape={tuple(x.shape)} dtype={x.dtype} finite={int(finite.sum())}/{x.numel()} "
            f"min={vals.min().item():.6g} max={vals.max().item():.6g} absmax={vals.abs().max().item():.6g}"
        )
    else:
        msg = f"{name}: shape={tuple(x.shape)} dtype={x.dtype} finite=0/{x.numel()}"
    return msg, ok


def smoke_one(fusion: str, steps: int, lr: float, device) -> list[str]:
    _load_torch_deps()
    torch.manual_seed(0)
    local = fusion == "local_attention"
    enc = GazeTrajectoryEncoder(
        embed_dim=1408,
        hidden_dim=64,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
        num_tokens=8,
        video_feat_dim=1408,
        video_proj_dim=128,
        video_fusion=fusion,
    ).to(device)
    enc.train()
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=0.1)
    bsz, traj_len = 2, 64
    traj = torch.randn(bsz, traj_len, 3, device=device).clamp(-0.5, 0.5)
    lengths = torch.full((bsz,), traj_len, dtype=torch.long, device=device)
    sample_valid = torch.ones(bsz, dtype=torch.bool, device=device)
    if local:
        video = torch.randn(bsz, traj_len, 9, 1408, device=device)
    else:
        video = torch.randn(bsz, traj_len, 1408, device=device)
    lines = [f"smoke fusion={fusion} steps={steps} lr={lr} device={device}"]
    for step in range(steps + 1):
        out = enc(traj, lengths=lengths, sample_valid=sample_valid, video_features=video)
        msg, ok = tensor_stats(f"step={step} gaze_tokens", out)
        lines.append(msg)
        monitor = enc._last_video_gate_mean if enc._last_video_gate_mean is not None else enc._last_fusion_alpha
        lines.append(f"step={step} monitor={monitor}")
        if not ok:
            break
        if step == steps:
            break
        loss = out.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_bad = []
        for name, param in enc.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                grad_bad.append(name)
        if grad_bad:
            lines.append(f"nonfinite gradients: {grad_bad[:10]}")
            break
        opt.step()
        param_bad = [name for name, param in enc.named_parameters() if not torch.isfinite(param).all()]
        if param_bad:
            lines.append(f"nonfinite params after step {step}: {param_bad[:10]}")
            break
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/hdepic_lora_action_anticipation/action_anticipation_frozen"))
    parser.add_argument("--config", type=Path, default=Path("configs/generated/hdepic_lora_rnn_gaze.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--smoke-lr", type=float, default=0.025)
    args = parser.parse_args()

    runs = [
        ("gated_nearest", "9365092", "hdepic-lora-rnn-gaze-gated-nearest-5ep-syncfix"),
        ("residual_conditioned", "9365101", "hdepic-lora-rnn-gaze-residual-alpha-5ep-syncfix"),
        ("local_attention", "9365106", "hdepic-lora-rnn-gaze-local-attn-5ep-syncfix"),
    ]
    print("# Log diagnosis")
    for fusion, job_id, tag in runs:
        out_log = args.logs_dir / f"hdepic_lora_rnn_gaze_{job_id}.out"
        err_log = args.logs_dir / f"hdepic_lora_rnn_gaze_{job_id}.err"
        csv_path = args.outputs_dir / tag / "log_r0.csv"
        print(f"\n## {fusion} job={job_id}")
        if out_log.exists():
            summary = summarize_log(out_log, FUSION_PATTERNS[fusion])
            print(f"monitor_count={summary['monitor_count']} nonfinite_count={summary['nonfinite_count']}")
            print(f"first_values={summary['first_values']} last_values={summary['last_values']}")
            print(f"first_nonfinite={summary['first_nonfinite']}")
            print(f"last_monitor={summary['last_monitor']}")
        else:
            print(f"missing log: {out_log}")
        rows = read_csv_rows(csv_path)
        print(f"validation_rows={len(rows)}")
        for row in rows:
            print(
                "epoch={epoch} action_top3={val-acc} action_recall5={val-recall} "
                "verb_top3={val-acc-verb} noun_top3={val-acc-noun}".format(**row)
            )
        errors = summarize_err(err_log)
        print(f"stderr_hits={len(errors)}")
        for line in errors[-5:]:
            print(f"  {line}")

    lrs = load_effective_lrs(args.config)
    if lrs:
        print("\n# Effective gaze LR from config")
        print("head,base_start_lr,gaze_start_lr")
        for idx, base_lr, gaze_lr in lrs:
            print(f"{idx},{base_lr:.6g},{gaze_lr:.6g}")
        print("note: monitor logs use classifier 0, so its gaze_start_lr is the first row.")

    if args.smoke:
        print("\n# Synthetic finite-input smoke test")
        _load_torch_deps()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for fusion, _job_id, _tag in runs:
            for line in smoke_one(fusion, args.smoke_steps, args.smoke_lr, device):
                print(line)


if __name__ == "__main__":
    main()
