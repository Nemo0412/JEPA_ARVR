#!/usr/bin/env python3
"""Classify HD-EPIC DataLoader/runtime failures from pulled Slurm logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "cgroup_oom": [
        "oom_kill",
        "oom-kill",
        "Detected ",
        "killed by signal: Killed",
        "signal: Killed",
    ],
    "cuda_oom": [
        "CUDA out of memory",
        "torch.OutOfMemoryError",
        "CUBLAS_STATUS_ALLOC_FAILED",
    ],
    "native_sigsegv": [
        "Unexpected segmentation fault encountered in worker",
        "killed by signal: Segmentation fault",
        "munmap_chunk()",
        "invalid pointer",
    ],
    "external_cancel": [
        "CANCELLED",
        "DUE TO TIME LIMIT",
        "SIGNAL Terminated",
        "externally cancelled",
    ],
    "python_traceback": [
        "Traceback (most recent call last)",
        "RuntimeError:",
        "AttributeError:",
        "ValueError:",
    ],
    "known_shutdown_warning": [
        "destroy_process_group() was not called",
    ],
    "completed_epoch": [
        re.compile(r"\]\s+\[\s*\d+\]\s+train acc"),
        "Wrote binary input adapter checkpoint",
    ],
}


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _hits(text: str, patterns: list[str | re.Pattern[str]]) -> list[str]:
    found = []
    for pat in patterns:
        if isinstance(pat, str):
            if pat in text:
                found.append(pat)
        elif pat.search(text):
            found.append(pat.pattern)
    return found


def classify(paths: list[Path]) -> dict:
    text_by_path = {str(path): _read(path) for path in paths}
    text = "\n".join(text_by_path.values())
    hits = {name: _hits(text, pats) for name, pats in PATTERNS.items()}

    completed = bool(hits["completed_epoch"])
    cgroup_oom = bool(hits["cgroup_oom"])
    cuda_oom = bool(hits["cuda_oom"])
    native_sigsegv = bool(hits["native_sigsegv"])
    external_cancel = bool(hits["external_cancel"])
    python_traceback = bool(hits["python_traceback"])
    only_shutdown_warning = bool(hits["known_shutdown_warning"]) and not (
        cgroup_oom or cuda_oom or native_sigsegv or external_cancel or python_traceback
    )

    if cgroup_oom:
        label = "cgroup-oom"
    elif cuda_oom:
        label = "cuda-oom"
    elif native_sigsegv and completed:
        label = "teardown-or-postmetric-sigsegv"
    elif native_sigsegv:
        label = "blocking-native-sigsegv"
    elif external_cancel:
        label = "external-cancel"
    elif only_shutdown_warning:
        label = "clean-with-known-shutdown-warning"
    elif completed:
        label = "clean-completed"
    elif python_traceback:
        label = "python-runtime-error"
    else:
        label = "unknown-or-incomplete"

    return {
        "label": label,
        "completed_epoch_or_checkpoint": completed,
        "hits": {name: vals for name, vals in hits.items() if vals},
        "files": [str(path) for path in paths],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Pulled .out/.err logs to classify")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    result = classify(args.paths)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"label: {result['label']}")
        print(f"completed_epoch_or_checkpoint: {result['completed_epoch_or_checkpoint']}")
        if result["hits"]:
            print("hits:")
            for name, vals in result["hits"].items():
                print(f"  {name}: {', '.join(vals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
