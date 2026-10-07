"""Verify the shared adapter's official populations and optional source parity.

Run on a compute node in the prepared project environment. This validation
does not train a model or produce scientific accuracy results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    (1, "train"): (23943, 19403, 19340),
    (1, "val"): (16227, 4678, 4655),
    (2, "train"): (24027, 19417, 19340),
    (2, "val"): (13386, 4664, 4655),
    (3, "train"): (23822, 19360, 19302),
    (3, "val"): (13159, 4721, 4693),
}
PINS = {
    "scalable-anticipation": "f7d8710435af4c6abc9dee28f1fb890099d7ec43",
    "rulstm": "e648c7ee40a1a71381bfaa7563f2e8a6935275ab",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(root: Path) -> None:
    sys.path.insert(0, str(root))
    from app.scalant_baseline.official_stream_data import build_official_split1_ticks
    from app.scalant_baseline.official_stream_train import source_identity
    from app.scalant_baseline.stream_data import StreamingTSNReader

    report = {}
    reader = StreamingTSNReader()
    try:
        for split in (1, 2, 3):
            for population in ("train", "val"):
                ticks = build_official_split1_ticks(population, split)
                counts = [sum(t.target_actions[i] >= 0 for t in ticks) for i in (0, 1)]
                if (len(ticks), *counts) != EXPECTED[split, population]:
                    raise ValueError(f"Official population changed: {split}/{population}")
                tick_hash = hashlib.sha256()
                supported = [Counter(), Counter()]
                for tick in ticks:
                    # Include all feature keys and past/future labels, not only
                    # counts: equal denominators do not prove equal populations.
                    tick_hash.update(json.dumps(asdict(tick), sort_keys=True,
                                                separators=(",", ":")).encode())
                    tick_hash.update(b"\n")
                    for i, label in enumerate(tick.target_actions):
                        if label >= 0:
                            supported[i][label] += 1
                samples = [reader.read(tick).numpy().tobytes() for tick in (ticks[0], ticks[-1])]
                report[f"s{split}_{population}"] = {
                    "processed_ticks": len(ticks), "scored_h2_h6": counts,
                    "all_tick_fields_sha256": tick_hash.hexdigest(),
                    "by_class_h2_h6": [dict(sorted(c.items())) for c in supported],
                    "sample_feature_sha256": [hashlib.sha256(value).hexdigest() for value in samples],
                    "source_identity": source_identity(split),
                }
    finally:
        reader.close()
    print(json.dumps(report, sort_keys=True), flush=True)


def collect(root: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             "--worker", str(root)], cwd=root, env=env,
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Population check failed for {root}:\n{result.stderr[-4000:]}")
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--report", type=Path, required=False)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker.resolve())
        return
    compiled = 0
    for path in list((ROOT / "app/scalant_baseline").glob("*.py")) + list(
            (ROOT / "scripts/egtea").glob("*scalant*.py")):
        compile(path.read_text(), str(path), "exec")
        compiled += 1
    for name, expected in PINS.items():
        observed = subprocess.check_output(
            ["git", "-C", str(ROOT / "refer_repo" / name), "rev-parse", "HEAD"],
            text=True).strip()
        if observed != expected:
            raise ValueError(f"Upstream pin changed: {name}/{observed}")
    import yaml
    routes = yaml.safe_load((ROOT / "configs/evaluation_protocols/scalant_egtea.yaml").read_text())
    if len(routes["protocols"]) != 6:
        raise ValueError("Expected six protocol routes")
    for record in routes["protocols"].values():
        for key, value in record.items():
            if not isinstance(value, str):
                continue
            path = ROOT / (value.replace(".", "/") + ".py" if value.startswith("app.") else value)
            if not path.is_file():
                raise ValueError(f"Missing protocol route {key}: {value}")
    shared = collect(ROOT)
    unchanged = []
    if args.source_root is not None:
        source_root = args.source_root.resolve()
        source = collect(source_root)
        if shared != source:
            raise ValueError("Shared and original tick/source/feature identities differ")
        ported = {"egtea_cache.py", "stream_source_audit.py", "stream_data.py",
                  "decoder_fix_cpu_check.py"}
        for path in sorted((source_root / "app/scalant_baseline").glob("*.py")):
            if path.name in ported:
                continue
            if digest(path) != digest(ROOT / "app/scalant_baseline" / path.name):
                raise ValueError(f"Unexpected model/train/eval code change: {path.name}")
            unchanged.append(path.name)
    report = {"passed": True, "compiled_python_files": compiled,
              "source_parity_checked": args.source_root is not None,
              "byte_identical_adapter_modules": unchanged,
              "upstream_pins": PINS, "populations": shared}
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "populations"}), flush=True)


if __name__ == "__main__":
    main()
