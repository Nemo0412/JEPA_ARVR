#!/usr/bin/env python3
"""Project the frozen EGTEA +2/+4/+6 labels to one selected horizon.

The pinned ll Vanilla trainer indexes labels by the *position within the
requested horizon list*.  A one-horizon invocation therefore always reads
label column zero.  This adapter preserves every split row and every
non-label field, but rewrites ``mtp_verbs``, ``mtp_nouns`` and ``mtp_mask`` to
the single source column selected by ``--horizon``.  The ll trainer remains
byte-identical to its reference commit.

For bounded smoke tests, the projected training rows are reduced only after
projection, while preserving the complete selected-horizon class space.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


HORIZON_TO_INDEX = {2: 0, 4: 1, 6: 2}
EXPECTED_SOURCE = {
    "train": ("dcdc98c9993ae028dbb4a2877deff09a", 25_256),
    "val": ("cb04d6eddc2e65910899fde8b3c08659", 22_741),
}
EXPECTED_LABEL_COUNTS = (19, 51, 106)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",")]


def read_source(path: Path, split: str) -> tuple[list[str], list[dict[str, str]]]:
    expected_md5, expected_rows = EXPECTED_SOURCE[split]
    actual_md5 = md5(path)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"{split} source MD5 mismatch: {actual_md5} != {expected_md5}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != expected_rows:
        raise RuntimeError(f"{split} row count mismatch: {len(rows)} != {expected_rows}")
    required = {"mtp_verbs", "mtp_nouns", "mtp_mask", "context_sec"}
    if not required.issubset(header):
        raise RuntimeError(f"{split} CSV missing fields: {sorted(required.difference(header))}")
    return header, rows


def project_rows(rows: list[dict[str, str]], index: int, split: str) -> list[dict[str, str]]:
    projected: list[dict[str, str]] = []
    for line_no, row in enumerate(rows, start=2):
        values = {field: split_values(row[field]) for field in ("mtp_verbs", "mtp_nouns", "mtp_mask")}
        lengths = {field: len(parts) for field, parts in values.items()}
        if any(length != 3 for length in lengths.values()):
            raise RuntimeError(f"{split}:{line_no} expected three horizon labels, got {lengths}")
        out = dict(row)
        for field, parts in values.items():
            out[field] = parts[index]
        projected.append(out)
    return projected


def row_label(row: dict[str, str]) -> tuple[int, int] | None:
    verb = int(row["mtp_verbs"])
    noun = int(row["mtp_nouns"])
    mask = float(row["mtp_mask"])
    if mask < 0.5 or verb < 0 or noun < 0:
        return None
    return verb, noun


def label_sets(rows: list[dict[str, str]]) -> tuple[set[int], set[int], set[tuple[int, int]]]:
    verbs: set[int] = set()
    nouns: set[int] = set()
    actions: set[tuple[int, int]] = set()
    for row in rows:
        label = row_label(row)
        if label is None:
            continue
        verb, noun = label
        verbs.add(verb)
        nouns.add(noun)
        actions.add(label)
    return verbs, nouns, actions


def bounded_smoke(
    train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    all_verbs, all_nouns, all_actions = label_sets(train_rows)
    actual_counts = (len(all_verbs), len(all_nouns), len(all_actions))
    if actual_counts != EXPECTED_LABEL_COUNTS:
        raise RuntimeError(
            f"selected-horizon class space mismatch: {actual_counts} != {EXPECTED_LABEL_COUNTS}"
        )

    seen_verbs: set[int] = set()
    seen_nouns: set[int] = set()
    seen_actions: set[tuple[int, int]] = set()
    selected_train: list[dict[str, str]] = []
    for row in train_rows:
        label = row_label(row)
        if label is None:
            continue
        verb, noun = label
        if verb not in seen_verbs or noun not in seen_nouns or label not in seen_actions:
            selected_train.append(row)
            seen_verbs.add(verb)
            seen_nouns.add(noun)
            seen_actions.add(label)
        if (seen_verbs, seen_nouns, seen_actions) == (all_verbs, all_nouns, all_actions):
            break
    if (seen_verbs, seen_nouns, seen_actions) != (all_verbs, all_nouns, all_actions):
        raise RuntimeError("bounded train rows did not preserve the complete class space")

    selected_val: list[dict[str, str]] = []
    seen_contexts: set[str] = set()
    for row in val_rows:
        label = row_label(row)
        context = row["context_sec"]
        if label is not None and label in all_actions and context not in seen_contexts:
            selected_val.append(row)
            seen_contexts.add(context)
    if seen_contexts != {"4.0", "6.0", "8.0", "10.0"}:
        raise RuntimeError(f"bounded validation contexts are incomplete: {sorted(seen_contexts)}")
    return selected_train, selected_val


def write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=sorted(HORIZON_TO_INDEX), required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    args = parser.parse_args()

    train_header, train_source = read_source(args.train_csv, "train")
    val_header, val_source = read_source(args.val_csv, "val")
    if train_header != val_header:
        raise RuntimeError("train/val headers differ")

    index = HORIZON_TO_INDEX[args.horizon]
    train_rows = project_rows(train_source, index, "train")
    val_rows = project_rows(val_source, index, "val")
    full_counts = tuple(len(values) for values in label_sets(train_rows))
    if full_counts != EXPECTED_LABEL_COUNTS:
        raise RuntimeError(f"+{args.horizon}s class space mismatch: {full_counts} != {EXPECTED_LABEL_COUNTS}")
    if args.mode == "smoke":
        train_rows, val_rows = bounded_smoke(train_rows, val_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.out_dir / "train.csv"
    val_out = args.out_dir / "val.csv"
    write_csv(train_out, train_header, train_rows)
    write_csv(val_out, val_header, val_rows)
    manifest = {
        "adapter": "frozen_triplet_to_single_selected_horizon_v1",
        "horizon_sec": args.horizon,
        "source_column_index": index,
        "mode": args.mode,
        "source": {
            "train_md5": EXPECTED_SOURCE["train"][0],
            "val_md5": EXPECTED_SOURCE["val"][0],
        },
        "rows": {"train": len(train_rows), "val": len(val_rows)},
        "label_counts": {"verb": full_counts[0], "noun": full_counts[1], "action": full_counts[2]},
        "outputs": {"train_md5": md5(train_out), "val_md5": md5(val_out)},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
