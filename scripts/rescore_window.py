"""Rescore HD-EPIC anticipation predictions with a relaxed time-window label.

For each val sample dumped by PredictionDumper, the strict label is the verb /
noun / action that the model was trained to predict at one timestamp. This
script relaxes that label to "any action segment whose ground truth falls
inside a [t - W/2, t + W/2] window around the original target time," where
t = start_frame / vfps as agreed with the user.

Two membership criteria are reported side-by-side:

    start_in_window   segment.start_frame / vfps ∈ window
    any_overlap       segment overlaps the window at all

For every (ckpt, criterion) pair the script reports Top-1 / Top-3 / Top-5 /
Top-10 hit rate and class-mean Recall@5 (using the GT *set* — a sample
counts as a hit if any member of its label set appears in the predicted
top-k). Window-set-size statistics are emitted to highlight how much of the
gain is just "the set got bigger."

Usage:
    python scripts/rescore_window.py \\
        --predictions outputs/.../val_predictions.csv \\
        --annotations data/hdepic_vjepa_annotations/HD_EPIC_val_vjepa.csv \\
        --window-sec 5.0 \\
        --out-dir outputs/.../rescore

Pass --predictions multiple times to compare several ckpts in one run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("rescore_window")

TOPK_LEVELS = (1, 3, 5, 10)
CRITERIA = ("strict", "start_in_window", "any_overlap")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Path to val_predictions.csv. Pass multiple times to compare runs.",
    )
    p.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional short label per --predictions (must match length). Defaults to parent dir name.",
    )
    p.add_argument("--annotations", required=True, help="HD_EPIC_val_vjepa.csv")
    p.add_argument("--annotation-fps", type=float, default=30.0, help="FPS used to write start/stop_frame")
    p.add_argument("--window-sec", type=float, default=5.0, help="Total window width centered on t")
    p.add_argument(
        "--class-maps",
        default=None,
        help="Optional explicit path to <predictions>_class_maps.json. If omitted, inferred per predictions file.",
    )
    p.add_argument("--out-dir", required=True, help="Where to write rescore tables")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s")


def load_class_maps(predictions_path: Path, explicit: Path | None) -> dict[str, Any]:
    if explicit is not None:
        path = Path(explicit)
    else:
        path = predictions_path.with_name(f"{predictions_path.stem}_class_maps.json")
    if not path.exists():
        logger.warning("No class_maps.json next to %s; action set will use raw (verb,noun) tuples.", predictions_path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, dict] = {}
    for name, entries in raw.items():
        d: dict = {}
        if isinstance(entries, list):
            for entry in entries:
                key = entry["key"]
                val = entry["value"]
                if isinstance(key, list):
                    key = tuple(key)
                d[key] = val
        elif isinstance(entries, dict):
            d = entries
        out[name] = d
    return out


def load_annotations(path: Path) -> dict[str, list[dict]]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                by_video[str(row["video_id"])].append(
                    {
                        "start_frame": int(row["start_frame"]),
                        "stop_frame": int(row["stop_frame"]),
                        "verb_class": int(row["verb_class"]),
                        "noun_class": int(row["noun_class"]),
                    }
                )
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed annotation row: %r (%s)", row, exc)
    for segs in by_video.values():
        segs.sort(key=lambda s: s["start_frame"])
    return by_video


def collect_window_labels(
    video_segs: list[dict],
    t_center_sec: float,
    half_w_sec: float,
    annotation_fps: float,
    criterion: str,
) -> tuple[set[int], set[int], set[tuple[int, int]]]:
    """Return (verbs, nouns, (verb,noun) pairs) inside the window."""
    lo = (t_center_sec - half_w_sec) * annotation_fps
    hi = (t_center_sec + half_w_sec) * annotation_fps
    verbs: set[int] = set()
    nouns: set[int] = set()
    pairs: set[tuple[int, int]] = set()
    for seg in video_segs:
        if criterion == "start_in_window":
            inside = lo <= seg["start_frame"] <= hi
        elif criterion == "any_overlap":
            inside = seg["stop_frame"] >= lo and seg["start_frame"] <= hi
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        if inside:
            verbs.add(seg["verb_class"])
            nouns.add(seg["noun_class"])
            pairs.add((seg["verb_class"], seg["noun_class"]))
    return verbs, nouns, pairs


def remap_set(
    raw_set: set,
    mapping: dict,
) -> set:
    """Map raw class IDs (or (verb,noun) tuples) through PredictionDumper class_maps."""
    if not mapping:
        return raw_set
    out: set = set()
    for item in raw_set:
        if item in mapping:
            out.add(mapping[item])
        elif isinstance(item, tuple) and item in mapping:
            out.add(mapping[item])
    return out


def topk_hit(pred_list: list[int], gt_set: set[int], k: int) -> int:
    if not gt_set:
        return 0
    for p in pred_list[:k]:
        if p in gt_set:
            return 1
    return 0


def rescore_one(
    predictions_path: Path,
    annotations: dict[str, list[dict]],
    annotation_fps: float,
    half_w_sec: float,
    class_maps: dict,
    out_dir: Path,
    label: str,
) -> dict[str, dict]:
    verb_map = class_maps.get("verb", {})
    noun_map = class_maps.get("noun", {})
    action_map = class_maps.get("action", {})

    per_criterion: dict[str, dict] = {}
    set_sizes: dict[str, list[int]] = {c: [] for c in CRITERIA if c != "strict"}
    hits_by_criterion: dict[str, dict] = {
        c: {
            name: {k: 0 for k in TOPK_LEVELS}
            for name in ("verb", "noun", "action")
        }
        for c in CRITERIA
    }
    per_class_state: dict[str, dict] = {
        c: {
            name: {"tp": Counter(), "fn": Counter(), "labels": set()}
            for name in ("verb", "noun", "action")
        }
        for c in CRITERIA
    }
    total = 0
    missing_video = 0

    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            video_id = str(row.get("video_id", "")).strip()
            try:
                start_frame = int(row["start_frame"])
                vfps = float(row.get("vfps") or annotation_fps)
            except (KeyError, ValueError):
                logger.warning("Missing start_frame/vfps in row %d, skipping", total)
                continue
            t_center = start_frame / max(vfps, 1e-6)

            verb_label = int(row["verb_label"])
            noun_label = int(row["noun_label"])
            action_label = int(row["action_label"])
            verb_preds = json.loads(row.get("verb_top10", "[]"))
            noun_preds = json.loads(row.get("noun_top10", "[]"))
            action_preds = json.loads(row.get("action_top10", "[]"))

            segs = annotations.get(video_id, [])
            if not segs:
                missing_video += 1

            for criterion in CRITERIA:
                if criterion == "strict":
                    verb_set = {verb_label}
                    noun_set = {noun_label}
                    action_set = {action_label}
                else:
                    raw_verbs, raw_nouns, raw_pairs = collect_window_labels(
                        segs, t_center, half_w_sec, annotation_fps, criterion
                    )
                    raw_verbs.add(int(row["verb_raw"]))
                    raw_nouns.add(int(row["noun_raw"]))
                    raw_pairs.add((int(row["verb_raw"]), int(row["noun_raw"])))
                    verb_set = remap_set(raw_verbs, verb_map) or {verb_label}
                    noun_set = remap_set(raw_nouns, noun_map) or {noun_label}
                    action_set = remap_set(raw_pairs, action_map) or {action_label}
                    set_sizes[criterion].append(len(action_set))

                for name, pred_list, gt_set in (
                    ("verb", verb_preds, verb_set),
                    ("noun", noun_preds, noun_set),
                    ("action", action_preds, action_set),
                ):
                    for k in TOPK_LEVELS:
                        hit = topk_hit(pred_list, gt_set, k)
                        hits_by_criterion[criterion][name][k] += hit
                    state = per_class_state[criterion][name]
                    primary_gt = {"verb": verb_label, "noun": noun_label, "action": action_label}[name]
                    state["labels"].add(primary_gt)
                    if topk_hit(pred_list, gt_set, 5):
                        state["tp"][primary_gt] += 1
                    else:
                        state["fn"][primary_gt] += 1

    rows = []
    for criterion in CRITERIA:
        for name in ("verb", "noun", "action"):
            row_out = {"label": label, "criterion": criterion, "metric": name, "total": total}
            for k in TOPK_LEVELS:
                h = hits_by_criterion[criterion][name][k]
                row_out[f"top{k}"] = 100.0 * h / max(1, total)
            state = per_class_state[criterion][name]
            recalls = []
            for cls in state["labels"]:
                tp = state["tp"][cls]
                fn = state["fn"][cls]
                if tp + fn > 0:
                    recalls.append(tp / (tp + fn))
            row_out["class_mean_recall5"] = 100.0 * sum(recalls) / max(1, len(recalls))
            rows.append(row_out)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"rescore_{label}.csv"
    fieldnames = ["label", "criterion", "metric", "total"] + [f"top{k}" for k in TOPK_LEVELS] + ["class_mean_recall5"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s", out_csv)

    set_size_summary = {}
    for criterion, sizes in set_sizes.items():
        if not sizes:
            continue
        sizes_sorted = sorted(sizes)
        set_size_summary[criterion] = {
            "n": len(sizes),
            "mean": sum(sizes) / len(sizes),
            "median": sizes_sorted[len(sizes_sorted) // 2],
            "p90": sizes_sorted[max(0, int(0.9 * len(sizes_sorted)) - 1)],
            "max": sizes_sorted[-1],
        }
    set_csv = out_dir / f"rescore_{label}_set_sizes.csv"
    with set_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["criterion", "n", "mean", "median", "p90", "max"])
        w.writeheader()
        for criterion, stats in set_size_summary.items():
            w.writerow({"criterion": criterion, **stats})
    logger.info("Wrote %s", set_csv)

    if missing_video:
        logger.warning("%d / %d rows had no matching video_id in annotations", missing_video, total)

    return {"rows": rows, "set_sizes": set_size_summary, "total": total, "missing_video": missing_video}


def write_combined(results: list[tuple[str, dict]], out_dir: Path, window_sec: float) -> None:
    combined_csv = out_dir / "rescore_summary.csv"
    fieldnames = ["label", "criterion", "metric", "total"] + [f"top{k}" for k in TOPK_LEVELS] + ["class_mean_recall5"]
    with combined_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _, result in results:
            w.writerows(result["rows"])
    logger.info("Wrote combined table: %s", combined_csv)

    note = out_dir / "README.txt"
    with note.open("w", encoding="utf-8") as f:
        f.write(f"Window width: {window_sec:.3f}s (half = {window_sec/2:.3f}s)\n")
        f.write("Window center t = start_frame / vfps per dumped row.\n")
        f.write("strict           = original single-label hit (matches eval-time Top-k).\n")
        f.write("start_in_window  = segment.start_frame / vfps inside [t - W/2, t + W/2].\n")
        f.write("any_overlap      = segment overlaps the window at all.\n")
        f.write("class_mean_recall5 counts a sample as hit-for-class if any window-set label is in Top-5.\n")
        f.write(f"\nLabels: {[lbl for lbl, _ in results]}\n")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if args.label is not None and len(args.label) != len(args.predictions):
        raise SystemExit("--label must be passed as many times as --predictions, or omitted entirely")

    out_dir = Path(args.out_dir)
    annotations = load_annotations(Path(args.annotations))
    logger.info("Loaded annotations for %d videos", len(annotations))

    half_w = args.window_sec / 2.0
    results: list[tuple[str, dict]] = []
    for idx, pred_path_str in enumerate(args.predictions):
        pred_path = Path(pred_path_str)
        label = args.label[idx] if args.label else pred_path.parent.name
        class_maps = load_class_maps(pred_path, Path(args.class_maps) if args.class_maps else None)
        logger.info("Rescoring %s (label=%s)", pred_path, label)
        res = rescore_one(
            pred_path,
            annotations,
            args.annotation_fps,
            half_w,
            class_maps,
            out_dir,
            label,
        )
        results.append((label, res))

    write_combined(results, out_dir, args.window_sec)


if __name__ == "__main__":
    main()
