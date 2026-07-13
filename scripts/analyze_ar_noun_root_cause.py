"""Offline root-cause analysis for AR noun vs encoder baseline (B6).

Phases covered:
  A — metric comparison matrix (old action_top3, native/vjepa2, 5s window)
  B — 5s window rescore on B6 sample_hits (encoder / ar / oracle split)
  D — sample-level encoder-hit / AR-miss attribution
  E — B5-old vs B5-distance-lrm05 anomaly comparison

Usage:
    python scripts/analyze_ar_noun_root_cause.py --out-dir outputs/ar_noun_root_cause
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rescore_window import CRITERIA, TOPK_LEVELS, collect_window_labels, load_annotations, topk_hit

logger = logging.getLogger("ar_noun_root_cause")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / "data/hdepic_vjepa_annotations/HD_EPIC_val_vjepa.csv"
DEFAULT_OUT = ROOT / "outputs/ar_noun_root_cause"

PROBE_RUNS = {
    "B1-clean": {
        "10s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_failure_modes_native/B1-clean-10s",
        "1s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_failure_modes_native_1s/B1-clean-1s",
        "path_y_10s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_compare_path_y/B1-clean-10s/future_latent_compare.csv",
    },
    "B2-rnn-gaze": {
        "10s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_failure_modes_native/B2-rnn-gaze-10s",
        "1s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_failure_modes_native_1s/B2-rnn-gaze-1s",
        "path_y_10s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_compare_path_y/B2-rnn-gaze-10s/future_latent_compare.csv",
    },
    "B5-old-binary": {
        "10s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_failure_modes_native/B5-binary-input-adapter-10s",
        "1s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_failure_modes_native_1s/B5-binary-input-adapter-1s",
        "path_y_10s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_compare_path_y/B5-binary-input-adapter-10s/future_latent_compare.csv",
    },
    "B5-distance-lrm05": {
        "10s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_failure_modes_native/B5-binary-distance-lrm05-10s",
        "1s": ROOT / "outputs/hdepic_lora_action_anticipation/future_latent_failure_modes_native_1s/B5-binary-distance-lrm05-1s",
    },
}

RESCORE_5S_STANDARD = ROOT / "outputs/rescore_window_5s_native_vjepa2/rescore_summary.csv"
RESCORE_5S_LABELS = {
    "B1-clean": "B1-clean-native-10s",
    "B2-rnn-gaze": "B2-rnn-gaze-native-10s",
    "B5-old-binary": "B5-binary-native-10s",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def vjepa2_summary_from_head_metrics(head_rows: list[dict], horizon: str) -> dict[str, dict]:
    methods = sorted({r["method"] for r in head_rows})
    out: dict[str, dict] = {}
    metric_keys = ["noun_top3", "noun_recall5", "action_top3", "action_recall5"]
    for method in methods:
        subset = [r for r in head_rows if r["method"] == method]
        row_out = {"method": method, "horizon": horizon, "metric_scope": "native/vjepa2"}
        for key in metric_keys:
            best = max(subset, key=lambda r: float(r[key]))
            row_out[key] = float(best[key])
            row_out[f"{key}_head"] = int(best["head"])
        out[method] = row_out
    return out


def action_top3_same_head(head_rows: list[dict], horizon: str) -> dict[str, dict]:
    methods = sorted({r["method"] for r in head_rows})
    out: dict[str, dict] = {}
    for method in methods:
        subset = [r for r in head_rows if r["method"] == method]
        action_head = max(subset, key=lambda r: float(r["action_top3"]))["head"]
        row = next(r for r in subset if int(r["head"]) == int(action_head))
        out[method] = {
            "method": method,
            "horizon": horizon,
            "head": int(action_head),
            "noun_top3": float(row["noun_top3"]),
            "noun_recall5": float(row["noun_recall5"]),
        }
    return out


def load_path_y_noun(path: Path, horizon: float = 10.0) -> dict[str, dict]:
    rows = _read_csv(path)
    out = {}
    for row in rows:
        if float(row.get("horizon_sec", 0)) != horizon:
            continue
        method = row["method"]
        if method in {"direct_single", "direct_dense"}:
            continue
        out[method] = {
            "noun_top3": float(row["noun_top3"]),
            "noun_recall5": float(row["noun_recall5"]),
            "best_classifier": row.get("best_classifier", ""),
            "metric_scope": "filtered+action-head (Path Y CSV)",
        }
    return out


def load_standard_5s_noun(label: str) -> dict[str, float]:
    rows = _read_csv(RESCORE_5S_STANDARD)
    out = {}
    for row in rows:
        if row.get("label") != label or row.get("metric") != "noun":
            continue
        crit = row["criterion"]
        out[f"top3_{crit}"] = float(row["top3"])
        out[f"recall5_{crit}"] = float(row["class_mean_recall5"])
    out["note"] = "standard val AR rollout only; no encoder/AR split"
    return out


def phase_a_comparison_matrix(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for probe, horizons in PROBE_RUNS.items():
        for hz_key in ("1s", "10s"):
            run_dir = horizons.get(hz_key)
            if run_dir is None or not Path(run_dir).exists():
                continue
            head_rows = _read_csv(Path(run_dir) / "head_metrics.csv")
            if not head_rows:
                continue
            vjepa2 = vjepa2_summary_from_head_metrics(head_rows, hz_key)
            same_head = action_top3_same_head(head_rows, hz_key)
            for method in ("encoder", "ar"):
                if method not in vjepa2:
                    continue
                enc = vjepa2.get("encoder", {})
                ar = vjepa2.get("ar", {})
                row = {
                    "probe": probe,
                    "horizon": hz_key,
                    "method": method,
                    "metric_scope": "native/vjepa2",
                    "noun_top3": vjepa2[method]["noun_top3"],
                    "noun_recall5": vjepa2[method]["noun_recall5"],
                    "noun_top3_head": vjepa2[method]["noun_top3_head"],
                    "ar_minus_encoder_top3": (
                        vjepa2[method]["noun_top3"] - enc["noun_top3"] if method == "ar" and enc else ""
                    ),
                }
                rows.append(row)
            if hz_key == "10s" and "path_y_10s" in horizons:
                py = load_path_y_noun(Path(horizons["path_y_10s"]))
                for method in ("encoder", "ar"):
                    if method not in py:
                        continue
                    rows.append(
                        {
                            "probe": probe,
                            "horizon": hz_key,
                            "method": method,
                            "metric_scope": "filtered+action-head",
                            "noun_top3": py[method]["noun_top3"],
                            "noun_recall5": py[method]["noun_recall5"],
                            "noun_top3_head": py[method].get("best_classifier", ""),
                            "ar_minus_encoder_top3": (
                                py["ar"]["noun_top3"] - py["encoder"]["noun_top3"] if method == "ar" else ""
                            ),
                        }
                    )
                for method in ("encoder", "ar"):
                    if method not in same_head:
                        continue
                    rows.append(
                        {
                            "probe": probe,
                            "horizon": hz_key,
                            "method": method,
                            "metric_scope": "native action_top3 same-head",
                            "noun_top3": same_head[method]["noun_top3"],
                            "noun_recall5": same_head[method]["noun_recall5"],
                            "noun_top3_head": same_head[method]["head"],
                            "ar_minus_encoder_top3": (
                                same_head["ar"]["noun_top3"] - same_head["encoder"]["noun_top3"]
                                if method == "ar"
                                else ""
                            ),
                        }
                    )
        if probe in RESCORE_5S_LABELS and horizons.get("10s"):
            s5 = load_standard_5s_noun(RESCORE_5S_LABELS[probe])
            rows.append(
                {
                    "probe": probe,
                    "horizon": "10s",
                    "method": "standard_val_ar",
                    "metric_scope": "native/vjepa2 5s-window (no encoder/AR split)",
                    "noun_top3_strict": s5.get("top3_strict", ""),
                    "noun_top3_any_overlap": s5.get("top3_any_overlap", ""),
                    "noun_recall5_any_overlap": s5.get("recall5_any_overlap", ""),
                    "note": s5.get("note", ""),
                }
            )

    _write_csv(out_dir / "phase_a_comparison_matrix.csv", rows)
    logger.info("Phase A: wrote %d rows", len(rows))
    return rows


def _lookup_raw_labels(
    annotations: dict[str, list[dict]], video_id: str, start_frame: int, stop_frame: int
) -> tuple[int | None, int | None]:
    for seg in annotations.get(video_id, []):
        if seg["start_frame"] == start_frame and seg["stop_frame"] == stop_frame:
            return seg["verb_class"], seg["noun_class"]
    return None, None


def _best_heads_for_noun(head_rows: list[dict]) -> dict[str, int]:
    out = {}
    for method in sorted({r["method"] for r in head_rows}):
        subset = [r for r in head_rows if r["method"] == method]
        best = max(subset, key=lambda r: float(r["noun_top3"]))
        out[method] = int(best["head"])
    return out


def rescore_sample_hits_window(
    run_dir: Path,
    probe: str,
    annotations: dict[str, list[dict]],
    half_w_sec: float,
    annotation_fps: float,
    vfps_default: float,
) -> list[dict]:
    sample_rows = _read_csv(run_dir / "sample_hits.csv")
    head_rows = _read_csv(run_dir / "head_metrics.csv")
    if not sample_rows or not head_rows:
        return []

    best_noun_head = _best_heads_for_noun(head_rows)
    methods = [m for m in ("encoder", "ar", "oracle") if m in best_noun_head]

    hits: dict[str, dict] = {
        method: {c: {k: 0 for k in TOPK_LEVELS} for c in CRITERIA} for method in methods
    }
    totals: Counter[str] = Counter()
    per_class_tp: dict[str, dict[str, Counter]] = {
        method: {c: Counter() for c in CRITERIA} for method in methods
    }
    per_class_fn: dict[str, dict[str, Counter]] = {
        method: {c: Counter() for c in CRITERIA} for method in methods
    }

    by_sample_method: dict[tuple[int, str], dict] = {}
    for row in sample_rows:
        method = row["method"]
        if method not in best_noun_head:
            continue
        if int(row["head"]) != best_noun_head[method]:
            continue
        key = (int(row["sample_index"]), method)
        by_sample_method[key] = row

    for method in methods:
        seen_indices = sorted({idx for (idx, m) in by_sample_method if m == method})
        for sample_index in seen_indices:
            row = by_sample_method.get((sample_index, method))
            if row is None:
                continue
            totals[method] += 1
            video_id = str(row["video_id"])
            start_frame = int(row["start_frame"])
            stop_frame = int(row["stop_frame"])
            anchor_frame = stop_frame  # anticipation_point=[0,0] -> anchor=stop_frame
            vfps = vfps_default
            t_center = anchor_frame / max(vfps, 1e-6)
            noun_label = int(row["noun_label"])
            noun_preds = json.loads(row.get("noun_top10", "[]"))
            verb_raw, noun_raw = _lookup_raw_labels(annotations, video_id, start_frame, stop_frame)
            segs = annotations.get(video_id, [])

            for criterion in CRITERIA:
                if criterion == "strict":
                    noun_set = {noun_label}
                else:
                    _, raw_nouns, _ = collect_window_labels(segs, t_center, half_w_sec, annotation_fps, criterion)
                    if noun_raw is not None:
                        raw_nouns.add(noun_raw)
                    if raw_nouns:
                        noun_set = raw_nouns
                    else:
                        noun_set = {noun_label}

                for k in TOPK_LEVELS:
                    hits[method][criterion][k] += topk_hit(noun_preds, noun_set, k)
                if topk_hit(noun_preds, noun_set, 5):
                    per_class_tp[method][criterion][noun_label] += 1
                else:
                    per_class_fn[method][criterion][noun_label] += 1

    out_rows = []
    for method in methods:
        n = max(1, totals[method])
        for criterion in CRITERIA:
            recalls = []
            labels = set(per_class_tp[method][criterion]) | set(per_class_fn[method][criterion])
            for cls in labels:
                tp = per_class_tp[method][criterion][cls]
                fn = per_class_fn[method][criterion][cls]
                if tp + fn > 0:
                    recalls.append(tp / (tp + fn))
            out_rows.append(
                {
                    "probe": probe,
                    "method": method,
                    "criterion": criterion,
                    "metric": "noun",
                    "head_selection": "vjepa2 noun_top3 best head",
                    "total": totals[method],
                    "top3": 100.0 * hits[method][criterion][3] / n,
                    "top5": 100.0 * hits[method][criterion][5] / n,
                    "class_mean_recall5": 100.0 * sum(recalls) / max(1, len(recalls)),
                }
            )
    return out_rows


def phase_b_5s_window_b6(out_dir: Path, annotations_path: Path, window_sec: float) -> list[dict]:
    annotations = load_annotations(annotations_path)
    half_w = window_sec / 2.0
    all_rows: list[dict] = []
    for probe in ("B1-clean", "B2-rnn-gaze", "B5-old-binary"):
        run_dir = PROBE_RUNS[probe]["10s"]
        if not run_dir.exists():
            logger.warning("Missing run dir for %s", probe)
            continue
        rows = rescore_sample_hits_window(run_dir, probe, annotations, half_w, 30.0, 30.0)
        all_rows.extend(rows)

    _write_csv(out_dir / "phase_b_b6_5s_window_noun.csv", all_rows)

    pivot_rows = []
    for probe in ("B1-clean", "B2-rnn-gaze", "B5-old-binary"):
        for criterion in ("strict", "start_in_window", "any_overlap"):
            enc = next(
                (r for r in all_rows if r["probe"] == probe and r["method"] == "encoder" and r["criterion"] == criterion),
                None,
            )
            ar = next(
                (r for r in all_rows if r["probe"] == probe and r["method"] == "ar" and r["criterion"] == criterion),
                None,
            )
            if enc and ar:
                pivot_rows.append(
                    {
                        "probe": probe,
                        "criterion": criterion,
                        "encoder_top3": enc["top3"],
                        "ar_top3": ar["top3"],
                        "ar_minus_encoder_top3": ar["top3"] - enc["top3"],
                        "encoder_recall5": enc["class_mean_recall5"],
                        "ar_recall5": ar["class_mean_recall5"],
                    }
                )
    _write_csv(out_dir / "phase_b_ar_vs_encoder_5s_pivot.csv", pivot_rows)
    logger.info("Phase B: wrote %d rescore rows, %d pivot rows", len(all_rows), len(pivot_rows))
    return all_rows


def phase_d_sample_attribution(out_dir: Path) -> list[dict]:
    probe = "B1-clean"
    run_dir = PROBE_RUNS[probe]["10s"]
    sample_rows = _read_csv(run_dir / "sample_hits.csv")
    latent_rows = _read_csv(run_dir / "latent_stats.csv")
    head_rows = _read_csv(run_dir / "head_metrics.csv")
    best_heads = _best_heads_for_noun(head_rows)
    ar_head = best_heads.get("ar")
    enc_head = best_heads.get("encoder")

    latent_by_idx = {int(r["sample_index"]): r for r in latent_rows}

    enc_hits: dict[int, bool] = {}
    ar_hits: dict[int, bool] = {}
    enc_preds: dict[int, int] = {}
    ar_preds: dict[int, int] = {}
    labels: dict[int, int] = {}

    for row in sample_rows:
        idx = int(row["sample_index"])
        if int(row["head"]) == enc_head and row["method"] == "encoder":
            enc_hits[idx] = bool(int(row["noun_top3_hit"]))
            enc_preds[idx] = int(row["noun_top1"])
            labels[idx] = int(row["noun_label"])
        if int(row["head"]) == ar_head and row["method"] == "ar":
            ar_hits[idx] = bool(int(row["noun_top3_hit"]))
            ar_preds[idx] = int(row["noun_top1"])

    categories = Counter()
    category_samples: dict[str, list[dict]] = defaultdict(list)
    per_label_delta: dict[int, list[int]] = defaultdict(list)

    for idx in sorted(set(enc_hits) & set(ar_hits)):
        e_hit = enc_hits[idx]
        a_hit = ar_hits[idx]
        if e_hit and not a_hit:
            cat = "encoder_hit_ar_miss"
        elif not e_hit and a_hit:
            cat = "encoder_miss_ar_hit"
        elif e_hit and a_hit:
            cat = "both_hit"
        else:
            cat = "both_miss"
        categories[cat] += 1
        lat = latent_by_idx.get(idx, {})
        cos = float(lat.get("ar_to_oracle_cos") or 0)
        entry = {
            "sample_index": idx,
            "category": cat,
            "noun_label": labels.get(idx, -1),
            "encoder_pred": enc_preds.get(idx, -1),
            "ar_pred": ar_preds.get(idx, -1),
            "ar_to_oracle_cos": cos,
            "ar_target_norm": lat.get("ar_target_norm", ""),
        }
        category_samples[cat].append(entry)
        per_label_delta[labels[idx]].append(int(a_hit) - int(e_hit))

    summary_rows = []
    for cat, count in categories.items():
        subset = category_samples[cat]
        cos_vals = [s["ar_to_oracle_cos"] for s in subset if s["ar_to_oracle_cos"]]
        mean_cos = sum(cos_vals) / len(cos_vals) if cos_vals else 0.0
        label_ctr = Counter(s["noun_label"] for s in subset)
        summary_rows.append(
            {
                "probe": probe,
                "category": cat,
                "count": count,
                "pct": 100.0 * count / max(1, sum(categories.values())),
                "mean_ar_to_oracle_cos": mean_cos,
                "top_noun_labels": json.dumps(label_ctr.most_common(5)),
            }
        )

    class_delta_rows = []
    for label, deltas in sorted(per_label_delta.items()):
        class_delta_rows.append(
            {
                "noun_label": label,
                "n_samples": len(deltas),
                "ar_minus_encoder_hit_rate": sum(deltas) / len(deltas),
            }
        )

    detail_rows = [s for rows in category_samples.values() for s in rows]
    _write_csv(out_dir / "phase_d_category_summary.csv", summary_rows)
    _write_csv(out_dir / "phase_d_per_class_delta.csv", class_delta_rows)
    _write_csv(out_dir / "phase_d_sample_details.csv", detail_rows)
    logger.info("Phase D: categories=%s", dict(categories))
    return summary_rows


def phase_c_input_ablation(out_dir: Path) -> list[dict]:
    ablation_dir = (
        ROOT
        / "outputs/hdepic_lora_action_anticipation/future_latent_failure_modes_native/B1-clean-10s-input-ablation"
    )
    head_path = ablation_dir / "head_metrics.csv"
    if not head_path.exists():
        logger.warning("Phase C output not found at %s (job may still be running)", ablation_dir)
        return []

    head_rows = _read_csv(head_path)
    vjepa2 = vjepa2_summary_from_head_metrics(head_rows, "10s")
    order = ["encoder", "observed_plus_observed_tail", "ar", "oracle_target_only", "oracle"]
    rows = []
    for method in order:
        if method not in vjepa2:
            continue
        rows.append(
            {
                "probe": "B1-clean",
                "horizon": "10s",
                "method": method,
                "noun_top3": vjepa2[method]["noun_top3"],
                "noun_recall5": vjepa2[method]["noun_recall5"],
                "noun_top3_head": vjepa2[method]["noun_top3_head"],
            }
        )
    enc_top3 = vjepa2.get("encoder", {}).get("noun_top3")
    for row in rows:
        if enc_top3 is not None and row["method"] != "encoder":
            row["minus_encoder_noun_top3"] = float(row["noun_top3"]) - float(enc_top3)

    _write_csv(out_dir / "phase_c_input_ablation.csv", rows)
    logger.info("Phase C: wrote %d ablation rows", len(rows))
    return rows
    

def phase_e_b5_anomaly(out_dir: Path) -> list[dict]:
    rows = []
    for probe_key in ("B5-old-binary", "B5-distance-lrm05"):
        run_dir = PROBE_RUNS[probe_key]["10s"]
        if not run_dir.exists():
            continue
        head_rows = _read_csv(run_dir / "head_metrics.csv")
        vjepa2 = vjepa2_summary_from_head_metrics(head_rows, "10s")
        latent_summary = _read_csv(run_dir / "latent_summary.csv")
        confusions = _read_csv(run_dir / "noun_top1_confusions.csv")

        lat = latent_summary[0] if latent_summary else {}
        ar_conf = [r for r in confusions if r.get("method") == "ar"]
        top_conf = sorted(ar_conf, key=lambda r: int(r.get("count", 0)), reverse=True)[:3]

        enc = vjepa2.get("encoder", {})
        ar = vjepa2.get("ar", {})
        rows.append(
            {
                "probe": probe_key,
                "encoder_noun_top3": enc.get("noun_top3", ""),
                "ar_noun_top3": ar.get("noun_top3", ""),
                "ar_minus_encoder_top3": (
                    float(ar.get("noun_top3", 0)) - float(enc.get("noun_top3", 0))
                    if enc and ar
                    else ""
                ),
                "ar_to_oracle_cos_mean": lat.get("ar_to_oracle_cos_mean", ""),
                "ar_target_norm_mean": lat.get("ar_target_norm_mean", ""),
                "oracle_target_norm_mean": lat.get("oracle_target_norm_mean", ""),
                "top_ar_noun_confusions": json.dumps(
                    [(r.get("noun_label"), r.get("noun_top1"), r.get("count")) for r in top_conf]
                ),
            }
        )

    _write_csv(out_dir / "phase_e_b5_anomaly.csv", rows)
    logger.info("Phase E: wrote %d probe rows", len(rows))
    return rows


def write_markdown_report(out_dir: Path) -> None:
    a = _read_csv(out_dir / "phase_a_comparison_matrix.csv")
    b = _read_csv(out_dir / "phase_b_ar_vs_encoder_5s_pivot.csv")
    c = _read_csv(out_dir / "phase_c_input_ablation.csv")
    d = _read_csv(out_dir / "phase_d_category_summary.csv")
    e = _read_csv(out_dir / "phase_e_b5_anomaly.csv")

    lines = [
        "# AR Noun Root Cause Analysis Report",
        "",
        "## Phase A: Metric existence summary",
        "",
        "| probe | horizon | scope | encoder noun Top-3 | AR noun Top-3 | AR − encoder |",
        "|---|---|---|---:|---:|---:|",
    ]
    for probe in ("B1-clean", "B2-rnn-gaze", "B5-old-binary", "B5-distance-lrm05"):
        for hz in ("1s", "10s"):
            enc = next(
                (
                    r
                    for r in a
                    if r.get("probe") == probe
                    and r.get("horizon") == hz
                    and r.get("method") == "encoder"
                    and r.get("metric_scope") == "native/vjepa2"
                ),
                None,
            )
            ar = next(
                (
                    r
                    for r in a
                    if r.get("probe") == probe
                    and r.get("horizon") == hz
                    and r.get("method") == "ar"
                    and r.get("metric_scope") == "native/vjepa2"
                ),
                None,
            )
            if enc and ar:
                delta = float(ar["noun_top3"]) - float(enc["noun_top3"])
                lines.append(
                    f"| {probe} | {hz} | native/vjepa2 | {float(enc['noun_top3']):.2f} | "
                    f"{float(ar['noun_top3']):.2f} | {delta:+.2f} |"
                )

    lines.extend(
        [
            "",
            "## Phase B: 5s window B6 encoder vs AR (noun Top-3)",
            "",
            "| probe | criterion | encoder | AR | AR − encoder |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in b:
        lines.append(
            f"| {row['probe']} | {row['criterion']} | {float(row['encoder_top3']):.2f} | "
            f"{float(row['ar_top3']):.2f} | {float(row['ar_minus_encoder_top3']):+.2f} |"
        )

    lines.extend(["", "## Phase C: B1 10s input-construction ablation", ""])
    if c:
        lines.extend(
            [
                "| method | noun Top-3 | vs encoder |",
                "|---|---:|---:|",
            ]
        )
        enc = next((float(r["noun_top3"]) for r in c if r["method"] == "encoder"), None)
        for row in c:
            delta = float(row["noun_top3"]) - enc if enc is not None and row["method"] != "encoder" else 0.0
            delta_s = f"{delta:+.2f}" if row["method"] != "encoder" else "—"
            lines.append(f"| {row['method']} | {float(row['noun_top3']):.2f} | {delta_s} |")
    else:
        lines.append("- Pending: Slurm job input ablation (see raw log for job ID).")

    lines.extend(["", "## Phase D: B1 10s sample attribution", ""])
    for row in d:
        lines.append(
            f"- {row['category']}: n={row['count']} ({float(row['pct']):.1f}%), "
            f"mean ar_to_oracle_cos={float(row['mean_ar_to_oracle_cos']):.3f}"
        )

    lines.extend(["", "## Phase E: B5 checkpoint comparison", ""])
    for row in e:
        lines.append(
            f"- {row['probe']}: encoder noun Top-3={row['encoder_noun_top3']}, "
            f"AR={row['ar_noun_top3']}, delta={row['ar_minus_encoder_top3']}"
        )

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--phases", default="a,b,c,d,e", help="Comma-separated phases to run")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phases = {x.strip().lower() for x in args.phases.split(",")}

    if "a" in phases:
        phase_a_comparison_matrix(args.out_dir)
    if "b" in phases:
        phase_b_5s_window_b6(args.out_dir, args.annotations, args.window_sec)
    if "c" in phases:
        phase_c_input_ablation(args.out_dir)
    if "d" in phases:
        phase_d_sample_attribution(args.out_dir)
    if "e" in phases:
        phase_e_b5_anomaly(args.out_dir)

    write_markdown_report(args.out_dir)
    logger.info("Done. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
