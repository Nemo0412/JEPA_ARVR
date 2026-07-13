"""Dedicated diagnostics for B6 future-latent failure modes.

This script intentionally lives outside the main eval path. It reuses the
standalone future_latent_compare loader/model helpers, then writes removable
CSV/JSON artifacts for:

1. per-head action/verb/noun metrics and head-selection disagreement;
2. per-sample AR latent cosine/MSE versus hit/miss outcomes;
3. noun top-1 prior/concentration/confusion checks;
4. latent distribution summaries for observed-tail, oracle-target, and AR-target
   tokens so binary-map checkpoints can be analyzed separately from clean.

By default metrics use the upstream/native V-JEPA2 action-anticipation
convention: do not pass valid_classes into the metric/top-k calculation.
Use --metric-scope filtered only when explicitly matching project-local
future_latent_compare CSVs that filter to val split classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation.future_latent_compare import (
    _build_gaze_components,
    _build_samples,
    _collate,
    _labels,
    FutureOracleDataset,
    _last_layer,
    _load_classifiers,
    _load_encoder_predictor,
    _predict_ar,
)
from app.hdepic_lora_action_anticipation.gaze_rnn import call_classifier, encode_gaze_tokens
from evals.action_anticipation_frozen.dataloader import filter_annotations

logger = logging.getLogger("future_latent_failure_modes")


class MetricTracker:
    def __init__(self):
        self.total = 0
        self.top1_hits = 0
        self.top3_hits = 0
        self.top5_hits = 0
        self.label_total = Counter()
        self.label_top5_hits = Counter()

    def update(self, label: int, preds: list[int]):
        self.total += 1
        self.top1_hits += int(preds[:1] == [label])
        self.top3_hits += int(label in preds[:3])
        self.top5_hits += int(label in preds[:5])
        self.label_total[label] += 1
        self.label_top5_hits[label] += int(label in preds[:5])

    def values(self, valid_labels: set[int]):
        seen = [label for label in valid_labels if self.label_total[label] > 0]
        recall = 0.0
        if seen:
            recall = sum(self.label_top5_hits[label] / self.label_total[label] for label in seen) / len(seen)
        denom = max(1, self.total)
        return {
            "top1": 100.0 * self.top1_hits / denom,
            "top3": 100.0 * self.top3_hits / denom,
            "top5": 100.0 * self.top5_hits / denom,
            "recall5": 100.0 * recall,
        }


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = [count / total for count in counter.values() if count > 0]
    return float(-sum(p * math.log(p) for p in probs))


def _topk(logits: torch.Tensor, k: int, valid_classes: set[int] | None = None) -> tuple[list[int], list[float]]:
    scores = torch.sigmoid(logits.float())
    if valid_classes is not None:
        filtered = torch.zeros_like(scores)
        valid = [c for c in valid_classes if 0 <= c < scores.numel()]
        if valid:
            idx = torch.tensor(valid, device=scores.device, dtype=torch.long)
            filtered[idx] = scores[idx]
        scores = filtered
    kk = min(k, scores.numel())
    vals, idx = scores.topk(kk)
    return [int(x) for x in idx.detach().cpu().tolist()], [float(x) for x in vals.detach().cpu().tolist()]


def _valid_label_sets(annotations: dict) -> dict[str, set[int]]:
    return {
        "verb": {int(x) for x in annotations["val_verbs"]},
        "noun": {int(x) for x in annotations["val_nouns"]},
        "action": {int(x) for x in annotations["val_actions"]},
    }


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().flatten(1)


def _row_stats(values: list[float], prefix: str) -> dict:
    if not values:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_std": "",
            f"{prefix}_p10": "",
            f"{prefix}_p50": "",
            f"{prefix}_p90": "",
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
    }


def _latent_summary(latent_rows: list[dict]) -> list[dict]:
    keys = [
        "observed_tail_norm",
        "oracle_target_norm",
        "ar_target_norm",
        "observed_tail_to_oracle_cos",
        "observed_tail_to_oracle_mse",
        "ar_to_oracle_cos",
        "ar_to_oracle_mse",
        "ar_to_observed_tail_cos",
        "ar_to_observed_tail_mse",
    ]
    row = {"samples": len(latent_rows)}
    for key in keys:
        vals = [float(r[key]) for r in latent_rows if r.get(key) not in {"", None}]
        row.update(_row_stats(vals, key))
    return [row]


def _select_heads(head_rows: list[dict]) -> dict[str, dict[str, int]]:
    selected: dict[str, dict[str, int]] = defaultdict(dict)
    for method in sorted({row["method"] for row in head_rows}):
        subset = [row for row in head_rows if row["method"] == method]
        for task in ["action", "verb", "noun"]:
            best = max(subset, key=lambda row: float(row[f"{task}_top3"]))
            selected[method][f"{task}_top3"] = int(best["head"])
            best_r5 = max(subset, key=lambda row: float(row[f"{task}_recall5"]))
            selected[method][f"{task}_recall5"] = int(best_r5["head"])
    return {method: dict(values) for method, values in selected.items()}


def _add_cosine_bins(sample_rows: list[dict], selected_heads: dict[str, dict[str, int]]) -> list[dict]:
    rows = []
    ar_rows = [
        row
        for row in sample_rows
        if row["method"] == "ar" and row.get("latent_cos_to_oracle") not in {"", None}
    ]
    if not ar_rows:
        return rows

    cos_values = np.array([float(row["latent_cos_to_oracle"]) for row in ar_rows], dtype=np.float64)
    quantiles = np.quantile(cos_values, [0.0, 0.25, 0.50, 0.75, 1.0])
    quantiles[0] -= 1e-12
    quantiles[-1] += 1e-12
    for metric_name, head in selected_heads.get("ar", {}).items():
        if metric_name not in {"action_top3", "noun_top3", "noun_recall5"}:
            continue
        head_rows = [row for row in ar_rows if int(row["head"]) == head]
        for idx in range(4):
            lo, hi = quantiles[idx], quantiles[idx + 1]
            bucket = [row for row in head_rows if lo < float(row["latent_cos_to_oracle"]) <= hi]
            n = len(bucket)
            if n == 0:
                continue
            rows.append(
                {
                    "selection": metric_name,
                    "head": head,
                    "bin": idx,
                    "cos_min": min(float(row["latent_cos_to_oracle"]) for row in bucket),
                    "cos_max": max(float(row["latent_cos_to_oracle"]) for row in bucket),
                    "samples": n,
                    "action_top3_rate": 100.0 * sum(int(row["action_top3_hit"]) for row in bucket) / n,
                    "action_top5_rate": 100.0 * sum(int(row["action_top5_hit"]) for row in bucket) / n,
                    "noun_top3_rate": 100.0 * sum(int(row["noun_top3_hit"]) for row in bucket) / n,
                    "noun_top5_rate": 100.0 * sum(int(row["noun_top5_hit"]) for row in bucket) / n,
                    "verb_top3_rate": 100.0 * sum(int(row["verb_top3_hit"]) for row in bucket) / n,
                    "verb_top5_rate": 100.0 * sum(int(row["verb_top5_hit"]) for row in bucket) / n,
                }
            )
    return rows


def _noun_prior_outputs(sample_rows: list[dict], head_rows: list[dict], selected_heads: dict[str, dict[str, int]]):
    head_metrics = {(row["method"], int(row["head"])): row for row in head_rows}
    summary_rows = []
    confusion_rows = []

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in sample_rows:
        grouped[(row["method"], int(row["head"]))].append(row)

    selected_tags: dict[tuple[str, int], list[str]] = defaultdict(list)
    for method, picks in selected_heads.items():
        for metric, head in picks.items():
            selected_tags[(method, int(head))].append(metric)

    for (method, head), rows in sorted(grouped.items()):
        labels = Counter(int(row["noun_label"]) for row in rows)
        preds = Counter(int(row["noun_top1"]) for row in rows)
        correct_preds = Counter(int(row["noun_top1"]) for row in rows if int(row["noun_top1_hit"]) == 1)
        top_pred = preds.most_common(10)
        top_pred_share = 100.0 * top_pred[0][1] / max(1, len(rows)) if top_pred else 0.0
        metric_row = head_metrics.get((method, head), {})
        summary_rows.append(
            {
                "method": method,
                "head": head,
                "selected_as": json.dumps(selected_tags.get((method, head), [])),
                "samples": len(rows),
                "noun_top1": metric_row.get("noun_top1", ""),
                "noun_top3": metric_row.get("noun_top3", ""),
                "noun_top5": metric_row.get("noun_top5", ""),
                "noun_recall5": metric_row.get("noun_recall5", ""),
                "label_entropy_nats": _entropy(labels),
                "pred_entropy_nats": _entropy(preds),
                "unique_label_nouns": len(labels),
                "unique_pred_nouns": len(preds),
                "top_pred_share": top_pred_share,
                "top_pred_nouns_json": json.dumps(top_pred),
            }
        )

        confusion = Counter((int(row["noun_label"]), int(row["noun_top1"])) for row in rows)
        for (label, pred), count in confusion.most_common(100):
            confusion_rows.append(
                {
                    "method": method,
                    "head": head,
                    "selected_as": json.dumps(selected_tags.get((method, head), [])),
                    "noun_label": label,
                    "noun_top1": pred,
                    "count": count,
                    "label_total": labels[label],
                    "pred_total": preds[pred],
                    "pred_top1_precision": 100.0 * correct_preds[pred] / max(1, preds[pred]),
                    "label_to_pred_share": 100.0 * count / max(1, labels[label]),
                }
            )
    return summary_rows, confusion_rows


def _head_selection_disagreement(head_rows: list[dict], selected_heads: dict[str, dict[str, int]]) -> list[dict]:
    rows = []
    for method in sorted({row["method"] for row in head_rows}):
        picks = selected_heads.get(method, {})
        rows.append(
            {
                "method": method,
                "action_top3_head": picks.get("action_top3", ""),
                "action_recall5_head": picks.get("action_recall5", ""),
                "verb_top3_head": picks.get("verb_top3", ""),
                "verb_recall5_head": picks.get("verb_recall5", ""),
                "noun_top3_head": picks.get("noun_top3", ""),
                "noun_recall5_head": picks.get("noun_recall5", ""),
                "action_top3_vs_noun_top3_same": int(picks.get("action_top3") == picks.get("noun_top3")),
                "action_top3_vs_action_recall5_same": int(picks.get("action_top3") == picks.get("action_recall5")),
                "noun_top3_vs_noun_recall5_same": int(picks.get("noun_top3") == picks.get("noun_recall5")),
            }
        )
    return rows


def _vjepa2_native_summary(head_rows: list[dict]) -> list[dict]:
    rows = []
    metric_keys = [
        "action_top3",
        "action_recall5",
        "verb_top3",
        "verb_recall5",
        "noun_top3",
        "noun_recall5",
    ]
    for method in sorted({row["method"] for row in head_rows}):
        subset = [row for row in head_rows if row["method"] == method]
        row_out = {
            "method": method,
            "horizon_sec": subset[0].get("horizon_sec", "") if subset else "",
            "metric_scope": subset[0].get("metric_scope", "") if subset else "",
            "metric_aggregation": "metric_wise_max",
        }
        for key in metric_keys:
            best = max(subset, key=lambda row: float(row[key]))
            row_out[key] = best[key]
            row_out[f"{key}_head"] = best["head"]
        rows.append(row_out)
    return rows


@torch.no_grad()
def run(args):
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["experiment"]["data"]
    annotations = filter_annotations(
        data_cfg["dataset"],
        data_cfg["base_path"],
        data_cfg["dataset_train"],
        data_cfg["dataset_val"],
        file_format=data_cfg.get("file_format", 1),
    )
    samples = _build_samples(annotations["val"])
    encoder, predictor = _load_encoder_predictor(cfg, device)
    classifiers = _load_classifiers(cfg, annotations, encoder.embed_dim, device)
    gaze_components = _build_gaze_components(cfg, classifiers, device)

    ds = FutureOracleDataset(
        samples=samples,
        horizon_sec=args.horizon,
        frames_per_clip=data_cfg["frames_per_clip"],
        fps=data_cfg["frames_per_second"],
        anticipation_point=tuple(data_cfg.get("val_anticipation_point", [0.0, 0.0])),
        resolution=data_cfg["resolution"],
        drop_incomplete_history=args.drop_incomplete_history,
        max_samples=args.max_samples or None,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )
    logger.info("Running diagnostics over %d samples at horizon %.3fs", len(ds), args.horizon)

    methods = [m.strip() for m in args.methods.replace(",", " ").split() if m.strip()]
    valid_labels = _valid_label_sets(annotations)
    metric_scope = str(args.metric_scope).lower()
    use_valid_filter = metric_scope == "filtered"
    logger.info("Metric scope: %s", metric_scope)
    trackers = {
        method: {
            head: {task: MetricTracker() for task in ["verb", "noun", "action"]}
            for head in range(len(classifiers))
        }
        for method in methods
    }
    sample_rows: list[dict] = []
    latent_rows: list[dict] = []
    sample_index = 0

    use_bfloat16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False)) and device.type == "cuda"
    gaze_mode = gaze_components.get("mode", "none")
    adapter = gaze_components.get("adapter")
    map_builder = gaze_components.get("map_builder")
    traj_loader = gaze_components.get("traj_loader")
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})

    for batch_idx, batch in enumerate(loader):
        observed = batch["observed"].to(device, non_blocking=True)
        oracle_clip = batch["oracle"].to(device, non_blocking=True)
        metadata = batch["metadata"]
        labels = _labels(batch, annotations, device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            if gaze_mode == "binary_input_adapter" and adapter is not None and map_builder is not None:
                obs_map = map_builder.build(observed, metadata)
                oracle_meta = [
                    {**m, "frame_indices": m.get("oracle_frame_indices", m.get("frame_indices"))}
                    for m in metadata
                ]
                oracle_map = map_builder.build(oracle_clip, oracle_meta)
                observed = adapter(observed, obs_map)
                oracle_clip = adapter(oracle_clip, oracle_map)

            observed_tokens = encoder(observed)
            observed_last = _last_layer(observed_tokens, encoder.embed_dim)
            oracle_tokens = encoder(oracle_clip)
            oracle_last = _last_layer(oracle_tokens, encoder.embed_dim)
            n_pred = (data_cfg["resolution"] // encoder.patch_size) ** 2
            n_pred *= max(int(wrapper_cfg.get("num_output_frames", 2)), encoder.tubelet_size) // encoder.tubelet_size
            observed_tail = observed_last[:, -n_pred:, :]
            oracle_target = oracle_last[:, -n_pred:, :]

            tokens_by_method = {}
            latent_by_method: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            ar_target = None
            if "encoder" in methods:
                tokens_by_method["encoder"] = observed_last
            if "observed_plus_observed_tail" in methods:
                tokens_by_method["observed_plus_observed_tail"] = torch.cat([observed_last, observed_tail], dim=1)
            if "oracle_target_only" in methods:
                tokens_by_method["oracle_target_only"] = oracle_target
            if "oracle" in methods:
                tokens_by_method["oracle"] = torch.cat([observed_last, oracle_target], dim=1)
            if "ar" in methods:
                ar_target, ar_info = _predict_ar(encoder, predictor, observed_tokens, args.horizon, cfg, device)
                if ar_target is None:
                    logger.warning("Skipping AR batch %d because status=%s", batch_idx, ar_info)
                else:
                    tokens_by_method["ar"] = torch.cat([observed_last, ar_target], dim=1)
                    mse = torch.mean((ar_target.float() - oracle_target.float()) ** 2, dim=(1, 2))
                    cos = torch.nn.functional.cosine_similarity(
                        ar_target.float().flatten(1),
                        oracle_target.float().flatten(1),
                        dim=1,
                    )
                    latent_by_method["ar"] = (mse.detach().cpu(), cos.detach().cpu())

            observed_tail_f = _flat(observed_tail)
            oracle_target_f = _flat(oracle_target)
            observed_oracle_cos = torch.nn.functional.cosine_similarity(observed_tail_f, oracle_target_f, dim=1)
            observed_oracle_mse = torch.mean((observed_tail.float() - oracle_target.float()) ** 2, dim=(1, 2))
            observed_tail_norm = torch.linalg.vector_norm(observed_tail_f, dim=1)
            oracle_target_norm = torch.linalg.vector_norm(oracle_target_f, dim=1)
            if ar_target is not None:
                ar_target_f = _flat(ar_target)
                ar_target_norm = torch.linalg.vector_norm(ar_target_f, dim=1)
                ar_oracle_cos = torch.nn.functional.cosine_similarity(ar_target_f, oracle_target_f, dim=1)
                ar_oracle_mse = torch.mean((ar_target.float() - oracle_target.float()) ** 2, dim=(1, 2))
                ar_observed_cos = torch.nn.functional.cosine_similarity(ar_target_f, observed_tail_f, dim=1)
                ar_observed_mse = torch.mean((ar_target.float() - observed_tail.float()) ** 2, dim=(1, 2))
            else:
                ar_target_norm = ar_oracle_cos = ar_oracle_mse = ar_observed_cos = ar_observed_mse = None

            for i, meta in enumerate(metadata):
                latent_row = {
                    "sample_index": sample_index + i,
                    "video_id": meta.get("video_id", ""),
                    "start_frame": meta.get("start_frame", ""),
                    "stop_frame": meta.get("stop_frame", ""),
                    "horizon_sec": args.horizon,
                    "metric_scope": metric_scope,
                    "observed_tail_norm": float(observed_tail_norm[i].detach().cpu()),
                    "oracle_target_norm": float(oracle_target_norm[i].detach().cpu()),
                    "observed_tail_to_oracle_cos": float(observed_oracle_cos[i].detach().cpu()),
                    "observed_tail_to_oracle_mse": float(observed_oracle_mse[i].detach().cpu()),
                    "ar_target_norm": "",
                    "ar_to_oracle_cos": "",
                    "ar_to_oracle_mse": "",
                    "ar_to_observed_tail_cos": "",
                    "ar_to_observed_tail_mse": "",
                }
                if ar_target is not None:
                    latent_row.update(
                        {
                            "ar_target_norm": float(ar_target_norm[i].detach().cpu()),
                            "ar_to_oracle_cos": float(ar_oracle_cos[i].detach().cpu()),
                            "ar_to_oracle_mse": float(ar_oracle_mse[i].detach().cpu()),
                            "ar_to_observed_tail_cos": float(ar_observed_cos[i].detach().cpu()),
                            "ar_to_observed_tail_mse": float(ar_observed_mse[i].detach().cpu()),
                        }
                    )
                latent_rows.append(latent_row)

            gaze_tokens_per_classifier = [None] * len(classifiers)
            if gaze_mode in {"rnn_fuse", "mlp_fuse"} and traj_loader is not None:
                for idx, classifier in enumerate(classifiers):
                    gaze_tokens_per_classifier[idx] = encode_gaze_tokens(
                        classifier,
                        metadata,
                        traj_loader,
                        device,
                        video_tokens=observed_last if traj_loader.use_video_tokens else None,
                    )

            label_cpu = {
                task: [int(x) for x in labels[task].detach().cpu().tolist()]
                for task in ["verb", "noun", "action"]
            }
            for method, tokens in tokens_by_method.items():
                for head, classifier in enumerate(classifiers):
                    outputs = call_classifier(classifier, tokens, gaze_tokens_per_classifier[head])
                    topk_by_task = {}
                    for task in ["verb", "noun", "action"]:
                        preds = []
                        scores = []
                        for i in range(outputs[task].shape[0]):
                            valid = valid_labels[task] if use_valid_filter else None
                            p, s = _topk(outputs[task][i], args.topk, valid)
                            preds.append(p)
                            scores.append(s)
                            trackers[method][head][task].update(label_cpu[task][i], p)
                        topk_by_task[task] = (preds, scores)

                    for i, meta in enumerate(metadata):
                        latent_mse = ""
                        latent_cos = ""
                        if method in latent_by_method:
                            latent_mse = float(latent_by_method[method][0][i])
                            latent_cos = float(latent_by_method[method][1][i])
                        row = {
                            "sample_index": sample_index + i,
                            "method": method,
                            "head": head,
                            "video_id": meta.get("video_id", ""),
                            "start_frame": meta.get("start_frame", ""),
                            "stop_frame": meta.get("stop_frame", ""),
                            "horizon_sec": args.horizon,
                            "metric_scope": metric_scope,
                            "latent_mse_to_oracle": latent_mse,
                            "latent_cos_to_oracle": latent_cos,
                        }
                        for task in ["verb", "noun", "action"]:
                            preds, scores = topk_by_task[task]
                            label = label_cpu[task][i]
                            row[f"{task}_label"] = label
                            row[f"{task}_top1"] = preds[i][0] if preds[i] else -1
                            row[f"{task}_top1_hit"] = int(label in preds[i][:1])
                            row[f"{task}_top3_hit"] = int(label in preds[i][:3])
                            row[f"{task}_top5_hit"] = int(label in preds[i][:5])
                            row[f"{task}_top{args.topk}"] = json.dumps(preds[i])
                            row[f"{task}_scores_top{args.topk}"] = json.dumps(scores[i])
                        sample_rows.append(row)

        sample_index += observed.size(0)
        if batch_idx % args.log_every == 0:
            logger.info("batch=%d samples=%d", batch_idx, sample_index)

    head_rows = []
    for method in methods:
        for head in range(len(classifiers)):
            row = {"method": method, "head": head, "horizon_sec": args.horizon, "metric_scope": metric_scope}
            for task in ["verb", "noun", "action"]:
                vals = trackers[method][head][task].values(valid_labels[task])
                for metric, value in vals.items():
                    row[f"{task}_{metric}"] = value
            head_rows.append(row)

    selected_heads = _select_heads(head_rows)
    disagreement_rows = _head_selection_disagreement(head_rows, selected_heads)
    native_summary_rows = _vjepa2_native_summary(head_rows)
    cosine_rows = _add_cosine_bins(sample_rows, selected_heads)
    noun_prior_rows, noun_confusion_rows = _noun_prior_outputs(sample_rows, head_rows, selected_heads)
    latent_summary_rows = _latent_summary(latent_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "head_metrics.csv", head_rows)
    _write_csv(args.out_dir / "head_selection_disagreement.csv", disagreement_rows)
    _write_csv(args.out_dir / "vjepa2_native_summary.csv", native_summary_rows)
    _write_csv(args.out_dir / "sample_hits.csv", sample_rows)
    _write_csv(args.out_dir / "ar_cosine_hit_bins.csv", cosine_rows)
    _write_csv(args.out_dir / "noun_prior_summary.csv", noun_prior_rows)
    _write_csv(args.out_dir / "noun_top1_confusions.csv", noun_confusion_rows)
    _write_csv(args.out_dir / "latent_stats.csv", latent_rows)
    _write_csv(args.out_dir / "latent_summary.csv", latent_summary_rows)

    summary = {
        "config": str(args.config),
        "tag": cfg.get("tag", ""),
        "horizon_sec": args.horizon,
        "metric_scope": metric_scope,
        "samples": sample_index,
        "methods": methods,
        "selected_heads": selected_heads,
        "metric_aggregation": "metric_wise_max",
        "outputs": {
            "head_metrics": "head_metrics.csv",
            "head_selection_disagreement": "head_selection_disagreement.csv",
            "vjepa2_native_summary": "vjepa2_native_summary.csv",
            "sample_hits": "sample_hits.csv",
            "ar_cosine_hit_bins": "ar_cosine_hit_bins.csv",
            "noun_prior_summary": "noun_prior_summary.csv",
            "noun_top1_confusions": "noun_top1_confusions.csv",
            "latent_stats": "latent_stats.csv",
            "latent_summary": "latent_summary.csv",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote diagnostics to %s", args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--methods", default="encoder,ar,oracle")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--drop-incomplete-history", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--metric-scope", choices=["native", "filtered"], default="native")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
