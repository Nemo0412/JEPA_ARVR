"""Shared validation metric aggregation for HD-EPIC LoRA eval."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


def per_head_val_rows(
    action_metrics: list[dict],
    verb_metrics: list[dict] | None,
    noun_metrics: list[dict] | None,
) -> list[dict]:
    rows: list[dict] = []
    for head_idx, action_row in enumerate(action_metrics):
        row: dict[str, Any] = {
            "head": head_idx,
            "action_top1": float(action_row.get("top1_accuracy", action_row["accuracy"])),
            "action_top3": float(action_row["accuracy"]),
            "action_recall5": float(action_row["recall"]),
            "action_top5": float(action_row.get("top5_accuracy", action_row["accuracy"])),
        }
        if verb_metrics is not None and noun_metrics is not None:
            row["verb_top1"] = float(verb_metrics[head_idx].get("top1_accuracy", verb_metrics[head_idx]["accuracy"]))
            row["verb_top3"] = float(verb_metrics[head_idx]["accuracy"])
            row["verb_recall5"] = float(verb_metrics[head_idx]["recall"])
            row["verb_top5"] = float(verb_metrics[head_idx].get("top5_accuracy", verb_metrics[head_idx]["accuracy"]))
            row["noun_top1"] = float(noun_metrics[head_idx].get("top1_accuracy", noun_metrics[head_idx]["accuracy"]))
            row["noun_top3"] = float(noun_metrics[head_idx]["accuracy"])
            row["noun_recall5"] = float(noun_metrics[head_idx]["recall"])
            row["noun_top5"] = float(noun_metrics[head_idx].get("top5_accuracy", noun_metrics[head_idx]["accuracy"]))
        rows.append(row)
    return rows


def _metrics_from_head_row(row: dict) -> dict:
    return {
        "action": {
            "top1_accuracy": row["action_top1"],
            "accuracy": row["action_top3"],
            "recall": row["action_recall5"],
            "top5_accuracy": row["action_top5"],
        },
        "verb": {
            "top1_accuracy": row["verb_top1"],
            "accuracy": row["verb_top3"],
            "recall": row["verb_recall5"],
            "top5_accuracy": row["verb_top5"],
        },
        "noun": {
            "top1_accuracy": row["noun_top1"],
            "accuracy": row["noun_top3"],
            "recall": row["noun_recall5"],
            "top5_accuracy": row["noun_top5"],
        },
    }


def _max_metric_wise(
    action_metrics: list[dict],
    verb_metrics: list[dict] | None,
    noun_metrics: list[dict] | None,
) -> dict:
    ret = {
        "action": {
            "top1_accuracy": max(a.get("top1_accuracy", a["accuracy"]) for a in action_metrics),
            "accuracy": max(a["accuracy"] for a in action_metrics),
            "recall": max(a["recall"] for a in action_metrics),
            "top5_accuracy": max(a.get("top5_accuracy", a["accuracy"]) for a in action_metrics),
        },
    }
    if verb_metrics is not None and noun_metrics is not None:
        ret["verb"] = {
            "top1_accuracy": max(v.get("top1_accuracy", v["accuracy"]) for v in verb_metrics),
            "accuracy": max(v["accuracy"] for v in verb_metrics),
            "recall": max(v["recall"] for v in verb_metrics),
            "top5_accuracy": max(v.get("top5_accuracy", v["accuracy"]) for v in verb_metrics),
        }
        ret["noun"] = {
            "top1_accuracy": max(n.get("top1_accuracy", n["accuracy"]) for n in noun_metrics),
            "accuracy": max(n["accuracy"] for n in noun_metrics),
            "recall": max(n["recall"] for n in noun_metrics),
            "top5_accuracy": max(n.get("top5_accuracy", n["accuracy"]) for n in noun_metrics),
        }
    return ret


def summarize_metric_lists(
    action_metrics: list[dict],
    verb_metrics: list[dict] | None,
    noun_metrics: list[dict] | None,
) -> dict:
    """Summarize per-head metric logger rows for train/eval wrappers."""

    return _max_metric_wise(action_metrics, verb_metrics, noun_metrics)


def train_one_epoch_with_standard_model(
    base_eval,
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
):
    """Project-local copy of upstream standard training that preserves Top-1/5 metrics."""
    from app.hdepic_lora_action_anticipation.gaze import labels_from_udata

    _data_loader = iter(data_loader)
    for c in classifiers:
        c.train(mode=True)
    if action_is_verb_noun:
        verb_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers
        ]
        noun_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers
        ]
    action_metric_loggers = [
        base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers
    ]
    data_elapsed_time_meter = base_eval.AverageMeter()

    for itr in range(ipe):
        itr_start_time = time.time()

        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)

        [s.step() for s in scheduler]
        [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = udata[0].to(device)
            anticipation_times = udata[-1].to(device)
            labels = labels_from_udata(
                udata,
                device,
                action_is_verb_noun,
                verb_classes,
                noun_classes,
                action_classes,
            )
            data_elapsed_time_meter.update((time.time() - itr_start_time) * 1000.0)

            with torch.no_grad():
                tokens = model(clips, anticipation_times)
            outputs = [c(tokens) for c in classifiers]

        if action_is_verb_noun:
            verb_loss = [criterion(o["verb"], labels["verb"]) for o in outputs]
            noun_loss = [criterion(o["noun"], labels["noun"]) for o in outputs]
            action_loss = [criterion(o["action"], labels["action"]) for o in outputs]
            loss = [v + n + a for v, n, a in zip(verb_loss, noun_loss, action_loss)]
        else:
            loss = [criterion(o["action"], labels["action"]) for o in outputs]

        if use_bfloat16:
            [s.scale(l).backward() for s, l in zip(scaler, loss)]
            [s.step(o) for s, o in zip(scaler, optimizer)]
            [s.update() for s in scaler]
        else:
            [L.backward() for L in loss]
            [o.step() for o in optimizer]
        [o.zero_grad() for o in optimizer]

        with torch.no_grad():
            action_metrics = [
                m(o["action"], labels["action"]) for o, m in zip(outputs, action_metric_loggers)
            ]
            if action_is_verb_noun:
                verb_metrics = [
                    m(o["verb"], labels["verb"]) for o, m in zip(outputs, verb_metric_loggers)
                ]
                noun_metrics = [
                    m(o["noun"], labels["noun"]) for o, m in zip(outputs, noun_metric_loggers)
                ]
            else:
                verb_metrics = noun_metrics = None

        if itr % 10 == 0 or itr == ipe - 1:
            if action_is_verb_noun:
                logger.info(
                    "[%5d] acc (a/v/n): %.1f%% %.1f%% %.1f%% recall (a/v/n): %.1f%% %.1f%% %.1f%% [mem: %.2e] [data: %.1f ms]",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(v["accuracy"] for v in verb_metrics),
                    max(n["accuracy"] for n in noun_metrics),
                    max(a["recall"] for a in action_metrics),
                    max(v["recall"] for v in verb_metrics),
                    max(n["recall"] for n in noun_metrics),
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    data_elapsed_time_meter.avg,
                )
            else:
                logger.info(
                    "[%5d] acc: %.1f%% recall: %.1f%% [mem: %.2e] [data: %.1f ms]",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(a["recall"] for a in action_metrics),
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    data_elapsed_time_meter.avg,
                )

    del _data_loader
    return summarize_metric_lists(
        action_metrics,
        verb_metrics if action_is_verb_noun else None,
        noun_metrics if action_is_verb_noun else None,
    )


def summarize_val_metrics(
    action_metrics: list[dict],
    verb_metrics: list[dict] | None,
    noun_metrics: list[dict] | None,
    metric_scope: str,
    metric_aggregation: str = "metric_wise_max",
    val_fixed_head_index: int | None = None,
) -> dict:
    """Build validation summary for native/filtered scopes.

    ``metric_scope`` (logit masking only):
      - ``native``: full verb/noun/action vocab before top-k.
      - ``filtered`` / ``class_filtered``: mask logits to classes seen in the val split.

    ``metric_aggregation`` (how to combine the 20 classifier heads):
      - ``metric_wise_max``: each metric uses its own best head (V-JEPA2 default).
      - ``single_head`` (alias ``action_top3_single_head``): verb/noun/action all from one head,
        chosen by max action Top-3 under the current ``metric_scope`` (native or class_filtered).
        Optional ``val_fixed_head_index`` overrides the auto-picked head.
    """
    scope = str(metric_scope).lower()
    aggregation = str(metric_aggregation).lower()
    if aggregation == "action_top3_single_head":
        aggregation = "single_head"
    if scope in {"class_filtered"}:
        scope = "filtered"
    if scope not in {"native", "filtered"}:
        raise ValueError(f"Unsupported metric_scope={metric_scope!r}; expected native or filtered")
    if aggregation not in {"metric_wise_max", "single_head"}:
        raise ValueError(
            f"Unsupported metric_aggregation={metric_aggregation!r}; "
            "expected metric_wise_max or single_head"
        )

    rows = per_head_val_rows(action_metrics, verb_metrics, noun_metrics)
    ret: dict[str, Any] = {"metric_scope": scope, "metric_aggregation": aggregation}

    if aggregation == "metric_wise_max":
        ret.update(_max_metric_wise(action_metrics, verb_metrics, noun_metrics))
        return ret

    if verb_metrics is None or noun_metrics is None:
        raise ValueError("single_head aggregation requires verb/noun metrics for HD-EPIC")

    if val_fixed_head_index is not None:
        head_idx = int(val_fixed_head_index)
        if head_idx < 0 or head_idx >= len(rows):
            raise ValueError(f"val_fixed_head_index={head_idx} out of range for {len(rows)} classifier heads")
        best = rows[head_idx]
        head_pick = f"fixed_index_{head_idx}"
    else:
        best = max(rows, key=lambda r: r["action_top3"])
        head_idx = int(best["head"])
        head_pick = "action_top3_under_native" if scope == "native" else "action_top3_under_class_filtered"

    picked = _metrics_from_head_row(best)
    ret.update(picked)
    ret["single_head"] = {
        "head": head_idx,
        "head_pick": head_pick,
        **picked,
    }

    scope_label = "class_filtered" if scope == "filtered" else "native"
    logger.info(
        "[val] scope=%s aggregation=single_head head=%d (%s) | "
        "action Top-1/3/5/Recall@5=%.1f/%.1f/%.1f/%.1f "
        "verb Top-1/3/5/Recall@5=%.1f/%.1f/%.1f/%.1f "
        "noun Top-1/3/5/Recall@5=%.1f/%.1f/%.1f/%.1f",
        scope_label,
        head_idx,
        head_pick,
        best["action_top1"],
        best["action_top3"],
        best["action_top5"],
        best["action_recall5"],
        best["verb_top1"],
        best["verb_top3"],
        best["verb_top5"],
        best["verb_recall5"],
        best["noun_top1"],
        best["noun_top3"],
        best["noun_top5"],
        best["noun_recall5"],
    )
    return ret


@torch.no_grad()
def validate_with_standard_model(
    base_eval,
    dumper,
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    data_loader,
    use_bfloat16,
    valid_nouns,
    valid_verbs,
    valid_actions,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
    val_metric_scope: str = "native",
    val_metric_aggregation: str = "metric_wise_max",
    val_fixed_head_index: int | None = None,
):
    """Validation for the default frozen V-JEPA path (B1 clean baseline)."""
    from app.hdepic_lora_action_anticipation.gaze import labels_from_udata

    metric_scope = str(val_metric_scope).lower()
    use_valid_filter = metric_scope == "filtered"
    logger.info(
        "Running val with standard model (metric_scope=%s, aggregation=%s)...",
        metric_scope,
        val_metric_aggregation,
    )
    _data_loader = iter(data_loader)
    for c in classifiers:
        c.train(mode=False)
    if action_is_verb_noun:
        verb_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers
        ]
        noun_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers
        ]
    action_metric_loggers = [
        base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers
    ]

    for itr in range(ipe):
        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = udata[0].to(device)
            anticipation_times = udata[-1].to(device)
            labels = labels_from_udata(udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes)
            tokens = model(clips, anticipation_times)
            outputs = [c(tokens) for c in classifiers]
            valid_actions_arg = valid_actions if use_valid_filter else None
            valid_verbs_arg = valid_verbs if use_valid_filter else None
            valid_nouns_arg = valid_nouns if use_valid_filter else None
            action_metrics = [
                m(o["action"], labels["action"], valid_actions_arg) for o, m in zip(outputs, action_metric_loggers)
            ]
            if action_is_verb_noun:
                verb_metrics = [
                    m(o["verb"], labels["verb"], valid_verbs_arg) for o, m in zip(outputs, verb_metric_loggers)
                ]
                noun_metrics = [
                    m(o["noun"], labels["noun"], valid_nouns_arg) for o, m in zip(outputs, noun_metric_loggers)
                ]
                loss = sum(criterion(o["verb"], labels["verb"]) for o in outputs)
                loss = loss + sum(criterion(o["noun"], labels["noun"]) for o in outputs)
                loss = loss + sum(criterion(o["action"], labels["action"]) for o in outputs)
            else:
                verb_metrics = noun_metrics = None
                loss = sum(criterion(o["action"], labels["action"]) for o in outputs)
        best_head_idx = max(range(len(action_metrics)), key=lambda i: action_metrics[i]["accuracy"])
        dumper.add_batch(
            udata,
            [outputs[best_head_idx]],
            labels,
            {"verb": verb_classes, "noun": noun_classes, "action": action_classes},
        )
        if itr % 10 == 0 or itr == ipe - 1:
            if action_is_verb_noun:
                logger.info(
                    "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) loss (v/n): %.3f [mem: %.2e]",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(v["accuracy"] for v in verb_metrics),
                    max(n["accuracy"] for n in noun_metrics),
                    max(a["recall"] for a in action_metrics),
                    max(v["recall"] for v in verb_metrics),
                    max(n["recall"] for n in noun_metrics),
                    loss,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
    dumper.write()
    verb_arg = verb_metrics if action_is_verb_noun else None
    noun_arg = noun_metrics if action_is_verb_noun else None
    return summarize_val_metrics(
        action_metrics,
        verb_arg,
        noun_arg,
        metric_scope,
        metric_aggregation=val_metric_aggregation,
        val_fixed_head_index=val_fixed_head_index,
    )
