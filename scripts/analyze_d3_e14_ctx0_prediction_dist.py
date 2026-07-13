"""D3 E14: predicted-class distribution for ctx0 + direct_rope @ 10s.

Diagnoses whether low R@5 comes from prior collapse vs broad confusion.
Optionally compares against oracle_target_only rows from an existing sample_hits.csv.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation.future_latent_compare import (
    FutureOracleDataset,
    _build_gaze_components,
    _build_samples,
    _collate,
    _labels,
    _last_layer,
    _load_classifiers,
    _load_encoder_predictor,
    _predict_direct,
)
from app.hdepic_lora_action_anticipation.gaze_rnn import call_classifier
from evals.action_anticipation_frozen.dataloader import filter_annotations

logger = logging.getLogger("d3_e14_prediction_dist")


def _load_fm():
    path = Path(__file__).resolve().parent / "analyze_future_latent_failure_modes.py"
    spec = importlib.util.spec_from_file_location("fm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def _dist_summary(counter: Counter, num_classes: int, top_k: int = 15) -> dict:
    total = sum(counter.values())
    top = counter.most_common(top_k)
    return {
        "total": total,
        "unique_preds": len(counter),
        "entropy": _entropy(counter),
        "max_class_share": (top[0][1] / total) if total and top else 0.0,
        "top1_class": top[0][0] if top else -1,
        "top1_count": top[0][1] if top else 0,
        "top_preds": top,
        "num_classes": num_classes,
    }


def _load_baseline_top1(csv_path: Path, method: str, task: str, head: int | None) -> list[int]:
    preds = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("method") != method:
                continue
            if head is not None and int(row.get("head", -1)) != head:
                continue
            preds.append(int(row[f"{task}_top1"]))
    return preds


@torch.no_grad()
def collect_direct_ctx0_preds(args, cfg, device, fm):
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

    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    grid = int(data_cfg["resolution"] // encoder.patch_size)
    tubelet = int(encoder.tubelet_size)
    num_output_frames = max(int(wrapper_cfg.get("num_output_frames", 2)), tubelet)
    spatial = int(grid * grid * (num_output_frames // tubelet))

    ds = FutureOracleDataset(
        samples=samples,
        horizon_sec=args.horizon,
        frames_per_clip=data_cfg["frames_per_clip"],
        fps=float(data_cfg["frames_per_second"]),
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
    use_bf16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False)) and device.type == "cuda"

    per_head = {
        h: {t: [] for t in ["verb", "noun", "action"]}
        for h in range(len(classifiers))
    }
    rows_out = []

    for batch_idx, batch in enumerate(loader):
        observed = batch["observed"].to(device, non_blocking=True)
        metadata = batch["metadata"]
        labels = _labels(batch, annotations, device)
        label_cpu = {t: [int(x) for x in labels[t].detach().cpu().tolist()] for t in ["verb", "noun", "action"]}

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
            obs_tok = encoder(observed)
            obs_last = _last_layer(obs_tok, encoder.embed_dim)
            direct_target, info = _predict_direct(
                encoder,
                predictor,
                obs_tok,
                args.horizon,
                cfg,
                device,
                dense=False,
                rope_scale_mode="ntk_temporal",
            )
            if direct_target is None:
                raise RuntimeError(f"direct_rope failed: {info}")
            tokens = direct_target  # ctx0

            for head, clf in enumerate(classifiers):
                out = call_classifier(clf, tokens, None)
                for i, meta in enumerate(metadata):
                    row = {
                        "sample_index": batch_idx * args.batch_size + i,
                        "method": "direct_rope_ctx0",
                        "head": head,
                        "horizon_sec": args.horizon,
                        "video_id": meta.get("video_id", ""),
                    }
                    for task in ["verb", "noun", "action"]:
                        preds, scores = fm._topk(out[task][i], args.topk, None)
                        top1 = preds[0] if preds else -1
                        per_head[head][task].append(top1)
                        row[f"{task}_label"] = label_cpu[task][i]
                        row[f"{task}_top1"] = top1
                        row[f"{task}_top1_hit"] = int(label_cpu[task][i] in preds[:1])
                        row[f"{task}_top5_hit"] = int(label_cpu[task][i] in preds[:5])
                    rows_out.append(row)

        if batch_idx % args.log_every == 0:
            logger.info("batch=%d", batch_idx)

    return per_head, rows_out, {
        "verb": len(annotations["verbs"]),
        "noun": len(annotations["nouns"]),
        "action": len(annotations["actions"]),
    }


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _plot_distributions(out_dir: Path, summaries: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for task in ["action", "noun", "verb"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, (label, summ) in zip(axes, summaries[task].items()):
            top = summ["top_preds"][:12]
            if not top:
                continue
            classes, counts = zip(*top)
            ax.bar(range(len(classes)), counts, color="steelblue")
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
            ax.set_title(f"{label}\nH={summ['entropy']:.2f} max_share={summ['max_class_share']:.2%}")
            ax.set_ylabel("count")
        fig.suptitle(f"E14 top-1 prediction frequency ({task})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"top1_freq_{task}.png", dpi=150)
        plt.close(fig)


def run(args):
    fm = _load_fm()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    per_head, rows, num_classes = collect_direct_ctx0_preds(args, cfg, device, fm)
    _write_csv(args.out_dir / "sample_top1.csv", rows)

    summaries: dict[str, dict] = {t: {} for t in ["verb", "noun", "action"]}
    head_summaries = {}

    def _jsonable_summary(summ: dict) -> dict:
        out = dict(summ)
        out["top_preds"] = [[int(c), int(n)] for c, n in summ.get("top_preds", [])]
        return out

    for head, task_preds in per_head.items():
        head_summaries[str(head)] = {}
        for task, preds in task_preds.items():
            c = Counter(preds)
            summ = _dist_summary(c, num_classes[task])
            head_summaries[str(head)][task] = _jsonable_summary(summ)
            summaries[task][f"head_{head}"] = _jsonable_summary(summ)

    # metric-wise max head per task (vjepa2 style on top1 frequency diversity - report best head by recall proxy)
    if args.baseline_sample_hits and args.baseline_sample_hits.is_file():
        for task in ["verb", "noun", "action"]:
            for method, label in [
                (args.baseline_method, "baseline_oracle_target_only"),
            ]:
                preds = _load_baseline_top1(
                    args.baseline_sample_hits,
                    method,
                    task,
                    args.baseline_head,
                )
                if preds:
                    summaries[task][label] = _jsonable_summary(
                        _dist_summary(Counter(preds), num_classes[task])
                    )

    _plot_distributions(args.out_dir, summaries)

    out = {
        "experiment": "D3-E14-ctx0-prediction-distribution",
        "horizon_sec": args.horizon,
        "method": "direct_rope_ctx0",
        "per_head": head_summaries,
        "interpretation_hints": {
            "collapse": "max_class_share > 0.5 or entropy very low vs oracle baseline",
            "confused": "high unique_preds but low hit rate (see E7 R@5)",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Wrote E14 to %s", args.out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--horizon", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--drop-incomplete-history", action="store_true")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--baseline-sample-hits", type=Path, default=None)
    p.add_argument("--baseline-method", default="oracle_target_only")
    p.add_argument("--baseline-head", type=int, default=17)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
