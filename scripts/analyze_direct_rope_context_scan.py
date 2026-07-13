"""D3 E7: sweep observed context length with direct_single_rope future tokens.

Probe input is built from rope-scaled direct predictor target at fixed horizon.
Only observed prefix length changes, in units of temporal chunks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

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
from app.hdepic_lora_action_anticipation.gaze_rnn import call_classifier, encode_gaze_tokens
from evals.action_anticipation_frozen.dataloader import filter_annotations


def _load_failure_modes_module():
    path = Path(__file__).resolve().parent / "analyze_future_latent_failure_modes.py"
    spec = importlib.util.spec_from_file_location("analyze_future_latent_failure_modes", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load failure-modes helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fm = _load_failure_modes_module()
MetricTracker = _fm.MetricTracker
_select_heads = _fm._select_heads
_topk = _fm._topk
_valid_label_sets = _fm._valid_label_sets
_vjepa2_native_summary = _fm._vjepa2_native_summary
_write_csv = _fm._write_csv

logger = logging.getLogger("direct_rope_context_scan")

DEFAULT_CONTEXT_CHUNKS = (16, 8, 4, 2, 1, 0)


def _spatial_tokens_per_chunk(cfg: dict, encoder) -> int:
    data_cfg = cfg["experiment"]["data"]
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    grid = int(data_cfg["resolution"] // encoder.patch_size)
    tubelet = int(encoder.tubelet_size)
    num_output_frames = max(int(wrapper_cfg.get("num_output_frames", 2)), tubelet)
    return int(grid * grid * (num_output_frames // tubelet))


def _probe_tokens(observed_last: torch.Tensor, direct_target: torch.Tensor, context_chunks: int, spatial: int):
    if context_chunks <= 0:
        return direct_target
    prefix = observed_last[:, -context_chunks * spatial :, :]
    return torch.cat([prefix, direct_target], dim=1)


def _method_label(context_chunks: int) -> str:
    return "direct_rope_ctx0" if context_chunks <= 0 else f"direct_rope_ctx{context_chunks}"


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

    context_chunks_list = tuple(
        int(x) for x in str(args.context_chunks).replace(",", " ").split() if str(x).strip()
    )
    if not context_chunks_list:
        context_chunks_list = DEFAULT_CONTEXT_CHUNKS

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
    spatial = _spatial_tokens_per_chunk(cfg, encoder)
    metric_scope = str(args.metric_scope).lower()
    use_valid_filter = metric_scope == "filtered"
    valid_labels = _valid_label_sets(annotations)

    trackers = {
        k: {
            head: {task: MetricTracker() for task in ["verb", "noun", "action"]}
            for head in range(len(classifiers))
        }
        for k in context_chunks_list
    }
    sample_rows: list[dict] = []
    sample_index = 0
    use_bfloat16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False)) and device.type == "cuda"
    gaze_mode = gaze_components.get("mode", "none")
    adapter = gaze_components.get("adapter")
    map_builder = gaze_components.get("map_builder")
    traj_loader = gaze_components.get("traj_loader")

    logger.info(
        "E7 direct_rope context scan horizon=%.3fs chunks=%s spatial=%d samples=%d",
        args.horizon,
        context_chunks_list,
        spatial,
        len(ds),
    )

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
            direct_target, direct_info = _predict_direct(
                encoder,
                predictor,
                observed_tokens,
                args.horizon,
                cfg,
                device,
                dense=False,
                rope_scale_mode="ntk_temporal",
            )
            if direct_target is None:
                raise RuntimeError(f"E7 requires direct_rope target; got status={direct_info}")

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

            for context_chunks in context_chunks_list:
                method = _method_label(context_chunks)
                tokens = _probe_tokens(observed_last, direct_target, context_chunks, spatial)
                for head, classifier in enumerate(classifiers):
                    outputs = call_classifier(classifier, tokens, gaze_tokens_per_classifier[head])
                    for task in ["verb", "noun", "action"]:
                        for i in range(outputs[task].shape[0]):
                            valid = valid_labels[task] if use_valid_filter else None
                            preds, _ = _topk(outputs[task][i], args.topk, valid)
                            trackers[context_chunks][head][task].update(label_cpu[task][i], preds)

                    if args.write_sample_hits:
                        for i, meta in enumerate(metadata):
                            row = {
                                "sample_index": sample_index + i,
                                "method": method,
                                "context_chunks": context_chunks,
                                "head": head,
                                "horizon_sec": args.horizon,
                                "metric_scope": metric_scope,
                                "video_id": meta.get("video_id", ""),
                            }
                            for task in ["verb", "noun", "action"]:
                                valid = valid_labels[task] if use_valid_filter else None
                                preds, _ = _topk(outputs[task][i], args.topk, valid)
                                label = label_cpu[task][i]
                                row[f"{task}_label"] = label
                                row[f"{task}_top3_hit"] = int(label in preds[:3])
                                row[f"{task}_top5_hit"] = int(label in preds[:5])
                            sample_rows.append(row)

        sample_index += observed.size(0)
        if batch_idx % args.log_every == 0:
            logger.info("batch=%d samples=%d", batch_idx, sample_index)

    head_rows = []
    for context_chunks in context_chunks_list:
        method = _method_label(context_chunks)
        for head in range(len(classifiers)):
            row = {
                "method": method,
                "context_chunks": context_chunks,
                "head": head,
                "horizon_sec": args.horizon,
                "metric_scope": metric_scope,
            }
            for task in ["verb", "noun", "action"]:
                vals = trackers[context_chunks][head][task].values(valid_labels[task])
                for metric, value in vals.items():
                    row[f"{task}_{metric}"] = value
            head_rows.append(row)

    native_summary_rows = _vjepa2_native_summary(head_rows)
    curve_rows = []
    for row in native_summary_rows:
        method = row["method"]
        chunks = 0 if method == "direct_rope_ctx0" else int(method.removeprefix("direct_rope_ctx"))
        curve_rows.append(
            {
                "context_chunks": chunks,
                "method": method,
                "horizon_sec": row.get("horizon_sec", args.horizon),
                "metric_scope": row.get("metric_scope", metric_scope),
                "action_recall5": row.get("action_recall5"),
                "action_top3": row.get("action_top3"),
                "noun_recall5": row.get("noun_recall5"),
                "noun_top3": row.get("noun_top3"),
                "verb_recall5": row.get("verb_recall5"),
                "verb_top3": row.get("verb_top3"),
            }
        )
    curve_rows.sort(key=lambda r: r["context_chunks"], reverse=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "head_metrics.csv", head_rows)
    _write_csv(args.out_dir / "vjepa2_native_summary.csv", native_summary_rows)
    _write_csv(args.out_dir / "context_length_curve.csv", curve_rows)
    if args.write_sample_hits:
        _write_csv(args.out_dir / "sample_hits.csv", sample_rows)

    summary = {
        "experiment": "D3-E7-direct-rope-context-scan",
        "config": str(args.config),
        "tag": cfg.get("tag", ""),
        "horizon_sec": args.horizon,
        "metric_scope": metric_scope,
        "context_chunks": list(context_chunks_list),
        "spatial_tokens_per_chunk": spatial,
        "samples": sample_index,
        "selected_heads": _select_heads(head_rows),
        "outputs": {
            "context_length_curve": "context_length_curve.csv",
            "vjepa2_native_summary": "vjepa2_native_summary.csv",
            "head_metrics": "head_metrics.csv",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote E7 direct_rope context scan to %s", args.out_dir)
    for row in curve_rows:
        logger.info(
            "ctx=%2d action_r5=%.2f noun_r5=%.2f action_top3=%.2f noun_top3=%.2f",
            row["context_chunks"],
            float(row["action_recall5"] or 0),
            float(row["noun_recall5"] or 0),
            float(row["action_top3"] or 0),
            float(row["noun_top3"] or 0),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--context-chunks", default=",".join(str(x) for x in DEFAULT_CONTEXT_CHUNKS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--drop-incomplete-history", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--metric-scope", choices=["native", "filtered"], default="native")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--write-sample-hits", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
