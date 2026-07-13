"""D3 E17: ctx16 future-token replacement ablation @ 10s (+ ctx0 1s/10s direct diagnostic).

Measures how much probe @ ctx16 relies on direct_rope future tokens vs context-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
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
from app.hdepic_lora_action_anticipation.gaze_rnn import call_classifier, encode_gaze_tokens
from evals.action_anticipation_frozen.dataloader import filter_annotations

logger = logging.getLogger("d3_e17_junk_future")


def _load_fm():
    path = Path(__file__).resolve().parent / "analyze_future_latent_failure_modes.py"
    spec = importlib.util.spec_from_file_location("fm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _spatial(cfg, encoder) -> int:
    data_cfg = cfg["experiment"]["data"]
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    grid = int(data_cfg["resolution"] // encoder.patch_size)
    tubelet = int(encoder.tubelet_size)
    n_out = max(int(wrapper_cfg.get("num_output_frames", 2)), tubelet)
    return int(grid * grid * (n_out // tubelet))


def _load_e11_gaussian_stats(stats_dir: Path | None, device: torch.device):
    if stats_dir is None:
        return None, None
    mu = torch.from_numpy(np.load(stats_dir / "per_dim_mean_oracle.npy")).to(device)
    std = torch.from_numpy(np.load(stats_dir / "per_dim_std_oracle.npy")).to(device)
    return mu.view(1, 1, -1), std.view(1, 1, -1)


def _build_future(
    mode: str,
    observed_last: torch.Tensor,
    oracle_target: torch.Tensor,
    direct_10s: torch.Tensor,
    direct_1s: torch.Tensor,
    mu_o: torch.Tensor | None,
    std_o: torch.Tensor | None,
) -> torch.Tensor:
    if mode == "direct_rope_10s":
        return direct_10s
    if mode == "oracle_target":
        return oracle_target
    if mode == "zero":
        return torch.zeros_like(direct_10s)
    if mode == "gaussian":
        if mu_o is None or std_o is None:
            raise ValueError("gaussian mode requires --e11-stats-dir")
        return torch.randn_like(direct_10s) * std_o + mu_o
    if mode == "unrelated_oracle":
        # Permute oracle targets within batch (same marginals, wrong video pairing)
        perm = torch.randperm(oracle_target.size(0), device=oracle_target.device)
        return oracle_target[perm]
    if mode == "observed_tail_repeat":
        return observed_last[:, -direct_10s.size(1) :, :]
    if mode == "direct_rope_1s":
        return direct_1s
    raise ValueError(f"Unknown future mode: {mode}")


@torch.no_grad()
def run(args):
    fm = _load_fm()
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
    spatial = _spatial(cfg, encoder)
    mu_o, std_o = _load_e11_gaussian_stats(args.e11_stats_dir, device)

    label_horizon = float(args.horizon)
    modes = [m.strip() for m in str(args.future_modes).replace(";", ",").split(",") if m.strip()]
    context_list = tuple(int(x) for x in str(args.context_chunks).replace(",", " ").split() if x.strip())

    ds = FutureOracleDataset(
        samples=samples,
        horizon_sec=label_horizon,
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

    metric_scope = str(args.metric_scope).lower()
    use_valid_filter = metric_scope == "filtered"
    valid_labels = fm._valid_label_sets(annotations)
    use_bf16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False)) and device.type == "cuda"
    gaze_mode = gaze_components.get("mode", "none")
    adapter = gaze_components.get("adapter")
    map_builder = gaze_components.get("map_builder")
    traj_loader = gaze_components.get("traj_loader")

    trackers = {
        (m, c): {h: {t: fm.MetricTracker() for t in ["verb", "noun", "action"]} for h in range(len(classifiers))}
        for m in modes
        for c in context_list
    }

    logger.info("E17 label_horizon=%.3fs modes=%s ctx=%s samples=%d", label_horizon, modes, context_list, len(ds))

    for batch_idx, batch in enumerate(loader):
        observed = batch["observed"].to(device, non_blocking=True)
        oracle_clip = batch["oracle"].to(device, non_blocking=True)
        metadata = batch["metadata"]
        labels = _labels(batch, annotations, device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
            if gaze_mode == "binary_input_adapter" and adapter is not None and map_builder is not None:
                obs_map = map_builder.build(observed, metadata)
                oracle_meta = [
                    {**m, "frame_indices": m.get("oracle_frame_indices", m.get("frame_indices"))}
                    for m in metadata
                ]
                oracle_map = map_builder.build(oracle_clip, oracle_meta)
                observed = adapter(observed, obs_map)
                oracle_clip = adapter(oracle_clip, oracle_map)

            obs_tok = encoder(observed)
            obs_last = _last_layer(obs_tok, encoder.embed_dim)
            ora_tok = encoder(oracle_clip)
            ora_last = _last_layer(ora_tok, encoder.embed_dim)
            oracle_target = ora_last[:, -spatial:, :]

            direct_10s, info10 = _predict_direct(
                encoder, predictor, obs_tok, label_horizon, cfg, device, dense=False, rope_scale_mode="ntk_temporal"
            )
            if direct_10s is None:
                raise RuntimeError(f"direct 10s failed: {info10}")
            direct_1s, info1 = _predict_direct(
                encoder, predictor, obs_tok, 1.0, cfg, device, dense=False, rope_scale_mode="ntk_temporal"
            )
            if direct_1s is None:
                raise RuntimeError(f"direct 1s failed: {info1}")

            gaze_per_head = [None] * len(classifiers)
            if gaze_mode in {"rnn_fuse", "mlp_fuse"} and traj_loader is not None:
                for idx, classifier in enumerate(classifiers):
                    gaze_per_head[idx] = encode_gaze_tokens(
                        classifier, metadata, traj_loader, device,
                        video_tokens=obs_last if traj_loader.use_video_tokens else None,
                    )

            label_cpu = {t: [int(x) for x in labels[t].detach().cpu().tolist()] for t in ["verb", "noun", "action"]}

            for mode in modes:
                future = _build_future(mode, obs_last, oracle_target, direct_10s, direct_1s, mu_o, std_o)
                for ctx in context_list:
                    if ctx <= 0:
                        tokens = future
                    else:
                        prefix = obs_last[:, -ctx * spatial :, :]
                        tokens = torch.cat([prefix, future], dim=1)
                    method = f"{mode}_ctx{ctx}"
                    for head, clf in enumerate(classifiers):
                        out = call_classifier(clf, tokens, gaze_per_head[head])
                        for task in ["verb", "noun", "action"]:
                            for i in range(out[task].shape[0]):
                                valid = valid_labels[task] if use_valid_filter else None
                                preds, _ = fm._topk(out[task][i], args.topk, valid)
                                trackers[(mode, ctx)][head][task].update(label_cpu[task][i], preds)

        if batch_idx % args.log_every == 0:
            logger.info("batch=%d", batch_idx)

    head_rows = []
    for mode in modes:
        for ctx in context_list:
            method = f"{mode}_ctx{ctx}"
            for head in range(len(classifiers)):
                row = {
                    "method": method,
                    "future_mode": mode,
                    "context_chunks": ctx,
                    "label_horizon_sec": label_horizon,
                    "head": head,
                    "horizon_sec": label_horizon,
                    "metric_scope": metric_scope,
                }
                for task in ["verb", "noun", "action"]:
                    vals = trackers[(mode, ctx)][head][task].values(valid_labels[task])
                    for metric, value in vals.items():
                        row[f"{task}_{metric}"] = value
                head_rows.append(row)

    native_rows = fm._vjepa2_native_summary(head_rows)
    curve_rows = []
    for row in native_rows:
        curve_rows.append(
            {
                "method": row["method"],
                "future_mode": row.get("future_mode", ""),
                "context_chunks": row.get("context_chunks", 16),
                "label_horizon_sec": label_horizon,
                "horizon_sec": label_horizon,
                "metric_scope": metric_scope,
                "action_recall5": row.get("action_recall5"),
                "action_top3": row.get("action_top3"),
                "noun_recall5": row.get("noun_recall5"),
                "noun_top3": row.get("noun_top3"),
                "verb_recall5": row.get("verb_recall5"),
                "verb_top3": row.get("verb_top3"),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fm._write_csv(args.out_dir / "head_metrics.csv", head_rows)
    fm._write_csv(args.out_dir / "vjepa2_native_summary.csv", native_rows)
    fm._write_csv(args.out_dir / "junk_future_curve.csv", curve_rows)

    summary = {
        "experiment": "D3-E17-junk-future-ctx16",
        "label_horizon_sec": label_horizon,
        "future_modes": modes,
        "context_chunks": list(context_list),
        "selected_heads": fm._select_heads(head_rows),
        "reference_action_r5": {
            "oracle_ctx16": 9.551,
            "direct_rope_ctx16": 8.156,
            "direct_rope_ctx0": 1.900,
            "observed_plus_tail_noun_top3_10s": 27.11,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for row in curve_rows:
        if int(row.get("context_chunks", -1)) == 16:
            logger.info(
                "%s action_r5=%.2f noun_top3=%.2f",
                row["method"],
                float(row["action_recall5"] or 0),
                float(row["noun_top3"] or 0),
            )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--e11-stats-dir", type=Path, default=None)
    p.add_argument("--horizon", type=float, default=10.0, help="Label / dataset anticipation horizon")
    p.add_argument(
        "--future-modes",
        default="direct_rope_10s,oracle_target,zero,gaussian,unrelated_oracle,observed_tail_repeat,direct_rope_1s",
    )
    p.add_argument("--context-chunks", default="16,0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--drop-incomplete-history", action="store_true")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--metric-scope", choices=["native", "filtered"], default="native")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
