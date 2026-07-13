"""Check whether B5 binary input adapter changes future-latent diagnostics.

This is a removable HPC diagnostic. It compares the same decoded samples with
and without the binary input adapter, then writes compact JSON/CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation.future_latent_compare import (
    _build_gaze_components,
    _build_samples,
    _collate,
    _last_layer,
    _load_encoder_predictor,
    _predict_ar,
    FutureOracleDataset,
)
from evals.action_anticipation_frozen.dataloader import filter_annotations

logger = logging.getLogger("binary_adapter_latent_effect")


def _flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().flatten(1)


def _mean_abs(tensor: torch.Tensor) -> float:
    return float(tensor.float().abs().mean().detach().cpu())


def _max_abs(tensor: torch.Tensor) -> float:
    return float(tensor.float().abs().max().detach().cpu())


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(_flat(a), _flat(b), dim=1)


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a.float() - b.float()) ** 2, dim=tuple(range(1, a.ndim)))


def _sum(tensor: torch.Tensor) -> float:
    return float(tensor.float().sum().detach().cpu())


def _summ(values: list[float], prefix: str) -> dict:
    if not values:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_min": "",
            f"{prefix}_max": "",
        }
    return {
        f"{prefix}_mean": sum(values) / len(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    gaze_components = _build_gaze_components(cfg, classifiers=[], device=device)
    if gaze_components.get("mode") not in {"binary_input_adapter", "binary_input_adapter_gaze_pose_matrix"}:
        raise ValueError(
            f"Expected binary_input_adapter* config, got mode={gaze_components.get('mode')!r}"
        )
    adapter = gaze_components["adapter"]
    map_builder = gaze_components["map_builder"]
    if adapter is None or map_builder is None:
        raise RuntimeError("Binary adapter or map builder was not constructed")

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
    logger.info("Checking %d samples horizon=%.3fs", len(ds), args.horizon)

    use_bfloat16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False)) and device.type == "cuda"
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    n_pred = (data_cfg["resolution"] // encoder.patch_size) ** 2
    n_pred *= max(int(wrapper_cfg.get("num_output_frames", 2)), encoder.tubelet_size) // encoder.tubelet_size

    rows = []
    for batch_idx, batch in enumerate(loader):
        observed_raw = batch["observed"].to(device, non_blocking=True)
        oracle_raw = batch["oracle"].to(device, non_blocking=True)
        metadata = batch["metadata"]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            obs_map = map_builder.build(observed_raw, metadata)
            oracle_meta = [
                {**m, "frame_indices": m.get("oracle_frame_indices", m.get("frame_indices"))}
                for m in metadata
            ]
            oracle_map = map_builder.build(oracle_raw, oracle_meta)
            observed_adapted = adapter(observed_raw, obs_map)
            oracle_adapted = adapter(oracle_raw, oracle_map)

            obs_clean_tokens = encoder(observed_raw)
            obs_adapted_tokens = encoder(observed_adapted)
            oracle_clean_tokens = encoder(oracle_raw)
            oracle_adapted_tokens = encoder(oracle_adapted)

            obs_clean_last = _last_layer(obs_clean_tokens, encoder.embed_dim)
            obs_adapted_last = _last_layer(obs_adapted_tokens, encoder.embed_dim)
            oracle_clean_last = _last_layer(oracle_clean_tokens, encoder.embed_dim)
            oracle_adapted_last = _last_layer(oracle_adapted_tokens, encoder.embed_dim)

            obs_clean_tail = obs_clean_last[:, -n_pred:, :]
            obs_adapted_tail = obs_adapted_last[:, -n_pred:, :]
            oracle_clean_target = oracle_clean_last[:, -n_pred:, :]
            oracle_adapted_target = oracle_adapted_last[:, -n_pred:, :]

            ar_clean, ar_clean_info = _predict_ar(encoder, predictor, obs_clean_tokens, args.horizon, cfg, device)
            ar_adapted, ar_adapted_info = _predict_ar(encoder, predictor, obs_adapted_tokens, args.horizon, cfg, device)

            for i, meta in enumerate(metadata):
                row = {
                    "sample_index": len(rows),
                    "video_id": meta.get("video_id", ""),
                    "start_frame": meta.get("start_frame", ""),
                    "stop_frame": meta.get("stop_frame", ""),
                    "obs_map_nonzero_frac": float((obs_map[i] != 0).float().mean().detach().cpu()),
                    "obs_map_mean": _mean_abs(obs_map[i]),
                    "obs_map_max": _max_abs(obs_map[i]),
                    "obs_map_sum": _sum(obs_map[i]),
                    "oracle_map_nonzero_frac": float((oracle_map[i] != 0).float().mean().detach().cpu()),
                    "oracle_map_mean": _mean_abs(oracle_map[i]),
                    "oracle_map_max": _max_abs(oracle_map[i]),
                    "oracle_map_sum": _sum(oracle_map[i]),
                    "obs_adapter_delta_abs_mean": _mean_abs(observed_adapted[i] - observed_raw[i]),
                    "obs_adapter_delta_abs_max": _max_abs(observed_adapted[i] - observed_raw[i]),
                    "oracle_adapter_delta_abs_mean": _mean_abs(oracle_adapted[i] - oracle_raw[i]),
                    "oracle_adapter_delta_abs_max": _max_abs(oracle_adapted[i] - oracle_raw[i]),
                    "obs_token_clean_vs_adapted_cos": float(_cos(obs_clean_last[i : i + 1], obs_adapted_last[i : i + 1])[0].detach().cpu()),
                    "obs_token_clean_vs_adapted_mse": float(_mse(obs_clean_last[i : i + 1], obs_adapted_last[i : i + 1])[0].detach().cpu()),
                    "oracle_token_clean_vs_adapted_cos": float(_cos(oracle_clean_last[i : i + 1], oracle_adapted_last[i : i + 1])[0].detach().cpu()),
                    "oracle_token_clean_vs_adapted_mse": float(_mse(oracle_clean_last[i : i + 1], oracle_adapted_last[i : i + 1])[0].detach().cpu()),
                    "obs_tail_clean_vs_adapted_cos": float(_cos(obs_clean_tail[i : i + 1], obs_adapted_tail[i : i + 1])[0].detach().cpu()),
                    "obs_tail_clean_vs_adapted_mse": float(_mse(obs_clean_tail[i : i + 1], obs_adapted_tail[i : i + 1])[0].detach().cpu()),
                    "oracle_target_clean_vs_adapted_cos": float(_cos(oracle_clean_target[i : i + 1], oracle_adapted_target[i : i + 1])[0].detach().cpu()),
                    "oracle_target_clean_vs_adapted_mse": float(_mse(oracle_clean_target[i : i + 1], oracle_adapted_target[i : i + 1])[0].detach().cpu()),
                    "clean_obs_to_oracle_cos": float(_cos(obs_clean_tail[i : i + 1], oracle_clean_target[i : i + 1])[0].detach().cpu()),
                    "adapted_obs_to_oracle_cos": float(_cos(obs_adapted_tail[i : i + 1], oracle_adapted_target[i : i + 1])[0].detach().cpu()),
                    "ar_clean_status": ar_clean_info.get("status"),
                    "ar_adapted_status": ar_adapted_info.get("status"),
                }
                if ar_clean is not None and ar_adapted is not None:
                    row.update(
                        {
                            "ar_clean_vs_adapted_cos": float(_cos(ar_clean[i : i + 1], ar_adapted[i : i + 1])[0].detach().cpu()),
                            "ar_clean_vs_adapted_mse": float(_mse(ar_clean[i : i + 1], ar_adapted[i : i + 1])[0].detach().cpu()),
                            "ar_clean_to_oracle_cos": float(_cos(ar_clean[i : i + 1], oracle_clean_target[i : i + 1])[0].detach().cpu()),
                            "ar_adapted_to_oracle_cos": float(_cos(ar_adapted[i : i + 1], oracle_adapted_target[i : i + 1])[0].detach().cpu()),
                        }
                    )
                rows.append(row)

        if batch_idx % args.log_every == 0:
            logger.info("batch=%d rows=%d", batch_idx, len(rows))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "adapter_latent_effect.csv", rows)
    summary = {
        "config": str(args.config),
        "tag": cfg.get("tag", ""),
        "binary_map_type": cfg.get("experiment", {}).get("lora", {}).get("gaze", {}).get("binary_map_type", "binary"),
        "horizon_sec": args.horizon,
        "samples": len(rows),
        "output": "adapter_latent_effect.csv",
    }
    numeric_keys = [
        "obs_map_nonzero_frac",
        "obs_map_mean",
        "obs_map_max",
        "obs_map_sum",
        "oracle_map_nonzero_frac",
        "oracle_map_mean",
        "oracle_map_max",
        "oracle_map_sum",
        "obs_adapter_delta_abs_mean",
        "oracle_adapter_delta_abs_mean",
        "obs_token_clean_vs_adapted_cos",
        "obs_token_clean_vs_adapted_mse",
        "oracle_token_clean_vs_adapted_cos",
        "oracle_token_clean_vs_adapted_mse",
        "obs_tail_clean_vs_adapted_cos",
        "obs_tail_clean_vs_adapted_mse",
        "oracle_target_clean_vs_adapted_cos",
        "oracle_target_clean_vs_adapted_mse",
        "clean_obs_to_oracle_cos",
        "adapted_obs_to_oracle_cos",
        "ar_clean_vs_adapted_cos",
        "ar_clean_vs_adapted_mse",
        "ar_clean_to_oracle_cos",
        "ar_adapted_to_oracle_cos",
    ]
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
        summary.update(_summ(vals, key))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--drop-incomplete-history", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
