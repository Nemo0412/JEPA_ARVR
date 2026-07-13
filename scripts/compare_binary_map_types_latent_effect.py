"""Compare binary vs distance gaze maps on latent change (same adapter checkpoint).

For each validation sample, builds both map types with the same gaze centers,
runs the binary input adapter + frozen encoder/predictor, and reports how much
clean latents move under each map type at a chosen anticipation horizon.
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

from app.hdepic_lora_action_anticipation.binary_map_utils import normalize_map_type
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

logger = logging.getLogger("compare_binary_map_types_latent_effect")

MAP_TYPES = ("binary", "distance")


def _flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().flatten(1)


def _mean_abs(tensor: torch.Tensor) -> float:
    return float(tensor.float().abs().mean().detach().cpu())


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(_flat(a), _flat(b), dim=1)


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a.float() - b.float()) ** 2, dim=tuple(range(1, a.ndim)))


def _summ(values: list[float], prefix: str) -> dict:
    if not values:
        return {f"{prefix}_mean": "", f"{prefix}_min": "", f"{prefix}_max": ""}
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


def _build_maps(map_builder, clips, metadata, map_type: str):
    prev = map_builder.map_type
    map_builder.map_type = normalize_map_type(map_type)
    try:
        return map_builder.build(clips, metadata)
    finally:
        map_builder.map_type = prev


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
    if gaze_components.get("mode") != "binary_input_adapter":
        raise ValueError(f"Expected binary_input_adapter config, got mode={gaze_components.get('mode')!r}")
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
    logger.info("Comparing map types on %d samples horizon=%.3fs", len(ds), args.horizon)

    use_bfloat16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False)) and device.type == "cuda"
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    n_pred = (data_cfg["resolution"] // encoder.patch_size) ** 2
    n_pred *= max(int(wrapper_cfg.get("num_output_frames", 2)), encoder.tubelet_size) // encoder.tubelet_size

    rows = []
    for batch_idx, batch in enumerate(loader):
        observed_raw = batch["observed"].to(device, non_blocking=True)
        oracle_raw = batch["oracle"].to(device, non_blocking=True)
        metadata = batch["metadata"]
        oracle_meta = [
            {**m, "frame_indices": m.get("oracle_frame_indices", m.get("frame_indices"))}
            for m in metadata
        ]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            obs_clean_tokens = encoder(observed_raw)
            oracle_clean_tokens = encoder(oracle_raw)
            obs_clean_last = _last_layer(obs_clean_tokens, encoder.embed_dim)
            oracle_clean_last = _last_layer(oracle_clean_tokens, encoder.embed_dim)
            obs_clean_tail = obs_clean_last[:, -n_pred:, :]
            oracle_clean_target = oracle_clean_last[:, -n_pred:, :]
            ar_clean, ar_clean_info = _predict_ar(encoder, predictor, obs_clean_tokens, args.horizon, cfg, device)

            per_type = {}
            for map_type in MAP_TYPES:
                obs_map = _build_maps(map_builder, observed_raw, metadata, map_type)
                oracle_map = _build_maps(map_builder, oracle_raw, oracle_meta, map_type)
                observed_adapted = adapter(observed_raw, obs_map)
                oracle_adapted = adapter(oracle_raw, oracle_map)
                obs_adapted_tokens = encoder(observed_adapted)
                oracle_adapted_tokens = encoder(oracle_adapted)
                obs_adapted_last = _last_layer(obs_adapted_tokens, encoder.embed_dim)
                oracle_adapted_last = _last_layer(oracle_adapted_tokens, encoder.embed_dim)
                obs_adapted_tail = obs_adapted_last[:, -n_pred:, :]
                oracle_adapted_target = oracle_adapted_last[:, -n_pred:, :]
                ar_adapted, ar_adapted_info = _predict_ar(encoder, predictor, obs_adapted_tokens, args.horizon, cfg, device)
                per_type[map_type] = {
                    "obs_map": obs_map,
                    "observed_adapted": observed_adapted,
                    "obs_adapted_last": obs_adapted_last,
                    "obs_adapted_tail": obs_adapted_tail,
                    "oracle_adapted_target": oracle_adapted_target,
                    "ar_adapted": ar_adapted,
                    "ar_adapted_info": ar_adapted_info,
                }

            for i, meta in enumerate(metadata):
                row = {
                    "sample_index": len(rows),
                    "video_id": meta.get("video_id", ""),
                    "start_frame": meta.get("start_frame", ""),
                    "stop_frame": meta.get("stop_frame", ""),
                    "ar_clean_status": ar_clean_info.get("status"),
                }
                for map_type in MAP_TYPES:
                    pack = per_type[map_type]
                    prefix = map_type
                    obs_map = pack["obs_map"]
                    observed_adapted = pack["observed_adapted"]
                    row.update(
                        {
                            f"{prefix}_obs_map_nonzero_frac": float((obs_map[i] != 0).float().mean().detach().cpu()),
                            f"{prefix}_obs_map_mean": _mean_abs(obs_map[i]),
                            f"{prefix}_obs_adapter_delta_abs_mean": _mean_abs(observed_adapted[i] - observed_raw[i]),
                            f"{prefix}_obs_token_clean_vs_adapted_cos": float(
                                _cos(obs_clean_last[i : i + 1], pack["obs_adapted_last"][i : i + 1])[0].detach().cpu()
                            ),
                            f"{prefix}_obs_token_clean_vs_adapted_mse": float(
                                _mse(obs_clean_last[i : i + 1], pack["obs_adapted_last"][i : i + 1])[0].detach().cpu()
                            ),
                            f"{prefix}_obs_tail_clean_vs_adapted_cos": float(
                                _cos(obs_clean_tail[i : i + 1], pack["obs_adapted_tail"][i : i + 1])[0].detach().cpu()
                            ),
                            f"{prefix}_obs_tail_clean_vs_adapted_mse": float(
                                _mse(obs_clean_tail[i : i + 1], pack["obs_adapted_tail"][i : i + 1])[0].detach().cpu()
                            ),
                            f"{prefix}_adapted_obs_to_oracle_cos": float(
                                _cos(pack["obs_adapted_tail"][i : i + 1], pack["oracle_adapted_target"][i : i + 1])[0].detach().cpu()
                            ),
                        }
                    )
                    ar_adapted = pack["ar_adapted"]
                    row[f"{prefix}_ar_adapted_status"] = pack["ar_adapted_info"].get("status")
                    if ar_clean is not None and ar_adapted is not None:
                        row[f"{prefix}_ar_clean_vs_adapted_cos"] = float(
                            _cos(ar_clean[i : i + 1], ar_adapted[i : i + 1])[0].detach().cpu()
                        )
                        row[f"{prefix}_ar_clean_vs_adapted_mse"] = float(
                            _mse(ar_clean[i : i + 1], ar_adapted[i : i + 1])[0].detach().cpu()
                        )
                        row[f"{prefix}_ar_adapted_to_oracle_cos"] = float(
                            _cos(ar_adapted[i : i + 1], pack["oracle_adapted_target"][i : i + 1])[0].detach().cpu()
                        )

                row["binary_vs_distance_obs_token_cos"] = float(
                    _cos(per_type["binary"]["obs_adapted_last"][i : i + 1], per_type["distance"]["obs_adapted_last"][i : i + 1])[
                        0
                    ].detach().cpu()
                )
                row["binary_vs_distance_obs_token_mse"] = float(
                    _mse(per_type["binary"]["obs_adapted_last"][i : i + 1], per_type["distance"]["obs_adapted_last"][i : i + 1])[
                        0
                    ].detach().cpu()
                )
                row["binary_vs_distance_obs_tail_mse"] = float(
                    _mse(per_type["binary"]["obs_adapted_tail"][i : i + 1], per_type["distance"]["obs_adapted_tail"][i : i + 1])[
                        0
                    ].detach().cpu()
                )
                if per_type["binary"]["ar_adapted"] is not None and per_type["distance"]["ar_adapted"] is not None:
                    row["binary_vs_distance_ar_mse"] = float(
                        _mse(per_type["binary"]["ar_adapted"][i : i + 1], per_type["distance"]["ar_adapted"][i : i + 1])[
                            0
                        ].detach().cpu()
                    )
                    row["binary_vs_distance_ar_cos"] = float(
                        _cos(per_type["binary"]["ar_adapted"][i : i + 1], per_type["distance"]["ar_adapted"][i : i + 1])[
                            0
                        ].detach().cpu()
                    )
                rows.append(row)

        if batch_idx % args.log_every == 0:
            logger.info("batch=%d rows=%d", batch_idx, len(rows))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "map_type_latent_compare.csv", rows)

    summary = {
        "config": str(args.config),
        "tag": cfg.get("tag", ""),
        "trained_binary_map_type": cfg.get("experiment", {}).get("lora", {}).get("gaze", {}).get("binary_map_type", "binary"),
        "horizon_sec": args.horizon,
        "samples": len(rows),
        "map_types_compared": list(MAP_TYPES),
    }
    metric_keys = [k for k in rows[0] if k not in {"sample_index", "video_id", "start_frame", "stop_frame"} and not k.endswith("_status")]
    for key in sorted(metric_keys):
        vals = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
        summary.update(_summ(vals, key))

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all filtered val samples")
    parser.add_argument("--drop-incomplete-history", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.max_samples == 0:
        args.max_samples = None
    run(args)


if __name__ == "__main__":
    main()
