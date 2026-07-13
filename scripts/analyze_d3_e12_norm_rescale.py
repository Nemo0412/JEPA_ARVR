"""D3 E12: norm rescaling for direct_rope targets before probe readout.

Variants:
  12a global — multiply by constant scale (default: oracle_mean_norm / direct_mean_norm)
  12b per_token — each sample chunk rescaled to match oracle chunk L2 norm
  12c per_dim — z-score using E11 global per-dim mean/std, remap to oracle stats

Evaluates ctx0 and ctx16 (configurable).
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

logger = logging.getLogger("d3_e12_norm_rescale")

RESCALE_MODES = ("none", "global", "per_token", "per_dim")


def _load_failure_modes_module():
    path = Path(__file__).resolve().parent / "analyze_future_latent_failure_modes.py"
    spec = importlib.util.spec_from_file_location("analyze_future_latent_failure_modes", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load failure-modes helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spatial_tokens_per_chunk(cfg: dict, encoder) -> int:
    data_cfg = cfg["experiment"]["data"]
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    grid = int(data_cfg["resolution"] // encoder.patch_size)
    tubelet = int(encoder.tubelet_size)
    num_output_frames = max(int(wrapper_cfg.get("num_output_frames", 2)), tubelet)
    return int(grid * grid * (num_output_frames // tubelet))


def _chunk_norm(tokens: torch.Tensor) -> torch.Tensor:
    flat = tokens.float().reshape(tokens.size(0), -1)
    return torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp(min=1e-8)


def _rescale_global(direct: torch.Tensor, scale: float) -> torch.Tensor:
    return direct * scale


def _rescale_per_token(direct: torch.Tensor, oracle: torch.Tensor) -> torch.Tensor:
    d_norm = _chunk_norm(direct)
    o_norm = _chunk_norm(oracle)
    ratio = (o_norm / d_norm).view(-1, 1, 1)
    return direct * ratio


def _rescale_per_dim(
    direct: torch.Tensor,
    mu_d: torch.Tensor,
    std_d: torch.Tensor,
    mu_o: torch.Tensor,
    std_o: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    x = direct.float()
    return ((x - mu_d) / (std_d + eps)) * std_o + mu_o


def _load_e11_stats(stats_dir: Path, device: torch.device):
    mu_o = torch.from_numpy(np.load(stats_dir / "per_dim_mean_oracle.npy")).to(device)
    std_o = torch.from_numpy(np.load(stats_dir / "per_dim_std_oracle.npy")).to(device)
    mu_d = torch.from_numpy(np.load(stats_dir / "per_dim_mean_direct.npy")).to(device)
    std_d = torch.from_numpy(np.load(stats_dir / "per_dim_std_direct.npy")).to(device)
    summary = json.loads((stats_dir / "summary.json").read_text(encoding="utf-8"))
    global_scale = float(summary["oracle_norm_mean"] / summary["direct_norm_mean"])
    return mu_o, std_o, mu_d, std_d, global_scale


def _probe_tokens(
    observed_last: torch.Tensor,
    future: torch.Tensor,
    context_chunks: int,
    spatial: int,
):
    if context_chunks <= 0:
        return future
    prefix = observed_last[:, -context_chunks * spatial :, :]
    return torch.cat([prefix, future], dim=1)


def _method_label(rescale: str, context_chunks: int) -> str:
    tag = "direct_rope" if rescale == "none" else f"direct_rope_{rescale}"
    return f"{tag}_ctx{context_chunks}"


def _parse_method(method: str) -> tuple[str, int]:
    if "_ctx" not in method:
        return "none", 0
    base, ctx_s = method.rsplit("_ctx", 1)
    ctx = int(ctx_s)
    if base == "direct_rope":
        return "none", ctx
    if base.startswith("direct_rope_"):
        return base.removeprefix("direct_rope_"), ctx
    return "none", ctx


@torch.no_grad()
def run(args):
    fm = _load_failure_modes_module()
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

    spatial = _spatial_tokens_per_chunk(cfg, encoder)
    context_list = tuple(int(x) for x in str(args.context_chunks).replace(",", " ").split() if x.strip())
    rescale_list = tuple(
        m.strip()
        for m in str(args.rescale_modes).replace(";", ",").split(",")
        if m.strip() in RESCALE_MODES
    )
    if not rescale_list:
        rescale_list = ("none", "global", "per_token", "per_dim")

    mu_o = std_o = mu_d = std_d = None
    global_scale = float(args.global_scale) if args.global_scale > 0 else None
    if any(m in rescale_list for m in ("global", "per_token", "per_dim")):
        if args.e11_stats_dir:
            mu_o, std_o, mu_d, std_d, gs = _load_e11_stats(args.e11_stats_dir, device)
            if global_scale is None:
                global_scale = gs
        elif global_scale is None:
            raise ValueError("Set --e11-stats-dir or --global-scale for rescaling modes")
    if global_scale is None:
        global_scale = 1.0

    if mu_o is not None:
        mu_o = mu_o.view(1, 1, -1)
        std_o = std_o.view(1, 1, -1)
        mu_d = mu_d.view(1, 1, -1)
        std_d = std_d.view(1, 1, -1)

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
    metric_scope = str(args.metric_scope).lower()
    use_valid_filter = metric_scope == "filtered"
    valid_labels = fm._valid_label_sets(annotations)

    trackers = {
        (r, c): {
            h: {t: fm.MetricTracker() for t in ["verb", "noun", "action"]}
            for h in range(len(classifiers))
        }
        for r in rescale_list
        for c in context_list
    }
    norm_after: dict[str, list[float]] = {r: [] for r in rescale_list if r != "none"}

    use_bf16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False)) and device.type == "cuda"
    gaze_mode = gaze_components.get("mode", "none")
    adapter = gaze_components.get("adapter")
    map_builder = gaze_components.get("map_builder")
    traj_loader = gaze_components.get("traj_loader")

    logger.info(
        "E12 horizon=%.3fs rescale=%s ctx=%s global_scale=%.4f samples=%d",
        args.horizon,
        rescale_list,
        context_list,
        global_scale,
        len(ds),
    )

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
            direct_target, direct_info = _predict_direct(
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
                raise RuntimeError(f"direct_rope failed: {direct_info}")

            targets_by_mode = {"none": direct_target}
            if "global" in rescale_list:
                targets_by_mode["global"] = _rescale_global(direct_target, global_scale)
            if "per_token" in rescale_list:
                targets_by_mode["per_token"] = _rescale_per_token(direct_target, oracle_target)
            if "per_dim" in rescale_list:
                targets_by_mode["per_dim"] = _rescale_per_dim(
                    direct_target, mu_d, std_d, mu_o, std_o
                ).to(direct_target.dtype)

            for mode in rescale_list:
                if mode != "none" and mode in targets_by_mode:
                    norm_after[mode].extend(_chunk_norm(targets_by_mode[mode]).squeeze(-1).cpu().tolist())

            gaze_per_head = [None] * len(classifiers)
            if gaze_mode in {"rnn_fuse", "mlp_fuse"} and traj_loader is not None:
                for idx, classifier in enumerate(classifiers):
                    gaze_per_head[idx] = encode_gaze_tokens(
                        classifier,
                        metadata,
                        traj_loader,
                        device,
                        video_tokens=obs_last if traj_loader.use_video_tokens else None,
                    )

            label_cpu = {
                t: [int(x) for x in labels[t].detach().cpu().tolist()]
                for t in ["verb", "noun", "action"]
            }

            for rescale in rescale_list:
                future = targets_by_mode[rescale]
                for ctx in context_list:
                    tokens = _probe_tokens(obs_last, future, ctx, spatial)
                    for head, clf in enumerate(classifiers):
                        out = call_classifier(clf, tokens, gaze_per_head[head])
                        for task in ["verb", "noun", "action"]:
                            for i in range(out[task].shape[0]):
                                valid = valid_labels[task] if use_valid_filter else None
                                preds, _ = fm._topk(out[task][i], args.topk, valid)
                                trackers[(rescale, ctx)][head][task].update(label_cpu[task][i], preds)

        if batch_idx % args.log_every == 0:
            logger.info("batch=%d", batch_idx)

    head_rows = []
    for rescale in rescale_list:
        for ctx in context_list:
            method = _method_label(rescale, ctx)
            for head in range(len(classifiers)):
                row = {
                    "method": method,
                    "rescale_mode": rescale,
                    "context_chunks": ctx,
                    "head": head,
                    "horizon_sec": args.horizon,
                    "metric_scope": metric_scope,
                    "global_scale": global_scale if rescale == "global" else "",
                }
                for task in ["verb", "noun", "action"]:
                    vals = trackers[(rescale, ctx)][head][task].values(valid_labels[task])
                    for metric, value in vals.items():
                        row[f"{task}_{metric}"] = value
                head_rows.append(row)

    native_rows = fm._vjepa2_native_summary(head_rows)
    curve_rows = []
    for row in native_rows:
        method = row["method"]
        rescale_tag, chunks = _parse_method(method)
        curve_rows.append(
            {
                "context_chunks": chunks,
                "method": method,
                "rescale_mode": rescale_tag,
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
    curve_rows.sort(key=lambda r: (r.get("rescale_mode", r["method"]), -r["context_chunks"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fm._write_csv(args.out_dir / "head_metrics.csv", head_rows)
    fm._write_csv(args.out_dir / "vjepa2_native_summary.csv", native_rows)
    fm._write_csv(args.out_dir / "rescale_context_curve.csv", curve_rows)

    norm_summary = {}
    for mode, vals in norm_after.items():
        if vals:
            norm_summary[mode] = {
                "post_rescale_norm_mean": float(np.mean(vals)),
                "post_rescale_norm_std": float(np.std(vals)),
            }

    summary = {
        "experiment": "D3-E12-norm-rescale",
        "horizon_sec": args.horizon,
        "global_scale": global_scale,
        "rescale_modes": list(rescale_list),
        "context_chunks": list(context_list),
        "e11_stats_dir": str(args.e11_stats_dir) if args.e11_stats_dir else "",
        "post_rescale_norm": norm_summary,
        "selected_heads": fm._select_heads(head_rows),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for row in curve_rows:
        logger.info(
            "%s ctx=%s action_r5=%.2f noun_top3=%.2f",
            row["method"],
            row["context_chunks"],
            float(row["action_recall5"] or 0),
            float(row["noun_top3"] or 0),
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--e11-stats-dir", type=Path, default=None)
    p.add_argument("--horizon", type=float, default=10.0)
    p.add_argument("--context-chunks", default="16,0")
    p.add_argument("--rescale-modes", default="none,global,per_token,per_dim")
    p.add_argument("--global-scale", type=float, default=0.0, help="Override; 0 = from E11 summary")
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
