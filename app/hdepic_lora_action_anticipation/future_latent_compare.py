"""Compare encoder, predictor, and oracle future latents on HD-EPIC.

This is a standalone validation tool so the oracle branch can decode a second
future clip for the same action sample. It intentionally lives outside
``vjepa2/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset

from app.hdepic_lora_action_anticipation.eval import Top3AccuracyRecallAt5, _make_lora_init_classifier
from evals.action_anticipation_frozen.dataloader import filter_annotations, make_transforms
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger("future_latent_compare")


@dataclass
class FutureSample:
    video_path: str
    video_id: str
    start_frame: int
    stop_frame: int
    verb_raw: int
    noun_raw: int


class FutureOracleDataset(Dataset):
    def __init__(
        self,
        samples: list[FutureSample],
        horizon_sec: float,
        frames_per_clip: int,
        fps: int,
        anticipation_point: tuple[float, float],
        resolution: int,
        drop_incomplete_history: bool,
        max_samples: int | None = None,
    ):
        self.samples = samples[:max_samples] if max_samples else samples
        self.horizon_sec = float(horizon_sec)
        self.frames_per_clip = int(frames_per_clip)
        self.fps = int(fps)
        self.anticipation_point = anticipation_point
        self.transform = make_transforms(training=False, crop_size=resolution)
        self.drop_incomplete_history = bool(drop_incomplete_history)
        if self.drop_incomplete_history:
            self.samples = self._filter_full_history(self.samples)

    def _filter_full_history(self, samples: list[FutureSample]) -> list[FutureSample]:
        kept = []
        for sample in samples:
            try:
                vr = VideoReader(sample.video_path, num_threads=1, ctx=cpu(0))
                vfps = float(vr.get_avg_fps())
            except Exception as exc:
                logger.info("Skipping unreadable video during history filter: %s error=%r", sample.video_path, exc)
                continue
            frame_step = max(1, int(vfps / self.fps))
            nframes = int(self.frames_per_clip * frame_step)
            anchor = self._anchor_frame(sample)
            observed_end = anchor - int(self.horizon_sec * vfps)
            if observed_end - nframes >= 0:
                kept.append(sample)
        logger.info(
            "Horizon %.3fs full-history filter kept %d/%d samples",
            self.horizon_sec,
            len(kept),
            len(samples),
        )
        return kept

    def _anchor_frame(self, sample: FutureSample) -> int:
        # Validation configs normally use [0, 0], i.e. action stop frame.
        ap = float(sum(self.anticipation_point) / 2.0)
        return int(sample.start_frame * ap + (1.0 - ap) * sample.stop_frame)

    def __len__(self):
        return len(self.samples)

    def _decode_clip(self, vr: VideoReader, vfps: float, end_frame: int):
        frame_step = max(1, int(vfps / self.fps))
        nframes = int(self.frames_per_clip * frame_step)
        indices = np.arange(end_frame - nframes, end_frame, frame_step).astype(np.int64)
        indices[indices < 0] = 0
        n_total = len(vr)
        if n_total > 0:
            indices[indices >= n_total] = n_total - 1
        return self.transform(vr.get_batch(indices).asnumpy())

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        vr = VideoReader(sample.video_path, num_threads=1, ctx=cpu(0))
        vr.seek(0)
        vfps = float(vr.get_avg_fps())
        anchor = self._anchor_frame(sample)
        observed_end = anchor - int(self.horizon_sec * vfps)
        obs = self._decode_clip(vr, vfps, observed_end)
        oracle = self._decode_clip(vr, vfps, anchor)
        return {
            "observed": obs,
            "oracle": oracle,
            "verb_raw": torch.tensor(sample.verb_raw, dtype=torch.long),
            "noun_raw": torch.tensor(sample.noun_raw, dtype=torch.long),
            "metadata": {
                "video_id": sample.video_id,
                "video_path": sample.video_path,
                "start_frame": sample.start_frame,
                "stop_frame": sample.stop_frame,
                "anchor_frame": anchor,
                "observed_end_frame": observed_end,
                "horizon_sec": self.horizon_sec,
            },
        }


def _build_samples(val_annotations) -> list[FutureSample]:
    paths, annotations = val_annotations
    path_by_id = {Path(path).stem: path for path in paths}
    samples = []
    for video_id, df in annotations.items():
        path = path_by_id.get(str(video_id))
        if path is None:
            continue
        for row in df.itertuples(index=False):
            samples.append(
                FutureSample(
                    video_path=str(path),
                    video_id=str(video_id),
                    start_frame=int(getattr(row, "start_frame")),
                    stop_frame=int(getattr(row, "stop_frame")),
                    verb_raw=int(getattr(row, "verb_class")),
                    noun_raw=int(getattr(row, "noun_class")),
                )
            )
    return samples


def _get_model_modules(pretrain_kwargs):
    if pretrain_kwargs.get("use_v2_1", False):
        import app.vjepa_2_1.models.predictor as vit_pred
        import app.vjepa_2_1.models.vision_transformer as vit
    else:
        import src.models.predictor as vit_pred
        import src.models.vision_transformer as vit
    return vit, vit_pred


def _load_encoder_predictor(cfg: dict, device: torch.device):
    model_kwargs = cfg["model_kwargs"]
    pretrain_kwargs = model_kwargs["pretrain_kwargs"]
    checkpoint_data = torch.load(model_kwargs["checkpoint"], map_location="cpu")
    vit, vit_pred = _get_model_modules(pretrain_kwargs)

    enc_kwargs = dict(pretrain_kwargs["encoder"])
    encoder = vit.__dict__[enc_kwargs["model_name"]](
        img_size=cfg["experiment"]["data"]["resolution"],
        num_frames=cfg["experiment"]["data"]["frames_per_clip"],
        **enc_kwargs,
    )
    enc_state = checkpoint_data[enc_kwargs["checkpoint_key"]]
    enc_state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc_state.items()}
    enc_state = {k: enc_state.get(k, v) if enc_state.get(k, v).shape == v.shape else v for k, v in encoder.state_dict().items()}
    logger.info("Loaded encoder: %s", encoder.load_state_dict(enc_state, strict=False))

    prd_kwargs = dict(pretrain_kwargs["predictor"])
    teacher_embed_dim = prd_kwargs.get("teacher_embed_dim")
    n_output_distillation = prd_kwargs.get("n_output_distillation", 4)
    out_embed_dim = teacher_embed_dim // n_output_distillation if teacher_embed_dim is not None else None
    predictor = vit_pred.__dict__[prd_kwargs["model_name"]](
        img_size=cfg["experiment"]["data"]["resolution"],
        embed_dim=encoder.embed_dim,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        out_embed_dim=out_embed_dim,
        **prd_kwargs,
    )
    pred_state = checkpoint_data[prd_kwargs["checkpoint_key"]]
    pred_state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in pred_state.items()}
    pred_state = {k: pred_state.get(k, v) if pred_state.get(k, v).shape == v.shape else v for k, v in predictor.state_dict().items()}
    logger.info("Loaded predictor: %s", predictor.load_state_dict(pred_state, strict=False))

    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    for module in (encoder, predictor):
        for param in module.parameters():
            param.requires_grad = False
    if hasattr(predictor, "hierarchical_layers") and len(predictor.hierarchical_layers) > 1:
        encoder.return_hierarchical = True
    return encoder, predictor


def _load_classifiers(cfg: dict, annotations: dict, embed_dim: int, device: torch.device):
    lora_cfg = cfg["experiment"].get("lora", {})
    factory = _make_lora_init_classifier(lora_cfg)
    classifiers = factory(
        embed_dim=embed_dim,
        num_heads=cfg["experiment"]["classifier"]["num_heads"],
        num_blocks=cfg["experiment"]["classifier"]["num_probe_blocks"],
        device=device,
        num_classifiers=len(cfg["experiment"]["optimization"]["multihead_kwargs"]),
        action_classes=annotations["actions"],
        verb_classes=annotations["verbs"],
        noun_classes=annotations["nouns"],
    )
    latest = Path(cfg["folder"]) / "action_anticipation_frozen" / cfg["tag"] / "latest.pt"
    checkpoint = robust_checkpoint_loader(str(latest), map_location=torch.device("cpu"))
    for classifier, state in zip(classifiers, checkpoint["classifiers"]):
        clean = {k.removeprefix("module."): v for k, v in state.items()}
        msg = classifier.load_state_dict(clean, strict=False)
        logger.info("Loaded classifier from %s: %s", latest, msg)
        classifier.eval()
    return classifiers


def _last_layer(tokens: torch.Tensor, embed_dim: int) -> torch.Tensor:
    return tokens[:, :, -embed_dim:] if tokens.size(-1) > embed_dim else tokens


def _predict_direct(encoder, predictor, observed_tokens, horizon_sec, cfg, device, dense: bool = False):
    data_cfg = cfg["experiment"]["data"]
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    B, N, _ = observed_tokens.shape
    grid = data_cfg["resolution"] // encoder.patch_size
    spatial = grid * grid
    tubelet = encoder.tubelet_size
    num_output_frames = max(int(wrapper_cfg.get("num_output_frames", 2)), tubelet)
    n_pred = int(spatial * (num_output_frames // tubelet))
    anticipation_steps = int(horizon_sec * data_cfg["frames_per_second"] / tubelet)
    start = N + spatial * anticipation_steps
    mask_start = N if dense else start
    mask_tokens = (start - N) + n_pred if dense else n_pred
    max_position = int(getattr(predictor, "num_patches", start + n_pred))
    if start + n_pred > max_position:
        return None, {
            "status": "unsupported_position",
            "target_start": start,
            "target_end": start + n_pred - 1,
            "max_position": max_position - 1,
        }
    if mask_tokens <= 0:
        return None, {"status": "empty_mask", "target_start": start, "mask_tokens": mask_tokens}
    masks_x = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
    masks_y = torch.arange(mask_tokens, device=device).unsqueeze(0).repeat(B, 1) + mask_start
    pred = predictor(observed_tokens, masks_x=masks_x, masks_y=masks_y)
    pred = pred[0] if isinstance(pred, tuple) else pred
    return _last_layer(pred[:, -n_pred:, :], encoder.embed_dim), {
        "status": "ok",
        "mask_tokens": mask_tokens,
        "target_start": start,
    }


def _predict_ar(encoder, predictor, observed_tokens, horizon_sec, cfg, device):
    data_cfg = cfg["experiment"]["data"]
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    B, N, _ = observed_tokens.shape
    grid = data_cfg["resolution"] // encoder.patch_size
    spatial = grid * grid
    tubelet = encoder.tubelet_size
    num_output_frames = max(int(wrapper_cfg.get("num_output_frames", 2)), tubelet)
    n_pred = int(spatial * (num_output_frames // tubelet))
    local_x = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
    local_y = torch.arange(n_pred, device=device).unsqueeze(0).repeat(B, 1) + N
    horizon_chunks = int(horizon_sec * data_cfg["frames_per_second"] / tubelet)
    rollout_steps = max(1, horizon_chunks + (num_output_frames // tubelet))
    max_steps = int(wrapper_cfg.get("max_rollout_steps", 512))
    if rollout_steps > max_steps:
        return None, {"status": "too_many_steps", "steps": rollout_steps, "max_steps": max_steps}
    window = observed_tokens
    target = None
    for step in range(rollout_steps):
        pred = predictor(window, masks_x=local_x, masks_y=local_y)
        pred = pred[0] if isinstance(pred, tuple) else pred
        pred_last = _last_layer(pred, encoder.embed_dim)
        if step == rollout_steps - 1:
            target = pred_last
        pred_in = pred if pred.size(-1) == window.size(-1) else pred_last
        window = torch.cat([window[:, n_pred:, :], pred_in], dim=1)
    return target, {"status": "ok", "steps": rollout_steps}


def _labels(batch, annotations: dict, device: torch.device):
    verbs = batch["verb_raw"]
    nouns = batch["noun_raw"]
    verb = torch.tensor([annotations["verbs"][int(v)] for v in verbs], device=device, dtype=torch.long)
    noun = torch.tensor([annotations["nouns"][int(n)] for n in nouns], device=device, dtype=torch.long)
    action = torch.tensor(
        [annotations["actions"][(int(v), int(n))] for v, n in zip(verbs, nouns)],
        device=device,
        dtype=torch.long,
    )
    return {"verb": verb, "noun": noun, "action": action}


def _metric_pack(annotations: dict, device: torch.device):
    return {
        "verb": Top3AccuracyRecallAt5(len(annotations["verbs"]), device),
        "noun": Top3AccuracyRecallAt5(len(annotations["nouns"]), device),
        "action": Top3AccuracyRecallAt5(len(annotations["actions"]), device),
    }


def _update_metrics(metrics, outputs, labels, annotations):
    metrics["verb"](outputs["verb"], labels["verb"], annotations["val_verbs"])
    metrics["noun"](outputs["noun"], labels["noun"], annotations["val_nouns"])
    metrics["action"](outputs["action"], labels["action"], annotations["val_actions"])


def _metric_values(metric: Top3AccuracyRecallAt5):
    top3_total = torch.sum(metric.top3_tp + metric.top3_fn).clamp(min=1.0)
    top3 = 100.0 * torch.sum(metric.top3_tp) / top3_total
    seen = torch.sum((metric.r5_tp + metric.r5_fn) > 0).clamp(min=1)
    recall = 100.0 * torch.sum(metric.r5_tp / (metric.r5_tp + metric.r5_fn + 1e-8)) / seen
    return float(top3), float(recall)


def _final_metrics(metrics):
    out = {}
    for name, metric in metrics.items():
        top3, recall = _metric_values(metric)
        out[f"{name}_top3"] = top3
        out[f"{name}_recall5"] = recall
    return out


def _latent_stats(pred: torch.Tensor, oracle: torch.Tensor):
    pred_f = pred.float()
    oracle_f = oracle.float()
    mse = torch.mean((pred_f - oracle_f) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(pred_f.flatten(1), oracle_f.flatten(1), dim=1).mean().item()
    return mse, cos


@torch.no_grad()
def run_horizon(args, cfg, annotations, samples, encoder, predictor, classifiers, device, horizon: float):
    data_cfg = cfg["experiment"]["data"]
    ds = FutureOracleDataset(
        samples=samples,
        horizon_sec=horizon,
        frames_per_clip=data_cfg["frames_per_clip"],
        fps=data_cfg["frames_per_second"],
        anticipation_point=tuple(data_cfg.get("val_anticipation_point", [0.0, 0.0])),
        resolution=data_cfg["resolution"],
        drop_incomplete_history=args.drop_incomplete_history,
        max_samples=args.max_samples,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logger.info("Running horizon %.3fs over %d samples", horizon, len(ds))

    methods = ["encoder", "direct_single", "direct_dense", "ar", "oracle"]
    metrics = {m: [_metric_pack(annotations, device) for _ in classifiers] for m in methods}
    latent_rows = {m: [] for m in ["direct_single", "direct_dense", "ar"]}
    status_counts: dict[str, int] = {}
    sample_count = 0
    use_bfloat16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False)) and device.type == "cuda"

    for batch_idx, batch in enumerate(loader):
        observed = batch["observed"].to(device, non_blocking=True)
        oracle_clip = batch["oracle"].to(device, non_blocking=True)
        labels = _labels(batch, annotations, device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            observed_tokens = encoder(observed)
            observed_last = _last_layer(observed_tokens, encoder.embed_dim)
            oracle_tokens = encoder(oracle_clip)
            oracle_last = _last_layer(oracle_tokens, encoder.embed_dim)
            wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
            n_pred = (data_cfg["resolution"] // encoder.patch_size) ** 2
            n_pred *= max(int(wrapper_cfg.get("num_output_frames", 2)), encoder.tubelet_size) // encoder.tubelet_size
            oracle_target = oracle_last[:, -n_pred:, :]

            tokens_by_method = {"encoder": observed_last}
            target_by_method = {"oracle": oracle_target}
            direct_target, direct_info = _predict_direct(
                encoder, predictor, observed_tokens, horizon, cfg, device, dense=False
            )
            status_counts[f"direct_single:{direct_info['status']}"] = (
                status_counts.get(f"direct_single:{direct_info['status']}", 0) + observed.size(0)
            )
            if direct_target is not None:
                target_by_method["direct_single"] = direct_target
                mse, cos = _latent_stats(direct_target, oracle_target)
                latent_rows["direct_single"].append((mse, cos, observed.size(0)))
            dense_target, dense_info = _predict_direct(
                encoder, predictor, observed_tokens, horizon, cfg, device, dense=True
            )
            status_counts[f"direct_dense:{dense_info['status']}"] = (
                status_counts.get(f"direct_dense:{dense_info['status']}", 0) + observed.size(0)
            )
            if dense_target is not None:
                target_by_method["direct_dense"] = dense_target
                mse, cos = _latent_stats(dense_target, oracle_target)
                latent_rows["direct_dense"].append((mse, cos, observed.size(0)))
            ar_target, ar_info = _predict_ar(encoder, predictor, observed_tokens, horizon, cfg, device)
            status_counts[f"ar:{ar_info['status']}"] = status_counts.get(f"ar:{ar_info['status']}", 0) + observed.size(0)
            if ar_target is not None:
                target_by_method["ar"] = ar_target
                mse, cos = _latent_stats(ar_target, oracle_target)
                latent_rows["ar"].append((mse, cos, observed.size(0)))

            for method, target in target_by_method.items():
                tokens_by_method[method] = torch.cat([observed_last, target], dim=1)
            for method, tokens in tokens_by_method.items():
                for idx, classifier in enumerate(classifiers):
                    outputs = classifier(tokens)
                    _update_metrics(metrics[method][idx], outputs, labels, annotations)
        sample_count += observed.size(0)
        if batch_idx % args.log_every == 0:
            logger.info("horizon %.3fs batch %d samples=%d statuses=%s", horizon, batch_idx, sample_count, status_counts)

    rows = []
    for method in methods:
        best_idx = None
        best_action = -math.inf
        best_metrics = None
        for idx, metric_pack in enumerate(metrics[method]):
            vals = _final_metrics(metric_pack)
            if vals["action_top3"] > best_action:
                best_idx = idx
                best_action = vals["action_top3"]
                best_metrics = vals
        latent_mse = ""
        latent_cos = ""
        if method in latent_rows and latent_rows[method]:
            denom = sum(n for _, _, n in latent_rows[method])
            latent_mse = sum(mse * n for mse, _, n in latent_rows[method]) / max(1, denom)
            latent_cos = sum(cos * n for _, cos, n in latent_rows[method]) / max(1, denom)
        status = "ok"
        if method not in {"encoder", "oracle"}:
            ok = status_counts.get(f"{method}:ok", 0)
            status = "ok" if ok == sample_count else f"partial_ok_{ok}_of_{sample_count}"
        row = {
            "horizon_sec": horizon,
            "method": method,
            "status": status,
            "samples": sample_count,
            "best_classifier": best_idx,
            "latent_mse_to_oracle": latent_mse,
            "latent_cos_to_oracle": latent_cos,
            **(best_metrics or {}),
            "status_counts": json.dumps(status_counts, sort_keys=True),
        }
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--horizons", default="1,1.5,2,2.5,3,4,5,6,7,8,9,10,60")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--drop-incomplete-history", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
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

    horizons = [float(x) for x in args.horizons.replace(",", " ").split()]
    rows = []
    for horizon in horizons:
        rows.extend(run_horizon(args, cfg, annotations, samples, encoder, predictor, classifiers, device, horizon))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote future latent comparison: %s", out_path)


if __name__ == "__main__":
    main()
