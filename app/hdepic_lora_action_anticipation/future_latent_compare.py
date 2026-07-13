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
import os
from dataclasses import dataclass
from pathlib import Path

# Large HD-EPIC clips (e.g. P01_20240204-130448) need a high EOF retry budget when
# decord seeks near the file tail. Must be set before importing decord.
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "65536")

import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset

from app.hdepic_lora_action_anticipation.eval import Top3AccuracyRecallAt5, _make_lora_init_classifier
from app.hdepic_lora_action_anticipation.binary_input_adapter import (
    BinaryGazeMapBuilder,
    BinaryMapInputAdapter,
)
from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate
from app.hdepic_lora_action_anticipation.gaze_rnn import (
    GazeTrajectoryLoader,
    call_classifier,
    encode_gaze_tokens,
)
from app.hdepic_lora_action_anticipation.rope_position_scaling import remap_mask_pair_ntk_temporal
from evals.action_anticipation_frozen.dataloader import filter_annotations, make_transforms
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger("future_latent_compare")

_DEFAULT_DECORD_BLOCKLIST = (
    Path(__file__).resolve().parents[2] / "data/hdepic_vjepa_annotations/decord_eof_videos.txt"
)


def load_decord_blocklist(path: Path | str | None = None) -> set[str]:
    blocklist_path = Path(path or os.environ.get("HDEPIC_DECORD_BLOCKLIST", _DEFAULT_DECORD_BLOCKLIST))
    if not blocklist_path.exists():
        return set()
    entries: set[str] = set()
    for line in blocklist_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.add(line)
    return entries


# Per-worker single VideoReader cache (clip-balanced style; one open reader per worker).
_WORKER_READER_PATH: str | None = None
_WORKER_READER_STATE: tuple[VideoReader, float, int] | None = None


def decord_worker_init(_worker_id: int) -> None:
    os.environ.setdefault("DECORD_EOF_RETRY_MAX", "65536")
    _reset_worker_reader()


def _reset_worker_reader() -> None:
    global _WORKER_READER_PATH, _WORKER_READER_STATE
    if _WORKER_READER_STATE is not None:
        vr, _, _ = _WORKER_READER_STATE
        del vr
    _WORKER_READER_PATH = None
    _WORKER_READER_STATE = None


def _cached_video_reader(video_path: str, *, cache_reader: bool = True) -> tuple[VideoReader, float, int]:
    global _WORKER_READER_PATH, _WORKER_READER_STATE
    path = str(video_path)
    if cache_reader and _WORKER_READER_PATH == path and _WORKER_READER_STATE is not None:
        return _WORKER_READER_STATE
    if _WORKER_READER_STATE is not None:
        vr, _, _ = _WORKER_READER_STATE
        del vr
        _WORKER_READER_PATH = None
        _WORKER_READER_STATE = None
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    state = (vr, float(vr.get_avg_fps()), len(vr))
    if cache_reader:
        _WORKER_READER_PATH = path
        _WORKER_READER_STATE = state
    return state


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
        fps: float,
        anticipation_point: tuple[float, float],
        resolution: int,
        drop_incomplete_history: bool,
        max_samples: int | None = None,
        training: bool = False,
        auto_augment: bool = True,
        reprob: float = 0.25,
        random_resize_scale: tuple[float, float] = (0.08, 1.0),
        decord_blocklist_path: Path | str | None = None,
        probe_decodable: bool = False,
    ):
        self.samples = samples[:max_samples] if max_samples else samples
        self.horizon_sec = float(horizon_sec)
        self.frames_per_clip = int(frames_per_clip)
        self.fps = float(fps)
        self.anticipation_point = anticipation_point
        self.transform = make_transforms(
            training=training,
            crop_size=resolution,
            auto_augment=auto_augment,
            reprob=reprob,
            random_resize_scale=random_resize_scale,
        )
        self.drop_incomplete_history = bool(drop_incomplete_history)
        blocklist = load_decord_blocklist(decord_blocklist_path)
        if blocklist:
            before = len(self.samples)
            self.samples = [
                s for s in self.samples if s.video_id not in blocklist and s.video_path not in blocklist
            ]
            logger.info(
                "Decord blocklist removed %d/%d samples (entries=%d)",
                before - len(self.samples),
                before,
                len(blocklist),
            )
        if self.drop_incomplete_history:
            self.samples = self._filter_full_history(self.samples)
        if probe_decodable:
            self.samples = self._filter_decodable_videos(self.samples)

    def _filter_full_history(self, samples: list[FutureSample]) -> list[FutureSample]:
        meta_cache: dict[str, tuple[float, int]] = {}
        bad_videos: set[str] = set()
        kept = []
        for sample in samples:
            path = sample.video_path
            if path in bad_videos:
                continue
            if path not in meta_cache:
                try:
                    meta_cache[path] = self._read_video_length(path)
                except Exception as exc:
                    logger.info(
                        "Skipping unreadable video during history filter: %s error=%r",
                        path,
                        exc,
                    )
                    bad_videos.add(path)
                    continue
            vfps, n_total = meta_cache[path]
            frame_step = max(1, int(vfps / self.fps))
            nframes = int(self.frames_per_clip * frame_step)
            anchor = self._anchor_frame(sample)
            observed_end = anchor - int(self.horizon_sec * vfps)
            if observed_end - nframes >= 0 and anchor < n_total:
                kept.append(sample)
        logger.info(
            "Horizon %.3fs full-history filter kept %d/%d samples",
            self.horizon_sec,
            len(kept),
            len(samples),
        )
        return kept

    def _filter_decodable_videos(self, samples: list[FutureSample]) -> list[FutureSample]:
        by_path: dict[str, list[FutureSample]] = {}
        for sample in samples:
            by_path.setdefault(sample.video_path, []).append(sample)
        kept: list[FutureSample] = []
        kept_videos = 0
        for path, group in by_path.items():
            probe = max(group, key=lambda s: s.stop_frame)
            try:
                self._probe_sample_decode(probe)
            except Exception as exc:
                logger.warning(
                    "Excluding decord-failing video %s (%d samples): %r",
                    path,
                    len(group),
                    exc,
                )
                continue
            kept.extend(group)
            kept_videos += 1
        logger.info(
            "Decodable probe kept %d/%d samples across %d/%d videos",
            len(kept),
            len(samples),
            kept_videos,
            len(by_path),
        )
        return kept

    @staticmethod
    def _read_video_length(video_path: str) -> tuple[float, int]:
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        vfps = float(vr.get_avg_fps())
        n_total = len(vr)
        del vr
        return vfps, n_total

    def _clip_indices(self, vfps: float, end_frame: int, n_total: int) -> np.ndarray:
        frame_step = max(1, int(vfps / self.fps))
        nframes = int(self.frames_per_clip * frame_step)
        indices = np.arange(end_frame - nframes, end_frame, frame_step).astype(np.int64)
        indices[indices < 0] = 0
        if n_total > 0:
            safe_max = max(0, n_total - 1)
            indices[indices >= n_total] = safe_max
        return indices

    def _decode_clip(self, vr: VideoReader, vfps: float, end_frame: int, n_total: int):
        indices = self._clip_indices(vfps, end_frame, n_total)
        clip = self.transform(vr.get_batch(indices).asnumpy())
        return clip, indices

    def _probe_sample_decode(self, sample: FutureSample) -> None:
        vr, vfps, n_total = _cached_video_reader(sample.video_path)
        anchor = self._anchor_frame(sample)
        observed_end = anchor - int(self.horizon_sec * vfps)
        for end_frame in (observed_end, anchor):
            indices = self._clip_indices(vfps, end_frame, n_total)
            vr.get_batch(indices).asnumpy()

    def _anchor_frame(self, sample: FutureSample) -> int:
        # Validation configs normally use [0, 0], i.e. action stop frame.
        ap = float(sum(self.anticipation_point) / 2.0)
        return int(sample.start_frame * ap + (1.0 - ap) * sample.stop_frame)

    def __len__(self):
        return len(self.samples)

    def _getitem_one(self, sample: FutureSample):
        vr, vfps, n_total = _cached_video_reader(sample.video_path)
        first = vr.get_batch([0]).asnumpy()
        h0, w0 = int(first.shape[1]), int(first.shape[2])
        anchor = self._anchor_frame(sample)
        observed_end = anchor - int(self.horizon_sec * vfps)
        obs, obs_indices = self._decode_clip(vr, vfps, observed_end, n_total)
        oracle, oracle_indices = self._decode_clip(vr, vfps, anchor, n_total)
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
                "frame_indices": obs_indices.tolist(),
                "oracle_frame_indices": oracle_indices.tolist(),
                "vfps": vfps,
                "height": h0,
                "width": w0,
            },
        }

    def __getitem__(self, idx: int):
        n = len(self.samples)
        last_exc = None
        for attempt in range(12):
            j = (idx + attempt * 17) % n
            sample = self.samples[j]
            try:
                return self._getitem_one(sample)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "FutureOracleDataset skip video=%s attempt=%d: %r",
                    sample.video_path,
                    attempt,
                    exc,
                )
        raise RuntimeError(f"Failed to load sample after retries: {last_exc}") from last_exc


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
    gaze_cfg = dict(lora_cfg.get("gaze", {}))
    gaze_mode = str(gaze_cfg.get("mode", "none")).lower()
    traj_mode = gaze_mode if gaze_mode in {"rnn_fuse", "mlp_fuse"} else None
    rnn_cfg = dict(gaze_cfg.get("rnn", {}))
    factory = _make_lora_init_classifier(lora_cfg, traj_mode=traj_mode, rnn_cfg=rnn_cfg)
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


def _predictor_trained_grid_depth(predictor, encoder) -> int:
    grid_depth = getattr(predictor, "grid_depth", None)
    if grid_depth is not None:
        return int(grid_depth)
    num_patches = int(getattr(predictor, "num_patches", 0))
    spatial = (encoder.grid_height * encoder.grid_width) if hasattr(encoder, "grid_height") else None
    if num_patches > 0 and spatial:
        return max(int(num_patches // spatial), 1)
    pretrain_frames = int(getattr(predictor, "num_frames", encoder.num_frames))
    tubelet = int(encoder.tubelet_size)
    return max(pretrain_frames // tubelet, 1)


def _predict_direct(
    encoder,
    predictor,
    observed_tokens,
    horizon_sec,
    cfg,
    device,
    dense: bool = False,
    rope_scale_mode: str | None = None,
):
    data_cfg = cfg["experiment"]["data"]
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    downsample_factor = float(data_cfg.get("video_downsample_factor", 1.0) or 1.0)
    B, N, _ = observed_tokens.shape
    grid = data_cfg["resolution"] // encoder.patch_size
    spatial = grid * grid
    tubelet = encoder.tubelet_size
    num_output_frames = max(int(wrapper_cfg.get("num_output_frames", 2)), tubelet)
    n_pred = int(spatial * (num_output_frames // tubelet))
    anticipation_steps = int((horizon_sec / downsample_factor) * data_cfg["frames_per_second"] / tubelet)
    start = N + spatial * anticipation_steps
    mask_start = N if dense else start
    mask_tokens = (start - N) + n_pred if dense else n_pred
    max_position = int(getattr(predictor, "num_patches", start + n_pred))
    use_rope_scale = str(rope_scale_mode or "").lower() == "ntk_temporal"
    if start + n_pred > max_position and not use_rope_scale:
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
    rope_scale = 1.0
    if use_rope_scale:
        trained_grid_depth = _predictor_trained_grid_depth(predictor, encoder)
        masks_x, masks_y, rope_scale = remap_mask_pair_ntk_temporal(
            masks_x, masks_y, spatial, trained_grid_depth
        )
        remapped_max = int(torch.cat([masks_x.reshape(-1), masks_y.reshape(-1)]).max().item())
        if remapped_max >= max_position:
            return None, {
                "status": "unsupported_position_after_remap",
                "target_start": start,
                "remapped_max": remapped_max,
                "max_position": max_position - 1,
                "rope_scale": rope_scale,
            }
    pred = predictor(observed_tokens, masks_x=masks_x, masks_y=masks_y)
    pred = pred[0] if isinstance(pred, tuple) else pred
    return _last_layer(pred[:, -n_pred:, :], encoder.embed_dim), {
        "status": "ok",
        "mask_tokens": mask_tokens,
        "target_start": start,
        "rope_scale_mode": rope_scale_mode or "",
        "rope_scale": rope_scale,
        "dense": dense,
    }


def _predict_ar(encoder, predictor, observed_tokens, horizon_sec, cfg, device):
    data_cfg = cfg["experiment"]["data"]
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    downsample_factor = float(data_cfg.get("video_downsample_factor", 1.0) or 1.0)
    B, N, _ = observed_tokens.shape
    grid = data_cfg["resolution"] // encoder.patch_size
    spatial = grid * grid
    tubelet = encoder.tubelet_size
    num_output_frames = max(int(wrapper_cfg.get("num_output_frames", 2)), tubelet)
    n_pred = int(spatial * (num_output_frames // tubelet))
    local_x = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
    local_y = torch.arange(n_pred, device=device).unsqueeze(0).repeat(B, 1) + N
    horizon_chunks = int((horizon_sec / downsample_factor) * data_cfg["frames_per_second"] / tubelet)
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


def _update_metrics(metrics, outputs, labels, annotations, metric_scope: str = "native"):
    if metric_scope == "filtered":
        metrics["verb"](outputs["verb"], labels["verb"], annotations["val_verbs"])
        metrics["noun"](outputs["noun"], labels["noun"], annotations["val_nouns"])
        metrics["action"](outputs["action"], labels["action"], annotations["val_actions"])
    elif metric_scope == "native":
        metrics["verb"](outputs["verb"], labels["verb"])
        metrics["noun"](outputs["noun"], labels["noun"])
        metrics["action"](outputs["action"], labels["action"])
    else:
        raise ValueError(f"Unsupported metric_scope={metric_scope!r}; expected native or filtered")


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


def _select_head_metrics(metrics_per_head, head_selection: str):
    per_head = []
    for idx, metric_pack in enumerate(metrics_per_head):
        vals = _final_metrics(metric_pack)
        vals["head"] = idx
        per_head.append(vals)

    if not per_head:
        return None, {}, {}

    metric_keys = [
        "action_top3",
        "action_recall5",
        "verb_top3",
        "verb_recall5",
        "noun_top3",
        "noun_recall5",
    ]
    action_head = max(per_head, key=lambda row: row["action_top3"])
    selected_heads = {}

    if head_selection == "action_top3":
        report = {key: action_head[key] for key in metric_keys}
        for key in metric_keys:
            selected_heads[f"{key}_head"] = int(action_head["head"])
        return int(action_head["head"]), report, selected_heads

    if head_selection != "vjepa2":
        raise ValueError(f"Unsupported head_selection={head_selection!r}; expected vjepa2 or action_top3")

    report = {}
    for key in metric_keys:
        best = max(per_head, key=lambda row: row[key])
        report[key] = best[key]
        selected_heads[f"{key}_head"] = int(best["head"])
    return int(action_head["head"]), report, selected_heads


def _latent_stats(pred: torch.Tensor, oracle: torch.Tensor):
    pred_f = pred.float()
    oracle_f = oracle.float()
    mse = torch.mean((pred_f - oracle_f) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(pred_f.flatten(1), oracle_f.flatten(1), dim=1).mean().item()
    pred_norm = torch.linalg.vector_norm(pred_f.flatten(1), dim=1).mean().item()
    oracle_norm = torch.linalg.vector_norm(oracle_f.flatten(1), dim=1).mean().item()
    norm_ratio = pred_norm / max(oracle_norm, 1e-8)
    return mse, cos, norm_ratio


def _collate(batch):
    out = {
        "observed": torch.stack([b["observed"] for b in batch], dim=0),
        "oracle": torch.stack([b["oracle"] for b in batch], dim=0),
        "verb_raw": torch.stack([b["verb_raw"] for b in batch], dim=0),
        "noun_raw": torch.stack([b["noun_raw"] for b in batch], dim=0),
        "metadata": [b["metadata"] for b in batch],
    }
    return out


def _build_gaze_components(cfg, classifiers, device):
    """Construct gaze runtime components from cfg + checkpoint, if a gaze mode is set.

    Returns a dict with keys:
        - mode: "none" | "binary_input_adapter" | "rnn_fuse" | "mlp_fuse"
        - adapter: BinaryMapInputAdapter | None
        - map_builder: BinaryGazeMapBuilder | None
        - traj_loader: GazeTrajectoryLoader | None
    """
    lora_cfg = cfg.get("experiment", {}).get("lora", {})
    gaze_cfg = dict(lora_cfg.get("gaze", {}))
    mode = str(gaze_cfg.get("mode", "none")).lower()
    out = {"mode": mode, "adapter": None, "map_builder": None, "traj_loader": None}
    if mode == "none":
        return out

    data_cfg = cfg.get("experiment", {}).get("data", {})
    enc_kwargs = cfg["model_kwargs"]["pretrain_kwargs"]["encoder"]
    gaze_cfg.setdefault("crop_size", data_cfg.get("resolution", 384))
    gaze_cfg.setdefault("frames_per_clip", data_cfg.get("frames_per_clip", 32))
    gaze_cfg.setdefault("patch_size", enc_kwargs.get("patch_size", 16))
    gaze_cfg.setdefault("tubelet_size", enc_kwargs.get("tubelet_size", 2))

    if mode == "binary_input_adapter":
        ad_cfg = dict(gaze_cfg.get("input_adapter", {}))
        adapter = BinaryMapInputAdapter(
            hidden_dim=int(ad_cfg.get("hidden_dim", 8)),
            scale=float(ad_cfg.get("scale", 1.0)),
            temporal_kernel=int(ad_cfg.get("temporal_kernel", 1)),
            binary_center=float(ad_cfg.get("binary_center", 0.0)),
            residual_clamp=float(ad_cfg.get("residual_clamp", 1.0)),
        ).to(device).eval()
        for p in adapter.parameters():
            p.requires_grad = False
        ckpt_path = Path(cfg["folder"]) / "action_anticipation_frozen" / cfg["tag"] / "binary_input_adapter_latest.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"binary_input_adapter checkpoint not found: {ckpt_path}")
        ckpt = robust_checkpoint_loader(str(ckpt_path), map_location=torch.device("cpu"))
        state = ckpt.get("input_adapter", ckpt)
        if any(str(k).startswith("module.input_adapter.") for k in state):
            state = {str(k).removeprefix("module.input_adapter."): v for k, v in state.items() if str(k).startswith("module.input_adapter.")}
        elif any(str(k).startswith("input_adapter.") for k in state):
            state = {str(k).removeprefix("input_adapter."): v for k, v in state.items() if str(k).startswith("input_adapter.")}
        missing, unexpected = adapter.load_state_dict(state, strict=False)
        logger.info("Loaded binary_input_adapter from %s missing=%d unexpected=%d", ckpt_path, len(missing), len(unexpected))
        gate = GazeTokenGate({**gaze_cfg, "mode": "token_gate"})
        map_builder = BinaryGazeMapBuilder(gaze_cfg, gate=gate)
        out["adapter"] = adapter
        out["map_builder"] = map_builder
        return out

    if mode == "binary_input_adapter_gaze_pose_matrix":
        from app.hdepic_lora_action_anticipation.pose_map_builder import GazePoseInputMapBuilder

        ad_cfg = dict(gaze_cfg.get("input_adapter", {}))
        in_channels = int(ad_cfg.get("in_channels", 5))
        adapter = BinaryMapInputAdapter(
            hidden_dim=int(ad_cfg.get("hidden_dim", 8)),
            scale=float(ad_cfg.get("scale", 1.0)),
            temporal_kernel=int(ad_cfg.get("temporal_kernel", 3)),
            binary_center=float(ad_cfg.get("binary_center", 0.0)),
            residual_clamp=float(ad_cfg.get("residual_clamp", 1.0)),
            in_channels=in_channels,
        ).to(device).eval()
        for p in adapter.parameters():
            p.requires_grad = False
        ckpt_path = Path(cfg["folder"]) / "action_anticipation_frozen" / cfg["tag"] / "binary_input_adapter_latest.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"binary_input_adapter checkpoint not found: {ckpt_path}")
        ckpt = robust_checkpoint_loader(str(ckpt_path), map_location=torch.device("cpu"))
        state = ckpt.get("input_adapter", ckpt)
        if any(str(k).startswith("module.input_adapter.") for k in state):
            state = {str(k).removeprefix("module.input_adapter."): v for k, v in state.items() if str(k).startswith("module.input_adapter.")}
        elif any(str(k).startswith("input_adapter.") for k in state):
            state = {str(k).removeprefix("input_adapter."): v for k, v in state.items() if str(k).startswith("input_adapter.")}
        missing, unexpected = adapter.load_state_dict(state, strict=False)
        logger.info(
            "Loaded binary_input_adapter_gaze_pose_matrix from %s missing=%d unexpected=%d",
            ckpt_path,
            len(missing),
            len(unexpected),
        )
        gate = GazeTokenGate({**gaze_cfg, "mode": "token_gate"})
        map_builder = GazePoseInputMapBuilder(gaze_cfg, gate=gate)
        out["adapter"] = adapter
        out["map_builder"] = map_builder
        return out

    if mode in {"rnn_fuse", "mlp_fuse"}:
        gate = GazeTokenGate({**gaze_cfg, "mode": mode})
        traj_loader = GazeTrajectoryLoader(gaze_cfg, gate=gate)
        out["traj_loader"] = traj_loader
        return out

    raise ValueError(f"Unsupported gaze mode for future_latent_compare: {mode}")


@torch.no_grad()
def run_horizon(args, cfg, annotations, samples, encoder, predictor, classifiers, device, horizon: float, gaze_components: dict | None = None):
    data_cfg = cfg["experiment"]["data"]
    downsample_factor = float(data_cfg.get("video_downsample_factor", 1.0) or 1.0)
    ds = FutureOracleDataset(
        samples=samples,
        horizon_sec=horizon,
        frames_per_clip=data_cfg["frames_per_clip"],
        fps=float(data_cfg["frames_per_second"]) / downsample_factor,
        anticipation_point=tuple(data_cfg.get("val_anticipation_point", [0.0, 0.0])),
        resolution=data_cfg["resolution"],
        drop_incomplete_history=args.drop_incomplete_history,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    logger.info("Running horizon %.3fs over %d samples", horizon, len(ds))
    metric_scope = str(args.metric_scope).lower()
    head_selection = str(args.head_selection).lower()
    logger.info("Metric scope: %s head_selection: %s", metric_scope, head_selection)

    methods = ["encoder", "direct_single", "direct_dense", "ar", "oracle"]
    if getattr(args, "include_rope_scale", False):
        methods.extend(["direct_single_rope", "direct_dense_rope"])
    metrics = {m: [_metric_pack(annotations, device) for _ in classifiers] for m in methods}
    latent_rows = {m: [] for m in methods if m not in {"encoder", "oracle"}}
    status_counts: dict[str, int] = {}
    sample_count = 0
    use_bfloat16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False)) and device.type == "cuda"
    gaze_components = gaze_components or {"mode": "none"}
    gaze_mode = gaze_components.get("mode", "none")
    adapter = gaze_components.get("adapter")
    map_builder = gaze_components.get("map_builder")
    traj_loader = gaze_components.get("traj_loader")

    for batch_idx, batch in enumerate(loader):
        observed = batch["observed"].to(device, non_blocking=True)
        oracle_clip = batch["oracle"].to(device, non_blocking=True)
        metadata = batch["metadata"]
        labels = _labels(batch, annotations, device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            if gaze_mode == "binary_input_adapter" and adapter is not None and map_builder is not None:
                obs_meta = [
                    {**m, "frame_indices": m.get("frame_indices")} for m in metadata
                ]
                oracle_meta = [
                    {**m, "frame_indices": m.get("oracle_frame_indices", m.get("frame_indices"))}
                    for m in metadata
                ]
                obs_map = map_builder.build(observed, obs_meta)
                oracle_map = map_builder.build(oracle_clip, oracle_meta)
                observed = adapter(observed, obs_map)
                oracle_clip = adapter(oracle_clip, oracle_map)
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
                mse, cos, norm_ratio = _latent_stats(direct_target, oracle_target)
                latent_rows["direct_single"].append((mse, cos, norm_ratio, observed.size(0)))
            dense_target, dense_info = _predict_direct(
                encoder, predictor, observed_tokens, horizon, cfg, device, dense=True
            )
            status_counts[f"direct_dense:{dense_info['status']}"] = (
                status_counts.get(f"direct_dense:{dense_info['status']}", 0) + observed.size(0)
            )
            if dense_target is not None:
                target_by_method["direct_dense"] = dense_target
                mse, cos, norm_ratio = _latent_stats(dense_target, oracle_target)
                latent_rows["direct_dense"].append((mse, cos, norm_ratio, observed.size(0)))
            if getattr(args, "include_rope_scale", False):
                rope_target, rope_info = _predict_direct(
                    encoder,
                    predictor,
                    observed_tokens,
                    horizon,
                    cfg,
                    device,
                    dense=False,
                    rope_scale_mode="ntk_temporal",
                )
                status_counts[f"direct_single_rope:{rope_info['status']}"] = (
                    status_counts.get(f"direct_single_rope:{rope_info['status']}", 0) + observed.size(0)
                )
                if rope_target is not None:
                    target_by_method["direct_single_rope"] = rope_target
                    mse, cos, norm_ratio = _latent_stats(rope_target, oracle_target)
                    latent_rows["direct_single_rope"].append((mse, cos, norm_ratio, observed.size(0)))
                rope_dense_target, rope_dense_info = _predict_direct(
                    encoder,
                    predictor,
                    observed_tokens,
                    horizon,
                    cfg,
                    device,
                    dense=True,
                    rope_scale_mode="ntk_temporal",
                )
                status_counts[f"direct_dense_rope:{rope_dense_info['status']}"] = (
                    status_counts.get(f"direct_dense_rope:{rope_dense_info['status']}", 0) + observed.size(0)
                )
                if rope_dense_target is not None:
                    target_by_method["direct_dense_rope"] = rope_dense_target
                    mse, cos, norm_ratio = _latent_stats(rope_dense_target, oracle_target)
                    latent_rows["direct_dense_rope"].append((mse, cos, norm_ratio, observed.size(0)))
            ar_target, ar_info = _predict_ar(encoder, predictor, observed_tokens, horizon, cfg, device)
            status_counts[f"ar:{ar_info['status']}"] = status_counts.get(f"ar:{ar_info['status']}", 0) + observed.size(0)
            if ar_target is not None:
                target_by_method["ar"] = ar_target
                mse, cos, norm_ratio = _latent_stats(ar_target, oracle_target)
                latent_rows["ar"].append((mse, cos, norm_ratio, observed.size(0)))

            for method, target in target_by_method.items():
                tokens_by_method[method] = torch.cat([observed_last, target], dim=1)
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
            for method, tokens in tokens_by_method.items():
                for idx, classifier in enumerate(classifiers):
                    outputs = call_classifier(classifier, tokens, gaze_tokens_per_classifier[idx])
                    _update_metrics(metrics[method][idx], outputs, labels, annotations, metric_scope)
        sample_count += observed.size(0)
        if batch_idx % args.log_every == 0:
            logger.info("horizon %.3fs batch %d samples=%d statuses=%s", horizon, batch_idx, sample_count, status_counts)

    rows = []
    for method in methods:
        action_top3_head, report_metrics, selected_heads = _select_head_metrics(metrics[method], head_selection)
        latent_mse = ""
        latent_cos = ""
        latent_norm_ratio = ""
        if method in latent_rows and latent_rows[method]:
            denom = sum(n for _, _, _, n in latent_rows[method])
            latent_mse = sum(mse * n for mse, _, _, n in latent_rows[method]) / max(1, denom)
            latent_cos = sum(cos * n for _, cos, _, n in latent_rows[method]) / max(1, denom)
            latent_norm_ratio = sum(nr * n for _, _, nr, n in latent_rows[method]) / max(1, denom)
        status = "ok"
        if method not in {"encoder", "oracle"}:
            ok = status_counts.get(f"{method}:ok", 0)
            status = "ok" if ok == sample_count else f"partial_ok_{ok}_of_{sample_count}"
        row = {
            "horizon_sec": horizon,
            "metric_scope": metric_scope,
            "head_selection": head_selection,
            "metric_aggregation": "metric_wise_max" if head_selection == "vjepa2" else "action_top3_single_head",
            "method": method,
            "status": status,
            "samples": sample_count,
            "best_classifier": action_top3_head,
            "latent_mse_to_oracle": latent_mse,
            "latent_cos_to_oracle": latent_cos,
            "latent_norm_ratio_to_oracle": latent_norm_ratio,
            **report_metrics,
            **selected_heads,
            "selected_heads_json": json.dumps(selected_heads, sort_keys=True),
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
    parser.add_argument("--metric-scope", choices=["native", "filtered"], default="native")
    parser.add_argument("--head-selection", choices=["vjepa2", "action_top3"], default="vjepa2")
    parser.add_argument(
        "--include-rope-scale",
        action="store_true",
        help="Also evaluate direct_single_rope / direct_dense_rope with NTK temporal scaling.",
    )
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
    gaze_components = _build_gaze_components(cfg, classifiers, device)
    logger.info("Gaze mode: %s", gaze_components["mode"])

    horizons = [float(x) for x in args.horizons.replace(",", " ").split()]
    rows = []
    for horizon in horizons:
        rows.extend(
            run_horizon(
                args, cfg, annotations, samples, encoder, predictor, classifiers, device, horizon, gaze_components,
            )
        )

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
