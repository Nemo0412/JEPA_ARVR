"""Binary gaze-map input adapter for HD-EPIC LoRA action anticipation."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation.binary_map_utils import normalize_map_type, rasterize_gaze_disk
from app.hdepic_lora_action_anticipation.encoder_lora import (
    encoder_lora_grads_finite,
    save_encoder_lora_checkpoint,
    trainable_encoder_lora_named_params,
    zero_encoder_lora_grads,
)
from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate
from app.hdepic_lora_action_anticipation.gaze import labels_from_udata
from app.hdepic_lora_action_anticipation.val_metrics import summarize_metric_lists
from app.hdepic_lora_action_anticipation.val_metrics import summarize_val_metrics
from app.hdepic_lora_action_anticipation.latency_breakdown import (
    LatencyBreakdown,
    instrument_model_for_breakdown,
)
from src.utils.logging import AverageMeter

logger = logging.getLogger(__name__)


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def _save_trainable_sidecars(model: nn.Module, adapter_checkpoint_path: str | None, rank: int):
    if not adapter_checkpoint_path or rank != 0:
        return
    path = Path(adapter_checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_adapter": unwrap_ddp(model).input_adapter.state_dict()}, path)
    logger.info("Wrote binary input adapter checkpoint: %s", path)
    encoder_lora_path = path.with_name("encoder_lora_latest.pt")
    save_encoder_lora_checkpoint(model, encoder_lora_path)


class BinaryMapInputAdapter(nn.Module):
    """Tiny residual adapter that maps RGB + auxiliary maps back to RGB space.

    The last projection is zero-initialized, so the adapter starts as an exact
    identity on the RGB input and learns only a residual correction.
    """

    def __init__(
        self,
        hidden_dim: int = 8,
        scale: float = 1.0,
        temporal_kernel: int = 1,
        binary_center: float = 0.0,
        residual_clamp: float = 1.0,
        in_channels: int = 4,
    ):
        super().__init__()
        self.scale = float(scale)
        self.binary_center = float(binary_center)
        self.residual_clamp = float(residual_clamp)
        self.in_channels = int(in_channels)
        if self.in_channels < 4:
            raise ValueError(f"in_channels must be >= 4 (RGB + at least one aux), got {in_channels}")
        self.aux_channels = self.in_channels - 3
        tk = int(temporal_kernel)
        if tk not in {1, 3}:
            raise ValueError(f"Unsupported temporal_kernel={temporal_kernel}; expected 1 or 3")
        padding = (tk // 2, 1, 1)
        self.net = nn.Sequential(
            nn.Conv3d(self.in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(tk, 3, 3), padding=padding, groups=hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 3, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rgb: torch.Tensor, aux_map: torch.Tensor) -> torch.Tensor:
        aux_map = aux_map.to(dtype=rgb.dtype, device=rgb.device)
        if aux_map.shape[1] != self.aux_channels:
            raise ValueError(
                f"Expected aux_map with {self.aux_channels} channels, got shape {tuple(aux_map.shape)}"
            )
        aux = aux_map.clone()
        if self.aux_channels >= 1:
            aux[:, 0:1] = aux[:, 0:1] - self.binary_center
        x = torch.cat([rgb, aux], dim=1)
        residual = self.scale * self.net(x)
        if self.residual_clamp > 0:
            residual = residual.clamp(-self.residual_clamp, self.residual_clamp)
        return rgb + residual


class BinaryInputAdaptedModel(nn.Module):
    """Wrap a frozen V-JEPA model with a trainable input adapter."""

    def __init__(self, base_model: nn.Module, adapter: BinaryMapInputAdapter):
        super().__init__()
        self.base_model = base_model
        self.input_adapter = adapter
        self.embed_dim = base_model.embed_dim

    def forward(self, clips: torch.Tensor, anticipation_times: torch.Tensor, binary_map: torch.Tensor | None = None):
        if binary_map is not None:
            clips = self.input_adapter(clips, binary_map)
        tokens = self.base_model(clips, anticipation_times)
        if torch.is_tensor(tokens) and not torch.isfinite(tokens).all():
            return None
        return tokens


class BinaryGazeMapBuilder:
    """Build per-frame binary gaze disks aligned to the model crop size."""

    def __init__(self, cfg: dict[str, Any], gate: GazeTokenGate | None = None):
        self.cfg = dict(cfg)
        self.crop_size = int(cfg.get("crop_size", 384))
        self.radius_px = float(cfg.get("binary_radius_px", cfg.get("binary_radius", 64.0)))
        self.map_type = normalize_map_type(cfg.get("binary_map_type", cfg.get("map_type", "binary")))
        self.fallback_full_frame = bool(cfg.get("fallback_full_frame", False))
        self.force_zero_map = bool(cfg.get("force_zero_map", False))
        self.adapter_checkpoint_path = cfg.get("adapter_checkpoint_path")
        self.rank = int(cfg.get("rank", 0))
        self.gate = gate or GazeTokenGate({**cfg, "mode": "token_gate"})
        self._grid_cache: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def build(self, clips: torch.Tensor, metadata) -> torch.Tensor:
        bsz, _, frames, height, width = clips.shape
        if self.force_zero_map:
            return clips.new_zeros((bsz, 1, frames, height, width))
        if height != self.crop_size or width != self.crop_size:
            logger.debug("Binary map crop size differs from clips: cfg=%d clip=%sx%s", self.crop_size, height, width)
        maps = clips.new_zeros((bsz, 1, frames, height, width))
        yy, xx = self._grid(clips.device, height, width)
        radius2 = float(self.radius_px) ** 2

        for idx in range(bsz):
            meta = metadata[idx] if isinstance(metadata, list) else metadata
            xy = self._query_xy(meta)
            if xy is None:
                if self.fallback_full_frame:
                    maps[idx] = 1.0
                continue
            nframes = min(frames, xy.shape[0])
            xy_t = torch.as_tensor(xy[:nframes], device=clips.device, dtype=torch.float32)
            x = xy_t[:, 0].view(nframes, 1, 1) * (width - 1) / max(1, self.crop_size - 1)
            y = xy_t[:, 1].view(nframes, 1, 1) * (height - 1) / max(1, self.crop_size - 1)
            maps[idx, 0, :nframes] = rasterize_gaze_disk(
                xx,
                yy,
                x,
                y,
                radius2**0.5,
                map_type=self.map_type,
                dtype=maps.dtype,
            )
        return maps

    def _grid(self, device: torch.device, height: int, width: int):
        key = (str(device), int(height), int(width))
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        yy = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1)
        xx = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width)
        self._grid_cache[key] = (yy, xx)
        return yy, xx

    def _query_xy(self, meta):
        if meta is None:
            return None
        video_id = str(meta.get("video_id"))
        record = self.gate._load_record(video_id)  # noqa: SLF001 - reuse the existing gaze loader/sync logic
        if record is None:
            return None
        frame_indices = meta.get("frame_indices")
        if torch.is_tensor(frame_indices):
            frame_indices = frame_indices.detach().cpu().numpy()
        vfps = meta.get("vfps", 30.0)
        if torch.is_tensor(vfps):
            vfps = float(vfps.detach().cpu())
        h0 = int(meta.get("height", self.crop_size))
        w0 = int(meta.get("width", self.crop_size))
        return self.gate._query_crop_xy(record, frame_indices, vfps, h0, w0)  # noqa: SLF001


def binary_input_adapter_param_names(model: nn.Module) -> set[str]:
    model = unwrap_ddp(model)
    return {f"input_adapter.{name}" for name, _ in model.input_adapter.named_parameters()}


def trainable_binary_input_adapter_params(model: nn.Module):
    model = unwrap_ddp(model)
    return [param for param in model.input_adapter.parameters() if param.requires_grad]


def resolve_binary_input_map(
    model: nn.Module,
    map_builder,
    clips: torch.Tensor,
    metadata,
    binary_map: torch.Tensor | None,
) -> torch.Tensor:
    """Use dataloader map when channel count matches adapter; otherwise build online."""
    expected_aux = unwrap_ddp(model).input_adapter.aux_channels
    if binary_map is not None and int(binary_map.shape[1]) == expected_aux:
        return binary_map
    return map_builder.build(clips, metadata)


def _summarize_val_metrics(
    action_metrics: list[dict],
    verb_metrics: list[dict] | None,
    noun_metrics: list[dict] | None,
    metric_scope: str,
    metric_aggregation: str = "metric_wise_max",
    val_fixed_head_index: int | None = None,
) -> dict:
    return summarize_val_metrics(
        action_metrics,
        verb_metrics,
        noun_metrics,
        metric_scope,
        metric_aggregation=metric_aggregation,
        val_fixed_head_index=val_fixed_head_index,
    )


def train_one_epoch_with_binary_input_adapter_and_pose(
    base_eval,
    map_builder: BinaryGazeMapBuilder,
    pose_loader,
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
    """Binary gaze map input adapter + pose_rnn_fuse tokens injected at probe.

    This is a hybrid of:
    - `train_one_epoch_with_binary_input_adapter` (binary_map -> input_adapter)
    - `train_one_epoch_with_gaze` (pose_loader -> probe cross-attn extra K/V)
    """
    from app.hdepic_lora_action_anticipation.gaze_rnn import call_classifier, encode_fusion_tokens

    _data_loader = iter(data_loader)
    model_inner = unwrap_ddp(model)
    model_inner.base_model.eval()
    model_inner.input_adapter.train(mode=True)
    for c in classifiers:
        c.train(mode=True)

    if action_is_verb_noun:
        verb_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]

    data_elapsed_time_meter = AverageMeter()
    try:
        max_train_iters = int(os.environ.get("EVAL_MAX_TRAIN_ITERS", os.environ.get("MAX_TRAIN_ITERS", "0")) or "0")
    except ValueError:
        max_train_iters = 0
    grad_clip = _grad_clip_max_norm()
    if grad_clip > 0.0:
        logger.info("Using binary_input_adapter grad clip max_norm=%.3f", grad_clip)
    if max_train_iters > 0 and max_train_iters < ipe:
        logger.info("Limiting train_one_epoch_with_binary_input_adapter_and_pose to %d/%d iters via EVAL_MAX_TRAIN_ITERS", max_train_iters, ipe)
        ipe = max_train_iters

    successful_steps = 0
    for itr in range(ipe):
        itr_start_time = time.time()
        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)

        [s.step() for s in scheduler]
        [wds_.step() for wds_ in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = udata[0].to(device, non_blocking=True)
            metadata = udata[3] if len(udata) > 3 else None
            if metadata is None:
                raise ValueError("binary_input_adapter_and_pose requires metadata-aware dataloader")
            anticipation_times = udata[4].to(device, non_blocking=True)
            binary_map = udata[5].to(device, non_blocking=True) if len(udata) > 5 else None

            labels = labels_from_udata(udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes)

            data_elapsed_time_meter.update((time.time() - itr_start_time) * 1000.0)
            binary_map = resolve_binary_input_map(model, map_builder, clips, metadata, binary_map)

            tokens = model(clips, anticipation_times, binary_map=binary_map)
            if tokens is None:
                logger.warning("Skipping binary_input_adapter_and_pose step because encoder output is non-finite at itr=%d", itr)
                optimizer[0].zero_grad()
                continue

            # Pose tokens do NOT depend on `tokens_proxy`; we still need per-head pose encoder.
            pose_batch = pose_loader.load_batch(metadata, device)

            tokens_proxy = tokens.detach().requires_grad_(True)
            outputs = []
            for c in classifiers:
                fusion_tokens = encode_fusion_tokens(
                    c,
                    metadata,
                    device,
                    pose_loader=pose_loader,
                    pose_batch=pose_batch,
                    video_tokens=tokens,
                )
                outputs.append(call_classifier(c, tokens_proxy, fusion_tokens))

        if action_is_verb_noun:
            loss = [
                criterion(o["verb"], labels["verb"])
                + criterion(o["noun"], labels["noun"])
                + criterion(o["action"], labels["action"])
                for o in outputs
            ]
        else:
            loss = [criterion(o["action"], labels["action"]) for o in outputs]

        tokens_grad_accum = torch.zeros_like(tokens_proxy)
        healthy_heads = 0

        for head_idx, (l, c) in enumerate(zip(loss, classifiers)):
            if not torch.isfinite(l.detach()):
                logger.warning("Skipping per-head contribution because loss is non-finite: head=%d loss=%s", head_idx, float(l.detach().float()))
                _zero_classifier_grads(c)
                continue
            if tokens_proxy.grad is not None:
                tokens_proxy.grad.zero_()
            scaled = scaler[0].scale(l) if use_bfloat16 else l
            scaled.backward(retain_graph=(head_idx < len(loss) - 1))
            head_token_grad = tokens_proxy.grad
            if head_token_grad is None or not torch.isfinite(head_token_grad).all():

                _log_nonfinite_grad_diagnostics(
                    itr,
                    "head_tokens_grad_nonfinite",
                    use_bfloat16=use_bfloat16,
                    tokens_proxy_grad=head_token_grad,
                    classifiers=classifiers,
                    optimizer=optimizer,
                    head_idx=head_idx,
                )
                logger.warning("Discarding head %d gradient contribution because tokens grad is non-finite", head_idx)
                _zero_classifier_grads(c)
                continue
            head_param_ok = _classifier_grads_finite(c)
            if not head_param_ok:

                _log_nonfinite_grad_diagnostics(
                    itr,
                    "head_param_grad_nonfinite",
                    use_bfloat16=use_bfloat16,
                    tokens_proxy_grad=head_token_grad,
                    classifiers=classifiers,
                    optimizer=optimizer,
                    head_idx=head_idx,
                )
                logger.warning("Discarding head %d gradient contribution because head param grads are non-finite", head_idx)
                _zero_classifier_grads(c)
                continue

            tokens_grad_accum.add_(head_token_grad)
            healthy_heads += 1

        if healthy_heads == 0:

            _log_nonfinite_grad_diagnostics(
                itr,
                "all_heads_nonfinite",
                use_bfloat16=use_bfloat16,
                tokens_grad_accum=tokens_grad_accum,
                classifiers=classifiers,
                optimizer=optimizer,
            )
            logger.warning("All %d heads produced non-finite grads at itr=%d; skipping optimizer step", len(loss), itr)
            optimizer[0].zero_grad()
            if use_bfloat16:
                scaler[0].update()
            continue

        tokens_grad_accum.mul_(1.0 / float(healthy_heads))
        tokens.backward(gradient=tokens_grad_accum)

        if _should_log_grad_diag(itr):
            _log_grad_snapshot(
                itr,
                "proxy_post_token_backward",
                use_bfloat16=use_bfloat16,
                tokens_grad_accum=tokens_grad_accum,
                tokens_proxy_grad=tokens_proxy.grad,
                tokens_grad=tokens.grad,
                model=model,
                classifiers=classifiers,
                optimizer=optimizer,
            )


        adapter_ok = _adapter_grads_finite(model)
        if not adapter_ok:

            _log_nonfinite_grad_diagnostics(
                itr,
                "adapter_grad_nonfinite",
                use_bfloat16=use_bfloat16,
                tokens_grad_accum=tokens_grad_accum,
                tokens_grad=tokens.grad,
                model=model,
                classifiers=classifiers,
                optimizer=optimizer,
            )
            logger.warning("Discarding adapter step at itr=%d because adapter grads are non-finite after token backward", itr)
            _zero_adapter_grads(model)
        encoder_ok = _encoder_lora_grads_finite(model)
        if not encoder_ok:
            bad = _first_nonfinite_named_grad(trainable_encoder_lora_named_params(model))
            _log_nonfinite_grad_diagnostics(
                itr,
                "encoder_lora_grad_nonfinite",
                use_bfloat16=use_bfloat16,
                tokens_grad_accum=tokens_grad_accum,
                tokens_grad=tokens.grad,
                model=model,
                classifiers=classifiers,
                optimizer=optimizer,
            )
            if bad is None:
                logger.warning("Discarding encoder-LoRA grads at itr=%d because they are non-finite after token backward", itr)
            else:
                logger.warning(
                    "Discarding encoder-LoRA grads at itr=%d because they are non-finite after token backward; first_bad=%s bad_elems=%s",
                    itr,
                    bad[0],
                    bad[1],
                )
            _zero_encoder_lora_grads(model)
        if not _clip_optimizer_grads(optimizer, scaler, use_bfloat16, grad_clip, itr):
            if use_bfloat16:
                scaler[0].update()
            continue

        if use_bfloat16:
            scaler[0].step(optimizer[0])
            scaler[0].update()
        else:
            optimizer[0].step()
        optimizer[0].zero_grad()

        with torch.no_grad():
            action_metrics = [m(o["action"], labels["action"]) for o, m in zip(outputs, action_metric_loggers)]
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], labels["verb"]) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], labels["noun"]) for o, m in zip(outputs, noun_metric_loggers)]
        successful_steps += 1

        if itr % 10 == 0 or itr == ipe - 1:
            if action_is_verb_noun:
                logger.info(
                    "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) healthy_heads=%d/%d adapter_ok=%s [mem: %.2e] [data: %.1f ms]",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(v["accuracy"] for v in verb_metrics),
                    max(n["accuracy"] for n in noun_metrics),
                    max(a["recall"] for a in action_metrics),
                    max(v["recall"] for v in verb_metrics),
                    max(n["recall"] for n in noun_metrics),
                    healthy_heads,
                    len(loss),
                    adapter_ok,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    data_elapsed_time_meter.avg,
                )
            else:
                logger.info(
                    "[%5d] acc: %.1f%% recall: %.1f%% healthy_heads=%d/%d adapter_ok=%s [mem: %.2e] [data: %.1f ms]",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(a["recall"] for a in action_metrics),
                    healthy_heads,
                    len(loss),
                    adapter_ok,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    data_elapsed_time_meter.avg,
                )

    if successful_steps == 0:
        raise RuntimeError(
            "No finite optimizer steps completed in train_one_epoch_with_binary_input_adapter_and_pose; "
            "inspect preceding non-finite gradient diagnostics"
        )
    verb_arg = verb_metrics if action_is_verb_noun else None
    noun_arg = noun_metrics if action_is_verb_noun else None
    return summarize_metric_lists(action_metrics, verb_arg, noun_arg)


@torch.no_grad()
def validate_with_binary_input_adapter_and_pose(
    base_eval,
    map_builder: BinaryGazeMapBuilder,
    pose_loader,
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
    from app.hdepic_lora_action_anticipation.gaze_rnn import call_classifier, encode_fusion_tokens

    metric_scope = str(val_metric_scope).lower()
    if metric_scope not in {"native", "filtered"}:
        raise ValueError(f"Unsupported val_metric_scope={val_metric_scope!r}; expected native or filtered")
    use_valid_filter = metric_scope == "filtered"
    logger.info("Running val with binary input adapter + pose_rnn_fuse (metric_scope=%s)...", metric_scope)
    if use_valid_filter:
        logger.info("Using filtered val metrics: passing valid_* class sets into ClassMeanRecall")

    _data_loader = iter(data_loader)
    model_inner = unwrap_ddp(model)
    model_inner.base_model.eval()
    model_inner.input_adapter.eval()
    for c in classifiers:
        c.train(mode=False)

    if action_is_verb_noun:
        verb_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]

    for itr in range(ipe):
        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = udata[0].to(device, non_blocking=True)
            metadata = udata[3] if len(udata) > 4 else None
            if metadata is None:
                raise ValueError("binary_input_adapter_and_pose requires metadata-aware dataloader")
            anticipation_times = udata[4].to(device, non_blocking=True)
            binary_map = udata[5].to(device, non_blocking=True) if len(udata) > 5 else None
            labels = labels_from_udata(udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes)

            binary_map = resolve_binary_input_map(model, map_builder, clips, metadata, binary_map)

            tokens = model(clips, anticipation_times, binary_map=binary_map)
            if tokens is None:
                logger.warning("Skipping binary_input_adapter_and_pose val batch because encoder output is non-finite at itr=%d", itr)
                continue

            pose_batch = pose_loader.load_batch(metadata, device)

            outputs = []
            for c in classifiers:
                fusion_tokens = encode_fusion_tokens(
                    c,
                    metadata,
                    device,
                    pose_loader=pose_loader,
                    pose_batch=pose_batch,
                    video_tokens=tokens,
                )
                outputs.append(call_classifier(c, tokens, fusion_tokens))

            valid_actions_arg = valid_actions if use_valid_filter else None
            valid_verbs_arg = valid_verbs if use_valid_filter else None
            valid_nouns_arg = valid_nouns if use_valid_filter else None

            action_metrics = [m(o["action"], labels["action"], valid_actions_arg) for o, m in zip(outputs, action_metric_loggers)]
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], labels["verb"], valid_verbs_arg) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], labels["noun"], valid_nouns_arg) for o, m in zip(outputs, noun_metric_loggers)]
                verb_loss = sum(criterion(o["verb"], labels["verb"]) for o in outputs)
                noun_loss = sum(criterion(o["noun"], labels["noun"]) for o in outputs)
                action_loss = sum(criterion(o["action"], labels["action"]) for o in outputs)
                loss = verb_loss + noun_loss + action_loss
            else:
                loss = sum(criterion(o["action"], labels["action"]) for o in outputs)

        best_head_idx = max(range(len(action_metrics)), key=lambda i: action_metrics[i]["accuracy"])
        dumper.add_batch(udata, [outputs[best_head_idx]], labels, {"verb": verb_classes, "noun": noun_classes, "action": action_classes})

        if itr % 10 == 0 or itr == ipe - 1:
            if action_is_verb_noun:
                logger.info(
                    "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) loss (v/n): %.3f (%.3f %.3f) [mem: %.2e]",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(v["accuracy"] for v in verb_metrics),
                    max(n["accuracy"] for n in noun_metrics),
                    max(a["recall"] for a in action_metrics),
                    max(v["recall"] for v in verb_metrics),
                    max(n["recall"] for n in noun_metrics),
                    float(loss.detach().float()),
                    float(verb_loss.detach().float()),
                    float(noun_loss.detach().float()),
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )

    dumper.write()
    _save_trainable_sidecars(model, map_builder.adapter_checkpoint_path, map_builder.rank)

    # Note: keep return shape aligned with upstream binary validate
    verb_arg = verb_metrics if action_is_verb_noun else None
    noun_arg = noun_metrics if action_is_verb_noun else None
    return _summarize_val_metrics(
        action_metrics,
        verb_arg,
        noun_arg,
        metric_scope,
        metric_aggregation=val_metric_aggregation,
        val_fixed_head_index=val_fixed_head_index,
    )


def normalize_binary_input_adapter_grads(model: nn.Module, divisor: int):
    model = unwrap_ddp(model)
    if divisor <= 1:
        return
    scale = 1.0 / float(divisor)
    for param in model.input_adapter.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)


def _classifier_grads_finite(classifier: nn.Module) -> bool:
    for param in classifier.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


def _zero_classifier_grads(classifier: nn.Module) -> None:
    for param in classifier.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def _adapter_grads_finite(model: nn.Module) -> bool:
    model = unwrap_ddp(model)
    for param in model.input_adapter.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


def _zero_adapter_grads(model: nn.Module) -> None:
    model = unwrap_ddp(model)
    for param in model.input_adapter.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def _encoder_lora_grads_finite(model: nn.Module) -> bool:
    return encoder_lora_grads_finite(model)


def _zero_encoder_lora_grads(model: nn.Module) -> None:
    zero_encoder_lora_grads(model)


def _first_nonfinite_named_grad(named_params) -> tuple[str, str] | None:
    for name, param in named_params:
        if param.grad is None:
            continue
        finite = torch.isfinite(param.grad)
        if not finite.all():
            bad = int((~finite).sum().detach().cpu())
            total = param.grad.numel()
            return name, f"{bad}/{total}"
    return None


def _safe_grad_norm(grad: torch.Tensor | None) -> str:
    if grad is None:
        return "none"
    finite = torch.isfinite(grad)
    if not finite.all():
        bad = int((~finite).sum().detach().cpu())
        finite_vals = grad[finite]
        if finite_vals.numel() == 0:
            return f"nonfinite(all {bad}/{grad.numel()})"
        fnorm = float(torch.linalg.vector_norm(finite_vals.float()).detach().cpu())
        return f"nonfinite(bad={bad}/{grad.numel()},finite_norm={fnorm:.4e})"
    return f"{float(torch.linalg.vector_norm(grad.float()).detach().cpu()):.4e}"


def _params_grad_norm(params: list[nn.Parameter]) -> str:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return "none"
    flat = [g.float().reshape(-1) for g in grads]
    concat = torch.cat(flat)
    finite = torch.isfinite(concat)
    if not finite.all():
        bad = int((~finite).sum().detach().cpu())
        finite_vals = concat[finite]
        if finite_vals.numel() == 0:
            return f"nonfinite(all {bad}/{concat.numel()})"
        fnorm = float(torch.linalg.vector_norm(finite_vals).detach().cpu())
        return f"nonfinite(bad={bad}/{concat.numel()},finite_norm={fnorm:.4e})"
    return f"{float(torch.linalg.vector_norm(concat).detach().cpu()):.4e}"


def _grad_diag_interval() -> int:
    raw = os.environ.get("EVAL_GRAD_DIAG_INTERVAL", "10")
    try:
        return max(0, int(raw or "0"))
    except ValueError:
        return 10


def _should_log_grad_diag(itr: int) -> bool:
    if os.environ.get("EVAL_GRAD_DIAG_ALWAYS", "0").lower() in {"1", "true", "yes", "on"}:
        return True
    interval = _grad_diag_interval()
    return interval > 0 and (itr % interval == 0 or itr == 0)


def _use_direct_encoder_backward() -> bool:
    raw = os.environ.get(
        "BINARY_INPUT_ADAPTER_DIRECT_BACKWARD",
        os.environ.get("ENCODER_DIRECT_BACKWARD", "0"),
    )
    return raw.lower() in {"1", "true", "yes", "on"}


def _annotate_grad_norm(label: str, norm_str: str) -> str:
    if norm_str in {"none"}:
        return f"{label}: {norm_str}"
    if norm_str.startswith("nonfinite"):
        return f"{label}: {norm_str}"
    try:
        value = float(norm_str)
    except ValueError:
        return f"{label}: {norm_str}"
    if value > 1e4:
        tag = "EXPLOSIVE"
    elif value > 1e2:
        tag = "large"
    else:
        tag = "ok"
    return f"{label}: {norm_str} ({tag})"


def _log_grad_snapshot(
    itr: int,
    reason: str,
    *,
    use_bfloat16: bool,
    tokens_grad_accum: torch.Tensor | None = None,
    tokens_proxy_grad: torch.Tensor | None = None,
    tokens_grad: torch.Tensor | None = None,
    model: nn.Module | None = None,
    classifiers: list[nn.Module] | None = None,
    optimizer=None,
    head_idx: int | None = None,
) -> None:
    lines = [
        f"[grad-snapshot] itr={itr} reason={reason} use_bfloat16={use_bfloat16}"
        + (f" head={head_idx}" if head_idx is not None else "")
    ]
    if tokens_grad_accum is not None:
        lines.append("  " + _annotate_grad_norm("tokens_grad_accum", _safe_grad_norm(tokens_grad_accum)))
    if tokens_proxy_grad is not None:
        lines.append("  " + _annotate_grad_norm("tokens_proxy_grad", _safe_grad_norm(tokens_proxy_grad)))
    if tokens_grad is not None:
        lines.append("  " + _annotate_grad_norm("tokens.grad", _safe_grad_norm(tokens_grad)))

    if classifiers:
        for idx, classifier in enumerate(classifiers):
            params = [p for p in classifier.parameters() if p.grad is not None]
            lines.append("  " + _annotate_grad_norm(f"head[{idx}]", _params_grad_norm(params)))
            bad = _first_nonfinite_named_grad(classifier.named_parameters())
            if bad is not None:
                lines.append(f"    head[{idx}].first_bad={bad[0]} bad_elems={bad[1]}")

    if model is not None:
        model_inner = unwrap_ddp(model)
        if hasattr(model_inner, "input_adapter"):
            adapter_params = [p for p in model_inner.input_adapter.parameters() if p.grad is not None]
            lines.append("  " + _annotate_grad_norm("input_adapter", _params_grad_norm(adapter_params)))
            bad = _first_nonfinite_named_grad(model_inner.input_adapter.named_parameters())
            if bad is not None:
                lines.append(f"    input_adapter.first_bad={bad[0]} bad_elems={bad[1]}")

        enc_named = trainable_encoder_lora_named_params(model)
        enc_params = [p for _, p in enc_named if p.grad is not None]
        lines.append("  " + _annotate_grad_norm("encoder_lora", _params_grad_norm(enc_params)))
        bad = _first_nonfinite_named_grad(enc_named)
        if bad is not None:
            lines.append(f"    encoder_lora.first_bad={bad[0]} bad_elems={bad[1]}")

    if optimizer is not None:
        opt = optimizer[0] if isinstance(optimizer, (list, tuple)) else optimizer
        for idx, group in enumerate(opt.param_groups):
            label = group.get("diagnostic_name") or group.get("name") or f"group{idx}"
            params = [p for p in group["params"] if p.grad is not None]
            lines.append("  " + _annotate_grad_norm(f"opt[{label}]", _params_grad_norm(params)))

    logger.info("\n".join(lines))


def _log_nonfinite_grad_diagnostics(
    itr: int,
    reason: str,
    *,
    use_bfloat16: bool,
    tokens_grad_accum: torch.Tensor | None = None,
    tokens_proxy_grad: torch.Tensor | None = None,
    tokens_grad: torch.Tensor | None = None,
    model: nn.Module | None = None,
    classifiers: list[nn.Module] | None = None,
    optimizer=None,
    head_idx: int | None = None,
) -> None:
    lines = [
        f"[grad-diag] itr={itr} reason={reason} use_bfloat16={use_bfloat16}"
        + (f" head={head_idx}" if head_idx is not None else "")
    ]
    if tokens_grad_accum is not None:
        lines.append(f"  tokens_grad_accum: {_safe_grad_norm(tokens_grad_accum)}")
    if tokens_proxy_grad is not None:
        lines.append(f"  tokens_proxy_grad: {_safe_grad_norm(tokens_proxy_grad)}")
    if tokens_grad is not None:
        lines.append(f"  tokens.grad: {_safe_grad_norm(tokens_grad)}")

    if classifiers:
        for idx, classifier in enumerate(classifiers):
            params = [p for p in classifier.parameters() if p.grad is not None]
            lines.append(f"  head[{idx}]: {_params_grad_norm(params)}")
            bad = _first_nonfinite_named_grad(classifier.named_parameters())
            if bad is not None:
                lines.append(f"    head[{idx}].first_bad={bad[0]} bad_elems={bad[1]}")

    if model is not None:
        model_inner = unwrap_ddp(model)
        if hasattr(model_inner, "input_adapter"):
            adapter_params = [p for p in model_inner.input_adapter.parameters() if p.grad is not None]
            lines.append(f"  input_adapter: {_params_grad_norm(adapter_params)}")
            bad = _first_nonfinite_named_grad(model_inner.input_adapter.named_parameters())
            if bad is not None:
                lines.append(f"    input_adapter.first_bad={bad[0]} bad_elems={bad[1]}")

        enc_named = trainable_encoder_lora_named_params(model)
        enc_params = [p for _, p in enc_named if p.grad is not None]
        lines.append(f"  encoder_lora: {_params_grad_norm(enc_params)}")
        bad = _first_nonfinite_named_grad(enc_named)
        if bad is not None:
            lines.append(f"    encoder_lora.first_bad={bad[0]} bad_elems={bad[1]}")

    if optimizer is not None:
        opt = optimizer[0] if isinstance(optimizer, (list, tuple)) else optimizer
        for idx, group in enumerate(opt.param_groups):
            label = group.get("diagnostic_name") or group.get("name") or f"group{idx}"
            params = [p for p in group["params"] if p.grad is not None]
            lines.append(f"  opt[{label}]: {_params_grad_norm(params)}")

    logger.warning("\n".join(lines))


def _optimizer_nonfinite_group_summary(opt) -> str:
    summaries = []
    for idx, group in enumerate(opt.param_groups):
        label = group.get("diagnostic_name") or group.get("name") or f"group{idx}"
        with_grad = 0
        bad_params = 0
        bad_elems = 0
        total_elems = 0
        for param in group["params"]:
            if param.grad is None:
                continue
            with_grad += 1
            finite = torch.isfinite(param.grad)
            if not finite.all():
                bad_params += 1
                bad = int((~finite).sum().detach().cpu())
                bad_elems += bad
                total_elems += param.grad.numel()
        if bad_params:
            summaries.append(f"{label}:bad_params={bad_params}/{with_grad},bad_elems={bad_elems}/{total_elems}")
    return "; ".join(summaries) if summaries else "no per-parameter non-finite grads found"


def _grad_clip_max_norm() -> float:
    raw = os.environ.get("EVAL_GRAD_CLIP", os.environ.get("GRAD_CLIP", "0"))
    try:
        return float(raw or "0")
    except ValueError:
        logger.warning("Ignoring invalid EVAL_GRAD_CLIP=%r", raw)
        return 0.0


def _output_regularization_cfg() -> dict[str, Any]:
    raw_weight = os.environ.get("LORA_OUTPUT_REG_WEIGHT", os.environ.get("OUTPUT_REG_WEIGHT", "0"))
    try:
        weight = float(raw_weight or "0")
    except ValueError as exc:
        raise ValueError(f"Invalid LORA_OUTPUT_REG_WEIGHT={raw_weight!r}") from exc
    mode = os.environ.get("LORA_OUTPUT_REG_MODE", "confidence_penalty").strip().lower()
    if mode == "entropy":
        mode = "confidence_penalty"
    if mode not in {"confidence_penalty", "logit_l2"}:
        raise ValueError(
            f"Unsupported LORA_OUTPUT_REG_MODE={mode!r}; expected confidence_penalty or logit_l2"
        )
    targets_raw = os.environ.get("LORA_OUTPUT_REG_TARGETS", "verb,noun,action")
    targets = tuple(
        target.strip().lower()
        for target in targets_raw.replace("|", ",").split(",")
        if target.strip()
    )
    if any(target not in {"verb", "noun", "action"} for target in targets):
        raise ValueError(f"Invalid LORA_OUTPUT_REG_TARGETS={targets_raw!r}")
    return {"weight": weight, "mode": mode, "targets": targets}


def _regularize_output_loss(base_loss: torch.Tensor, output: dict[str, torch.Tensor], cfg: dict[str, Any]) -> torch.Tensor:
    weight = float(cfg.get("weight", 0.0) or 0.0)
    if weight == 0.0:
        return base_loss
    mode = str(cfg.get("mode", "confidence_penalty"))
    terms = []
    for target in cfg.get("targets", ()):
        logits = output.get(target)
        if logits is None:
            continue
        logits_f = logits.float()
        if mode == "confidence_penalty":
            log_probs = torch.log_softmax(logits_f, dim=-1)
            probs = log_probs.exp()
            terms.append((probs * log_probs).sum(dim=-1).mean())
        elif mode == "logit_l2":
            terms.append(logits_f.square().mean())
        else:
            raise ValueError(f"Unsupported output regularization mode={mode!r}")
    if not terms:
        return base_loss
    return base_loss + weight * torch.stack(terms).mean()


def _clip_optimizer_grads(optimizer, scaler, use_bfloat16: bool, max_norm: float, itr: int) -> bool:
    if max_norm <= 0.0:
        return True
    opt = optimizer[0]
    if use_bfloat16 and scaler and scaler[0] is not None:
        scaler[0].unscale_(opt)
    params = [p for group in opt.param_groups for p in group["params"] if p.grad is not None]
    if not params:
        return True
    bad_groups = _optimizer_nonfinite_group_summary(opt)
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=False)
    if not torch.isfinite(total_norm):
        _log_nonfinite_grad_diagnostics(
            itr,
            "clip_grad_nonfinite",
            use_bfloat16=use_bfloat16,
            model=None,
            classifiers=None,
            optimizer=optimizer,
        )
        logger.warning(
            "Skipping optimizer step at itr=%d because clipped grad norm is non-finite: %s; bad_grad_groups=%s",
            itr,
            float(total_norm),
            bad_groups,
        )
        opt.zero_grad()
        return False
    return True


def _use_direct_encoder_backward() -> bool:
    return os.environ.get(
        "BINARY_INPUT_ADAPTER_DIRECT_BACKWARD",
        os.environ.get("ENCODER_DIRECT_BACKWARD", "0"),
    ).lower() in {"1", "true", "yes", "on"}


def train_one_epoch_with_binary_input_adapter_direct(
    base_eval,
    map_builder: BinaryGazeMapBuilder,
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
    """JEPA_ARVR-style autograd with latency breakdown (single loss.backward)."""
    _data_loader = iter(data_loader)
    model_inner = unwrap_ddp(model)
    model_inner.base_model.train(mode=True)
    model_inner.input_adapter.train(mode=True)
    for c in classifiers:
        c.train(mode=True)
    if action_is_verb_noun:
        verb_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]
    data_elapsed_time_meter = AverageMeter()
    try:
        max_train_iters = int(os.environ.get("EVAL_MAX_TRAIN_ITERS", os.environ.get("MAX_TRAIN_ITERS", "0")) or "0")
    except ValueError:
        max_train_iters = 0
    grad_clip = _grad_clip_max_norm()
    logger.info("Using JEPA_ARVR-style direct encoder backward (no tokens_proxy detach)")
    if grad_clip > 0.0:
        logger.info("Using binary_input_adapter grad clip max_norm=%.3f", grad_clip)
    if max_train_iters > 0 and max_train_iters < ipe:
        logger.info(
            "Limiting train_one_epoch_with_binary_input_adapter_direct to %d/%d iterations via EVAL_MAX_TRAIN_ITERS",
            max_train_iters,
            ipe,
        )
        ipe = max_train_iters
    output_reg_cfg = _output_regularization_cfg()
    if float(output_reg_cfg["weight"]) != 0.0:
        logger.info(
            "Using output regularization: mode=%s weight=%g targets=%s",
            output_reg_cfg["mode"],
            output_reg_cfg["weight"],
            ",".join(output_reg_cfg["targets"]),
        )

    breakdown = LatencyBreakdown()
    if breakdown.enabled:
        logger.info("Binary-input-adapter direct-backward latency breakdown enabled (EVAL_LATENCY_BREAKDOWN=1)")
        instrument_model_for_breakdown(model, breakdown)

    successful_steps = 0
    for itr in range(ipe):
        itr_start_time = time.time()
        with breakdown.section("data_load", sync_before=False):
            try:
                udata = next(_data_loader)
            except Exception:
                _data_loader = iter(data_loader)
                udata = next(_data_loader)
        [s.step() for s in scheduler]
        [wds_.step() for wds_ in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            with breakdown.section("h2d"):
                clips = udata[0].to(device, non_blocking=True)
                metadata = udata[3] if len(udata) > 4 else None
                if metadata is None:
                    raise ValueError("binary_input_adapter requires metadata-aware dataloader")
                anticipation_times = udata[4].to(device, non_blocking=True)
                binary_map = udata[5].to(device, non_blocking=True) if len(udata) > 5 else None
                labels = labels_from_udata(udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes)
            data_elapsed_time_meter.update((time.time() - itr_start_time) * 1000.0)
            with breakdown.section("binary_map"):
                binary_map = resolve_binary_input_map(model, map_builder, clips, metadata, binary_map)
            with breakdown.section("fwd_model"):
                tokens = model(clips, anticipation_times, binary_map=binary_map)
            if tokens is None:
                logger.warning("Skipping direct-backward step because encoder output is non-finite at itr=%d", itr)
                optimizer[0].zero_grad()
                continue
            with breakdown.section("fwd_classifier"):
                outputs = [c(tokens) for c in classifiers]

        with breakdown.section("loss"):
            if action_is_verb_noun:
                loss = [
                    _regularize_output_loss(
                        criterion(o["verb"], labels["verb"])
                        + criterion(o["noun"], labels["noun"])
                        + criterion(o["action"], labels["action"]),
                        o,
                        output_reg_cfg,
                    )
                    for o in outputs
                ]
            else:
                loss = [
                    _regularize_output_loss(criterion(o["action"], labels["action"]), o, output_reg_cfg)
                    for o in outputs
                ]
            total_loss = sum(loss) / max(1, len(loss))

        if not torch.isfinite(total_loss.detach()):
            logger.warning("Skipping direct-backward step because loss is non-finite at itr=%d", itr)
            optimizer[0].zero_grad()
            if use_bfloat16:
                scaler[0].update()
            continue

        optimizer[0].zero_grad()
        with breakdown.section("bwd_total"):
            if use_bfloat16:
                scaler[0].scale(total_loss).backward()
            else:
                total_loss.backward()

        if _should_log_grad_diag(itr):
            _log_grad_snapshot(
                itr,
                "direct_post_backward",
                use_bfloat16=use_bfloat16,
                tokens_grad=tokens.grad,
                model=model,
                classifiers=classifiers,
                optimizer=optimizer,
            )

        adapter_ok = _adapter_grads_finite(model)
        if not adapter_ok:
            logger.warning("Discarding adapter grads at itr=%d (direct backward, non-finite)", itr)
            _log_nonfinite_grad_diagnostics(
                itr,
                "adapter_grad_nonfinite",
                use_bfloat16=use_bfloat16,
                tokens_grad=tokens.grad,
                model=model,
                classifiers=classifiers,
                optimizer=optimizer,
            )
            _zero_adapter_grads(model)
        encoder_ok = _encoder_lora_grads_finite(model)
        if not encoder_ok:
            bad = _first_nonfinite_named_grad(trainable_encoder_lora_named_params(model))
            _log_nonfinite_grad_diagnostics(
                itr,
                "encoder_lora_grad_nonfinite",
                use_bfloat16=use_bfloat16,
                tokens_grad=tokens.grad,
                model=model,
                classifiers=classifiers,
                optimizer=optimizer,
            )
            if bad is not None:
                logger.warning(
                    "Discarding encoder-LoRA grads at itr=%d (direct backward); first_bad=%s bad_elems=%s",
                    itr,
                    bad[0],
                    bad[1],
                )
            else:
                logger.warning("Discarding encoder-LoRA grads at itr=%d (direct backward, non-finite)", itr)
            _zero_encoder_lora_grads(model)
        with breakdown.section("grad_clip"):
            clip_ok = _clip_optimizer_grads(optimizer, scaler, use_bfloat16, grad_clip, itr)
        if not clip_ok:
            if use_bfloat16:
                scaler[0].update()
            continue

        with breakdown.section("optimizer"):
            if use_bfloat16:
                scaler[0].step(optimizer[0])
                scaler[0].update()
            else:
                optimizer[0].step()
            optimizer[0].zero_grad()

        with torch.no_grad():
            action_metrics = [m(o["action"], labels["action"]) for o, m in zip(outputs, action_metric_loggers)]
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], labels["verb"]) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], labels["noun"]) for o, m in zip(outputs, noun_metric_loggers)]
        successful_steps += 1
        breakdown.iter_wall_ms.update((time.time() - itr_start_time) * 1000.0)
        breakdown.log(itr, force=(itr == ipe - 1))
        if itr % 10 == 0 or itr == ipe - 1:
            if action_is_verb_noun:
                logger.info(
                    "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) adapter_ok=%s enc_ok=%s direct=1 [mem: %.2e] [data: %.1f ms]",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(v["accuracy"] for v in verb_metrics),
                    max(n["accuracy"] for n in noun_metrics),
                    max(a["recall"] for a in action_metrics),
                    max(v["recall"] for v in verb_metrics),
                    max(n["recall"] for n in noun_metrics),
                    adapter_ok,
                    encoder_ok,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    data_elapsed_time_meter.avg,
                )

    breakdown.write_report()
    if successful_steps == 0:
        raise RuntimeError(
            "No finite optimizer steps completed in train_one_epoch_with_binary_input_adapter_direct; "
            "inspect preceding gradient diagnostics"
        )
    verb_arg = verb_metrics if action_is_verb_noun else None
    noun_arg = noun_metrics if action_is_verb_noun else None
    return summarize_metric_lists(action_metrics, verb_arg, noun_arg)


def train_one_epoch_with_binary_input_adapter(
    base_eval,
    map_builder: BinaryGazeMapBuilder,
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
    if _use_direct_encoder_backward():
        return train_one_epoch_with_binary_input_adapter_direct(
            base_eval,
            map_builder,
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
        )
    _data_loader = iter(data_loader)
    model_inner = unwrap_ddp(model)
    model_inner.base_model.eval()
    model_inner.input_adapter.train(mode=True)
    for c in classifiers:
        c.train(mode=True)
    if action_is_verb_noun:
        verb_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]
    data_elapsed_time_meter = AverageMeter()
    try:
        max_train_iters = int(os.environ.get("EVAL_MAX_TRAIN_ITERS", os.environ.get("MAX_TRAIN_ITERS", "0")) or "0")
    except ValueError:
        max_train_iters = 0
    grad_clip = _grad_clip_max_norm()
    if grad_clip > 0.0:
        logger.info("Using binary_input_adapter grad clip max_norm=%.3f", grad_clip)
    if max_train_iters > 0 and max_train_iters < ipe:
        logger.info("Limiting train_one_epoch_with_binary_input_adapter to %d/%d iterations via EVAL_MAX_TRAIN_ITERS", max_train_iters, ipe)
        ipe = max_train_iters
    output_reg_cfg = _output_regularization_cfg()
    if float(output_reg_cfg["weight"]) != 0.0:
        logger.info(
            "Using output regularization: mode=%s weight=%g targets=%s",
            output_reg_cfg["mode"],
            output_reg_cfg["weight"],
            ",".join(output_reg_cfg["targets"]),
        )

    breakdown = LatencyBreakdown()
    if breakdown.enabled:
        logger.info("Binary-input-adapter latency breakdown enabled (EVAL_LATENCY_BREAKDOWN=1)")
        instrument_model_for_breakdown(model, breakdown)

    successful_steps = 0
    for itr in range(ipe):
        itr_start_time = time.time()
        with breakdown.section("data_load", sync_before=False):
            try:
                udata = next(_data_loader)
            except Exception:
                _data_loader = iter(data_loader)
                udata = next(_data_loader)
        [s.step() for s in scheduler]
        [wds_.step() for wds_ in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            with breakdown.section("h2d"):
                clips = udata[0].to(device, non_blocking=True)
                metadata = udata[3] if len(udata) > 4 else None
                if metadata is None:
                    raise ValueError("binary_input_adapter requires metadata-aware dataloader")
                anticipation_times = udata[4].to(device, non_blocking=True)
                binary_map = udata[5].to(device, non_blocking=True) if len(udata) > 5 else None
                labels = labels_from_udata(udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes)
            data_elapsed_time_meter.update((time.time() - itr_start_time) * 1000.0)
            with breakdown.section("binary_map"):
                binary_map = resolve_binary_input_map(model, map_builder, clips, metadata, binary_map)
            with breakdown.section("fwd_model"):
                tokens = model(clips, anticipation_times, binary_map=binary_map)
            if tokens is None:
                logger.warning("Skipping binary_input_adapter optimizer step because encoder output is non-finite at itr=%d", itr)
                optimizer[0].zero_grad()
                continue
            tokens_proxy = tokens.detach().requires_grad_(True)
            with breakdown.section("fwd_classifier"):
                outputs = [c(tokens_proxy) for c in classifiers]

        with breakdown.section("loss"):
            if action_is_verb_noun:
                loss = [
                    _regularize_output_loss(
                        criterion(o["verb"], labels["verb"])
                        + criterion(o["noun"], labels["noun"])
                        + criterion(o["action"], labels["action"]),
                        o,
                        output_reg_cfg,
                    )
                    for o in outputs
                ]
            else:
                loss = [
                    _regularize_output_loss(criterion(o["action"], labels["action"]), o, output_reg_cfg)
                    for o in outputs
                ]

        tokens_grad_accum = torch.zeros_like(tokens_proxy)
        healthy_heads = 0
        adapter_param_names = binary_input_adapter_param_names(model)
        with breakdown.section("bwd_probe"):
            for head_idx, (l, c) in enumerate(zip(loss, classifiers)):
                if not torch.isfinite(l.detach()):
                    logger.warning(
                        "Skipping per-head contribution because loss is non-finite: head=%d loss=%s",
                        head_idx,
                        float(l.detach().float()),
                    )
                    _zero_classifier_grads(c)
                    continue
                if tokens_proxy.grad is not None:
                    tokens_proxy.grad.zero_()
                scaled = scaler[0].scale(l) if use_bfloat16 else l
                scaled.backward(retain_graph=(head_idx < len(loss) - 1))
                head_token_grad = tokens_proxy.grad
                if head_token_grad is None or not torch.isfinite(head_token_grad).all():
                    logger.warning(
                        "Discarding head %d gradient contribution because tokens grad is non-finite",
                        head_idx,
                    )
                    _log_nonfinite_grad_diagnostics(
                        itr,
                        "head_tokens_grad_nonfinite",
                        use_bfloat16=use_bfloat16,
                        tokens_proxy_grad=head_token_grad,
                        classifiers=classifiers,
                        optimizer=optimizer,
                        head_idx=head_idx,
                    )
                    _zero_classifier_grads(c)
                    continue
                head_param_ok = _classifier_grads_finite(c)
                if not head_param_ok:
                    logger.warning(
                        "Discarding head %d gradient contribution because head param grads are non-finite",
                        head_idx,
                    )
                    _log_nonfinite_grad_diagnostics(
                        itr,
                        "head_param_grad_nonfinite",
                        use_bfloat16=use_bfloat16,
                        tokens_proxy_grad=head_token_grad,
                        classifiers=classifiers,
                        optimizer=optimizer,
                        head_idx=head_idx,
                    )
                    _zero_classifier_grads(c)
                    continue
                tokens_grad_accum.add_(head_token_grad)
                healthy_heads += 1

        if healthy_heads == 0:
            logger.warning("All %d heads produced non-finite grads at itr=%d; skipping optimizer step", len(loss), itr)
            _log_nonfinite_grad_diagnostics(
                itr,
                "all_heads_nonfinite",
                use_bfloat16=use_bfloat16,
                tokens_grad_accum=tokens_grad_accum,
                classifiers=classifiers,
                optimizer=optimizer,
            )
            optimizer[0].zero_grad()
            if use_bfloat16:
                scaler[0].update()
            continue

        tokens_grad_accum.mul_(1.0 / float(healthy_heads))
        with breakdown.section("bwd_encoder"):
            tokens.backward(gradient=tokens_grad_accum)

        if _should_log_grad_diag(itr):
            _log_grad_snapshot(
                itr,
                "proxy_post_token_backward",
                use_bfloat16=use_bfloat16,
                tokens_grad_accum=tokens_grad_accum,
                tokens_proxy_grad=tokens_proxy.grad,
                tokens_grad=tokens.grad,
                model=model,
                classifiers=classifiers,
                optimizer=optimizer,
            )

        adapter_ok = _adapter_grads_finite(model)
        if not adapter_ok:
            logger.warning("Discarding adapter step at itr=%d because adapter grads are non-finite after token backward", itr)
            _log_nonfinite_grad_diagnostics(
                itr,
                "adapter_grad_nonfinite",
                use_bfloat16=use_bfloat16,
                tokens_grad_accum=tokens_grad_accum,
                tokens_grad=tokens.grad,
                model=model,
                classifiers=classifiers,
                optimizer=optimizer,
            )
            _zero_adapter_grads(model)
        encoder_ok = _encoder_lora_grads_finite(model)
        if not encoder_ok:
            bad = _first_nonfinite_named_grad(trainable_encoder_lora_named_params(model))
            _log_nonfinite_grad_diagnostics(
                itr,
                "encoder_lora_grad_nonfinite",
                use_bfloat16=use_bfloat16,
                tokens_grad_accum=tokens_grad_accum,
                tokens_grad=tokens.grad,
                model=model,
                classifiers=classifiers,
                optimizer=optimizer,
            )
            if bad is None:
                logger.warning("Discarding encoder-LoRA grads at itr=%d because they are non-finite after token backward", itr)
            else:
                logger.warning(
                    "Discarding encoder-LoRA grads at itr=%d because they are non-finite after token backward; first_bad=%s bad_elems=%s",
                    itr,
                    bad[0],
                    bad[1],
                )
            _zero_encoder_lora_grads(model)
        with breakdown.section("grad_clip"):
            clip_ok = _clip_optimizer_grads(optimizer, scaler, use_bfloat16, grad_clip, itr)
        if not clip_ok:
            if use_bfloat16:
                scaler[0].update()
            continue

        with breakdown.section("optimizer"):
            if use_bfloat16:
                scaler[0].step(optimizer[0])
                scaler[0].update()
            else:
                optimizer[0].step()
            optimizer[0].zero_grad()

        with torch.no_grad():
            action_metrics = [m(o["action"], labels["action"]) for o, m in zip(outputs, action_metric_loggers)]
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], labels["verb"]) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], labels["noun"]) for o, m in zip(outputs, noun_metric_loggers)]
        successful_steps += 1
        breakdown.iter_wall_ms.update((time.time() - itr_start_time) * 1000.0)
        breakdown.log(itr, force=(itr == ipe - 1))
        if itr % 10 == 0 or itr == ipe - 1:
            if action_is_verb_noun:
                logger.info(
                    "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) healthy_heads=%d/%d adapter_ok=%s [mem: %.2e] [data: %.1f ms]",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(v["accuracy"] for v in verb_metrics),
                    max(n["accuracy"] for n in noun_metrics),
                    max(a["recall"] for a in action_metrics),
                    max(v["recall"] for v in verb_metrics),
                    max(n["recall"] for n in noun_metrics),
                    healthy_heads,
                    len(loss),
                    adapter_ok,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    data_elapsed_time_meter.avg,
                )

    breakdown.write_report()

    if successful_steps == 0:
        raise RuntimeError(
            "No finite optimizer steps completed in train_one_epoch_with_binary_input_adapter; "
            "inspect preceding non-finite gradient diagnostics"
        )
    verb_arg = verb_metrics if action_is_verb_noun else None
    noun_arg = noun_metrics if action_is_verb_noun else None
    return summarize_metric_lists(action_metrics, verb_arg, noun_arg)


@torch.no_grad()
def validate_with_binary_input_adapter(
    base_eval,
    map_builder: BinaryGazeMapBuilder,
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
    metric_scope = str(val_metric_scope).lower()
    if metric_scope not in {"native", "filtered"}:
        raise ValueError(f"Unsupported val_metric_scope={val_metric_scope!r}; expected native or filtered")
    use_valid_filter = metric_scope == "filtered"
    logger.info(
        "Running val with binary input adapter (metric_scope=%s, aggregation=%s)...",
        metric_scope,
        val_metric_aggregation,
    )
    if use_valid_filter:
        logger.info("Using filtered val metrics: passing valid_* class sets into ClassMeanRecall")
    _data_loader = iter(data_loader)
    model_inner = unwrap_ddp(model)
    model_inner.base_model.eval()
    model_inner.input_adapter.eval()
    for c in classifiers:
        c.train(mode=False)
    if action_is_verb_noun:
        verb_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]

    for itr in range(ipe):
        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = udata[0].to(device, non_blocking=True)
            metadata = udata[3] if len(udata) > 4 else None
            if metadata is None:
                raise ValueError("binary_input_adapter requires metadata-aware dataloader")
            anticipation_times = udata[4].to(device, non_blocking=True)
            binary_map = udata[5].to(device, non_blocking=True) if len(udata) > 5 else None
            labels = labels_from_udata(udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes)
            binary_map = resolve_binary_input_map(model, map_builder, clips, metadata, binary_map)
            tokens = model(clips, anticipation_times, binary_map=binary_map)
            if tokens is None:
                logger.warning("Skipping binary_input_adapter val batch because encoder output is non-finite at itr=%d", itr)
                continue
            outputs = [c(tokens) for c in classifiers]
            valid_actions_arg = valid_actions if use_valid_filter else None
            valid_verbs_arg = valid_verbs if use_valid_filter else None
            valid_nouns_arg = valid_nouns if use_valid_filter else None
            action_metrics = [m(o["action"], labels["action"], valid_actions_arg) for o, m in zip(outputs, action_metric_loggers)]
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], labels["verb"], valid_verbs_arg) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], labels["noun"], valid_nouns_arg) for o, m in zip(outputs, noun_metric_loggers)]
                verb_loss = sum(criterion(o["verb"], labels["verb"]) for o in outputs)
                noun_loss = sum(criterion(o["noun"], labels["noun"]) for o in outputs)
                action_loss = sum(criterion(o["action"], labels["action"]) for o in outputs)
                loss = verb_loss + noun_loss + action_loss
            else:
                loss = sum(criterion(o["action"], labels["action"]) for o in outputs)
        best_head_idx = max(range(len(action_metrics)), key=lambda i: action_metrics[i]["accuracy"])
        dumper.add_batch(udata, [outputs[best_head_idx]], labels, {"verb": verb_classes, "noun": noun_classes, "action": action_classes})
        if itr % 10 == 0 or itr == ipe - 1:
            if action_is_verb_noun:
                logger.info(
                    "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) loss (v/n): %.3f (%.3f %.3f) [mem: %.2e]",
                    itr,
                    max(a["accuracy"] for a in action_metrics),
                    max(v["accuracy"] for v in verb_metrics),
                    max(n["accuracy"] for n in noun_metrics),
                    max(a["recall"] for a in action_metrics),
                    max(v["recall"] for v in verb_metrics),
                    max(n["recall"] for n in noun_metrics),
                    loss,
                    verb_loss,
                    noun_loss,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
    dumper.write()
    _save_trainable_sidecars(model, map_builder.adapter_checkpoint_path, map_builder.rank)
    verb_arg = verb_metrics if action_is_verb_noun else None
    noun_arg = noun_metrics if action_is_verb_noun else None
    return _summarize_val_metrics(
        action_metrics,
        verb_arg,
        noun_arg,
        metric_scope,
        metric_aggregation=val_metric_aggregation,
        val_fixed_head_index=val_fixed_head_index,
    )
