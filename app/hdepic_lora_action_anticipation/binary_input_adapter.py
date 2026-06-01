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
from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate
from app.hdepic_lora_action_anticipation.gaze import labels_from_udata
from src.utils.logging import AverageMeter

logger = logging.getLogger(__name__)


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


class BinaryMapInputAdapter(nn.Module):
    """Tiny residual adapter that maps RGB + binary gaze map back to RGB space.

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
    ):
        super().__init__()
        self.scale = float(scale)
        self.binary_center = float(binary_center)
        self.residual_clamp = float(residual_clamp)
        tk = int(temporal_kernel)
        if tk not in {1, 3}:
            raise ValueError(f"Unsupported temporal_kernel={temporal_kernel}; expected 1 or 3")
        padding = (tk // 2, 1, 1)
        self.net = nn.Sequential(
            nn.Conv3d(4, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(tk, 3, 3), padding=padding, groups=hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 3, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rgb: torch.Tensor, binary_map: torch.Tensor) -> torch.Tensor:
        binary_map = binary_map.to(dtype=rgb.dtype, device=rgb.device) - self.binary_center
        x = torch.cat([rgb, binary_map], dim=1)
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
    if max_train_iters > 0 and max_train_iters < ipe:
        logger.info("Limiting train_one_epoch_with_binary_input_adapter to %d/%d iterations via EVAL_MAX_TRAIN_ITERS", max_train_iters, ipe)
        ipe = max_train_iters

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
            metadata = udata[3] if len(udata) > 4 else None
            if metadata is None:
                raise ValueError("binary_input_adapter requires metadata-aware dataloader")
            anticipation_times = udata[4].to(device, non_blocking=True)
            binary_map = udata[5].to(device, non_blocking=True) if len(udata) > 5 else None
            labels = labels_from_udata(udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes)
            data_elapsed_time_meter.update((time.time() - itr_start_time) * 1000.0)
            if binary_map is None:
                binary_map = map_builder.build(clips, metadata)
            tokens = model(clips, anticipation_times, binary_map=binary_map)
            if tokens is None:
                logger.warning("Skipping binary_input_adapter optimizer step because encoder output is non-finite at itr=%d", itr)
                optimizer[0].zero_grad()
                continue
            tokens_proxy = tokens.detach().requires_grad_(True)
            outputs = [c(tokens_proxy) for c in classifiers]

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
        adapter_param_names = binary_input_adapter_param_names(model)
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
                _zero_classifier_grads(c)
                continue
            head_param_ok = _classifier_grads_finite(c)
            if not head_param_ok:
                logger.warning(
                    "Discarding head %d gradient contribution because head param grads are non-finite",
                    head_idx,
                )
                _zero_classifier_grads(c)
                continue
            tokens_grad_accum.add_(head_token_grad)
            healthy_heads += 1

        if healthy_heads == 0:
            logger.warning("All %d heads produced non-finite grads at itr=%d; skipping optimizer step", len(loss), itr)
            optimizer[0].zero_grad()
            if use_bfloat16:
                scaler[0].update()
            continue

        tokens_grad_accum.mul_(1.0 / float(healthy_heads))
        tokens.backward(gradient=tokens_grad_accum)

        adapter_ok = _adapter_grads_finite(model)
        if not adapter_ok:
            logger.warning("Discarding adapter step at itr=%d because adapter grads are non-finite after token backward", itr)
            _zero_adapter_grads(model)

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

    ret = {"action": {"accuracy": max(a["accuracy"] for a in action_metrics), "recall": max(a["recall"] for a in action_metrics)}}
    if action_is_verb_noun:
        ret.update(
            {
                "verb": {"accuracy": max(v["accuracy"] for v in verb_metrics), "recall": max(v["recall"] for v in verb_metrics)},
                "noun": {"accuracy": max(n["accuracy"] for n in noun_metrics), "recall": max(n["recall"] for n in noun_metrics)},
            }
        )
    return ret


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
):
    metric_scope = str(val_metric_scope).lower()
    if metric_scope not in {"native", "filtered"}:
        raise ValueError(f"Unsupported val_metric_scope={val_metric_scope!r}; expected native or filtered")
    use_valid_filter = metric_scope == "filtered"
    logger.info("Running val with binary input adapter (metric_scope=%s)...", metric_scope)
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
            if binary_map is None:
                binary_map = map_builder.build(clips, metadata)
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
    if map_builder.adapter_checkpoint_path and map_builder.rank == 0:
        path = Path(map_builder.adapter_checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"input_adapter": unwrap_ddp(model).input_adapter.state_dict()}, path)
        logger.info("Wrote binary input adapter checkpoint: %s", path)
    ret = {"action": {"accuracy": max(a["accuracy"] for a in action_metrics), "recall": max(a["recall"] for a in action_metrics)}}
    ret["metric_scope"] = metric_scope
    if action_is_verb_noun:
        ret.update(
            {
                "verb": {"accuracy": max(v["accuracy"] for v in verb_metrics), "recall": max(v["recall"] for v in verb_metrics)},
                "noun": {"accuracy": max(n["accuracy"] for n in noun_metrics), "recall": max(n["recall"] for n in noun_metrics)},
            }
        )
    return ret
