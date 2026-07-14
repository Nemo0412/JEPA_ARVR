"""Predictor LoRA fine-tuning for HD-EPIC action anticipation.

Mirrors `encoder_lora.py`'s strategy (same `LoRALinear`, same default
rank/alpha/dropout/target_suffixes) but targets the V-JEPA2 predictor
(`model.predictor`) instead of the encoder. The predictor uses
`.predictor_blocks` (not `.blocks`) for its transformer stack, otherwise the
block internals (`attn.qkv`, `attn.proj`) are the same `Block` class as the
encoder.

This module intentionally does not share state with `encoder_lora.py` so the
two LoRA strategies can be enabled independently or together: a model can
carry both encoder-LoRA-flagged and predictor-LoRA-flagged `LoRALinear`
modules at once, distinguished by separate marker attributes.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation.encoder_lora import (
    _clip_optimizer_grads,
    _grad_clip_max_norm,
    _unwrap,
    keep_nonfinite_grads_enabled,
)
from app.hdepic_lora_action_anticipation.latency_breakdown import (
    LatencyBreakdown,
    instrument_model_for_breakdown,
)
from src.utils.logging import AverageMeter

logger = logging.getLogger(__name__)

# Predictor LoRA is a project-only feature (no JEPA_ARVR reference counterpart;
# the reference probe pipeline has no predictor rollout). Default kept consistent
# with the aligned encoder-LoRA target set: attention qkv + proj only.
_DEFAULT_TARGET_SUFFIXES = ("attn.qkv", "attn.proj")
_PREDICTOR_LORA_FLAG = "_is_predictor_lora"


def _find_predictor(model: nn.Module) -> nn.Module:
    inner = _unwrap(model)
    predictor = getattr(inner, "predictor", None)
    if predictor is None:
        raise AttributeError(
            "predictor_lora: could not locate `.predictor` on the model "
            f"(type={type(inner).__name__}); cannot inject predictor LoRA"
        )
    return predictor


def _iter_blocks(predictor: nn.Module):
    blocks = getattr(predictor, "predictor_blocks", None)
    if blocks is None:
        blocks = getattr(predictor, "blocks", None)
    if blocks is None:
        raise AttributeError("predictor_lora: predictor has no `.predictor_blocks`/`.blocks` ModuleList")
    return list(blocks)


def _get_submodule(block: nn.Module, dotted: str):
    obj = block
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not hasattr(obj, p):
            return None, None, None
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        return None, None, None
    return obj, leaf, getattr(obj, leaf)


def inject_predictor_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    last_n_blocks: int = 0,
    target_suffixes: Iterable[str] = _DEFAULT_TARGET_SUFFIXES,
) -> int:
    from app.hdepic_lora_action_anticipation.eval import LoRALinear

    # last_n_blocks < 0 means skip LoRA (used with full_ft_last_n_blocks for extra depth).
    if int(last_n_blocks) < 0:
        logger.info("predictor_lora.last_n_blocks=%d < 0; skipping LoRA injection", last_n_blocks)
        return 0

    predictor = _find_predictor(model)
    blocks = _iter_blocks(predictor)
    n_total = len(blocks)
    last_n = n_total if last_n_blocks <= 0 else min(int(last_n_blocks), n_total)
    target_block_idxs = range(n_total - last_n, n_total)
    target_suffixes = tuple(target_suffixes)

    wrapped = 0
    for bi in target_block_idxs:
        block = blocks[bi]
        for suffix in target_suffixes:
            parent, leaf, child = _get_submodule(block, suffix)
            if child is None or not isinstance(child, nn.Linear):
                continue
            lora = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(lora, _PREDICTOR_LORA_FLAG, True)
            setattr(parent, leaf, lora)
            wrapped += 1

    set_predictor_lora_trainable(model, trainable=True)
    n_trainable = sum(p.numel() for p in trainable_predictor_lora_params(model))
    logger.info(
        "Injected predictor LoRA into %d Linear layers across last %d/%d blocks "
        "(rank=%d alpha=%.1f dropout=%.3f); trainable predictor-LoRA params=%d",
        wrapped,
        last_n,
        n_total,
        rank,
        alpha,
        dropout,
        n_trainable,
    )
    if wrapped == 0:
        logger.warning(
            "predictor_lora: wrapped 0 layers -- check target_suffixes=%s against the predictor block structure",
            target_suffixes,
        )
    return wrapped


def copy_init_extra_predictor_blocks(model: nn.Module, n_pretrained: int = 12) -> int:
    """Warm-start extra predictor blocks (depth > checkpoint) from the last pretrained block."""
    predictor = _find_predictor(model)
    blocks = _iter_blocks(predictor)
    if len(blocks) <= int(n_pretrained):
        return 0
    src_sd = blocks[int(n_pretrained) - 1].state_dict()
    n_copied = 0
    for block in blocks[int(n_pretrained) :]:
        block.load_state_dict(src_sd, strict=True)
        n_copied += 1
    logger.info(
        "Copy-initialized %d extra predictor block(s) from block %d (total depth=%d)",
        n_copied,
        int(n_pretrained) - 1,
        len(blocks),
    )
    return n_copied


def set_predictor_full_ft_last_n(model: nn.Module, last_n: int) -> int:
    """Fully unfreeze predictor weights for fine-tuning.

    - ``last_n >= num_blocks`` (or a very large value): train the **entire**
      predictor (all blocks + embeds / heads).
    - ``0 < last_n < num_blocks``: freeze everything, then unfreeze only the
      last ``last_n`` blocks (legacy partial-FT path).
    """
    if last_n <= 0:
        return 0
    predictor = _find_predictor(model)
    blocks = _iter_blocks(predictor)
    n_total = len(blocks)
    last_n = int(last_n)
    if last_n >= n_total:
        n_params = 0
        for p in predictor.parameters():
            p.requires_grad = True
            n_params += p.numel()
        set_predictor_lora_trainable(model, trainable=True)
        logger.info(
            "Enabled full fine-tune on ENTIRE predictor (%d blocks, %d params)",
            n_total,
            n_params,
        )
        return n_params
    for p in predictor.parameters():
        p.requires_grad = False
    n_params = 0
    for block in blocks[n_total - last_n :]:
        for p in block.parameters():
            p.requires_grad = True
            n_params += p.numel()
    # Re-enable any LoRA adapters that may have been frozen above.
    set_predictor_lora_trainable(model, trainable=True)
    logger.info(
        "Enabled full fine-tune on last %d/%d predictor blocks (%d params)",
        last_n,
        n_total,
        n_params,
    )
    return n_params


def trainable_predictor_full_ft_params(model: nn.Module) -> list[nn.Parameter]:
    """Trainable non-LoRA predictor params (extra-depth full-FT blocks)."""
    predictor = _find_predictor(model)
    params: list[nn.Parameter] = []
    for name, p in predictor.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            continue
        params.append(p)
    return params


def _predictor_lora_modules(model: nn.Module):
    for name, module in _unwrap(model).named_modules():
        if getattr(module, _PREDICTOR_LORA_FLAG, False):
            yield name, module


def set_predictor_lora_trainable(model: nn.Module, trainable: bool = True) -> int:
    count = 0
    for _, module in _predictor_lora_modules(model):
        module.lora_A.weight.requires_grad = trainable
        module.lora_B.weight.requires_grad = trainable
        count += module.lora_A.weight.numel() + module.lora_B.weight.numel()
    return count


def trainable_predictor_lora_params(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for _, module in _predictor_lora_modules(model):
        for param in (module.lora_A.weight, module.lora_B.weight):
            if param.requires_grad:
                params.append(param)
    return params


def trainable_predictor_lora_named_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    params: list[tuple[str, nn.Parameter]] = []
    for name, module in _predictor_lora_modules(model):
        for child_name, param in (("lora_A.weight", module.lora_A.weight), ("lora_B.weight", module.lora_B.weight)):
            if param.requires_grad:
                params.append((f"{name}.{child_name}", param))
    return params


def predictor_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in _predictor_lora_modules(model):
        state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
        state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def save_predictor_lora_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> int:
    state = predictor_lora_state_dict(model)
    if not state:
        return 0
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"predictor_lora": state}, path)
    logger.info("Wrote predictor LoRA checkpoint: %s", path)
    return len(state)


def load_predictor_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor], strict: bool = False):
    modules = dict(_unwrap(model).named_modules())
    missing: list[str] = []
    unexpected: list[str] = []
    used: set[str] = set()
    for module_name, module in modules.items():
        if not getattr(module, _PREDICTOR_LORA_FLAG, False):
            continue
        for leaf, param in (("lora_A.weight", module.lora_A.weight), ("lora_B.weight", module.lora_B.weight)):
            key = f"{module_name}.{leaf}"
            value = state.get(key)
            if value is None or tuple(value.shape) != tuple(param.shape):
                missing.append(key)
                continue
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))
            used.add(key)
    for key in state:
        if key not in used:
            unexpected.append(key)
    if strict and (missing or unexpected):
        raise RuntimeError(f"predictor LoRA checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return missing, unexpected


def load_predictor_lora_checkpoint(model: nn.Module, checkpoint_path: str | Path, strict: bool = False):
    checkpoint = torch.load(Path(checkpoint_path), map_location=torch.device("cpu"))
    state = checkpoint.get("predictor_lora", checkpoint)
    return load_predictor_lora_state_dict(model, state, strict=strict)


def assert_predictor_lora_device_consistency(model: nn.Module) -> None:
    for name, module in _predictor_lora_modules(model):
        base_device = module.base.weight.device
        for child_name, param in module.named_parameters():
            if child_name.startswith("base."):
                continue
            if param.device != base_device:
                raise RuntimeError(
                    f"predictor LoRA parameter {name}.{child_name} is on {param.device}, "
                    f"but base Linear is on {base_device}"
                )


def predictor_lora_grads_finite(model: nn.Module) -> bool:
    for p in trainable_predictor_lora_params(model):
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def zero_predictor_lora_grads(model: nn.Module) -> None:
    for p in trainable_predictor_lora_params(model):
        if p.grad is not None:
            p.grad.zero_()


def parse_predictor_lora_cfg(lora_cfg: dict) -> dict | None:
    cfg = dict(lora_cfg.get("predictor_lora", {}) or {})
    env = os.environ.get("PREDICTOR_LORA_ENABLED")
    if env is not None:
        enabled = env.lower() in {"1", "true", "yes", "on"}
    else:
        enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return None
    parsed = {
        "rank": int(cfg.get("rank", lora_cfg.get("rank", 8))),
        "alpha": float(cfg.get("alpha", lora_cfg.get("alpha", 16.0))),
        "dropout": float(cfg.get("dropout", lora_cfg.get("dropout", 0.05))),
        "last_n_blocks": int(cfg.get("last_n_blocks", 0)),
        "full_ft_last_n_blocks": int(cfg.get("full_ft_last_n_blocks", 0) or 0),
        "copy_init_from_pretrained": int(cfg.get("copy_init_from_pretrained", 12) or 0),
        "lr_mult": float(cfg.get("lr_mult", 0.5)),
        "weight_decay": float(cfg.get("weight_decay", 0.0001)),
        "activation_checkpointing": bool(cfg.get("activation_checkpointing", False)),
        "freeze": bool(cfg.get("freeze", False)),
        "warm_start_at_init": bool(cfg.get("warm_start_at_init", False)),
    }
    target_suffixes = cfg.get("target_suffixes")
    if isinstance(target_suffixes, str):
        target_suffixes = [part.strip() for part in target_suffixes.split(",") if part.strip()]
    if target_suffixes:
        parsed["target_suffixes"] = tuple(str(suffix) for suffix in target_suffixes)
    for key in ("checkpoint_path", "load_checkpoint_path"):
        if key in cfg:
            parsed[key] = cfg[key]
    return parsed


def train_one_epoch_predictor_lora(
    base_eval,
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
    """Same training-loop shape as `train_one_epoch_encoder_lora`, but the
    non-finite-grad discard/zero hooks check predictor-LoRA params instead of
    encoder-LoRA params. Kept as a separate function (rather than a generic
    `target=` parameter on the encoder-LoRA loop) so encoder-LoRA runs are
    untouched by this change.
    """
    model_inner = _unwrap(model)
    model_inner.train(mode=True)
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
        logger.info("Using predictor-LoRA grad clip max_norm=%.3f", grad_clip)
    if max_train_iters > 0 and max_train_iters < ipe:
        logger.info("Limiting train_one_epoch_predictor_lora to %d/%d iterations via EVAL_MAX_TRAIN_ITERS", max_train_iters, ipe)
        ipe = max_train_iters

    breakdown = LatencyBreakdown()
    if breakdown.enabled:
        logger.info("Predictor-LoRA latency breakdown enabled (EVAL_LATENCY_BREAKDOWN=1)")
        instrument_model_for_breakdown(model, breakdown)

    from app.hdepic_lora_action_anticipation.data_prefetch import DataLoaderPrefetcher

    prefetcher = DataLoaderPrefetcher(data_loader, name="pred-lora-prefetch")
    try:
        for itr in range(ipe):
            itr_start_time = time.time()
            with breakdown.section("data_load", sync_before=False):
                udata, fetch_ms = prefetcher.get()
                data_elapsed_time_meter.update(float(fetch_ms))

            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                with breakdown.section("h2d"):
                    clips = udata[0].to(device, non_blocking=True)
                    anticipation_times = udata[-1].to(device, non_blocking=True)

                    if action_is_verb_noun:
                        _verbs, _nouns = udata[1], udata[2]
                        verb_labels, noun_labels, action_labels = [], [], []
                        for v, n in zip(_verbs, _nouns):
                            verb_labels.append(verb_classes[int(v)])
                            noun_labels.append(noun_classes[int(n)])
                            action_labels.append(action_classes[(int(v), int(n))])
                        verb_labels = torch.tensor(verb_labels).to(device).to(_verbs.dtype)
                        noun_labels = torch.tensor(noun_labels).to(device).to(_nouns.dtype)
                        action_labels = torch.tensor(action_labels).to(device).to(_verbs.dtype)
                    else:
                        _actions = udata[1]
                        action_labels = [action_classes[str(int(a))] for a in _actions]
                        action_labels = torch.tensor(action_labels).to(device).to(_actions.dtype)

                with breakdown.section("fwd_model"):
                    outputs_tokens = model(clips, anticipation_times)
                with breakdown.section("fwd_classifier"):
                    outputs = [c(outputs_tokens) for c in classifiers]

            with breakdown.section("loss"):
                if action_is_verb_noun:
                    loss = [
                        criterion(o["verb"], verb_labels)
                        + criterion(o["noun"], noun_labels)
                        + criterion(o["action"], action_labels)
                        for o in outputs
                    ]
                else:
                    loss = [criterion(o["action"], action_labels) for o in outputs]
                total_loss = sum(loss) / max(1, len(loss))

            if not torch.isfinite(total_loss.detach()):
                logger.warning("Skipping optimizer step at itr=%d because loss is non-finite", itr)
                optimizer[0].zero_grad()
                if use_bfloat16:
                    scaler[0].update()
                continue

            with breakdown.section("bwd_total"):
                if use_bfloat16:
                    scaler[0].scale(total_loss).backward()
                else:
                    total_loss.backward()

            from app.hdepic_lora_action_anticipation.binary_input_adapter import (
                _log_grad_snapshot,
                _should_log_grad_diag,
            )

            if _should_log_grad_diag(itr):
                _log_grad_snapshot(
                    itr,
                    "baseline_post_backward",
                    use_bfloat16=use_bfloat16,
                    tokens_grad=outputs_tokens.grad,
                    model=model,
                    classifiers=classifiers,
                    optimizer=optimizer,
                )

            pred_ok = predictor_lora_grads_finite(model)
            if not pred_ok:
                if keep_nonfinite_grads_enabled():
                    logger.warning(
                        "Keeping non-finite predictor-LoRA grads at itr=%d (EVAL_KEEP_NONFINITE_GRADS=1)",
                        itr,
                    )
                else:
                    logger.warning("Discarding predictor-LoRA grads at itr=%d (non-finite)", itr)
                    zero_predictor_lora_grads(model)
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
                action_metrics = [
                    m(o["action"], action_labels) for o, m in zip(outputs, action_metric_loggers)
                ]
                if action_is_verb_noun:
                    verb_metrics = [
                        m(o["verb"], verb_labels) for o, m in zip(outputs, verb_metric_loggers)
                    ]
                    noun_metrics = [
                        m(o["noun"], noun_labels) for o, m in zip(outputs, noun_metric_loggers)
                    ]

            breakdown.iter_wall_ms.update((time.time() - itr_start_time) * 1000.0)
            breakdown.log(itr, force=(itr == ipe - 1))

            if itr % 10 == 0 or itr == ipe - 1:
                step_ms = (time.time() - itr_start_time) * 1000.0
                if action_is_verb_noun:
                    logger.info(
                        "[%5d] loss=%.4f acc (a/v/n): %.1f%% %.1f%% %.1f%% "
                        "recall (a/v/n): %.1f%% %.1f%% %.1f%% pred_lora_ok=%s "
                        "[mem: %.2e] [fetch: %.1f ms] [step: %.0f ms]",
                        itr,
                        float(total_loss.detach().float()),
                        max(a["accuracy"] for a in action_metrics),
                        max(v["accuracy"] for v in verb_metrics),
                        max(n["accuracy"] for n in noun_metrics),
                        max(a["recall"] for a in action_metrics),
                        max(v["recall"] for v in verb_metrics),
                        max(n["recall"] for n in noun_metrics),
                        pred_ok,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                        data_elapsed_time_meter.avg,
                        step_ms,
                    )
                else:
                    logger.info(
                        "[%5d] loss=%.4f acc: %.1f%% recall: %.1f%% pred_lora_ok=%s "
                        "[mem: %.2e] [fetch: %.1f ms] [step: %.0f ms]",
                        itr,
                        float(total_loss.detach().float()),
                        max(a["accuracy"] for a in action_metrics),
                        max(a["recall"] for a in action_metrics),
                        pred_ok,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                        data_elapsed_time_meter.avg,
                        step_ms,
                    )
    finally:
        prefetcher.close()

    breakdown.write_report()

    from app.hdepic_lora_action_anticipation.val_metrics import summarize_metric_lists

    return summarize_metric_lists(
        action_metrics,
        verb_metrics if action_is_verb_noun else None,
        noun_metrics if action_is_verb_noun else None,
    )
