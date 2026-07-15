"""Encoder (trunk) LoRA fine-tuning for HD-EPIC action anticipation."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation.latency_breakdown import (
    LatencyBreakdown,
    instrument_model_for_breakdown,
)
from src.utils.logging import AverageMeter

logger = logging.getLogger(__name__)

# PhD-reference / JEPA_ARVR aligned default: inject LoRA only into attention
# qkv + proj (NOT the MLP). See docs/PHD_CODE_COMPARISON.md "Encoder-LoRA
# hyper-parameter table". The legacy ViT-G default also wrapped mlp.fc1/fc2.
_DEFAULT_TARGET_SUFFIXES = ("attn.qkv", "attn.proj")
_ENCODER_LORA_FLAG = "_is_encoder_lora"


def _unwrap(model: nn.Module) -> nn.Module:
    inner = model
    if hasattr(inner, "module") and not hasattr(inner, "encoder"):
        inner = inner.module
    if hasattr(inner, "base_model"):
        inner = inner.base_model
    if hasattr(inner, "module") and not hasattr(inner, "encoder"):
        inner = inner.module
    return inner


def _find_encoder(model: nn.Module) -> nn.Module:
    inner = _unwrap(model)
    encoder = getattr(inner, "encoder", None)
    if encoder is None:
        raise AttributeError(
            "encoder_lora: could not locate `.encoder` on the model "
            f"(type={type(inner).__name__}); cannot inject encoder LoRA"
        )
    return encoder


def _iter_blocks(encoder: nn.Module):
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise AttributeError("encoder_lora: encoder has no `.blocks` ModuleList")
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


def apply_encoder_arch_depth(model: nn.Module, depth: int) -> int:
    """Force encoder ``blocks`` length to ``depth`` (keep first ``depth`` blocks).

    Used for −n layer drop after the native-depth encoder and its full pretrained
    checkpoint have been loaded. The pinned upstream ``vit_large`` factories
    hard-code their native depth, so shortening configs must not also pass a
    ``depth`` keyword to the factory. Deepening (+n) requires a factory that can
    build the requested depth, followed by ``copy_init``.
    """
    depth = int(depth)
    if depth <= 0:
        return 0
    encoder = _find_encoder(model)
    blocks = _iter_blocks(encoder)
    n_before = len(blocks)
    if depth == n_before:
        logger.info("Encoder arch depth already %d", depth)
        return depth
    if depth > n_before:
        raise ValueError(
            f"Cannot deepen encoder from {n_before} to {depth} via truncate; "
            "first add/use a factory that supports the requested depth, then "
            "set copy_init_from_pretrained"
        )
    encoder.blocks = nn.ModuleList(blocks[:depth])
    logger.info("Truncated encoder blocks %d -> %d (dropped last %d)", n_before, depth, n_before - depth)
    return depth


def copy_init_extra_encoder_blocks(model: nn.Module, n_pretrained: int = 24) -> int:
    """Warm-start extra encoder blocks (depth > checkpoint) from the last pretrained block.

    Mirrors ``copy_init_extra_predictor_blocks``. For −n (truncation), load with
    ``strict=False`` and skip this helper (``n_pretrained`` >= current depth).
    """
    encoder = _find_encoder(model)
    blocks = _iter_blocks(encoder)
    if len(blocks) <= int(n_pretrained):
        return 0
    src_sd = blocks[int(n_pretrained) - 1].state_dict()
    n_copied = 0
    for block in blocks[int(n_pretrained) :]:
        block.load_state_dict(src_sd, strict=True)
        n_copied += 1
    logger.info(
        "Copy-initialized %d extra encoder block(s) from block %d (total depth=%d)",
        n_copied,
        int(n_pretrained) - 1,
        len(blocks),
    )
    return n_copied


def inject_encoder_lora(
    model: nn.Module,
    rank: int = 8,        # PhD-reference / JEPA_ARVR aligned default (was legacy ViT-G 16)
    alpha: float = 16.0,  # PhD-reference / JEPA_ARVR aligned default (was legacy ViT-G 32)
    dropout: float = 0.05,  # project regularization; reference uses 0.0 (no dropout)
    last_n_blocks: int = 0,  # 0/<=0 => all blocks (reference injects every block); was legacy 12
    target_suffixes: Iterable[str] = _DEFAULT_TARGET_SUFFIXES,
) -> int:
    from app.hdepic_lora_action_anticipation.eval import LoRALinear

    encoder = _find_encoder(model)
    blocks = _iter_blocks(encoder)
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
            setattr(lora, _ENCODER_LORA_FLAG, True)
            setattr(parent, leaf, lora)
            wrapped += 1

    set_encoder_lora_trainable(model, trainable=True)
    n_trainable = sum(p.numel() for p in trainable_encoder_lora_params(model))
    logger.info(
        "Injected encoder LoRA into %d Linear layers across last %d/%d blocks "
        "(rank=%d alpha=%.1f dropout=%.3f); trainable encoder-LoRA params=%d",
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
            "encoder_lora: wrapped 0 layers -- check target_suffixes=%s against the encoder block structure",
            target_suffixes,
        )
    return wrapped


def _encoder_lora_modules(model: nn.Module):
    for name, module in _unwrap(model).named_modules():
        if getattr(module, _ENCODER_LORA_FLAG, False):
            yield name, module


def set_encoder_lora_trainable(model: nn.Module, trainable: bool = True) -> int:
    count = 0
    for _, module in _encoder_lora_modules(model):
        module.lora_A.weight.requires_grad = trainable
        module.lora_B.weight.requires_grad = trainable
        count += module.lora_A.weight.numel() + module.lora_B.weight.numel()
    return count


def trainable_encoder_lora_params(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for _, module in _encoder_lora_modules(model):
        for param in (module.lora_A.weight, module.lora_B.weight):
            if param.requires_grad:
                params.append(param)
    return params


def trainable_encoder_lora_named_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    params: list[tuple[str, nn.Parameter]] = []
    for name, module in _encoder_lora_modules(model):
        for child_name, param in (("lora_A.weight", module.lora_A.weight), ("lora_B.weight", module.lora_B.weight)):
            if param.requires_grad:
                params.append((f"{name}.{child_name}", param))
    return params


def encoder_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in _encoder_lora_modules(model):
        state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
        state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def save_encoder_lora_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> int:
    state = encoder_lora_state_dict(model)
    if not state:
        return 0
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_lora": state}, path)
    logger.info("Wrote encoder LoRA checkpoint: %s", path)
    return len(state)


def load_encoder_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor], strict: bool = False):
    modules = dict(_unwrap(model).named_modules())
    missing: list[str] = []
    unexpected: list[str] = []
    used: set[str] = set()
    for module_name, module in modules.items():
        if not getattr(module, _ENCODER_LORA_FLAG, False):
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
        raise RuntimeError(f"encoder LoRA checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return missing, unexpected


def load_encoder_lora_checkpoint(model: nn.Module, checkpoint_path: str | Path, strict: bool = False):
    checkpoint = torch.load(Path(checkpoint_path), map_location=torch.device("cpu"))
    state = checkpoint.get("encoder_lora", checkpoint)
    return load_encoder_lora_state_dict(model, state, strict=strict)


def assert_encoder_lora_device_consistency(model: nn.Module) -> None:
    for name, module in _encoder_lora_modules(model):
        base_device = module.base.weight.device
        for child_name, param in module.named_parameters():
            if child_name.startswith("base."):
                continue
            if param.device != base_device:
                raise RuntimeError(
                    f"encoder LoRA parameter {name}.{child_name} is on {param.device}, "
                    f"but base Linear is on {base_device}"
                )


def encoder_lora_grads_finite(model: nn.Module) -> bool:
    for p in trainable_encoder_lora_params(model):
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def zero_encoder_lora_grads(model: nn.Module) -> None:
    for p in trainable_encoder_lora_params(model):
        if p.grad is not None:
            p.grad.zero_()


def keep_nonfinite_grads_enabled() -> bool:
    return os.environ.get("EVAL_KEEP_NONFINITE_GRADS", "0").lower() in {"1", "true", "yes", "on"}


def make_grad_scaler(use_bfloat16: bool):
    """Build the AMP GradScaler for a training loop.

    GradScaler exists to prevent fp16 gradient *underflow* via loss scaling
    (default init_scale=65536). bf16 shares fp32's 8-bit exponent range, so it
    needs no loss scaling; scaling a large initial loss by 65536 instead pushes
    the deep-encoder backward into bf16 overflow -> inf upstream grads, which
    become NaN in zero-initialised LoRA-A grads (0 * inf). So under bf16 we
    return a disabled scaler (all scale/step/update calls pass through).

    Set EVAL_BF16_GRAD_SCALER=1 to restore the legacy enabled-scaler behaviour
    for A/B comparison.
    """
    if not use_bfloat16:
        return None
    legacy = os.environ.get("EVAL_BF16_GRAD_SCALER", "0").lower() in {"1", "true", "yes", "on"}
    if legacy:
        logger.warning("EVAL_BF16_GRAD_SCALER=1: using enabled GradScaler under bf16 (legacy/overflow-prone)")
        return torch.cuda.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=False)


def _grad_clip_max_norm() -> float:
    raw = os.environ.get("EVAL_GRAD_CLIP", os.environ.get("GRAD_CLIP", "0"))
    try:
        return float(raw or "0")
    except ValueError:
        logger.warning("Ignoring invalid EVAL_GRAD_CLIP=%r", raw)
        return 0.0


def _clip_optimizer_grads(optimizer, scaler, use_bfloat16: bool, max_norm: float, itr: int) -> bool:
    if max_norm <= 0.0:
        return True
    opt = optimizer[0]
    if use_bfloat16 and scaler and scaler[0] is not None:
        scaler[0].unscale_(opt)
    params = [p for group in opt.param_groups for p in group["params"] if p.grad is not None]
    if not params:
        return True
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=False)
    if not torch.isfinite(total_norm):
        if keep_nonfinite_grads_enabled():
            logger.warning(
                "Proceeding at itr=%d despite non-finite clipped grad norm=%s (EVAL_KEEP_NONFINITE_GRADS=1)",
                itr,
                total_norm,
            )
            return True
        logger.warning("Skipping optimizer step at itr=%d because clipped grad norm is non-finite: %s", itr, float(total_norm))
        opt.zero_grad()
        return False
    return True


def parse_encoder_lora_cfg(lora_cfg: dict) -> dict | None:
    cfg = dict(lora_cfg.get("encoder_lora", {}) or {})
    env = os.environ.get("ENCODER_LORA_ENABLED")
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
        "lr_mult": float(cfg.get("lr_mult", 0.5)),
        "weight_decay": float(cfg.get("weight_decay", 0.0001)),
        "activation_checkpointing": bool(cfg.get("activation_checkpointing", True)),
        # 0 = no copy-init (use for −n truncation). For +n set to pretrained depth (24).
        "copy_init_from_pretrained": int(cfg.get("copy_init_from_pretrained", 0) or 0),
        # Optional: truncate encoder.blocks to this depth after init (encoder −n).
        "arch_depth": int(cfg.get("arch_depth", 0) or 0),
    }
    target_suffixes = cfg.get("target_suffixes")
    if isinstance(target_suffixes, str):
        target_suffixes = [part.strip() for part in target_suffixes.split(",") if part.strip()]
    if target_suffixes:
        parsed["target_suffixes"] = tuple(str(suffix) for suffix in target_suffixes)
    for key in ("checkpoint_path", "load_checkpoint_path"):
        if key in cfg:
            parsed[key] = cfg[key]
    if "warm_start_at_init" in cfg:
        parsed["warm_start_at_init"] = bool(cfg.get("warm_start_at_init"))
    if "freeze" in cfg:
        parsed["freeze"] = bool(cfg.get("freeze"))
    return parsed


def train_one_epoch_encoder_lora(
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
        logger.info("Using encoder-LoRA grad clip max_norm=%.3f", grad_clip)
    if max_train_iters > 0 and max_train_iters < ipe:
        logger.info("Limiting train_one_epoch_encoder_lora to %d/%d iterations via EVAL_MAX_TRAIN_ITERS", max_train_iters, ipe)
        ipe = max_train_iters

    breakdown = LatencyBreakdown()
    if breakdown.enabled:
        logger.info("Encoder-LoRA latency breakdown enabled (EVAL_LATENCY_BREAKDOWN=1)")
        instrument_model_for_breakdown(model, breakdown)

    from app.hdepic_lora_action_anticipation.data_prefetch import DataLoaderPrefetcher

    prefetcher = DataLoaderPrefetcher(data_loader, name="enc-lora-prefetch")
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

            enc_ok = encoder_lora_grads_finite(model)
            if not enc_ok:
                if keep_nonfinite_grads_enabled():
                    logger.warning(
                        "Keeping non-finite encoder-LoRA grads at itr=%d (EVAL_KEEP_NONFINITE_GRADS=1)",
                        itr,
                    )
                else:
                    logger.warning("Discarding encoder-LoRA grads at itr=%d (non-finite)", itr)
                    zero_encoder_lora_grads(model)
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
                        "recall (a/v/n): %.1f%% %.1f%% %.1f%% enc_lora_ok=%s "
                        "[mem: %.2e] [fetch: %.1f ms] [step: %.0f ms]",
                        itr,
                        float(total_loss.detach().float()),
                        max(a["accuracy"] for a in action_metrics),
                        max(v["accuracy"] for v in verb_metrics),
                        max(n["accuracy"] for n in noun_metrics),
                        max(a["recall"] for a in action_metrics),
                        max(v["recall"] for v in verb_metrics),
                        max(n["recall"] for n in noun_metrics),
                        enc_ok,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                        data_elapsed_time_meter.avg,
                        step_ms,
                    )
                else:
                    logger.info(
                        "[%5d] loss=%.4f acc: %.1f%% recall: %.1f%% enc_lora_ok=%s "
                        "[mem: %.2e] [fetch: %.1f ms] [step: %.0f ms]",
                        itr,
                        float(total_loss.detach().float()),
                        max(a["accuracy"] for a in action_metrics),
                        max(a["recall"] for a in action_metrics),
                        enc_ok,
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
