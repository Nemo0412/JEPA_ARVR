"""
HD-EPIC LoRA fine-tuning — encoder + probe/pooler variant.

Extends app.hdepic_lora_action_anticipation (probe-only LoRA) by also inserting
LoRA adapters into the frozen encoder and routing gradients back through it.

Config structure (under experiment:):
  lora:                         # probe / pooler LoRA  (same keys as probe-only module)
    enabled: true
    rank: 8
    alpha: 16
    dropout: 0.05
    train_heads: true
    pretrained_probe: <path>
    align_reference_metrics: true

  encoder_lora:                 # encoder LoRA  (new section)
    enabled: true               # set false to fall back to probe-only LoRA
    rank: 4                     # lower rank than probe — encoder is larger
    alpha: 8
    dropout: 0.0
    target_modules: ["attn"]    # substring filter for layer names; null → all Linear
    lr: 2.0e-5                  # encoder LoRA learning rate
    weight_decay: 0.0

Monkeypatching strategy (all patches are applied to the base eval module only,
vjepa2/ source files are never modified):
  1. base_eval.ClassMeanRecall  → Top3AccuracyRecallAt5       (same as probe-only)
  2. base_eval.init_classifier  → LoRA-wrapped AttentiveClassifier  (same as probe-only)
  3. base_eval.init_module      → LoRA-wrapped encoder + EnableGradWrapper
  4. base_eval.init_opt         → adds encoder LoRA params to optimizer[0]
  5. torch.save                 → injects encoder_lora state dict into checkpoint
  6. base_eval.load_checkpoint  → restores encoder_lora state dict on resume
"""

import logging
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from app.hdepic_lora_action_anticipation.eval import (
    Top3AccuracyRecallAt5,
    LoRALinear,
    _replace_linears_with_lora,
    _load_pooler_from_probe,
    _freeze_for_lora,
    _log_trainable_params,
    _make_lora_init_classifier,
)
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EnableGradWrapper
# ─────────────────────────────────────────────────────────────────────────────

class EnableGradWrapper(nn.Module):
    """Re-enable gradient tracking inside torch.no_grad() blocks.

    The base eval's train_one_epoch wraps the encoder forward in
    ``with torch.no_grad():``.  We need gradients to flow through encoder LoRA
    adapters, so we use ``torch.enable_grad()`` inside the wrapper's forward.
    ``torch.enable_grad()`` overrides an outer ``torch.no_grad()`` context.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        with torch.enable_grad():
            return self.inner(*args, **kwargs)

    @property
    def embed_dim(self):
        return self.inner.embed_dim


# ─────────────────────────────────────────────────────────────────────────────
# Module-level shared state (set during init_module, read by later patches)
# ─────────────────────────────────────────────────────────────────────────────

_ENCODER_WRAPPER: "EnableGradWrapper | None" = None
_ORIG_TORCH_SAVE = torch.save  # captured before any patching


# ─────────────────────────────────────────────────────────────────────────────
# Filtered LoRA replacement for the encoder
# ─────────────────────────────────────────────────────────────────────────────

def _replace_linears_with_lora_filtered(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_substrings: "list[str] | None",
    prefix: str = "",
) -> list:
    """Like _replace_linears_with_lora but optionally filters by name substring.

    When target_substrings is None every nn.Linear is replaced (same as the
    probe helper).  When it is a list only layers whose full dotted name
    contains at least one of the substrings are replaced.
    """
    replaced = []
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if target_substrings is None or any(s in child_prefix for s in target_substrings):
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(child_prefix)
        else:
            replaced.extend(
                _replace_linears_with_lora_filtered(
                    child, rank, alpha, dropout, target_substrings, child_prefix
                )
            )
    return replaced


# ─────────────────────────────────────────────────────────────────────────────
# Encoder init patch
# ─────────────────────────────────────────────────────────────────────────────

def _log_encoder_trainable(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Encoder LoRA trainable params: %d / %d (%.2f%%)",
        trainable,
        total,
        100.0 * trainable / max(1, total),
    )


def _make_lora_init_module(encoder_lora_cfg: dict):
    rank = int(encoder_lora_cfg.get("rank", 4))
    alpha = float(encoder_lora_cfg.get("alpha", 8.0))
    dropout = float(encoder_lora_cfg.get("dropout", 0.0))
    # target_modules: list of substrings to match against layer full names.
    # Default ["attn"] keeps LoRA on attention blocks only (qkv, proj, etc.).
    # Set to null / [] in config to apply to all Linear layers.
    target_modules = encoder_lora_cfg.get("target_modules", ["attn"])
    if not target_modules:
        target_modules = None  # None means "replace all"

    def _init_module_with_encoder_lora(
        module_name,
        device,
        frames_per_clip,
        frames_per_second,
        resolution,
        checkpoint,
        model_kwargs,
        wrapper_kwargs,
    ):
        global _ENCODER_WRAPPER
        import evals.action_anticipation_frozen.models as _base_models

        # Build original frozen encoder (uses base init_module from models.py)
        model = _base_models.init_module(
            module_name=module_name,
            device=device,
            frames_per_clip=frames_per_clip,
            frames_per_second=frames_per_second,
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )

        # Insert LoRA into encoder's (filtered) Linear layers.
        # init_module already froze all params; LoRALinear.__init__ re-freezes
        # the base weight and leaves lora_A / lora_B with requires_grad=True.
        replaced = _replace_linears_with_lora_filtered(
            model, rank=rank, alpha=alpha, dropout=dropout,
            target_substrings=target_modules,
        )
        logger.info(
            "Inserted encoder LoRA into %d Linear layers (filter=%s, rank=%d, alpha=%.1f)",
            len(replaced), target_modules, rank, alpha,
        )
        _log_encoder_trainable(model)

        wrapper = EnableGradWrapper(model)
        _ENCODER_WRAPPER = wrapper
        return wrapper

    return _init_module_with_encoder_lora


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer patch: add encoder LoRA params to optimizer[0] param groups
# ─────────────────────────────────────────────────────────────────────────────

def _make_lora_init_opt(encoder_lora_cfg: dict):
    enc_lr = float(encoder_lora_cfg.get("lr", 2e-5))
    enc_wd = float(encoder_lora_cfg.get("weight_decay", 0.0))

    from evals.action_anticipation_frozen.utils import init_opt as _orig_init_opt

    def _init_opt_with_encoder(
        classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False
    ):
        optimizers, scalers, schedulers, wd_schedulers = _orig_init_opt(
            classifiers=classifiers,
            iterations_per_epoch=iterations_per_epoch,
            opt_kwargs=opt_kwargs,
            num_epochs=num_epochs,
            use_bfloat16=use_bfloat16,
        )

        if _ENCODER_WRAPPER is not None:
            enc_lora_params = [
                p
                for m in _ENCODER_WRAPPER.modules()
                if isinstance(m, LoRALinear)
                for p in [m.lora_A.weight, m.lora_B.weight]
            ]
            if enc_lora_params:
                # Attach to optimizer[0] with its own LR/WD.
                # The WarmupCosineLRSchedule uses mc_* keys so we set them too.
                warmup_steps = int(opt_kwargs[0].get("warmup", 1) * iterations_per_epoch)
                optimizers[0].add_param_group(
                    {
                        "params": enc_lora_params,
                        "mc_warmup_steps": warmup_steps,
                        "mc_start_lr": enc_lr * 0.1,
                        "mc_ref_lr": enc_lr,
                        "mc_final_lr": enc_lr * 0.01,
                        "mc_ref_wd": enc_wd,
                        "mc_final_wd": enc_wd,
                        "weight_decay": enc_wd,
                        "lr": enc_lr,
                    }
                )
                logger.info(
                    "Added %d encoder LoRA tensors to optimizer[0] "
                    "(lr=%.1e, wd=%.1e)",
                    len(enc_lora_params),
                    enc_lr,
                    enc_wd,
                )

        return optimizers, scalers, schedulers, wd_schedulers

    return _init_opt_with_encoder


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load hooks for encoder LoRA state
# ─────────────────────────────────────────────────────────────────────────────

def _install_save_hook():
    """Patch torch.save so every classifier checkpoint also stores encoder_lora."""

    def _patched_save(obj, f, *args, **kwargs):
        if isinstance(obj, dict) and "classifiers" in obj and _ENCODER_WRAPPER is not None:
            obj = dict(obj)  # shallow copy — don't mutate the original dict
            obj["encoder_lora"] = _ENCODER_WRAPPER.inner.state_dict()
        _ORIG_TORCH_SAVE(obj, f, *args, **kwargs)

    torch.save = _patched_save
    logger.info("Installed encoder-LoRA save hook on torch.save")


def _install_load_hook(base_eval):
    """Patch base_eval.load_checkpoint to also restore encoder LoRA state."""
    _orig_load = base_eval.load_checkpoint

    def _load_with_encoder(device, r_path, classifiers, opt, scaler, val_only=False):
        classifiers, opt, scaler, epoch = _orig_load(
            device=device,
            r_path=r_path,
            classifiers=classifiers,
            opt=opt,
            scaler=scaler,
            val_only=val_only,
        )
        if _ENCODER_WRAPPER is not None:
            ckpt = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
            if "encoder_lora" in ckpt:
                missing, unexpected = _ENCODER_WRAPPER.inner.load_state_dict(
                    ckpt["encoder_lora"], strict=False
                )
                logger.info(
                    "Restored encoder LoRA from checkpoint: missing=%d unexpected=%d",
                    len(missing),
                    len(unexpected),
                )
            else:
                logger.warning(
                    "Checkpoint %s has no 'encoder_lora' key; "
                    "encoder LoRA weights not restored.",
                    r_path,
                )
        return classifiers, opt, scaler, epoch

    base_eval.load_checkpoint = _load_with_encoder


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(args_eval, resume_preempt=False):
    lora_cfg = args_eval.get("experiment", {}).get("lora", {})
    if not lora_cfg.get("enabled", True):
        raise ValueError(
            "app.hdepic_lora_encoder_action_anticipation requires "
            "experiment.lora.enabled=true"
        )

    encoder_lora_cfg = args_eval.get("experiment", {}).get("encoder_lora", {})
    encoder_lora_enabled = bool(encoder_lora_cfg.get("enabled", True))

    import evals.action_anticipation_frozen.eval as base_eval

    # ── 1. Metrics patch (same as probe-only) ─────────────────────────────────
    if bool(lora_cfg.get("align_reference_metrics", True)):
        logger.info("Using aligned metrics: accuracy=Top-3, recall=class-mean Recall@5")
        base_eval.ClassMeanRecall = Top3AccuracyRecallAt5

    # ── 2. Probe / pooler LoRA patch (same as probe-only) ────────────────────
    base_eval.init_classifier = _make_lora_init_classifier(lora_cfg)

    # ── 3–6. Encoder LoRA patches ─────────────────────────────────────────────
    if encoder_lora_enabled:
        logger.info(
            "Encoder LoRA enabled: rank=%s alpha=%s dropout=%s target=%s",
            encoder_lora_cfg.get("rank", 4),
            encoder_lora_cfg.get("alpha", 8),
            encoder_lora_cfg.get("dropout", 0.0),
            encoder_lora_cfg.get("target_modules", ["attn"]),
        )
        base_eval.init_module = _make_lora_init_module(encoder_lora_cfg)
        base_eval.init_opt = _make_lora_init_opt(encoder_lora_cfg)
        _install_save_hook()
        _install_load_hook(base_eval)
    else:
        logger.info("Encoder LoRA disabled; running probe-only LoRA (same as hdepic_lora_action_anticipation)")

    return base_eval.main(args_eval=args_eval, resume_preempt=resume_preempt)
