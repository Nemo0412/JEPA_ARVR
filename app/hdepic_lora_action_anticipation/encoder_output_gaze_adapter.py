"""Encoder-output gaze injection adapter (B8).

Inject RNN-encoded gaze tokens into the frozen V-JEPA token stream between the
encoder output and the predictor input. The adapter is a zero-initialized
cross-attention residual: at step 0 the modified ``x_full`` is bitwise equal
to the original, so the wrapped model starts as an exact identity. Only the
predictor input path sees the gaze-conditioned tokens; the classifier's
observed-window path (``x_accumulate``) keeps the unmodified encoder output,
matching B1 on the observed window.

Reimplements ``AnticipativeWrapper.forward`` locally to avoid editing
``vjepa2/``. Reuses ``GazeTrajectoryEncoder`` from ``gaze_rnn.py``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation.gaze import labels_from_udata
from app.hdepic_lora_action_anticipation.gaze_rnn import (
    GazeTrajectoryEncoder,
    GazeTrajectoryLoader,
    load_gaze_batch,
)
from src.utils.logging import AverageMeter

logger = logging.getLogger(__name__)


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


class EncoderOutputGazeAdapter(nn.Module):
    """Zero-init cross-attention residual that mixes gaze tokens into video tokens.

    ``forward(x_last, gaze_tokens)``:
        - ``x_last``: ``[B, N, D]`` last-layer slice of encoder output.
        - ``gaze_tokens``: ``[B, K, D]`` gaze tokens from ``GazeTrajectoryEncoder``.

    Returns ``x_last + out_proj(cross_attn(x_last, gaze_tokens))``. ``out_proj``
    is zero-initialized, so at step 0 the residual is exactly zero.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_last: torch.Tensor, gaze_tokens: torch.Tensor) -> torch.Tensor:
        delta, _ = self.cross_attn(query=x_last, key=gaze_tokens, value=gaze_tokens)
        return x_last + self.out_proj(delta)


class EncoderOutputGazeAdaptedModel(nn.Module):
    """Wrap a frozen ``AnticipativeWrapper`` with a gaze injection step between
    encoder output and predictor input.

    The forward reimplements
    ``vjepa2/evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.AnticipativeWrapper.forward``
    locally; ``vjepa2/`` is not edited.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter: EncoderOutputGazeAdapter,
        gaze_encoder: GazeTrajectoryEncoder,
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.gaze_encoder = gaze_encoder
        self.embed_dim = base_model.embed_dim

    def _encode_gaze(self, gaze_batch) -> Optional[torch.Tensor]:
        if gaze_batch is None:
            return None
        traj, lengths, sample_valid, video_features = gaze_batch
        return self.gaze_encoder(
            traj,
            lengths=lengths,
            sample_valid=sample_valid,
            video_features=video_features,
        )

    def forward(
        self,
        clips: torch.Tensor,
        anticipation_times: torch.Tensor,
        gaze_batch: Optional[tuple] = None,
    ) -> Optional[torch.Tensor]:
        base = self.base_model
        x_full = base.encoder(clips)
        if torch.is_tensor(x_full) and not torch.isfinite(x_full).all():
            return None

        if base.no_predictor:
            return x_full

        B, N, D_full = x_full.size()
        embed_dim = base.encoder.embed_dim
        use_hierarchical = D_full > embed_dim

        # Last-layer slice used both for classifier accumulator and for gaze
        # injection target.
        if use_hierarchical:
            x_last_obs = x_full[:, :, -embed_dim:]
        else:
            x_last_obs = x_full

        if base.no_encoder:
            x_accumulate = torch.rand(B, 0, embed_dim, device=x_full.device)
        else:
            # Observed-window classifier path stays exactly B1: no gaze influence.
            x_accumulate = x_last_obs.clone()

        # Build the predictor input. Inject gaze only into the predictor path.
        if gaze_batch is not None:
            gaze_tokens = self._encode_gaze(gaze_batch)
        else:
            gaze_tokens = None

        if gaze_tokens is not None:
            x_last_pred = self.adapter(x_last_obs, gaze_tokens.to(x_last_obs.dtype))
            if use_hierarchical:
                x_pred_input = torch.cat(
                    [x_full[:, :, :-embed_dim], x_last_pred], dim=-1
                )
            else:
                x_pred_input = x_last_pred
        else:
            x_pred_input = x_full

        if torch.is_tensor(x_pred_input) and not torch.isfinite(x_pred_input).all():
            return None

        # Position IDs of encoder patch tokens [B, N].
        ctxt_positions = torch.arange(N, device=x_full.device).unsqueeze(0).repeat(B, 1)

        anticipation_steps = (
            anticipation_times * base.frames_per_second / base.tubelet_size
        ).to(torch.int64)
        skip_positions = N + int(base.grid_size**2) * anticipation_steps

        N_pred = int(base.grid_size**2 * (base.num_output_frames // base.tubelet_size))
        tgt_positions = (
            torch.arange(N_pred, device=x_full.device).unsqueeze(0).repeat(B, 1)
        )
        tgt_positions = tgt_positions + skip_positions.unsqueeze(1).repeat(1, N_pred)

        for _ in range(base.num_steps):
            pred_out = base.predictor(
                x_pred_input, masks_x=ctxt_positions, masks_y=tgt_positions
            )
            x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out

            if x_pred_full.size(-1) != embed_dim:
                x_pred = x_pred_full[:, :, -embed_dim:]
            else:
                x_pred = x_pred_full

            x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)
            x_pred_for_input = (
                x_pred_full
                if x_pred_full.size(-1) == x_pred_input.size(-1)
                else x_pred
            )
            x_pred_input = torch.cat(
                [x_pred_input[:, N_pred:, :], x_pred_for_input], dim=1
            )

        if torch.is_tensor(x_accumulate) and not torch.isfinite(x_accumulate).all():
            return None
        return x_accumulate


def encoder_output_gaze_param_names(model: nn.Module) -> set[str]:
    """Return the set of parameter names that belong to the adapter + gaze encoder."""
    model = unwrap_ddp(model)
    names: set[str] = set()
    for prefix in ("adapter.", "gaze_encoder."):
        for name, _ in model.named_parameters():
            bare = name.split("module.", 1)[-1] if name.startswith("module.") else name
            if bare.startswith(prefix):
                names.add(name)
    return names


def trainable_encoder_output_gaze_params(model: nn.Module) -> list[nn.Parameter]:
    model = unwrap_ddp(model)
    params: list[nn.Parameter] = []
    for module in (model.adapter, model.gaze_encoder):
        for p in module.parameters():
            if p.requires_grad:
                params.append(p)
    return params


def _adapter_grads_finite(model: nn.Module) -> bool:
    model = unwrap_ddp(model)
    for module in (model.adapter, model.gaze_encoder):
        for param in module.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False
    return True


def _zero_adapter_grads(model: nn.Module) -> None:
    model = unwrap_ddp(model)
    for module in (model.adapter, model.gaze_encoder):
        for param in module.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()


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


def train_one_epoch_with_encoder_output_gaze(
    base_eval,
    traj_loader: GazeTrajectoryLoader,
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
    model_inner.adapter.train(mode=True)
    model_inner.gaze_encoder.train(mode=True)
    for c in classifiers:
        c.train(mode=True)
    if action_is_verb_noun:
        verb_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5)
            for _ in classifiers
        ]
        noun_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5)
            for _ in classifiers
        ]
    action_metric_loggers = [
        base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5)
        for _ in classifiers
    ]
    data_elapsed_time_meter = AverageMeter()
    try:
        max_train_iters = int(
            os.environ.get("EVAL_MAX_TRAIN_ITERS", os.environ.get("MAX_TRAIN_ITERS", "0")) or "0"
        )
    except ValueError:
        max_train_iters = 0
    if max_train_iters > 0 and max_train_iters < ipe:
        logger.info(
            "Limiting train_one_epoch_with_encoder_output_gaze to %d/%d iterations via EVAL_MAX_TRAIN_ITERS",
            max_train_iters,
            ipe,
        )
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
                raise ValueError("encoder_output_inject requires metadata-aware dataloader")
            anticipation_times = udata[4].to(device, non_blocking=True)
            labels = labels_from_udata(
                udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes
            )
            data_elapsed_time_meter.update((time.time() - itr_start_time) * 1000.0)
            gaze_batch = load_gaze_batch(metadata, traj_loader, device, video_tokens=None)
            tokens = model(clips, anticipation_times, gaze_batch=gaze_batch)
            if tokens is None:
                logger.warning(
                    "Skipping encoder_output_inject optimizer step because tokens are non-finite at itr=%d",
                    itr,
                )
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
            if not _classifier_grads_finite(c):
                logger.warning(
                    "Discarding head %d gradient contribution because head param grads are non-finite",
                    head_idx,
                )
                _zero_classifier_grads(c)
                continue
            tokens_grad_accum.add_(head_token_grad)
            healthy_heads += 1

        if healthy_heads == 0:
            logger.warning(
                "All %d heads produced non-finite grads at itr=%d; skipping optimizer step",
                len(loss),
                itr,
            )
            optimizer[0].zero_grad()
            if use_bfloat16:
                scaler[0].update()
            continue

        tokens_grad_accum.mul_(1.0 / float(healthy_heads))
        tokens.backward(gradient=tokens_grad_accum)

        adapter_ok = _adapter_grads_finite(model)
        if not adapter_ok:
            logger.warning(
                "Discarding adapter+gaze step at itr=%d because adapter/gaze grads are non-finite",
                itr,
            )
            _zero_adapter_grads(model)

        if use_bfloat16:
            scaler[0].step(optimizer[0])
            scaler[0].update()
        else:
            optimizer[0].step()
        optimizer[0].zero_grad()

        with torch.no_grad():
            action_metrics = [
                m(o["action"], labels["action"]) for o, m in zip(outputs, action_metric_loggers)
            ]
            if action_is_verb_noun:
                verb_metrics = [
                    m(o["verb"], labels["verb"]) for o, m in zip(outputs, verb_metric_loggers)
                ]
                noun_metrics = [
                    m(o["noun"], labels["noun"]) for o, m in zip(outputs, noun_metric_loggers)
                ]
        if itr % 10 == 0 or itr == ipe - 1:
            if action_is_verb_noun:
                logger.info(
                    "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "healthy_heads=%d/%d adapter_ok=%s [mem: %.2e] [data: %.1f ms]",
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

    ret = {
        "action": {
            "accuracy": max(a["accuracy"] for a in action_metrics),
            "recall": max(a["recall"] for a in action_metrics),
        }
    }
    if action_is_verb_noun:
        ret.update(
            {
                "verb": {
                    "accuracy": max(v["accuracy"] for v in verb_metrics),
                    "recall": max(v["recall"] for v in verb_metrics),
                },
                "noun": {
                    "accuracy": max(n["accuracy"] for n in noun_metrics),
                    "recall": max(n["recall"] for n in noun_metrics),
                },
            }
        )
    return ret


@torch.no_grad()
def validate_with_encoder_output_gaze(
    base_eval,
    dumper,
    traj_loader: GazeTrajectoryLoader,
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
    logger.info("Running val with encoder-output gaze inject adapter (metric_scope=%s)...", metric_scope)
    if use_valid_filter:
        logger.info("Using filtered val metrics: passing valid_* class sets into ClassMeanRecall")
    _data_loader = iter(data_loader)
    model_inner = unwrap_ddp(model)
    model_inner.base_model.eval()
    model_inner.adapter.eval()
    model_inner.gaze_encoder.eval()
    for c in classifiers:
        c.train(mode=False)
    if action_is_verb_noun:
        verb_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5)
            for _ in classifiers
        ]
        noun_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5)
            for _ in classifiers
        ]
    action_metric_loggers = [
        base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5)
        for _ in classifiers
    ]

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
                raise ValueError("encoder_output_inject requires metadata-aware dataloader")
            anticipation_times = udata[4].to(device, non_blocking=True)
            labels = labels_from_udata(
                udata, device, action_is_verb_noun, verb_classes, noun_classes, action_classes
            )
            gaze_batch = load_gaze_batch(metadata, traj_loader, device, video_tokens=None)
            tokens = model(clips, anticipation_times, gaze_batch=gaze_batch)
            if tokens is None:
                logger.warning(
                    "Skipping encoder_output_inject val batch because tokens are non-finite at itr=%d",
                    itr,
                )
                continue
            outputs = [c(tokens) for c in classifiers]
            valid_actions_arg = valid_actions if use_valid_filter else None
            valid_verbs_arg = valid_verbs if use_valid_filter else None
            valid_nouns_arg = valid_nouns if use_valid_filter else None
            action_metrics = [
                m(o["action"], labels["action"], valid_actions_arg)
                for o, m in zip(outputs, action_metric_loggers)
            ]
            if action_is_verb_noun:
                verb_metrics = [
                    m(o["verb"], labels["verb"], valid_verbs_arg)
                    for o, m in zip(outputs, verb_metric_loggers)
                ]
                noun_metrics = [
                    m(o["noun"], labels["noun"], valid_nouns_arg)
                    for o, m in zip(outputs, noun_metric_loggers)
                ]
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
                    "[%5d] acc (v/n): %.1f%% (%.1f%% %.1f%%) recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "loss (v/n): %.3f (%.3f %.3f) [mem: %.2e]",
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
    ret = {
        "action": {
            "accuracy": max(a["accuracy"] for a in action_metrics),
            "recall": max(a["recall"] for a in action_metrics),
        }
    }
    ret["metric_scope"] = metric_scope
    if action_is_verb_noun:
        ret.update(
            {
                "verb": {
                    "accuracy": max(v["accuracy"] for v in verb_metrics),
                    "recall": max(v["recall"] for v in verb_metrics),
                },
                "noun": {
                    "accuracy": max(n["accuracy"] for n in noun_metrics),
                    "recall": max(n["recall"] for n in noun_metrics),
                },
            }
        )
    return ret
