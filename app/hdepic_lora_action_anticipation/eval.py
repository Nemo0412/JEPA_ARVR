import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from evals.action_anticipation_frozen.models import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader

from app.hdepic_lora_action_anticipation.gaze import (
    GazeTokenGate,
    PredictionDumper,
    patch_clip_balanced_dataloader,
    patch_metadata_dataloader,
    train_one_epoch_with_gaze,
    validate_with_gaze,
)
from app.hdepic_lora_action_anticipation.binary_input_adapter import (
    BinaryGazeMapBuilder,
    BinaryInputAdaptedModel,
    BinaryMapInputAdapter,
    train_one_epoch_with_binary_input_adapter,
    trainable_binary_input_adapter_params,
    validate_with_binary_input_adapter,
)
from app.hdepic_lora_action_anticipation.gaze_rnn import (
    GazeFusedAttentiveClassifier,
    GazeHiddenDump,
    GazeTrajectoryLoader,
    attach_gaze_encoder_to_classifier,
    gaze_encoder_param_names,
)

logger = logging.getLogger(__name__)
logging.raiseExceptions = False


def _parse_past_window_curriculum(past_window_cfg: dict):
    cfg = dict(past_window_cfg.get("curriculum", {}))
    if not bool(cfg.get("enabled", False)):
        return None

    stages = cfg.get("stages")
    if not stages:
        raise ValueError("past_window_baseline.curriculum.enabled=true requires non-empty stages")

    parsed = []
    for idx, stage in enumerate(stages):
        label_h = stage.get("label_horizon_sec", stage.get("anticipation_time_sec"))
        if label_h is None:
            raise ValueError(f"curriculum stage {idx} is missing label_horizon_sec")
        if isinstance(label_h, (int, float)):
            label_h = [float(label_h), float(label_h)]
        if len(label_h) != 2:
            raise ValueError(f"curriculum stage {idx} label_horizon_sec must have two values")
        lo, hi = float(label_h[0]), float(label_h[1])
        if lo < 0 or hi < 0 or hi < lo:
            raise ValueError(f"curriculum stage {idx} has invalid label_horizon_sec={label_h}")
        until_epoch = stage.get("until_epoch")
        parsed.append(
            {
                "until_epoch": None if until_epoch is None else int(until_epoch),
                "label_horizon_sec": [lo, hi],
            }
        )

    for prev, cur in zip(parsed, parsed[1:]):
        if prev["until_epoch"] is not None and cur["until_epoch"] is not None and cur["until_epoch"] <= prev["until_epoch"]:
            raise ValueError("curriculum stage until_epoch values must increase")
    if parsed[-1]["until_epoch"] is not None:
        logger.warning("Last curriculum stage has until_epoch=%s; it will repeat after that epoch", parsed[-1]["until_epoch"])
    return parsed


class Top3AccuracyRecallAt5:
    """Metric adapter for matching the reference script's reporting convention.

    The upstream V-JEPA action anticipation eval instantiates ClassMeanRecall(k=5)
    and logs the returned "accuracy" and "recall" fields. In upstream code,
    both are based on top-5 predictions. The reference HD-EPIC script reports
    Top-3 accuracy and class-mean Recall@5, so this adapter keeps the same return
    keys while changing only "accuracy" to Top-3.
    """

    def __init__(self, num_classes: int, device: torch.device, k=5):
        self.num_classes = num_classes
        self.top3_tp = torch.zeros(num_classes).to(device)
        self.top3_fn = torch.zeros(num_classes).to(device)
        self.r5_tp = torch.zeros(num_classes).to(device)
        self.r5_fn = torch.zeros(num_classes).to(device)

    def __call__(self, logits, labels, valid_classes=None, eps=1e-8):
        logits = F.sigmoid(logits)

        if valid_classes is not None:
            filtered = torch.zeros(logits.shape).to(logits.device)
            for c in valid_classes:
                filtered[:, c] = logits[:, c]
            logits = filtered

        k3 = min(3, logits.shape[1])
        k5 = min(5, logits.shape[1])
        preds3 = logits.topk(k3, dim=1).indices
        preds5 = logits.topk(k5, dim=1).indices

        for p3, p5, gt in zip(preds3, preds5, labels):
            if gt in p3:
                self.top3_tp[gt] += 1
            else:
                self.top3_fn[gt] += 1
            if gt in p5:
                self.r5_tp[gt] += 1
            else:
                self.r5_fn[gt] += 1

        top3_tp, top3_fn = self.top3_tp.clone(), self.top3_fn.clone()
        r5_tp, r5_fn = self.r5_tp.clone(), self.r5_fn.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(top3_tp)
            dist.all_reduce(top3_fn)
            dist.all_reduce(r5_tp)
            dist.all_reduce(r5_fn)

        top3_total = torch.sum(top3_tp + top3_fn)
        top3_accuracy = 100.0 * torch.sum(top3_tp) / torch.clamp(top3_total, min=1.0)

        r5_seen = torch.sum((r5_tp + r5_fn) > 0)
        r5_recall = 100.0 * torch.sum(r5_tp / (r5_tp + r5_fn + eps)) / torch.clamp(r5_seen, min=1)

        return dict(recall=r5_recall, accuracy=top3_accuracy)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _replace_linears_with_lora(module: nn.Module, rank: int, alpha: float, dropout: float, prefix: str = ""):
    replaced = []
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(child_prefix)
        else:
            replaced.extend(_replace_linears_with_lora(child, rank, alpha, dropout, child_prefix))
    return replaced


def _load_pooler_from_probe(classifier: AttentiveClassifier, checkpoint_path: str):
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning("LoRA pretrained probe not found: %s", checkpoint_path)
        return

    checkpoint = robust_checkpoint_loader(str(path), map_location=torch.device("cpu"))
    state_dicts = checkpoint.get("classifiers", [])
    if not state_dicts:
        logger.warning("No classifier state dicts found in probe checkpoint: %s", checkpoint_path)
        return

    source = state_dicts[0]
    target = classifier.state_dict()
    pooler_state = {}
    for key, value in source.items():
        clean_key = key.removeprefix("module.")
        if clean_key.startswith("pooler.") and clean_key in target and target[clean_key].shape == value.shape:
            pooler_state[clean_key] = value

    missing, unexpected = classifier.load_state_dict(pooler_state, strict=False)
    logger.info(
        "Loaded %d pooler tensors from %s; ignored heads and mismatches. missing=%d unexpected=%d",
        len(pooler_state),
        checkpoint_path,
        len(missing),
        len(unexpected),
    )


def _freeze_for_lora(classifier: AttentiveClassifier, train_heads: bool):
    for param in classifier.parameters():
        param.requires_grad = False
    for module in classifier.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True
    if train_heads:
        for name, param in classifier.named_parameters():
            if name.startswith(("verb_classifier.", "noun_classifier.", "action_classifier.")):
                param.requires_grad = True
    if isinstance(classifier, GazeFusedAttentiveClassifier) and classifier.gaze_encoder is not None:
        for param in classifier.gaze_encoder.parameters():
            param.requires_grad = True


def _log_trainable_params(classifier: nn.Module):
    total = sum(p.numel() for p in classifier.parameters())
    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    logger.info("LoRA classifier trainable params: %d / %d (%.2f%%)", trainable, total, pct)


def _make_lora_init_classifier(lora_cfg, traj_mode: str | None = None, rnn_cfg: dict | None = None):
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16.0))
    dropout = float(lora_cfg.get("dropout", 0.05))
    train_heads = bool(lora_cfg.get("train_heads", True))
    pretrained_probe = lora_cfg.get("pretrained_probe", None)
    use_gaze_fusion = traj_mode in {"rnn_fuse", "mlp_fuse"}
    rnn_cfg = dict(rnn_cfg or {})
    if traj_mode == "mlp_fuse":
        rnn_cfg["mode_impl"] = "mlp"
    elif traj_mode == "rnn_fuse":
        rnn_cfg.setdefault("mode_impl", "rnn")

    def init_classifier(
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        device: torch.device,
        num_classifiers: int,
        action_classes: dict,
        verb_classes: dict,
        noun_classes: dict,
    ):
        cls = GazeFusedAttentiveClassifier if use_gaze_fusion else AttentiveClassifier
        classifiers = []
        for _ in range(num_classifiers):
            classifier = cls(
                verb_classes=verb_classes,
                noun_classes=noun_classes,
                action_classes=action_classes,
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=num_blocks,
                use_activation_checkpointing=True,
            )
            _load_pooler_from_probe(classifier, pretrained_probe)
            replaced = _replace_linears_with_lora(classifier.pooler, rank=rank, alpha=alpha, dropout=dropout)
            if use_gaze_fusion:
                attach_gaze_encoder_to_classifier(classifier, embed_dim=embed_dim, rnn_cfg=rnn_cfg)
            _freeze_for_lora(classifier, train_heads=train_heads)
            logger.info("Inserted LoRA into %d pooler Linear layers", len(replaced))
            if use_gaze_fusion:
                enc = classifier.gaze_encoder
                logger.info(
                    "Attached Gaze%sEncoder: hidden=%d, layers=%d, bidir=%s, num_tokens=%d",
                    rnn_cfg.get("mode_impl", "rnn").upper(),
                    int(rnn_cfg.get("hidden_dim", 256)),
                    int(rnn_cfg.get("num_layers", 2)),
                    bool(rnn_cfg.get("bidirectional", True)),
                    enc.num_tokens,
                )
                if bool(rnn_cfg.get("use_video_tokens", False)):
                    logger.info(
                        "Gaze encoder video-token conditioning enabled: fusion=%s, video_proj_dim=%d, local_radius=(t=%d,s=%d), residual_alpha_init=%.4f",
                        str(rnn_cfg.get("video_fusion", "nearest_concat")),
                        int(rnn_cfg.get("video_proj_dim", 128)),
                        int(rnn_cfg.get("local_temporal_radius", 0)),
                        int(rnn_cfg.get("local_spatial_radius", 1)),
                        float(rnn_cfg.get("residual_alpha_init", 0.01)),
                    )
            _log_trainable_params(classifier)
            classifiers.append(classifier.to(device))

        print(classifiers[0])
        return classifiers

    return init_classifier


def main(args_eval, resume_preempt=False):
    lora_cfg = args_eval.get("experiment", {}).get("lora", {})
    if not lora_cfg.get("enabled", True):
        raise ValueError("app.hdepic_lora_action_anticipation requires experiment.lora.enabled=true")

    import evals.action_anticipation_frozen.eval as base_eval

    if bool(lora_cfg.get("align_reference_metrics", True)):
        logger.info("Using aligned metrics: accuracy=Top-3, recall=class-mean Recall@5")
        base_eval.ClassMeanRecall = Top3AccuracyRecallAt5

    gaze_cfg = dict(lora_cfg.get("gaze", {}))
    pred_dump_cfg = dict(lora_cfg.get("prediction_dump", {}))
    gaze_mode = str(gaze_cfg.get("mode", "none")).lower()
    binary_input_adapter_enabled = gaze_mode == "binary_input_adapter"
    data_cfg = args_eval.get("experiment", {}).get("data", {})
    past_window_cfg = dict(lora_cfg.get("past_window_baseline", {}))
    model_anticipation_time_sec = None
    drop_incomplete_history = False
    train_label_horizon_schedule = None
    if past_window_cfg.get("enabled", False):
        pred_h = float(past_window_cfg["prediction_horizon_sec"])
        label_h = float(past_window_cfg["label_horizon_sec"])
        model_anticipation_time_sec = (pred_h, pred_h)
        drop_incomplete_history = bool(past_window_cfg.get("drop_incomplete_history", True))
        past_window_apply_to_train = bool(past_window_cfg.get("apply_to_train", False))
        train_label_horizon_schedule = _parse_past_window_curriculum(past_window_cfg)
        logger.info(
            "Enabled past-window baseline: sample clip %.3fs before target action, pass %.3fs anticipation time to model, apply_to_train=%s",
            label_h,
            pred_h,
            past_window_apply_to_train,
        )
        if train_label_horizon_schedule:
            logger.info("Past-window train curriculum enabled: %s", train_label_horizon_schedule)
    else:
        past_window_apply_to_train = False

    clip_balanced = bool(data_cfg.get("clip_balanced", True))
    if gaze_mode in {"rnn_fuse", "mlp_fuse", "binary_input_adapter"} and bool(gaze_cfg.get("use_motion", False)):
        # The token_gate motion path is unused when mode != token_gate, but warn so the
        # ablation matrix stays interpretable.
        logger.warning("gaze.use_motion=true is ignored when mode=%s (token_gate path is disabled)", gaze_mode)
    gaze_cfg.setdefault("crop_size", data_cfg.get("resolution", 384))
    gaze_cfg.setdefault("frames_per_clip", data_cfg.get("frames_per_clip", 32))
    gaze_cfg.setdefault("patch_size", args_eval.get("model_kwargs", {}).get("pretrain_kwargs", {}).get("encoder", {}).get("patch_size", 16))
    gaze_cfg.setdefault("tubelet_size", args_eval.get("model_kwargs", {}).get("pretrain_kwargs", {}).get("encoder", {}).get("tubelet_size", 2))
    if binary_input_adapter_enabled:
        disable_train_aug_env = os.environ.get("BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG")
        gaze_cfg["disable_train_aug"] = (
            disable_train_aug_env.lower() in {"1", "true", "yes", "on"}
            if disable_train_aug_env is not None
            else True
        )
    traj_mode = gaze_mode if gaze_mode in {"rnn_fuse", "mlp_fuse"} else None
    rnn_cfg = dict(gaze_cfg.get("rnn", {}))
    needs_metadata = gaze_mode in {"token_gate", "rnn_fuse", "mlp_fuse", "binary_input_adapter"} or bool(pred_dump_cfg.get("enabled", False))
    traj_loader = None
    hidden_dump = None
    if needs_metadata:
        logger.info("Using clip-balanced metadata-aware HD-EPIC dataloader for gaze/prediction dump hooks")
        patch_metadata_dataloader(
            model_anticipation_time_sec=model_anticipation_time_sec,
            drop_incomplete_history=drop_incomplete_history,
            apply_to_train=past_window_apply_to_train,
            train_label_horizon_schedule=train_label_horizon_schedule,
            emit_binary_map=binary_input_adapter_enabled,
            binary_map_cfg=gaze_cfg if binary_input_adapter_enabled else None,
        )

        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

        gate = GazeTokenGate(gaze_cfg)
        folder = Path(args_eval.get("folder", "."))
        tag = args_eval.get("tag")
        run_dir = folder / "action_anticipation_frozen" / tag if tag else folder / "action_anticipation_frozen"
        if pred_dump_cfg.get("enabled", False):
            pred_dump_cfg.setdefault("path", str(run_dir / "val_predictions.csv"))
        dumper = PredictionDumper(pred_dump_cfg, run_dir, rank)

        if binary_input_adapter_enabled:
            logger.info("Enabling binary_input_adapter: RGB + online binary gaze map -> tiny residual RGB adapter")
            gaze_cfg.setdefault("adapter_checkpoint_path", str(run_dir / "binary_input_adapter_latest.pt"))
            gaze_cfg.setdefault("rank", rank)
            _patch_init_module_for_binary_input_adapter(base_eval, gaze_cfg)
            _patch_opt_for_binary_input_adapter(base_eval, gaze_cfg)
            map_builder = BinaryGazeMapBuilder(gaze_cfg, gate=gate)
            base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_binary_input_adapter(
                base_eval, map_builder, **kwargs
            )
            base_eval.validate = lambda **kwargs: validate_with_binary_input_adapter(
                base_eval, map_builder, dumper, **kwargs
            )
            if args_eval.get("resume_checkpoint", False) and not bool(args_eval.get("val_only", False)):
                logger.warning(
                    "binary_input_adapter disables training resume_checkpoint to avoid classifier-only adapter mismatch; "
                    "val_only resume is allowed when input_adapter.load_checkpoint_path is set"
                )
                args_eval["resume_checkpoint"] = False

        elif traj_mode is not None:
            traj_loader = GazeTrajectoryLoader(gaze_cfg, gate=gate)
            hidden_dump = GazeHiddenDump(dict(gaze_cfg.get("hidden_dump", {})), run_dir, rank)

            base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_gaze(
                base_eval, gate, traj_loader=traj_loader, **kwargs
            )
            base_eval.validate = lambda **kwargs: validate_with_gaze(
                base_eval, gate, dumper, traj_loader=traj_loader, hidden_dump=hidden_dump, **kwargs
            )
        elif gaze_mode == "token_gate" or bool(pred_dump_cfg.get("enabled", False)):
            base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_gaze(
                base_eval, gate, traj_loader=traj_loader, **kwargs
            )
            base_eval.validate = lambda **kwargs: validate_with_gaze(
                base_eval, gate, dumper, traj_loader=traj_loader, hidden_dump=hidden_dump, **kwargs
            )
    elif clip_balanced:
        logger.info("Using clip-balanced HD-EPIC dataloader")
        patch_clip_balanced_dataloader(
            model_anticipation_time_sec=model_anticipation_time_sec,
            drop_incomplete_history=drop_incomplete_history,
            apply_to_train=past_window_apply_to_train,
            train_label_horizon_schedule=train_label_horizon_schedule,
        )

    base_eval.init_classifier = _make_lora_init_classifier(lora_cfg, traj_mode=traj_mode, rnn_cfg=rnn_cfg)
    if traj_mode is not None:
        _patch_opt_for_gaze_encoder(base_eval, gaze_lr_mult=float(rnn_cfg.get("gaze_lr_mult", 5.0)))
    return base_eval.main(args_eval=args_eval, resume_preempt=resume_preempt)


def _patch_init_module_for_binary_input_adapter(base_eval, gaze_cfg: dict):
    original_init_module = base_eval.init_module

    def init_module_with_binary_adapter(*args, **kwargs):
        model = original_init_module(*args, **kwargs)
        cfg = dict(gaze_cfg.get("input_adapter", {}))
        adapter = BinaryMapInputAdapter(
            hidden_dim=int(cfg.get("hidden_dim", 8)),
            scale=float(cfg.get("scale", 1.0)),
            temporal_kernel=int(cfg.get("temporal_kernel", 1)),
            binary_center=float(cfg.get("binary_center", 0.0)),
        ).to(next(model.parameters()).device)
        adapter_ckpt = cfg.get("load_checkpoint_path")
        if adapter_ckpt:
            _load_binary_input_adapter_checkpoint(adapter, str(adapter_ckpt))
        wrapped = BinaryInputAdaptedModel(model, adapter)
        wrapped.embed_dim = model.embed_dim
        for param in wrapped.base_model.parameters():
            param.requires_grad = False
        for param in wrapped.input_adapter.parameters():
            param.requires_grad = True
        if bool(cfg.get("activation_checkpointing", False)):
            if hasattr(wrapped.base_model.encoder, "use_activation_checkpointing"):
                wrapped.base_model.encoder.use_activation_checkpointing = True
                logger.info("Enabled encoder activation checkpointing for binary_input_adapter")
            if hasattr(wrapped.base_model.predictor, "use_activation_checkpointing"):
                wrapped.base_model.predictor.use_activation_checkpointing = True
                logger.info("Enabled predictor activation checkpointing for binary_input_adapter")
        trainable = sum(p.numel() for p in wrapped.input_adapter.parameters() if p.requires_grad)
        logger.info("Attached BinaryMapInputAdapter: trainable_params=%d cfg=%s", trainable, cfg)
        base_eval._binary_input_adapter_model = wrapped
        return wrapped

    base_eval.init_module = init_module_with_binary_adapter


def _load_binary_input_adapter_checkpoint(adapter: BinaryMapInputAdapter, checkpoint_path: str):
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning("Binary input adapter checkpoint not found: %s", checkpoint_path)
        return
    checkpoint = robust_checkpoint_loader(str(path), map_location=torch.device("cpu"))
    state = checkpoint.get("input_adapter", checkpoint)
    if any(str(k).startswith("module.input_adapter.") for k in state):
        state = {str(k).removeprefix("module.input_adapter."): v for k, v in state.items() if str(k).startswith("module.input_adapter.")}
    elif any(str(k).startswith("input_adapter.") for k in state):
        state = {str(k).removeprefix("input_adapter."): v for k, v in state.items() if str(k).startswith("input_adapter.")}
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    logger.info(
        "Loaded binary input adapter checkpoint from %s; missing=%d unexpected=%d",
        checkpoint_path,
        len(missing),
        len(unexpected),
    )


def _patch_opt_for_binary_input_adapter(base_eval, gaze_cfg: dict):
    from evals.action_anticipation_frozen.utils import CosineWDSchedule, WarmupCosineLRSchedule

    adapter_cfg = dict(gaze_cfg.get("input_adapter", {}))
    lr_mult = float(adapter_cfg.get("lr_mult", 1.0))
    wd = float(adapter_cfg.get("weight_decay", 0.0001))

    def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
        if not classifiers:
            raise ValueError("binary_input_adapter requires at least one classifier")
        model = getattr(base_eval, "_binary_input_adapter_model", None)
        if model is None:
            # The model is passed to train/validate later, but init_opt has no model
            # argument in upstream eval. The init_module patch stores it there.
            raise RuntimeError("binary_input_adapter model was not registered before init_opt")
        adapter_params = trainable_binary_input_adapter_params(model)
        param_groups = []
        classifier_param_count = 0
        first_kwargs = opt_kwargs[0]
        for classifier, kwargs in zip(classifiers, opt_kwargs):
            base_params = [p for p in classifier.parameters() if p.requires_grad]
            classifier_param_count += sum(p.numel() for p in base_params)
            warmup_steps = int((kwargs.get("warmup") or 0.0) * iterations_per_epoch)
            param_groups.append(
                {
                    "params": base_params,
                    "mc_warmup_steps": warmup_steps,
                    "mc_start_lr": kwargs.get("start_lr"),
                    "mc_ref_lr": kwargs.get("ref_lr"),
                    "mc_final_lr": kwargs.get("final_lr"),
                    "mc_ref_wd": kwargs.get("ref_wd"),
                    "mc_final_wd": kwargs.get("final_wd"),
                }
            )
        adapter_warmup_steps = int((first_kwargs.get("warmup") or 0.0) * iterations_per_epoch)
        param_groups.append(
            {
                "params": adapter_params,
                "mc_warmup_steps": adapter_warmup_steps,
                "mc_start_lr": (first_kwargs.get("start_lr") or 0.0) * lr_mult,
                "mc_ref_lr": (first_kwargs.get("ref_lr") or 0.0) * lr_mult,
                "mc_final_lr": (first_kwargs.get("final_lr") or 0.0) * lr_mult,
                "mc_ref_wd": wd,
                "mc_final_wd": wd,
            }
        )
        logger.info(
            "Optimizer split: classifiers=%d params across %d heads, binary_input_adapter=%d params, adapter_lr_mult=%.3f",
            classifier_param_count,
            len(classifiers),
            sum(p.numel() for p in adapter_params),
            lr_mult,
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = WarmupCosineLRSchedule(optimizer, T_max=int(num_epochs * iterations_per_epoch))
        wd_scheduler = CosineWDSchedule(optimizer, T_max=int(num_epochs * iterations_per_epoch))
        scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
        return [optimizer], [scaler], [scheduler], [wd_scheduler]

    base_eval.init_opt = init_opt


def _patch_opt_for_gaze_encoder(base_eval, gaze_lr_mult: float):
    """Wrap base_eval.init_opt to put gaze_encoder params into their own LR group.

    The GRU/MLP is trained from scratch while LoRA only fine-tunes pretrained
    linears, so they need different learning rates. The schedule keys read by
    ``WarmupCosineLRSchedule`` (``mc_ref_lr`` / ``mc_start_lr`` / ``mc_final_lr``)
    are scaled by ``gaze_lr_mult`` for the gaze group.
    """
    from evals.action_anticipation_frozen.utils import (
        CosineWDSchedule,
        WarmupCosineLRSchedule,
    )

    def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
        optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
        for c, kwargs in zip(classifiers, opt_kwargs):
            gaze_names = gaze_encoder_param_names(c)
            base_params = [p for n, p in c.named_parameters() if n not in gaze_names and p.requires_grad]
            gaze_params = [p for n, p in c.named_parameters() if n in gaze_names and p.requires_grad]
            warmup_steps = int((kwargs.get("warmup") or 0.0) * iterations_per_epoch)
            base_group = {
                "params": base_params,
                "mc_warmup_steps": warmup_steps,
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
            param_groups = [base_group]
            if gaze_params:
                gaze_group = {
                    "params": gaze_params,
                    "mc_warmup_steps": warmup_steps,
                    "mc_start_lr": (kwargs.get("start_lr") or 0.0) * gaze_lr_mult,
                    "mc_ref_lr": (kwargs.get("ref_lr") or 0.0) * gaze_lr_mult,
                    "mc_final_lr": (kwargs.get("final_lr") or 0.0) * gaze_lr_mult,
                    "mc_ref_wd": kwargs.get("ref_wd"),
                    "mc_final_wd": kwargs.get("final_wd"),
                }
                param_groups.append(gaze_group)
                logger.info(
                    "Optimizer split: lora/heads=%d params, gaze=%d params, gaze_lr_mult=%.2f",
                    sum(p.numel() for p in base_params),
                    sum(p.numel() for p in gaze_params),
                    gaze_lr_mult,
                )
            logger.info("Using AdamW")
            optimizers.append(torch.optim.AdamW(param_groups))
            schedulers.append(WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch)))
            wd_schedulers.append(CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch)))
            scalers.append(torch.cuda.amp.GradScaler() if use_bfloat16 else None)
        return optimizers, scalers, schedulers, wd_schedulers

    base_eval.init_opt = init_opt
