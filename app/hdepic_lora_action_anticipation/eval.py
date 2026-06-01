import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

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
    GazeTrajectoryEncoder,
    GazeTrajectoryLoader,
    PoseTrajectoryLoader,
    attach_gaze_encoder_to_classifier,
    attach_pose_encoder_to_classifier,
    gaze_encoder_param_names,
)
from app.hdepic_lora_action_anticipation.pose_slam import feature_dim_for_set
from app.hdepic_lora_action_anticipation.encoder_output_gaze_adapter import (
    EncoderOutputGazeAdapter,
    EncoderOutputGazeAdaptedModel,
    train_one_epoch_with_encoder_output_gaze,
    trainable_encoder_output_gaze_params,
    validate_with_encoder_output_gaze,
)

logger = logging.getLogger(__name__)
logging.raiseExceptions = False


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def _wrap_trainable_model_for_ddp(model: nn.Module) -> nn.Module:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        kwargs = {}
        if torch.cuda.is_available():
            kwargs = {"device_ids": [torch.cuda.current_device()], "output_device": torch.cuda.current_device()}
        logger.info("Wrapping trainable LoRA side model with DDP for world_size=%d", dist.get_world_size())
        wrapped = DistributedDataParallel(model, **kwargs)
        if hasattr(model, "embed_dim"):
            wrapped.embed_dim = model.embed_dim
        return wrapped
    return model


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


def _as_time_pair(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a two-value time range, got {value}")
        return (float(value[0]), float(value[1]))
    v = float(value)
    return (v, v)


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
    if isinstance(classifier, GazeFusedAttentiveClassifier):
        for enc in (classifier.gaze_encoder, classifier.pose_encoder):
            if enc is not None:
                for param in enc.parameters():
                    param.requires_grad = True


def _log_trainable_params(classifier: nn.Module):
    total = sum(p.numel() for p in classifier.parameters())
    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    logger.info("LoRA classifier trainable params: %d / %d (%.2f%%)", trainable, total, pct)


def _make_lora_init_classifier(
    lora_cfg,
    traj_mode: str | None = None,
    rnn_cfg: dict | None = None,
    token_gate: GazeTokenGate | None = None,
):
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16.0))
    dropout = float(lora_cfg.get("dropout", 0.05))
    train_heads = bool(lora_cfg.get("train_heads", True))
    pretrained_probe = lora_cfg.get("pretrained_probe", None)
    use_gaze_fusion = traj_mode in {"rnn_fuse", "mlp_fuse", "pose_rnn_fuse", "multimodal_rnn_fuse"}
    use_gaze_branch = traj_mode in {"rnn_fuse", "mlp_fuse", "multimodal_rnn_fuse"}
    use_pose_branch = traj_mode in {"pose_rnn_fuse", "multimodal_rnn_fuse"}
    rnn_cfg = dict(rnn_cfg or {})
    pose_cfg = dict(lora_cfg.get("gaze", {}).get("pose", {}))
    if use_pose_branch:
        feature_set = str(pose_cfg.get("feature_set", "pose_6d"))
        rnn_cfg.setdefault("input_dim", feature_dim_for_set(feature_set))
    if traj_mode == "mlp_fuse":
        rnn_cfg["mode_impl"] = "mlp"
    elif traj_mode in {"rnn_fuse", "pose_rnn_fuse", "multimodal_rnn_fuse"}:
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
        for head_idx in range(num_classifiers):
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
                if use_gaze_branch:
                    attach_gaze_encoder_to_classifier(classifier, embed_dim=embed_dim, rnn_cfg={**rnn_cfg, "input_dim": 3})
                if use_pose_branch:
                    attach_pose_encoder_to_classifier(classifier, embed_dim=embed_dim, rnn_cfg=rnn_cfg)
            _freeze_for_lora(classifier, train_heads=train_heads)
            if token_gate is not None and head_idx == 0:
                classifier.gaze_token_gate = token_gate
                for param in classifier.gaze_token_gate.parameters():
                    param.requires_grad = bool(getattr(classifier.gaze_token_gate, "learnable_gate", False))
                logger.info(
                    "Attached %s GazeTokenGate to classifier head 0: gamma_init=%.4f trainable_params=%d",
                    "learnable" if getattr(classifier.gaze_token_gate, "learnable_gate", False) else "fixed",
                    float(classifier.gaze_token_gate.current_gamma().detach().float().cpu()),
                    sum(p.numel() for p in classifier.gaze_token_gate.parameters() if p.requires_grad),
                )
            logger.info("Inserted LoRA into %d pooler Linear layers", len(replaced))
            if use_gaze_fusion:
                if classifier.gaze_encoder is not None:
                    enc = classifier.gaze_encoder
                    logger.info(
                        "Attached Gaze%sEncoder: hidden=%d, layers=%d, bidir=%s, num_tokens=%d, input_dim=%d",
                        rnn_cfg.get("mode_impl", "rnn").upper(),
                        int(rnn_cfg.get("hidden_dim", 256)),
                        int(rnn_cfg.get("num_layers", 2)),
                        bool(rnn_cfg.get("bidirectional", True)),
                        enc.num_tokens,
                        int(enc.gaze_input_dim),
                    )
                if classifier.pose_encoder is not None:
                    enc = classifier.pose_encoder
                    logger.info(
                        "Attached Pose%sEncoder: hidden=%d, layers=%d, bidir=%s, num_tokens=%d, input_dim=%d",
                        rnn_cfg.get("mode_impl", "rnn").upper(),
                        int(rnn_cfg.get("hidden_dim", 256)),
                        int(rnn_cfg.get("num_layers", 2)),
                        bool(rnn_cfg.get("bidirectional", True)),
                        enc.num_tokens,
                        int(enc.gaze_input_dim),
                    )
                if use_gaze_branch and bool(rnn_cfg.get("use_video_tokens", False)):
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


def _patch_load_checkpoint_for_learnable_token_gate(base_eval):
    def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
        logger.info(f"read-path: {r_path}")
        checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
        messages = []
        for classifier, state in zip(classifiers, checkpoint["classifiers"]):
            try:
                messages.append(classifier.load_state_dict(state))
            except RuntimeError as exc:
                msg = classifier.load_state_dict(state, strict=False)
                logger.warning(
                    "Loaded classifier checkpoint with strict=False for learnable token gate compatibility: %s; msg=%s",
                    exc,
                    msg,
                )
                messages.append(msg)

        if val_only:
            logger.info(f"loaded pretrained classifier from epoch with msg: {messages}")
            return classifiers, opt, scaler, 0

        epoch = checkpoint["epoch"]
        logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {messages}")
        try:
            [o.load_state_dict(c) for o, c in zip(opt, checkpoint["opt"])]
            if scaler is not None:
                [s.load_state_dict(c) for s, c in zip(scaler, checkpoint["scaler"])]
            logger.info(f"loaded optimizers from epoch {epoch}")
        except ValueError as exc:
            logger.warning(
                "Skipping optimizer/scaler restore after adding learnable token gate because state shapes changed: %s",
                exc,
            )
        return classifiers, opt, scaler, epoch

    base_eval.load_checkpoint = load_checkpoint


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
    data_cfg = args_eval.get("experiment", {}).get("data", {})

    downsample_factor = float(
        data_cfg.get(
            "video_downsample_factor",
            lora_cfg.get("video_downsample_factor", 1.0),
        )
        or 1.0
    )
    if downsample_factor < 1.0:
        raise ValueError(f"video_downsample_factor must be >= 1, got {downsample_factor}")
    if not math.isclose(downsample_factor, 1.0):
        raw_fps = float(data_cfg.get("frames_per_second"))
        if raw_fps <= 0:
            raise ValueError("experiment.data.frames_per_second must be positive when video_downsample_factor is enabled")
        semantic_fps = raw_fps
        scaled_fps = raw_fps / downsample_factor
        data_cfg["video_downsample_factor"] = downsample_factor
        data_cfg["frames_per_second"] = scaled_fps
        logger.info(
            "Applied video downsample factor %.3fx: model fps stays %.3f, dataloader fps becomes %.3f; "
            "sample horizons stay in real seconds and model mask horizons are divided by the factor",
            downsample_factor,
            semantic_fps,
            scaled_fps,
        )

        original_init_module = base_eval.init_module

        def init_module_with_video_downsample(*args, **kwargs):
            kwargs["frames_per_second"] = semantic_fps
            return original_init_module(*args, **kwargs)

        base_eval.init_module = init_module_with_video_downsample

    val_metric_scope = str(os.environ.get("LORA_VAL_METRIC_SCOPE", lora_cfg.get("val_metric_scope", "native"))).lower()
    if val_metric_scope not in {"native", "filtered"}:
        raise ValueError(f"Unsupported lora.val_metric_scope={val_metric_scope!r}; expected native or filtered")
    logger.info(
        "Validation metric scope: %s (%s)",
        val_metric_scope,
        "upstream V-JEPA2 native/unfiltered" if val_metric_scope == "native" else "filtered to val split valid classes",
    )
    gaze_mode = str(gaze_cfg.get("mode", "none")).lower()
    binary_input_adapter_enabled = gaze_mode == "binary_input_adapter"
    encoder_output_inject_enabled = gaze_mode == "encoder_output_inject"
    past_window_cfg = dict(lora_cfg.get("past_window_baseline", {}))
    model_anticipation_time_sec = None
    drop_incomplete_history = False
    train_label_horizon_schedule = None
    if past_window_cfg.get("enabled", False):
        pred_h = float(past_window_cfg["prediction_horizon_sec"])
        label_h = float(past_window_cfg["label_horizon_sec"])
        if not math.isclose(downsample_factor, 1.0):
            pred_h = pred_h / downsample_factor
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

    if model_anticipation_time_sec is None and not math.isclose(downsample_factor, 1.0):
        val_h = _as_time_pair(data_cfg.get("anticipation_time_sec"))
        if val_h is None:
            raise ValueError("video_downsample_factor requires experiment.data.anticipation_time_sec for validation")
        model_anticipation_time_sec = tuple(x / downsample_factor for x in val_h)
        logger.info(
            "Video-downsample validation: real sample horizon=%s sec, model mask horizon=%s sec",
            val_h,
            model_anticipation_time_sec,
        )

    clip_balanced = bool(data_cfg.get("clip_balanced", True))
    if gaze_mode in {"rnn_fuse", "mlp_fuse", "binary_input_adapter", "encoder_output_inject"} and bool(gaze_cfg.get("use_motion", False)):
        # The token_gate motion path is unused when mode != token_gate, but warn so the
        # ablation matrix stays interpretable.
        logger.warning("gaze.use_motion=true is ignored when mode=%s (token_gate path is disabled)", gaze_mode)
    gaze_cfg.setdefault("crop_size", data_cfg.get("resolution", 384))
    gaze_cfg.setdefault("frames_per_clip", data_cfg.get("frames_per_clip", 32))
    gaze_cfg.setdefault("patch_size", args_eval.get("model_kwargs", {}).get("pretrain_kwargs", {}).get("encoder", {}).get("patch_size", 16))
    gaze_cfg.setdefault("tubelet_size", args_eval.get("model_kwargs", {}).get("pretrain_kwargs", {}).get("encoder", {}).get("tubelet_size", 2))
    if binary_input_adapter_enabled:
        aug_aware_env = os.environ.get("BINARY_INPUT_ADAPTER_AUG_AWARE")
        aug_aware = (
            aug_aware_env.lower() in {"1", "true", "yes", "on"}
            if aug_aware_env is not None
            else bool(gaze_cfg.get("aug_aware", False))
        )
        gaze_cfg["aug_aware"] = aug_aware
        if aug_aware:
            # Aug-aware joint transform replays V-JEPA2 training aug on RGB while
            # synchronizing the geometric ops (RRC + flip + center-crop) onto the
            # binary gaze map. The legacy disable_train_aug switch is bypassed in
            # this path because aug is handled inside the joint transform.
            gaze_cfg["disable_train_aug"] = False
            gaze_cfg.setdefault("random_resize_scale", list(data_cfg.get("random_resize_scale", [0.08, 1.0])))
            gaze_cfg.setdefault("auto_augment", bool(data_cfg.get("auto_augment", True)))
            gaze_cfg.setdefault("reprob", float(data_cfg.get("reprob", 0.25)))
        else:
            disable_train_aug_env = os.environ.get("BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG")
            gaze_cfg["disable_train_aug"] = (
                disable_train_aug_env.lower() in {"1", "true", "yes", "on"}
                if disable_train_aug_env is not None
                else True
            )
    traj_mode = gaze_mode if gaze_mode in {"rnn_fuse", "mlp_fuse", "pose_rnn_fuse", "multimodal_rnn_fuse"} else None
    rnn_cfg = dict(gaze_cfg.get("rnn", {}))
    pose_cfg = dict(gaze_cfg.get("pose", {}))
    if traj_mode == "pose_rnn_fuse":
        rnn_cfg["use_video_tokens"] = False
    if traj_mode == "multimodal_rnn_fuse":
        rnn_cfg["use_video_tokens"] = False
    needs_metadata = gaze_mode in {
        "token_gate",
        "rnn_fuse",
        "mlp_fuse",
        "pose_rnn_fuse",
        "multimodal_rnn_fuse",
        "binary_input_adapter",
        "encoder_output_inject",
    } or bool(pred_dump_cfg.get("enabled", False))
    debug_subset_path = os.environ.get("DEBUG_SUBSET_PATH", "").strip() or None
    if debug_subset_path:
        logger.warning(
            "DEBUG_SUBSET_PATH=%s — this is a debug-only run; DO NOT report these metrics as final results.",
            debug_subset_path,
        )
    traj_loader = None
    pose_loader = None
    hidden_dump = None
    local_validate_patched = False
    token_gate_module = None
    if needs_metadata:
        logger.info("Using clip-balanced metadata-aware HD-EPIC dataloader for gaze/prediction dump hooks")
        patch_metadata_dataloader(
            model_anticipation_time_sec=model_anticipation_time_sec,
            drop_incomplete_history=drop_incomplete_history,
            apply_to_train=past_window_apply_to_train,
            train_label_horizon_schedule=train_label_horizon_schedule,
            emit_binary_map=binary_input_adapter_enabled,
            binary_map_cfg=gaze_cfg if binary_input_adapter_enabled else None,
            debug_subset_path=debug_subset_path,
        )

        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

        gate = GazeTokenGate(gaze_cfg)
        if gaze_mode == "token_gate":
            token_gate_module = gate
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
                base_eval, map_builder, dumper, val_metric_scope=val_metric_scope, **kwargs
            )
            local_validate_patched = True
            if args_eval.get("resume_checkpoint", False) and not bool(args_eval.get("val_only", False)):
                logger.warning(
                    "binary_input_adapter disables training resume_checkpoint to avoid classifier-only adapter mismatch; "
                    "val_only resume is allowed when input_adapter.load_checkpoint_path is set"
                )
                args_eval["resume_checkpoint"] = False

        elif encoder_output_inject_enabled:
            logger.info(
                "Enabling encoder_output_inject: zero-init cross-attn adapter between encoder output and predictor input"
            )
            # Force gaze-only branch (no video-token conditioning); B8 keeps the
            # architectural axis isolated from B2's video-conditioned RNN gaze.
            rnn_cfg["use_video_tokens"] = False
            traj_loader = GazeTrajectoryLoader(gaze_cfg, gate=gate)
            _patch_init_module_for_encoder_output_gaze(base_eval, gaze_cfg, rnn_cfg)
            _patch_opt_for_encoder_output_gaze(base_eval, gaze_cfg, rnn_cfg)
            base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_encoder_output_gaze(
                base_eval, traj_loader, **kwargs
            )
            base_eval.validate = lambda **kwargs: validate_with_encoder_output_gaze(
                base_eval, dumper, traj_loader, val_metric_scope=val_metric_scope, **kwargs
            )
            local_validate_patched = True

        elif traj_mode is not None:
            gate_for_pose = gate
            if traj_mode in {"pose_rnn_fuse", "multimodal_rnn_fuse"}:
                pose_loader = PoseTrajectoryLoader(gaze_cfg, gate=gate_for_pose)
            if traj_mode in {"rnn_fuse", "mlp_fuse", "multimodal_rnn_fuse"}:
                traj_loader = GazeTrajectoryLoader(gaze_cfg, gate=gate)
            hidden_dump = GazeHiddenDump(dict(gaze_cfg.get("hidden_dump", {})), run_dir, rank)

            base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_gaze(
                base_eval, gate, traj_loader=traj_loader, pose_loader=pose_loader, **kwargs
            )
            base_eval.validate = lambda **kwargs: validate_with_gaze(
                base_eval,
                gate,
                dumper,
                traj_loader=traj_loader,
                pose_loader=pose_loader,
                hidden_dump=hidden_dump,
                val_metric_scope=val_metric_scope,
                **kwargs,
            )
            local_validate_patched = True
        elif gaze_mode == "token_gate" or bool(pred_dump_cfg.get("enabled", False)):
            base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_gaze(
                base_eval, gate, traj_loader=traj_loader, **kwargs
            )
            base_eval.validate = lambda **kwargs: validate_with_gaze(
                base_eval, gate, dumper, traj_loader=traj_loader, hidden_dump=hidden_dump, val_metric_scope=val_metric_scope, **kwargs
            )
            local_validate_patched = True
    elif clip_balanced:
        logger.info("Using clip-balanced HD-EPIC dataloader")
        patch_clip_balanced_dataloader(
            model_anticipation_time_sec=model_anticipation_time_sec,
            drop_incomplete_history=drop_incomplete_history,
            apply_to_train=past_window_apply_to_train,
            train_label_horizon_schedule=train_label_horizon_schedule,
            debug_subset_path=debug_subset_path,
        )
    if val_metric_scope == "filtered" and not local_validate_patched:
        raise ValueError(
            "LORA_VAL_METRIC_SCOPE=filtered requires a project-local validate wrapper "
            "(gaze/prediction_dump/binary_input_adapter/encoder_output_inject). "
            "The upstream V-JEPA2 validate path is native/unfiltered and does not pass valid_classes."
        )

    base_eval.init_classifier = _make_lora_init_classifier(
        lora_cfg,
        traj_mode=traj_mode,
        rnn_cfg=rnn_cfg,
        token_gate=token_gate_module if getattr(token_gate_module, "learnable_gate", False) else None,
    )
    if getattr(token_gate_module, "learnable_gate", False):
        _patch_load_checkpoint_for_learnable_token_gate(base_eval)
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
            residual_clamp=float(cfg.get("residual_clamp", 1.0)),
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
        wrapped = _wrap_trainable_model_for_ddp(wrapped)
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


def _patch_init_module_for_encoder_output_gaze(base_eval, gaze_cfg: dict, rnn_cfg: dict):
    original_init_module = base_eval.init_module
    adapter_cfg = dict(gaze_cfg.get("encoder_output_adapter", {}))
    rnn_cfg = dict(rnn_cfg)
    rnn_cfg["use_video_tokens"] = False

    def init_module_with_encoder_output_gaze(*args, **kwargs):
        model = original_init_module(*args, **kwargs)
        device = next(model.parameters()).device
        embed_dim = int(model.embed_dim)
        adapter = EncoderOutputGazeAdapter(
            embed_dim=embed_dim,
            num_heads=int(adapter_cfg.get("num_heads", 4)),
            dropout=float(adapter_cfg.get("dropout", 0.0)),
        ).to(device)
        gaze_encoder = GazeTrajectoryEncoder(
            embed_dim=embed_dim,
            mode=str(rnn_cfg.get("mode_impl", "rnn")),
            input_dim=int(rnn_cfg.get("input_dim", 3)),
            hidden_dim=int(rnn_cfg.get("hidden_dim", 256)),
            num_layers=int(rnn_cfg.get("num_layers", 2)),
            bidirectional=bool(rnn_cfg.get("bidirectional", True)),
            dropout=float(rnn_cfg.get("dropout", 0.1)),
            num_tokens=int(rnn_cfg.get("num_tokens", 64)),
            modality_embed_std=float(rnn_cfg.get("modality_embed_std", 0.02)),
            video_feat_dim=0,
            video_proj_dim=int(rnn_cfg.get("video_proj_dim", 128)),
            video_fusion=str(rnn_cfg.get("video_fusion", "nearest_concat")),
            residual_alpha_init=float(rnn_cfg.get("residual_alpha_init", 0.01)),
        ).to(device)
        wrapped = EncoderOutputGazeAdaptedModel(model, adapter, gaze_encoder)
        wrapped.embed_dim = model.embed_dim
        for param in wrapped.base_model.parameters():
            param.requires_grad = False
        for param in wrapped.adapter.parameters():
            param.requires_grad = True
        for param in wrapped.gaze_encoder.parameters():
            param.requires_grad = True
        adapter_n = sum(p.numel() for p in wrapped.adapter.parameters() if p.requires_grad)
        gaze_n = sum(p.numel() for p in wrapped.gaze_encoder.parameters() if p.requires_grad)
        logger.info(
            "Attached EncoderOutputGazeAdapter: embed_dim=%d, num_heads=%d, gaze_num_tokens=%d, "
            "adapter_params=%d, gaze_params=%d",
            embed_dim,
            int(adapter_cfg.get("num_heads", 4)),
            int(rnn_cfg.get("num_tokens", 64)),
            adapter_n,
            gaze_n,
        )
        wrapped = _wrap_trainable_model_for_ddp(wrapped)
        base_eval._encoder_output_gaze_model = wrapped
        return wrapped

    base_eval.init_module = init_module_with_encoder_output_gaze


def _patch_opt_for_encoder_output_gaze(base_eval, gaze_cfg: dict, rnn_cfg: dict):
    from evals.action_anticipation_frozen.utils import CosineWDSchedule, WarmupCosineLRSchedule

    adapter_cfg = dict(gaze_cfg.get("encoder_output_adapter", {}))
    lr_mult = float(adapter_cfg.get("lr_mult", 0.05))
    wd = float(adapter_cfg.get("weight_decay", 0.0001))

    def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
        if not classifiers:
            raise ValueError("encoder_output_inject requires at least one classifier")
        model = getattr(base_eval, "_encoder_output_gaze_model", None)
        if model is None:
            raise RuntimeError("encoder_output_inject model was not registered before init_opt")
        gaze_adapter_params = trainable_encoder_output_gaze_params(model)
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
                "params": gaze_adapter_params,
                "mc_warmup_steps": adapter_warmup_steps,
                "mc_start_lr": (first_kwargs.get("start_lr") or 0.0) * lr_mult,
                "mc_ref_lr": (first_kwargs.get("ref_lr") or 0.0) * lr_mult,
                "mc_final_lr": (first_kwargs.get("final_lr") or 0.0) * lr_mult,
                "mc_ref_wd": wd,
                "mc_final_wd": wd,
            }
        )
        logger.info(
            "Optimizer split: classifiers=%d params across %d heads, encoder_output_gaze=%d params, lr_mult=%.3f",
            classifier_param_count,
            len(classifiers),
            sum(p.numel() for p in gaze_adapter_params),
            lr_mult,
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = WarmupCosineLRSchedule(optimizer, T_max=int(num_epochs * iterations_per_epoch))
        wd_scheduler = CosineWDSchedule(optimizer, T_max=int(num_epochs * iterations_per_epoch))
        scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
        return [optimizer], [scaler], [scheduler], [wd_scheduler]

    base_eval.init_opt = init_opt
