import logging
import math
import os
import shutil
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
    train_one_epoch_with_binary_input_adapter_and_pose,
    trainable_binary_input_adapter_params,
    validate_with_binary_input_adapter,
    validate_with_binary_input_adapter_and_pose,
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
from app.hdepic_lora_action_anticipation.encoder_lora import (
    assert_encoder_lora_device_consistency,
    inject_encoder_lora,
    load_encoder_lora_checkpoint,
    make_grad_scaler,
    parse_encoder_lora_cfg,
    save_encoder_lora_checkpoint,
    set_encoder_lora_trainable,
    train_one_epoch_encoder_lora,
    trainable_encoder_lora_params,
)
from app.hdepic_lora_action_anticipation.pose_slam import feature_dim_for_set
from app.hdepic_lora_action_anticipation.predictor_lora import (
    assert_predictor_lora_device_consistency,
    inject_predictor_lora,
    load_predictor_lora_checkpoint,
    parse_predictor_lora_cfg,
    save_predictor_lora_checkpoint,
    train_one_epoch_predictor_lora,
    trainable_predictor_lora_params,
)
from app.hdepic_lora_action_anticipation.pose_map_builder import GazePoseInputMapBuilder
from app.hdepic_lora_action_anticipation.encoder_output_gaze_adapter import (
    EncoderOutputGazeAdapter,
    EncoderOutputGazeAdaptedModel,
    train_one_epoch_with_encoder_output_gaze,
    trainable_encoder_output_gaze_params,
    validate_with_encoder_output_gaze,
)

logger = logging.getLogger(__name__)
logging.raiseExceptions = False

# Reference to the pristine upstream train_one_epoch, captured on first entry to
# main() before any path patches base_eval.train_one_epoch. Used by encoder LoRA
# to detect the plain baseline path.
_UPSTREAM_TRAIN_ONE_EPOCH = None


class _EarlyStopTraining(Exception):
    pass


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hdepic_reference_paths(data_cfg: dict, lora_cfg: dict) -> dict[str, Path]:
    cfg = dict(lora_cfg.get("hdepic_reference", {}))
    if data_cfg.get("dataset_train"):
        data_root = Path(data_cfg["dataset_train"]).expanduser().resolve().parent.parent
    else:
        data_root = _project_root() / "data"
    ann_root = data_root / "hd-epic-annotations" / "narrations-and-action-segments"
    return {
        "annotations_pkl": Path(
            cfg.get(
                "annotations_pkl",
                data_cfg.get("annotations_pkl", ann_root / "HD_EPIC_Narrations.pkl"),
            )
        ),
        "verb_classes_csv": Path(
            cfg.get(
                "verb_classes_csv",
                data_cfg.get("verb_classes_csv", ann_root / "HD_EPIC_verb_classes.csv"),
            )
        ),
        "noun_classes_csv": Path(
            cfg.get(
                "noun_classes_csv",
                data_cfg.get("noun_classes_csv", ann_root / "HD_EPIC_noun_classes.csv"),
            )
        ),
    }


def _build_annotations_from_df(base_path, df, file_format=1):
    video_paths, annotations = [], {}
    unique_videos = list(dict.fromkeys(df["video_id"].values))
    for uv in unique_videos:
        pid = uv.split("_")[0]
        if file_format == 0:
            fpath = os.path.join(base_path, pid, "videos", uv + ".MP4")
        else:
            fpath = os.path.join(base_path, pid, uv + ".MP4")
        if not os.path.exists(fpath):
            logging.info("file path not found fpath=%s", fpath)
            continue
        video_paths.append(fpath)
        annotations[uv] = df[df["video_id"] == uv].sort_values(by="start_frame")
    return video_paths, annotations


def _filter_annotations_hdepic_reference(
    data_cfg,
    lora_cfg,
    base_path,
    train_annotations_path,
    val_annotations_path,
    file_format=1,
):
    import pandas as pd

    paths = _hdepic_reference_paths(data_cfg, lora_cfg)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "HD-EPIC PhD-reference class-space files not found: "
            f"{missing}. Configure experiment.lora.hdepic_reference.* paths."
        )

    train_df = pd.read_csv(train_annotations_path)
    val_df = pd.read_csv(val_annotations_path)
    narr_df = pd.read_pickle(paths["annotations_pkl"])
    p01_df = narr_df[narr_df["video_id"].str.startswith("P01")].copy()
    vdf = pd.read_csv(paths["verb_classes_csv"])
    ndf = pd.read_csv(paths["noun_classes_csv"])

    # Reference code uses HD-EPIC taxonomy ids directly for verb/noun heads.
    verb_classes = {int(r["id"]): int(r["id"]) for _, r in vdf.iterrows()}
    noun_classes = {int(r["id"]): int(r["id"]) for _, r in ndf.iterrows()}

    # Reference code builds action ids from all P01 primary verb/noun pairs,
    # before splitting train and val.
    pairs = set()
    for _, row in p01_df.iterrows():
        vcs = row["verb_classes"]
        ncs = row["noun_classes"]
        if isinstance(vcs, list) and isinstance(ncs, list) and vcs and ncs:
            pairs.add((int(vcs[0]), int(ncs[0])))
    action_classes = {k: i for i, k in enumerate(pairs)}

    missing_val_pairs = sorted(
        {
            (int(v), int(n))
            for v, n in zip(val_df["verb_class"].values, val_df["noun_class"].values)
            if (int(v), int(n)) not in action_classes
        }
    )
    if missing_val_pairs:
        raise ValueError(
            "HD-EPIC PhD-reference action map does not cover all validation pairs; "
            f"missing {len(missing_val_pairs)}, first={missing_val_pairs[:5]}"
        )

    val_verb_classes = {verb_classes[int(v)] for v in val_df["verb_class"].values}
    val_noun_classes = {noun_classes[int(n)] for n in val_df["noun_class"].values}
    val_action_classes = {
        action_classes[(int(v), int(n))]
        for v, n in zip(val_df["verb_class"].values, val_df["noun_class"].values)
    }

    logger.info(
        "HD-EPIC PhD-reference class space: verbs=%d nouns=%d actions=%d "
        "p01_rows=%d train_rows=%d val_rows=%d",
        len(verb_classes),
        len(noun_classes),
        len(action_classes),
        len(p01_df),
        len(train_df),
        len(val_df),
    )

    return dict(
        verbs=verb_classes,
        nouns=noun_classes,
        actions=action_classes,
        val_verbs=val_verb_classes,
        val_nouns=val_noun_classes,
        val_actions=val_action_classes,
        train=_build_annotations_from_df(base_path, train_df, file_format=file_format),
        val=_build_annotations_from_df(base_path, val_df, file_format=file_format),
    )


def _patch_hdepic_class_space(base_eval, data_cfg: dict, lora_cfg: dict):
    import evals.action_anticipation_frozen.dataloader as dl

    if not hasattr(base_eval, "_original_filter_annotations"):
        base_eval._original_filter_annotations = base_eval.filter_annotations
    if not hasattr(dl, "_original_filter_annotations"):
        dl._original_filter_annotations = dl.filter_annotations

    raw = os.environ.get("LORA_CLASS_SPACE", lora_cfg.get("class_space", "phd_reference"))
    class_space = str(raw).lower()
    if class_space in {"train", "train_only", "upstream", "upstream_train", "legacy"}:
        base_eval.filter_annotations = base_eval._original_filter_annotations
        dl.filter_annotations = dl._original_filter_annotations
        logger.info("HD-EPIC class space: using upstream train-only behavior")
        return
    if class_space not in {"phd", "phd_reference", "jepa_arvr", "reference"}:
        raise ValueError(
            f"Unsupported experiment.lora.class_space={raw!r}; expected phd_reference or train_only"
        )

    def filter_annotations(dataset, base_path, train_annotations_path, val_annotations_path, **kwargs):
        if "ek100" not in str(dataset).lower():
            return base_eval._original_filter_annotations(
                dataset,
                base_path,
                train_annotations_path,
                val_annotations_path,
                **kwargs,
            )
        return _filter_annotations_hdepic_reference(
            data_cfg=data_cfg,
            lora_cfg=lora_cfg,
            base_path=base_path,
            train_annotations_path=train_annotations_path,
            val_annotations_path=val_annotations_path,
            **kwargs,
        )

    base_eval.filter_annotations = filter_annotations
    dl.filter_annotations = filter_annotations
    logger.info("HD-EPIC class space: PhD-reference default enabled; set class_space=train_only for upstream behavior")


def _patch_hdepic_temporal_sampling(data_cfg: dict, lora_cfg: dict):
    raw = os.environ.get("LORA_TEMPORAL_SAMPLING", lora_cfg.get("temporal_sampling", "phd_reference"))
    mode = str(raw).lower()
    if mode in {"legacy", "upstream", "current", "end_anchor", "mixed"}:
        logger.info("HD-EPIC temporal sampling: using configured legacy/current anticipation points and horizons")
        return
    if mode not in {"phd", "phd_reference", "jepa_arvr", "reference"}:
        raise ValueError(
            f"Unsupported experiment.lora.temporal_sampling={raw!r}; expected phd_reference or legacy"
        )

    val_horizon = _as_time_pair(data_cfg.get("anticipation_time_sec"))
    if val_horizon is None:
        raise ValueError("PhD-reference temporal sampling requires experiment.data.anticipation_time_sec")

    data_cfg["val_anticipation_point"] = [1.0, 1.0]
    data_cfg["train_anticipation_point"] = [1.0, 1.0]
    data_cfg["train_anticipation_time_sec"] = [float(val_horizon[0]), float(val_horizon[1])]
    logger.info(
        "HD-EPIC temporal sampling: PhD-reference default enabled "
        "(obs_end=action_start-horizon); train/val anticipation_point=[1,1], train_horizon=%s. "
        "Set temporal_sampling=legacy for configured/current behavior.",
        data_cfg["train_anticipation_time_sec"],
    )


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


def _metric_scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _metric_or_nan(metrics: dict | None, group: str, key: str) -> float:
    if not metrics or group not in metrics:
        return float("nan")
    row = metrics[group]
    if key not in row:
        return float("nan")
    return _metric_scalar(row[key])


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))


def _run_dir(args_eval: dict) -> Path:
    folder = Path(args_eval.get("folder", "."))
    tag = args_eval.get("tag")
    return folder / "action_anticipation_frozen" / tag if tag else folder / "action_anticipation_frozen"


def _extract_best_metric(val_metrics: dict, metric_name: str) -> float:
    metric_map = {
        "val-action-acc": ("action", "accuracy"),
        "val-action-recall": ("action", "recall"),
        "val-verb-acc": ("verb", "accuracy"),
        "val-verb-recall": ("verb", "recall"),
        "val-noun-acc": ("noun", "accuracy"),
        "val-noun-recall": ("noun", "recall"),
    }
    if metric_name not in metric_map:
        raise ValueError(f"Unsupported best_metric={metric_name!r}; expected one of {sorted(metric_map)}")
    group, key = metric_map[metric_name]
    if group not in val_metrics:
        raise ValueError(f"best_metric={metric_name!r} requires {group!r} metrics")
    return _metric_scalar(val_metrics[group][key])


def _patch_top5_epoch_reporting(base_eval, args_eval):
    """Project-local Top-1/3/5 CSV sidecar without changing upstream log_r0.csv schema."""

    original_train_one_epoch = base_eval.train_one_epoch
    original_validate = base_eval.validate
    state = {"last_train_metrics": None, "validation_idx": 0}
    rank = _rank()
    run_dir = _run_dir(args_eval)
    csv_path = run_dir / f"topk_log_r{rank}.csv"

    def _csv_metric(metrics: dict | None, group: str, key: str) -> float:
        if not metrics or group not in metrics or key not in metrics[group]:
            return float("nan")
        return _metric_scalar(metrics[group][key])

    def write_row(validation_idx: int, train_metrics: dict | None, val_metrics: dict):
        if rank != 0:
            return
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8") as f:
            if new_file:
                f.write(
                    "validation,"
                    "train-action-top1,train-action-top3,train-action-top5,"
                    "train-verb-top1,train-verb-top3,train-verb-top5,"
                    "train-noun-top1,train-noun-top3,train-noun-top5,"
                    "val-action-top1,val-action-top3,val-action-top5,"
                    "val-verb-top1,val-verb-top3,val-verb-top5,"
                    "val-noun-top1,val-noun-top3,val-noun-top5\n"
                )
            row = [str(validation_idx)]
            for split_metrics in (train_metrics, val_metrics):
                for group in ("action", "verb", "noun"):
                    row.extend(
                        [
                            f"{_csv_metric(split_metrics, group, 'top1_accuracy'):.5f}",
                            f"{_csv_metric(split_metrics, group, 'accuracy'):.5f}",
                            f"{_csv_metric(split_metrics, group, 'top5_accuracy'):.5f}",
                        ]
                    )
            f.write(",".join(row) + "\n")

    def train_one_epoch_with_top5_tracking(*args, **kwargs):
        train_metrics = original_train_one_epoch(*args, **kwargs)
        state["last_train_metrics"] = train_metrics
        return train_metrics

    def validate_with_top5_reporting(*args, **kwargs):
        val_metrics = original_validate(*args, **kwargs)
        state["validation_idx"] += 1
        write_row(int(state["validation_idx"]), state.get("last_train_metrics"), val_metrics)
        return val_metrics

    base_eval.train_one_epoch = train_one_epoch_with_top5_tracking
    base_eval.validate = validate_with_top5_reporting
    logger.info("Project-local Top-1/3/5 epoch CSV enabled: %s", csv_path)

    def restore():
        base_eval.train_one_epoch = original_train_one_epoch
        base_eval.validate = original_validate

    return restore


def _resolve_test_annotations_path(data_cfg: dict) -> Path | None:
    raw = data_cfg.get("dataset_test") or data_cfg.get("test_dataset")
    if raw:
        return Path(raw).expanduser()
    val_path = data_cfg.get("dataset_val")
    if not val_path:
        return None
    candidate = Path(val_path).expanduser().with_name("HD_EPIC_test_vjepa.csv")
    return candidate if candidate.exists() else None


def _split_valid_classes_from_csv(path: Path, action_is_verb_noun: bool, verb_classes, noun_classes, action_classes):
    if not action_is_verb_noun:
        return {}, {}, None

    import pandas as pd

    df = pd.read_csv(path)
    missing_pairs = sorted(
        {
            (int(v), int(n))
            for v, n in zip(df["verb_class"].values, df["noun_class"].values)
            if (int(v), int(n)) not in action_classes
        }
    )
    if missing_pairs:
        raise ValueError(
            f"dataset_test={path} contains action pairs outside the classifier action map: "
            f"missing={len(missing_pairs)}, first={missing_pairs[:5]}"
        )
    missing_verbs = sorted({int(v) for v in df["verb_class"].values if int(v) not in verb_classes})
    missing_nouns = sorted({int(n) for n in df["noun_class"].values if int(n) not in noun_classes})
    if missing_verbs or missing_nouns:
        raise ValueError(
            f"dataset_test={path} contains classes outside verb/noun maps: "
            f"verbs={missing_verbs[:5]} nouns={missing_nouns[:5]}"
        )

    return (
        {verb_classes[int(v)] for v in df["verb_class"].values},
        {noun_classes[int(n)] for n in df["noun_class"].values},
        {action_classes[(int(v), int(n))] for v, n in zip(df["verb_class"].values, df["noun_class"].values)},
    )


def _write_split_metric_row(path: Path, split: str, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    groups = ("action", "verb", "noun") if "verb" in metrics else ("action",)
    with path.open("a", encoding="utf-8") as f:
        if new_file:
            header = ["split"]
            for group in ("action", "verb", "noun"):
                header.extend(
                    [
                        f"{group}-top1",
                        f"{group}-top3",
                        f"{group}-top5",
                        f"{group}-recall5",
                    ]
                )
            f.write(",".join(header) + "\n")
        row = [split]
        for group in ("action", "verb", "noun"):
            if group in groups:
                row.extend(
                    [
                        f"{_metric_or_nan(metrics, group, 'top1_accuracy'):.5f}",
                        f"{_metric_or_nan(metrics, group, 'accuracy'):.5f}",
                        f"{_metric_or_nan(metrics, group, 'top5_accuracy'):.5f}",
                        f"{_metric_or_nan(metrics, group, 'recall'):.5f}",
                    ]
                )
            else:
                row.extend(["nan", "nan", "nan", "nan"])
        f.write(",".join(row) + "\n")


def _patch_post_train_test_eval(base_eval, args_eval, dumper=None):
    exp_cfg = args_eval.get("experiment", {}) or {}
    data_cfg = exp_cfg.get("data", {}) or {}
    lora_cfg = exp_cfg.get("lora", {}) or {}
    test_cfg = dict(lora_cfg.get("post_train_test", {}) or {})
    enabled = os.environ.get("LORA_POST_TRAIN_TEST", test_cfg.get("enabled", True))
    if isinstance(enabled, str):
        enabled = enabled.lower() not in {"0", "false", "no", "off"}
    if not enabled or bool(args_eval.get("val_only", False)):
        return (lambda: None), (lambda: None)

    test_path = _resolve_test_annotations_path(data_cfg)
    if test_path is None or not test_path.exists():
        logger.info("Post-train test eval disabled: no dataset_test/HD_EPIC_test_vjepa.csv found")
        return (lambda: None), (lambda: None)

    original_validate = base_eval.validate
    state = {"last_kwargs": None, "ran": False}
    rank = _rank()
    run_dir = _run_dir(args_eval)
    csv_path = run_dir / f"test_log_r{rank}.csv"

    def validate_with_test_capture(*args, **kwargs):
        metrics = original_validate(*args, **kwargs)
        state["last_kwargs"] = dict(kwargs)
        return metrics

    def run_pending():
        if state["ran"] or state["last_kwargs"] is None:
            return None
        state["ran"] = True
        kwargs = dict(state["last_kwargs"])
        action_is_verb_noun = bool(kwargs["action_is_verb_noun"])
        valid_verbs, valid_nouns, valid_actions = _split_valid_classes_from_csv(
            test_path,
            action_is_verb_noun,
            kwargs["verb_classes"],
            kwargs["noun_classes"],
            kwargs["action_classes"],
        )
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank_local = dist.get_rank() if dist.is_available() and dist.is_initialized() else rank
        # All training/val dataloader paths in this app (clip_balanced, metadata,
        # d3_concat) monkeypatch ek.make_webvid/dl.ek100_make_webvid to require
        # annotations_path as a (paths, annotations) tuple, not a raw CSV path --
        # build it the same way filter_annotations does for train/val.
        import pandas as pd

        test_df = pd.read_csv(test_path)
        test_annotations_path = _build_annotations_from_df(
            data_cfg.get("base_path"), test_df, file_format=data_cfg.get("file_format", 1)
        )
        _, test_loader, _ = base_eval.init_data(
            dataset=data_cfg.get("dataset"),
            training=False,
            base_path=data_cfg.get("base_path"),
            annotations_path=test_annotations_path,
            batch_size=exp_cfg.get("optimization", {}).get("batch_size"),
            frames_per_clip=data_cfg.get("frames_per_clip"),
            fps=data_cfg.get("frames_per_second"),
            anticipation_time_sec=data_cfg.get("anticipation_time_sec"),
            anticipation_point=data_cfg.get("val_anticipation_point", [0.0, 0.0]),
            crop_size=data_cfg.get("resolution", 224),
            world_size=world_size,
            rank=rank_local,
            num_workers=data_cfg.get("val_num_workers", data_cfg.get("num_workers", 12)),
            pin_mem=data_cfg.get("pin_memory", True),
            persistent_workers=False,
        )
        kwargs.update(
            data_loader=test_loader,
            ipe=test_loader.num_batches,
            valid_verbs=valid_verbs,
            valid_nouns=valid_nouns,
            valid_actions=valid_actions,
        )
        logger.info("Running post-train test eval on %s (%d iterations)", test_path, test_loader.num_batches)
        dumper_state = None
        if dumper is not None and getattr(dumper, "enabled", False):
            dumper_state = (
                getattr(dumper, "path", None),
                list(getattr(dumper, "rows", [])),
                dict(getattr(dumper, "class_maps", {})),
            )
            dumper.path = run_dir / "test_predictions.csv"
            dumper.rows = []
            dumper.class_maps = {}
        try:
            metrics = original_validate(**kwargs)
        finally:
            if dumper_state is not None:
                dumper.path, dumper.rows, dumper.class_maps = dumper_state
        if rank_local == 0:
            _write_split_metric_row(csv_path, "test", metrics)
            logger.info("Wrote post-train test metrics: %s", csv_path)
        return metrics

    base_eval.validate = validate_with_test_capture
    logger.info("Post-train test eval enabled: %s", test_path)

    def restore():
        base_eval.validate = original_validate

    return restore, run_pending


def _patch_best_checkpointing_and_early_stop(base_eval, args_eval):
    opt_cfg = dict(args_eval.get("experiment", {}).get("optimization", {}) or {})
    metric_name = str(opt_cfg.get("best_metric", "val-action-acc"))
    patience = int(opt_cfg.get("early_stopping_patience", 0) or 0)
    original_validate = base_eval.validate
    original_torch_save = torch.save
    state = {
        "pending_metric": None,
        "best_metric": None,
        "best_epoch": None,
        "bad_validations": 0,
    }

    def validate_with_best_tracking(*args, **kwargs):
        val_metrics = original_validate(*args, **kwargs)
        state["pending_metric"] = _extract_best_metric(val_metrics, metric_name)
        return val_metrics

    def copy_best_sidecars(run_dir: Path):
        for filename in ("binary_input_adapter_latest.pt", "encoder_lora_latest.pt", "predictor_lora_latest.pt"):
            src = run_dir / filename
            if src.exists():
                dst = run_dir / filename.replace("_latest.pt", "_best.pt")
                shutil.copy2(src, dst)
                logger.info("Copied best sidecar checkpoint: %s", dst)

    def save_with_best_tracking(obj, f, *args, **kwargs):
        path = Path(f) if isinstance(f, (str, os.PathLike)) else None
        is_latest = path is not None and path.name == "latest.pt"
        if not is_latest or not isinstance(obj, dict) or state["pending_metric"] is None:
            return original_torch_save(obj, f, *args, **kwargs)

        epoch = int(obj.get("epoch", 0) or 0)
        current = float(state["pending_metric"])
        improved = state["best_metric"] is None or current > float(state["best_metric"])
        if improved:
            state["best_metric"] = current
            state["best_epoch"] = epoch
            state["bad_validations"] = 0
        else:
            state["bad_validations"] += 1

        obj["best_metric"] = state["best_metric"]
        obj["best_metric_name"] = metric_name
        obj["best_epoch"] = state["best_epoch"]
        original_torch_save(obj, f, *args, **kwargs)

        predictor_lora_model = getattr(base_eval, "_predictor_lora_model", None)
        if predictor_lora_model is not None:
            save_predictor_lora_checkpoint(predictor_lora_model, path.parent / "predictor_lora_latest.pt")

        # Video-only / encoder-LoRA baseline path also needs sidecars: previously only the
        # binary_input_adapter validate wrappers wrote encoder_lora_latest.pt, so
        # gaze:none runs left encoder LoRA unrecoverable after preempt/cancel.
        encoder_lora_model = getattr(base_eval, "_encoder_lora_model", None)
        if encoder_lora_model is None:
            encoder_lora_model = getattr(base_eval, "_binary_input_adapter_model", None)
        if encoder_lora_model is not None:
            save_encoder_lora_checkpoint(encoder_lora_model, path.parent / "encoder_lora_latest.pt")

        if improved:
            best_path = path.with_name("best.pt")
            original_torch_save(obj, best_path, *args, **kwargs)
            copy_best_sidecars(path.parent)
            logger.info("New best checkpoint at epoch %d: %s=%.5f -> %s", epoch, metric_name, current, best_path)
        else:
            logger.info(
                "No best improvement at epoch %d: %s=%.5f best=%.5f@%s patience=%d/%d",
                epoch,
                metric_name,
                current,
                float(state["best_metric"]),
                state["best_epoch"],
                state["bad_validations"],
                patience,
            )
            if patience > 0 and state["bad_validations"] >= patience:
                raise _EarlyStopTraining(
                    f"Early stopping at epoch {epoch}: {state['bad_validations']} validations without {metric_name} improvement"
                )
        return None

    base_eval.validate = validate_with_best_tracking
    torch.save = save_with_best_tracking
    base_eval.torch.save = save_with_best_tracking
    logger.info("Project-local best checkpointing enabled: best_metric=%s early_stopping_patience=%d", metric_name, patience)

    def restore():
        base_eval.validate = original_validate
        torch.save = original_torch_save
        base_eval.torch.save = original_torch_save

    return restore


class Top3AccuracyRecallAt5:
    """Metric adapter for matching the reference script's reporting convention.

    The upstream V-JEPA action anticipation eval instantiates ClassMeanRecall(k=5)
    and logs the returned "accuracy" and "recall" fields. In upstream code,
    both are based on top-5 predictions. The reference HD-EPIC script reports
    Top-3 accuracy and class-mean Recall@5, so this adapter keeps the same return
    keys while changing only "accuracy" to Top-3 and also exposing Top-1/Top-5.
    """

    def __init__(self, num_classes: int, device: torch.device, k=5):
        self.num_classes = num_classes
        self.top1_tp = torch.zeros(num_classes).to(device)
        self.top1_fn = torch.zeros(num_classes).to(device)
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
        preds1 = logits.argmax(dim=1)
        preds3 = logits.topk(k3, dim=1).indices
        preds5 = logits.topk(k5, dim=1).indices

        for p1, p3, p5, gt in zip(preds1, preds3, preds5, labels):
            if gt == p1:
                self.top1_tp[gt] += 1
            else:
                self.top1_fn[gt] += 1
            if gt in p3:
                self.top3_tp[gt] += 1
            else:
                self.top3_fn[gt] += 1
            if gt in p5:
                self.r5_tp[gt] += 1
            else:
                self.r5_fn[gt] += 1

        top1_tp, top1_fn = self.top1_tp.clone(), self.top1_fn.clone()
        top3_tp, top3_fn = self.top3_tp.clone(), self.top3_fn.clone()
        r5_tp, r5_fn = self.r5_tp.clone(), self.r5_fn.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(top1_tp)
            dist.all_reduce(top1_fn)
            dist.all_reduce(top3_tp)
            dist.all_reduce(top3_fn)
            dist.all_reduce(r5_tp)
            dist.all_reduce(r5_fn)

        top1_total = torch.sum(top1_tp + top1_fn)
        top1_accuracy = 100.0 * torch.sum(top1_tp) / torch.clamp(top1_total, min=1.0)

        top3_total = torch.sum(top3_tp + top3_fn)
        top3_accuracy = 100.0 * torch.sum(top3_tp) / torch.clamp(top3_total, min=1.0)

        r5_seen = torch.sum((r5_tp + r5_fn) > 0)
        r5_recall = 100.0 * torch.sum(r5_tp / (r5_tp + r5_fn + eps)) / torch.clamp(r5_seen, min=1)

        top5_total = torch.sum(r5_tp + r5_fn)
        top5_accuracy = 100.0 * torch.sum(r5_tp) / torch.clamp(top5_total, min=1.0)

        return dict(
            recall=r5_recall,
            accuracy=top3_accuracy,
            top1_accuracy=top1_accuracy,
            top5_accuracy=top5_accuracy,
        )


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        ref = base.weight
        self.dropout.to(device=ref.device)
        self.lora_A.to(device=ref.device, dtype=ref.dtype)
        self.lora_B.to(device=ref.device, dtype=ref.dtype)
        with torch.no_grad():
            self.lora_A.weight.copy_(torch.randn_like(self.lora_A.weight) * 0.02)
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


def _log_trainable_params(classifier: nn.Module, label: str = "LoRA classifier"):
    total = sum(p.numel() for p in classifier.parameters())
    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    logger.info("%s trainable params: %d / %d (%.2f%%)", label, trainable, total, pct)


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
    probe_train_mode = str(lora_cfg.get("probe_train_mode", "lora")).lower()
    if probe_train_mode not in {"lora", "full"}:
        raise ValueError(f"Unsupported lora.probe_train_mode={probe_train_mode!r}; expected 'lora' or 'full'")
    # rnn_fuse/mlp_fuse/pose_rnn_fuse/multimodal_rnn_fuse: pre-ViT-L token-level gaze/pose
    # fusion modes (archived branches B2, B7, B10). Not used in current ViT-L singleprobe.
    # Active mode: binary_input_adapter_gaze_pose_matrix (B11 ViT-L).
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
            if pretrained_probe:
                _load_pooler_from_probe(classifier, pretrained_probe)
            replaced = []
            if probe_train_mode == "lora":
                replaced = _replace_linears_with_lora(classifier.pooler, rank=rank, alpha=alpha, dropout=dropout)
            if use_gaze_fusion:
                if use_gaze_branch:
                    attach_gaze_encoder_to_classifier(classifier, embed_dim=embed_dim, rnn_cfg={**rnn_cfg, "input_dim": 3})
                if use_pose_branch:
                    attach_pose_encoder_to_classifier(classifier, embed_dim=embed_dim, rnn_cfg=rnn_cfg)
            if probe_train_mode == "lora":
                _freeze_for_lora(classifier, train_heads=train_heads)
            else:
                for param in classifier.parameters():
                    param.requires_grad = train_heads
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
            if probe_train_mode == "lora":
                logger.info("Inserted LoRA into %d pooler Linear layers", len(replaced))
            else:
                logger.info("Using full AttentiveClassifier probe training for classifier head %d", head_idx)
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
            _log_trainable_params(
                classifier,
                label="Full probe" if probe_train_mode == "full" else "LoRA classifier",
            )
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


def _patch_load_checkpoint_for_binary_input_adapter(base_eval, gaze_cfg: dict):
    """Extend checkpoint resume to also restore binary input-adapter weights.

    Upstream `latest.pt` stores classifier/optimizer/scaler only. For binary-map
    modes, adapter weights are saved as a sidecar file
    (`binary_input_adapter_latest.pt`) by the local validate wrapper.
    """

    def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
        logger.info(f"read-path: {r_path}")
        checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

        # Restore classifiers (includes pose GRU encoder inside classifier state).
        msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, checkpoint["classifiers"])]
        model = getattr(base_eval, "_binary_input_adapter_model", None)
        if model is None:
            logger.warning("binary_input_adapter model missing during resume; skipping adapter weight restore")
        else:
            adapter_cfg = dict(gaze_cfg.get("input_adapter", {}))
            adapter_path = adapter_cfg.get("load_checkpoint_path") or gaze_cfg.get("adapter_checkpoint_path")
            if adapter_path:
                _load_binary_input_adapter_checkpoint(_unwrap_ddp(model).input_adapter, str(adapter_path))
            else:
                logger.warning("No adapter checkpoint path configured; resuming classifier/optimizer only")

        if val_only:
            logger.info(f"loaded pretrained classifier from epoch with msg: {msg}")
            return classifiers, opt, scaler, 0

        epoch = checkpoint["epoch"]
        logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")
        [o.load_state_dict(c) for o, c in zip(opt, checkpoint["opt"])]
        if scaler is not None:
            [s.load_state_dict(c) for s, c in zip(scaler, checkpoint["scaler"])]
        logger.info(f"loaded optimizers from epoch {epoch}")
        return classifiers, opt, scaler, epoch

    base_eval.load_checkpoint = load_checkpoint


def _patch_load_checkpoint_for_encoder_lora(base_eval, encoder_lora_cfg: dict):
    original_load_checkpoint = base_eval.load_checkpoint

    def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
        classifiers, opt, scaler, epoch = original_load_checkpoint(
            device,
            r_path,
            classifiers,
            opt,
            scaler,
            val_only=val_only,
        )
        path = encoder_lora_cfg.get("load_checkpoint_path") or encoder_lora_cfg.get("checkpoint_path")
        model = getattr(base_eval, "_binary_input_adapter_model", None)
        if model is None:
            model = getattr(base_eval, "_encoder_lora_model", None)
        if path and model is not None and Path(path).exists():
            missing, unexpected = load_encoder_lora_checkpoint(model, str(path), strict=False)
            logger.info(
                "Loaded encoder LoRA checkpoint from %s; missing=%d unexpected=%d",
                path,
                len(missing),
                len(unexpected),
            )
        elif path:
            logger.warning("Encoder LoRA checkpoint not found: %s", path)
        return classifiers, opt, scaler, epoch

    base_eval.load_checkpoint = load_checkpoint


def main(args_eval, resume_preempt=False):
    lora_cfg = args_eval.get("experiment", {}).get("lora", {})
    if not lora_cfg.get("enabled", True):
        raise ValueError("app.hdepic_lora_action_anticipation requires experiment.lora.enabled=true")

    import evals.action_anticipation_frozen.eval as base_eval

    # Capture the upstream train loop before any path patches it, so encoder LoRA
    # can tell whether it must supply its own grad-flowing baseline loop.
    global _UPSTREAM_TRAIN_ONE_EPOCH
    if _UPSTREAM_TRAIN_ONE_EPOCH is None:
        _UPSTREAM_TRAIN_ONE_EPOCH = base_eval.train_one_epoch

    if bool(lora_cfg.get("align_reference_metrics", True)):
        logger.info(
            "Using aligned metrics: Top-1/Top-3/Top-5 accuracy tracked; "
            "accuracy field=Top-3, recall=class-mean Recall@5"
        )
        base_eval.ClassMeanRecall = Top3AccuracyRecallAt5

    gaze_cfg = dict(lora_cfg.get("gaze", {}))
    pred_dump_cfg = dict(lora_cfg.get("prediction_dump", {}))
    data_cfg = args_eval.get("experiment", {}).get("data", {})
    _patch_hdepic_class_space(base_eval, data_cfg, lora_cfg)
    _patch_hdepic_temporal_sampling(data_cfg, lora_cfg)

    # Encoder (trunk) LoRA: optional fine-tuning of the frozen V-JEPA2 encoder.
    # Independent of the gaze path so baseline and gaze runs can be matched.
    encoder_lora_cfg = parse_encoder_lora_cfg(lora_cfg)
    if encoder_lora_cfg is not None:
        logger.info("Encoder LoRA enabled: %s", encoder_lora_cfg)
        # Must set these outside the gaze/metadata branch: video-only (gaze:none)
        # never entered that block, so encoder_lora_latest.pt was never written/loaded.
        encoder_run_dir = _run_dir(args_eval)
        encoder_lora_cfg.setdefault("checkpoint_path", str(encoder_run_dir / "encoder_lora_latest.pt"))
        if args_eval.get("resume_checkpoint", False):
            encoder_lora_cfg.setdefault("load_checkpoint_path", str(encoder_lora_cfg["checkpoint_path"]))

    # Predictor LoRA: same strategy as encoder LoRA (same LoRALinear, same
    # default rank/alpha/dropout/target_suffixes) but targets model.predictor
    # instead of model.encoder. Independent config block so it can run with
    # the encoder frozen (validated path) or alongside encoder LoRA.
    predictor_lora_cfg = parse_predictor_lora_cfg(lora_cfg)
    if predictor_lora_cfg is not None:
        logger.info("Predictor LoRA enabled: %s", predictor_lora_cfg)
        # Default this unconditionally so the no-gaze baseline path can also
        # resume predictor LoRA weights via RESUME_CHECKPOINT=1 -- see the save
        # hook in _patch_best_checkpointing_and_early_stop, which also writes
        # predictor_lora_latest.pt on every latest.pt save.
        predictor_run_dir = _run_dir(args_eval)
        predictor_lora_cfg.setdefault("checkpoint_path", str(predictor_run_dir / "predictor_lora_latest.pt"))
        if args_eval.get("resume_checkpoint", False):
            predictor_lora_cfg.setdefault("load_checkpoint_path", str(predictor_lora_cfg["checkpoint_path"]))

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

    # Current default is native/metric_wise_max (ViT-L singleprobe standard).
    # scope=filtered and aggregation=single_head/action_top3_single_head are pre-ViT-L modes
    # (ViT-G B5/B11-matrix filtered+action-head eval); kept only for reproducing archived runs.
    val_metric_scope = str(os.environ.get("LORA_VAL_METRIC_SCOPE", lora_cfg.get("val_metric_scope", "native"))).lower()
    if val_metric_scope not in {"native", "filtered"}:
        raise ValueError(f"Unsupported lora.val_metric_scope={val_metric_scope!r}; expected native or filtered")
    val_metric_aggregation = str(
        os.environ.get("LORA_VAL_METRIC_AGGREGATION", lora_cfg.get("val_metric_aggregation", "metric_wise_max"))
    ).lower()
    if val_metric_aggregation == "action_top3_single_head":  # legacy alias
        val_metric_aggregation = "single_head"
    if val_metric_aggregation not in {"metric_wise_max", "single_head"}:
        raise ValueError(
            f"Unsupported lora.val_metric_aggregation={val_metric_aggregation!r}; "
            "expected metric_wise_max or single_head"
        )
    fixed_raw = os.environ.get("VAL_FIXED_HEAD_INDEX", lora_cfg.get("val_fixed_head_index"))
    val_fixed_head_index = int(fixed_raw) if fixed_raw not in (None, "") else None
    val_metric_kwargs = dict(
        val_metric_scope=val_metric_scope,
        val_metric_aggregation=val_metric_aggregation,
        val_fixed_head_index=val_fixed_head_index,
    )
    logger.info(
        "Validation metrics: scope=%s aggregation=%s fixed_head=%s",
        val_metric_scope,
        val_metric_aggregation,
        val_fixed_head_index,
    )
    gaze_mode = str(gaze_cfg.get("mode", "none")).lower()
    binary_input_adapter_pose_rnn_fuse_enabled = gaze_mode == "binary_input_adapter_pose_rnn_fuse"
    binary_input_adapter_gaze_pose_matrix_enabled = gaze_mode == "binary_input_adapter_gaze_pose_matrix"
    binary_input_adapter_enabled = gaze_mode in {
        "binary_input_adapter",
        "binary_input_adapter_pose_rnn_fuse",
        "binary_input_adapter_gaze_pose_matrix",
    }
    # encoder_output_inject: zero-init encoder-output gaze inject (B8, design phase, never trained).
    encoder_output_inject_enabled = gaze_mode == "encoder_output_inject"
    # past_window_baseline: B3 long-horizon past-window curriculum (pre-ViT-L, archived).
    # Not used in current ViT-L singleprobe; kept for reproducing B3 runs.
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
    if gaze_mode in {
        "rnn_fuse",
        "mlp_fuse",
        "binary_input_adapter",
        "binary_input_adapter_pose_rnn_fuse",
        "binary_input_adapter_gaze_pose_matrix",
        "encoder_output_inject",
    } and bool(
        gaze_cfg.get("use_motion", False)
    ):
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
    traj_mode = (
        "pose_rnn_fuse"
        if binary_input_adapter_pose_rnn_fuse_enabled
        else (gaze_mode if gaze_mode in {"rnn_fuse", "mlp_fuse", "pose_rnn_fuse", "multimodal_rnn_fuse"} else None)
    )
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
        "binary_input_adapter_pose_rnn_fuse",
        "binary_input_adapter_gaze_pose_matrix",
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
    dumper = None
    local_validate_patched = False
    token_gate_module = None
    d3_concat_cfg = dict(lora_cfg.get("d3_concat_train") or {})
    if not d3_concat_cfg and lora_cfg.get("d3_oracle_concat_train"):
        d3_concat_cfg = dict(lora_cfg["d3_oracle_concat_train"])
        d3_concat_cfg.setdefault("future_mode", "oracle")
        d3_concat_cfg.setdefault("experiment_id", "E20")
        d3_concat_cfg.setdefault("eval_json_name", "d3_e20_oracle_ctx_eval.json")
    d3_concat_enabled = bool(d3_concat_cfg.get("enabled", False))
    lora_init_classifier_fn = _make_lora_init_classifier(
        lora_cfg,
        traj_mode=traj_mode,
        rnn_cfg=rnn_cfg,
        token_gate=token_gate_module if getattr(token_gate_module, "learnable_gate", False) else None,
    )
    if d3_concat_enabled:
        if past_window_cfg.get("enabled", False):
            raise ValueError("d3_concat_train cannot be combined with past_window_baseline")
        from app.hdepic_lora_action_anticipation.d3_concat_probe_train import enable_d3_concat_probe_training

        enable_d3_concat_probe_training(
            base_eval, args_eval, d3_concat_cfg, data_cfg, lora_init_classifier_fn
        )
        local_validate_patched = True
    elif needs_metadata:
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
        if encoder_lora_cfg is not None:
            encoder_lora_cfg.setdefault("checkpoint_path", str(run_dir / "encoder_lora_latest.pt"))
            if args_eval.get("resume_checkpoint", False):
                encoder_lora_cfg.setdefault("load_checkpoint_path", str(encoder_lora_cfg["checkpoint_path"]))
        if pred_dump_cfg.get("enabled", False):
            pred_dump_cfg.setdefault("path", str(run_dir / "val_predictions.csv"))
        dumper = PredictionDumper(pred_dump_cfg, run_dir, rank)

        if binary_input_adapter_enabled:
            logger.info("Enabling binary_input_adapter: RGB + online binary gaze map -> tiny residual RGB adapter")
            gaze_cfg.setdefault("adapter_checkpoint_path", str(run_dir / "binary_input_adapter_latest.pt"))
            if args_eval.get("resume_checkpoint", False):
                gaze_cfg.setdefault("input_adapter", {})
                gaze_cfg["input_adapter"].setdefault("load_checkpoint_path", str(gaze_cfg["adapter_checkpoint_path"]))
            gaze_cfg.setdefault("rank", rank)
            if binary_input_adapter_gaze_pose_matrix_enabled:
                gaze_cfg.setdefault("input_adapter", {})
                gaze_cfg["input_adapter"].setdefault("in_channels", 5)
                gaze_cfg["input_adapter"].setdefault("temporal_kernel", 3)
                gaze_cfg.setdefault("pose", {})
                gaze_cfg["pose"].setdefault("enabled", True)
                gaze_cfg["pose"].setdefault("interframe_k_max", 128)
                gaze_cfg.setdefault("pose_map", {})
                gaze_cfg["pose_map"].setdefault("patch_height", 128)
                gaze_cfg["pose_map"].setdefault("patch_width", 9)
                gaze_cfg["pose_map"].setdefault("layout", "topleft")
                gaze_cfg["pose_map"].setdefault("normalize", "none")
            _patch_init_module_for_binary_input_adapter(base_eval, gaze_cfg)
            _patch_opt_for_binary_input_adapter(base_eval, gaze_cfg)
            if args_eval.get("resume_checkpoint", False):
                _patch_load_checkpoint_for_binary_input_adapter(base_eval, gaze_cfg)
            map_builder = BinaryGazeMapBuilder(gaze_cfg, gate=gate)
            if binary_input_adapter_gaze_pose_matrix_enabled:
                logger.info(
                    "Enabling binary_input_adapter_gaze_pose_matrix: gaze binary map + inter-frame pose matrix -> 5ch adapter (no probe GRU)"
                )
                map_builder = GazePoseInputMapBuilder(gaze_cfg, gate=gate)
                base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_binary_input_adapter(
                    base_eval, map_builder, **kwargs
                )
                base_eval.validate = lambda **kwargs: validate_with_binary_input_adapter(
                    base_eval, map_builder, dumper, **val_metric_kwargs, **kwargs
                )
            elif binary_input_adapter_pose_rnn_fuse_enabled:
                logger.info("Enabling hybrid mode: binary gaze adapter + pose_rnn_fuse tokens at probe")
                pose_loader = PoseTrajectoryLoader(gaze_cfg, gate=gate)
                base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_binary_input_adapter_and_pose(
                    base_eval, map_builder, pose_loader, **kwargs
                )
                base_eval.validate = lambda **kwargs: validate_with_binary_input_adapter_and_pose(
                    base_eval, map_builder, pose_loader, dumper, **val_metric_kwargs, **kwargs
                )
            else:
                base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_binary_input_adapter(
                    base_eval, map_builder, **kwargs
                )
                base_eval.validate = lambda **kwargs: validate_with_binary_input_adapter(
                    base_eval, map_builder, dumper, **val_metric_kwargs, **kwargs
                )
            local_validate_patched = True

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
                base_eval, dumper, traj_loader, **val_metric_kwargs, **kwargs
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
                **val_metric_kwargs,
                **kwargs,
            )
            local_validate_patched = True
        elif gaze_mode == "token_gate" or bool(pred_dump_cfg.get("enabled", False)):
            base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_gaze(
                base_eval, gate, traj_loader=traj_loader, **kwargs
            )
            base_eval.validate = lambda **kwargs: validate_with_gaze(
                base_eval, gate, dumper, traj_loader=traj_loader, hidden_dump=hidden_dump, **val_metric_kwargs, **kwargs
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
    if not local_validate_patched:
        from app.hdepic_lora_action_anticipation.val_metrics import validate_with_standard_model

        if dumper is None:
            rank = _rank()
            run_dir = _run_dir(args_eval)
            dumper = PredictionDumper(pred_dump_cfg, run_dir, rank)

        base_eval.validate = lambda **kwargs: validate_with_standard_model(
            base_eval, dumper, **val_metric_kwargs, **kwargs
        )
        local_validate_patched = True
        logger.info("Using app-local standard validate wrapper for native/filtered metric logging")

    if not d3_concat_enabled:
        base_eval.init_classifier = lora_init_classifier_fn
    if getattr(token_gate_module, "learnable_gate", False):
        _patch_load_checkpoint_for_learnable_token_gate(base_eval)
    if traj_mode is not None and not binary_input_adapter_pose_rnn_fuse_enabled:
        _patch_opt_for_gaze_encoder(base_eval, gaze_lr_mult=float(rnn_cfg.get("gaze_lr_mult", 5.0)))
    if encoder_lora_cfg is not None:
        # train_one_epoch is left at the upstream (no_grad) default only on the
        # plain baseline path; gaze/binary paths set a project-local grad-flowing
        # loop. Detect that so we only swap in the encoder-LoRA baseline loop when
        # nothing else owns training.
        if args_eval.get("resume_checkpoint", False):
            _patch_load_checkpoint_for_encoder_lora(base_eval, encoder_lora_cfg)
        baseline_train_loop = base_eval.train_one_epoch is _UPSTREAM_TRAIN_ONE_EPOCH
        _patch_for_encoder_lora(base_eval, encoder_lora_cfg, baseline_train_loop=baseline_train_loop)
    if predictor_lora_cfg is not None:
        # Same baseline-train-loop detection as encoder LoRA above. NOTE: if both
        # encoder_lora_cfg and predictor_lora_cfg are enabled together, whichever
        # block runs first claims the grad-flowing train_one_epoch and only its
        # own LoRA params get the non-finite-grad discard/zero check each step --
        # the other LoRA's grads still flow and get optimizer steps, just without
        # that per-step guard. Not an issue for the validated predictor-LoRA-only
        # (encoder frozen) path.
        if args_eval.get("resume_checkpoint", False):
            _patch_load_checkpoint_for_predictor_lora(base_eval, predictor_lora_cfg)
        baseline_train_loop = base_eval.train_one_epoch is _UPSTREAM_TRAIN_ONE_EPOCH
        _patch_for_predictor_lora(base_eval, predictor_lora_cfg, baseline_train_loop=baseline_train_loop)
    if base_eval.train_one_epoch is _UPSTREAM_TRAIN_ONE_EPOCH:
        from app.hdepic_lora_action_anticipation.val_metrics import train_one_epoch_with_standard_model

        base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_with_standard_model(base_eval, **kwargs)
        logger.info("Using app-local standard train wrapper for Top-1/3/5 metric logging")
    restore_post_train_test, run_post_train_test = _patch_post_train_test_eval(base_eval, args_eval, dumper=dumper)
    restore_top5_reporting = _patch_top5_epoch_reporting(base_eval, args_eval)
    restore_best_checkpointing = _patch_best_checkpointing_and_early_stop(base_eval, args_eval)
    try:
        result = base_eval.main(args_eval=args_eval, resume_preempt=resume_preempt)
        run_post_train_test()
        return result
    except _EarlyStopTraining as exc:
        logger.info("%s", exc)
        run_post_train_test()
        return None
    finally:
        restore_best_checkpointing()
        restore_top5_reporting()
        restore_post_train_test()


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
            in_channels=int(cfg.get("in_channels", 4)),
        ).to(next(model.parameters()).device)
        adapter_ckpt = cfg.get("load_checkpoint_path")
        if adapter_ckpt:
            _load_binary_input_adapter_checkpoint(adapter, str(adapter_ckpt))
        wrapped = BinaryInputAdaptedModel(model, adapter)
        wrapped.embed_dim = model.embed_dim
        for param in wrapped.base_model.parameters():
            param.requires_grad = False
        if cfg.get("freeze", False):
            for param in wrapped.input_adapter.parameters():
                param.requires_grad = False
            logger.info("Froze binary input adapter params for stage-2 warm start")
        else:
            for param in wrapped.input_adapter.parameters():
                param.requires_grad = True
        restored_encoder_lora = set_encoder_lora_trainable(wrapped.base_model, trainable=True)
        if restored_encoder_lora and cfg.get("freeze_encoder_lora", False):
            set_encoder_lora_trainable(wrapped.base_model, trainable=False)
            logger.info("Froze encoder LoRA params after binary adapter init")
        elif restored_encoder_lora:
            logger.info("Restored encoder-LoRA trainable params after binary adapter freeze: %d", restored_encoder_lora)
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


def _opt_ref_lr(kwargs: dict) -> float | None:
    ref_lr = kwargs.get("ref_lr")
    return kwargs.get("lr") if ref_lr is None else ref_lr


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
            head_idx = len(param_groups)
            base_params = [p for p in classifier.parameters() if p.requires_grad]
            classifier_param_count += sum(p.numel() for p in base_params)
            warmup_steps = int((kwargs.get("warmup") or 0.0) * iterations_per_epoch)
            param_groups.append(
                {
                    "diagnostic_name": f"classifier_head_{head_idx}",
                    "params": base_params,
                    "mc_warmup_steps": warmup_steps,
                    "mc_start_lr": kwargs.get("start_lr"),
                    "mc_ref_lr": _opt_ref_lr(kwargs),
                    "mc_final_lr": kwargs.get("final_lr"),
                    "mc_ref_wd": kwargs.get("ref_wd"),
                    "mc_final_wd": kwargs.get("final_wd"),
                }
            )
        adapter_warmup_steps = int((first_kwargs.get("warmup") or 0.0) * iterations_per_epoch)
        adapter_ref_lr = _opt_ref_lr(first_kwargs)
        param_groups.append(
            {
                "diagnostic_name": "binary_input_adapter",
                "params": adapter_params,
                "mc_warmup_steps": adapter_warmup_steps,
                "mc_start_lr": (first_kwargs.get("start_lr") or 0.0) * lr_mult,
                "mc_ref_lr": (adapter_ref_lr or 0.0) * lr_mult,
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
        scaler = make_grad_scaler(use_bfloat16)
        return [optimizer], [scaler], [scheduler], [wd_scheduler]

    base_eval.init_opt = init_opt


def _patch_for_encoder_lora(base_eval, enc_cfg: dict, baseline_train_loop: bool):
    """Inject encoder LoRA, add its params to the optimizer, and (for the plain
    baseline path) install a grad-flowing train loop.

    Wraps whatever ``init_module`` / ``init_opt`` are currently installed so it
    composes with the binary-input-adapter path (which also patches them).
    """
    inner_init_module = base_eval.init_module
    inner_init_opt = base_eval.init_opt

    def init_module_with_encoder_lora(*args, **kwargs):
        model = inner_init_module(*args, **kwargs)
        inject_encoder_lora(
            model,
            rank=enc_cfg["rank"],
            alpha=enc_cfg["alpha"],
            dropout=enc_cfg["dropout"],
            last_n_blocks=enc_cfg["last_n_blocks"],
            target_suffixes=enc_cfg.get("target_suffixes", ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")),
            )
        if enc_cfg.get("activation_checkpointing", True):
            mdl = model.module if hasattr(model, "module") and not hasattr(model, "encoder") else model
            enc = getattr(getattr(mdl, "base_model", mdl), "encoder", getattr(mdl, "encoder", None))
            if enc is not None and hasattr(enc, "use_activation_checkpointing"):
                enc.use_activation_checkpointing = True
                logger.info("Enabled encoder activation checkpointing for encoder LoRA")
        warm_path = enc_cfg.get("load_checkpoint_path")
        if enc_cfg.get("warm_start_at_init") and warm_path and Path(warm_path).exists():
            missing, unexpected = load_encoder_lora_checkpoint(model, str(warm_path), strict=False)
            logger.info(
                "Warm-started encoder LoRA from %s; missing=%d unexpected=%d",
                warm_path,
                len(missing),
                len(unexpected),
            )
        elif enc_cfg.get("warm_start_at_init") and warm_path:
            logger.warning("Encoder LoRA warm-start checkpoint not found: %s", warm_path)
        if enc_cfg.get("freeze"):
            frozen = set_encoder_lora_trainable(model, trainable=False)
            logger.info("Froze encoder LoRA params after init: %d", frozen)
        assert_encoder_lora_device_consistency(model)
        base_eval._encoder_lora_model = model
        return model

    base_eval.init_module = init_module_with_encoder_lora

    def init_opt_with_encoder_lora(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
        optimizers, scalers, schedulers, wd_schedulers = inner_init_opt(
            classifiers=classifiers,
            iterations_per_epoch=iterations_per_epoch,
            opt_kwargs=opt_kwargs,
            num_epochs=num_epochs,
            use_bfloat16=use_bfloat16,
        )
        model = getattr(base_eval, "_binary_input_adapter_model", None)
        if model is None:
            model = getattr(base_eval, "_encoder_output_gaze_model", None)
        if model is None:
            model = getattr(base_eval, "_encoder_lora_model", None)
        if model is None:
            raise RuntimeError("encoder_lora model was not registered before init_opt")
        enc_params = trainable_encoder_lora_params(model)
        if not enc_params:
            raise RuntimeError("encoder LoRA is enabled but no trainable encoder-LoRA params were found")
        first_kwargs = opt_kwargs[0]
        lr_mult = enc_cfg["lr_mult"]
        ref_lr = _opt_ref_lr(first_kwargs)
        warmup_steps = int((first_kwargs.get("warmup") or 0.0) * iterations_per_epoch)
        enc_group = {
            "diagnostic_name": "encoder_lora",
            "params": enc_params,
            "mc_warmup_steps": warmup_steps,
            "mc_start_lr": (first_kwargs.get("start_lr") or 0.0) * lr_mult,
            "mc_ref_lr": (ref_lr or 0.0) * lr_mult,
            "mc_final_lr": (first_kwargs.get("final_lr") or 0.0) * lr_mult,
            "mc_ref_wd": enc_cfg["weight_decay"],
            "mc_final_wd": enc_cfg["weight_decay"],
        }
        # encoder-LoRA grads flow through the single shared encoder, so they only
        # make sense on optimizer[0] (the loop steps optimizer[0] for the model).
        optimizers[0].add_param_group(enc_group)
        logger.info(
            "Added encoder-LoRA optimizer group to optimizer[0]: params=%d lr_mult=%.3f wd=%.5f",
            sum(p.numel() for p in enc_params),
            lr_mult,
            enc_cfg["weight_decay"],
        )
        return optimizers, scalers, schedulers, wd_schedulers

    base_eval.init_opt = init_opt_with_encoder_lora

    if baseline_train_loop:
        logger.info("Encoder LoRA baseline path: installing grad-flowing train_one_epoch")
        base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_encoder_lora(base_eval, **kwargs)


def _patch_load_checkpoint_for_predictor_lora(base_eval, predictor_lora_cfg: dict):
    original_load_checkpoint = base_eval.load_checkpoint

    def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
        classifiers, opt, scaler, epoch = original_load_checkpoint(
            device,
            r_path,
            classifiers,
            opt,
            scaler,
            val_only=val_only,
        )
        path = predictor_lora_cfg.get("load_checkpoint_path") or predictor_lora_cfg.get("checkpoint_path")
        model = getattr(base_eval, "_binary_input_adapter_model", None)
        if model is None:
            model = getattr(base_eval, "_encoder_lora_model", None)
        if model is None:
            model = getattr(base_eval, "_predictor_lora_model", None)
        if path and model is not None and Path(path).exists():
            missing, unexpected = load_predictor_lora_checkpoint(model, str(path), strict=False)
            logger.info(
                "Loaded predictor LoRA checkpoint from %s; missing=%d unexpected=%d",
                path,
                len(missing),
                len(unexpected),
            )
        elif path:
            logger.warning("Predictor LoRA checkpoint not found: %s", path)
        return classifiers, opt, scaler, epoch

    base_eval.load_checkpoint = load_checkpoint


def _patch_for_predictor_lora(base_eval, pred_cfg: dict, baseline_train_loop: bool):
    """Inject predictor LoRA, add its params to the optimizer, and (for the
    plain baseline path) install a grad-flowing train loop.

    Mirrors `_patch_for_encoder_lora`; see that function for the composition
    rationale with the binary-input-adapter path.
    """
    inner_init_module = base_eval.init_module
    inner_init_opt = base_eval.init_opt

    def init_module_with_predictor_lora(*args, **kwargs):
        model = inner_init_module(*args, **kwargs)
        inject_predictor_lora(
            model,
            rank=pred_cfg["rank"],
            alpha=pred_cfg["alpha"],
            dropout=pred_cfg["dropout"],
            last_n_blocks=pred_cfg["last_n_blocks"],
            target_suffixes=pred_cfg.get("target_suffixes", ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")),
        )
        if pred_cfg.get("activation_checkpointing", False):
            mdl = model.module if hasattr(model, "module") and not hasattr(model, "predictor") else model
            pred = getattr(getattr(mdl, "base_model", mdl), "predictor", getattr(mdl, "predictor", None))
            if pred is not None and hasattr(pred, "use_activation_checkpointing"):
                pred.use_activation_checkpointing = True
                logger.info("Enabled predictor activation checkpointing for predictor LoRA")
        assert_predictor_lora_device_consistency(model)
        base_eval._predictor_lora_model = model
        return model

    base_eval.init_module = init_module_with_predictor_lora

    def init_opt_with_predictor_lora(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
        optimizers, scalers, schedulers, wd_schedulers = inner_init_opt(
            classifiers=classifiers,
            iterations_per_epoch=iterations_per_epoch,
            opt_kwargs=opt_kwargs,
            num_epochs=num_epochs,
            use_bfloat16=use_bfloat16,
        )
        model = getattr(base_eval, "_binary_input_adapter_model", None)
        if model is None:
            model = getattr(base_eval, "_encoder_output_gaze_model", None)
        if model is None:
            model = getattr(base_eval, "_encoder_lora_model", None)
        if model is None:
            model = getattr(base_eval, "_predictor_lora_model", None)
        if model is None:
            raise RuntimeError("predictor_lora model was not registered before init_opt")
        pred_params = trainable_predictor_lora_params(model)
        if not pred_params:
            raise RuntimeError("predictor LoRA is enabled but no trainable predictor-LoRA params were found")
        first_kwargs = opt_kwargs[0]
        lr_mult = pred_cfg["lr_mult"]
        ref_lr = _opt_ref_lr(first_kwargs)
        warmup_steps = int((first_kwargs.get("warmup") or 0.0) * iterations_per_epoch)
        pred_group = {
            "diagnostic_name": "predictor_lora",
            "params": pred_params,
            "mc_warmup_steps": warmup_steps,
            "mc_start_lr": (first_kwargs.get("start_lr") or 0.0) * lr_mult,
            "mc_ref_lr": (ref_lr or 0.0) * lr_mult,
            "mc_final_lr": (first_kwargs.get("final_lr") or 0.0) * lr_mult,
            "mc_ref_wd": pred_cfg["weight_decay"],
            "mc_final_wd": pred_cfg["weight_decay"],
        }
        optimizers[0].add_param_group(pred_group)
        logger.info(
            "Added predictor-LoRA optimizer group to optimizer[0]: params=%d lr_mult=%.3f wd=%.5f",
            sum(p.numel() for p in pred_params),
            lr_mult,
            pred_cfg["weight_decay"],
        )
        return optimizers, scalers, schedulers, wd_schedulers

    base_eval.init_opt = init_opt_with_predictor_lora

    if baseline_train_loop:
        logger.info("Predictor LoRA baseline path: installing grad-flowing train_one_epoch")
        base_eval.train_one_epoch = lambda **kwargs: train_one_epoch_predictor_lora(base_eval, **kwargs)


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
                "mc_ref_lr": _opt_ref_lr(kwargs),
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
                    "mc_ref_lr": (_opt_ref_lr(kwargs) or 0.0) * gaze_lr_mult,
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
            scalers.append(make_grad_scaler(use_bfloat16))
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
            head_idx = len(param_groups)
            base_params = [p for p in classifier.parameters() if p.requires_grad]
            classifier_param_count += sum(p.numel() for p in base_params)
            warmup_steps = int((kwargs.get("warmup") or 0.0) * iterations_per_epoch)
            param_groups.append(
                {
                    "diagnostic_name": f"classifier_head_{head_idx}",
                    "params": base_params,
                    "mc_warmup_steps": warmup_steps,
                    "mc_start_lr": kwargs.get("start_lr"),
                    "mc_ref_lr": _opt_ref_lr(kwargs),
                    "mc_final_lr": kwargs.get("final_lr"),
                    "mc_ref_wd": kwargs.get("ref_wd"),
                    "mc_final_wd": kwargs.get("final_wd"),
                }
        )
        adapter_warmup_steps = int((first_kwargs.get("warmup") or 0.0) * iterations_per_epoch)
        adapter_ref_lr = _opt_ref_lr(first_kwargs)
        param_groups.append(
            {
                "diagnostic_name": "encoder_output_gaze",
                "params": gaze_adapter_params,
                "mc_warmup_steps": adapter_warmup_steps,
                "mc_start_lr": (first_kwargs.get("start_lr") or 0.0) * lr_mult,
                "mc_ref_lr": (adapter_ref_lr or 0.0) * lr_mult,
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
        scaler = make_grad_scaler(use_bfloat16)
        return [optimizer], [scaler], [scheduler], [wd_scheduler]

    base_eval.init_opt = init_opt
