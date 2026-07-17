#!/usr/bin/env python
"""Measure action entropy at intermediate predictor depths using trained weights."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
from pathlib import Path

import torch
import yaml

from app.hdepic_lora_action_anticipation.binary_input_adapter import BinaryMapInputAdapter
from app.hdepic_lora_action_anticipation.encoder_lora import (
    inject_encoder_lora,
    load_encoder_lora_checkpoint,
)
from app.hdepic_lora_action_anticipation.encoder_early_exit import (
    anticipation_tokens_by_encoder_depth,
)
from app.hdepic_lora_action_anticipation.eval import (
    _make_lora_init_classifier,
    _patch_hdepic_temporal_sampling,
)
from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate, labels_from_udata, patch_metadata_dataloader
from app.hdepic_lora_action_anticipation.pose_map_builder import GazePoseInputMapBuilder
from app.hdepic_lora_action_anticipation.predictor_early_exit import (
    anticipation_tokens_by_predictor_depth,
)
from app.hdepic_lora_action_anticipation.predictor_lora import (
    inject_predictor_lora,
    load_predictor_lora_checkpoint,
)
from evals.action_anticipation_frozen.dataloader import filter_annotations, init_data
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger("predictor_early_exit_entropy")


def _parse_depths(raw: str) -> tuple[int, ...]:
    depths = tuple(sorted({int(x) for x in raw.replace(",", " ").replace("|", " ").split()}))
    if not depths:
        raise ValueError("at least one exit depth is required")
    return depths


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _validate_split(cfg: dict, expected_split_dir: Path | None, split_label: str) -> dict:
    data = cfg["experiment"]["data"]
    train_csv = Path(data["dataset_train"]).expanduser().resolve()
    val_csv = Path(data["dataset_val"]).expanduser().resolve()
    actual_dirs = {train_csv.parent, val_csv.parent}
    if len(actual_dirs) != 1:
        raise RuntimeError(
            "train/val split provenance mismatch: "
            f"train={train_csv} val={val_csv} must share one split directory"
        )
    actual = train_csv.parent
    if expected_split_dir is not None and actual != expected_split_dir.expanduser().resolve():
        raise RuntimeError(
            "checkpoint/split provenance mismatch: "
            f"expected train+val under {expected_split_dir.expanduser().resolve()}, "
            f"got train={train_csv} val={val_csv}"
        )
    identity = split_label.strip() or actual.name
    split = {
        "identity": identity,
        "directory": str(actual),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "raw_csv_rows": {
            "train": _csv_row_count(train_csv),
            "val": _csv_row_count(val_csv),
        },
    }
    logger.info(
        "Validated split identity=%s train_rows=%d val_rows=%d directory=%s",
        split["identity"],
        split["raw_csv_rows"]["train"],
        split["raw_csv_rows"]["val"],
        split["directory"],
    )
    return split


def _load_model(cfg: dict, checkpoint_dir: Path, device: torch.device):
    data = cfg["experiment"]["data"]
    model_cfg = cfg["model_kwargs"]
    module = importlib.import_module(model_cfg["module_name"])
    model = module.init_module(
        frames_per_clip=data["frames_per_clip"],
        frames_per_second=data["frames_per_second"],
        resolution=data["resolution"],
        checkpoint=model_cfg["checkpoint"],
        model_kwargs=model_cfg["pretrain_kwargs"],
        wrapper_kwargs=model_cfg["wrapper_kwargs"],
    ).to(device)
    lora = cfg["experiment"]["lora"]
    enc = lora["encoder_lora"]
    inject_encoder_lora(
        model,
        rank=int(enc["rank"]),
        alpha=float(enc["alpha"]),
        dropout=float(enc["dropout"]),
        last_n_blocks=int(enc["last_n_blocks"]),
        target_suffixes=enc["target_suffixes"],
    )
    load_encoder_lora_checkpoint(model, checkpoint_dir / "encoder_lora_best.pt", strict=True)
    pred = lora["predictor_lora"]
    inject_predictor_lora(
        model,
        rank=int(pred["rank"]),
        alpha=float(pred["alpha"]),
        dropout=float(pred["dropout"]),
        last_n_blocks=int(pred["last_n_blocks"]),
        target_suffixes=pred["target_suffixes"],
    )
    load_predictor_lora_checkpoint(model, checkpoint_dir / "predictor_lora_best.pt", strict=True)
    model.encoder.use_activation_checkpointing = False
    model.predictor.use_activation_checkpointing = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _load_classifier(cfg: dict, annotations: dict, checkpoint_dir: Path, embed_dim: int, device):
    lora = dict(cfg["experiment"]["lora"])
    # The complete Joint checkpoint supersedes the Stage-1 probe. Avoid a
    # dependency on the original Stage-1 directory during analysis.
    lora["pretrained_probe"] = ""
    factory = _make_lora_init_classifier(lora)
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
    checkpoint = robust_checkpoint_loader(str(checkpoint_dir / "best.pt"), map_location=torch.device("cpu"))
    if len(classifiers) != len(checkpoint["classifiers"]):
        raise RuntimeError("classifier count differs between config and checkpoint")
    for classifier, state in zip(classifiers, checkpoint["classifiers"]):
        clean = {key.removeprefix("module."): value for key, value in state.items()}
        classifier.load_state_dict(clean, strict=True)
        classifier.eval()
        for parameter in classifier.parameters():
            parameter.requires_grad = False
    return classifiers


def _load_gaze(cfg: dict, checkpoint_dir: Path, device):
    gaze = dict(cfg["experiment"]["lora"].get("gaze", {}))
    mode = str(gaze.get("mode", "none")).lower()
    if mode == "none":
        return None, None
    if mode != "binary_input_adapter_gaze_pose_matrix":
        raise ValueError(f"unsupported gaze mode for this diagnostic: {mode}")
    data = cfg["experiment"]["data"]
    enc = cfg["model_kwargs"]["pretrain_kwargs"]["encoder"]
    gaze.setdefault("crop_size", data["resolution"])
    gaze.setdefault("frames_per_clip", data["frames_per_clip"])
    gaze.setdefault("patch_size", enc["patch_size"])
    gaze.setdefault("tubelet_size", enc["tubelet_size"])
    adapter_cfg = gaze["input_adapter"]
    adapter = BinaryMapInputAdapter(
        hidden_dim=int(adapter_cfg.get("hidden_dim", 8)),
        scale=float(adapter_cfg.get("scale", 1.0)),
        temporal_kernel=int(adapter_cfg.get("temporal_kernel", 3)),
        binary_center=float(adapter_cfg.get("binary_center", 0.0)),
        residual_clamp=float(adapter_cfg.get("residual_clamp", 1.0)),
        in_channels=int(adapter_cfg.get("in_channels", 5)),
    ).to(device)
    checkpoint = robust_checkpoint_loader(
        str(checkpoint_dir / "binary_input_adapter_best.pt"), map_location=torch.device("cpu")
    )
    state = checkpoint.get("input_adapter", checkpoint)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    gate = GazeTokenGate({**gaze, "mode": "token_gate"})
    return adapter, GazePoseInputMapBuilder(gaze, gate=gate)


def _build_val_loader(cfg: dict, batch_size: int, num_workers: int):
    data = cfg["experiment"]["data"]
    _patch_hdepic_temporal_sampling(data, cfg["experiment"]["lora"])
    annotations = filter_annotations(
        data["dataset"],
        data["base_path"],
        data["dataset_train"],
        data["dataset_val"],
        file_format=data.get("file_format", 1),
    )
    patch_metadata_dataloader()
    _, loader, _ = init_data(
        dataset=data["dataset"],
        training=False,
        base_path=data["base_path"],
        annotations_path=annotations["val"],
        batch_size=batch_size,
        frames_per_clip=data["frames_per_clip"],
        fps=data["frames_per_second"],
        anticipation_time_sec=data["anticipation_time_sec"],
        anticipation_point=data["val_anticipation_point"],
        crop_size=data["resolution"],
        world_size=1,
        rank=0,
        num_workers=num_workers,
        pin_mem=True,
        persistent_workers=num_workers > 0,
    )
    return annotations, loader


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1) / math.log(logits.shape[-1])


def _metadata_value(metadata, index: int, key: str, default=""):
    if not isinstance(metadata, list) or index >= len(metadata):
        return default
    return metadata[index].get(key, default)


@torch.inference_mode()
def run(args) -> None:
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    split = _validate_split(cfg, args.expected_split_dir, args.split_label)
    depths = _parse_depths(args.depths)
    device = torch.device("cuda:0")
    annotations, loader = _build_val_loader(cfg, args.batch_size, args.num_workers)
    model = _load_model(cfg, args.checkpoint_dir, device)
    classifiers = _load_classifier(cfg, annotations, args.checkpoint_dir, model.embed_dim, device)
    if len(classifiers) != 1:
        raise RuntimeError("this diagnostic expects the transferred single-classifier Joint checkpoints")
    classifier = classifiers[0]
    adapter, map_builder = _load_gaze(cfg, args.checkpoint_dir, device)
    if args.exit_module == "predictor":
        full_depth = len(model.predictor.predictor_blocks)
        token_fn = anticipation_tokens_by_predictor_depth
    else:
        full_depth = len(model.encoder.blocks)
        token_fn = anticipation_tokens_by_encoder_depth
    if max(depths) != full_depth:
        raise ValueError(
            f"depth list must include the full {args.exit_module} depth={full_depth} "
            "for the equivalence gate"
        )

    rows = []
    logits_by_depth = {depth: [] for depth in depths}
    labels_saved = []
    processed = 0
    equivalence = None
    use_bfloat16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False))

    for batch_index, udata in enumerate(loader):
        clips = udata[0].to(device, non_blocking=True)
        metadata = udata[3]
        anticipation_times = udata[4].to(device, non_blocking=True)
        labels = labels_from_udata(
            udata,
            device,
            True,
            annotations["verbs"],
            annotations["nouns"],
            annotations["actions"],
        )
        if args.max_samples and processed + clips.shape[0] > args.max_samples:
            keep = args.max_samples - processed
            clips = clips[:keep]
            anticipation_times = anticipation_times[:keep]
            metadata = metadata[:keep]
            labels = {name: value[:keep] for name, value in labels.items()}
        if clips.shape[0] == 0:
            break

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            model_input = clips
            if adapter is not None:
                aux_map = map_builder.build(clips, metadata)
                model_input = adapter(clips, aux_map)
            tokens = token_fn(model, model_input, anticipation_times, depths)
            outputs = {depth: classifier(tokens[depth]) for depth in depths}

            if equivalence is None:
                reference_tokens = model(model_input, anticipation_times)
                reference_logits = classifier(reference_tokens)["action"]
                token_diff = (reference_tokens.float() - tokens[full_depth].float()).abs().max().item()
                logit_diff = (reference_logits.float() - outputs[full_depth]["action"].float()).abs().max().item()
                top1_equal = bool(
                    torch.equal(reference_logits.argmax(dim=-1), outputs[full_depth]["action"].argmax(dim=-1))
                )
                equivalence = {
                    "max_abs_token_diff": token_diff,
                    "max_abs_action_logit_diff": logit_diff,
                    "top1_equal": top1_equal,
                }
                if token_diff > args.equivalence_atol or logit_diff > args.equivalence_atol or not top1_equal:
                    raise RuntimeError(f"full-depth equivalence gate failed: {equivalence}")

        final_top1 = outputs[full_depth]["action"].argmax(dim=-1)
        final_top5 = outputs[full_depth]["action"].topk(5, dim=-1).indices
        batch_labels = labels["action"]
        labels_saved.append(batch_labels.detach().cpu())
        for depth in depths:
            logits = outputs[depth]["action"].float()
            entropy = _entropy(logits)
            top1 = logits.argmax(dim=-1)
            top5 = logits.topk(5, dim=-1).indices
            logits_by_depth[depth].append(logits.detach().to(dtype=torch.float16).cpu())
            for i in range(logits.shape[0]):
                top5_set = set(int(x) for x in top5[i].detach().cpu().tolist())
                final_top5_set = set(int(x) for x in final_top5[i].detach().cpu().tolist())
                rows.append(
                    {
                        "sample_index": processed + i,
                        "video_id": _metadata_value(metadata, i, "video_id"),
                        "start_frame": _metadata_value(metadata, i, "start_frame"),
                        "stop_frame": _metadata_value(metadata, i, "stop_frame"),
                        "depth": depth,
                        "action_label": int(batch_labels[i]),
                        "action_entropy_normalized": float(entropy[i]),
                        "action_top1": int(top1[i]),
                        "action_top1_hit": int(top1[i] == batch_labels[i]),
                        "action_top5_hit": int(int(batch_labels[i]) in top5_set),
                        "full_depth_top1_agree": int(top1[i] == final_top1[i]),
                        "full_depth_top5_jaccard": len(top5_set & final_top5_set) / len(top5_set | final_top5_set),
                    }
                )
        processed += clips.shape[0]
        logger.info("batch=%d processed=%d/%s", batch_index, processed, args.max_samples or "all")
        if args.max_samples and processed >= args.max_samples:
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "per_sample_depth_entropy.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "samples": processed,
        "exit_module": args.exit_module,
        "depths": list(depths),
        "equivalence": equivalence,
        "split": split,
        "per_depth": {},
    }
    for depth in depths:
        depth_rows = [row for row in rows if row["depth"] == depth]
        entropies = torch.tensor([row["action_entropy_normalized"] for row in depth_rows])
        summary["per_depth"][str(depth)] = {
            "mean_entropy_normalized": float(entropies.mean()),
            "median_entropy_normalized": float(entropies.median()),
            "action_top1": 100.0 * sum(row["action_top1_hit"] for row in depth_rows) / len(depth_rows),
            "action_top5": 100.0 * sum(row["action_top5_hit"] for row in depth_rows) / len(depth_rows),
            "full_depth_top1_agreement": 100.0 * sum(row["full_depth_top1_agree"] for row in depth_rows) / len(depth_rows),
            "mean_full_depth_top5_jaccard": sum(row["full_depth_top5_jaccard"] for row in depth_rows) / len(depth_rows),
        }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    torch.save(
        {
            "depths": depths,
            "action_labels": torch.cat(labels_saved),
            "action_logits": {depth: torch.cat(parts) for depth, parts in logits_by_depth.items()},
        },
        args.out_dir / "action_logits_fp16.pt",
    )
    logger.info("wrote %s", args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-split-dir",
        type=Path,
        default=None,
        help="optional checkpoint-bound split directory; mismatch aborts before model loading",
    )
    parser.add_argument("--split-label", default="", help="split identity written to summary.json")
    parser.add_argument("--exit-module", choices=("predictor", "encoder"), default="predictor")
    parser.add_argument("--depths", default="3,6,9,12")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--equivalence-atol", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())
