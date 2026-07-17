#!/usr/bin/env python
"""Train only an encoder-prefix output-space projector for B14."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from app.hdepic_lora_action_anticipation.encoder_early_exit import (
    EncoderExitProjector,
    anticipation_tokens_from_encoded,
    encoder_prefix_outputs,
)
from app.hdepic_lora_action_anticipation.eval import _patch_hdepic_temporal_sampling
from app.hdepic_lora_action_anticipation.gaze import (
    labels_from_udata,
    patch_metadata_dataloader,
)
from evals.action_anticipation_frozen.dataloader import filter_annotations, init_data
from scripts.analyze_predictor_early_exit_entropy import (
    _load_classifier,
    _load_model,
    _validate_split,
)

logger = logging.getLogger("encoder_exit_projector")


def _load_run_config(path: Path) -> tuple[dict, dict]:
    with path.open("r", encoding="utf-8") as handle:
        run_cfg = yaml.safe_load(handle)
    source_path = Path(run_cfg["source_config"]).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        source_cfg = yaml.safe_load(handle)
    return run_cfg, source_cfg


def _build_loaders(source_cfg: dict, batch_size: int, num_workers: int, val_num_workers: int):
    data = source_cfg["experiment"]["data"]
    _patch_hdepic_temporal_sampling(data, source_cfg["experiment"]["lora"])
    annotations = filter_annotations(
        data["dataset"],
        data["base_path"],
        data["dataset_train"],
        data["dataset_val"],
        file_format=data.get("file_format", 1),
    )
    patch_metadata_dataloader()
    common = dict(
        dataset=data["dataset"],
        base_path=data["base_path"],
        frames_per_clip=data["frames_per_clip"],
        fps=data["frames_per_second"],
        crop_size=data["resolution"],
        world_size=1,
        rank=0,
        pin_mem=True,
    )
    _, train_loader, _ = init_data(
        **common,
        training=True,
        annotations_path=annotations["train"],
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        anticipation_time_sec=data["train_anticipation_time_sec"],
        anticipation_point=data["train_anticipation_point"],
        random_resize_scale=tuple(data.get("random_resize_scale", (0.08, 1.0))),
        reprob=float(data.get("reprob", 0.25)),
        auto_augment=bool(data.get("auto_augment", True)),
        motion_shift=bool(data.get("motion_shift", False)),
    )
    _, val_loader, _ = init_data(
        **common,
        training=False,
        annotations_path=annotations["val"],
        batch_size=batch_size,
        num_workers=val_num_workers,
        persistent_workers=val_num_workers > 0,
        anticipation_time_sec=data["anticipation_time_sec"],
        anticipation_point=data["val_anticipation_point"],
    )
    return annotations, train_loader, val_loader


def _labels(udata, device, annotations):
    return labels_from_udata(
        udata,
        device,
        True,
        annotations["verbs"],
        annotations["nouns"],
        annotations["actions"],
    )


def _task_loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = [F.cross_entropy(outputs[name].float(), labels[name]) for name in ("action", "verb", "noun")]
    return torch.stack(losses).mean()


def _repr_loss(projected: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projected_f = projected.float()
    teacher_f = teacher.float()
    mse = F.mse_loss(projected_f, teacher_f)
    cosine = 1.0 - F.cosine_similarity(projected_f, teacher_f, dim=-1).mean()
    return mse + cosine, mse, cosine


def _atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _checkpoint(projector, optimizer, epoch, global_step, best_metric, run_cfg):
    return {
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "run_config": run_cfg,
    }


def _topk_hits(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    return int(logits.topk(min(k, logits.shape[-1]), dim=-1).indices.eq(labels[:, None]).any(dim=-1).sum())


@torch.inference_mode()
def validate(model, classifier, projector, loader, annotations, device, depth, use_bfloat16, max_samples):
    projector.eval()
    totals = {
        "samples": 0,
        "student_top1": 0,
        "student_top5": 0,
        "teacher_top1": 0,
        "teacher_top5": 0,
        "top1_agree": 0,
        "top5_jaccard_sum": 0.0,
        "repr_mse_sum": 0.0,
        "repr_cosine_sum": 0.0,
    }
    for udata in loader:
        clips = udata[0].to(device, non_blocking=True)
        anticipation_times = udata[4].to(device, non_blocking=True)
        labels = _labels(udata, device, annotations)
        if max_samples and totals["samples"] + clips.shape[0] > max_samples:
            keep = max_samples - totals["samples"]
            clips = clips[:keep]
            anticipation_times = anticipation_times[:keep]
            labels = {name: value[:keep] for name, value in labels.items()}
        if clips.shape[0] == 0:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            encoded = encoder_prefix_outputs(model.encoder, clips, (depth, len(model.encoder.blocks)))
            projected = projector(encoded[depth])
            student_tokens = anticipation_tokens_from_encoded(model, projected, anticipation_times)
            teacher_tokens = anticipation_tokens_from_encoded(
                model, encoded[len(model.encoder.blocks)], anticipation_times
            )
            student = classifier(student_tokens)["action"].float()
            teacher = classifier(teacher_tokens)["action"].float()
        batch = clips.shape[0]
        action_labels = labels["action"]
        student_top1 = student.argmax(dim=-1)
        teacher_top1 = teacher.argmax(dim=-1)
        student_top5 = student.topk(5, dim=-1).indices
        teacher_top5 = teacher.topk(5, dim=-1).indices
        repr_mse = F.mse_loss(projected.float(), encoded[len(model.encoder.blocks)].float())
        repr_cos = F.cosine_similarity(
            projected.float(), encoded[len(model.encoder.blocks)].float(), dim=-1
        ).mean()
        totals["samples"] += batch
        totals["student_top1"] += int(student_top1.eq(action_labels).sum())
        totals["student_top5"] += _topk_hits(student, action_labels, 5)
        totals["teacher_top1"] += int(teacher_top1.eq(action_labels).sum())
        totals["teacher_top5"] += _topk_hits(teacher, action_labels, 5)
        totals["top1_agree"] += int(student_top1.eq(teacher_top1).sum())
        for s5, t5 in zip(student_top5, teacher_top5):
            sset = set(s5.cpu().tolist())
            tset = set(t5.cpu().tolist())
            totals["top5_jaccard_sum"] += len(sset & tset) / len(sset | tset)
        totals["repr_mse_sum"] += float(repr_mse) * batch
        totals["repr_cosine_sum"] += float(repr_cos) * batch
        if max_samples and totals["samples"] >= max_samples:
            break
    n = totals["samples"]
    if n == 0:
        raise RuntimeError("validation loader produced no samples")
    return {
        "samples": n,
        "student_action_top1": 100.0 * totals["student_top1"] / n,
        "student_action_top5": 100.0 * totals["student_top5"] / n,
        "teacher_action_top1": 100.0 * totals["teacher_top1"] / n,
        "teacher_action_top5": 100.0 * totals["teacher_top5"] / n,
        "teacher_top1_agreement": 100.0 * totals["top1_agree"] / n,
        "teacher_top5_jaccard": totals["top5_jaccard_sum"] / n,
        "repr_mse": totals["repr_mse_sum"] / n,
        "repr_cosine": totals["repr_cosine_sum"] / n,
    }


def train(run_config_path: Path) -> None:
    run_cfg, source_cfg = _load_run_config(run_config_path)
    device = torch.device("cuda:0")
    checkpoint_dir = Path(run_cfg["checkpoint_dir"])
    out_dir = Path(run_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    split = _validate_split(
        source_cfg,
        Path(run_cfg["expected_split_dir"]),
        run_cfg["split_label"],
    )
    opt_cfg = run_cfg["optimization"]
    depth = int(run_cfg["projector"]["encoder_depth"])
    annotations, train_loader, val_loader = _build_loaders(
        source_cfg,
        int(opt_cfg["batch_size"]),
        int(opt_cfg["num_workers"]),
        int(opt_cfg["val_num_workers"]),
    )
    model = _load_model(source_cfg, checkpoint_dir, device)
    model.predictor.use_activation_checkpointing = bool(opt_cfg.get("predictor_activation_checkpointing", True))
    classifiers = _load_classifier(source_cfg, annotations, checkpoint_dir, model.embed_dim, device)
    if len(classifiers) != 1:
        raise RuntimeError("projector training expects exactly one transferred classifier")
    classifier = classifiers[0]
    projector = EncoderExitProjector(int(model.encoder.embed_dim)).to(device)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    use_bfloat16 = bool(opt_cfg.get("use_bfloat16", True))
    lambda_repr = float(opt_cfg["lambda_repr"])
    lambda_task = float(opt_cfg["lambda_task"])
    max_steps = int(opt_cfg.get("max_train_steps", 0))
    max_val_samples = int(opt_cfg.get("max_val_samples", 0))
    checkpoint_every = int(opt_cfg.get("checkpoint_every_steps", 100))
    global_step = 0
    start_epoch = 0
    best_metric = -math.inf
    resume_path = Path(run_cfg.get("resume_checkpoint", "")) if run_cfg.get("resume_checkpoint") else None
    if resume_path and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu")
        projector.load_state_dict(state["projector"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best_metric = float(state.get("best_metric", -math.inf))
        logger.info("Resumed projector from %s at epoch=%d step=%d", resume_path, start_epoch, global_step)

    trainable = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in classifier.parameters())
    logger.info(
        "run_tag=%s split=%s depth=%d trainable_projector=%d frozen_original=%d batch=%d max_steps=%d",
        run_cfg["run_tag"], split["identity"], depth, trainable, frozen,
        int(opt_cfg["batch_size"]), max_steps,
    )
    metrics_path = out_dir / "metrics.jsonl"
    start_wall = time.monotonic()
    last_step_end = start_wall
    stop = False
    for epoch in range(start_epoch, int(opt_cfg["epochs"])):
        projector.train()
        for batch_index, udata in enumerate(train_loader):
            step_start = time.monotonic()
            data_time = step_start - start_wall if global_step == 0 else step_start - last_step_end
            clips = udata[0].to(device, non_blocking=True)
            anticipation_times = udata[4].to(device, non_blocking=True)
            labels = _labels(udata, device, annotations)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                encoded = encoder_prefix_outputs(model.encoder, clips, (depth, len(model.encoder.blocks)))
            h_early = encoded[depth].detach()
            h_teacher = encoded[len(model.encoder.blocks)].detach()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                projected = projector(h_early)
                tokens = anticipation_tokens_from_encoded(model, projected, anticipation_times)
                outputs = classifier(tokens)
                task_loss = _task_loss(outputs, labels)
                repr_loss, repr_mse, repr_cosine = _repr_loss(projected, h_teacher)
                loss = lambda_task * task_loss + lambda_repr * repr_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), float(opt_cfg["grad_clip"]))
            optimizer.step()
            global_step += 1
            last_step_end = time.monotonic()
            step_time = last_step_end - step_start
            if global_step == 1 or global_step % int(opt_cfg.get("log_every_steps", 10)) == 0:
                row = {
                    "kind": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "batch": batch_index,
                    "loss": float(loss.detach()),
                    "task_loss": float(task_loss.detach()),
                    "repr_loss": float(repr_loss.detach()),
                    "repr_mse": float(repr_mse.detach()),
                    "repr_cosine_loss": float(repr_cosine.detach()),
                    "grad_norm": float(grad_norm),
                    "data_time_sec": data_time,
                    "step_time_sec": step_time,
                    "elapsed_sec": last_step_end - start_wall,
                }
                logger.info("train %s", json.dumps(row, sort_keys=True))
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                _atomic_save(
                    _checkpoint(projector, optimizer, epoch, global_step, best_metric, run_cfg),
                    out_dir / "latest.pt",
                )
            if max_steps and global_step >= max_steps:
                stop = True
                break

        val = validate(
            model, classifier, projector, val_loader, annotations, device, depth,
            use_bfloat16, max_val_samples,
        )
        val.update({"kind": "val", "epoch": epoch, "step": global_step})
        logger.info("val %s", json.dumps(val, sort_keys=True))
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(val, sort_keys=True) + "\n")
        current = float(val["student_action_top5"])
        if current > best_metric:
            best_metric = current
            _atomic_save(
                _checkpoint(projector, optimizer, epoch + 1, global_step, best_metric, run_cfg),
                out_dir / "best.pt",
            )
        _atomic_save(
            _checkpoint(projector, optimizer, epoch + 1, global_step, best_metric, run_cfg),
            out_dir / "latest.pt",
        )
        with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_tag": run_cfg["run_tag"],
                    "split": split,
                    "trainable_projector_parameters": trainable,
                    "global_step": global_step,
                    "best_student_action_top5": best_metric,
                    "latest_validation": val,
                    "elapsed_sec": time.monotonic() - start_wall,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        if stop:
            break
    logger.info("Training complete: step=%d best_top5=%.4f out=%s", global_step, best_metric, out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(parse_args().run_config)
