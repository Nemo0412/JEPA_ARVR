"""Official split-1 full-prefix ScalAnt streaming training adapter.

The pinned upstream model is unchanged. Video order may shuffle between
epochs, but each video is processed chronologically from its first 4s tick.
Only official training actions supervise either past or future heads. The
released EK100 mixup cannot mix variable-length video prefixes, so this
adaptation accumulates 128 supervised ticks per AdamW update instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .egtea_cache import ANNOTATIONS, MAIN_ROOT, ROOT
from .legacy_autograd_fix import install_clone_per_layer_fix
from .official_stream_data import build_official_split1_ticks
from .stream_data import StreamingTSNReader, StreamTick
from .stream_eval import ActionTickScorer
from .stream_model import FullPrefixStreamingScalAnt, masked_stream_loss


PROTOCOL = "egtea-action-anticipation/official-split1-stream4s-ticks-h2h6-recall5-v1"
UPSTREAM = ROOT / "refer_repo/scalable-anticipation"
SOURCE_CONFIG = UPSTREAM / "configs/ek100/default.yaml"
ACCUMULATION = 128
EPOCHS = 50


def protocol_for_split(split: int) -> str:
    if split == 1:
        return PROTOCOL
    if split not in (2, 3):
        raise ValueError("Expected official EGTEA split 1, 2 or 3")
    return f"egtea-action-anticipation/official-split{split}-stream4s-ticks-h2h6-recall5-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(split: int = 1) -> dict:
    return {
        f"training{split}_sha256": file_sha256(ANNOTATIONS / f"training{split}.csv"),
        f"validation{split}_sha256": file_sha256(ANNOTATIONS / f"validation{split}.csv"),
        "actions_sha256": file_sha256(ANNOTATIONS / "actions.csv"),
        "scalant_config_sha256": file_sha256(SOURCE_CONFIG),
    }


def make_model_and_optimizer(horizon: int):
    if horizon not in (2, 6):
        raise ValueError("Independent horizon must be +2s or +6s")
    sys.path.insert(0, str(UPSTREAM))
    from scalant.config import load_config
    from scalant.models import QueryPredictor
    from scalant.utils import WarmUpCosineAnnealingLR

    install_clone_per_layer_fix()
    cfg = load_config(str(SOURCE_CONFIG))
    cfg.DATA.TAU_O = 4.0
    cfg.DATA.TAU_A = float(horizon)
    cfg.DATA.LONG_MEMORY_LENGTH = 0.0
    cfg.MODEL.INPUT_DIM = 1024
    cfg.MODEL.IGNORE_INDEX = -1
    cfg.MODEL.N_QUERIES = 1
    cfg.TRAIN.USE_MIXUP = False
    cfg.TRAIN.EPOCHS = EPOCHS
    model = QueryPredictor(cfg, {"action": 106, "verb": 19, "noun": 53}, None).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    scheduler = WarmUpCosineAnnealingLR(optimizer, T_max=EPOCHS,
                                        warmup_epochs=5, eta_min=1e-7)
    return model, optimizer, scheduler


def group_by_video(ticks: list[StreamTick]) -> dict[str, list[StreamTick]]:
    by_video = defaultdict(list)
    for tick in ticks:
        by_video[tick.video].append(tick)
    for video, rows in by_video.items():
        if [t.tick_frame for t in rows] != sorted(t.tick_frame for t in rows):
            raise ValueError(f"Nonchronological ticks in {video}")
    return dict(by_video)


def video_order(videos, epoch: int) -> list[str]:
    ordered = sorted(videos)
    generator = torch.Generator().manual_seed(42 + epoch)
    permutation = torch.randperm(len(ordered), generator=generator).tolist()
    return [ordered[i] for i in permutation]


def _optimizer_step(model, optimizer, accumulated: int) -> None:
    if accumulated <= 0:
        raise ValueError("No supervised ticks in optimizer group")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(accumulated)
            if not torch.isfinite(parameter.grad).all():
                raise ValueError("Nonfinite accumulated gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(model, optimizer, reader: StreamingTSNReader,
                by_video: dict[str, list[StreamTick]], horizon: int,
                epoch: int, max_ticks: int | None = None,
                loss_trace: list[float] | None = None,
                before_step=None) -> dict:
    model.train()
    stream = FullPrefixStreamingScalAnt(model)
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    processed = supervised = steps = future_labels = past_positions = 0
    loss_sum = 0.0
    for video in video_order(by_video, epoch):
        for tick in by_video[video]:
            if max_ticks is not None and processed >= max_ticks:
                break
            chunk = reader.read(tick).unsqueeze(0).cuda(non_blocking=True)
            has_label = (tick.target_actions[0 if horizon == 2 else 1] >= 0
                         or any(label >= 0 for label in tick.past_actions))
            with torch.set_grad_enabled(has_label):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = stream.process(video, chunk)
                    loss = masked_stream_loss(output, tick, horizon) if has_label else None
                if has_label:
                    if loss is None or loss.value is None or not torch.isfinite(loss.value):
                        raise ValueError(f"Missing/nonfinite loss at {video}/{tick.tick_frame}")
                    loss.value.backward()
                    accumulated += 1
                    supervised += 1
                    future_labels += loss.valid_future
                    past_positions += loss.valid_past
                    loss_sum += float(loss.value.detach())
                    if loss_trace is not None:
                        loss_trace.append(float(loss.value.detach()))
                    if accumulated == ACCUMULATION:
                        if before_step is not None:
                            before_step(model, optimizer, accumulated, steps)
                        _optimizer_step(model, optimizer, accumulated)
                        accumulated = 0
                        steps += 1
            processed += 1
        if max_ticks is not None and processed >= max_ticks:
            break
    if accumulated:
        if before_step is not None:
            before_step(model, optimizer, accumulated, steps)
        _optimizer_step(model, optimizer, accumulated)
        steps += 1
    return {"epoch": epoch, "processed_ticks": processed,
            "supervised_ticks": supervised, "optimizer_steps": steps,
            "future_labels": future_labels, "known_past_positions": past_positions,
            "mean_loss_per_supervised_tick": loss_sum / supervised if supervised else None}


@torch.inference_mode()
def validate(model, reader: StreamingTSNReader,
             by_video: dict[str, list[StreamTick]], horizon: int,
             split: int = 1) -> dict:
    model.eval()
    scorer = ActionTickScorer(horizon)
    auxiliary = {name: {"total": Counter(), "hit": Counter(), "classes": classes}
                 for name, classes in (("verb_direct", 19), ("noun_direct", 53))} if split != 1 else {}
    stream = FullPrefixStreamingScalAnt(model)
    processed = 0
    for video in sorted(by_video):
        for tick in by_video[video]:
            chunk = reader.read(tick).unsqueeze(0).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = stream.process(video, chunk)
            scorer.update(tick, output.future_actions[0, -1].float())
            if auxiliary:
                index = 0 if horizon == 2 else 1
                if tick.target_actions[index] >= 0:
                    for name, logits, label in (("verb_direct", output.future_verbs[0, -1].float(),
                                                 tick.target_verbs[index]),
                                                ("noun_direct", output.future_nouns[0, -1].float(),
                                                 tick.target_nouns[index])):
                        stat = auxiliary[name]
                        if (not 0 <= label < stat["classes"]
                                or logits.shape != (stat["classes"],)
                                or not torch.isfinite(logits).all()):
                            raise ValueError(f"Invalid {name} target/logits at {video}/{tick.tick_frame}")
                        stat["total"][label] += 1
                        stat["hit"][label] += int(label in logits.topk(5).indices.tolist())
            processed += 1
    report = scorer.report()
    if auxiliary:
        report["metric_scope"] = f"split{split}_streaming_tick"
        for name, stat in auxiliary.items():
            counts, hits = stat["total"], stat["hit"]
            n = sum(counts.values())
            report[name] = {
                "scored_tick_count": n,
                "classes_with_scored_ticks": len(counts),
                "top5_micro": sum(hits.values()) / n if n else None,
                "mean_recall_at5_present_classes":
                    sum(hits[c] / count for c, count in counts.items()) / len(counts) if counts else None,
            }
    report["processed_ticks"] = processed
    return report


def rng_state() -> dict:
    name, state, position, has_gauss, gauss = np.random.get_state()
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "python": random.getstate(),
            "numpy": (name, state.tolist(), position, has_gauss, gauss)}


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    name, values, position, has_gauss, gauss = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32),
                         position, has_gauss, gauss))


def checkpoint_payload(model, optimizer, scheduler, horizon: int,
                       completed_epoch: int, best_recall: float,
                       best_epoch: int, split: int = 1) -> dict:
    payload = {"protocol": protocol_for_split(split), "horizon_seconds": horizon,
            "completed_epoch": completed_epoch,
            "best_recall": best_recall, "best_epoch": best_epoch,
            "source_identity": source_identity(split),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": rng_state()}
    if split != 1:
        payload["split"] = split
    return payload


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, optimizer, scheduler,
                    horizon: int, split: int = 1) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["protocol"] != protocol_for_split(split)
            or payload["horizon_seconds"] != horizon
            or (split != 1 and payload.get("split") != split)):
        raise ValueError("Checkpoint protocol or horizon mismatch")
    if payload["source_identity"] != source_identity(split):
        raise ValueError("Checkpoint source identities changed")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    return payload


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _assert_population(ticks: list[StreamTick], population: str,
                       expected_ticks: int | None = None,
                       expected_pairs: tuple[int, int] | None = None) -> None:
    if expected_ticks is None:
        expected_ticks = 23943 if population == "train" else 16227
    if expected_pairs is None:
        expected_pairs = (19403, 19340) if population == "train" else (4678, 4655)
    scored = tuple(sum(t.target_actions[h] >= 0 for t in ticks) for h in (0, 1))
    if len(ticks) != expected_ticks or scored != expected_pairs:
        raise ValueError(f"Official {population} stream coverage changed: "
                         f"{len(ticks)} ticks / {scored} pairs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--horizon", type=int, choices=(2, 6), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs-this-job", type=int, default=1)
    parser.add_argument("--expected-start-epoch", type=int)
    args = parser.parse_args()
    if args.epochs_this_job < 1:
        raise ValueError("Each segment must complete at least one full epoch")
    split = args.split
    train_ticks = build_official_split1_ticks("train", split)
    val_ticks = build_official_split1_ticks("val", split)
    audit = json.loads((MAIN_ROOT / f"outputs/b19_scalant/implementation/official_split{split}_stream_audit.json").read_text())
    if split == 1:
        if audit["kind"] != "official_split1_stream_source_audit_not_training":
            raise ValueError("Missing recognized official split-1 source audit")
        _assert_population(train_ticks, "train")
        _assert_population(val_ticks, "val")
        for population in ("train", "val"):
            if (audit["population"][population].get("missing_feature_queries", 0)
                    or audit["population"][population].get("bad_feature_payloads", 0)):
                raise ValueError(f"Official {population} source feature gap")
    else:
        if (audit["kind"] != "official_stream_split_source_audit_not_training"
                or audit["split"] != split):
            raise ValueError(f"Missing recognized official split-{split} source audit")
        identity = source_identity(split)
        for population in ("train", "val"):
            record = audit["population"][population]
            name = "training" if population == "train" else "validation"
            if (identity[f"{name}{split}_sha256"] != audit["annotation_sha256"][population]
                    or identity["actions_sha256"] != audit["annotation_sha256"]["actions"]
                    or record["missing_queries"] or record["malformed_queries"]):
                raise ValueError(f"Official split-{split} {population} source audit mismatch")
            _assert_population(train_ticks if population == "train" else val_ticks,
                               population, record["processed_ticks"],
                               (record["scored_h2"], record["scored_h6"]))
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    model, optimizer, scheduler = make_model_and_optimizer(args.horizon)
    latest = args.output_dir / "latest.pt"
    best = args.output_dir / "best.pt"
    start_epoch = 0
    best_recall = -1.0
    best_epoch = -1
    if latest.exists():
        payload = load_checkpoint(latest, model, optimizer, scheduler, args.horizon, split)
        start_epoch = payload["completed_epoch"] + 1
        best_recall = payload["best_recall"]
        best_epoch = payload["best_epoch"]
        if best_epoch >= 0 and not best.exists():
            raise ValueError("Resume checkpoint names missing selected best model")
    if start_epoch >= EPOCHS:
        raise ValueError("All 50 epochs already completed")
    if (args.expected_start_epoch is not None
            and start_epoch != args.expected_start_epoch):
        raise ValueError(f"Expected epoch {args.expected_start_epoch}, checkpoint starts {start_epoch}")
    end_epoch = min(EPOCHS, start_epoch + args.epochs_this_job)
    train_by_video = group_by_video(train_ticks)
    val_by_video = group_by_video(val_ticks)
    reader = StreamingTSNReader()
    try:
        for epoch in range(start_epoch, end_epoch):
            train = train_epoch(model, optimizer, reader, train_by_video,
                                args.horizon, epoch)
            if train["processed_ticks"] != len(train_ticks):
                raise ValueError("Training epoch did not process every tick")
            scheduler.step()
            validation = validate(model, reader, val_by_video, args.horizon, split)
            if validation["processed_ticks"] != len(val_ticks):
                raise ValueError("Validation did not process every tick")
            if split == 1:
                expected_scored = 4678 if args.horizon == 2 else 4655
            else:
                expected_scored = audit["population"]["val"][f"scored_h{args.horizon}"]
            if validation["scored_tick_count"] != expected_scored:
                raise ValueError("Validation scored denominator changed")
            recall = validation["action_mean_recall_at5_present_classes"]
            if recall is None or not 0 <= recall <= 1:
                raise ValueError("Invalid selection recall")
            improved = recall > best_recall
            if improved:
                best_recall, best_epoch = recall, epoch
            record = {"protocol": protocol_for_split(split), "horizon_seconds": args.horizon,
                      "epoch": epoch, "training": train, "validation": validation,
                      "selected_best_epoch": best_epoch,
                      "selected_best_recall": best_recall,
                      "lr_after_epoch": optimizer.param_groups[0]["lr"],
                      "new_best": improved}
            if split != 1:
                record["split"] = split
            payload = checkpoint_payload(model, optimizer, scheduler,
                                         args.horizon, epoch,
                                         best_recall, best_epoch, split)
            payload["epoch_record"] = record
            if improved:
                save_checkpoint(best, payload)
            save_checkpoint(latest, payload)
            _atomic_json(args.output_dir / f"epoch_{epoch:02d}.json", record)
            print(json.dumps({"split": split, "epoch": epoch, "horizon": args.horizon,
                              "train_ticks": train["processed_ticks"],
                              "val_scored": validation["scored_tick_count"],
                              "action_recall5": recall,
                              "action_top5": validation["action_top5_micro"],
                              "verb_recall5": (validation.get("verb_direct") or {}).get("mean_recall_at5_present_classes"),
                              "noun_recall5": (validation.get("noun_direct") or {}).get("mean_recall_at5_present_classes"),
                              "best_epoch": best_epoch}), flush=True)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
