#!/usr/bin/env python3
"""Video-only streaming anticipation @ +2/+4/+6s — **no MTP communicating heads**.

Same half-split streaming protocol as ``train_stream_mtp.py`` (grow 4→10s,
tick every 2s, prune-before-predictor), but:

  * encode once → optional attention prune
  * run the anticipative **predictor independently** at each horizon (2/4/6s)
  * shared ``AttentiveClassifier`` per horizon (no CommunicatingMLP / cascade)

Use for a clean video streaming baseline vs MTP / concat+CA.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402

from app.hdepic_lora_action_anticipation import train_stream_mtp as base  # noqa: E402
from app.hdepic_lora_action_anticipation.mtp import (  # noqa: E402
    MultiHorizonAnticipativeWrapper,
)

logger = logging.getLogger("stream_video")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class IndependentHorizonClassifier(nn.Module):
    """Shared AttentiveClassifier applied independently at each horizon."""

    def __init__(self, base_classifier: nn.Module, horizons_sec: list[float]):
        super().__init__()
        self.base = base_classifier
        self.horizons_sec = [float(h) for h in horizons_sec]

    def forward(self, tokens_by_horizon: list[torch.Tensor]) -> dict[float, dict[str, torch.Tensor]]:
        if len(tokens_by_horizon) != len(self.horizons_sec):
            raise ValueError(
                f"Expected {len(self.horizons_sec)} token tensors, got {len(tokens_by_horizon)}"
            )
        return {float(h): self.base(tok) for h, tok in zip(self.horizons_sec, tokens_by_horizon)}


class PrunedMultiHorizonStreamModel(nn.Module):
    """Encode → prune → independent anticipative predict at each horizon."""

    def __init__(
        self,
        base_model: nn.Module,
        pruner: base.TokenPruner | None,
        prune_threshold: int,
        horizons_sec: list[float],
    ):
        super().__init__()
        self.base = base_model
        self.pruner = pruner
        self.prune_threshold = int(prune_threshold)
        self.horizons_sec = [float(h) for h in horizons_sec]
        self.embed_dim = getattr(base_model, "embed_dim", None)
        # Reuse MultiHorizon helpers (direct skip / AR rollout) without wrapping
        # the full encode path — we inject pruned tokens below.
        self._mh = MultiHorizonAnticipativeWrapper(base_model, horizons_sec=self.horizons_sec)

    def forward(self, x, anticipation_times=None):
        del anticipation_times
        core = self.base
        x_full = core.encoder(x)
        B, N, D_full = x_full.size()
        embed_dim = core.encoder.embed_dim
        if self.pruner is not None and N > self.prune_threshold:
            x_full, _ = self.pruner.prune(x_full)
            B, N, D_full = x_full.size()
        x_ctx = x_full[:, :, -embed_dim:] if D_full > embed_dim else x_full
        x_base = (
            torch.zeros(B, 0, embed_dim, device=x.device, dtype=x_ctx.dtype)
            if getattr(core, "no_encoder", False)
            else x_ctx
        )
        grid2 = int(core.grid_size**2)
        max_skip = self._mh._max_direct_skip(core, N)
        outs: list[torch.Tensor] = []
        for h in self.horizons_sec:
            steps = int(float(h) * core.frames_per_second / core.tubelet_size)
            skip = N + grid2 * steps
            if skip <= max_skip:
                outs.append(self._mh._predict_direct(core, x_full, x_base, h))
            else:
                outs.append(self._mh._predict_ar_rollout(core, x_full, x_base, h))
        return outs


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    step: int,
    model,
    classifier,
    optimizer,
    scaler,
    best: float,
    horizons,
    verb_map,
    noun_map,
    action_map,
    history,
    phase: str = "train",
    metric_state=None,
):
    ck = {
        "epoch": int(epoch),
        "step": int(step),
        "phase": str(phase),
        "model": model.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best": float(best),
        "horizons": list(horizons),
        "verb_map": verb_map,
        "noun_map": noun_map,
        "action_map": {f"{v},{n}": i for (v, n), i in action_map.items()},
        "history": history,
        "metric_state": metric_state,
        "head_type": "independent",
        "backbone_mode": "multi_predict",
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)


def run_epoch(
    model,
    classifier,
    loader,
    device,
    horizons,
    weights,
    primary_idx,
    verb_map,
    noun_map,
    action_map,
    optimizer=None,
    scaler=None,
    train: bool = True,
    log_every: int = 20,
    start_step: int = 0,
    save_every: int = 0,
    save_fn=None,
    stop_flag=None,
    metric_state=None,
):
    model.train(mode=train)
    classifier.train(mode=train)
    crit = nn.CrossEntropyLoss()
    totals = defaultdict(float)
    counts = defaultdict(int)
    loss_meter = 0.0
    n_steps = 0
    if metric_state:
        for k, v in (metric_state.get("totals") or {}).items():
            totals[k] = float(v)
        for k, v in (metric_state.get("counts") or {}).items():
            counts[k] = int(v)
        loss_meter = float(metric_state.get("loss_meter", 0.0))
        n_steps = int(metric_state.get("n_steps", 0))
    t0 = time.time()
    stopped_early = False
    last_it = start_step - 1

    for local_it, batch in enumerate(loader):
        it = start_step + local_it
        if stop_flag is not None and stop_flag["stop"]:
            stopped_early = True
            break
        last_it = it
        clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
        clips = clips.sub_(base.IMAGENET_MEAN.to(device)).div_(base.IMAGENET_STD.to(device))
        mtp_verbs = batch["mtp_verbs"].to(device, non_blocking=True)
        mtp_nouns = batch["mtp_nouns"].to(device, non_blocking=True)
        mtp_mask = batch["mtp_mask"].to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            tokens_list = model(clips, None)
            if tokens_list is None:
                continue
            outputs = classifier(tokens_list)
            head_loss = clips.new_zeros(())
            for hi, h in enumerate(horizons):
                valid = mtp_mask[:, hi] > 0.5
                if not bool(valid.any()):
                    continue
                v_lab, n_lab, a_lab, keep = base.map_labels(
                    mtp_verbs[valid, hi],
                    mtp_nouns[valid, hi],
                    verb_map,
                    noun_map,
                    action_map,
                    device,
                )
                if not keep:
                    continue
                valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                o = outputs[float(h)]
                step_loss = (
                    crit(o["verb"][valid_pos], v_lab)
                    + crit(o["noun"][valid_pos], n_lab)
                    + crit(o["action"][valid_pos], a_lab)
                )
                head_loss = head_loss + float(weights[hi]) * step_loss
                with torch.no_grad():
                    key = f"action_top5@{h:g}s"
                    acc = base.topk_acc(o["action"][valid_pos].float(), a_lab, k=5) * len(keep)
                    totals[key] += acc
                    counts[key] += len(keep)
                    ctx = float(batch["context_sec"][0])
                    totals[f"{key}|ctx{ctx:g}"] += acc
                    counts[f"{key}|ctx{ctx:g}"] += len(keep)
                    for kind, logits, labs in (
                        ("verb", o["verb"], v_lab),
                        ("noun", o["noun"], n_lab),
                        ("action", o["action"], a_lab),
                    ):
                        for kk in (1, 5):
                            kname = f"{kind}_top{kk}@{h:g}s"
                            totals[kname] += base.topk_acc(logits[valid_pos].float(), labs, k=kk) * len(keep)
                            counts[kname] += len(keep)

            h0 = horizons[primary_idx]
            valid = mtp_mask[:, primary_idx] > 0.5
            if bool(valid.any()):
                v_lab, n_lab, a_lab, keep = base.map_labels(
                    mtp_verbs[valid, primary_idx],
                    mtp_nouns[valid, primary_idx],
                    verb_map,
                    noun_map,
                    action_map,
                    device,
                )
                if keep:
                    valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                    o = outputs[float(h0)]
                    with torch.no_grad():
                        totals["primary_action_top5"] += base.topk_acc(
                            o["action"][valid_pos].float(), a_lab, k=5
                        ) * len(keep)
                        counts["primary_action_top5"] += len(keep)
                        totals["primary_action_top1"] += base.topk_acc(
                            o["action"][valid_pos].float(), a_lab, k=1
                        ) * len(keep)
                        counts["primary_action_top1"] += len(keep)

        if train and optimizer is not None and scaler is not None:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(head_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_meter += float(head_loss.detach().float().item())
        n_steps += 1
        if log_every > 0 and (it + 1) % log_every == 0:
            elapsed = time.time() - t0
            prim = totals["primary_action_top5"] / max(1, counts["primary_action_top5"])
            logger.info(
                "%s step=%d loss=%.4f primary@%gs_top5=%.4f (%.1fs)",
                "train" if train else "val",
                it + 1,
                loss_meter / max(1, n_steps),
                horizons[primary_idx],
                prim,
                elapsed,
            )
        if train and save_every > 0 and save_fn is not None and (it + 1) % save_every == 0:
            save_fn(
                it + 1,
                "train",
                {
                    "totals": dict(totals),
                    "counts": dict(counts),
                    "loss_meter": loss_meter,
                    "n_steps": n_steps,
                },
            )

    metrics = {
        k: (totals[k] / counts[k] if counts[k] > 0 else float("nan"))
        for k in sorted(set(list(totals) + list(counts)))
    }
    metrics["loss"] = loss_meter / max(1, n_steps)
    metrics["seconds"] = time.time() - t0
    metrics["last_step"] = last_it + 1
    metrics["stopped_early"] = stopped_early
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Video streaming 2/4/6s finetune (no MTP)")
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--loss-weights", type=str, default="1.0,0.7,0.5")
    ap.add_argument("--primary-horizon-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument(
        "--init-from-ckpt",
        type=Path,
        default=None,
        help="Load model+classifier (no optimizer). For resume / val-only.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    weights = [float(x) for x in args.loss_weights.split(",")]
    assert len(horizons) == len(weights)
    primary_h = float(args.primary_horizon_sec)
    primary_idx = horizons.index(primary_h) if primary_h in horizons else 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = args.out_dir / "TRAINING_DONE"
    if done_flag.is_file() and not args.val_only:
        logger.info("TRAINING_DONE present (%s); exiting.", done_flag)
        return

    verb_map, noun_map, action_map = base.load_action_maps(args.train_csv)
    logger.info(
        "Video stream (no MTP): horizons=%s vocab v=%d n=%d a=%d",
        horizons,
        len(verb_map),
        len(noun_map),
        len(action_map),
    )

    train_ds = base.StreamMTPDataset(args.train_csv, args.video_root, args.img_size)
    val_ds = base.StreamMTPDataset(args.val_csv, args.video_root, args.img_size)
    train_sampler = base.ContextBucketBatchSampler(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_sampler = base.ContextBucketBatchSampler(val_ds, args.batch_size, shuffle=False, seed=args.seed)
    loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=base.collate_stream,
        pin_memory=False,
        persistent_workers=False,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kwargs)

    core = base.build_model(device, args.max_frames, args.fps, args.img_size, str(args.checkpoint))
    for p in core.encoder.parameters():
        p.requires_grad = False
    base.load_lora_sidecars(
        core,
        str(args.encoder_lora) if args.encoder_lora else None,
        str(args.predictor_lora) if args.predictor_lora else None,
    )
    gp = int(core.grid_size**2)
    pruner = base.TokenPruner(core.encoder, keep_count=args.keep_count, gp=gp)
    model = PrunedMultiHorizonStreamModel(
        core, pruner, prune_threshold=args.keep_count, horizons_sec=horizons
    ).to(device)

    base_clf = AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=int(core.encoder.embed_dim),
        num_heads=16,
        depth=4,
        use_activation_checkpointing=True,
    ).to(device)
    for name, p in base_clf.named_parameters():
        p.requires_grad = name.startswith(("verb_classifier.", "noun_classifier.", "action_classifier."))
        if name.startswith("pooler."):
            p.requires_grad = True
    classifier = IndependentHorizonClassifier(base_clf, horizons).to(device)

    if args.init_from_ckpt is not None and Path(args.init_from_ckpt).is_file():
        init_ck = torch.load(Path(args.init_from_ckpt), map_location="cpu", weights_only=False)
        m_miss, m_unexp = model.load_state_dict(init_ck["model"], strict=False)
        clf_key = "classifier" if "classifier" in init_ck else "mtp_classifier"
        if clf_key in init_ck:
            c_miss, c_unexp = classifier.load_state_dict(init_ck[clf_key], strict=False)
        else:
            c_miss, c_unexp = [], []
        logger.info(
            "Init from %s best=%s model(miss=%d unexp=%d) clf(miss=%d unexp=%d)",
            args.init_from_ckpt,
            init_ck.get("best"),
            len(m_miss),
            len(m_unexp),
            len(c_miss),
            len(c_unexp),
        )
        del init_ck

    params = [p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    logger.info("Trainable params: %d", sum(p.numel() for p in params))

    best = -1.0
    history = []
    start_epoch = 0
    start_step = 0
    resume_phase = "train"
    resume_metric_state = None
    latest = args.out_dir / "latest.pt"
    if latest.is_file():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        classifier.load_state_dict(ck["classifier"], strict=False)
        if ck.get("optimizer") is not None:
            optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scaler") is not None:
            try:
                scaler.load_state_dict(ck["scaler"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not restore GradScaler: %s", exc)
        best = float(ck.get("best", -1.0))
        history = list(ck.get("history") or [])
        if "step" in ck:
            start_epoch = int(ck.get("epoch", 0))
            start_step = int(ck.get("step", 0))
            resume_phase = str(ck.get("phase", "train"))
            resume_metric_state = ck.get("metric_state")
        else:
            start_epoch = int(ck.get("epoch", 0)) + 1
            start_step = 0
        logger.info(
            "Resumed %s epoch=%d step=%d phase=%s best=%.4f",
            latest,
            start_epoch,
            start_step,
            resume_phase,
            best,
        )

    stop_flag = {"stop": False}
    _ckpt_ctx = {"epoch": start_epoch, "step": start_step, "phase": resume_phase}

    def _periodic_save(step: int, phase: str, metric_state=None):
        _ckpt_ctx["step"] = int(step)
        _ckpt_ctx["phase"] = str(phase)
        save_checkpoint(
            latest,
            epoch=_ckpt_ctx["epoch"],
            step=step,
            model=model,
            classifier=classifier,
            optimizer=optimizer,
            scaler=scaler,
            best=best,
            horizons=horizons,
            verb_map=verb_map,
            noun_map=noun_map,
            action_map=action_map,
            history=history,
            phase=phase,
            metric_state=metric_state,
        )
        logger.info("Saved mid-step checkpoint step=%d phase=%s", step, phase)

    def _on_signal(signum, _frame):
        logger.warning("Signal %s — flag stop + emergency save", signum)
        stop_flag["stop"] = True
        try:
            save_checkpoint(
                latest,
                epoch=_ckpt_ctx["epoch"],
                step=_ckpt_ctx["step"],
                model=model,
                classifier=classifier,
                optimizer=optimizer,
                scaler=scaler,
                best=best,
                horizons=horizons,
                verb_map=verb_map,
                noun_map=noun_map,
                action_map=action_map,
                history=history,
                phase=_ckpt_ctx["phase"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Emergency save failed: %s", exc)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    if args.val_only:
        init_path = args.init_from_ckpt or (args.out_dir / "best.pt")
        if Path(init_path).is_file() and args.init_from_ckpt is None:
            ck = torch.load(Path(init_path), map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"], strict=False)
            classifier.load_state_dict(ck["classifier"], strict=False)
            best = float(ck.get("best", best))
            logger.info("Val-only loaded %s best=%.4f", init_path, best)
        val_sampler.set_epoch(0)
        val_metrics = run_epoch(
            model,
            classifier,
            val_loader,
            device,
            horizons,
            weights,
            primary_idx,
            verb_map,
            noun_map,
            action_map,
            train=False,
        )
        out = {
            "best": best,
            "val": val_metrics,
            "horizons": horizons,
            "head_type": "independent",
            "backbone_mode": "multi_predict",
        }
        (args.out_dir / "val_only_metrics.json").write_text(json.dumps(out, indent=2))
        logger.info("Val-only primary_action_top5=%.4f", val_metrics.get("primary_action_top5", float("nan")))
        for h in horizons:
            logger.info("  action_top5@%gs = %.4f", h, val_metrics.get(f"action_top5@{h:g}s", float("nan")))
        return

    for epoch in range(start_epoch, args.epochs):
        _ckpt_ctx["epoch"] = epoch
        train_sampler.set_epoch(epoch)
        skip_train = resume_phase == "val" and epoch == start_epoch
        train_start = start_step if (epoch == start_epoch and resume_phase == "train") else 0
        train_metric_state = resume_metric_state if (epoch == start_epoch and resume_phase == "train") else None
        if epoch == start_epoch:
            start_step = 0
            resume_phase = "train"
            resume_metric_state = None

        if skip_train:
            train_metrics = {"loss": float("nan"), "stopped_early": False, "last_step": 0}
        else:
            train_sampler.set_start_batch(train_start)
            train_metrics = run_epoch(
                model,
                classifier,
                train_loader,
                device,
                horizons,
                weights,
                primary_idx,
                verb_map,
                noun_map,
                action_map,
                optimizer=optimizer,
                scaler=scaler,
                train=True,
                start_step=train_start,
                save_every=args.save_every,
                save_fn=_periodic_save,
                stop_flag=stop_flag,
                metric_state=train_metric_state,
            )
            train_sampler.set_start_batch(0)
            if train_metrics.get("stopped_early"):
                break

        val_sampler.set_epoch(epoch)
        _ckpt_ctx["phase"] = "val"
        val_metrics = run_epoch(
            model,
            classifier,
            val_loader,
            device,
            horizons,
            weights,
            primary_idx,
            verb_map,
            noun_map,
            action_map,
            train=False,
            stop_flag=stop_flag,
        )
        prim = float(val_metrics.get("primary_action_top5", float("nan")))
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
        logger.info(
            "Epoch %d val primary@%gs_top5=%.4f action@2/4/6=%.4f/%.4f/%.4f",
            epoch,
            horizons[primary_idx],
            prim,
            val_metrics.get("action_top5@2s", float("nan")),
            val_metrics.get("action_top5@4s", float("nan")),
            val_metrics.get("action_top5@6s", float("nan")),
        )

        is_best = prim > best
        if is_best:
            best = prim
        save_checkpoint(
            latest,
            epoch=epoch,
            step=0,
            model=model,
            classifier=classifier,
            optimizer=optimizer,
            scaler=scaler,
            best=best,
            horizons=horizons,
            verb_map=verb_map,
            noun_map=noun_map,
            action_map=action_map,
            history=history,
            phase="train",
        )
        if is_best:
            save_checkpoint(
                args.out_dir / "best.pt",
                epoch=epoch,
                step=0,
                model=model,
                classifier=classifier,
                optimizer=None,
                scaler=None,
                best=best,
                horizons=horizons,
                verb_map=verb_map,
                noun_map=noun_map,
                action_map=action_map,
                history=history,
            )
            # Sidecars for warm-start downstream jobs.
            try:
                from app.hdepic_lora_action_anticipation.predictor_lora import (
                    extract_predictor_lora_state_dict,
                )

                pred_sd = extract_predictor_lora_state_dict(core)
                torch.save(pred_sd, args.out_dir / "predictor_lora_from_stream_best.pt")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not dump predictor LoRA sidecar: %s", exc)
            torch.save(classifier.state_dict(), args.out_dir / "classifier_best.pt")
            logger.info("New best=%.4f @ epoch %d", best, epoch)

        if stop_flag["stop"]:
            break

    if not stop_flag["stop"] and start_epoch < args.epochs:
        done_flag.write_text(f"best={best}\n")
        logger.info("TRAINING_DONE best=%.4f", best)


if __name__ == "__main__":
    main()
