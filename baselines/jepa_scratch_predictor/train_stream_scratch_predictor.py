#!/usr/bin/env python3
"""From-scratch JEPA predictor control (vs Exp A causal decoder).

Same frozen ViT-L (+ frozen encoder LoRA) + MTP as Exp A / vanilla stream.
Difference vs vanilla: predictor weights are **re-initialized** and trained
fully (no predictor pretrain, no predictor LoRA).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402
from app.hdepic_lora_action_anticipation.mtp import CommunicatingMLPMTPClassifier  # noqa: E402
from app.hdepic_lora_action_anticipation import train_stream_mtp as stm  # noqa: E402

logger = logging.getLogger("stream_scratch_predictor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def reinit_module_(module: nn.Module) -> None:
    """Re-initialize Linear / LayerNorm / Parameters (drop pretrained predictor)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    for name, p in module.named_parameters(recurse=True):
        if p.dim() >= 2 and "weight" not in name.split(".")[-1]:
            # mask tokens / free params
            if "mask_token" in name or name.endswith("tokens"):
                nn.init.trunc_normal_(p, std=0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--loss-weights", type=str, default="1.0,0.7,0.5")
    ap.add_argument("--primary-horizon-sec", type=float, default=2.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    weights = [float(x) for x in args.loss_weights.split(",")]
    primary_h = float(args.primary_horizon_sec)
    primary_idx = horizons.index(primary_h) if primary_h in horizons else 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = args.out_dir / "TRAINING_DONE"
    if done_flag.is_file() and not args.val_only:
        logger.info("TRAINING_DONE present; exiting")
        return

    verb_map, noun_map, action_map = stm.load_action_maps(args.train_csv)
    logger.info("vocab v=%d n=%d a=%d", len(verb_map), len(noun_map), len(action_map))

    train_ds = stm.StreamMTPDataset(args.train_csv, args.video_root, args.img_size)
    val_ds = stm.StreamMTPDataset(args.val_csv, args.video_root, args.img_size)
    train_sampler = stm.ContextBucketBatchSampler(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_sampler = stm.ContextBucketBatchSampler(val_ds, args.batch_size, shuffle=False, seed=args.seed)
    loader_kwargs = dict(num_workers=args.num_workers, collate_fn=stm.collate_stream, pin_memory=False)
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kwargs)

    base = stm.build_model(device, args.max_frames, args.fps, args.img_size, str(args.checkpoint))
    # Drop pretrained predictor → random init, then train fully.
    n_before = sum(p.numel() for p in base.predictor.parameters())
    reinit_module_(base.predictor)
    # Also reinit ParameterList mask tokens if present
    if getattr(base.predictor, "mask_tokens", None) is not None:
        for p in base.predictor.mask_tokens:
            nn.init.trunc_normal_(p, std=0.02)
    logger.info("Re-initialized predictor (%.2fM params) from scratch", n_before / 1e6)

    for p in base.encoder.parameters():
        p.requires_grad = False
    for p in base.predictor.parameters():
        p.requires_grad = True
    # Frozen encoder LoRA only (same backbone as Exp A / vanilla); NO predictor LoRA.
    stm.load_lora_sidecars(base, str(args.encoder_lora) if args.encoder_lora else None, None)

    gp = int(base.grid_size**2)
    pruner = stm.TokenPruner(base.encoder, keep_count=args.keep_count, gp=gp)
    model = stm.PrunedAnticipativeModel(base, pruner, prune_threshold=args.keep_count).to(device)

    classifier = AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=int(base.encoder.embed_dim),
        num_heads=16,
        depth=4,
        use_activation_checkpointing=True,
    ).to(device)
    for name, p in classifier.named_parameters():
        p.requires_grad = name.startswith(("verb_classifier.", "noun_classifier.", "action_classifier."))
        if name.startswith("pooler."):
            p.requires_grad = True
    mtp_clf = CommunicatingMLPMTPClassifier(
        classifier, horizons_sec=horizons, comm_layers=2, comm_heads=4
    ).to(device)

    params = [p for p in list(model.parameters()) + list(mtp_clf.parameters()) if p.requires_grad]
    n_pred = sum(p.numel() for p in base.predictor.parameters() if p.requires_grad)
    n_train = sum(p.numel() for p in params)
    logger.info(
        "Trainable: predictor=%.2fM + heads/mtp=%.2fM → total %.2fM",
        n_pred / 1e6,
        (n_train - n_pred) / 1e6,
        n_train / 1e6,
    )
    (args.out_dir / "param_report.json").write_text(
        json.dumps(
            {
                "predictor_m": n_pred / 1e6,
                "trainable_m": n_train / 1e6,
                "note": "From-scratch JEPA predictor (no pred pretrain/LoRA); frozen ViT-L + enc LoRA",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

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
        mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
        if ck.get("optimizer") is not None:
            optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scaler") is not None:
            try:
                scaler.load_state_dict(ck["scaler"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("GradScaler restore failed: %s", exc)
        best = float(ck.get("best", -1.0))
        history = list(ck.get("history") or [])
        if "step" in ck:
            start_epoch = int(ck.get("epoch", 0))
            start_step = int(ck.get("step", 0))
            resume_phase = str(ck.get("phase", "train"))
            resume_metric_state = ck.get("metric_state")
        else:
            start_epoch = int(ck.get("epoch", 0)) + 1
        logger.info(
            "Resumed epoch=%d step=%d phase=%s best=%.4f",
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
        stm.save_checkpoint(
            latest,
            epoch=_ckpt_ctx["epoch"],
            step=step,
            model=model,
            mtp_clf=mtp_clf,
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

    def _on_signal(signum, _frame):
        logger.warning("Caught signal %s — saving latest", signum)
        stop_flag["stop"] = True
        try:
            _periodic_save(step=max(0, int(_ckpt_ctx["step"])), phase=str(_ckpt_ctx["phase"]))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Emergency save failed: %s", exc)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)

    if args.val_only:
        metrics = stm.run_epoch(
            model, mtp_clf, val_loader, device, horizons, weights, primary_idx,
            verb_map, noun_map, action_map, train=False, anticipation_sec=args.anticipation_sec,
            stop_flag=stop_flag,
        )
        pub = {k: v for k, v in metrics.items() if not str(k).startswith("_")}
        (args.out_dir / "val_only_metrics.json").write_text(json.dumps(pub, indent=2))
        logger.info("VAL_ONLY %s", pub)
        return

    for epoch in range(start_epoch, args.epochs):
        _ckpt_ctx["epoch"] = epoch
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        epoch_start_step = start_step if (epoch == start_epoch and resume_phase == "train") else 0
        val_start_step = start_step if (epoch == start_epoch and resume_phase == "val") else 0
        skip_train = bool(epoch == start_epoch and resume_phase == "val")

        if skip_train:
            tr = {"loss": float("nan"), "stopped_early": False, "last_step": 0}
            logger.info("Skipping train epoch=%d (resume val step=%d)", epoch, val_start_step)
        else:
            _ckpt_ctx["phase"] = "train"
            train_sampler.set_start_batch(epoch_start_step)
            tr = stm.run_epoch(
                model, mtp_clf, train_loader, device, horizons, weights, primary_idx,
                verb_map, noun_map, action_map, optimizer=optimizer, scaler=scaler, train=True,
                anticipation_sec=args.anticipation_sec, start_step=epoch_start_step,
                save_every=int(args.save_every), save_fn=_periodic_save, stop_flag=stop_flag,
                metric_state=(resume_metric_state if epoch_start_step > 0 else None),
            )
            train_sampler.set_start_batch(0)
            if tr.get("stopped_early"):
                _periodic_save(step=int(tr["last_step"]), phase="train", metric_state=tr.get("_metric_state"))
                return

        if not skip_train:
            _periodic_save(step=0, phase="val", metric_state=None)
        _ckpt_ctx["phase"] = "val"
        val_sampler.set_start_batch(val_start_step)
        va = stm.run_epoch(
            model, mtp_clf, val_loader, device, horizons, weights, primary_idx,
            verb_map, noun_map, action_map, train=False, anticipation_sec=args.anticipation_sec,
            start_step=val_start_step, save_every=int(args.save_every), save_fn=_periodic_save,
            stop_flag=stop_flag,
            metric_state=(resume_metric_state if (skip_train and val_start_step > 0) else None),
        )
        val_sampler.set_start_batch(0)
        if va.get("stopped_early"):
            _periodic_save(step=int(va["last_step"]), phase="val", metric_state=va.get("_metric_state"))
            return

        primary = float(va.get("primary_action_top5", 0.0))
        logger.info(
            "epoch %d train_loss=%.4f val_primary_top5=%.2f%% %s",
            epoch, tr.get("loss"), 100.0 * primary,
            {k: round(100.0 * va[k], 2) for k in va if k.startswith("action_top5")},
        )
        history.append({
            "epoch": epoch,
            "train": {k: v for k, v in tr.items() if not str(k).startswith("_")},
            "val": {k: v for k, v in va.items() if not str(k).startswith("_")},
        })
        stm.save_checkpoint(
            latest, epoch=epoch + 1, step=0, model=model, mtp_clf=mtp_clf,
            optimizer=optimizer, scaler=scaler, best=best, horizons=horizons,
            verb_map=verb_map, noun_map=noun_map, action_map=action_map,
            history=history, phase="train", metric_state=None,
        )
        if primary > best:
            best = primary
            stm.save_checkpoint(
                args.out_dir / "best.pt", epoch=epoch + 1, step=0, model=model, mtp_clf=mtp_clf,
                optimizer=optimizer, scaler=scaler, best=best, horizons=horizons,
                verb_map=verb_map, noun_map=noun_map, action_map=action_map,
                history=history, phase="train", metric_state=None,
            )
            logger.info("New best primary_action_top5=%.4f", best)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        start_step = 0
        resume_phase = "train"
        resume_metric_state = None

    done_flag.write_text(f"ok best={best}\n", encoding="utf-8")
    logger.info("TRAINING_DONE best=%.4f", best)


if __name__ == "__main__":
    main()
