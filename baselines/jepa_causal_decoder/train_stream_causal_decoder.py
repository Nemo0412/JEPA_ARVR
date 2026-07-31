#!/usr/bin/env python3
"""Experiment A: causal decoder on frozen V-JEPA latents (stream MTP).

Same protocol / encoder / MTP heads as ``train_stream_mtp.py``; only the
anticipative JEPA predictor is replaced by a param-matched causal
Transformer decoder (d=384, depth=12, heads=12 ≈29M).
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

# baselines/jepa_causal_decoder → repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402
from app.hdepic_lora_action_anticipation.mtp import CommunicatingMLPMTPClassifier  # noqa: E402
from app.hdepic_lora_action_anticipation import train_stream_mtp as stm  # noqa: E402

from model import CausalDecoderStreamModel, CausalFutureDecoder, count_trainable  # noqa: E402

logger = logging.getLogger("stream_causal_decoder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_encoder_only(device, max_frames: int, fps: int, img_size: int, checkpoint: str):
    """Same ViT-L as stream MTP, but wrapper skips predictor forward."""
    model_kwargs = {
        "use_v2_1": False,
        "encoder": {
            "model_name": "vit_large",
            "checkpoint_key": "target_encoder",
            "tubelet_size": 2,
            "patch_size": 16,
            "uniform_power": True,
            "use_rope": True,
        },
        "predictor": {
            "model_name": "vit_predictor",
            "checkpoint_key": "predictor",
            "num_frames": 64,
            "depth": 12,
            "num_heads": 12,
            "predictor_embed_dim": 384,
            "num_mask_tokens": 10,
            "uniform_power": True,
            "use_mask_tokens": True,
            "use_sdpa": True,
            "use_silu": False,
            "wide_silu": False,
            "use_rope": True,
        },
    }
    wrapper_kwargs = {"no_predictor": True, "num_output_frames": 2, "num_steps": 1}
    model = stm.init_anticipative_module(
        frames_per_clip=max_frames,
        frames_per_second=fps,
        resolution=img_size,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    ).to(device)
    return model


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
    ap.add_argument("--decoder-depth", type=int, default=12)
    ap.add_argument("--decoder-dim", type=int, default=384)
    ap.add_argument("--decoder-heads", type=int, default=12)
    ap.add_argument("--val-only", action="store_true")
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

    verb_map, noun_map, action_map = stm.load_action_maps(args.train_csv)
    logger.info("vocab verbs=%d nouns=%d actions=%d", len(verb_map), len(noun_map), len(action_map))

    train_ds = stm.StreamMTPDataset(args.train_csv, args.video_root, args.img_size)
    val_ds = stm.StreamMTPDataset(args.val_csv, args.video_root, args.img_size)
    train_sampler = stm.ContextBucketBatchSampler(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_sampler = stm.ContextBucketBatchSampler(val_ds, args.batch_size, shuffle=False, seed=args.seed)
    loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=stm.collate_stream,
        pin_memory=False,
        persistent_workers=False,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kwargs)

    base = build_encoder_only(device, args.max_frames, args.fps, args.img_size, str(args.checkpoint))
    for p in base.encoder.parameters():
        p.requires_grad = False
    # Encoder LoRA warm-start (frozen) — same backbone as JEPA vanilla stream.
    stm.load_lora_sidecars(
        base,
        str(args.encoder_lora) if args.encoder_lora else None,
        None,
    )
    # Drop unused predictor weights to free GPU RAM.
    if hasattr(base, "predictor") and base.predictor is not None:
        n_pred = sum(p.numel() for p in base.predictor.parameters())
        base.predictor = None
        logger.info("Dropped JEPA predictor weights (%.1fM) — using causal decoder instead", n_pred / 1e6)

    gp = int(base.grid_size**2)
    n_future = gp * max(1, int(base.num_output_frames // base.tubelet_size))
    decoder = CausalFutureDecoder(
        embed_dim=int(base.encoder.embed_dim),
        predictor_embed_dim=int(args.decoder_dim),
        depth=int(args.decoder_depth),
        num_heads=int(args.decoder_heads),
        num_future_tokens=n_future,
    )
    logger.info(
        "Causal decoder: dim=%d depth=%d heads=%d n_future=%d params=%.2fM",
        args.decoder_dim,
        args.decoder_depth,
        args.decoder_heads,
        n_future,
        decoder.param_count() / 1e6,
    )

    pruner = stm.TokenPruner(base.encoder, keep_count=args.keep_count, gp=gp)
    model = CausalDecoderStreamModel(base, decoder, pruner, prune_threshold=args.keep_count).to(device)

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
    n_train = sum(p.numel() for p in params)
    logger.info(
        "Trainable: decoder=%.2fM + heads/mtp=%.2fM → total %.2fM (encoder frozen)",
        count_trainable(decoder) / 1e6,
        (n_train - count_trainable(decoder)) / 1e6,
        n_train / 1e6,
    )
    (args.out_dir / "param_report.json").write_text(
        json.dumps(
            {
                "decoder_m": decoder.param_count() / 1e6,
                "trainable_m": n_train / 1e6,
                "decoder_dim": args.decoder_dim,
                "decoder_depth": args.decoder_depth,
                "decoder_heads": args.decoder_heads,
                "n_future": n_future,
                "note": "Swap JEPA anticipative predictor → causal cross-attn decoder; same frozen ViT-L + MTP",
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
                logger.warning("Could not restore GradScaler state: %s", exc)
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
            "Resumed from %s epoch=%d step=%d phase=%s best=%.4f",
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
        logger.warning("Caught signal %s — saving latest and stopping after current step", signum)
        stop_flag["stop"] = True
        try:
            _periodic_save(step=max(0, int(_ckpt_ctx["step"])), phase=str(_ckpt_ctx["phase"]))
            logger.info("Emergency checkpoint written to %s", latest)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed emergency checkpoint: %s", exc)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)

    if args.val_only:
        metrics = stm.run_epoch(
            model,
            mtp_clf,
            val_loader,
            device,
            horizons,
            weights,
            primary_idx,
            verb_map,
            noun_map,
            action_map,
            train=False,
            anticipation_sec=args.anticipation_sec,
            stop_flag=stop_flag,
        )
        metrics_pub = {k: v for k, v in metrics.items() if not str(k).startswith("_")}
        logger.info("VAL_ONLY metrics: %s", json.dumps(metrics_pub, indent=2, default=str))
        (args.out_dir / "val_only_metrics.json").write_text(json.dumps(metrics_pub, indent=2), encoding="utf-8")
        return

    for epoch in range(start_epoch, args.epochs):
        _ckpt_ctx["epoch"] = epoch
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        epoch_start_step = start_step if (epoch == start_epoch and resume_phase == "train") else 0
        val_start_step = start_step if (epoch == start_epoch and resume_phase == "val") else 0
        skip_train = bool(epoch == start_epoch and resume_phase == "val")
        if skip_train:
            logger.info("Skipping train for epoch %d (resume_phase=val step=%d)", epoch, val_start_step)
            tr = {"loss": float("nan"), "stopped_early": False, "last_step": 0}
        else:
            _ckpt_ctx["phase"] = "train"
            train_sampler.set_start_batch(epoch_start_step)
            tr = stm.run_epoch(
                model,
                mtp_clf,
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
                anticipation_sec=args.anticipation_sec,
                start_step=epoch_start_step,
                save_every=int(args.save_every),
                save_fn=_periodic_save,
                stop_flag=stop_flag,
                metric_state=(resume_metric_state if epoch_start_step > 0 else None),
            )
            train_sampler.set_start_batch(0)
            if tr.get("stopped_early"):
                _periodic_save(step=int(tr["last_step"]), phase="train", metric_state=tr.get("_metric_state"))
                logger.info("Stopped early during train; exiting for resubmit")
                return

        _ckpt_ctx["phase"] = "val"
        val_sampler.set_start_batch(val_start_step)
        va = stm.run_epoch(
            model,
            mtp_clf,
            val_loader,
            device,
            horizons,
            weights,
            primary_idx,
            verb_map,
            noun_map,
            action_map,
            train=False,
            anticipation_sec=args.anticipation_sec,
            start_step=val_start_step,
            save_every=int(args.save_every),
            save_fn=_periodic_save,
            stop_flag=stop_flag,
            metric_state=(resume_metric_state if (skip_train and val_start_step > 0) else None),
        )
        val_sampler.set_start_batch(0)
        if va.get("stopped_early"):
            _periodic_save(step=int(va["last_step"]), phase="val", metric_state=va.get("_metric_state"))
            logger.info("Stopped early during val; exiting for resubmit")
            return

        primary = float(va.get("primary_action_top5", 0.0))
        logger.info(
            "epoch %d train_loss=%.4f val_primary_top5=%.2f%% %s",
            epoch,
            tr.get("loss"),
            100.0 * primary,
            {k: round(100.0 * va[k], 2) for k in va if k.startswith("action_top5")},
        )
        tr_pub = {k: v for k, v in tr.items() if not str(k).startswith("_")}
        va_pub = {k: v for k, v in va.items() if not str(k).startswith("_")}
        history.append({"epoch": epoch, "train": tr_pub, "val": va_pub})
        stm.save_checkpoint(
            latest,
            epoch=epoch + 1,
            step=0,
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
            phase="train",
            metric_state=None,
        )
        if primary > best:
            best = primary
            stm.save_checkpoint(
                args.out_dir / "best.pt",
                epoch=epoch + 1,
                step=0,
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
                phase="train",
                metric_state=None,
            )
            logger.info("New best primary_action_top5=%.4f", best)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        start_step = 0
        resume_phase = "train"
        resume_metric_state = None

    done_flag.write_text("ok\n", encoding="utf-8")
    logger.info("TRAINING_DONE best_primary_action_top5=%.4f", best)


if __name__ == "__main__":
    main()
