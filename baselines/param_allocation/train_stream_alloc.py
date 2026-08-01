#!/usr/bin/env python3
"""Encoder-heavy vs decoder-heavy param allocation on the same ViT-L backbone.

Hypothesis (why Qwen underperforms despite more total params): on streaming
closed-set anticipation, **where** capacity sits matters — vision/encoder side
beats language/decoder side.

Matched ~25–30M trainable budget (excluding shared MTP heads):
  - encoder_heavy: freeze most of pretrained ViT-L; train **last N blocks** + norm;
    no predictor / no causal decoder (pool encoder tokens → MTP).
  - decoder_heavy: freeze entire ViT-L; train Exp-A-style causal decoder (~29M);
    no encoder LoRA (fair vs encoder_heavy).

Same stream half-split protocol / MTP heads / lr schedule.
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
sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "jepa_causal_decoder"))

from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402
from app.hdepic_lora_action_anticipation.mtp import CommunicatingMLPMTPClassifier  # noqa: E402
from app.hdepic_lora_action_anticipation import train_stream_mtp as stm  # noqa: E402
from model import CausalDecoderStreamModel, CausalFutureDecoder, count_trainable  # noqa: E402

logger = logging.getLogger("stream_param_alloc")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_backbone(device, max_frames, fps, img_size, checkpoint, no_predictor: bool):
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
    wrapper_kwargs = {
        "no_predictor": bool(no_predictor),
        "num_output_frames": 2,
        "num_steps": 1,
    }
    return stm.init_anticipative_module(
        frames_per_clip=max_frames,
        frames_per_second=fps,
        resolution=img_size,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    ).to(device)


class EncoderOnlyStreamModel(nn.Module):
    """Trainable encoder (selected blocks) → tokens for MTP (no future module)."""

    def __init__(self, base, pruner=None, prune_threshold: int = 4096):
        super().__init__()
        self.base = base
        self.pruner = pruner
        self.prune_threshold = int(prune_threshold)
        self.embed_dim = int(base.encoder.embed_dim)

    def forward(self, x, anticipation_times):
        del anticipation_times  # unused — allocation probe has no +Δt module
        x_full = self.base.encoder(x)
        B, N, D_full = x_full.size()
        embed_dim = self.base.encoder.embed_dim
        if self.pruner is not None and N > self.prune_threshold:
            x_full, _ = self.pruner.prune(x_full)
            B, N, D_full = x_full.size()
        if D_full > embed_dim:
            return x_full[:, :, -embed_dim:]
        return x_full


def unfreeze_last_encoder_blocks(encoder: nn.Module, n_blocks: int) -> int:
    for p in encoder.parameters():
        p.requires_grad = False
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise RuntimeError("encoder.blocks not found")
    n = min(int(n_blocks), len(blocks))
    for blk in blocks[-n:]:
        for p in blk.parameters():
            p.requires_grad = True
    if hasattr(encoder, "norm") and encoder.norm is not None:
        for p in encoder.norm.parameters():
            p.requires_grad = True
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("encoder_heavy", "decoder_heavy"), required=True)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
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
    ap.add_argument("--encoder-last-blocks", type=int, default=2, help="~12.6M each → 2≈25M")
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
    primary_idx = horizons.index(float(args.primary_horizon_sec))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = args.out_dir / "TRAINING_DONE"
    if done_flag.is_file() and not args.val_only:
        logger.info("TRAINING_DONE present; exiting")
        return

    verb_map, noun_map, action_map = stm.load_action_maps(args.train_csv)
    train_ds = stm.StreamMTPDataset(args.train_csv, args.video_root, args.img_size)
    val_ds = stm.StreamMTPDataset(args.val_csv, args.video_root, args.img_size)
    train_sampler = stm.ContextBucketBatchSampler(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_sampler = stm.ContextBucketBatchSampler(val_ds, args.batch_size, shuffle=False, seed=args.seed)
    loader_kwargs = dict(num_workers=args.num_workers, collate_fn=stm.collate_stream, pin_memory=False)
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kwargs)

    if args.mode == "encoder_heavy":
        base = build_backbone(
            device, args.max_frames, args.fps, args.img_size, str(args.checkpoint), no_predictor=True
        )
        if hasattr(base, "predictor"):
            base.predictor = None
        n_unfrozen = unfreeze_last_encoder_blocks(base.encoder, args.encoder_last_blocks)
        enc_train = sum(p.numel() for p in base.encoder.parameters() if p.requires_grad)
        gp = int(base.grid_size**2)
        pruner = stm.TokenPruner(base.encoder, keep_count=args.keep_count, gp=gp)
        model = EncoderOnlyStreamModel(base, pruner, prune_threshold=args.keep_count).to(device)
        alloc_report = {
            "mode": "encoder_heavy",
            "unfrozen_last_blocks": n_unfrozen,
            "encoder_trainable_m": enc_train / 1e6,
            "note": "Last-N ViT-L blocks trainable; no future decoder/predictor",
        }
        logger.info(
            "encoder_heavy: last %d blocks trainable = %.2fM",
            n_unfrozen,
            enc_train / 1e6,
        )
    else:
        base = build_backbone(
            device, args.max_frames, args.fps, args.img_size, str(args.checkpoint), no_predictor=True
        )
        for p in base.encoder.parameters():
            p.requires_grad = False
        if hasattr(base, "predictor") and base.predictor is not None:
            base.predictor = None
        gp = int(base.grid_size**2)
        n_future = gp * max(1, int(base.num_output_frames // base.tubelet_size))
        decoder = CausalFutureDecoder(
            embed_dim=int(base.encoder.embed_dim),
            predictor_embed_dim=int(args.decoder_dim),
            depth=int(args.decoder_depth),
            num_heads=int(args.decoder_heads),
            num_future_tokens=n_future,
        )
        pruner = stm.TokenPruner(base.encoder, keep_count=args.keep_count, gp=gp)
        # Freeze encoder path in CausalDecoderStreamModel uses no_grad — good.
        model = CausalDecoderStreamModel(base, decoder, pruner, prune_threshold=args.keep_count).to(device)
        alloc_report = {
            "mode": "decoder_heavy",
            "decoder_m": decoder.param_count() / 1e6,
            "decoder_dim": args.decoder_dim,
            "decoder_depth": args.decoder_depth,
            "note": "Frozen ViT-L + causal decoder; no encoder LoRA (fair vs encoder_heavy)",
        }
        logger.info("decoder_heavy: causal decoder = %.2fM", decoder.param_count() / 1e6)

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
    alloc_report["trainable_total_m"] = n_train / 1e6
    (args.out_dir / "param_report.json").write_text(json.dumps(alloc_report, indent=2), encoding="utf-8")
    logger.info("param_report %s", alloc_report)

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
        logger.info("Resumed epoch=%d step=%d phase=%s best=%.4f", start_epoch, start_step, resume_phase, best)

    stop_flag = {"stop": False}
    _ckpt_ctx = {"epoch": start_epoch, "step": start_step, "phase": resume_phase}

    def _periodic_save(step: int, phase: str, metric_state=None):
        _ckpt_ctx.update(step=int(step), phase=str(phase))
        stm.save_checkpoint(
            latest, epoch=_ckpt_ctx["epoch"], step=step, model=model, mtp_clf=mtp_clf,
            optimizer=optimizer, scaler=scaler, best=best, horizons=horizons,
            verb_map=verb_map, noun_map=noun_map, action_map=action_map,
            history=history, phase=phase, metric_state=metric_state,
        )

    def _on_signal(signum, _frame):
        logger.warning("Caught signal %s", signum)
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
            "[%s] epoch %d val_primary_top5=%.2f%% %s",
            args.mode, epoch, 100.0 * primary,
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
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        start_step = 0
        resume_phase = "train"
        resume_metric_state = None

    done_flag.write_text(f"ok mode={args.mode} best={best}\n", encoding="utf-8")
    logger.info("TRAINING_DONE mode=%s best=%.4f", args.mode, best)


if __name__ == "__main__":
    main()
