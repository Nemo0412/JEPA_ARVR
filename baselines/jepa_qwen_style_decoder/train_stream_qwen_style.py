#!/usr/bin/env python3
"""V-JEPA vision + Qwen-style causal decoder (KV-cache AR) on stream MTP.

Same frozen ViT-L / protocol / MTP as JEPA vanilla & Exp A; future module is a
**decoder-only** causal LM over vision prefix, with teacher-forcing train and
**KV-cache autoregressive** val (Qwen train/infer split).
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402
from app.hdepic_lora_action_anticipation.mtp import CommunicatingMLPMTPClassifier  # noqa: E402
from app.hdepic_lora_action_anticipation import train_stream_mtp as stm  # noqa: E402

from model import QwenStyleARDecoder, QwenStyleStreamModel  # noqa: E402

logger = logging.getLogger("stream_qwen_style_dec")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_encoder_only(device, max_frames, fps, img_size, checkpoint):
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
    return stm.init_anticipative_module(
        frames_per_clip=max_frames,
        frames_per_second=fps,
        resolution=img_size,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        wrapper_kwargs={"no_predictor": True, "num_output_frames": 2, "num_steps": 1},
    ).to(device)


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
    ap.add_argument("--num-future-tokens", type=int, default=16)
    ap.add_argument("--aux-ar-weight", type=float, default=0.1)
    ap.add_argument("--train-decode-mode", type=str, default="teacher", choices=("teacher", "ar_kv"))
    ap.add_argument("--val-decode-mode", type=str, default="ar_kv", choices=("teacher", "ar_kv"))
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

    base = build_encoder_only(device, args.max_frames, args.fps, args.img_size, str(args.checkpoint))
    for p in base.encoder.parameters():
        p.requires_grad = False
    stm.load_lora_sidecars(base, str(args.encoder_lora) if args.encoder_lora else None, None)
    if hasattr(base, "predictor"):
        base.predictor = None

    decoder = QwenStyleARDecoder(
        embed_dim=int(base.encoder.embed_dim),
        d_model=int(args.decoder_dim),
        depth=int(args.decoder_depth),
        num_heads=int(args.decoder_heads),
        num_future_tokens=int(args.num_future_tokens),
        max_prefix=int(args.keep_count),
    )
    gp = int(base.grid_size**2)
    pruner = stm.TokenPruner(base.encoder, keep_count=args.keep_count, gp=gp)
    model = QwenStyleStreamModel(
        base, decoder, pruner, prune_threshold=args.keep_count, decode_mode=args.train_decode_mode
    ).to(device)

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
    report = {
        "decoder_m": decoder.param_count() / 1e6,
        "trainable_m": n_train / 1e6,
        "num_future_tokens": args.num_future_tokens,
        "train_decode_mode": args.train_decode_mode,
        "val_decode_mode": args.val_decode_mode,
        "note": "V-JEPA ViT-L + Qwen-style decoder-only AR (KV cache on val); same MTP/protocol",
    }
    (args.out_dir / "param_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("param_report %s", report)

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
                logger.warning("scaler restore failed: %s", exc)
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

    def _periodic_save(step, phase, metric_state=None):
        _ckpt_ctx.update(step=int(step), phase=str(phase))
        stm.save_checkpoint(
            latest, epoch=_ckpt_ctx["epoch"], step=step, model=model, mtp_clf=mtp_clf,
            optimizer=optimizer, scaler=scaler, best=best, horizons=horizons,
            verb_map=verb_map, noun_map=noun_map, action_map=action_map,
            history=history, phase=phase, metric_state=metric_state,
        )

    def _on_signal(signum, _frame):
        logger.warning("signal %s", signum)
        stop_flag["stop"] = True
        try:
            _periodic_save(max(0, int(_ckpt_ctx["step"])), str(_ckpt_ctx["phase"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("emergency save failed: %s", exc)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)

    from collections import defaultdict
    import time
    import torch.nn as nn
    from app.hdepic_lora_action_anticipation.train_stream_mtp import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        map_labels,
        topk_acc,
    )

    def run_epoch(loader, train: bool, start_step: int = 0, metric_state=None, save_fn=None):
        model.decode_mode = args.train_decode_mode if train else args.val_decode_mode
        model.train(mode=train)
        mtp_clf.train(mode=train)
        crit = nn.CrossEntropyLoss()
        totals, counts = defaultdict(float), defaultdict(int)
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
        stopped = False
        last_it = start_step - 1
        for local_it, batch in enumerate(loader):
            it = start_step + local_it
            if stop_flag["stop"]:
                stopped = True
                break
            last_it = it
            clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
            clips = clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))
            mtp_verbs = batch["mtp_verbs"].to(device)
            mtp_nouns = batch["mtp_nouns"].to(device)
            mtp_mask = batch["mtp_mask"].to(device)
            B = clips.size(0)
            ant = torch.full((B,), float(args.anticipation_sec), device=device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                tokens = model(clips, ant)
                outputs = mtp_clf(tokens)
                head_loss = clips.new_zeros(())
                for hi, h in enumerate(horizons):
                    valid = mtp_mask[:, hi] > 0.5
                    if not bool(valid.any()):
                        continue
                    v_lab, n_lab, a_lab, keep = map_labels(
                        mtp_verbs[valid, hi], mtp_nouns[valid, hi], verb_map, noun_map, action_map, device
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
                        totals[f"action_top5@{h:g}s"] += topk_acc(o["action"][valid_pos].float(), a_lab, 5) * len(keep)
                        counts[f"action_top5@{h:g}s"] += len(keep)
                # primary
                h0 = horizons[primary_idx]
                valid = mtp_mask[:, primary_idx] > 0.5
                if bool(valid.any()):
                    v_lab, n_lab, a_lab, keep = map_labels(
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
                        totals["primary_action_top5"] += topk_acc(o["action"][valid_pos].float(), a_lab, 5) * len(keep)
                        counts["primary_action_top5"] += len(keep)
                aux = model.last_aux if model.last_aux is not None else head_loss.new_zeros(())
                loss = head_loss + float(args.aux_ar_weight) * aux

            if train:
                if not torch.isfinite(loss.detach()):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()

            loss_meter += float(loss.detach().item()) if torch.isfinite(loss.detach()) else 0.0
            n_steps += 1
            if it % 20 == 0:
                logger.info(
                    "%s itr=%d/%d loss=%.4f mode=%s primary@2s≈%.1f",
                    "train" if train else "val",
                    it,
                    len(loader),
                    loss_meter / max(1, n_steps),
                    model.decode_mode,
                    100.0 * totals["primary_action_top5"] / max(1, counts["primary_action_top5"]),
                )
            if train and args.save_every > 0 and (it + 1) % args.save_every == 0 and save_fn is not None:
                ms = {"totals": dict(totals), "counts": dict(counts), "loss_meter": loss_meter, "n_steps": n_steps}
                save_fn(step=it + 1, phase="train" if train else "val", metric_state=ms)
                logger.info("Periodic ckpt step=%d", it + 1)

        metrics = {k: totals[k] / max(1, counts[k]) for k in totals}
        metrics["loss"] = loss_meter / max(1, n_steps)
        metrics["seconds"] = time.time() - t0
        metrics["last_step"] = int(last_it + 1)
        metrics["stopped_early"] = stopped
        metrics["_metric_state"] = {
            "totals": dict(totals),
            "counts": dict(counts),
            "loss_meter": loss_meter,
            "n_steps": n_steps,
        }
        return metrics

    if args.val_only:
        va = run_epoch(val_loader, train=False)
        pub = {k: v for k, v in va.items() if not str(k).startswith("_")}
        (args.out_dir / "val_only_metrics.json").write_text(json.dumps(pub, indent=2))
        return

    for epoch in range(start_epoch, args.epochs):
        _ckpt_ctx["epoch"] = epoch
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        epoch_start = start_step if (epoch == start_epoch and resume_phase == "train") else 0
        val_start = start_step if (epoch == start_epoch and resume_phase == "val") else 0
        skip_train = bool(epoch == start_epoch and resume_phase == "val")

        if skip_train:
            tr = {"loss": float("nan"), "stopped_early": False, "last_step": 0}
        else:
            _ckpt_ctx["phase"] = "train"
            train_sampler.set_start_batch(epoch_start)
            tr = run_epoch(
                train_loader,
                train=True,
                start_step=epoch_start,
                metric_state=(resume_metric_state if epoch_start > 0 else None),
                save_fn=_periodic_save,
            )
            train_sampler.set_start_batch(0)
            if tr.get("stopped_early"):
                _periodic_save(int(tr["last_step"]), "train", tr.get("_metric_state"))
                return
            _periodic_save(0, "val", None)

        _ckpt_ctx["phase"] = "val"
        val_sampler.set_start_batch(val_start)
        va = run_epoch(
            val_loader,
            train=False,
            start_step=val_start,
            metric_state=(resume_metric_state if (skip_train and val_start > 0) else None),
            save_fn=_periodic_save,
        )
        val_sampler.set_start_batch(0)
        if va.get("stopped_early"):
            _periodic_save(int(va["last_step"]), "val", va.get("_metric_state"))
            return

        primary = float(va.get("primary_action_top5", 0.0))
        logger.info(
            "epoch %d train_loss=%.4f val_primary_top5=%.2f%% (val_mode=%s) %s",
            epoch,
            tr.get("loss"),
            100.0 * primary,
            args.val_decode_mode,
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
            logger.info("New best %.4f", best)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        start_step = 0
        resume_phase = "train"
        resume_metric_state = None

    done_flag.write_text(f"ok best={best}\n", encoding="utf-8")
    logger.info("TRAINING_DONE best=%.4f", best)


if __name__ == "__main__":
    main()
