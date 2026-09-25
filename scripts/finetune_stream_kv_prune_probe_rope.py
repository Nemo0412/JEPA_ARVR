#!/usr/bin/env python3
"""Streaming + KV-cache + probe-blk0-attn prune: joint encoder-LoRA + probe FT ± RoPE.

Protocol:

  * Streaming ticks ~every 4s: encode NEW_FRAMES=34 (no predictor).
  * KV cache = CACHE_FRAMES=128. After probe on the current 128, **Probe
    blocks[0] self-attn** received mass → slot scores (recorded/detached, not
    trained through the discrete keep/drop).
  * Next admit: drop lowest 34 frames by those scores; keep 94; encode new 34
    against kept KV; packed 94+34=128 → probe prediction.
  * One-clip bootstrap: fill 128 → probe-blk0 scores → step(+34) → probe.
  * **Joint FT:** encoder LoRA + probe/heads. History ``fill`` is ``no_grad``
    (detached K/V); the admit ``step`` and probe run with grad (same spirit as
    matched stream ``train_last_chunk_only``).
  * RoPE arm: 1D temporal RoPE on Probe.blocks[0] Q/K with abs surviving frame ids.
  * Horizon: ``--horizons 2`` or ``--horizons 6`` (default 2).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
os.environ.setdefault("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
sys.path.insert(0, os.environ["VJEPA_ROOT"])

import dump_probe_blk0_selfattn_16s_maps as P  # noqa: E402
from app.hdepic_lora_action_anticipation.encoder_lora import (  # noqa: E402
    encoder_lora_state_dict,
    load_encoder_lora_state_dict,
    set_encoder_lora_trainable,
    trainable_encoder_lora_params,
)
from app.hdepic_lora_action_anticipation.stream_kvcache_attn_prune import (  # noqa: E402
    CACHE_FRAMES,
    NEW_FRAMES,
    ProbeTemporalRoPE,
    StreamKVAttnPruneEncoder,
    probe_blk0_slot_scores,
    token_frame_ids_from_slots,
)
from eval_nopred_128_vs_stream16x8 import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from eval_nopred_kvcache_size_sweep import sample_even_per_video  # noqa: E402
from eval_64slot_pred0_prune_vs_last16 import Ctx64Dataset, collate, summarize  # noqa: E402
from eval_probe_posenc_kvcache_128_34 import (  # noqa: E402
    MTP_COLS,
    classify_independent,
    update_metrics_by_horizon,
)

logger = logging.getLogger("kv_prune_rope_ft")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GP = 256
METHOD = "StreamKVProbeBlk0Prune_JointEncProbe_RoPE_FT"
ARMS = ("zs_no_rope", "zs_rope", "ft_no_rope", "ft_rope")
ARM_LABELS = {
    "zs_no_rope": "KV+probePrune (no FT)",
    "zs_rope": "KV+probePrune+RoPE (no FT)",
    "ft_no_rope": "KV+probePrune joint FT",
    "ft_rope": "KV+probePrune+RoPE joint FT",
}
ARM_COLORS = {
    "zs_no_rope": "#9e9e9e",
    "zs_rope": "#90caf9",
    "ft_no_rope": "#c62828",
    "ft_rope": "#1565c0",
}


def normalize_clip(clip_uint8: torch.Tensor, device) -> torch.Tensor:
    clips = clip_uint8.to(device, non_blocking=True).float().div_(255.0)
    return clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))


def encode_probe_attn_pruned(
    stream: StreamKVAttnPruneEncoder,
    pooler: nn.Module,
    clips: torch.Tensor,
    embed_dim: int,
    *,
    chunk: int = 256,
    train_mode: bool = False,
):
    """fill 128 → probe-blk0 scores → step(+34 prune) → tokens + abs frame ids.

    ``train_mode``: history fill + prune scores under ``no_grad``; admit encode
    runs with grad so encoder LoRA + probe can jointly train.
    """
    hist = clips[:, :, :CACHE_FRAMES]
    new = clips[:, :, CACHE_FRAMES:]
    if train_mode:
        with torch.no_grad():
            state = stream.fill(hist)
            tok0 = state.tokens
            if tok0.size(-1) != embed_dim:
                tok0 = tok0[:, :, -embed_dim:]
            scores = probe_blk0_slot_scores(pooler, tok0, stream.gp, chunk=chunk).detach()
        state = stream.step(state, new, mode="attn", slot_scores=scores)
    else:
        state = stream.fill(hist)
        tok0 = state.tokens
        if tok0.size(-1) != embed_dim:
            tok0 = tok0[:, :, -embed_dim:]
        scores = probe_blk0_slot_scores(pooler, tok0, stream.gp, chunk=chunk)
        state = stream.step(state, new, mode="attn", slot_scores=scores)
    tok = state.tokens
    if tok.size(-1) != embed_dim:
        tok = tok[:, :, -embed_dim:]
    frame_ids = token_frame_ids_from_slots(state.slot_ids, stream.gp)
    return tok, frame_ids


def mtp_ce_loss(outputs, batch_dev, horizons, weights, verb_map, noun_map, action_map, device):
    crit = nn.CrossEntropyLoss()
    head_loss = None
    n_used = 0
    for hi, h in enumerate(horizons):
        col = MTP_COLS[float(h)]
        valid = batch_dev["mtp_mask"][:, col] > 0.5
        if not bool(valid.any()):
            continue
        v_lab, n_lab, a_lab, keep = P.S.map_labels(
            batch_dev["mtp_verbs"][valid, col],
            batch_dev["mtp_nouns"][valid, col],
            verb_map,
            noun_map,
            action_map,
            device,
        )
        if not keep:
            continue
        valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
        o = outputs[float(h)]
        step = crit(o["action"][valid_pos], a_lab)
        if "verb" in o and "noun" in o:
            step = step + crit(o["verb"][valid_pos], v_lab) + crit(o["noun"][valid_pos], n_lab)
        head_loss = step * float(weights[hi]) if head_loss is None else head_loss + step * float(weights[hi])
        n_used += 1
    return head_loss, n_used


@torch.no_grad()
def eval_arm(
    stream,
    mtp_clf,
    pooler,
    embed_dim,
    loader,
    device,
    ck_meta,
    rope: ProbeTemporalRoPE | None,
    prefix: str,
    chunk: int,
    horizons,
):
    totals = defaultdict(float)
    counts = defaultdict(int)
    verb_map, noun_map, action_map = ck_meta["verb_map"], ck_meta["noun_map"], ck_meta["action_map"]
    mtp_clf.eval()
    for batch in loader:
        clips = normalize_clip(batch["clip"], device)
        if clips.size(2) != CACHE_FRAMES + NEW_FRAMES:
            raise RuntimeError(f"expected T={CACHE_FRAMES + NEW_FRAMES}, got {clips.size(2)}")
        batch_dev = {
            "mtp_verbs": batch["mtp_verbs"].to(device),
            "mtp_nouns": batch["mtp_nouns"].to(device),
            "mtp_mask": batch["mtp_mask"].to(device),
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok, frame_ids = encode_probe_attn_pruned(
                stream, pooler, clips, embed_dim, chunk=chunk, train_mode=False
            )
            if rope is not None:
                rope.set_frame_ids(frame_ids)
            try:
                outputs = classify_independent(mtp_clf, tok, horizons)
            finally:
                if rope is not None:
                    rope.set_frame_ids(None)
        update_metrics_by_horizon(
            totals, counts, outputs, batch_dev, horizons, verb_map, noun_map, action_map, device, prefix=prefix
        )
    metrics = summarize(totals, counts)
    table = {
        f"@{h:g}s": round(100.0 * metrics.get(f"{prefix}/action_top5@{h:g}s", float("nan")), 4)
        for h in horizons
    }
    for h in horizons:
        table[f"n@{h:g}s"] = int(metrics.get(f"n|{prefix}/action_top5@{h:g}s", 0))
    return table, metrics


def set_joint_trainable(model, mtp_clf, *, train_encoder_lora: bool):
    for p in mtp_clf.parameters():
        p.requires_grad = True
    n_lora = set_encoder_lora_trainable(model, trainable=train_encoder_lora)
    # Keep non-LoRA encoder weights frozen.
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue
        p.requires_grad = False
    return n_lora


def finetune_joint(
    model,
    mtp_clf,
    pooler,
    stream,
    embed_dim,
    train_loader,
    *,
    rope: ProbeTemporalRoPE | None,
    epochs: int,
    lr: float,
    encoder_lr_mult: float,
    horizons,
    weights,
    ck_meta,
    device,
    chunk: int,
):
    n_lora = set_joint_trainable(model, mtp_clf, train_encoder_lora=True)
    mtp_clf.train()
    model.train()
    probe_params = [p for p in mtp_clf.parameters() if p.requires_grad]
    enc_params = trainable_encoder_lora_params(model)
    param_groups = [{"params": probe_params, "lr": lr}]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": lr * float(encoder_lr_mult)})
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    verb_map, noun_map, action_map = ck_meta["verb_map"], ck_meta["noun_map"], ck_meta["action_map"]
    logger.info(
        "joint FT: probe_params=%d enc_lora_params=%d (n_lora_tensors~%d) lr=%.2e enc_lr=%.2e",
        sum(p.numel() for p in probe_params),
        sum(p.numel() for p in enc_params),
        n_lora,
        lr,
        lr * float(encoder_lr_mult),
    )
    t0 = time.time()
    n_step = 0
    loss_meter = 0.0
    batches = list(train_loader)
    for ep in range(epochs):
        order = np.random.permutation(len(batches))
        for idx in order:
            batch = batches[int(idx)]
            clips = normalize_clip(batch["clip"], device)
            if clips.size(2) != CACHE_FRAMES + NEW_FRAMES:
                raise RuntimeError(f"expected T={CACHE_FRAMES + NEW_FRAMES}, got {clips.size(2)}")
            batch_dev = {
                "mtp_verbs": batch["mtp_verbs"].to(device),
                "mtp_nouns": batch["mtp_nouns"].to(device),
                "mtp_mask": batch["mtp_mask"].to(device),
            }
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tok, frame_ids = encode_probe_attn_pruned(
                        stream, pooler, clips, embed_dim, chunk=chunk, train_mode=True
                    )
                    if rope is not None:
                        rope.set_frame_ids(frame_ids)
                    outputs = classify_independent(mtp_clf, tok, horizons)
                    head_loss, n_used = mtp_ce_loss(
                        outputs, batch_dev, horizons, weights, verb_map, noun_map, action_map, device
                    )
            finally:
                if rope is not None:
                    rope.set_frame_ids(None)
            if head_loss is None or n_used == 0 or not torch.isfinite(head_loss.detach()):
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(head_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in probe_params + enc_params if p.requires_grad], 1.0
            )
            scaler.step(opt)
            scaler.update()
            loss_meter += float(head_loss.detach())
            n_step += 1
        logger.info(
            "epoch %d/%d steps=%d loss=%.4f rope=%s",
            ep + 1,
            epochs,
            n_step,
            loss_meter / max(1, n_step),
            rope is not None,
        )
    return time.time() - t0, loss_meter / max(1, n_step)


def plot_results(payload: dict, png: Path, copy_png: Path | None):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    table = payload["accuracy"]["table_action_top5"]
    cfg_h = payload.get("config", {}).get("horizons_sec") or [2.0]
    horizons = [f"{float(h):g}s" for h in cfg_h]
    arms = [a for a in ARMS if a in table]
    xh = np.arange(len(horizons))
    w = 0.2
    off = (len(arms) - 1) / 2.0
    for i, arm in enumerate(arms):
        ys = [table[arm].get(f"@{h}", float("nan")) for h in horizons]
        ax.bar(xh + (i - off) * w, ys, w, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax.set_xticks(xh)
    ax.set_xticklabels([f"+{h}" for h in horizons])
    ax.set_ylabel("Action Top-5 (%)")
    n = payload["accuracy"].get("n_val_clips", 0)
    ax.set_title(f"Stream KV + probe-blk0 prune joint enc+probe   val={n}")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=160)
    plt.close(fig)
    if copy_png is not None:
        copy_png.parent.mkdir(parents=True, exist_ok=True)
        copy_png.write_bytes(png.read_bytes())


def make_loader(csv_path, video_root, args, per_vid, max_n, shuffle=False):
    ds = Ctx64Dataset(
        csv_path,
        video_root,
        context_sec=(CACHE_FRAMES + NEW_FRAMES) / float(args.fps),
        model_fps=args.fps,
        img_size=args.img_size,
        max_samples=0,
        stride=1,
        require_ctx_sec=args.require_ctx_sec if args.require_ctx_sec > 0 else None,
    )
    ds = sample_even_per_video(ds, per_vid, max_n)
    return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=args.num_workers, collate_fn=collate), ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=Path(
        "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/HD_EPIC_train_stream_mtp.csv"
    ))
    ap.add_argument("--val-csv", type=Path, default=Path(
        "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/HD_EPIC_val_stream_mtp.csv"
    ))
    ap.add_argument("--video-root", type=Path, default=Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"))
    ap.add_argument("--checkpoint", type=Path, default=Path("/scratch/ll5914/models/vjepa2/vitl.pt"))
    ap.add_argument("--nopred-ckpt", type=Path, default=P.NOPRED_CKPT)
    ap.add_argument("--out-dir", type=Path, default=Path(
        "/scratch/ll5914/experiments/stream_kv_probe_blk0_prune_joint_ft_rope"
    ))
    ap.add_argument("--copy-json", type=Path, default=Path(
        "/home/ll5914/Jepa_yifan/JEPA_ARVR/configs/jepa_pe/stream_kv_probe_blk0_prune_joint_ft_rope.json"
    ))
    ap.add_argument("--copy-png", type=Path, default=Path(
        "/home/ll5914/Jepa_yifan/JEPA_ARVR/plots/stream_kv_probe_blk0_prune_joint_ft_rope.png"
    ))
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--train-clips-per-video", type=int, default=8)
    ap.add_argument("--val-clips-per-video", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=400)
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder-lr-mult", type=float, default=0.5)
    ap.add_argument("--require-ctx-sec", type=float, default=10.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument(
        "--horizons",
        type=str,
        default="2",
        help="Comma-separated horizons in seconds (single-horizon preferred: 2 or 6)",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_frames = CACHE_FRAMES + NEW_FRAMES
    horizons = [float(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    if not horizons:
        raise SystemExit("--horizons must list at least one value, e.g. 2 or 6")
    for h in horizons:
        if float(h) not in MTP_COLS:
            raise SystemExit(f"unsupported horizon {h}; known cols={sorted(MTP_COLS)}")
    weights = [1.0] * len(horizons)

    logger.info(
        "load nopred ViT-L  T=%d  cache=%d new=%d(~%.2fs) prune=probe_blk0  method=%s",
        total_frames,
        CACHE_FRAMES,
        NEW_FRAMES,
        NEW_FRAMES / float(args.fps),
        METHOD,
    )
    model, mtp_clf, pooler, ck_meta = P.load_nopred_pooler(
        device, total_frames, args.fps, args.img_size, args.nopred_ckpt, str(args.checkpoint)
    )
    if getattr(pooler, "blocks", None) is None or len(pooler.blocks) < 1:
        raise SystemExit("nopred probe must have blocks[0] (depth>1) for probe-blk0 prune")
    encoder = model.base.encoder
    embed_dim = int(encoder.embed_dim)
    gp = int(getattr(model.base, "grid_size", 16) ** 2)
    if gp != GP:
        raise RuntimeError(f"expected gp={GP}, got {gp}")
    stream = StreamKVAttnPruneEncoder(
        encoder, gp=gp, tubelet_size=2, cache_frames=CACHE_FRAMES, new_frames=NEW_FRAMES, chunk=args.chunk
    )

    init_probe = copy.deepcopy(mtp_clf.state_dict())
    init_enc_lora = encoder_lora_state_dict(model)

    train_loader, train_ds = make_loader(
        args.train_csv, args.video_root, args, args.train_clips_per_video, args.max_train
    )
    val_loader, val_ds = make_loader(
        args.val_csv, args.video_root, args, args.val_clips_per_video, args.max_val
    )
    logger.info("train=%d val=%d", len(train_ds), len(val_ds))

    # Zero-shot eval with frozen weights.
    set_joint_trainable(model, mtp_clf, train_encoder_lora=False)
    model.eval()
    mtp_clf.eval()
    logger.info("eval zs_no_rope")
    zs_no, _ = eval_arm(
        stream, mtp_clf, pooler, embed_dim, val_loader, device, ck_meta, rope=None,
        prefix="zs_no_rope", chunk=args.chunk, horizons=horizons,
    )
    rope = ProbeTemporalRoPE(pooler, rope_cross_attn_k=False, only_block0=True)
    logger.info("eval zs_rope")
    zs_rope, _ = eval_arm(
        stream, mtp_clf, pooler, embed_dim, val_loader, device, ck_meta, rope=rope,
        prefix="zs_rope", chunk=args.chunk, horizons=horizons,
    )
    rope.remove()
    rope = None

    def _reset_init():
        mtp_clf.load_state_dict(init_probe)
        load_encoder_lora_state_dict(model, init_enc_lora, strict=False)

    _reset_init()
    logger.info("joint finetune ft_no_rope (encoder LoRA + probe)")
    t_no, loss_no = finetune_joint(
        model, mtp_clf, pooler, stream, embed_dim, train_loader,
        rope=None, epochs=args.epochs, lr=args.lr, encoder_lr_mult=args.encoder_lr_mult,
        horizons=horizons, weights=weights, ck_meta=ck_meta, device=device, chunk=args.chunk,
    )
    ft_no_ckpt = args.out_dir / "probe_enc_ft_no_rope.pt"
    torch.save(
        {
            "mtp_classifier": mtp_clf.state_dict(),
            "encoder_lora": encoder_lora_state_dict(model),
            "rope": False,
            "prune": "probe_blk0",
            "train": "joint_encoder_lora_probe",
        },
        ft_no_ckpt,
    )
    model.eval()
    mtp_clf.eval()
    set_joint_trainable(model, mtp_clf, train_encoder_lora=False)
    ft_no, _ = eval_arm(
        stream, mtp_clf, pooler, embed_dim, val_loader, device, ck_meta, rope=None,
        prefix="ft_no_rope", chunk=args.chunk, horizons=horizons,
    )
    logger.info("ft_no_rope %s", json.dumps(ft_no))

    _reset_init()
    rope = ProbeTemporalRoPE(pooler, rope_cross_attn_k=False, only_block0=True)
    logger.info("joint finetune ft_rope (abs frame id; Probe.blocks[0] only)")
    t_rope, loss_rope = finetune_joint(
        model, mtp_clf, pooler, stream, embed_dim, train_loader,
        rope=rope, epochs=args.epochs, lr=args.lr, encoder_lr_mult=args.encoder_lr_mult,
        horizons=horizons, weights=weights, ck_meta=ck_meta, device=device, chunk=args.chunk,
    )
    ft_rope_ckpt = args.out_dir / "probe_enc_ft_rope.pt"
    torch.save(
        {
            "mtp_classifier": mtp_clf.state_dict(),
            "encoder_lora": encoder_lora_state_dict(model),
            "rope": "temporal_1d_abs_frame_index_blk0_only",
            "prune": "probe_blk0",
            "train": "joint_encoder_lora_probe",
            "cache_frames": CACHE_FRAMES,
            "new_frames": NEW_FRAMES,
            "horizons_sec": horizons,
        },
        ft_rope_ckpt,
    )
    model.eval()
    mtp_clf.eval()
    set_joint_trainable(model, mtp_clf, train_encoder_lora=False)
    ft_rope, _ = eval_arm(
        stream, mtp_clf, pooler, embed_dim, val_loader, device, ck_meta, rope=rope,
        prefix="ft_rope", chunk=args.chunk, horizons=horizons,
    )
    logger.info("ft_rope %s", json.dumps(ft_rope))
    rope.remove()

    table = {"zs_no_rope": zs_no, "zs_rope": zs_rope, "ft_no_rope": ft_no, "ft_rope": ft_rope}
    delta = {}
    for h in horizons:
        b0 = zs_no.get(f"@{h:g}s", float("nan"))
        fn = ft_no.get(f"@{h:g}s", float("nan"))
        fr = ft_rope.get(f"@{h:g}s", float("nan"))
        delta[f"ft_no_rope_minus_zs@{h:g}s"] = round(float(fn - b0), 4)
        delta[f"ft_rope_minus_zs@{h:g}s"] = round(float(fr - b0), 4)
        delta[f"ft_rope_minus_ft_no_rope@{h:g}s"] = round(float(fr - fn), 4)

    payload = {
        "method": METHOD,
        "note": (
            "Joint encoder-LoRA + probe FT under stream KV + prune by Probe "
            "blocks[0] self-attn received mass (scores detached for keep/drop). "
            "Bootstrap: fill-128 no_grad → admit +34 with grad → probe. "
            f"Single-horizon FT/eval on {horizons}s. "
            "RoPE uses abs surviving frame/slot ids."
        ),
        "device": torch.cuda.get_device_name(0),
        "accuracy": {
            "table_action_top5": table,
            "delta_pp": delta,
            "n_val_clips": len(val_ds),
            "n_train_clips": len(train_ds),
        },
        "config": {
            "cache_frames": CACHE_FRAMES,
            "new_frames": NEW_FRAMES,
            "keep_frames_after_prune": CACHE_FRAMES - NEW_FRAMES,
            "tick_sec_approx": NEW_FRAMES / float(args.fps),
            "fps": args.fps,
            "horizons_sec": horizons,
            "train": "joint_encoder_lora_probe",
            "encoder_lr_mult": args.encoder_lr_mult,
            "prune": "probe blocks[0] self-attn received mass → drop lowest 34 frames",
            "rope": "1D temporal RoPE on Probe.blocks[0] self-attn Q/K only",
            "epochs": args.epochs,
            "lr": args.lr,
            "train_seconds_no_rope": t_no,
            "train_seconds_rope": t_rope,
            "train_loss_no_rope": loss_no,
            "train_loss_rope": loss_rope,
            "ckpt": str(args.nopred_ckpt),
            "ft_no_rope_ckpt": str(ft_no_ckpt),
            "ft_rope_ckpt": str(ft_rope_ckpt),
        },
    }
    png = args.out_dir / "stream_kv_probe_blk0_prune_joint_ft_rope.png"
    plot_results(payload, png, args.copy_png)
    payload["png"] = str(png)
    text = json.dumps(payload, indent=2) + "\n"
    (args.out_dir / "metrics.json").write_text(text, encoding="utf-8")
    if args.copy_json is not None:
        args.copy_json.parent.mkdir(parents=True, exist_ok=True)
        args.copy_json.write_text(text, encoding="utf-8")
    logger.info("wrote %s", args.out_dir / "metrics.json")
    logger.info("table %s", json.dumps(table, indent=2))
    logger.info("delta_pp %s", json.dumps(delta, indent=2))


if __name__ == "__main__":
    main()
