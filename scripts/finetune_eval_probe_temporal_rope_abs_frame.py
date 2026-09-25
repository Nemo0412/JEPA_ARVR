#!/usr/bin/env python3
"""128 KV-cache + attn prune: baseline vs probe 1D temporal RoPE (abs Frame Index).

Protocol matches eval_probe_posenc_kvcache_128_34:
  encode 128f → stream +34f → attention-prune back to 128f → probe.

RoPE (no new params): default **all Probe.blocks[*]** self-attn
  Q'_p = R(p) Q_p,  K'_p = R(p) K_p,  θ_{p,m} = p · ω_m
with p = original Frame/slot Index surviving prune (abs_stream).

Fair accuracy needs **joint** encoder-LoRA + probe FT (RoPE changes Q/K
geometry; frozen-encoder probe-only FT underperforms). Arms:
  baseline       original ckpt, no probe RoPE
  rope_zeroshot  original ckpt + temporal RoPE (shows why FT is needed)
  rope_ft        joint enc-LoRA+probe FT with temporal RoPE on, then eval
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
    token_frame_ids_from_slots,
)
from eval_nopred_128_vs_stream16x8 import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from eval_nopred_kvcache_size_sweep import sample_even_per_video  # noqa: E402
from eval_64slot_pred0_prune_vs_last16 import Ctx64Dataset, collate, summarize  # noqa: E402
from eval_probe_posenc_kvcache_128_34 import (  # noqa: E402
    HORIZONS,
    MTP_COLS,
    classify_independent,
    update_metrics_by_horizon,
)

logger = logging.getLogger("probe_trope")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GP = 256
METHOD = "ProbeTemporalRoPEAbsFrame_JointEncProbe"
ARMS = ("baseline", "rope_zeroshot", "rope_ft")
ARM_LABELS = {
    "baseline": "128 KV (no probe RoPE)",
    "rope_zeroshot": "temporal RoPE zero-shot",
    "rope_ft": "temporal RoPE + joint FT",
}
ARM_COLORS = {"baseline": "#7f7f7f", "rope_zeroshot": "#1f4e79", "rope_ft": "#c45911"}


def normalize_clip(clip_uint8: torch.Tensor, device) -> torch.Tensor:
    clips = clip_uint8.to(device, non_blocking=True).float().div_(255.0)
    return clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))


def encode_pruned_128(
    stream: StreamKVAttnPruneEncoder,
    clips: torch.Tensor,
    embed_dim: int,
    *,
    train_mode: bool = False,
):
    """Return tokens [B,N,D] and abs-stream frame ids [B,N] after 128+34 attn prune.

    ``train_mode``: fill under no_grad; prune scores detached; admit encode with grad.
    """
    hist = clips[:, :, :CACHE_FRAMES]
    new = clips[:, :, CACHE_FRAMES:]
    if train_mode:
        with torch.no_grad():
            state = stream.fill(hist)
            scores = state.slot_scores.detach()
        state = stream.step(state, new, mode="attn", slot_scores=scores)
    else:
        state = stream.fill(hist)
        state = stream.step(state, new, mode="attn")
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
def eval_arm(stream, mtp_clf, embed_dim, loader, device, ck_meta, rope: ProbeTemporalRoPE | None, prefix: str):
    totals = defaultdict(float)
    counts = defaultdict(int)
    verb_map, noun_map, action_map = ck_meta["verb_map"], ck_meta["noun_map"], ck_meta["action_map"]
    horizons = list(HORIZONS)
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
            tok, frame_ids = encode_pruned_128(stream, clips, embed_dim, train_mode=False)
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
    table["n@2s"] = int(metrics.get(f"n|{prefix}/action_top5@2s", 0))
    return table, metrics


def set_joint_trainable(model, mtp_clf, *, train_encoder_lora: bool):
    for p in mtp_clf.parameters():
        p.requires_grad = True
    n_lora = set_encoder_lora_trainable(model, trainable=train_encoder_lora)
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue
        p.requires_grad = False
    return n_lora


def finetune_joint(
    model,
    mtp_clf,
    stream,
    embed_dim,
    train_loader,
    *,
    rope: ProbeTemporalRoPE,
    epochs: int,
    lr: float,
    encoder_lr_mult: float,
    horizons,
    weights,
    ck_meta,
    device,
):
    set_joint_trainable(model, mtp_clf, train_encoder_lora=True)
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
        "joint FT probe_params=%d enc_lora=%d lr=%.2e enc_lr=%.2e",
        sum(p.numel() for p in probe_params),
        sum(p.numel() for p in enc_params),
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
            batch_dev = {
                "mtp_verbs": batch["mtp_verbs"].to(device),
                "mtp_nouns": batch["mtp_nouns"].to(device),
                "mtp_mask": batch["mtp_mask"].to(device),
            }
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tok, frame_ids = encode_pruned_128(stream, clips, embed_dim, train_mode=True)
                    rope.set_frame_ids(frame_ids)
                    outputs = classify_independent(mtp_clf, tok, horizons)
                    head_loss, n_used = mtp_ce_loss(
                        outputs, batch_dev, horizons, weights, verb_map, noun_map, action_map, device
                    )
            finally:
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
            "epoch %d/%d steps=%d loss=%.4f",
            ep + 1,
            epochs,
            n_step,
            loss_meter / max(1, n_step),
        )
    return time.time() - t0, loss_meter / max(1, n_step)


def plot_results(payload: dict, png: Path, copy_png: Path | None):
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    table = payload["accuracy"]["table_action_top5"]
    horizons = ["2s", "6s"]
    arms = [a for a in ARMS if a in table]
    xh = np.arange(len(horizons))
    w = 0.24
    off = (len(arms) - 1) / 2.0
    for i, arm in enumerate(arms):
        ys = [table[arm].get(f"@{h}", float("nan")) for h in horizons]
        ax.bar(xh + (i - off) * w, ys, w, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax.set_xticks(xh)
    ax.set_xticklabels(["+2s", "+6s"])
    ax.set_ylabel("Action Top-5 (%)")
    n = payload["accuracy"].get("n_val_clips", 0)
    ax.set_title(f"128 KV + 34 attn-prune → joint enc+probe   val={n}")
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
        "/scratch/ll5914/experiments/probe_temporal_rope_abs_frame_joint_128kv"
    ))
    ap.add_argument("--copy-json", type=Path, default=Path(
        "/home/ll5914/Jepa_yifan/JEPA_ARVR/configs/jepa_pe/probe_temporal_rope_abs_frame_joint_128kv.json"
    ))
    ap.add_argument("--copy-png", type=Path, default=Path(
        "/home/ll5914/Jepa_yifan/JEPA_ARVR/plots/probe_temporal_rope_abs_frame_joint_128kv.png"
    ))
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--train-clips-per-video", type=int, default=4)
    ap.add_argument("--val-clips-per-video", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=108)
    ap.add_argument("--max-val", type=int, default=108)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder-lr-mult", type=float, default=0.5)
    ap.add_argument("--require-ctx-sec", type=float, default=10.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--skip-finetune", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_frames = CACHE_FRAMES + NEW_FRAMES
    horizons = list(HORIZONS)
    weights = [1.0] * len(horizons)

    logger.info("load nopred ViT-L  T=%d  method=%s", total_frames, METHOD)
    model, mtp_clf, pooler, ck_meta = P.load_nopred_pooler(
        device, total_frames, args.fps, args.img_size, args.nopred_ckpt, str(args.checkpoint)
    )
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

    set_joint_trainable(model, mtp_clf, train_encoder_lora=False)
    model.eval()
    mtp_clf.eval()

    logger.info("eval baseline (no probe RoPE)")
    base_table, _ = eval_arm(stream, mtp_clf, embed_dim, val_loader, device, ck_meta, rope=None, prefix="baseline")
    logger.info("baseline %s", json.dumps(base_table))

    rope = ProbeTemporalRoPE(pooler, rope_cross_attn_k=False, only_block0=False)
    logger.info("eval rope_zeroshot (RoPE on all probe self-attn blocks)")
    zs_table, _ = eval_arm(stream, mtp_clf, embed_dim, val_loader, device, ck_meta, rope=rope, prefix="rope_zeroshot")
    logger.info("rope_zeroshot %s", json.dumps(zs_table))

    ft_table = {f"@{h:g}s": float("nan") for h in horizons}
    ft_table["n@2s"] = 0
    train_seconds = 0.0
    train_loss = float("nan")
    ft_ckpt = args.out_dir / "probe_enc_temporal_rope_ft.pt"

    if not args.skip_finetune:
        mtp_clf.load_state_dict(init_probe)
        load_encoder_lora_state_dict(model, init_enc_lora, strict=False)
        logger.info("joint finetune rope_ft (encoder LoRA + probe)")
        train_seconds, train_loss = finetune_joint(
            model, mtp_clf, stream, embed_dim, train_loader,
            rope=rope, epochs=args.epochs, lr=args.lr, encoder_lr_mult=args.encoder_lr_mult,
            horizons=horizons, weights=weights, ck_meta=ck_meta, device=device,
        )
        torch.save(
            {
                "mtp_classifier": mtp_clf.state_dict(),
                "encoder_lora": encoder_lora_state_dict(model),
                "epochs": args.epochs,
                "lr": args.lr,
                "encoder_lr_mult": args.encoder_lr_mult,
                "rope": "temporal_1d_abs_frame_index_all_probe_blocks",
                "train": "joint_encoder_lora_probe",
                "cache_frames": CACHE_FRAMES,
                "new_frames": NEW_FRAMES,
            },
            ft_ckpt,
        )
        logger.info("wrote %s", ft_ckpt)
        model.eval()
        mtp_clf.eval()
        set_joint_trainable(model, mtp_clf, train_encoder_lora=False)
        logger.info("eval rope_ft")
        ft_table, _ = eval_arm(stream, mtp_clf, embed_dim, val_loader, device, ck_meta, rope=rope, prefix="rope_ft")
        logger.info("rope_ft %s", json.dumps(ft_table))

    rope.remove()

    table = {"baseline": base_table, "rope_zeroshot": zs_table, "rope_ft": ft_table}
    delta = {}
    for h in horizons:
        b = base_table.get(f"@{h:g}s", float("nan"))
        z = zs_table.get(f"@{h:g}s", float("nan"))
        f = ft_table.get(f"@{h:g}s", float("nan"))
        delta[f"zeroshot_minus_baseline@{h:g}s"] = round(float(z - b), 4)
        delta[f"ft_minus_baseline@{h:g}s"] = round(float(f - b), 4)
        delta[f"ft_minus_zeroshot@{h:g}s"] = round(float(f - z), 4)

    payload = {
        "method": METHOD,
        "note": (
            "Probe 1D temporal RoPE on all probe self-attn blocks with original "
            "Frame/slot Index after 128+34 attn prune. Joint encoder-LoRA + probe FT."
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
            "prune": "encoder last-block slot attention, replace lowest 34 frames",
            "rope": "1D temporal on all Probe.blocks[*] Q/K; abs stream slot id",
            "train": "joint_encoder_lora_probe",
            "epochs": args.epochs,
            "lr": args.lr,
            "encoder_lr_mult": args.encoder_lr_mult,
            "train_seconds": train_seconds,
            "train_loss": train_loss,
            "ckpt": str(args.nopred_ckpt),
            "ft_ckpt": str(ft_ckpt) if not args.skip_finetune else None,
        },
    }
    png = args.out_dir / "probe_temporal_rope_abs_frame_joint_128kv.png"
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
