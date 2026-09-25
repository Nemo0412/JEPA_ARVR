#!/usr/bin/env python3
"""128-frame KV cache + 34 new frames → probe, with vs without probe PE.

Encoder streams 34 new frames into a 128-frame KV cache and attention-prunes
34 (keep 94 + new 34 = 128). AttentivePooler has no positional encoding.

Because the cache is pruned every step, probe PE is *not* the raw stream slot
id (holes + unbounded growth). Default PE re-ranks survivors by original time
to a dense 0..63, and keeps original (h, w) inside each slot.

Arms: no PE / rel_rank (recommended) / abs_stream (ablation).
Accuracy is *not* MTP: private +2s / +6s heads, no horizon communication.
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
os.environ.setdefault("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
sys.path.insert(0, os.environ["VJEPA_ROOT"])

import dump_probe_blk0_selfattn_16s_maps as P  # noqa: E402
from app.hdepic_lora_action_anticipation.stream_kvcache_attn_prune import (  # noqa: E402
    CACHE_FRAMES,
    NEW_FRAMES,
    StreamKVAttnPruneEncoder,
    add_probe_positional_encoding,
)
from eval_nopred_128_vs_stream16x8 import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from eval_nopred_kvcache_size_sweep import sample_even_per_video  # noqa: E402
from eval_64slot_pred0_prune_vs_last16 import (  # noqa: E402
    Ctx64Dataset,
    collate,
    summarize,
)

logger = logging.getLogger("probe_posenc")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GP = 256
GRID = 16
HORIZONS = (2.0, 6.0)
MTP_COLS = {2.0: 0, 4.0: 1, 6.0: 2}
METHOD_NAME = "ProbePosEnc"
METHOD_NOTE = (
    "KV cache 128f + new 34f, attn-prune to 128 probe tokens. Probe PE is "
    "rel_rank: dense 0..63 among survivors (handles repeated prune). "
    "abs_stream is an ablation (raw slot ids). Accuracy: independent +2s/+6s, "
    "no MTP communication."
)
PE_ARMS = ("no_pe", "rel_rank", "abs_stream")
ARM_LABELS = {
    "no_pe": "no PE",
    "rel_rank": "PE rel-rank",
    "abs_stream": "PE abs-stream",
}
ARM_COLORS = {"no_pe": "#7f7f7f", "rel_rank": "#c45911", "abs_stream": "#1f4e79"}


def classify_independent(mtp_clf, tokens, horizons=HORIZONS):
    """Private horizon MLP on a shared pool; no cross-horizon attention."""
    fv, fn, fa = mtp_clf._pool_slots(tokens)
    action_only = getattr(mtp_clf.base, "action_only", False) or getattr(mtp_clf.base, "num_verb_classes", 1) == 0
    idx = {float(h): i for i, h in enumerate(mtp_clf.horizons_sec)}
    outputs = {}
    for h in horizons:
        i = idx[float(h)]
        z_a = mtp_clf.horizon_mlps[i](fa)
        if action_only:
            outputs[float(h)] = dict(action=mtp_clf.base.action_classifier(z_a))
        else:
            delta = z_a - fa
            outputs[float(h)] = dict(
                verb=mtp_clf.base.verb_classifier(fv + delta),
                noun=mtp_clf.base.noun_classifier(fn + delta),
                action=mtp_clf.base.action_classifier(z_a),
            )
    return outputs


def update_metrics_by_horizon(totals, counts, outputs, batch, horizons, verb_map, noun_map, action_map, device, prefix: str):
    mtp_verbs = batch["mtp_verbs"]
    mtp_nouns = batch["mtp_nouns"]
    mtp_mask = batch["mtp_mask"]
    for h in horizons:
        col = MTP_COLS[float(h)]
        valid = mtp_mask[:, col] > 0.5
        if not bool(valid.any()):
            continue
        _v, _n, a_lab, keep = P.S.map_labels(
            mtp_verbs[valid, col], mtp_nouns[valid, col], verb_map, noun_map, action_map, device
        )
        if not keep:
            continue
        valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
        acc = P.S.topk_acc(outputs[float(h)]["action"][valid_pos].float(), a_lab, k=5) * len(keep)
        key = f"{prefix}/action_top5@{h:g}s"
        totals[key] += acc
        counts[key] += len(keep)


def per_clip_by_horizon(outputs, batch, horizons, verb_map, noun_map, action_map, device):
    out = {}
    mtp_verbs = batch["mtp_verbs"]
    mtp_nouns = batch["mtp_nouns"]
    mtp_mask = batch["mtp_mask"]
    for h in horizons:
        col = MTP_COLS[float(h)]
        valid = mtp_mask[:, col] > 0.5
        rec = {"valid": bool(valid.any()), "hit_top5": None, "hit_top1": None, "label": None, "pred1": None}
        if rec["valid"]:
            _v, _n, a_lab, keep = P.S.map_labels(
                mtp_verbs[valid, col], mtp_nouns[valid, col], verb_map, noun_map, action_map, device
            )
            if keep:
                valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                logits = outputs[float(h)]["action"][valid_pos].float()
                pred1 = int(logits.argmax(dim=-1)[0].item())
                top5 = logits.topk(min(5, logits.size(-1)), dim=-1).indices[0]
                lab = int(a_lab[0].item())
                rec["label"] = lab
                rec["pred1"] = pred1
                rec["hit_top1"] = pred1 == lab
                rec["hit_top5"] = lab in top5.tolist()
        out[f"{h:g}s"] = rec
    return out


def plot_results(payload: dict, png_path: Path, copy_png: Path | None):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    table = payload.get("accuracy", {}).get("table_action_top5", {})
    horizons = ["2s", "6s"]
    arms = [k for k in PE_ARMS if k in table]
    if table and arms:
        xh = np.arange(len(horizons))
        w = 0.24
        off = (len(arms) - 1) / 2.0
        for i, arm in enumerate(arms):
            ys = [table[arm].get(f"@{h}", float("nan")) for h in horizons]
            ax.bar(xh + (i - off) * w, ys, w, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
        ax.set_xticks(xh)
        ax.set_xticklabels(["+2s", "+6s"])
        ax.set_ylabel("Action Top-5 (%)")
        n = payload["accuracy"].get("n_clips", 0)
        ax.set_title(f"128 KV-cache + 34 → probe  ({n} clips, no MTP)")
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(fontsize=9)
    else:
        ax.set_title("Accuracy (not run)")
        ax.axis("off")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    if copy_png is not None:
        copy_png.parent.mkdir(parents=True, exist_ok=True)
        copy_png.write_bytes(png_path.read_bytes())


def run_accuracy(stream, mtp_clf, embed_dim, device, args, ck_meta, gp: int) -> dict:
    horizons = list(HORIZONS)
    total_frames = CACHE_FRAMES + NEW_FRAMES
    context_sec = total_frames / float(args.fps)
    ds = Ctx64Dataset(
        args.val_csv,
        args.video_root,
        context_sec=context_sec,
        model_fps=args.fps,
        img_size=args.img_size,
        max_samples=0,
        stride=args.stride,
        require_ctx_sec=args.require_ctx_sec if args.require_ctx_sec > 0 else None,
    )
    ds = sample_even_per_video(ds, args.clips_per_video, args.max_samples)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=False
    )
    verb_map, noun_map, action_map = ck_meta["verb_map"], ck_meta["noun_map"], ck_meta["action_map"]
    totals = defaultdict(float)
    counts = defaultdict(int)
    per_clip = []
    t0 = time.time()

    with torch.no_grad():
        for it, batch in enumerate(loader):
            clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
            clips = clips.sub_(IMAGENET_MEAN.to(device)).div_(IMAGENET_STD.to(device))
            if clips.size(2) != total_frames:
                raise RuntimeError(f"expected T={total_frames}, got {clips.size(2)}")
            batch_dev = {
                "mtp_verbs": batch["mtp_verbs"].to(device),
                "mtp_nouns": batch["mtp_nouns"].to(device),
                "mtp_mask": batch["mtp_mask"].to(device),
            }
            vid = batch["video_id"][0]
            tick = int(batch["tick_frame"][0])

            with torch.autocast("cuda", dtype=torch.bfloat16):
                hist = clips[:, :, :CACHE_FRAMES]
                new = clips[:, :, CACHE_FRAMES:]
                state = stream.fill(hist)
                state = stream.step(state, new, mode="attn")
                tok = state.tokens
                if tok.size(-1) != embed_dim:
                    tok = tok[:, :, -embed_dim:]
                tok_rel = add_probe_positional_encoding(
                    tok, state.slot_ids, gp, pos_ids=state.pos, grid_size=GRID, mode="rel_rank"
                )
                tok_abs = add_probe_positional_encoding(
                    tok, state.slot_ids, gp, pos_ids=state.pos, grid_size=GRID, mode="abs_stream"
                )
                outs = {
                    "no_pe": classify_independent(mtp_clf, tok, horizons),
                    "rel_rank": classify_independent(mtp_clf, tok_rel, horizons),
                    "abs_stream": classify_independent(mtp_clf, tok_abs, horizons),
                }

            for prefix, out in outs.items():
                update_metrics_by_horizon(
                    totals, counts, out, batch_dev, horizons, verb_map, noun_map, action_map, device, prefix=prefix
                )
            rec = {
                "idx": it,
                "video_id": vid,
                "tick_frame": tick,
                **{
                    arm: per_clip_by_horizon(outs[arm], batch_dev, horizons, verb_map, noun_map, action_map, device)
                    for arm in PE_ARMS
                },
            }
            per_clip.append(rec)
            if (it + 1) % args.log_every == 0 or it == 0:
                logger.info(
                    "clip %d/%d %s  noPE@2s=%s  rel@2s=%s  abs@2s=%s  noPE@6s=%s  rel@6s=%s  abs@6s=%s",
                    it + 1,
                    len(loader),
                    vid,
                    rec["no_pe"]["2s"]["hit_top5"],
                    rec["rel_rank"]["2s"]["hit_top5"],
                    rec["abs_stream"]["2s"]["hit_top5"],
                    rec["no_pe"]["6s"]["hit_top5"],
                    rec["rel_rank"]["6s"]["hit_top5"],
                    rec["abs_stream"]["6s"]["hit_top5"],
                )

    metrics = summarize(totals, counts)
    table = {}
    for prefix in PE_ARMS:
        table[prefix] = {
            f"@{h:g}s": round(100.0 * metrics.get(f"{prefix}/action_top5@{h:g}s", float("nan")), 4)
            for h in horizons
        }
    delta = {}
    for h in horizons:
        no = metrics.get(f"no_pe/action_top5@{h:g}s", float("nan"))
        rel = metrics.get(f"rel_rank/action_top5@{h:g}s", float("nan"))
        ab = metrics.get(f"abs_stream/action_top5@{h:g}s", float("nan"))
        delta[f"rel_rank_minus_no_pe@{h:g}s"] = round(100.0 * (rel - no), 4)
        delta[f"abs_stream_minus_no_pe@{h:g}s"] = round(100.0 * (ab - no), 4)
        delta[f"rel_rank_minus_abs_stream@{h:g}s"] = round(100.0 * (rel - ab), 4)
    return {
        "method": METHOD_NAME,
        "table_action_top5": table,
        "delta_pp": delta,
        "n_clips": len(per_clip),
        "n_videos": len({r["video_id"] for r in per_clip}),
        "per_clip": per_clip,
        "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
        "seconds": time.time() - t0,
        "config": {
            "cache_frames": CACHE_FRAMES,
            "new_frames": NEW_FRAMES,
            "total_frames": total_frames,
            "context_sec": context_sec,
            "prune": "encoder last-block slot attention, replace lowest 34 frames",
            "probe_pe": (
                "rel_rank = dense 0..63 among current survivors (for repeated prune); "
                "abs_stream = raw stream slot id (holes, grows forever)"
            ),
            "mtp_communication": False,
            "horizons": list(horizons),
            "ckpt": str(args.nopred_ckpt),
            "ckpt_meta": {k: ck_meta[k] for k in ("epoch", "step", "phase")},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", type=Path, default=Path(
        "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/HD_EPIC_val_stream_mtp.csv"
    ))
    ap.add_argument("--video-root", type=Path, default=Path("/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"))
    ap.add_argument("--checkpoint", type=Path, default=Path("/scratch/ll5914/models/vjepa2/vitl.pt"))
    ap.add_argument("--nopred-ckpt", type=Path, default=P.NOPRED_CKPT)
    ap.add_argument("--out-dir", type=Path, default=Path("/scratch/ll5914/experiments/probe_posenc_kvcache_128_34"))
    ap.add_argument("--copy-json", type=Path, default=Path("/home/ll5914/Jepa_yifan/probe_posenc_kvcache_128_34.json"))
    ap.add_argument("--copy-png", type=Path, default=Path("/home/ll5914/Jepa_yifan/probe_posenc_kvcache_128_34.png"))
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=108)
    ap.add_argument("--clips-per-video", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--require-ctx-sec", type=float, default=10.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_frames = CACHE_FRAMES + NEW_FRAMES
    logger.info("loading nopred ViT-L  cache=%df new=%df T=%d  (%s)", CACHE_FRAMES, NEW_FRAMES, total_frames, METHOD_NAME)
    model, mtp_clf, _pooler, ck_meta = P.load_nopred_pooler(
        device, total_frames, args.fps, args.img_size, args.nopred_ckpt, str(args.checkpoint)
    )
    encoder = model.base.encoder
    embed_dim = int(encoder.embed_dim)
    gp = int(getattr(model.base, "grid_size", GRID) ** 2)
    if gp != GP:
        raise RuntimeError(f"expected gp={GP}, got {gp}")
    stream = StreamKVAttnPruneEncoder(
        encoder, gp=gp, tubelet_size=2, cache_frames=CACHE_FRAMES, new_frames=NEW_FRAMES, chunk=args.chunk
    )
    encoder.eval()
    mtp_clf.eval()

    payload = {
        "method": METHOD_NAME,
        "note": METHOD_NOTE,
        "device": torch.cuda.get_device_name(0),
        "accuracy": run_accuracy(stream, mtp_clf, embed_dim, device, args, ck_meta, gp),
    }
    png = args.out_dir / "probe_posenc_kvcache_128_34.png"
    plot_results(payload, png, args.copy_png)
    payload["png"] = str(png)

    out_path = args.out_dir / "metrics.json"
    text = json.dumps(payload, indent=2) + "\n"
    out_path.write_text(text, encoding="utf-8")
    if args.copy_json is not None:
        args.copy_json.parent.mkdir(parents=True, exist_ok=True)
        args.copy_json.write_text(text, encoding="utf-8")
    logger.info("wrote %s", out_path)
    logger.info("accuracy table %s", json.dumps(payload["accuracy"]["table_action_top5"]))
    logger.info("accuracy delta_pp %s", json.dumps(payload["accuracy"]["delta_pp"]))


if __name__ == "__main__":
    main()
