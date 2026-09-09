#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Visualize WHICH tokens a prune method DROPS: black out the dropped
# patches on the real frames (kept patches stay visible), across time slots. Works for every
# strategy — encoder attention, recent, online predictor-score (pred_attention_*), and the
# offline calibrated map (pred_offline_*). Single clip, same window builder as the overlay probe.
# Slurm only.
from __future__ import annotations

import argparse, os, sys
from pathlib import Path
import numpy as np, torch
from decord import VideoReader, cpu

CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(CODE_ROOT, "vjepa2"), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (  # noqa: E402
    FpsSubsampledStreamMTPDataset, enlarge_predictor_budget, RecentTokenPruner,
    OfflineCalibPruner, PredictorScorePrunedModel,
)
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import (  # noqa: E402
    build_finetuned_20,
)


def kept_indices(strategy, base, x_full, clip, ant, keep_count, gp, calib_path, score_block):
    """Return sorted kept absolute positions (1D LongTensor) for the given strategy."""
    N = x_full.shape[1]
    if strategy == "attention":
        pruner = T.TokenPruner(base.encoder, keep_count=keep_count, gp=gp)
        # TokenPruner scores during the encoder forward; re-run encoder once to populate _importance
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            base.encoder(clip)
        _, idx = pruner.prune(x_full)
        pruner.remove()
        return idx[0]
    if strategy == "recent":
        _, idx = RecentTokenPruner(keep_count, gp).prune(x_full)
        return idx[0]
    if strategy in ("pred_offline_high", "pred_offline_low"):
        mode = "high" if strategy.endswith("high") else "low"
        _, idx = OfflineCalibPruner(calib_path, keep_count, gp, mode).prune(x_full)
        return idx[0]
    if strategy in ("pred_attention_high", "pred_attention_low"):
        mode = "high" if strategy.endswith("high") else "low"
        m = PredictorScorePrunedModel(base, keep_count, gp, mode=mode, score_block=score_block)
        idx = m._score_select(base, x_full, ant)
        return idx[0]
    raise SystemExit(f"unknown strategy {strategy!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoder-lora", default=None); ap.add_argument("--predictor-lora", default=None)
    ap.add_argument("--init-from-ckpt", default=None)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--calib-path", default=None); ap.add_argument("--score-block", type=int, default=0)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--n-slots-show", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=16.0); ap.add_argument("--anticipation-sec", type=float, default=2.0)
    args = ap.parse_args()
    TUB = 2; GRID = args.img_size // 16; GP = GRID * GRID; T_SLOTS = args.frames // TUB
    device = torch.device("cuda")

    ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    cand = [r for r in ds.rows if abs(float(r["context_sec"]) - args.context_sec) < 1e-6]
    if args.video_id:
        cand = [r for r in cand if str(r["video_id"]) == args.video_id] or cand
    row = cand[0]; vid = str(row["video_id"])
    fi = [int(x) for x in T._parse_int_list(row["frame_indices"])]
    stride = (fi[1] - fi[0]) if len(fi) > 1 else args.src_fps // args.fps
    end = fi[-1]
    idxf = np.array([end - (args.frames - 1 - i) * stride for i in range(args.frames)], dtype=np.int64)
    pid = vid.split("_")[0] if "_" in vid else vid.split("-")[0]
    vpath = Path(args.video_root) / pid / f"{vid}.MP4"
    if not vpath.exists():
        vpath = next(Path(args.video_root).rglob(f"{vid}.MP4"))
    vr = VideoReader(str(vpath), ctx=cpu(0), num_threads=1, width=args.img_size, height=args.img_size)
    idxf = np.clip(idxf, 0, len(vr) - 1)
    rgb = vr.get_batch(idxf.tolist()).asnumpy()   # (frames,H,W,C) uint8
    del vr

    base, _ = build_finetuned_20(device, max_frames=args.frames, fps=args.fps, img_size=args.img_size,
                                 checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                 pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    enlarge_predictor_budget(base, T_SLOTS * GP, GP)
    clip = torch.from_numpy(np.ascontiguousarray(rgb)).permute(3, 0, 1, 2).contiguous().unsqueeze(0)
    clip = clip.to(device).float().div_(255.0).sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x_full = base.encoder(clip)
    ant = torch.full((1,), float(args.anticipation_sec), device=device)
    kept = kept_indices(args.strategy, base, x_full, clip, ant, args.keep_count, GP, args.calib_path, args.score_block)
    kept_set = set(int(i) for i in kept.cpu().tolist())
    N = T_SLOTS * GP
    n_kept = len(kept_set); n_drop = N - n_kept
    # per-slot kept-patch counts (how the budget is spread over time)
    slot_keep = np.array([sum(1 for p in range(GP) if (s * GP + p) in kept_set) for s in range(T_SLOTS)])

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    show_slots = np.linspace(0, T_SLOTS - 1, args.n_slots_show).round().astype(int)
    cell = args.img_size // GRID
    fig, axes = plt.subplots(1, len(show_slots), figsize=(2.5 * len(show_slots), 3.0))
    if len(show_slots) == 1:
        axes = [axes]
    for ax, s in zip(axes, show_slots):
        frame = rgb[min(s * TUB, args.frames - 1)].copy()
        for p in range(GP):
            if (s * GP + p) not in kept_set:            # dropped -> black square
                h, w = p // GRID, p % GRID
                frame[h * cell:(h + 1) * cell, w * cell:(w + 1) * cell] = 0
        ax.imshow(frame); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slot {s}/{T_SLOTS-1}\nkept {int(slot_keep[s])}/{GP}", fontsize=8)
    fig.suptitle(f"EGTEA {vid}  strategy={args.strategy}  keep={args.keep_count}/{N} "
                 f"({100*n_kept/N:.0f}% kept)  black=DROPPED patch", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    figpath = out / f"drop_{args.strategy}_{vid}_{T_SLOTS}slots.png"
    fig.savefig(figpath, dpi=140, bbox_inches="tight"); plt.close(fig)
    np.save(out / f"slot_keep_{args.strategy}_{vid}.npy", slot_keep)
    print(f"[{args.strategy}] {vid} kept={n_kept} dropped={n_drop} "
          f"slot_keep(min/mean/max)={slot_keep.min()}/{slot_keep.mean():.1f}/{slot_keep.max()} -> {figpath}", flush=True)


if __name__ == "__main__":
    main()
