#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · ll-style overlay: WHERE do the high-attention-score tokens land on
# the image. Replicates ll's L23 "persistent patches on mid-frame" figure for EGTEA.
#
# Attention score = head-summed received attention imp[j] = Σ_head Σ_query softmax(q·k/√d)
# (== train_stream_mtp.TokenPruner._importance, the actual pruning score; verified rel 1.7e-7).
# We time-average it over the [T_slots × 256] context grid to get per-spatial-token importance,
# take the top-K spatial tokens, and draw them as cyan squares on the clip's MID-FRAME, with a
# legend listing their (row, col) grid positions. Two videos A vs B side by side (ll's layout).
#
# --site encoder : encoder last block (ll-faithful "attention score", default).
# --site predictor: predictor block --block over the context tokens (our finding-15 site).
# Single clip per video, same window builder as probe_all_heads_mass_Tx256.py. Slurm only.
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
from matplotlib.patches import Rectangle  # noqa: E402
from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (  # noqa: E402
    FpsSubsampledStreamMTPDataset, enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import (  # noqa: E402
    build_finetuned_20, HeadAttnCapture20,
)


def load_window(ds, video_root, video_id, context_sec, frames, fps, src_fps, img_size):
    cand = [r for r in ds.rows if abs(float(r["context_sec"]) - context_sec) < 1e-6]
    cand2 = [r for r in cand if str(r["video_id"]) == video_id]
    row = (cand2 or cand)[0]
    vid = str(row["video_id"])
    fi = [int(x) for x in T._parse_int_list(row["frame_indices"])]
    stride = (fi[1] - fi[0]) if len(fi) > 1 else src_fps // fps
    end = fi[-1]
    idx = np.array([end - (frames - 1 - i) * stride for i in range(frames)], dtype=np.int64)
    pid = vid.split("_")[0] if "_" in vid else vid.split("-")[0]
    vpath = Path(video_root) / pid / f"{vid}.MP4"
    if not vpath.exists():
        vpath = next(Path(video_root).rglob(f"{vid}.MP4"))
    vr = VideoReader(str(vpath), ctx=cpu(0), num_threads=1, width=img_size, height=img_size)
    idx = np.clip(idx, 0, len(vr) - 1)
    rgb = vr.get_batch(idx.tolist()).asnumpy()   # (frames,H,W,C) uint8
    del vr
    return vid, rgb


def importance_grid(model, base, rgb, args, device, gp, grid):
    """Run the model on one clip, return per-spatial-token time-averaged importance (grid,grid)
    and the head-summed [T_slots,gp] map (for persistence). Head-summed = the pruning score."""
    clip = torch.from_numpy(np.ascontiguousarray(rgb)).permute(3, 0, 1, 2).contiguous().unsqueeze(0)
    clip = clip.to(device).float().div_(255.0).sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
    if args.site == "encoder":
        attn = base.encoder.blocks[-1].attn
    else:
        blk = args.block if args.block >= 0 else len(base.predictor.predictor_blocks) + args.block
        attn = base.predictor.predictor_blocks[blk].attn
    cap = HeadAttnCapture20(attn)
    ant = torch.full((1,), float(args.anticipation_sec), device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(clip, ant)
    cap.remove()
    imp = cap.importance[0].float().cpu().numpy()          # (H, N_total)
    summ = imp.sum(axis=0)                                  # (N_total,) head-summed = pruning score
    if args.site == "predictor":
        tub = int(base.tubelet_size)
        n_pred = int(base.grid_size ** 2 * (base.num_output_frames // tub))
        summ = summ[: summ.shape[0] - n_pred]
    T_slots = summ.shape[0] // gp
    M = summ.reshape(T_slots, gp)                           # (T, 256)
    time_mean = M.mean(axis=0)                              # (256,) per-spatial importance
    return time_mean.reshape(grid, grid), M, T_slots


def pick_topk(time_grid, M, grid, k, persist_frac):
    """Top-k spatial tokens by time-mean importance, keeping only 'persistent' ones (present in
    >= persist_frac of time slots above the uniform level)."""
    gp = grid * grid
    flat = time_grid.reshape(-1)
    uniform = flat.mean()
    frac_above = (M > uniform).mean(axis=0)                 # (256,) fraction of slots above uniform
    order = np.argsort(flat)[::-1]
    chosen = [j for j in order if frac_above[j] >= persist_frac][:k]
    if len(chosen) < k:                                     # backfill if persistence too strict
        chosen += [j for j in order if j not in chosen][: k - len(chosen)]
    return [(int(j // grid), int(j % grid), float(flat[j]), float(frac_above[j])) for j in chosen]


def draw(ax, rgb_mid, picks, grid, img_size, title):
    ax.imshow(rgb_mid)
    cell = img_size / grid
    for (r, c, sc, fa) in picks:
        ax.add_patch(Rectangle((c * cell, r * cell), cell, cell, fill=False, edgecolor="cyan", lw=2.2))
    ax.set_xticks([]); ax.set_yticks([])
    pos = ", ".join(f"({r},{c})" for (r, c, _, _) in picks)
    ax.set_title(f"{title}\n{pos}", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoder-lora", default=None); ap.add_argument("--predictor-lora", default=None)
    ap.add_argument("--init-from-ckpt", default=None)
    ap.add_argument("--video-a", required=True); ap.add_argument("--video-b", required=True)
    ap.add_argument("--site", choices=["encoder", "predictor"], default="encoder")
    ap.add_argument("--block", type=int, default=0, help="predictor block (only for --site predictor)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--persist-frac", type=float, default=0.6)
    ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=10.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    args = ap.parse_args()
    GRID = args.img_size // 16; GP = GRID * GRID
    device = torch.device("cuda")

    ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    base, _ = build_finetuned_20(device, max_frames=args.frames, fps=args.fps, img_size=args.img_size,
                                 checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                 pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    if args.site == "predictor":
        enlarge_predictor_budget(base, (args.frames // int(base.tubelet_size)) * GP, GP)
    model = T.PrunedAnticipativeModel(base, None, prune_threshold=10 ** 9).to(device); model.eval()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    site_tag = args.site if args.site == "encoder" else f"predictor_blk{args.block}"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    summary = [f"site={site_tag} frames={args.frames} slots={args.frames//2} topk={args.topk} "
               f"persist_frac={args.persist_frac} | (row,col) score frac_above_uniform"]
    for ax, vid_req, lab in ((axes[0], args.video_a, "A"), (axes[1], args.video_b, "B")):
        vid, rgb = load_window(ds, args.video_root, vid_req, args.context_sec, args.frames,
                               args.fps, args.src_fps, args.img_size)
        tgrid, M, T_slots = importance_grid(model, base, rgb, args, device, GP, GRID)
        picks = pick_topk(tgrid, M, GRID, args.topk, args.persist_frac)
        mid = rgb[len(rgb) // 2]
        draw(ax, mid, picks, GRID, args.img_size, f"{lab} {vid}")
        np.save(out / f"{lab}_{vid}_time_grid_{site_tag}.npy", tgrid.astype(np.float32))
        summary.append(f"[{lab}] {vid}")
        for (r, c, sc, fa) in picks:
            summary.append(f"    ({r:2d},{c:2d})  score={sc:.3f}  frac_above={fa:.2f}")
        print(f"[{lab}] {vid} picks={[(r,c) for r,c,_,_ in picks]}", flush=True)

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor="none", markeredgecolor="cyan",
                      markeredgewidth=2.2, markersize=12,
                      label=f"top-{args.topk} attention-score tokens\n(persistent ≥{args.persist_frac:.0%} of slots)")]
    fig.legend(handles=handles, loc="lower center", fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"High attention-score tokens on mid-frame (cyan) — {site_tag}, {args.frames}f "
                 f"({args.frames//2} slots)  |  compare A vs B", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    figpath = out / f"overlay_{site_tag}_{args.frames}f.png"
    fig.savefig(figpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    (out / f"summary_{site_tag}_{args.frames}f.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)
    print("wrote", figpath, flush=True)


if __name__ == "__main__":
    main()
