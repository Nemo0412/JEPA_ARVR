#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Predictor port of ll's L23 all-heads attn-mass [T x 256] figure.
# Encoder counterpart: probe_all_heads_mass_Tx256.py (finding 14, ENCODER last block).
# Here we run the FULL encoder->predictor path (no pruning) and capture the PREDICTOR's
# block-0 received-attention over the CONTEXT tokens, laid out [T_slots x 256 spatial],
# LINEAR, one row per head, corners at spatial 0/15/240/255. Same single-clip window
# builder as the encoder probe so the two figures are directly comparable.
#
# T_slots = frames // tubelet:
#   - 32 frames  -> 16 slots (4 s @ 8 fps): our post-KV-prune memory budget.
#   - 128 frames -> 64 slots (16 s @ 8 fps): the pre-prune streaming input. 64 slots
#     OVERFLOWS the predictor's pretrained num_patches (= num_frames/tubelet*grid^2 =
#     32*256 = 8192); enlarge_predictor_budget raises it (RoPE-only, no new params, the
#     predictor RoPE EXTRAPOLATES past its trained 32-slot depth -> OOD, by design here).
# Real V-JEPA 2.0 finetuned. Slurm only.
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
from decord import VideoReader, cpu

CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(CODE_ROOT, "vjepa2"), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (  # noqa: E402
    FpsSubsampledStreamMTPDataset, enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import (  # noqa: E402
    build_finetuned_20, HeadAttnCapture20,
)

CORNERS = {"TL": 0, "TR": 15, "BL": 240, "BR": 255}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoder-lora", default=None); ap.add_argument("--predictor-lora", default=None)
    ap.add_argument("--init-from-ckpt", default=None)
    ap.add_argument("--video-id", default=None, help="target EGTEA video_id (default: first 10s row)")
    ap.add_argument("--block", type=int, default=0, help="predictor block index to capture (0=first)")
    ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=10.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    args = ap.parse_args()
    TUBELET = 2
    T_SLOTS = args.frames // TUBELET
    GRID = args.img_size // 16; GP = GRID * GRID

    # ---- pick a row, build a `frames`-frame window @ fps ending at its last frame ----
    ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    cand = [r for r in ds.rows if abs(float(r["context_sec"]) - args.context_sec) < 1e-6]
    if args.video_id:
        cand = [r for r in cand if str(r["video_id"]) == args.video_id] or cand
    row = cand[0]
    vid = str(row["video_id"])
    fi = [int(x) for x in T._parse_int_list(row["frame_indices"])]
    stride = (fi[1] - fi[0]) if len(fi) > 1 else args.src_fps // args.fps
    end = fi[-1]
    idx = np.array([end - (args.frames - 1 - i) * stride for i in range(args.frames)], dtype=np.int64)

    pid = vid.split("_")[0] if "_" in vid else vid.split("-")[0]
    vpath = Path(args.video_root) / pid / f"{vid}.MP4"
    if not vpath.exists():
        vpath = next(Path(args.video_root).rglob(f"{vid}.MP4"))
    vr = VideoReader(str(vpath), ctx=cpu(0), num_threads=1, width=args.img_size, height=args.img_size)
    idx = np.clip(idx, 0, len(vr) - 1)
    frames = vr.get_batch(idx.tolist()).asnumpy()   # (frames,H,W,C)
    del vr
    clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).contiguous().unsqueeze(0)  # 1,C,F,H,W
    print(f"[data] vid={vid} frames={args.frames} stride={stride} span=[{idx[0]},{idx[-1]}] slots={T_SLOTS} GP={GP}", flush=True)

    device = torch.device("cuda")
    base, _ = build_finetuned_20(device, max_frames=args.frames, fps=args.fps, img_size=args.img_size,
                                 checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                 pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    # No pruning: raise the predictor position budget so the full (possibly OOD) context fits.
    num_tokens_full = T_SLOTS * GP
    enlarge_predictor_budget(base, num_tokens_full, GP)
    model = T.PrunedAnticipativeModel(base, None, prune_threshold=10 ** 9).to(device); model.eval()
    predictor = base.predictor
    n_blocks = len(predictor.predictor_blocks)
    blk = args.block if args.block >= 0 else n_blocks + args.block
    tub = int(base.tubelet_size)
    n_pred = int(base.grid_size ** 2 * (base.num_output_frames // tub))
    ph = int(predictor.predictor_blocks[blk].attn.num_heads)
    print(f"[model] predictor blocks={n_blocks} capture blk={blk} heads={ph} n_pred={n_pred} "
          f"num_patches={int(getattr(predictor,'num_patches',0))}", flush=True)

    clip = clip.to(device).float().div_(255.0).sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
    ant = torch.full((clip.size(0),), float(args.anticipation_sec), device=device)
    cap = HeadAttnCapture20(predictor.predictor_blocks[blk].attn)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(clip, ant)
    cap.remove()
    imp = cap.importance[0].float().cpu().numpy()   # (H, N_total = N_ctx + n_pred)
    H, N_total = imp.shape
    N_ctx = N_total - n_pred
    assert N_ctx == T_SLOTS * GP, (N_ctx, T_SLOTS, GP, N_total, n_pred)
    target_mass = float(imp[:, N_ctx:].sum() / (imp.sum() + 1e-12))
    M = imp[:, :N_ctx].reshape(H, T_SLOTS, GP)      # (H, T, 256) — context only

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"pred_blk{blk}_all_heads_mass_Hx{T_SLOTS}x256.npy", M.astype(np.float32))

    # summary: per-head corner mean vs other, ratio (ll's format), plus target-mass
    lines = [f"video={vid} PRED blk{blk} frames={args.frames} fps={args.fps} dur≈{args.frames/args.fps:.1f}s "
             f"slots={T_SLOTS} N_ctx={N_ctx} n_pred={n_pred} target_mass={target_mass:.4f} | I_j [T x 256] LINEAR", "",
             f"{'head':>4s} {'TL0':>8s} {'TR15':>8s} {'BL240':>8s} {'BR255':>8s} {'other':>8s} {'c/o':>7s}"]
    for h in range(H):
        mh = M[h]
        cvals = [float(mh[:, CORNERS[n]].mean()) for n in ("TL", "TR", "BL", "BR")]
        mask = np.ones(GP, dtype=bool)
        for i in CORNERS.values():
            mask[i] = False
        other = float(mh[:, mask].mean())
        lines.append(f"{h:4d} {cvals[0]:8.3f} {cvals[1]:8.3f} {cvals[2]:8.3f} {cvals[3]:8.3f} "
                     f"{other:8.3f} {float(np.mean(cvals)/(other+1e-12)):7.2f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(
        {"video": vid, "site": "predictor", "block": blk, "frames": args.frames, "slots": T_SLOTS,
         "N_ctx": N_ctx, "n_pred": n_pred, "target_mass": round(target_mass, 4),
         "num_patches": int(getattr(predictor, "num_patches", 0))}, indent=2))
    print("\n".join(lines), flush=True)

    yt = [0, T_SLOTS // 4, T_SLOTS // 2, 3 * T_SLOTS // 4, T_SLOTS - 1]
    # combined: one row per head
    fig, axes = plt.subplots(H, 1, figsize=(20, 2.2 * H + 1.0), sharex=True)
    if H == 1:
        axes = [axes]
    for h in range(H):
        mh = M[h]; vmax = float(np.percentile(mh, 99))
        im = axes[h].imshow(mh, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
        for name, i in CORNERS.items():
            axes[h].axvline(i, color="cyan", lw=0.6, alpha=0.8)
            if h == 0:
                axes[h].text(i, -1.5, name, color="cyan", ha="center", fontsize=8, fontweight="bold")
        axes[h].set_yticks(yt)
        axes[h].set_ylabel(f"h{h}\nt", fontsize=9, rotation=0, labelpad=22, va="center")
        cb = plt.colorbar(im, ax=axes[h], fraction=0.01, pad=0.006); cb.ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("spatial index 0..255  (TL=0, TR=15, BL=240, BR=255)", fontsize=11)
    fig.suptitle(f"EGTEA {vid}  PREDICTOR blk{blk}  {args.frames}f@{args.fps}fps≈{args.frames/args.fps:.0f}s  "
                 f"context attn mass I_j [{T_SLOTS} time × 256 spatial] LINEAR (target_mass={target_mass:.3f})\n"
                 f"one row = one head; cyan = corners",
                 fontsize=13, y=0.997)
    fig.tight_layout(rect=[0.02, 0.01, 1, 0.98])
    fig.savefig(out / f"pred_blk{blk}_all_heads_mass_{T_SLOTS}x256_linear.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    per = out / "per_head"; per.mkdir(exist_ok=True)
    for h in range(H):
        mh = M[h]; vmax = float(np.percentile(mh, 99))
        fig, ax = plt.subplots(figsize=(18, 4.5))
        im = ax.imshow(mh, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
        for name, i in CORNERS.items():
            ax.axvline(i, color="cyan", lw=0.7, alpha=0.85); ax.text(i, -1.2, name, color="cyan", ha="center", fontsize=8)
        ax.set_yticks(yt); ax.set_ylabel("time slot t"); ax.set_xlabel("spatial index 0..255")
        ax.set_title(f"EGTEA PRED blk{blk} h{h} context mass [{T_SLOTS}×256] linear  {args.frames}f≈{args.frames/args.fps:.0f}s")
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="I_j")
        fig.tight_layout(); fig.savefig(per / f"h{h:02d}_mass_{T_SLOTS}x256_linear.png", dpi=130, bbox_inches="tight")
        plt.close(fig); np.save(per / f"h{h:02d}_mass_{T_SLOTS}x256.npy", mh.astype(np.float32))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
