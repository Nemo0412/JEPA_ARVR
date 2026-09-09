#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Offline calibration of the predictor block-0 attention pattern.
# Advisor's ask: instead of the ONLINE per-sample predictor-score prune (two-pass), characterize
# the predictor first-layer received-attention pattern on a CALIBRATION set and make a FIXED,
# position-indexed keep/drop decision offline (cheaper; justified because the sink is
# position-anchored, findings 15g). Mirrors the loss_aware offline-config idea.
#
# Runs the full encoder->predictor (no prune) on M calibration clips at 64-slot (16 s) context,
# captures predictor block `--block` received attention (all queries, head-summed = the finding-15
# score) over the context tokens, and accumulates the MEAN per absolute position -> a [64, 256]
# map. Saved as .npy + JSON metadata; consumed at eval by the `pred_offline_{high,low}` strategy.
# Slurm only.
from __future__ import annotations

import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np, torch

CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(CODE_ROOT, "vjepa2"), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (  # noqa: E402
    FpsSubsampledStreamMTPDataset, enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import (  # noqa: E402
    build_finetuned_20, HeadAttnCapture20,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--calib-csv", required=True)
    ap.add_argument("--video-root", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoder-lora", default=None); ap.add_argument("--predictor-lora", default=None)
    ap.add_argument("--init-from-ckpt", default=None)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--n-clips", type=int, default=512)
    ap.add_argument("--context-sec", type=float, default=16.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    GRID = args.img_size // 16; GP = GRID * GRID
    device = torch.device("cuda")

    ds = FpsSubsampledStreamMTPDataset(args.calib_csv, args.video_root, args.img_size,
                                       src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(ds.rows))
            if abs(float(ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6]
    order = sorted(rows, key=lambda i: hashlib.md5(
        f"{args.seed}|{ds.rows[i]['video_id']}|{ds.rows[i].get('frame_indices','')}".encode()).hexdigest())
    picked = order[: args.n_clips]
    print(f"[calib] {len(picked)} clips @ context={args.context_sec}s from {args.calib_csv}", flush=True)

    base, _ = build_finetuned_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size,
                                 checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                 pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    enlarge_predictor_budget(base, (args.max_frames // int(base.tubelet_size)) * GP, GP)
    model = T.PrunedAnticipativeModel(base, None, prune_threshold=10 ** 9).to(device); model.eval()
    predictor = base.predictor
    blk = args.block if args.block >= 0 else len(predictor.predictor_blocks) + args.block
    tub = int(base.tubelet_size)
    n_pred = int(base.grid_size ** 2 * (base.num_output_frames // tub))
    ph = int(predictor.predictor_blocks[blk].attn.num_heads)

    mean = T.IMAGENET_MEAN.to(device); std = T.IMAGENET_STD.to(device)
    acc = None; n_used = 0; T_SLOTS = args.max_frames // tub
    for k, idx in enumerate(picked):
        try:
            batch = T.collate_stream([ds[idx]])
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {idx}: {e}", flush=True); continue
        clip = batch["clip"].to(device).float().div_(255.0).sub_(mean).div_(std)
        ant = torch.full((clip.size(0),), float(args.anticipation_sec), device=device)
        cap = HeadAttnCapture20(predictor.predictor_blocks[blk].attn)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(clip, ant)
        cap.remove()
        imp = cap.importance[0].float().cpu().numpy()   # (H, N_total)
        N_ctx = imp.shape[1] - n_pred
        if N_ctx != T_SLOTS * GP:
            print(f"  [skip] {idx}: N_ctx={N_ctx} != {T_SLOTS*GP}", flush=True); continue
        summ = imp.sum(axis=0)[:N_ctx]                  # head-summed received attn over context
        acc = summ if acc is None else acc + summ
        n_used += 1
        if n_used % 50 == 0:
            print(f"  {n_used}/{len(picked)}", flush=True)
    if n_used == 0:
        raise SystemExit("no usable calib clips")
    mean_map = (acc / n_used).reshape(T_SLOTS, GP)      # [64, 256] mean per absolute position

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    npy_path = out / f"calib_predblk{blk}_map_{T_SLOTS}x{GP}.npy"
    np.save(npy_path, mean_map.astype(np.float32))
    # quick pattern summary (corner vs other, per-slot recency)
    flat = mean_map.reshape(-1); uni = float(flat.mean())
    corners = [0, GRID - 1, GP - GRID, GP - 1]
    corner_mean = float(mean_map[:, corners].mean()); other_mean = float(
        mean_map[:, [j for j in range(GP) if j not in corners]].mean())
    slot_mass = mean_map.sum(axis=1); slot_mass = (slot_mass / slot_mass.sum()).tolist()
    meta = {"site": "predictor", "block": blk, "n_used": n_used, "slots": T_SLOTS, "gp": GP,
            "uniform": uni, "corner_mean": round(corner_mean, 4), "other_mean": round(other_mean, 4),
            "corner_over_other": round(corner_mean / (other_mean + 1e-12), 3),
            "recent8_slot_mass": round(float(np.sum(slot_mass[-8:])), 4),
            "oldest8_slot_mass": round(float(np.sum(slot_mass[:8])), 4),
            "map_path": str(npy_path), "context_sec": args.context_sec,
            "anticipation_sec": args.anticipation_sec, "calib_csv": args.calib_csv}
    (out / f"calib_predblk{blk}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print("wrote", npy_path, flush=True)


if __name__ == "__main__":
    main()
