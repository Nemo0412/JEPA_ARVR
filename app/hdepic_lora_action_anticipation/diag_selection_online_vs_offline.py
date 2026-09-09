#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Diagnostic: WHY offline-calibrated pruning ≳ online per-sample.
# Hypothesis (not a bug): the offline decision is topK(mean score) while online is per-sample
# topK(sample score); the two differ because top-K is a nonlinear argmax. Averaging first
# preserves the recency gradient (keeps whole recent slots); per-sample argmax lets the sharp
# spatial corner sink win within each slot. This dumps, over M clips, the online-vs-offline
# kept-set overlap + per-slot keep + recency/corner mass to show exactly that. No classifier.
from __future__ import annotations
from app.hdepic_lora_action_anticipation.share_paths import DATA_ROOT as SHARE_DATA_ROOT, VJEPA_ROOT as SHARE_VJEPA_ROOT

import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np, torch

CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (str(SHARE_VJEPA_ROOT), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (  # noqa: E402
    FpsSubsampledStreamMTPDataset, enlarge_predictor_budget, PredictorScorePrunedModel,
)
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import (  # noqa: E402
    build_finetuned_20, HeadAttnCapture20,
)

CORNERS4 = None  # set after grid known


def per_clip_score(base, x_full, ant, block, gp):
    """head-summed predictor-blk0 received attention over the context (B=1) -> np [N]."""
    from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import HeadAttnCapture20
    B, N, _ = x_full.size()
    ctxt = torch.arange(N, device=x_full.device).unsqueeze(0)
    steps = (ant * base.frames_per_second / base.tubelet_size).to(torch.int64)
    skip = N + int(base.grid_size ** 2) * steps
    N_pred = int(base.grid_size ** 2 * (base.num_output_frames // base.tubelet_size))
    tgt = torch.arange(N_pred, device=x_full.device).unsqueeze(0) + skip.unsqueeze(1)
    cap = HeadAttnCapture20(base.predictor.predictor_blocks[block].attn)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base.predictor(x_full, masks_x=ctxt, masks_y=tgt)
    cap.remove()
    return cap.importance[0, :, :N].sum(0).float().cpu().numpy()   # (N,)


def keep_stats(score, K, gp, grid, slots, mode="high"):
    order = np.argsort(score)[::-1] if mode == "high" else np.argsort(score)
    kept = np.zeros(len(score), dtype=bool); kept[order[:K]] = True
    per_slot = kept.reshape(slots, gp).sum(1)                      # kept patches per slot
    full_slots = int((per_slot == gp).sum())                       # slots kept entirely (recency signature)
    recent_half = float(kept.reshape(slots, gp)[slots // 2:].sum() / K)   # frac of kept in recent half
    corners = [0, grid - 1, gp - grid, gp - 1]
    cmask = np.zeros(gp, dtype=bool); cmask[corners] = True
    corner_frac = float(kept.reshape(slots, gp)[:, cmask].sum() / K)
    return kept, per_slot, full_slots, recent_half, corner_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calib-path", required=True)
    ap.add_argument("--encoder-lora", default=None); ap.add_argument("--predictor-lora", default=None)
    ap.add_argument("--init-from-ckpt", default=None)
    ap.add_argument("--block", type=int, default=0); ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--n-clips", type=int, default=64); ap.add_argument("--context-sec", type=float, default=16.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    grid = args.img_size // 16; gp = grid * grid; slots = args.max_frames // 2
    K = (args.keep_count // gp) * gp
    device = torch.device("cuda")

    offline_map = np.load(args.calib_path).astype(np.float32).reshape(-1)   # [slots*gp]
    _, off_per_slot, off_full, off_recent, off_corner = keep_stats(offline_map, K, gp, grid, slots, "high")
    off_kept_set = set(np.argsort(offline_map)[::-1][:K].tolist())

    ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(ds.rows)) if abs(float(ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6]
    order = sorted(rows, key=lambda i: hashlib.md5(
        f"{args.seed}|{ds.rows[i]['video_id']}|{ds.rows[i].get('frame_indices','')}".encode()).hexdigest())
    picked = order[: args.n_clips]

    base, _ = build_finetuned_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size,
                                 checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                 pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    enlarge_predictor_budget(base, slots * gp, gp)
    base.eval()
    mean = T.IMAGENET_MEAN.to(device); std = T.IMAGENET_STD.to(device)

    jaccard, on_full, on_recent, on_corner = [], [], [], []
    on_per_slot_acc = np.zeros(slots); score_acc = np.zeros(slots * gp); n = 0
    for idx in picked:
        try:
            batch = T.collate_stream([ds[idx]])
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {idx}: {e}", flush=True); continue
        clip = batch["clip"].to(device).float().div_(255.0).sub_(mean).div_(std)
        ant = torch.full((1,), float(args.anticipation_sec), device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x_full = base.encoder(clip)
        if x_full.shape[1] != slots * gp:
            print(f"[skip] {idx}: N={x_full.shape[1]}", flush=True); continue
        s = per_clip_score(base, x_full, ant, args.block, gp)
        score_acc += s; n += 1
        on_kept, on_ps, on_f, on_r, on_c = keep_stats(s, K, gp, grid, slots, "high")
        on_per_slot_acc += on_ps
        on_full.append(on_f); on_recent.append(on_r); on_corner.append(on_c)
        on_set = set(np.where(on_kept)[0].tolist())
        inter = len(on_set & off_kept_set); union = len(on_set | off_kept_set)
        jaccard.append(inter / union)
    if n == 0:
        raise SystemExit("no clips")

    mean_score = score_acc / n                                   # mean of per-sample scores over val clips
    # calibration sanity: does the val mean-score match the (train-calibrated) offline map?
    corr = float(np.corrcoef(mean_score, offline_map)[0, 1])
    # topK(mean) vs the offline map's topK (both are topK-of-a-mean -> should agree strongly)
    meanK = set(np.argsort(mean_score)[::-1][:K].tolist())
    jacc_meanK_off = len(meanK & off_kept_set) / len(meanK | off_kept_set)

    rep = {
        "n_clips": n, "K": K, "slots": slots, "gp": gp,
        "jaccard_online_vs_offline_mean": round(float(np.mean(jaccard)), 3),
        "jaccard_online_vs_offline_std": round(float(np.std(jaccard)), 3),
        "online_full_slots_kept_mean": round(float(np.mean(on_full)), 2),
        "offline_full_slots_kept": off_full,
        "online_recent_half_mass_mean": round(float(np.mean(on_recent)), 3),
        "offline_recent_half_mass": round(off_recent, 3),
        "online_corner_frac_mean": round(float(np.mean(on_corner)), 4),
        "offline_corner_frac": round(off_corner, 4),
        "val_meanscore_vs_train_calib_corr": round(corr, 4),
        "jaccard_topK-of-valmean_vs_offline": round(jacc_meanK_off, 3),
        "interpretation": ("offline=topK(mean) keeps whole recent slots (full_slots high, recency mass high); "
                           "online=per-sample topK spreads on the corner sink (full_slots~0, more corner mass); "
                           "low online-vs-offline Jaccard + high val-mean~train-calib corr => not a bug, it is "
                           "topK(mean)!=mean(topK)."),
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "diag_online_vs_offline.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)

    # figure: per-slot kept-count, online-avg vs offline
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(slots)
    ax[0].plot(x, on_per_slot_acc / n, "o-", ms=3, label="online (per-sample avg)")
    ax[0].plot(x, off_per_slot, "s-", ms=3, label="offline (calibrated fixed)")
    ax[0].axhline(gp, color="gray", ls=":", label="full slot (256)")
    ax[0].set_xlabel("time slot (0=oldest, 63=now)"); ax[0].set_ylabel("kept patches / slot")
    ax[0].set_title("Kept-per-slot: online spreads, offline keeps recent slots whole"); ax[0].legend(fontsize=8)
    ax[1].hist(jaccard, bins=20, color="tab:purple", alpha=0.8)
    ax[1].axvline(np.mean(jaccard), color="k", ls="--", label=f"mean={np.mean(jaccard):.2f}")
    ax[1].set_xlabel("Jaccard(online kept, offline kept) per clip"); ax[1].set_ylabel("clips")
    ax[1].set_title(f"Online vs offline keep-set overlap (corr val-mean~calib={corr:.2f})"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "diag_online_vs_offline.png", dpi=140); plt.close(fig)
    print("wrote", out / "diag_online_vs_offline.png", flush=True)


if __name__ == "__main__":
    main()
