#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · OFFLINE calibration of the sink-exclusion threshold.
# Runs the unmodified TokenPruner received-attention score on a calibration clip set
# and reports the distribution of imp/uniform (uniform=num_heads) so tau=mult*uniform
# can be chosen to separate the sink outliers from content -- WITHOUT touching val.
# Reports percentiles + how many tokens/slot exceed mult in {2,2.5,3,4,5,6}. Slurm only.
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(CODE_ROOT, "vjepa2"), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import FpsSubsampledStreamMTPDataset  # noqa: E402
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import build_finetuned_20  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True); ap.add_argument("--out-json", required=True)
    ap.add_argument("--encoder-lora", default=None); ap.add_argument("--predictor-lora", default=None)
    ap.add_argument("--init-from-ckpt", default=None)
    ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=10.0); ap.add_argument("--n-calib", type=int, default=60)
    args = ap.parse_args()

    device = torch.device("cuda")
    base, _ = build_finetuned_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size,
                                 checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                 pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    enc = base.encoder
    gp = int(base.grid_size ** 2)
    nheads = int(enc.blocks[-1].attn.num_heads)
    pruner = T.TokenPruner(enc, keep_count=gp, gp=gp)   # only need its received-attn capture
    val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(val_ds.rows)) if abs(float(val_ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6][: args.n_calib]
    mean = T.IMAGENET_MEAN.to(device); std = T.IMAGENET_STD.to(device)
    print(f"[calib] nheads={nheads} uniform={nheads} gp={gp} clips={len(rows)}", flush=True)

    ratios = []          # imp/uniform over all tokens/clips
    top_per_slot = []     # per-clip max ratio (the sink peak)
    mults = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    exceed_counts = {m: [] for m in mults}   # per-clip: #tokens with ratio>m, normalized per slot
    for i in rows:
        clip = T.collate_stream([val_ds[i]])["clip"].to(device).float().div_(255.0).sub_(mean).div_(std)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc(clip)
        imp = pruner._importance[0].float().cpu().numpy()   # (N,)
        r = imp / float(nheads)                              # ratio to uniform
        N = r.shape[0]; slots = max(1, N // gp)
        ratios.append(r)
        top_per_slot.append(float(r.max()))
        for m in mults:
            exceed_counts[m].append(float((r > m).sum()) / slots)
    pruner.remove()

    allr = np.concatenate(ratios)
    report = {
        "n_calib": len(rows), "num_heads": nheads, "uniform_level": nheads, "gp": gp,
        "ratio_percentiles": {p: round(float(np.percentile(allr, p)), 2)
                              for p in [50, 90, 99, 99.5, 99.9, 99.99]},
        "ratio_max": round(float(allr.max()), 2),
        "clip_peak_mean": round(float(np.mean(top_per_slot)), 2),
        "excluded_per_slot_at_mult": {str(m): round(float(np.mean(exceed_counts[m])), 2) for m in mults},
        "note": "tau = mult * uniform; pick mult that isolates the sink outlier cluster "
                "(~few tokens/slot) without cutting into the content bulk.",
    }
    outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
