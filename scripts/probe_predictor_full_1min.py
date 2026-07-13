#!/usr/bin/env python
"""Confirm: with num_patches lifted, the predictor takes the FULL 1-min encoder output (NO pruning).

One sample. Encoder(480 frames) -> 61440 context tokens -> predictor with masks_x = real positions
0..61439, masks_y = the 1s-future target at 61440.. . Stock predictor.num_patches=8192 would make the
target-token gather out-of-range; we lift it (config buffer, no new params under RoPE). Reports: runs?
output finite? shape, wall-time, peak GPU mem — for encoder and for the full-context predictor step.
"""
import os
import sys
import time

import numpy as np
import torch

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/path/to/VJEPA2-EXP")
import app.hdepic_lora_action_anticipation.train_vjepa_prune_anticipation as M  # noqa: E402

NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "480"))
USE_REAL = os.environ.get("REAL_CLIP", "1") == "1"


def real_clip(device):
    """Decode one real HD-EPIC clip (the exact window the pipeline uses), else fall back to noise."""
    import csv, glob
    from decord import VideoReader, cpu
    rows = list(csv.DictReader(open(f"{SHARED}/data/hdepic_vjepa_annotations/phd_split/HD_EPIC_test_vjepa.csv", newline="")))
    for r in rows:
        path = f"{SHARED}/data/hdepic_vjepa_videos/{r['participant_id']}/{r['video_id']}.MP4"
        if not os.path.exists(path):
            continue
        vr = VideoReader(path, num_threads=1, ctx=cpu(0), width=256, height=256)
        win = np.clip(M.compute_clip_window(int(r["start_frame"]), vr.get_avg_fps(), NUM_FRAMES, 8.0), 0, len(vr) - 1)
        if len(win) < NUM_FRAMES:
            continue
        f = vr.get_batch(win).asnumpy()
        clip = torch.from_numpy(f).permute(3, 0, 1, 2).float().div(255).sub(M.IMAGENET_MEAN).div(M.IMAGENET_STD)
        print(f"  real clip: {r['video_id']} start_frame={r['start_frame']} ('{r.get('narration','?')}')", flush=True)
        return clip.unsqueeze(0).to(device)
    return None


def main():
    dev = torch.device("cuda")
    model = M.build_model(dev, NUM_FRAMES, 8, 256, f"{SHARED}/checkpoints/vitl.pt")
    enc, prd = model.encoder, model.predictor
    gp = int(model.grid_size) ** 2
    tube = int(model.tubelet_size)
    n_full = (NUM_FRAMES // tube) * gp
    stock_cap = prd.num_patches
    anticip_steps = int(round(1.0 * int(model.frames_per_second) / tube))
    n_pred = gp * anticip_steps
    max_tgt = n_full + n_pred
    new_cap = ((max_tgt // gp) + 8) * gp
    prd.num_patches = new_cap
    print(f"[cfg] frames={NUM_FRAMES} -> n_full={n_full} ctx tokens | n_pred={n_pred} | "
          f"max target idx={max_tgt} | num_patches {stock_cap} -> {new_cap}", flush=True)

    clip = real_clip(dev) if USE_REAL else None
    if clip is None:
        print("  (using random clip)", flush=True)
        clip = torch.randn(1, 3, NUM_FRAMES, 256, 256, device=dev)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t0 = time.time(); x = enc(clip); torch.cuda.synchronize(); t_enc = time.time() - t0
        print(f"[encoder] out {tuple(x.shape)} finite={bool(torch.isfinite(x).all())} | {t_enc:.1f}s "
              f"| peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

        ctx_pos = torch.arange(n_full, device=dev).unsqueeze(0)
        tgt_pos = torch.arange(n_pred, device=dev).unsqueeze(0) + n_full
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        pred = prd(x, masks_x=ctx_pos, masks_y=tgt_pos)
        if isinstance(pred, tuple):
            pred = pred[0]
        torch.cuda.synchronize(); t_prd = time.time() - t0
    print(f"[predictor FULL-CONTEXT] in {n_full} ctx + {n_pred} tgt tokens -> out {tuple(pred.shape)} "
          f"finite={bool(torch.isfinite(pred).all())} | {t_prd:.1f}s "
          f"| peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    print("[result] PREDICTOR ACCEPTS THE FULL 1-MIN CONTEXT (no pruning) once num_patches is lifted.",
          flush=True)


if __name__ == "__main__":
    main()
