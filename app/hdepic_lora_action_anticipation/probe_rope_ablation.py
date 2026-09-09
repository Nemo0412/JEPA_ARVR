#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Causal probe #1: is RoPE the cause of the CORNER sink?
# Runs the real V-JEPA 2.1 forward with rotation DISABLED in every block and
# measures the last-block corner sink vs RoPE-on. If corners collapse to ~uniform
# with RoPE off, RoPE causes the corner location; if they persist, another
# positional source (e.g. patch-embed borders) does. Slurm only.
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(CODE_ROOT, "vjepa2"), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import FpsSubsampledStreamMTPDataset  # noqa: E402
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import build_base_21, HeadAttnCapture21  # noqa: E402
from app.hdepic_lora_action_anticipation.vjepa_testtime_register import make_clip_iter  # noqa: E402


class RoPEAblate:
    """Replace a 2.1 RoPEAttention.forward with a NO-rotation version (real SDPA on
    unrotated q,k). If capture, also record per-head received attention."""
    def __init__(self, attn, capture=False, chunk=256):
        self._m = attn; self._orig = attn.forward
        self.importance = None
        m = attn; cap = self
        def fwd(x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False):
            B, N, C = x.size()
            qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]           # NO RoPE
            if capture:
                with torch.no_grad():
                    imp = torch.zeros(B, m.num_heads, N, device=x.device, dtype=torch.float32)
                    for ci in range(0, N, chunk):
                        qc = q[:, :, ci:ci + chunk, :]
                        imp += ((qc @ k.transpose(-2, -1)) * m.scale).softmax(-1).sum(dim=2).float()
                    cap.importance = imp
            with torch.backends.cuda.sdp_kernel():
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=m.is_causal)
            y = y.transpose(1, 2).reshape(B, N, C)
            y = m.proj(y); y = m.proj_drop(y)
            return (y, None)
        m.forward = fwd
    def remove(self):
        self._m.forward = self._orig


def corner_stats(imp, grid, gp):
    H, N = imp.shape
    slots = N // gp
    sp = imp[:, :slots * gp].reshape(H, slots, grid, grid).sum(axis=1)
    sp = sp / (sp.sum(axis=(1, 2), keepdims=True) + 1e-12)
    uni = 1.0 / (grid * grid)
    b = max(1, grid // 8)
    cb = np.array([(sp[h, :b, :b].sum() + sp[h, :b, -b:].sum() + sp[h, -b:, :b].sum() + sp[h, -b:, -b:].sum())
                   / (4 * b * b) / uni for h in range(H)])
    denom = imp.sum(axis=1, keepdims=True) + 1e-12
    peak = ((imp / denom) / uni).max()
    return float(cb.max()), float(peak), sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=4.0)
    ap.add_argument("--n-eval", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda")
    base, _ = build_base_21(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size, checkpoint=args.checkpoint)
    enc = base.encoder
    grid = int(enc.blocks[-1].attn.grid_size); gp = grid * grid
    val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(val_ds.rows)) if abs(float(val_ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6]
    ev = rows[: args.n_eval]
    print(f"[data] grid={grid} eval={len(ev)}", flush=True)

    def run(cap):
        c_all, p_all, sps = [], [], []
        for clips in make_clip_iter(val_ds, ev, device):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc(clips)
            imp = cap.importance[0].float().cpu().numpy()
            cb, pk, sp = corner_stats(imp, grid, gp)
            c_all.append(cb); p_all.append(pk); sps.append(sp)
        return float(np.mean(c_all)), float(np.mean(p_all)), np.mean(sps, axis=0)

    # RoPE ON (normal capture)
    cap_on = HeadAttnCapture21(enc.blocks[-1].attn)
    on_c, on_p, on_sp = run(cap_on)
    cap_on.remove()

    # RoPE OFF: ablate rotation in EVERY block; capture on the last
    abls = [RoPEAblate(enc.blocks[i].attn, capture=(i == len(enc.blocks) - 1)) for i in range(len(enc.blocks))]
    off_c, off_p, off_sp = run(abls[-1])
    for a in abls:
        a.remove()

    report = {"model": "vjepa2.1_base", "grid": grid, "n_eval": len(ev),
              "rope_on": {"corner_block_over_uniform": round(on_c, 2), "patch_sink_peak": round(on_p, 2)},
              "rope_off": {"corner_block_over_uniform": round(off_c, 2), "patch_sink_peak": round(off_p, 2)},
              "verdict": ("RoPE causes the corner sink (collapses with RoPE off)"
                          if off_c < 1.4 else
                          "corner sink persists without RoPE -> another positional source")}
    outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp.with_suffix(".on_spatial.npy"), on_sp.astype(np.float32))
    np.save(outp.with_suffix(".off_spatial.npy"), off_sp.astype(np.float32))
    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
