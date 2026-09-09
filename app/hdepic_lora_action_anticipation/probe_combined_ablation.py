#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Causal probe: RoPE x border 2x2 factorial. Do RoPE-off AND
# border-neutralized TOGETHER collapse the corner sink (=> they jointly explain it),
# or does a residual softmax attention-sink persist? Real V-JEPA 2.1. Slurm only.
from __future__ import annotations
from app.hdepic_lora_action_anticipation.share_paths import DATA_ROOT as SHARE_DATA_ROOT, VJEPA_ROOT as SHARE_VJEPA_ROOT

import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (str(SHARE_VJEPA_ROOT), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch.nn.functional as F  # noqa: E402
from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import FpsSubsampledStreamMTPDataset  # noqa: E402
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import (  # noqa: E402
    build_base_20, build_base_21, build_finetuned_20, HeadAttnCapture20, HeadAttnCapture21,
)
from app.hdepic_lora_action_anticipation.vjepa_testtime_register import make_clip_iter  # noqa: E402
from app.hdepic_lora_action_anticipation.probe_rope_ablation import RoPEAblate as RoPEAblate21, corner_stats  # noqa: E402


class RoPEAblate20:
    """No-rotation replacement for a 2.0 RoPEAttention.forward (returns x, not a tuple)."""
    def __init__(self, attn, capture=False, chunk=256):
        self._m = attn; self._orig = attn.forward
        self.importance = None
        m = attn; cap = self
        def fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            B, N, C = x.size()
            qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]           # NO RoPE
            if capture:
                with torch.no_grad():
                    imp = torch.zeros(B, m.num_heads, N, device=x.device, dtype=torch.float32)
                    for ci in range(0, N, chunk):
                        imp += ((q[:, :, ci:ci + chunk, :] @ k.transpose(-2, -1)) * m.scale).softmax(-1).sum(dim=2).float()
                    cap.importance = imp
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=m.proj_drop_prob, is_causal=m.is_causal, attn_mask=attn_mask)
            x = x.transpose(1, 2).reshape(B, N, C)
            x = m.proj(x); x = m.proj_drop(x)
            return x
        m.forward = fwd
    def remove(self):
        self._m.forward = self._orig


def border_prehook(grid, gp, ring):
    def pre(mod, a, kw):
        x = a[0]
        B, N, D = x.shape
        slots = N // gp
        v = x.view(B, slots, grid, grid, D).clone()
        mask = torch.zeros(grid, grid, dtype=torch.bool, device=x.device)
        mask[:ring, :] = mask[-ring:, :] = mask[:, :ring] = mask[:, -ring:] = True
        mean_tok = v[:, :, ~mask, :].mean(dim=2, keepdim=True)
        v[:, :, mask, :] = mean_tok
        return (v.view(B, N, D),) + a[1:], kw
    return pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True); ap.add_argument("--out-json", required=True)
    ap.add_argument("--img-size", type=int, default=384); ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=4.0); ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--border", type=int, default=3)
    ap.add_argument("--variant", default="base21", choices=["base21", "base20", "finetuned20"])
    ap.add_argument("--encoder-lora", default=None); ap.add_argument("--predictor-lora", default=None)
    ap.add_argument("--init-from-ckpt", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.variant == "base21":
        base, kind = build_base_21(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size, checkpoint=args.checkpoint)
    elif args.variant == "base20":
        base, kind = build_base_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size, checkpoint=args.checkpoint)
    else:
        base, kind = build_finetuned_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size,
                                        checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                        pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    Ablate = RoPEAblate21 if kind == "21" else RoPEAblate20
    Capture = HeadAttnCapture21 if kind == "21" else HeadAttnCapture20
    enc = base.encoder
    grid = int(enc.blocks[-1].attn.grid_size); gp = grid * grid
    val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(val_ds.rows)) if abs(float(val_ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6]
    ev = rows[: args.n_eval]
    print(f"[data] grid={grid} eval={len(ev)} border={args.border}", flush=True)

    maps = {}   # cell -> (H, grid, grid) mean spatial

    def measure(rope_off, border_off, name):
        hooks = []
        if border_off:
            hooks.append(enc.blocks[0].register_forward_pre_hook(border_prehook(grid, gp, args.border), with_kwargs=True))
        if rope_off:
            abls = [Ablate(enc.blocks[i].attn, capture=(i == len(enc.blocks) - 1)) for i in range(len(enc.blocks))]
            cap = abls[-1]
        else:
            abls = []
            cap = Capture(enc.blocks[-1].attn)
        cs, ps, sps = [], [], []
        for clips in make_clip_iter(val_ds, ev, device):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc(clips)
            cb, pk, sp = corner_stats(cap.importance[0].float().cpu().numpy(), grid, gp)
            cs.append(cb); ps.append(pk); sps.append(sp)
        for a in abls:
            a.remove()
        if not rope_off:
            cap.remove()
        for h in hooks:
            h.remove()
        maps[name] = np.mean(sps, axis=0)   # (H, grid, grid)
        return {"corner_block": round(float(np.mean(cs)), 2), "patch_peak": round(float(np.mean(ps)), 2)}

    cells = {
        "rope_on__border_on": measure(False, False, "rope_on__border_on"),
        "rope_off__border_on": measure(True, False, "rope_off__border_on"),
        "rope_on__border_off": measure(False, True, "rope_on__border_off"),
        "rope_off__border_off": measure(True, True, "rope_off__border_off"),
    }
    base_c = cells["rope_on__border_on"]["corner_block"]
    comb_c = cells["rope_off__border_off"]["corner_block"]
    report = {"model": args.variant, "grid": grid, "n_eval": len(ev), "border_ring": args.border,
              "cells": cells,
              "combined_drop_frac": round(1 - comb_c / base_c, 3),
              "verdict": ("RoPE+border jointly explain the sink (collapses to ~uniform)"
                          if comb_c < 1.4 else
                          f"residual attention-sink persists ({comb_c}x) -> intrinsic softmax sink beyond RoPE+border")}
    outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    for k, m in maps.items():
        np.save(outp.parent / f"map_{k}.npy", m.astype(np.float32))
    print(json.dumps(report, indent=2), flush=True)

    # 2x2 figure: per cell, the max-peak head's spatial map (where the sink sits)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = [["rope_on__border_on", "rope_on__border_off"],
                 ["rope_off__border_on", "rope_off__border_off"]]
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for r in range(2):
            for c in range(2):
                k = order[r][c]; m = maps[k]
                mn = m / (m.sum(axis=(1, 2), keepdims=True) + 1e-12)
                uni = 1.0 / (grid * grid)
                peaks = (mn.reshape(mn.shape[0], -1).max(axis=1) / uni)
                h = int(peaks.argmax())
                ax = axes[r][c]; ax.imshow(mn[h], cmap="viridis")
                ax.set_title(f"{k}\nhead {h}  peak {peaks[h]:.0f}x  corner {cells[k]['corner_block']}x", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{args.variant} last-block sink map — RoPE × border ablation (max-peak head)", fontsize=11)
        fig.tight_layout(); fig.savefig(outp.parent / "fig_2x2_sink_maps.png", dpi=140); plt.close(fig)
        print("[fig] saved fig_2x2_sink_maps.png", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[fig] failed ({e}); maps saved", flush=True)


if __name__ == "__main__":
    main()
