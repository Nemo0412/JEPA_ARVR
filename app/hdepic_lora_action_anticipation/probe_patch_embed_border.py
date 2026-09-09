#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Causal probe #0: is the CORNER sink a patch-embed border effect?
# (A) descriptive: per-block token-norm by grid position -> where does the corner
#     high-norm/distinctness first appear (block-0 input = patch-embed output)?
# (B) causal: replace the border ring of tokens with the mean token at block-0 input;
#     if the last-block sink drops or relocates to the new border, border-token
#     distinctness drives it. Real V-JEPA 2.1 forward (RoPE on). Slurm only.
from __future__ import annotations
from app.hdepic_lora_action_anticipation.share_paths import DATA_ROOT as SHARE_DATA_ROOT, VJEPA_ROOT as SHARE_VJEPA_ROOT

import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (str(SHARE_VJEPA_ROOT), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import FpsSubsampledStreamMTPDataset  # noqa: E402
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import build_base_21, HeadAttnCapture21  # noqa: E402
from app.hdepic_lora_action_anticipation.vjepa_testtime_register import make_clip_iter  # noqa: E402


def spatial_from_tokens(vec, grid, gp):
    """vec: (N,) -> (grid,grid) mean over time slots."""
    slots = vec.shape[0] // gp
    return vec[: slots * gp].reshape(slots, grid, grid).mean(axis=0)


def corner_center(sp, b):
    g = sp.shape[0]
    corner = (sp[:b, :b].sum() + sp[:b, -b:].sum() + sp[-b:, :b].sum() + sp[-b:, -b:].sum()) / (4 * b * b)
    center = sp[g // 4:3 * g // 4, g // 4:3 * g // 4].mean()
    return float(corner), float(center)


def corner_peak(imp, grid, gp):
    H, N = imp.shape
    slots = N // gp
    sp = imp[:, :slots * gp].reshape(H, slots, grid, grid).sum(axis=1)
    sp = sp / (sp.sum(axis=(1, 2), keepdims=True) + 1e-12)
    uni = 1.0 / (grid * grid); b = max(1, grid // 8)
    cb = np.array([(sp[h, :b, :b].sum() + sp[h, :b, -b:].sum() + sp[h, -b:, :b].sum() + sp[h, -b:, -b:].sum())
                   / (4 * b * b) / uni for h in range(H)])
    denom = imp.sum(axis=1, keepdims=True) + 1e-12
    return float(cb.max()), float(((imp / denom) / uni).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True); ap.add_argument("--out-json", required=True)
    ap.add_argument("--img-size", type=int, default=384); ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=4.0); ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--border", type=int, default=3, help="border ring width in patches to neutralize")
    args = ap.parse_args()

    device = torch.device("cuda")
    base, _ = build_base_21(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size, checkpoint=args.checkpoint)
    enc = base.encoder
    grid = int(enc.blocks[-1].attn.grid_size); gp = grid * grid
    nblk = len(enc.blocks)
    val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(val_ds.rows)) if abs(float(val_ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6]
    ev = rows[: args.n_eval]
    b = max(1, grid // 8)
    print(f"[data] grid={grid} nblk={nblk} eval={len(ev)} corner_b={b} border_ring={args.border}", flush=True)

    # ── (A) descriptive: per-block token-norm spatial map (incl block-0 input) ──
    caps = {}
    handles = []
    # block-0 input = patch-embed(+modality) output
    def pre0(mod, args_, kwargs):
        caps["in"] = args_[0].detach()[0].float()
        return None
    handles.append(enc.blocks[0].register_forward_pre_hook(pre0, with_kwargs=True))
    def mk(i):
        def h(mod, inp, out):
            caps[i] = (out[0] if isinstance(out, tuple) else out).detach()[0].float()
        return h
    for i in range(nblk):
        handles.append(enc.blocks[i].register_forward_hook(mk(i)))

    norm_maps = {("in" if k == "in" else k): [] for k in (["in"] + list(range(nblk)))}
    for clips in make_clip_iter(val_ds, ev, device):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc(clips)
        for k in norm_maps:
            x = caps[k]; nrm = x.norm(dim=-1).cpu().numpy()
            norm_maps[k].append(spatial_from_tokens(nrm, grid, gp))
    for h in handles:
        h.remove()
    ratios = {}
    for k in norm_maps:
        sp = np.mean(norm_maps[k], axis=0)
        c, ce = corner_center(sp, b)
        ratios[str(k)] = round(c / (ce + 1e-9), 3)
    # find first block where corner/center norm ratio crosses 1.5
    emerge = next((k for k in list(range(nblk)) if ratios[str(k)] > 1.5), None)

    # ── (B) causal: neutralize border ring at block-0 input, measure sink ──
    def measure_sink(border_replace):
        cap = HeadAttnCapture21(enc.blocks[-1].attn)
        h = None
        if border_replace:
            R = args.border
            def pre(mod, a, kw):
                x = a[0]
                B, N, D = x.shape
                slots = N // gp
                v = x.view(B, slots, grid, grid, D).clone()
                mask = torch.zeros(grid, grid, dtype=torch.bool, device=x.device)
                mask[:R, :] = mask[-R:, :] = mask[:, :R] = mask[:, -R:] = True
                mean_tok = v[:, :, ~mask, :].mean(dim=2, keepdim=True)   # (B,slots,1,D) center mean
                v[:, :, mask, :] = mean_tok
                return (v.view(B, N, D),) + a[1:], kw
            h = enc.blocks[0].register_forward_pre_hook(pre, with_kwargs=True)
        cs, ps = [], []
        for clips in make_clip_iter(val_ds, ev, device):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc(clips)
            imp = cap.importance[0].float().cpu().numpy()
            c, p = corner_peak(imp, grid, gp); cs.append(c); ps.append(p)
        cap.remove()
        if h:
            h.remove()
        return round(float(np.mean(cs)), 2), round(float(np.mean(ps)), 2)

    base_c, base_p = measure_sink(False)
    repl_c, repl_p = measure_sink(True)

    report = {"model": "vjepa2.1_base", "grid": grid, "n_eval": len(ev), "border_ring": args.border,
              "norm_corner_over_center_by_layer": ratios,
              "corner_highnorm_emerges_block": emerge,
              "sink_baseline": {"corner_block": base_c, "patch_peak": base_p},
              "sink_border_neutralized": {"corner_block": repl_c, "patch_peak": repl_p},
              "verdict": ("border/patch-embed distinctness drives the sink (drops when neutralized)"
                          if repl_c < 0.75 * base_c else
                          "sink survives border neutralization -> not purely a patch-embed border effect")}
    outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
