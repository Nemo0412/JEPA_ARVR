#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · V-JEPA test-time-register port (no training).
# Ports nickjiang2378/test-time-registers to V-JEPA 2.1: (1) calibrate sink neurons,
# (2) append register tokens + redirect those neurons onto them (forward hooks),
# (3) measure the last-block patch attention sink OFF vs ON. Slurm only.
"""Remove the V-JEPA encoder corner attention sink with appended test-time registers."""
from __future__ import annotations
from app.hdepic_lora_action_anticipation.share_paths import DATA_ROOT as SHARE_DATA_ROOT, VJEPA_ROOT as SHARE_VJEPA_ROOT

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SHARED = os.environ.get("SHARED_PROJECT_ROOT", str(SHARE_DATA_ROOT))
CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (str(SHARE_VJEPA_ROOT), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (  # noqa: E402
    FpsSubsampledStreamMTPDataset,
)
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import (  # noqa: E402
    build_base_21, HeadAttnCapture21,
)


def _mlp_of(block):
    return block.mlp


# ── calibration: find sink neurons (their find_register_neurons, V-JEPA-adapted) ──
def find_sink_neurons(encoder, clips_iter, layers, norm_thresh, topk_per_layer, gp):
    acts = {}      # layer -> captured mlp.act output (seq, hidden)
    handles = []
    for L in layers:
        m = _mlp_of(encoder.blocks[L]).act
        def mk(L):
            def h(mod, inp, out):
                acts[L] = out.detach()[0].float()   # (N, hidden)
            return h
        handles.append(m.register_forward_hook(mk(L)))
    last_out = {}
    hlast = encoder.blocks[-1].register_forward_hook(
        lambda mod, inp, out: last_out.__setitem__("x", (out[0] if isinstance(out, tuple) else out).detach()[0].float()))

    n_layers_used = 0
    scores = {L: None for L in layers}
    for clips in clips_iter:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoder(clips)
        x = last_out["x"]                              # (N, D) last-block output
        norms = x.norm(dim=-1)                         # (N,)
        N = norms.shape[0]
        # patch tokens only (no registers during calibration)
        sink = torch.where(norms > norm_thresh)[0]
        if sink.numel() == 0:
            # fall back to the single max-norm token so calibration never empties
            sink = norms.topk(max(1, N // 256)).indices
        for L in layers:
            a = acts[L].abs()                          # (N, hidden)
            s = a[sink].mean(dim=0)                     # (hidden,)
            scores[L] = s if scores[L] is None else scores[L] + s
        n_layers_used += 1
    for h in handles + [hlast]:
        h.remove()
    sink_neurons = {}
    for L in layers:
        s = (scores[L] / max(1, n_layers_used))
        idx = s.topk(topk_per_layer).indices.tolist()
        sink_neurons[L] = idx
    return sink_neurons


# ── intervention: append registers + redirect sink neurons ────────────────────
class TestTimeRegisters:
    def __init__(self, encoder, n_registers, sink_neurons, scale=1.0, normal="zero"):
        self.encoder = encoder
        self.R = int(n_registers)
        self.sink_neurons = sink_neurons
        self.scale = scale
        self.normal = normal
        self._handles = []
        D = encoder.embed_dim
        self.reg = torch.zeros(1, self.R, D)           # appended zero register tokens
        # append registers at block-0 input; make all blocks register-aware
        b0 = encoder.blocks[0]
        def pre(mod, args, kwargs):
            x = args[0]
            reg = self.reg.to(x.device, x.dtype).expand(x.shape[0], -1, -1)
            return (torch.cat([x, reg], dim=1),) + args[1:], kwargs
        self._handles.append(b0.register_forward_pre_hook(pre, with_kwargs=True))
        self._orig_nreg = []
        for blk in encoder.blocks:
            self._orig_nreg.append(blk.attn.n_registers)
            blk.attn.n_registers = self.R
        # redirect hooks on sink-neuron layers' mlp.act
        for L, neurons in sink_neurons.items():
            idx = torch.tensor(neurons, dtype=torch.long)
            m = _mlp_of(encoder.blocks[L]).act
            self._handles.append(m.register_forward_hook(self._mk_redirect(idx)))

    def _mk_redirect(self, idx):
        R, scale, normal = self.R, self.scale, self.normal
        def hook(mod, inp, out):
            idx_d = idx.to(out.device)
            patch = out[0, :-R, idx_d].float()          # (N_patch, k)
            amax = patch.abs().argmax(dim=0)            # (k,) which patch is max-abs per neuron
            signed = patch[amax, torch.arange(patch.shape[1], device=out.device)]  # signed max-abs
            out[0, -R:, idx_d] = (scale * signed).unsqueeze(0).expand(R, -1).to(out.dtype)
            if normal == "zero":
                out[0, :-R, idx_d] = 0
            elif normal == "mean":
                out[0, :-R, idx_d] = patch.mean(dim=0).to(out.dtype)
            return out
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        for blk, n in zip(self.encoder.blocks, self._orig_nreg):
            blk.attn.n_registers = n


# ── measure last-block patch sink ─────────────────────────────────────────────
def measure(encoder, clips_iter, grid, gp, n_registers=0):
    cap = HeadAttnCapture21(encoder.blocks[-1].attn)
    peaks, reg_mass, corner = [], [], []
    for clips in clips_iter:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoder(clips)
        imp = cap.importance[0].float().cpu().numpy()   # (H, N)
        H, N = imp.shape
        n_patch = N - n_registers
        patch = imp[:, :n_patch]
        denom = imp.sum(axis=1, keepdims=True) + 1e-12
        uni = 1.0 / (grid * grid)
        # spatial per head (sum over time slots)
        slots = n_patch // gp
        sp = patch.reshape(H, slots, grid, grid).sum(axis=1)
        sp = sp / (sp.sum(axis=(1, 2), keepdims=True) + 1e-12)
        b = max(1, grid // 8)
        cb = np.stack([(sp[h, :b, :b].sum() + sp[h, :b, -b:].sum() + sp[h, -b:, :b].sum() + sp[h, -b:, -b:].sum())
                       / (4 * b * b) / uni for h in range(H)])
        corner.append(cb.max())
        peaks.append(((patch / denom) / uni).max())
        if n_registers > 0:
            reg_mass.append(float((imp[:, n_patch:].sum(axis=1) / denom.squeeze(1)).mean()))
    cap.remove()
    return {"patch_sink_peak_over_uniform": round(float(np.mean(peaks)), 2),
            "corner_block_over_uniform": round(float(np.mean(corner)), 2),
            "register_received_mass": round(float(np.mean(reg_mass)), 4) if reg_mass else None}


def make_clip_iter(val_ds, picked, device):
    mean = T.IMAGENET_MEAN.to(device); std = T.IMAGENET_STD.to(device)
    for idx in picked:
        b = T.collate_stream([val_ds[idx]])
        c = b["clip"].to(device).float().div_(255.0).sub_(mean).div_(std)
        yield c


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
    ap.add_argument("--n-calib", type=int, default=20)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--layers", type=str, default="14,15,16,17,18,19,20")
    ap.add_argument("--n-registers", type=int, default=4)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--norm-thresh", type=float, default=0.0, help="0 => auto per-clip (mean+3std)")
    ap.add_argument("--scale", type=float, default=1.0, help="register activation multiplier (sweep to test RoPE limit)")
    args = ap.parse_args()

    device = torch.device("cuda")
    base, _ = build_base_21(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size, checkpoint=args.checkpoint)
    encoder = base.encoder
    grid = int(encoder.blocks[-1].attn.grid_size)
    gp = grid * grid
    layers = [int(x) for x in args.layers.split(",")]

    val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(val_ds.rows)) if abs(float(val_ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6]
    calib = rows[: args.n_calib]
    ev = rows[args.n_calib: args.n_calib + args.n_eval]
    print(f"[data] grid={grid} calib={len(calib)} eval={len(ev)} layers={layers} R={args.n_registers}", flush=True)

    # auto norm threshold: probe one clip's last-block token norms
    if args.norm_thresh <= 0:
        probe = {}
        h = encoder.blocks[-1].register_forward_hook(lambda m, i, o: probe.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()[0].float()))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoder(next(make_clip_iter(val_ds, calib, device)))
        h.remove()
        nrm = probe["x"].norm(dim=-1)
        args.norm_thresh = float(nrm.mean() + 3 * nrm.std())
        print(f"[calib] auto norm_thresh={args.norm_thresh:.1f} (max token norm {float(nrm.max()):.1f})", flush=True)

    sink_neurons = find_sink_neurons(encoder, make_clip_iter(val_ds, calib, device), layers, args.norm_thresh, args.topk, gp)
    print(f"[calib] sink neurons per layer: {{{', '.join(f'{L}:{len(v)}' for L,v in sink_neurons.items())}}}", flush=True)

    before = measure(encoder, make_clip_iter(val_ds, ev, device), grid, gp, n_registers=0)
    tt = TestTimeRegisters(encoder, args.n_registers, sink_neurons, scale=args.scale, normal="zero")
    after = measure(encoder, make_clip_iter(val_ds, ev, device), grid, gp, n_registers=args.n_registers)
    tt.remove()

    report = {"model": "vjepa2.1_base", "img_size": args.img_size, "grid": grid,
              "n_registers": args.n_registers, "layers": layers, "topk_per_layer": args.topk,
              "norm_thresh": round(args.norm_thresh, 1),
              "sink_neurons": {str(k): v for k, v in sink_neurons.items()},
              "before": before, "after": after,
              "_summary": {
                  "patch_sink_peak": f'{before["patch_sink_peak_over_uniform"]}x -> {after["patch_sink_peak_over_uniform"]}x',
                  "corner_block": f'{before["corner_block_over_uniform"]}x -> {after["corner_block_over_uniform"]}x',
                  "register_absorbs": after["register_received_mass"]}}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
