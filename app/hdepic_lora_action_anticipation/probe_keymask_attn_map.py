#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · What does the last-block received-attention map look like
# AFTER ll's StreamingLLM-style corner key-masking? Forbidding all queries from
# attending to the corner sink columns (S[:,j]=-inf, every block) both (a) changes
# the last-block q,k via masked upstream representations and (b) removes the corner
# columns from the map. Question: where does the excess attention RELOCATE? A new
# sink (edge/interior), or spread ~uniform? Runs baseline vs masked on the same
# clips and plots them side-by-side. Real V-JEPA 2.0 finetuned. Slurm only.
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(CODE_ROOT, "vjepa2"), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (  # noqa: E402
    FpsSubsampledStreamMTPDataset, CornerKeyMasker,
)
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import (  # noqa: E402
    build_finetuned_20, HeadAttnCapture20,
)


class MaskedHeadCapture:
    """Like HeadAttnCapture20 but zeros the corner KEY columns in the softmax, so the
    captured received-attention reflects ll's masking (A[:,corner]=0). Local corner
    mask (gp,) tiled over time slots. Returns the real (masked) block output."""

    def __init__(self, attn_module, local_corner, gp, chunk_size: int = 256):
        from src.models.utils.modules import rotate_queries_or_keys
        m = attn_module
        self._m = m; self._orig = m.forward
        self.importance = None
        self.gp = int(gp)
        self.local_corner = local_corner  # (gp,) bool, True at corner cells
        cap = self

        def _fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            out = cap._orig(x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
            with torch.no_grad():
                B, N, C = x.size()
                grid_depth = int(N // (m.grid_size * m.grid_size))
                qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv[0], qkv[1], qkv[2]
                if mask is not None:
                    mp = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
                    d_mask, h_mask, w_mask = m.separate_positions(mp, H_patches, W_patches)
                else:
                    mp = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=x.device)
                    d_mask, h_mask, w_mask = m.separate_positions(mp, H_patches, W_patches)
                s = 0
                qd = rotate_queries_or_keys(q[..., s:s + m.d_dim], pos=d_mask)
                kd = rotate_queries_or_keys(k[..., s:s + m.d_dim], pos=d_mask); s += m.d_dim
                qh = rotate_queries_or_keys(q[..., s:s + m.h_dim], pos=h_mask)
                kh = rotate_queries_or_keys(k[..., s:s + m.h_dim], pos=h_mask); s += m.h_dim
                qw = rotate_queries_or_keys(q[..., s:s + m.w_dim], pos=w_mask)
                kw = rotate_queries_or_keys(k[..., s:s + m.w_dim], pos=w_mask); s += m.w_dim
                if s < m.head_dim:
                    q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
                    k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
                else:
                    q = torch.cat([qd, qh, qw], dim=-1)
                    k = torch.cat([kd, kh, kw], dim=-1)
                # corner key-column mask, tiled over slots
                slots = N // cap.gp
                col = cap.local_corner.to(x.device).repeat(slots)
                if col.numel() < N:
                    col = torch.cat([col, torch.zeros(N - col.numel(), dtype=torch.bool, device=x.device)])
                col = col.view(1, 1, 1, N)
                imp = torch.zeros(B, m.num_heads, N, device=x.device, dtype=torch.float32)
                for ci in range(0, N, chunk_size):
                    qc = q[:, :, ci:ci + chunk_size, :]
                    logits = (qc @ k.transpose(-2, -1)) * m.scale
                    logits = logits.masked_fill(col, float("-inf"))   # A[:,corner]=0
                    imp += logits.softmax(dim=-1).sum(dim=2).float()
                cap.importance = imp
            return out

        m.forward = _fwd

    def remove(self):
        self._m.forward = self._orig


def spatial(imp, grid, gp):
    """(heads,N) -> per-head (heads,grid,grid), time-summed, head-normalized to sum 1."""
    H, N = imp.shape
    slots = N // gp
    sp = imp[:, :slots * gp].reshape(H, slots, grid, grid).sum(axis=1)
    return sp / (sp.sum(axis=(1, 2), keepdims=True) + 1e-12)


def peak_info(sp_h, grid):
    uni = 1.0 / (grid * grid)
    flat = sp_h.reshape(-1)
    j = int(flat.argmax())
    return (j // grid, j % grid), float(flat[j] / uni)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--encoder-lora", default=None); ap.add_argument("--predictor-lora", default=None)
    ap.add_argument("--init-from-ckpt", default=None)
    ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=4.0); ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--ring", type=int, default=1); ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda")
    base, _ = build_finetuned_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size,
                                 checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                 pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    enc = base.encoder
    grid = int(enc.blocks[-1].attn.grid_size); gp = grid * grid
    nb = len(enc.blocks)
    val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(val_ds.rows)) if abs(float(val_ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6][: args.n_eval]
    print(f"[data] grid={grid} nb={nb} eval={len(rows)} ring={args.ring}", flush=True)

    lc = torch.zeros(grid, grid, dtype=torch.bool)
    r = args.ring; lc[:r, :r] = lc[:r, -r:] = lc[-r:, :r] = lc[-r:, -r:] = True
    lc = lc.reshape(-1)

    def get_clip(i):
        c = T.collate_stream([val_ds[rows[i]]])["clip"].to(device).float().div_(255.0)
        return c.sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))

    def run(masked):
        if masked:
            masker = CornerKeyMasker(enc, grid, gp, ring=args.ring, blocks=",".join(str(i) for i in range(nb - 1)))
            cap = MaskedHeadCapture(enc.blocks[-1].attn, lc, gp)
        else:
            masker = None; cap = HeadAttnCapture20(enc.blocks[-1].attn)
        acc = None
        for i in range(len(rows)):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc(get_clip(i))
            sp = spatial(cap.importance[0].float().cpu().numpy(), grid, gp)
            acc = sp if acc is None else acc + sp
        cap.remove()
        if masker is not None:
            masker.remove()
        return acc / len(rows)

    base_sp = run(False)     # (heads, grid, grid)
    mask_sp = run(True)
    uni = 1.0 / (grid * grid); b = max(1, grid // 8)

    def corner_mass(sp_h):
        return float((sp_h[:b, :b].sum() + sp_h[:b, -b:].sum() + sp_h[-b:, :b].sum() + sp_h[-b:, -b:].sum()) / (4 * b * b) / uni)

    order = sorted(range(base_sp.shape[0]), key=lambda h: corner_mass(base_sp[h]), reverse=True)[: args.topk]
    report = {"grid": grid, "ring": args.ring, "n_eval": len(rows), "topk_heads_by_baseline_corner": order, "heads": {}}
    for h in order:
        (br, bc), bp = peak_info(base_sp[h], grid)
        (mr, mc), mp = peak_info(mask_sp[h], grid)
        report["heads"][str(h)] = {
            "baseline_corner_over_uni": round(corner_mass(base_sp[h]), 2),
            "masked_corner_over_uni": round(corner_mass(mask_sp[h]), 2),
            "baseline_peak_rc": [br, bc], "baseline_peak_over_uni": round(bp, 1),
            "masked_peak_rc": [mr, mc], "masked_peak_over_uni": round(mp, 1),
        }
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "baseline_spatial.npy", base_sp.astype(np.float32))
    np.save(outdir / "masked_spatial.npy", mask_sp.astype(np.float32))
    (outdir / "keymask_attn_map.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    # side-by-side figure: rows=top heads, cols=[baseline, masked]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(order), 2, figsize=(5.2, 2.4 * len(order)), squeeze=False)
        for ri, h in enumerate(order):
            for ci, (sp, tag) in enumerate([(base_sp[h], "baseline"), (mask_sp[h], "keymask")]):
                ax = axes[ri][ci]; ax.imshow(sp, cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
                (pr, pc), pk = peak_info(sp, grid)
                ax.set_title(f"h{h} {tag}\npeak {pk:.0f}x @({pr},{pc})", fontsize=8)
        fig.suptitle(f"2.0 finetuned last-block received attn — baseline vs corner key-mask (ring{args.ring})", fontsize=10)
        fig.tight_layout(); fig.savefig(outdir / "fig_keymask_vs_baseline.png", dpi=140); plt.close(fig)
        print(f"[fig] saved {outdir/'fig_keymask_vs_baseline.png'}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[fig] failed ({e}); arrays saved", flush=True)


if __name__ == "__main__":
    main()
