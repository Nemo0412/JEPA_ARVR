#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Why does the RoPE-OFF sink still cluster on the LEFT edge?
# With RoPE off + no additive pos-embed, attention is permutation-equivariant, so any
# residual edge concentration must be CONTENT (or a degenerate artifact). Decisive test:
# horizontally mirror the input under RoPE-off; if the left-edge mass moves to the RIGHT,
# it is content-driven. Real V-JEPA 2.0 finetuned. Slurm only.
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
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import (  # noqa: E402
    build_finetuned_20, build_base_20, HeadAttnCapture20, _decode_video_clip,
)
from app.hdepic_lora_action_anticipation.probe_combined_ablation import RoPEAblate20  # noqa: E402


def corner_of(sp, w):
    g = sp.shape[-1]; spn = sp / (sp.sum(axis=(1, 2), keepdims=True) + 1e-12)
    b = max(1, g // 8); uni = 1.0 / (g * g)
    cb = np.array([(spn[h, :b, :b].sum() + spn[h, :b, -b:].sum() + spn[h, -b:, :b].sum() + spn[h, -b:, -b:].sum())
                   / (4 * b * b) / uni for h in range(spn.shape[0])])
    return float(cb.max())


def edge_mass(sp, w):
    """sp: (H, g, g) normalized per head. Return mean-over-heads left/right/top/bottom mass."""
    g = sp.shape[-1]
    spn = sp / (sp.sum(axis=(1, 2), keepdims=True) + 1e-12)
    L = spn[:, :, :w].sum(axis=(1, 2)).mean()
    R = spn[:, :, -w:].sum(axis=(1, 2)).mean()
    Tp = spn[:, :w, :].sum(axis=(1, 2)).mean()
    B = spn[:, -w:, :].sum(axis=(1, 2)).mean()
    return float(L), float(R), float(Tp), float(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--val-csv", default=None)
    ap.add_argument("--video-root", default=None); ap.add_argument("--video-dir", default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--variant", default="finetuned20", choices=["finetuned20", "base20"])
    ap.add_argument("--encoder-lora", default=None); ap.add_argument("--predictor-lora", default=None)
    ap.add_argument("--init-from-ckpt", default=None)
    ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=4.0); ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--edge", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.variant == "finetuned20":
        base, _ = build_finetuned_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size,
                                     checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                     pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    else:
        base, _ = build_base_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size, checkpoint=args.checkpoint)
    enc = base.encoder
    grid = int(enc.blocks[-1].attn.grid_size); gp = grid * grid
    mean = T.IMAGENET_MEAN.to(device); std = T.IMAGENET_STD.to(device)

    # clip source: EGTEA CSV or a directory of arbitrary MP4s
    use_dir = args.video_dir is not None
    if use_dir:
        import glob
        files = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")) + glob.glob(os.path.join(args.video_dir, "*.MP4")))[: args.n_eval]
        dataset_tag = os.path.basename(str(args.video_dir).rstrip("/"))
        def get_clip(i, hflip):
            clip = _decode_video_clip(files[i], args.img_size, args.max_frames).unsqueeze(0).to(device).float().div_(255.0)
            if hflip:
                clip = torch.flip(clip, dims=[-1])
            return clip.sub_(mean).div_(std)
        nev = len(files)
    else:
        val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
        rows = [i for i in range(len(val_ds.rows)) if abs(float(val_ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6][: args.n_eval]
        dataset_tag = "egtea"
        def get_clip(i, hflip):
            c = T.collate_stream([val_ds[rows[i]]])["clip"].to(device).float().div_(255.0)
            if hflip:
                c = torch.flip(c, dims=[-1])
            return c.sub_(mean).div_(std)
        nev = len(rows)
    print(f"[data] variant={args.variant} dataset={dataset_tag} grid={grid} eval={nev}", flush=True)

    def run(rope_off, hflip):
        if rope_off:
            abls = [RoPEAblate20(enc.blocks[i].attn, capture=(i == len(enc.blocks) - 1)) for i in range(len(enc.blocks))]
            cap = abls[-1]
        else:
            abls = []; cap = HeadAttnCapture20(enc.blocks[-1].attn)
        sps = []
        for i in range(nev):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc(get_clip(i, hflip))
            imp = cap.importance[0].float().cpu().numpy()
            H, N = imp.shape; slots = N // gp
            sps.append(imp[:, :slots * gp].reshape(H, slots, grid, grid).sum(axis=1))
        for a in abls:
            a.remove()
        if not rope_off:
            cap.remove()
        return np.mean(sps, axis=0)

    on = run(False, False)                        # RoPE on (baseline picture)
    orig = run(True, False); flip = run(True, True)   # RoPE off, orig + hflip
    Lo, Ro, To, Bo = edge_mass(orig, args.edge)
    Lf, Rf, Tf, Bf = edge_mass(flip, args.edge)
    report = {"variant": args.variant, "dataset": dataset_tag, "grid": grid, "n_eval": nev, "edge_w": args.edge,
              "rope_on_corner": round(corner_of(on, args.edge), 2),
              "rope_off_corner": round(corner_of(orig, args.edge), 2),
              "rope_off_original": {"left": round(Lo, 3), "right": round(Ro, 3), "top": round(To, 3), "bottom": round(Bo, 3)},
              "rope_off_hflip": {"left": round(Lf, 3), "right": round(Rf, 3), "top": round(Tf, 3), "bottom": round(Bf, 3)},
              "left_right_asym_orig": round(Lo - Ro, 3), "left_right_asym_hflip": round(Lf - Rf, 3),
              "verdict": ("CONTENT-driven: left/right asymmetry flips sign with the image"
                          if (Lo - Ro) * (Lf - Rf) < 0 else
                          "asymmetry does NOT flip -> residual position/artifact, not content")}
    outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp.with_suffix(".rope_on.npy"), on.astype(np.float32))
    np.save(outp.with_suffix(".orig.npy"), orig.astype(np.float32))
    np.save(outp.with_suffix(".hflip.npy"), flip.astype(np.float32))
    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
