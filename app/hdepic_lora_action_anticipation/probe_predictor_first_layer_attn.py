#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Predictor received-attention pattern (no pruning), multi-purpose.
# Prior work read the ENCODER's last block; the advisor asked to look at the PREDICTOR.
#   (1) --blocks 0,3,6,9,11  : per-block predictor context corner sink (depth sweep).
#   (2) --capture-encoder    : also capture the encoder last block on the SAME clips and emit
#                              a predictor-head × encoder-head spatial cosine-sim matrix (are
#                              the predictor sink heads "the same" as the encoder's?). NB the
#                              two modules have independent QKV and different head counts
#                              (encoder 16 vs predictor 12), so head-INDEX identity is undefined;
#                              we compare spatial MAPS, not indices.
#   (3) --rope-factorial     : encoder/predictor RoPE on/off 2x2, measure predictor block-0 sink
#                              (is the predictor sink inherited from the encoder's corner tokens,
#                              or made by the predictor's own RoPE?).
# The predictor input after its internal sort = [context patch tokens 0..N_ctx-1] + [target/mask
# tokens >= N_ctx], so the first N_ctx received-attn entries are the context grid.
# Frozen finetuned 2.0 pruning model. Slurm only.
from __future__ import annotations

import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np, torch

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/scratch/yh6416/VJEPA2-EXP")
CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(CODE_ROOT, "vjepa2"), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import FpsSubsampledStreamMTPDataset  # noqa: E402
from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import (  # noqa: E402
    build_finetuned_20, HeadAttnCapture20, corner_metrics,
)
from app.hdepic_lora_action_anticipation.probe_combined_ablation import RoPEAblate20  # noqa: E402


def _spatial(imp, n_take, gp, grid, n_heads):
    """(heads,N) importance -> per-head time-avg spatial (heads,grid,grid) over first n_take keys."""
    slots = n_take // gp
    per_head_total = imp.sum(axis=1) + 1e-12
    maps = imp[:, :n_take].reshape(n_heads, slots, grid, grid) / per_head_total[:, None, None, None]
    sp = maps.sum(axis=1)                                   # (heads, grid, grid)
    return sp / (sp.sum(axis=(1, 2), keepdims=True) + 1e-12), per_head_total


def _attach(encoder, predictor, blocks, enc_rope, pred_rope, capture_encoder):
    """De-RoPE and/or capture per config. enc RoPE-off ablates EVERY encoder block; predictor
    captured blocks use HeadAttnCapture (rope on) or RoPEAblate+capture (rope off)."""
    handles = []
    enc_cap = None
    n_enc = len(encoder.blocks)
    for i, blk in enumerate(encoder.blocks):
        want = capture_encoder and (i == n_enc - 1)
        if enc_rope == "off":
            w = RoPEAblate20(blk.attn, capture=want); handles.append(w)
            if want:
                enc_cap = w
        elif want:
            w = HeadAttnCapture20(blk.attn); handles.append(w); enc_cap = w
    pred_caps = {}
    for b in blocks:
        attn = predictor.predictor_blocks[b].attn
        w = RoPEAblate20(attn, capture=True) if pred_rope == "off" else HeadAttnCapture20(attn)
        handles.append(w); pred_caps[b] = w
    return handles, enc_cap, pred_caps


def _iter_clips(val_ds, picked, device):
    mean = T.IMAGENET_MEAN.to(device); std = T.IMAGENET_STD.to(device)
    for idx in picked:
        try:
            batch = T.collate_stream([val_ds[idx]])
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {idx}: {e}", flush=True); continue
        yield batch["clip"].to(device).float().div_(255.0).sub_(mean).div_(std)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _ = build_finetuned_20(
        device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size,
        checkpoint=args.checkpoint, enc_lora=args.encoder_lora, pred_lora=args.predictor_lora,
        parent_ckpt=args.init_from_ckpt)
    model = T.PrunedAnticipativeModel(base, None, prune_threshold=10 ** 9).to(device); model.eval()
    encoder, predictor = base.encoder, base.predictor
    grid = int(predictor.predictor_blocks[0].attn.grid_size); gp = grid * grid
    tub = int(base.tubelet_size)
    n_pred = int(base.grid_size ** 2 * (base.num_output_frames // tub))
    ph = int(predictor.predictor_blocks[0].attn.num_heads)
    eh = int(encoder.blocks[-1].attn.num_heads)
    blocks = [int(x) for x in str(args.blocks).split(",") if x.strip() != ""]

    val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size,
                                           src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(val_ds.rows))
            if abs(float(val_ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6]
    order = sorted(rows, key=lambda i: hashlib.md5(
        f"{args.seed}|{val_ds.rows[i]['video_id']}|{val_ds.rows[i].get('frame_indices','')}".encode()).hexdigest())
    picked = order[: args.n_samples]
    ant_val = float(args.anticipation_sec)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"[model] pred heads={ph} enc heads={eh} grid={grid} n_pred={n_pred} blocks={blocks} "
          f"clips={len(picked)}", flush=True)

    report = {"site": "predictor", "grid": grid, "n_pred_target_tokens": n_pred,
              "context_sec": args.context_sec, "anticipation_sec": ant_val,
              "predictor_blocks": blocks, "n_samples_req": len(picked)}

    # ── (3) RoPE factorial: predictor block-0 sink vs enc/pred RoPE on/off ──────
    if args.rope_factorial:
        fac = {}
        for er in ("on", "off"):
            for pr in ("on", "off"):
                handles, _, pcaps = _attach(encoder, predictor, [0], er, pr, False)
                cs, ts, n = [], [], 0
                for clip in _iter_clips(val_ds, picked, device):
                    ant = torch.full((clip.size(0),), ant_val, device=device)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        model(clip, ant)
                    imp = pcaps[0].importance[0].float().cpu().numpy()
                    N_ctx = imp.shape[1] - n_pred
                    if N_ctx <= 0 or N_ctx % gp != 0:
                        continue
                    sp, tot = _spatial(imp, N_ctx, gp, grid, ph)
                    cs.append(max(corner_metrics(sp[h])["cornerblk_over_uniform"] for h in range(ph)))
                    ts.append(float(imp[:, N_ctx:].sum() / (imp.sum() + 1e-12)))
                    n += 1
                for h in handles:
                    h.remove()
                fac[f"enc{er}_pred{pr}"] = {"pred_l0_corner_max_head": round(float(np.mean(cs)), 3),
                                            "target_mass": round(float(np.mean(ts)), 4), "n": n}
                print(f"[factorial] enc{er} pred{pr}: corner={np.mean(cs):.3f} tgt={np.mean(ts):.4f} n={n}", flush=True)
        report["rope_factorial"] = fac
        (outdir / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return report

    # ── (1)+(2) depth sweep (+ optional encoder correspondence) ────────────────
    handles, enc_cap, pred_caps = _attach(encoder, predictor, blocks, args.enc_rope, args.pred_rope,
                                          args.capture_encoder)
    acc = {b: None for b in blocks}; tgt = {b: np.zeros(ph) for b in blocks}
    enc_acc = None; n_used = 0
    for clip in _iter_clips(val_ds, picked, device):
        ant = torch.full((clip.size(0),), ant_val, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            model(clip, ant)
        ok = True
        cur = {}
        for b in blocks:
            imp = pred_caps[b].importance[0].float().cpu().numpy()
            N_ctx = imp.shape[1] - n_pred
            if N_ctx <= 0 or N_ctx % gp != 0:
                ok = False; break
            sp, tot = _spatial(imp, N_ctx, gp, grid, ph)
            cur[b] = (sp, imp[:, N_ctx:].sum(axis=1) / tot)
        if not ok:
            continue
        for b in blocks:
            sp, tm = cur[b]
            acc[b] = sp if acc[b] is None else acc[b] + sp
            tgt[b] += tm
        if enc_cap is not None:
            eimp = enc_cap.importance[0].float().cpu().numpy()
            esp, _ = _spatial(eimp, eimp.shape[1], gp, grid, eh)
            enc_acc = esp if enc_acc is None else enc_acc + esp
        n_used += 1
        if n_used % 25 == 0:
            print(f"  {n_used}", flush=True)
    for h in handles:
        h.remove()
    if n_used == 0:
        raise SystemExit("no usable clips")

    per_block = {}
    for b in blocks:
        sp = acc[b] / n_used
        sp = sp / (sp.sum(axis=(1, 2), keepdims=True) + 1e-12)
        m = [corner_metrics(sp[h]) for h in range(ph)]
        per_block[b] = {"spatial": sp, "metrics": m, "tgt": tgt[b] / n_used}
        np.save(outdir / f"pred_block{b}_spatial.npy", sp.astype(np.float32))
    report["n_used"] = n_used
    report["enc_rope"] = args.enc_rope; report["pred_rope"] = args.pred_rope
    report["per_block"] = {str(b): {
        "corner_by_head": [round(per_block[b]["metrics"][h]["cornerblk_over_uniform"], 3) for h in range(ph)],
        "corner_mean": round(float(np.mean([per_block[b]["metrics"][h]["cornerblk_over_uniform"] for h in range(ph)])), 3),
        "corner_max": round(float(np.max([per_block[b]["metrics"][h]["cornerblk_over_uniform"] for h in range(ph)])), 3),
        "target_mass_mean": round(float(per_block[b]["tgt"].mean()), 4),
    } for b in blocks}

    if enc_cap is not None:
        esp = enc_acc / n_used
        esp = esp / (esp.sum(axis=(1, 2), keepdims=True) + 1e-12)
        np.save(outdir / "encoder_last_spatial.npy", esp.astype(np.float32))
        em = [corner_metrics(esp[h]) for h in range(eh)]
        # cosine-sim of predictor L0 head maps vs encoder last-block head maps
        b0 = blocks[0]; psp = per_block[b0]["spatial"]
        pf = psp.reshape(ph, -1); ef = esp.reshape(eh, -1)
        pf = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-12)
        ef = ef / (np.linalg.norm(ef, axis=1, keepdims=True) + 1e-12)
        sim = pf @ ef.T                                     # (ph, eh)
        report["encoder_last_corner_by_head"] = [round(em[h]["cornerblk_over_uniform"], 3) for h in range(eh)]
        report["corr_best_enc_head_per_pred_head"] = {
            str(p): {"enc_head": int(sim[p].argmax()), "cos": round(float(sim[p].max()), 3)} for p in range(ph)}
        np.save(outdir / "pred_vs_enc_cosine.npy", sim.astype(np.float32))
        _plot_correspondence(sim, per_block[b0]["metrics"], em, ph, eh, outdir)

    (outdir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("site", "n_used", "per_block",
          "enc_rope", "pred_rope")}, indent=2), flush=True)
    _plot_depth(per_block, blocks, ph, args, outdir)
    return report


def _plot_depth(per_block, blocks, ph, args, outdir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] mpl unavailable ({e})", flush=True); return
    # corner mean/max vs predictor depth + target-mass vs depth
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cm = [np.mean([per_block[b]["metrics"][h]["cornerblk_over_uniform"] for h in range(ph)]) for b in blocks]
    cx = [np.max([per_block[b]["metrics"][h]["cornerblk_over_uniform"] for h in range(ph)]) for b in blocks]
    tm = [float(per_block[b]["tgt"].mean()) for b in blocks]
    ax[0].plot(blocks, cm, "o-", label="head-avg"); ax[0].plot(blocks, cx, "s-", label="max head")
    ax[0].axhline(1.0, color="gray", ls=":"); ax[0].set_xlabel("predictor block"); ax[0].set_ylabel("corner/uniform"); ax[0].legend()
    ax[0].set_title("Predictor corner sink vs depth")
    ax[1].plot(blocks, tm, "^-", color="tab:red"); ax[1].set_xlabel("predictor block")
    ax[1].set_ylabel("target-mass fraction"); ax[1].set_title("Attn to anticipation targets vs depth")
    fig.tight_layout(); fig.savefig(outdir / "predictor_depth_sweep.png", dpi=130); plt.close(fig)
    # per-block sink-head spatial maps
    show = blocks
    fig2, axes = plt.subplots(len(show), ph, figsize=(1.4 * ph, 1.5 * len(show)), squeeze=False)
    for r, b in enumerate(show):
        sp = per_block[b]["spatial"]; vmax = np.percentile(sp, 99.5)
        for h in range(ph):
            ax = axes[r][h]; ax.imshow(sp[h], cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            cb = per_block[b]["metrics"][h]["cornerblk_over_uniform"]
            if r == 0:
                ax.set_title(f"h{h}", fontsize=7)
            if h == 0:
                ax.set_ylabel(f"blk{b}", fontsize=8)
            ax.text(0.5, -0.08, f"{cb:.1f}", transform=ax.transAxes, ha="center", va="top",
                    fontsize=6, color=("red" if cb > 1.8 else "gray"))
    fig2.suptitle(f"Predictor per-head context maps by depth (enc RoPE {args.enc_rope}, pred RoPE {args.pred_rope})")
    fig2.tight_layout(); fig2.savefig(outdir / "predictor_depth_maps.png", dpi=110, bbox_inches="tight"); plt.close(fig2)
    print("[plot] saved predictor_depth_sweep.png + predictor_depth_maps.png", flush=True)


def _plot_correspondence(sim, pmetrics, emetrics, ph, eh, outdir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] mpl unavailable ({e})", flush=True); return
    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * eh, 1.0 + 0.5 * ph))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("encoder last-block head"); ax.set_ylabel("predictor block-0 head")
    ax.set_xticks(range(eh)); ax.set_yticks(range(ph))
    ax.set_title("Spatial map cosine sim (predictor L0 head × encoder L23 head)")
    for p in range(ph):
        e = int(sim[p].argmax()); ax.text(e, p, f"{sim[p][e]:.2f}", ha="center", va="center",
                                          color="white", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(outdir / "pred_vs_enc_correspondence.png", dpi=130); plt.close(fig)
    print("[plot] saved pred_vs_enc_correspondence.png", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description="[ATTN-CORNER-SINK] predictor attention pattern (multi-purpose)")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--init-from-ckpt", type=Path, default=None)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--blocks", default="0", help="comma list of predictor blocks (0=first)")
    ap.add_argument("--capture-encoder", action="store_true", help="also capture encoder last block + correspondence")
    ap.add_argument("--enc-rope", default="on", choices=["on", "off"])
    ap.add_argument("--pred-rope", default="on", choices=["on", "off"])
    ap.add_argument("--rope-factorial", action="store_true", help="enc/pred RoPE on/off 2x2 on predictor block-0")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=4.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=256)
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
