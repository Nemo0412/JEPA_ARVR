#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Mathematical diagnostics of the last-block corner sink.
# Not figures-and-words: measures the terms of the analytical decomposition on the 2.0
# finetuned pruning model, last block, sink head. Three tests:
#   (1) sink = query-INDEPENDENT attractor: per-key mean_i A_ij (=mu_j) and CV_j =
#       std_i A_ij / mean_i A_ij. Sink keys: high mu, LOW CV (all queries agree).
#   (2) bias-sink: b_j = mean_i S_ij (query-averaged logit); does softmax(b)*Nq rebuild
#       I_j's corner peak? (corr, corner metric).
#   (3) RoPE position kernel: b_pos_j = qbar^T M_j kbar with M_j = mean_i R(p_j - p_i),
#       computed as (mean_i R(p_i) qbar)·(R(p_j) kbar). Predicts: bright at the 4 corners
#       with RoPE ON; FLAT (const) with RoPE OFF (M_j -> I).
# Slurm only. Single head, B=0, averaged over clips.
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
from app.hdepic_lora_action_anticipation.vjepa_testtime_register import make_clip_iter  # noqa: E402


def rotated_qk(m, x):
    """Return (q_rot, k_rot, q_raw, k_raw), each (H,N,D), mirroring the 2.0 encoder RoPE."""
    from src.models.utils.modules import rotate_queries_or_keys
    B, N, C = x.size()
    grid_depth = int(N // (m.grid_size * m.grid_size))
    qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)  # (3,B,H,N,D)
    q, k = qkv[0, 0], qkv[1, 0]  # (H,N,D), B=0
    q_raw, k_raw = q.clone(), k.clone()
    mp = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=x.device)
    d_mask, h_mask, w_mask = m.separate_positions(mp, None, None)

    def rot(t):
        s = 0
        td = rotate_queries_or_keys(t[..., s:s + m.d_dim], pos=d_mask); s += m.d_dim
        th = rotate_queries_or_keys(t[..., s:s + m.h_dim], pos=h_mask); s += m.h_dim
        tw = rotate_queries_or_keys(t[..., s:s + m.w_dim], pos=w_mask); s += m.w_dim
        return torch.cat([td, th, tw, t[..., s:]], dim=-1) if s < m.head_dim else torch.cat([td, th, tw], dim=-1)

    return rot(q.unsqueeze(0))[0], rot(k.unsqueeze(0))[0], q_raw, k_raw


def rot_vec(m, vec, N, device):
    """Rotate a single per-head vector 'vec' (D,) as if placed at every position 0..N-1;
    return (N,D) = [R(p_i) vec]. Used for M_j = mean_i R(p_i) qbar and R(p_j) kbar."""
    D = vec.shape[0]
    tiled = vec.view(1, 1, 1, D).expand(1, 1, N, D).contiguous()  # (1,1,N,D): vec at every position
    from src.models.utils.modules import rotate_queries_or_keys
    grid_depth = int(N // (m.grid_size * m.grid_size))
    mp = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=device)
    d_mask, h_mask, w_mask = m.separate_positions(mp, None, None)
    s = 0
    td = rotate_queries_or_keys(tiled[..., s:s + m.d_dim], pos=d_mask); s += m.d_dim
    th = rotate_queries_or_keys(tiled[..., s:s + m.h_dim], pos=h_mask); s += m.h_dim
    tw = rotate_queries_or_keys(tiled[..., s:s + m.w_dim], pos=w_mask); s += m.w_dim
    out = torch.cat([td, th, tw, tiled[..., s:]], dim=-1) if s < m.head_dim else torch.cat([td, th, tw], dim=-1)
    return out[0, 0]  # (N,D)


def to_spatial(vec_N, slots, grid):
    """(N,) -> time-avg (grid,grid)."""
    gp = grid * grid
    return vec_N[:slots * gp].reshape(slots, grid, grid).mean(axis=0)


def corner4(m):
    g = m.shape[0]
    mn = m / (m.sum() + 1e-12)
    return float((mn[0, 0] + mn[0, -1] + mn[-1, 0] + mn[-1, -1]) / (4.0 / (g * g)))


def pearson(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--encoder-lora", required=True)
    ap.add_argument("--predictor-lora", required=True); ap.add_argument("--init-from-ckpt", required=True)
    ap.add_argument("--val-csv", required=True); ap.add_argument("--video-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=4.0); ap.add_argument("--n-eval", type=int, default=12)
    args = ap.parse_args()

    device = torch.device("cuda")
    base, _ = build_finetuned_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size,
                                 checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                 pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    enc = base.encoder
    m = enc.blocks[-1].attn
    grid = int(m.grid_size); gp = grid * grid
    val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(val_ds.rows)) if abs(float(val_ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6]
    ev = rows[: args.n_eval]
    print(f"[data] grid={grid} eval={len(ev)}", flush=True)

    # grab the input to the LAST block's attn (= norm1(x)), the exact tensor it rotates
    box = {}
    hh = m.register_forward_pre_hook(lambda mod, a, kw: box.__setitem__("x", a[0].detach()), with_kwargs=True)

    acc = {k: None for k in ["I", "recon", "b", "bpos_on", "bpos_off", "mu", "cv"]}
    scal = {k: [] for k in ["corner_I", "corner_recon", "corner_bpos_on", "corner_bpos_off",
                            "corr_I_recon", "corr_I_bpos", "cv_corner", "cv_interior", "hstar"]}
    mu_pts, cv_pts = [], []
    n = 0
    for clips in make_clip_iter(val_ds, ev, device):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc(clips)
        x = box["x"].float()
        with torch.no_grad():
            q, k, q_raw, k_raw = rotated_qk(m, x)          # (H,N,D)
            H, N, D = q.shape
            slots = N // gp
            # pick sink head on this clip = argmax corner4 of received attn
            scale = m.scale
            def recv(hh_):
                S = (q[hh_] @ k[hh_].t()) * scale
                A = S.softmax(dim=-1)                        # over keys
                return A.sum(0)                              # I_j (N,)
            hstar = int(np.argmax([corner4(to_spatial(recv(h).cpu().numpy(), slots, grid)) for h in range(H)]))
            qh, kh = q[hstar], k[hstar]                      # (N,D)
            S = (qh @ kh.t()) * scale                        # (N,N) logits
            A = S.softmax(dim=-1)                            # A_ij over keys j
            I = A.sum(0)                                     # received (N,)
            mu = A.mean(0); cv = A.std(0) / (mu + 1e-12)     # per-key query stats
            b = S.mean(0)                                    # query-averaged logit (N,)
            recon = A.shape[0] * torch.softmax(b, dim=0)     # Nq*softmax(b)
            # (3) position kernel: qbar,kbar = pre-rope per-head means; rotate & average
            qbar = q_raw[hstar].mean(0); kbar = k_raw[hstar].mean(0)   # (D,)
            qbar_hat = rot_vec(m, qbar, N, device).mean(0)            # mean_i R(p_i) qbar -> (D,)
            kbar_hat = rot_vec(m, kbar, N, device)                    # R(p_j) kbar -> (N,D)
            bpos_on = (kbar_hat @ qbar_hat) * scale                   # (N,)
            bpos_off = (kbar @ qbar).expand(N) * scale               # RoPE off: const qbar.kbar

        sp = {"I": to_spatial(I.cpu().numpy(), slots, grid),
              "recon": to_spatial(recon.cpu().numpy(), slots, grid),
              "b": to_spatial(b.cpu().numpy(), slots, grid),
              "bpos_on": to_spatial(bpos_on.cpu().numpy(), slots, grid),
              "bpos_off": to_spatial(bpos_off.cpu().numpy(), slots, grid),
              "mu": to_spatial(mu.cpu().numpy(), slots, grid),
              "cv": to_spatial(cv.cpu().numpy(), slots, grid)}
        for kk in acc:
            acc[kk] = sp[kk] if acc[kk] is None else acc[kk] + sp[kk]
        scal["corner_I"].append(corner4(sp["I"])); scal["corner_recon"].append(corner4(sp["recon"]))
        scal["corner_bpos_on"].append(corner4(np.abs(sp["bpos_on"] - sp["bpos_on"].min() + 1e-6)))
        scal["corner_bpos_off"].append(float(sp["bpos_off"].std() / (abs(sp["bpos_off"].mean()) + 1e-12)))
        scal["corr_I_recon"].append(pearson(sp["I"], sp["recon"]))
        scal["corr_I_bpos"].append(pearson(sp["I"], sp["bpos_on"]))
        cmask = np.zeros((grid, grid), bool); cmask[[0, 0, -1, -1], [0, -1, 0, -1]] = True
        imask = np.zeros((grid, grid), bool); imask[grid//4:-grid//4, grid//4:-grid//4] = True
        scal["cv_corner"].append(float(sp["cv"][cmask].mean())); scal["cv_interior"].append(float(sp["cv"][imask].mean()))
        scal["hstar"].append(hstar)
        mu_pts.append(sp["mu"].ravel()); cv_pts.append(sp["cv"].ravel())
        n += 1
    hh.remove()
    for kk in acc:
        acc[kk] /= n

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    report = {"grid": grid, "n_used": n, "sink_head_mode": int(np.bincount(scal["hstar"]).argmax())}
    for kk in ["corner_I", "corner_recon", "corner_bpos_on", "corner_bpos_off",
               "corr_I_recon", "corr_I_bpos", "cv_corner", "cv_interior"]:
        report[kk] = round(float(np.mean(scal[kk])), 4)
    (out / "sink_math.json").write_text(json.dumps(report, indent=2))
    for kk, v in acc.items():
        np.save(out / f"map_{kk}.npy", v.astype(np.float32))
    print(json.dumps(report, indent=2), flush=True)
    _plots(acc, np.concatenate(mu_pts), np.concatenate(cv_pts), grid, report, out)


def _plots(acc, mu, cv, grid, report, out):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] {e}", flush=True); return
    # spatial maps
    order = [("I", "received I_j"), ("recon", "softmax(b_j)·Nq  (bias-sink recon)"),
             ("b", "query-avg logit b_j"), ("bpos_on", "position kernel b_pos (RoPE ON)"),
             ("bpos_off", "b_pos (RoPE OFF)"), ("cv", "CV_j (query dispersion)")]
    fig, axes = plt.subplots(1, 6, figsize=(19, 3.3))
    for ax, (kk, lab) in zip(axes, order):
        im = ax.imshow(acc[kk], cmap=("viridis" if kk == "cv" else "magma"))
        ax.set_title(lab, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Sink math diagnostics (2.0 finetuned, last block, sink head {report['sink_head_mode']})  "
                 f"corr(I,recon)={report['corr_I_recon']:.2f} corr(I,b_pos)={report['corr_I_bpos']:.2f} "
                 f"CV corner {report['cv_corner']:.2f} vs interior {report['cv_interior']:.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "sink_math_maps.png", dpi=125); plt.close(fig)
    # mu vs CV scatter, colored by region
    g = grid; idx = np.arange(g * g).reshape(g, g)
    corner = np.zeros((g, g), bool); corner[[0, 0, -1, -1], [0, -1, 0, -1]] = True
    border = np.zeros((g, g), bool); border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    reg = np.where(corner.ravel(), 2, np.where(border.ravel(), 1, 0))
    fig2, ax = plt.subplots(figsize=(6, 5))
    for r, c, lab in [(0, "0.6", "interior"), (1, "tab:blue", "border"), (2, "red", "corner")]:
        s = reg == r
        ax.scatter(mu.reshape(-1, g * g).mean(0)[s], cv.reshape(-1, g * g).mean(0)[s],
                   c=c, s=(40 if r == 2 else 14), label=lab, alpha=0.8, edgecolors="k" if r == 2 else "none")
    ax.set_xlabel("mean_i A_ij  (received per query)"); ax.set_ylabel("CV_j = std_i/mean_i A_ij")
    ax.set_title("Sink = high received & LOW dispersion (query-independent)"); ax.legend()
    fig2.tight_layout(); fig2.savefig(out / "sink_math_mu_cv.png", dpi=130); plt.close(fig2)
    print("[plot] saved sink_math_maps.png + sink_math_mu_cv.png", flush=True)


if __name__ == "__main__":
    main()
