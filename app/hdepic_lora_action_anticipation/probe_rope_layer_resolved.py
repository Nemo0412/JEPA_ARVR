#!/usr/bin/env python3
# [ATTN-CORNER-SINK] B18 · Layer-resolved RoPE ablation + per-block pre/post-RoPE sweep.
# The earlier probe (job 16517340) turned RoPE OFF in EVERY block at once. This probe:
#   (A) --rope-off <spec>: disable rotation in an ARBITRARY block set (last only, one mid
#       block, top-k, all, none), run the REAL forward, measure the real last-block corner
#       sink. Answers "does RoPE at layer L carry the corner drop?".
#   (B) --sweep: RoPE fully ON, at EVERY block record received attention from BOTH the
#       unrotated q,k (pre-RoPE) and rotated q,k (post-RoPE). Locates the block where the
#       corner concentration first appears and decomposes it into content vs rotation.
# VARIANT-parametrized: finetuned20 (the PRUNING MODEL, @256) is the relevant target;
# base20 / base21 also available. Same clip set as the other mechanism probes. Slurm only.
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
    build_base_20, build_base_21, build_finetuned_20,
    HeadAttnCapture20, HeadAttnCapture21, _enable_vjepa_2_1_imports,
)
from app.hdepic_lora_action_anticipation.probe_rope_ablation import corner_stats  # noqa: E402
from app.hdepic_lora_action_anticipation.probe_rope_ablation import RoPEAblate as RoPEAblate21  # noqa: E402
from app.hdepic_lora_action_anticipation.probe_combined_ablation import RoPEAblate20  # noqa: E402
from app.hdepic_lora_action_anticipation.vjepa_testtime_register import make_clip_iter  # noqa: E402


def _received(q, k, scale, chunk=256):
    """Column-summed softmax received attention per head: (B, heads, N)."""
    B, Hh, N, _ = q.shape
    imp = torch.zeros(B, Hh, N, device=q.device, dtype=torch.float32)
    for ci in range(0, N, chunk):
        qc = q[:, :, ci:ci + chunk, :]
        imp += ((qc @ k.transpose(-2, -1)) * scale).softmax(-1).sum(dim=2).float()
    return imp


class PrePostRoPECapture20:
    """2.0 RoPEAttention: record received attn from unrotated (pre) and rotated (post)
    q,k. Rotation mirrors HeadAttnCapture20 byte-for-byte. Original forward runs."""

    def __init__(self, attn, chunk=256):
        from src.models.utils.modules import rotate_queries_or_keys
        m = attn
        self._m = m
        self._orig = m.forward
        self.imp_pre = None
        self.imp_post = None
        cap = self

        def _fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            out = cap._orig(x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
            with torch.no_grad():
                B, N, C = x.size()
                grid_depth = int(N // (m.grid_size * m.grid_size))
                qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv[0], qkv[1], qkv[2]
                cap.imp_pre = _received(q, k, m.scale, chunk)
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
                    qr = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
                    kr = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
                else:
                    qr = torch.cat([qd, qh, qw], dim=-1)
                    kr = torch.cat([kd, kh, kw], dim=-1)
                cap.imp_post = _received(qr, kr, m.scale, chunk)
            return out

        m.forward = _fwd

    def remove(self):
        self._m.forward = self._orig


class PrePostRoPECapture21:
    """2.1 RoPEAttention: pre/post received attention. Rotation mirrors HeadAttnCapture21."""

    def __init__(self, attn, chunk=256):
        _enable_vjepa_2_1_imports()
        from app.vjepa_2_1.models.utils.modules import rotate_queries_or_keys
        m = attn
        self._m = m
        self._orig = m.forward
        self.imp_pre = None
        self.imp_post = None
        cap = self

        def _fwd(x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False):
            out = cap._orig(x, mask=mask, T=T, H_patches=H_patches, W_patches=W_patches, return_attn=return_attn)
            with torch.no_grad():
                B, N, C = x.size()
                N_ctx = N - m.n_registers
                grid_depth = int(N_ctx // (m.grid_size * m.grid_size))
                qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv[0], qkv[1], qkv[2]
                cap.imp_pre = _received(q, k, m.scale, chunk)
                if T is None or H_patches is None or W_patches is None:
                    mp = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=x.device)
                else:
                    mp = torch.arange(int(T * H_patches * W_patches), device=x.device)
                d_mask, h_mask, w_mask = m.separate_positions(mp, H_patches, W_patches)
                if m.interpolate_rope:
                    Hp = int(m.grid_size) if H_patches is None else H_patches
                    Wp = int(m.grid_size) if W_patches is None else W_patches
                    h_mask = h_mask * (m.pretrained_grid_size - 1) / (Hp - 1)
                    w_mask = w_mask * (m.pretrained_grid_size - 1) / (Wp - 1)
                rk = dict(n_registers=m.n_registers, has_cls_first=m.has_cls_first)
                s = 0
                qd = rotate_queries_or_keys(q[..., s:s + m.d_dim], pos=d_mask, **rk)
                kd = rotate_queries_or_keys(k[..., s:s + m.d_dim], pos=d_mask, **rk); s += m.d_dim
                qh = rotate_queries_or_keys(q[..., s:s + m.h_dim], pos=h_mask, **rk)
                kh = rotate_queries_or_keys(k[..., s:s + m.h_dim], pos=h_mask, **rk); s += m.h_dim
                qw = rotate_queries_or_keys(q[..., s:s + m.w_dim], pos=w_mask, **rk)
                kw = rotate_queries_or_keys(k[..., s:s + m.w_dim], pos=w_mask, **rk); s += m.w_dim
                if s < m.head_dim:
                    qr = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
                    kr = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
                else:
                    qr = torch.cat([qd, qh, qw], dim=-1)
                    kr = torch.cat([kd, kh, kw], dim=-1)
                cap.imp_post = _received(qr, kr, m.scale, chunk)
            return out

        m.forward = _fwd

    def remove(self):
        self._m.forward = self._orig


def parse_spec(spec, n_blocks):
    """'none'|'all'|'last'|'prefix:L'|'suffix:L'|comma list -> sorted set of de-RoPE indices.
    prefix:L = de-RoPE blocks {0..L-1} (RoPE turned ON only from block L onward).
    suffix:L = de-RoPE blocks {L..n-1} (RoPE turned OFF from block L onward)."""
    spec = spec.strip().lower()
    if spec in ("none", ""):
        return set()
    if spec == "all":
        return set(range(n_blocks))
    if spec == "last":
        return {n_blocks - 1}
    if spec.startswith("prefix:"):
        return set(range(0, int(spec.split(":")[1])))
    if spec.startswith("suffix:"):
        return set(range(int(spec.split(":")[1]), n_blocks))
    out = set()
    for tok in spec.replace(" ", "").split(","):
        i = int(tok)
        out.add(i if i >= 0 else n_blocks + i)
    return out


# ── (C) prefix / suffix cumulative RoPE scan (measurement always last block) ───
def run_scan(enc, clip_iter_fn, grid, gp, n_blocks, Ablate, Capture, kinds, step, out_dir):
    """Sweep the RoPE-active RANGE and measure the last-block corner sink.
    suffix: de-RoPE {L..n-1} — 'turn RoPE OFF from block L onward' (front kept on).
    prefix: de-RoPE {0..L-1} — 'turn RoPE ON only from block L onward' (front off).
    L runs 0..n_blocks (inclusive) at `step` (+endpoints); L=none->all-on, other end->all-off."""
    Ls = sorted(set(list(range(0, n_blocks + 1, step)) + [n_blocks]))
    results = {}
    for kind in kinds:
        pts = []
        for L in Ls:
            off = set(range(L, n_blocks)) if kind == "suffix" else set(range(0, L))
            cb, pk = run_ablation(enc, clip_iter_fn, grid, gp, off, n_blocks, Ablate, Capture)
            pts.append({"L": L, "n_off": len(off), "corner": round(cb, 2), "peak": round(pk, 2)})
            print(f"[scan:{kind}] L={L:2d} off={len(off):2d} -> corner={cb:.2f} peak={pk:.2f}", flush=True)
        results[kind] = pts
    if out_dir is not None:
        _plot_scan(results, n_blocks, out_dir)
    return results


def _plot_scan(results, n_blocks, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e})", flush=True)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {"suffix": "suffix: de-RoPE [L..last]  (RoPE OFF from block L on)",
              "prefix": "prefix: de-RoPE [0..L-1]  (RoPE ON only from block L on)"}
    styles = {"suffix": "s-", "prefix": "o-"}
    base = allv = None
    for kind, pts in results.items():
        xs = [p["L"] for p in pts]; ys = [p["corner"] for p in pts]
        ax.plot(xs, ys, styles.get(kind, "d-"), label=labels.get(kind, kind))
        for p in pts:
            if p["n_off"] == 0:
                base = p["corner"]
            if p["n_off"] == n_blocks:
                allv = p["corner"]
    if base is not None:
        ax.axhline(base, color="green", ls=":", lw=1, label=f"RoPE all-on (baseline) {base}")
    if allv is not None:
        ax.axhline(allv, color="red", ls=":", lw=1, label=f"RoPE all-off {allv}")
    ax.set_xlabel("layer L"); ax.set_ylabel("last-block corner-block / uniform")
    ax.set_title("Cumulative RoPE on/off range vs last-block corner sink")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "rope_prefix_suffix_scan.png", dpi=130)
    plt.close(fig)
    print(f"[plot] saved rope_prefix_suffix_scan.png", flush=True)


# ── (A) layer-resolved ablation: RoPE off in a chosen block set ────────────────
def run_ablation(enc, clip_iter_fn, grid, gp, off_set, n_blocks, Ablate, Capture):
    last = n_blocks - 1
    # De-RoPE ONLY the blocks in off_set. Measure the real last-block received attention:
    # if the last block is itself de-RoPE'd, capture on its no-rotation path.
    abls = [Ablate(enc.blocks[i].attn, capture=(i == last)) for i in sorted(off_set)]
    if last in off_set:
        cap = next(a for a, i in zip(abls, sorted(off_set)) if i == last)
        cap_is_ablate = True
    else:
        cap = Capture(enc.blocks[last].attn)
        cap_is_ablate = False

    cbs, pks = [], []
    for clips in clip_iter_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc(clips)
        imp = cap.importance[0].float().cpu().numpy()
        cb, pk, _ = corner_stats(imp, grid, gp)
        cbs.append(cb); pks.append(pk)

    for a in abls:
        a.remove()
    if not cap_is_ablate:
        cap.remove()
    return float(np.mean(cbs)), float(np.mean(pks))


# ── (D) layerwise profile under a fixed RoPE config (switch-boundary view) ─────
def _map(imp, grid, gp):
    """(H,N) received attn -> per-head time-avg spatial map (H,grid,grid), each head sums to 1."""
    H, N = imp.shape
    slots = N // gp
    sp = imp[:, :slots * gp].reshape(H, slots, grid, grid).sum(axis=1)
    return sp / (sp.sum(axis=(1, 2), keepdims=True) + 1e-12)


def _corner4(m):
    g = m.shape[0]
    return float((m[0, 0] + m[0, -1] + m[-1, 0] + m[-1, -1]) / (4.0 / (g * g)))


# ── (E) LAST-layer received-attn map under several RoPE configs (side-by-side) ─
def _last_importance(enc, clip_iter_fn, off_set, n_blocks, Ablate, Capture, grid, gp):
    """Average per-head last-block spatial map (H,grid,grid) under a fixed RoPE config."""
    last = n_blocks - 1
    abls = [Ablate(enc.blocks[i].attn, capture=(i == last)) for i in sorted(off_set)]
    if last in off_set:
        cap = next(a for a, i in zip(abls, sorted(off_set)) if i == last); cap_is_ablate = True
    else:
        cap = Capture(enc.blocks[last].attn); cap_is_ablate = False
    acc = None; n = 0
    for clips in clip_iter_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc(clips)
        m = _map(cap.importance[0].float().cpu().numpy(), grid, gp)  # (H,grid,grid)
        acc = m if acc is None else acc + m; n += 1
    for a in abls:
        a.remove()
    if not cap_is_ablate:
        cap.remove()
    return acc / n


def run_lastlayer_maps(enc, clip_iter_fn, grid, gp, n_blocks, specs, Ablate, Capture, out_dir):
    maps = {s: _last_importance(enc, clip_iter_fn, parse_spec(s, n_blocks), n_blocks, Ablate, Capture, grid, gp)
            for s in specs}
    hstar = int(np.argmax([_corner4(maps[specs[0]][h]) for h in range(maps[specs[0]].shape[0])]))
    report = {s: {"off_blocks": sorted(parse_spec(s, n_blocks)),
                  "corner_sinkhead": round(_corner4(maps[s][hstar]), 3),
                  "corner_headavg": round(_corner4(maps[s].mean(0)), 3)} for s in specs}
    if out_dir is not None:
        _plot_lastlayer_maps(maps, specs, hstar, grid, out_dir)
        for s in specs:
            np.save(Path(out_dir) / f"lastmap_{s.replace(':','')}.npy", maps[s].astype(np.float32))
    return {"sink_head": hstar, "configs": report}


def _cfg_label(s, n_blocks):
    s = s.strip().lower()
    if s in ("none", ""):
        return "all RoPE ON"
    if s == "all":
        return "all RoPE OFF"
    if s.startswith("suffix:"):
        L = int(s.split(":")[1]); return f"OFF [{L}..{n_blocks-1}]\n(front on)"
    if s.startswith("prefix:"):
        L = int(s.split(":")[1]); return f"OFF [0..{L-1}]\n(back on)"
    if s == "last":
        return f"OFF last ({n_blocks-1})"
    return f"OFF {s}"


def _plot_lastlayer_maps(maps, specs, hstar, grid, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] {e}", flush=True); return
    n_blocks = 24
    ncol = len(specs)
    fig, axes = plt.subplots(2, ncol, figsize=(2.15 * ncol, 4.7), squeeze=False)
    sink_stack = np.array([maps[s][hstar] for s in specs])
    avg_stack = np.array([maps[s].mean(0) for s in specs])
    vmax0 = np.percentile(sink_stack, 99.0); vmax1 = np.percentile(avg_stack, 99.0)
    for c, s in enumerate(specs):
        axes[0][c].imshow(maps[s][hstar], cmap="magma", vmin=0, vmax=vmax0)
        axes[0][c].set_title(f"{_cfg_label(s, n_blocks)}\nhead {hstar}  c{_corner4(maps[s][hstar]):.2f}", fontsize=7)
        axes[1][c].imshow(maps[s].mean(0), cmap="magma", vmin=0, vmax=vmax1)
        axes[1][c].set_title(f"head-avg  c{_corner4(maps[s].mean(0)):.2f}", fontsize=7)
        for r in (0, 1):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    axes[0][0].set_ylabel(f"sink head {hstar}", fontsize=9)
    axes[1][0].set_ylabel("head-avg", fontsize=9)
    fig.suptitle("LAST-layer received attention under different RoPE configs (2.0 finetuned)", y=1.01)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "lastlayer_maps.png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved lastlayer_maps.png (sink head {hstar})", flush=True)


def run_profile(enc, clip_iter_fn, grid, gp, n_blocks, off_set, Ablate, Capture, out_dir, tag):
    """Under a FIXED RoPE config (off_set de-RoPE'd), measure the corner sink at EVERY block.
    RoPE-on blocks use HeadAttnCapture (real rotated q,k); RoPE-off blocks use RoPEAblate
    (unrotated q,k) — each captures the block's ACTUAL q,k. Shows how corner strength moves
    across the RoPE on/off switch boundary (does the sink persist via the residual stream?)."""
    caps = [Ablate(enc.blocks[i].attn, capture=True) if i in off_set else Capture(enc.blocks[i].attn)
            for i in range(n_blocks)]
    corner = np.zeros(n_blocks); peak = np.zeros(n_blocks)
    maps = None; n = 0
    for clips in clip_iter_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc(clips)
        for i in range(n_blocks):
            imp = caps[i].importance[0].float().cpu().numpy()
            cb, pk, _ = corner_stats(imp, grid, gp)
            corner[i] += cb; peak[i] += pk
            m = _map(imp, grid, gp)
            if maps is None:
                maps = np.zeros((n_blocks,) + m.shape)
            maps[i] += m
        n += 1
    for c in caps:
        c.remove()
    corner /= n; peak /= n; maps /= n
    if out_dir is not None:
        _plot_profile(corner, peak, maps, off_set, n_blocks, grid, out_dir, tag)
    return {"off_blocks": sorted(off_set), "corner_by_layer": corner.round(3).tolist(),
            "peak_by_layer": peak.round(2).tolist()}


def _plot_profile(corner, peak, maps, off_set, n_blocks, grid, out_dir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e})", flush=True)
        return
    boundaries = [i for i in range(1, n_blocks) if (i - 1 in off_set) != (i in off_set)]
    # corner-vs-layer line, shading RoPE-off blocks, marking switch boundaries
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(range(n_blocks), corner, "o-", color="tab:blue")
    for i in sorted(off_set):
        ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.08)
    for b in boundaries:
        ax.axvline(b - 0.5, color="k", ls="--", lw=1)
        ax.annotate(f"RoPE switch @L{b}", xy=(b - 0.5, ax.get_ylim()[1]), fontsize=8,
                    ha="center", va="top", rotation=90)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("encoder block (red shade = RoPE OFF)")
    ax.set_ylabel("corner-block / uniform")
    ax.set_title(f"Per-layer corner sink under config '{tag}' (de-RoPE {sorted(off_set)[:1]}..)")
    fig.tight_layout(); fig.savefig(Path(out_dir) / f"profile_{tag}_corner_vs_layer.png", dpi=130)
    plt.close(fig)

    # all-layer 2D maps (sink head), on/off in title, boundary highlighted
    hstar = int(np.argmax([_corner4(maps[-1, h]) for h in range(maps.shape[1])]))
    ncol = 6; nrow = int(np.ceil(n_blocks / ncol))
    fig2, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.1 * nrow), squeeze=False)
    vmax = np.percentile(maps[:, hstar], 99.5)
    for i in range(nrow * ncol):
        ax = axes.flat[i]
        if i < n_blocks:
            ax.imshow(maps[i, hstar], cmap="magma", vmin=0, vmax=vmax)
            st = "OFF" if i in off_set else "ON"
            col = "red" if i in off_set else "black"
            edge = "lime" if i in boundaries or (i + 1 in boundaries) else None
            ax.set_title(f"L{i} RoPE {st}  c{_corner4(maps[i, hstar]):.1f}", fontsize=7, color=col)
            if edge:
                for s in ax.spines.values():
                    s.set_color(edge); s.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
    fig2.suptitle(f"Per-layer received-attn map (sink head {hstar}) under '{tag}'", y=1.0)
    fig2.tight_layout()
    fig2.savefig(Path(out_dir) / f"profile_{tag}_maps.png", dpi=110, bbox_inches="tight")
    plt.close(fig2)
    print(f"[plot] saved profile_{tag}_corner_vs_layer.png + _maps.png (sink head {hstar}, boundaries {boundaries})", flush=True)


# ── (B) per-block pre/post-RoPE sweep (RoPE fully ON) ──────────────────────────
def run_sweep(enc, clip_iter_fn, grid, gp, n_blocks, PrePost):
    caps = [PrePost(enc.blocks[i].attn) for i in range(n_blocks)]
    pre_cb = np.zeros(n_blocks); post_cb = np.zeros(n_blocks)
    pre_pk = np.zeros(n_blocks); post_pk = np.zeros(n_blocks)
    n = 0
    for clips in clip_iter_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc(clips)
        for i in range(n_blocks):
            cbpre, pkpre, _ = corner_stats(caps[i].imp_pre[0].float().cpu().numpy(), grid, gp)
            cbpost, pkpost, _ = corner_stats(caps[i].imp_post[0].float().cpu().numpy(), grid, gp)
            pre_cb[i] += cbpre; post_cb[i] += cbpost
            pre_pk[i] += pkpre; post_pk[i] += pkpost
        n += 1
    for c in caps:
        c.remove()
    return {
        "pre_corner_block": (pre_cb / n).round(3).tolist(),
        "post_corner_block": (post_cb / n).round(3).tolist(),
        "pre_peak": (pre_pk / n).round(2).tolist(),
        "post_peak": (post_pk / n).round(2).tolist(),
    }


def build(args, device):
    if args.variant == "base21":
        base, kind = build_base_21(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size, checkpoint=args.checkpoint)
    elif args.variant == "base20":
        base, kind = build_base_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size, checkpoint=args.checkpoint)
    else:
        base, kind = build_finetuned_20(device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size,
                                        checkpoint=args.checkpoint, enc_lora=args.encoder_lora,
                                        pred_lora=args.predictor_lora, parent_ckpt=args.init_from_ckpt)
    return base, kind


def main():
    ap = argparse.ArgumentParser(description="[ATTN-CORNER-SINK] layer-resolved RoPE ablation + pre/post sweep")
    ap.add_argument("--variant", default="finetuned20", choices=["base21", "base20", "finetuned20"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--encoder-lora", default=None)
    ap.add_argument("--predictor-lora", default=None)
    ap.add_argument("--init-from-ckpt", default=None)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--context-sec", type=float, default=4.0)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--rope-off", default="", help="';'-separated specs; each: comma list / none / all / last / prefix:L / suffix:L")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--scan", default="none", choices=["none", "prefix", "suffix", "both"],
                    help="cumulative RoPE range scan (measurement = last block)")
    ap.add_argument("--scan-step", type=int, default=2)
    ap.add_argument("--profile", default="", help="';'-sep configs (e.g. 'suffix:12;prefix:12'): "
                    "per-LAYER corner + 2D maps under that fixed RoPE config (switch-boundary view)")
    ap.add_argument("--last-maps", default="", help="';'-sep configs: LAST-layer 2D attn map per config, side by side")
    args = ap.parse_args()

    device = torch.device("cuda")
    base, kind = build(args, device)
    enc = base.encoder
    Ablate = RoPEAblate21 if kind == "21" else RoPEAblate20
    Capture = HeadAttnCapture21 if kind == "21" else HeadAttnCapture20
    PrePost = PrePostRoPECapture21 if kind == "21" else PrePostRoPECapture20
    n_blocks = len(enc.blocks)
    grid = int(enc.blocks[-1].attn.grid_size); gp = grid * grid
    val_ds = FpsSubsampledStreamMTPDataset(args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
    rows = [i for i in range(len(val_ds.rows)) if abs(float(val_ds.rows[i]["context_sec"]) - args.context_sec) < 1e-6]
    ev = rows[: args.n_eval]
    print(f"[data] variant={args.variant} kind={kind} blocks={n_blocks} grid={grid} eval={len(ev)}", flush=True)

    def clip_iter_fn():
        return make_clip_iter(val_ds, ev, device)

    report = {"model": args.variant, "kind": kind, "grid": grid, "n_blocks": n_blocks, "n_eval": len(ev),
              "img_size": args.img_size, "max_frames": args.max_frames, "context_sec": args.context_sec}

    outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)

    # fixed-set ablation: only if explicitly requested, or if not scanning/profiling/last-maps
    if args.rope_off or (args.scan == "none" and not args.profile and not args.last_maps):
        specs = args.rope_off if args.rope_off else "none;last;-8;-12;all"
        abl = {}
        for spec in specs.split(";"):
            off = parse_spec(spec, n_blocks)
            cb, pk = run_ablation(enc, clip_iter_fn, grid, gp, off, n_blocks, Ablate, Capture)
            key = spec.strip().lower() or "none"
            abl[key] = {"off_blocks": sorted(off), "corner_block_over_uniform": round(cb, 2), "patch_sink_peak": round(pk, 2)}
            print(f"[ablate] rope-off={key:>6} blocks={sorted(off)} -> corner={cb:.2f} peak={pk:.2f}", flush=True)
        report["ablation"] = abl

    # (C) cumulative prefix/suffix RoPE range scan (measurement = last block)
    if args.scan != "none":
        kinds = ["prefix", "suffix"] if args.scan == "both" else [args.scan]
        report["scan"] = run_scan(enc, clip_iter_fn, grid, gp, n_blocks, Ablate, Capture,
                                  kinds, args.scan_step, outp.parent)

    # (E) last-layer 2D attn map under several RoPE configs, side by side
    if args.last_maps:
        specs = [s.strip() for s in args.last_maps.split(";") if s.strip()]
        report["last_maps"] = run_lastlayer_maps(enc, clip_iter_fn, grid, gp, n_blocks, specs,
                                                 Ablate, Capture, outp.parent)
        print(f"[last-maps] {json.dumps(report['last_maps']['configs'])}", flush=True)

    # (D) per-layer profile under a fixed RoPE config (switch-boundary view)
    if args.profile:
        prof = {}
        for spec in args.profile.split(";"):
            off = parse_spec(spec, n_blocks)
            tag = spec.strip().lower().replace(":", "")
            prof[spec.strip().lower()] = run_profile(enc, clip_iter_fn, grid, gp, n_blocks, off,
                                                     Ablate, Capture, outp.parent, tag)
            print(f"[profile] {spec}: corner_by_layer={prof[spec.strip().lower()]['corner_by_layer']}", flush=True)
        report["profile"] = prof

    if args.sweep:
        report["sweep"] = run_sweep(enc, clip_iter_fn, grid, gp, n_blocks, PrePost)
        pc = report["sweep"]["post_corner_block"]
        report["sweep"]["first_block_corner_gt_1p5"] = next((i for i, v in enumerate(pc) if v > 1.5), None)
        print(f"[sweep] post corner/block by layer: {pc}", flush=True)

    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
