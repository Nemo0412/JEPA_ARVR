#!/usr/bin/env python3
"""Ablate: is L23 corner sink about spatial POSITION (token id / RoPE) or VIDEO CONTENT?

Transforms applied to the same HD-EPIC clip *before* the encoder:
  A) original
  B) spatial roll by half-grid (content moves; geometric corners stay corners)
  C) fixed random permute of 16x16 patches (same perm all frames)
  D) constant gray video (no content)
  E) i.i.d. noise video (no structure)

Prediction:
  - If POSITION/token-id: geometric corners stay high under B/C/D/E
  - If CONTENT: mass follows moved patches under B/C; collapses under D/E
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/home/ll5914/Jepa_yifan/JEPA_ARVR")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
os.environ.setdefault(
    "TRI_MODAL_FRAME_CACHE",
    "/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1",
)

from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mc", PROJECT_ROOT / "scripts/analyze_encoder_last_layer_multiclip.py"
)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

OUT = Path("/home/ll5914/Jepa_yifan/encoder_L23_hdepic_sink_example/spatial_vs_content")
TARGET_VID = "P01_20240202-110250"
FRAMES = 32
FPS = 8
TUBELET = 2
T_SLOTS = FRAMES // TUBELET
GRID = 16
GP = GRID * GRID
PATCH = 16  # img 256 / grid 16
CHUNK = 128
CORNER_YX = ((0, 0), (0, 15), (15, 0), (15, 15))
CORNER_FLAT = [y * GRID + x for y, x in CORNER_YX]
CORNER_NAMES = ["TL", "TR", "BL", "BR"]
FOCUS_HEADS = [3, 1, 8, 12]  # sink-ish + content-ish
SEED = 0


def apply_rope(attn, q, k, T=None, Hp=None, Wp=None):
    m = attn
    _B, _H, N, _D = q.shape
    gd = int(N // (m.grid_size * m.grid_size))
    if T is None or Hp is None or Wp is None:
        mask_p = torch.arange(int(gd * m.grid_size * m.grid_size), device=q.device)
    else:
        mask_p = torch.arange(int(T * Hp * Wp), device=q.device)
    d_mask, h_mask, w_mask = m.separate_positions(mask_p, Hp, Wp)
    s = 0
    qd = rotate_queries_or_keys(q[..., s : s + m.d_dim], pos=d_mask)
    kd = rotate_queries_or_keys(k[..., s : s + m.d_dim], pos=d_mask)
    s += m.d_dim
    qh = rotate_queries_or_keys(q[..., s : s + m.h_dim], pos=h_mask)
    kh = rotate_queries_or_keys(k[..., s : s + m.h_dim], pos=h_mask)
    s += m.h_dim
    qw = rotate_queries_or_keys(q[..., s : s + m.w_dim], pos=w_mask)
    kw = rotate_queries_or_keys(k[..., s : s + m.w_dim], pos=w_mask)
    s += m.w_dim
    if s < m.head_dim:
        return (
            torch.cat([qd, qh, qw, q[..., s:]], dim=-1),
            torch.cat([kd, kh, kw, k[..., s:]], dim=-1),
        )
    return torch.cat([qd, qh, qw], dim=-1), torch.cat([kd, kh, kw], dim=-1)



def attn_mass_from_qk(q, k, scale, chunk=CHUNK):
    h, n, _ = q.shape
    out = torch.zeros(h, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)
    for ci in range(0, n, chunk):
        logits = torch.bmm(q[:, ci : ci + chunk].float(), k_t) * scale
        out += logits.softmax(dim=-1).sum(dim=1)
    return out


def yx_of(flat: int):
    return divmod(flat, GRID)


def transform_clip(clip: torch.Tensor, name: str, rng: np.random.Generator):
    """clip [C,T,H,W] -> transformed clip, plus content_src map [256] (for tracking).

    content_src[j] = original spatial index whose *content* now sits at j.
    identity: content_src[j]=j
    """
    c, t, h, w = clip.shape
    assert h == GRID * PATCH and w == GRID * PATCH
    content_src = np.arange(GP, dtype=np.int64)

    if name == "original":
        return clip.clone(), content_src

    if name == "roll_half":
        # shift by 8 patches = 128 px → geometric corners get former mid content
        dy = dx = 8 * PATCH
        out = torch.roll(clip, shifts=(dy, dx), dims=(-2, -1))
        # content now at (y,x) came from ((y-8)%16, (x-8)%16)
        for j in range(GP):
            y, x = yx_of(j)
            content_src[j] = ((y - 8) % GRID) * GRID + ((x - 8) % GRID)
        return out, content_src

    if name == "permute_patches":
        perm = rng.permutation(GP)  # new_pos -> old_pos content: content at j came from perm[j]
        # build patch tensor [T,16,16,C,P,P]
        x = clip.permute(1, 0, 2, 3)  # T,C,H,W
        patches = (
            x.unfold(2, PATCH, PATCH)
            .unfold(3, PATCH, PATCH)
            .contiguous()
        )  # T,C,16,16,P,P
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # T,Gy,Gx,C,P,P
        flat = patches.reshape(t, GP, c, PATCH, PATCH)
        # place: position j gets content from perm[j]
        flat2 = flat[:, perm]
        patches2 = flat2.reshape(t, GRID, GRID, c, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5)
        out = patches2.reshape(t, c, h, w).permute(1, 0, 2, 3).contiguous()
        content_src = perm.astype(np.int64)
        return out, content_src

    if name == "gray":
        # destroy content; keep shape. Use ImageNet-ish mid gray in model input space
        # clip is already normalized by VideoTransform — use mean over clip as constant
        mean = clip.mean(dim=(1, 2, 3), keepdim=True)
        return mean.expand_as(clip).clone(), content_src

    if name == "noise":
        # i.i.d. noise matching clip mean/std
        mu = clip.mean()
        std = clip.std().clamp_min(1e-3)
        out = torch.empty_like(clip).normal_(float(mu), float(std))
        return out, content_src

    raise ValueError(name)


def capture_mass(model, encoder, clip_b, ant):
    last = len(encoder.blocks) - 1
    block = encoder.blocks[last]
    attn = block.attn
    bag = {}
    orig = block.forward

    def wrapped(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        with torch.no_grad():
            qkv = attn.qkv(block.norm1(x)).unflatten(-1, (3, attn.num_heads, -1)).permute(
                2, 0, 3, 1, 4
            )
            q, k = apply_rope(attn, qkv[0], qkv[1], T, H_patches, W_patches)
            bag["mass"] = attn_mass_from_qk(q[0], k[0], float(attn.scale)).float().cpu()
            bag["N"] = int(x.shape[1])
            bag["H"] = int(attn.num_heads)
        raise RuntimeError("MASS_CAPTURE_DONE")

    block.forward = wrapped
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = model(clip_b, ant)
    except RuntimeError as exc:
        if "MASS_CAPTURE_DONE" not in str(exc):
            block.forward = orig
            raise
    finally:
        block.forward = orig
    return bag["mass"].numpy()  # [H,N]


def summarize(mass: np.ndarray, content_src: np.ndarray, cond: str):
    H, N = mass.shape
    assert N == T_SLOTS * GP
    M = mass.reshape(H, T_SLOTS, GP)  # [H,T,S]
    # time-mean spatial
    spat = M.mean(axis=1)  # [H,256]
    # where content from original corners landed
    dest_of_corner = {}
    for name, c in zip(CORNER_NAMES, CORNER_FLAT):
        # find j such that content_src[j]==c
        hits = np.where(content_src == c)[0]
        dest_of_corner[name] = int(hits[0]) if len(hits) else c

    rows = []
    for h in range(H):
        s = spat[h]
        corner_mean = float(s[CORNER_FLAT].mean())
        other_mean = float(np.delete(s, CORNER_FLAT).mean())
        # mass at positions that now hold original corner *content*
        content_corner_idx = [dest_of_corner[n] for n in CORNER_NAMES]
        content_corner_mean = float(s[content_corner_idx].mean())
        top4 = np.argsort(-s)[:4]
        rows.append(
            {
                "cond": cond,
                "head": h,
                "geom_corner_mean": corner_mean,
                "content_corner_mean": content_corner_mean,
                "other_mean": other_mean,
                "geom_ratio": corner_mean / (other_mean + 1e-12),
                "content_ratio": content_corner_mean / (other_mean + 1e-12),
                "top4": [int(i) for i in top4],
                "top4_yx": [yx_of(int(i)) for i in top4],
                "top4_is_geom_corner": [int(i) in CORNER_FLAT for i in top4],
                "n_geom_in_top4": int(sum(int(i) in CORNER_FLAT for i in top4)),
            }
        )
    return {
        "cond": cond,
        "content_src_dest_of_orig_corners": dest_of_corner,
        "heads": rows,
        "spat_mean_over_heads": spat.mean(0),
        "spat_h3": spat[3] if H > 3 else spat[0],
        "M_h3": M[3] if H > 3 else M[0],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    class A:
        pass

    args = A()
    args.n_sample = 1
    args.chunk = CHUNK
    args.frames = FRAMES
    args.fps = FPS
    args.img_size = 256
    args.anticipation = 1.0
    args.clips_dir = ""
    args.val_csv = (
        "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/"
        "clip_split/HD_EPIC_val_vjepa.csv"
    )
    args.video_root = "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos"
    args.vjepa_ckpt = "/scratch/ll5914/models/vjepa2/vitl.pt"
    args.video_enc_lora = (
        "/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/"
        "action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/"
        "encoder_lora_best.pt"
    )
    args.video_pred_lora = (
        "/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/"
        "action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/"
        "predictor_lora_best.pt"
    )

    df = pd.read_csv(args.val_csv)
    rows = df[df["video_id"].astype(str) == TARGET_VID].head(1)
    tmp = Path(tempfile.gettempdir()) / f"svc_{os.getpid()}.csv"
    rows.to_csv(tmp, index=False)
    args.val_csv = str(tmp)

    device = torch.device("cuda")
    samples = mc.load_diverse_clips(args)
    tmp.unlink(missing_ok=True)
    clip0, meta = samples[0]
    vid = meta["video_id"]
    assert clip0.shape[1] == FRAMES, clip0.shape
    print("loaded", vid, tuple(clip0.shape), flush=True)

    model = mc._fastgen.load_video_model(args, device)
    core = model.base_model if hasattr(model, "base_model") else model
    encoder = core.encoder
    ant = torch.full((1,), 1.0, device=device)

    conditions = ["original", "roll_half", "permute_patches", "gray", "noise"]
    results = {}
    spat_maps = {}

    for cond in conditions:
        print("===", cond, flush=True)
        clip_t, content_src = transform_clip(clip0, cond, rng)
        clip_b = clip_t.unsqueeze(0).to(device=device, dtype=torch.float32)
        mass = capture_mass(model, encoder, clip_b, ant)
        np.save(OUT / f"mass_{cond}_HxN.npy", mass)
        summary = summarize(mass, content_src, cond)
        results[cond] = {
            "content_src_dest_of_orig_corners": summary["content_src_dest_of_orig_corners"],
            "heads": summary["heads"],
        }
        spat_maps[cond] = summary["spat_h3"]
        np.save(OUT / f"spat_h3_{cond}.npy", summary["spat_h3"])
        np.save(OUT / f"M_h3_{cond}_Tx256.npy", summary["M_h3"])
        # quick print focus heads
        for r in summary["heads"]:
            if r["head"] in FOCUS_HEADS:
                print(
                    f"  h{r['head']}: geom_ratio={r['geom_ratio']:.2f} "
                    f"content_ratio={r['content_ratio']:.2f} "
                    f"geom_in_top4={r['n_geom_in_top4']} top4_yx={r['top4_yx']}",
                    flush=True,
                )

    # aggregate table: mean over sink-like heads (geom_ratio>2 on original)
    orig_heads = results["original"]["heads"]
    sink_heads = [r["head"] for r in orig_heads if r["geom_ratio"] >= 2.5]
    if not sink_heads:
        sink_heads = [3, 1, 12]

    lines = []
    lines.append(f"video={vid} frames={FRAMES} (~{FRAMES/FPS:.0f}s) L23 spatial-vs-content ablation")
    lines.append(f"sink-like heads (orig geom_ratio>=2.5): {sink_heads}")
    lines.append("")
    lines.append(
        "If POSITION/token-id matters: geom_ratio stays high under roll/permute/gray/noise;"
    )
    lines.append("content_ratio should NOT win over geom under roll/permute.")
    lines.append(
        "If CONTENT matters: under roll/permute, content_ratio >> geom_ratio; gray/noise kill sink."
    )
    lines.append("")
    hdr = f"{'cond':16s} {'geom_r':>8s} {'cont_r':>8s} {'#top4c':>7s}  note"
    lines.append(hdr)
    verdict_scores = {"position": 0, "content": 0}
    for cond in conditions:
        hs = [r for r in results[cond]["heads"] if r["head"] in sink_heads]
        geom_r = float(np.mean([r["geom_ratio"] for r in hs]))
        cont_r = float(np.mean([r["content_ratio"] for r in hs]))
        ntop = float(np.mean([r["n_geom_in_top4"] for r in hs]))
        note = ""
        if cond == "original":
            note = "baseline"
        elif cond in ("roll_half", "permute_patches"):
            if geom_r > cont_r * 1.3 and geom_r > 2:
                note = "POSITION wins (geom > content)"
                verdict_scores["position"] += 1
            elif cont_r > geom_r * 1.3 and cont_r > 2:
                note = "CONTENT wins (follows moved patches)"
                verdict_scores["content"] += 1
            else:
                note = "mixed/weak"
        elif cond in ("gray", "noise"):
            if geom_r > 2:
                note = "POSITION (sink without video content)"
                verdict_scores["position"] += 1
            else:
                note = "needs content (sink collapsed)"
                verdict_scores["content"] += 1
        lines.append(f"{cond:16s} {geom_r:8.2f} {cont_r:8.2f} {ntop:7.2f}  {note}")

    lines.append("")
    if verdict_scores["position"] > verdict_scores["content"]:
        lines.append(
            "VERDICT: spatial POSITION / token grid id (RoPE geometry) drives the corner sink, "
            "not the specific video pixels at those patches."
        )
    elif verdict_scores["content"] > verdict_scores["position"]:
        lines.append(
            "VERDICT: VIDEO CONTENT at those patches drives the sink more than fixed token ids."
        )
    else:
        lines.append("VERDICT: mixed — both position bias and content contribute.")
    lines.append(
        f"score position={verdict_scores['position']} content={verdict_scores['content']}"
    )

    # per-head full table for h3
    lines.append("")
    lines.append("Detail head h3:")
    for cond in conditions:
        r = results[cond]["heads"][3]
        dest = results[cond]["content_src_dest_of_orig_corners"]
        lines.append(
            f"  {cond:16s} geom_r={r['geom_ratio']:.2f} cont_r={r['content_ratio']:.2f} "
            f"top4_yx={r['top4_yx']} dest_of_orig_corners={dest}"
        )

    text = "\n".join(lines) + "\n"
    (OUT / "verdict.txt").write_text(text)
    print(text, flush=True)

    # JSON (no huge arrays)
    slim = {c: {"dest": results[c]["content_src_dest_of_orig_corners"], "heads": results[c]["heads"]} for c in conditions}
    (OUT / "results.json").write_text(json.dumps(slim, indent=2))

    # plot h3 time-mean 16x16 under each condition
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.2 * len(conditions), 3.4))
    for ax, cond in zip(axes, conditions):
        m = spat_maps[cond].reshape(GRID, GRID)
        vmax = np.percentile(m, 99)
        im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        for y, x in CORNER_YX:
            ax.plot(x, y, "c+", ms=10, mew=1.5)
        ax.set_title(cond, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{vid} L23 h3 time-mean mass — spatial vs content", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "h3_spat_maps_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close()

    # bar chart geom vs content ratio for sink heads mean
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(conditions))
    geom = []
    cont = []
    for cond in conditions:
        hs = [r for r in results[cond]["heads"] if r["head"] in sink_heads]
        geom.append(np.mean([r["geom_ratio"] for r in hs]))
        cont.append(np.mean([r["content_ratio"] for r in hs]))
    w = 0.35
    ax.bar(x - w / 2, geom, w, label="geometric corners (token id / grid pos)", color="#c44e52")
    ax.bar(x + w / 2, cont, w, label="where original corner CONTENT moved", color="#4c72b0")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15)
    ax.set_ylabel("mass ratio vs other patches")
    ax.set_title(f"Sink-like heads {sink_heads}: position vs content tracking")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "geom_vs_content_ratios.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 16x256 for h3 original vs roll vs permute
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    for ax, cond in zip(axes, ["original", "roll_half", "permute_patches"]):
        Mh = np.load(OUT / f"M_h3_{cond}_Tx256.npy")
        vmax = np.percentile(Mh, 99)
        im = ax.imshow(Mh, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        for idx in CORNER_FLAT:
            ax.axvline(idx, color="cyan", lw=0.6, alpha=0.8)
        dest = results[cond]["content_src_dest_of_orig_corners"]
        for name, j in dest.items():
            if j not in CORNER_FLAT:
                ax.axvline(j, color="lime", lw=0.6, alpha=0.7, ls="--")
        ax.set_ylabel(f"{cond}\nt")
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    axes[0].set_title(
        "h3 mass [T×256]: cyan=geom corners; lime dashed=where orig corner CONTENT went"
    )
    axes[-1].set_xlabel("spatial 0..255")
    fig.tight_layout()
    fig.savefig(OUT / "h3_16x256_orig_roll_permute.png", dpi=140, bbox_inches="tight")
    plt.close()

    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
