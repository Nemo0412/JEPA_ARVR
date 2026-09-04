#!/usr/bin/env python3
"""Decompose L23 attention chain: where do corner sinks emerge?

Stages (per token j, focus heads h3 sink vs h8 content):
  1) Input hidden      ||h_j||_2 / ||h_j||_inf
  2) After Norm1       ||hat{h}_j||_2 / ||hat{h}_j||_inf
  3) Q/K projection    ||q_j||_2 / ||k_j||_2  (pre-RoPE)
  4) Before RoPE       mean_i (q_i^T k_j / sqrt(d))
  5) After RoPE        mean_i ((Rq_i)^T (Rk_j) / sqrt(d))
  6) Softmax mass      I_j = sum_i A_ij   (post-RoPE attention)
  Also: mean_i A_ij for corners vs others; corner/other ratio per stage.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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

OUT = Path("/home/ll5914/Jepa_yifan/encoder_L23_hdepic_sink_example")
GRID = 16
HEADS = (3, 8)  # sink-heavy vs content
CHUNK = 128


def corner_mask(n: int, grid: int = GRID) -> torch.Tensor:
    gp = grid * grid
    t = n // gp
    m = torch.zeros(n, dtype=torch.bool)
    for ti in range(t):
        base = ti * gp
        for r, c in ((0, 0), (0, grid - 1), (grid - 1, 0), (grid - 1, grid - 1)):
            m[base + r * grid + c] = True
    return m


def center_mask(n: int, grid: int = GRID) -> torch.Tensor:
    gp = grid * grid
    t = n // gp
    m = torch.zeros(n, dtype=torch.bool)
    lo, hi = grid // 4, 3 * grid // 4
    for ti in range(t):
        base = ti * gp
        for r in range(lo, hi):
            for c in range(lo, hi):
                m[base + r * grid + c] = True
    return m


def spatial_mean(vec: np.ndarray, grid: int = GRID) -> np.ndarray:
    """[N] -> [grid,grid] mean over temporal slots."""
    gp = grid * grid
    t = vec.size // gp
    return vec.reshape(t, grid, grid).mean(axis=0)


def group_stats(vec: torch.Tensor, cmask: torch.Tensor, omask: torch.Tensor) -> dict:
    v = vec.float()
    return {
        "corner_mean": float(v[cmask].mean()),
        "corner_max": float(v[cmask].max()),
        "other_mean": float(v[omask].mean()),
        "other_max": float(v[omask].max()),
        "center_mean": float(v[omask].mean()),  # placeholder overwritten by caller if needed
        "ratio_corner_over_other": float(v[cmask].mean() / (v[omask].mean() + 1e-12)),
    }


def mean_key_logits(q: torch.Tensor, k: torch.Tensor, scale: float, chunk: int = CHUNK) -> torch.Tensor:
    """q,k: [H,N,D] -> mean_i (q_i·k_j)*scale  shape [H,N]."""
    h, n, _d = q.shape
    out = torch.zeros(h, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)  # [H,D,N]
    for ci in range(0, n, chunk):
        q_c = q[:, ci : ci + chunk, :].float()  # [H,c,D]
        logits = torch.bmm(q_c, k_t) * scale  # [H,c,N]
        out += logits.sum(dim=1)
    out /= float(n)
    return out


def attention_mass(q: torch.Tensor, k: torch.Tensor, scale: float, chunk: int = CHUNK) -> torch.Tensor:
    """I_j = sum_i softmax_j(q_i·k)_ij   shape [H,N]."""
    h, n, _d = q.shape
    out = torch.zeros(h, n, device=q.device, dtype=torch.float32)
    k_t = k.float().transpose(-2, -1)
    for ci in range(0, n, chunk):
        q_c = q[:, ci : ci + chunk, :].float()
        logits = torch.bmm(q_c, k_t) * scale
        out += logits.softmax(dim=-1).sum(dim=1)
    return out


def apply_rope(module, q, k, T=None, H_patches=None, W_patches=None):
    m = module
    B, H, N, D = q.shape
    grid_depth = int(N // (m.grid_size * m.grid_size))
    if T is None or H_patches is None or W_patches is None:
        mask_p = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=q.device)
    else:
        mask_p = torch.arange(int(T * H_patches * W_patches), device=q.device)
    d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)
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
        q2 = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
        k2 = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
    else:
        q2 = torch.cat([qd, qh, qw], dim=-1)
        k2 = torch.cat([kd, kh, kw], dim=-1)
    return q2, k2


def diagnose_block(block, x: torch.Tensor, T=None, H_patches=None, W_patches=None):
    """x: residual into block [1,N,C]. Returns dict of stage tensors [H,N] or [N]."""
    attn = block.attn
    h_in = x[0]  # [N,C]
    h_hat = block.norm1(x)[0]

    qkv = attn.qkv(block.norm1(x)).unflatten(-1, (3, attn.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q0, k0, v0 = qkv[0], qkv[1], qkv[2]  # [1,H,N,D]
    q0, k0 = q0[0], k0[0]  # [H,N,D]

    q_rope, k_rope = apply_rope(attn, q0.unsqueeze(0), k0.unsqueeze(0), T, H_patches, W_patches)
    q_rope, k_rope = q_rope[0], k_rope[0]

    scale = float(attn.scale)
    logits_pre = mean_key_logits(q0, k0, scale)
    logits_post = mean_key_logits(q_rope, k_rope, scale)
    mass = attention_mass(q_rope, k_rope, scale)

    # Also pre-RoPE attention mass (counterfactual: softmax without RoPE)
    mass_pre = attention_mass(q0, k0, scale)

    return {
        "h_l2": h_in.float().norm(dim=-1),
        "h_inf": h_in.float().abs().amax(dim=-1),
        "hn_l2": h_hat.float().norm(dim=-1),
        "hn_inf": h_hat.float().abs().amax(dim=-1),
        "q_l2": q0.float().norm(dim=-1),  # [H,N]
        "k_l2": k0.float().norm(dim=-1),
        "logits_pre_rope": logits_pre,
        "logits_post_rope": logits_post,
        "mass_pre_rope": mass_pre,
        "mass_post_rope": mass,
        "delta_logits": logits_post - logits_pre,
        "delta_mass": mass - mass_pre,
    }


def summarize_head(diag, head: int, cmask, omask, cenmask) -> dict:
    stages = {}

    def one(name, vec_n, is_head=False):
        v = vec_n[head] if is_head else vec_n
        st = {
            "corner_mean": float(v[cmask].mean()),
            "other_mean": float(v[omask].mean()),
            "center_mean": float(v[cenmask].mean()),
            "ratio": float(v[cmask].mean() / (v[omask].mean() + 1e-12)),
        }
        stages[name] = st
        return st

    one("1_input_h_l2", diag["h_l2"])
    one("1b_input_h_inf", diag["h_inf"])
    one("2_after_norm_l2", diag["hn_l2"])
    one("2b_after_norm_inf", diag["hn_inf"])
    one("3_q_l2", diag["q_l2"], is_head=True)
    one("3b_k_l2", diag["k_l2"], is_head=True)
    one("4_logits_pre_rope", diag["logits_pre_rope"], is_head=True)
    one("5_logits_post_rope", diag["logits_post_rope"], is_head=True)
    one("5b_delta_logits_rope", diag["delta_logits"], is_head=True)
    one("6_mass_pre_rope", diag["mass_pre_rope"], is_head=True)
    one("7_mass_post_rope", diag["mass_post_rope"], is_head=True)
    one("7b_delta_mass_rope", diag["delta_mass"], is_head=True)
    return stages


def plot_ratio_bars(stages_h3, stages_h8, out_png: Path):
    # Key stages for "when does corner become special"
    keys = [
        ("1_input_h_l2", "||h||₂"),
        ("2_after_norm_l2", "||Norm(h)||₂"),
        ("3b_k_l2", "||k||₂"),
        ("4_logits_pre_rope", "mean logit\npre-RoPE"),
        ("5_logits_post_rope", "mean logit\npost-RoPE"),
        ("6_mass_pre_rope", "attn mass\npre-RoPE"),
        ("7_mass_post_rope", "attn mass\npost-RoPE"),
    ]
    x = np.arange(len(keys))
    r3 = [stages_h3[k]["ratio"] for k, _ in keys]
    r8 = [stages_h8[k]["ratio"] for k, _ in keys]
    labels = [lab for _, lab in keys]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.38
    ax.bar(x - w / 2, r3, w, label="h3 (sink-heavy)", color="#2E7D32")
    ax.bar(x + w / 2, r8, w, label="h8 (content)", color="#E65100")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="ratio=1 (no corner bias)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("corner_mean / other_mean")
    ax.set_title("Where do corners become abnormal? (HD-EPIC L23, one clip)")
    ax.legend()
    # annotate post-softmax jump
    for i, (a, b) in enumerate(zip(r3, r8)):
        ax.text(i - w / 2, a + 0.05, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(i + w / 2, b + 0.05, f"{b:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_stages(diag, head: int, out_png: Path, title: str):
    maps = [
        ("||h||₂", spatial_mean(diag["h_l2"].cpu().numpy())),
        ("||Norm(h)||₂", spatial_mean(diag["hn_l2"].cpu().numpy())),
        ("||k||₂", spatial_mean(diag["k_l2"][head].cpu().numpy())),
        ("logit pre-RoPE", spatial_mean(diag["logits_pre_rope"][head].cpu().numpy())),
        ("logit post-RoPE", spatial_mean(diag["logits_post_rope"][head].cpu().numpy())),
        ("Δlogit (RoPE)", spatial_mean(diag["delta_logits"][head].cpu().numpy())),
        ("mass pre-RoPE", spatial_mean(diag["mass_pre_rope"][head].cpu().numpy())),
        ("mass post-RoPE", spatial_mean(diag["mass_post_rope"][head].cpu().numpy())),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (name, sp) in zip(axes.ravel(), maps):
        # center each map for signed delta; else min-max
        if "Δ" in name:
            lim = float(np.max(np.abs(sp)) + 1e-12)
            im = ax.imshow(sp, cmap="coolwarm", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(sp, cmap="viridis")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for r, c in ((0, 0), (0, 15), (15, 0), (15, 15)):
            ax.plot(c, r, "r.", markersize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    class A:
        pass

    args = A()
    args.n_sample = 1
    args.chunk = 128
    args.frames = 32
    args.fps = 8
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

    device = torch.device("cuda")
    samples = mc.load_diverse_clips(args)
    clip, meta = samples[0]
    vid = meta["video_id"]
    print("video", vid)

    model = mc._fastgen.load_video_model(args, device)
    core = model.base_model if hasattr(model, "base_model") else model
    encoder = core.encoder
    last = len(encoder.blocks) - 1
    block = encoder.blocks[last]

    captured = {}
    orig_fwd = block.forward

    def wrapped_fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        with torch.no_grad():
            captured["diag"] = diagnose_block(
                block, x, T=T, H_patches=H_patches, W_patches=W_patches
            )
            captured["meta"] = {
                "T": T,
                "H": H_patches,
                "W": W_patches,
                "N": int(x.shape[1]),
            }
        return orig_fwd(x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)

    block.forward = wrapped_fwd
    clip_b = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
    ant = torch.full((1,), float(args.anticipation), device=device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _ = model(clip_b, ant)
    block.forward = orig_fwd

    diag = captured["diag"]
    n = captured["meta"]["N"]
    cmask = corner_mask(n).to(device)
    omask = ~cmask
    cenmask = center_mask(n).to(device)

    report = {"video_id": vid, "layer": last, "N": n, "heads": {}}
    for head in HEADS:
        stages = summarize_head(diag, head, cmask, omask, cenmask)
        # fix center_mean properly already in summarize
        report["heads"][f"h{head}"] = stages
        print(f"\n===== head {head} =====")
        for k, st in stages.items():
            print(
                f"  {k:28s}  corner={st['corner_mean']:.5f}  other={st['other_mean']:.5f}  "
                f"center={st['center_mean']:.5f}  ratio={st['ratio']:.3f}"
            )

    (OUT / "sink_chain_diagnosis.json").write_text(json.dumps(report, indent=2))

    # Verdict table
    lines = []
    lines.append(f"video={vid}  L{last}  N={n}")
    lines.append("")
    lines.append("corner/other ratio by stage (h3 sink vs h8 content):")
    lines.append(f"{'stage':32s} {'h3':>8s} {'h8':>8s}  note")
    keys_order = list(report["heads"]["h3"].keys())
    for k in keys_order:
        r3 = report["heads"]["h3"][k]["ratio"]
        r8 = report["heads"]["h8"][k]["ratio"]
        note = ""
        if "logits_pre" in k and r3 < 1.2:
            note = "pre-RoPE: corners NOT special yet" if r3 < 1.15 else ""
        if "logits_post" in k:
            note = "AFTER RoPE logits"
        if k == "6_mass_pre_rope":
            note = "softmax w/o RoPE"
        if k == "7_mass_post_rope":
            note = "FINAL sink mass"
        if "delta_logits" in k:
            note = "RoPE-induced Δlogit"
        if "delta_mass" in k:
            note = "RoPE-induced Δmass"
        lines.append(f"{k:32s} {r3:8.3f} {r8:8.3f}  {note}")

    # Identify jump
    r_pre = report["heads"]["h3"]["4_logits_pre_rope"]["ratio"]
    r_post = report["heads"]["h3"]["5_logits_post_rope"]["ratio"]
    m_pre = report["heads"]["h3"]["6_mass_pre_rope"]["ratio"]
    m_post = report["heads"]["h3"]["7_mass_post_rope"]["ratio"]
    k_r = report["heads"]["h3"]["3b_k_l2"]["ratio"]
    h_r = report["heads"]["h3"]["1_input_h_l2"]["ratio"]
    lines.append("")
    lines.append("VERDICT for h3 (sink head):")
    lines.append(f"  ||h|| corner/other     = {h_r:.3f}")
    lines.append(f"  ||k|| corner/other     = {k_r:.3f}")
    lines.append(f"  logit ratio pre-RoPE   = {r_pre:.3f}")
    lines.append(f"  logit ratio post-RoPE  = {r_post:.3f}   (delta_ratio={r_post-r_pre:+.3f})")
    lines.append(f"  mass ratio pre-RoPE    = {m_pre:.3f}")
    lines.append(f"  mass ratio post-RoPE   = {m_post:.3f}   (delta_ratio={m_post-m_pre:+.3f})")
    if r_post > r_pre * 1.15 or (r_post - r_pre) > 0.2:
        lines.append("  → RoPE INCREASES corner logit bias.")
    else:
        lines.append("  → RoPE does NOT strongly create corner logit bias.")
    if m_post > m_pre * 1.2:
        lines.append("  → Softmax on post-RoPE logits is the main amplifier of sink mass.")
    if k_r > 1.2 and r_pre > 1.15:
        lines.append("  → Corner keys already large / high-dot before RoPE (projection / hidden).")
    if h_r < 1.1 and k_r > 1.2:
        lines.append("  → Hidden not outlier; W_K (or head subspace) makes corner keys large.")
    elif h_r > 1.2:
        lines.append("  → Corner tokens already representation outliers before attention.")

    text = "\n".join(lines) + "\n"
    (OUT / "sink_chain_verdict.txt").write_text(text)
    print("\n" + text)

    plot_ratio_bars(report["heads"]["h3"], report["heads"]["h8"], OUT / "sink_chain_ratios_bars.png")
    plot_spatial_stages(diag, 3, OUT / "sink_chain_spatial_h3.png", f"L{last} h3 chain — {vid}")
    plot_spatial_stages(diag, 8, OUT / "sink_chain_spatial_h8.png", f"L{last} h8 chain — {vid}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
