#!/usr/bin/env python3
# =============================================================================
# [ATTN-CORNER-SINK]  ADD-ON MODULE -- easy to merge or delete.
# -----------------------------------------------------------------------------
# Status  : experimental (worktree experimental/attn-corner-sink, 2026-08-24).
#           Touches NO existing file. `git rm` this one file to remove.
# Purpose : The pruning method's "attention score" = received attention at the
#           encoder's LAST block, column-summed over queries AND heads
#           (train_stream_mtp.TokenPruner). Broken down PER HEAD on hdepic (the
#           "ll" experiments) some heads concentrate their received attention on
#           the four FRAME CORNERS -- not the usual attention-sink shape. This
#           module reproduces that per-head signal on EGTEA and compares models:
#             #1 reproduce on our finetuned EGTEA model (base V-JEPA2 + ll enc-LoRA)
#             #2 3D (t,h,w) view per head -- per-slot small multiples, NOT T-avg
#             #3 base V-JEPA2 vs base V-JEPA 2.1 (register-token pretraining)
#           Existing tools sum over heads and average over time, washing this out.
# Score    : identical to TokenPruner -- imp[h,j] = sum_query softmax(q_i.k_j/vd),
#            just WITHOUT the final sum over heads. Two chunked captures mirror
#            each encoder's own q/k (2.0 src RoPEAttention; 2.1 register-aware).
# =============================================================================
"""Per-head encoder last-block received-attention corner analysis (EGTEA)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ── sys.path: code from CODE_ROOT (worktree) + vjepa2; data/ckpts from SHARED ──
SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/scratch/yh6416/VJEPA2-EXP")
CODE_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (os.path.join(CODE_ROOT, "vjepa2"), CODE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.hdepic_lora_action_anticipation import train_stream_mtp as T  # noqa: E402
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (  # noqa: E402
    FpsSubsampledStreamMTPDataset,
    enlarge_predictor_budget,
)


# ── per-head capture: V-JEPA 2.0 encoder (src RoPEAttention) ──────────────────
class HeadAttnCapture20:
    """Patch a 2.0 ``RoPEAttention`` to record per-head received attention.

    Byte-for-byte mirrors ``TokenPruner._patched_forward`` q/k (RoPE) but keeps
    the head axis: ``imp[b,h,j] = sum_i softmax(q_i.k_j/vd)`` -> (B, heads, N).
    Calls the ORIGINAL forward for the real block output (no output change).
    """

    def __init__(self, attn_module, chunk_size: int = 256):
        from src.models.utils.modules import rotate_queries_or_keys

        m = attn_module
        self._m = m
        self._orig = m.forward
        self.importance: torch.Tensor | None = None  # (B, heads, N)
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
                    if T is None or H_patches is None or W_patches is None:
                        mp = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=x.device)
                    else:
                        mp = torch.arange(int(T * H_patches * W_patches), device=x.device)
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
                imp = torch.zeros(B, m.num_heads, N, device=x.device, dtype=torch.float32)
                for ci in range(0, N, chunk_size):
                    qc = q[:, :, ci:ci + chunk_size, :]
                    logits = (qc @ k.transpose(-2, -1)) * m.scale
                    imp += logits.softmax(dim=-1).sum(dim=2).float()   # sum over queries, KEEP heads
                cap.importance = imp
            return out

        m.forward = _fwd

    def remove(self):
        self._m.forward = self._orig


# ── per-head capture: V-JEPA 2.1 encoder (register-aware RoPEAttention) ────────
class HeadAttnCapture21:
    """Patch a 2.1 ``RoPEAttention`` to record per-head received attention.

    Mirrors ``vjepa_2_1.models.utils.modules.RoPEAttention.forward`` q/k
    (interpolate_rope + register/cls-aware rotate) and column-sums per head.
    ``n_registers`` / ``has_cls_first`` (the register-token mechanism itself) are
    read off the module; context tokens are ``[n_cls : N - n_registers]``.
    """

    def __init__(self, attn_module, chunk_size: int = 256):
        _enable_vjepa_2_1_imports()
        from app.vjepa_2_1.models.utils.modules import rotate_queries_or_keys

        m = attn_module
        self._m = m
        self._orig = m.forward
        self.importance: torch.Tensor | None = None  # (B, heads, N) over ALL keys
        self.n_registers = int(getattr(m, "n_registers", 0))
        self.n_cls = 1 if getattr(m, "has_cls_first", False) else 0
        cap = self

        def _fwd(x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False):
            out = cap._orig(x, mask=mask, T=T, H_patches=H_patches, W_patches=W_patches, return_attn=return_attn)
            with torch.no_grad():
                B, N, C = x.size()
                N_ctx = N - m.n_registers
                grid_depth = int(N_ctx // (m.grid_size * m.grid_size))
                qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv[0], qkv[1], qkv[2]
                if mask is not None:
                    mp = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
                    d_mask, h_mask, w_mask = m.separate_positions(mp, H_patches, W_patches)
                else:
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
                    q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
                    k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
                else:
                    q = torch.cat([qd, qh, qw], dim=-1)
                    k = torch.cat([kd, kh, kw], dim=-1)
                imp = torch.zeros(B, m.num_heads, N, device=x.device, dtype=torch.float32)
                for ci in range(0, N, chunk_size):
                    qc = q[:, :, ci:ci + chunk_size, :]
                    logits = (qc @ k.transpose(-2, -1)) * m.scale
                    imp += logits.softmax(dim=-1).sum(dim=2).float()
                cap.importance = imp
            return out

        m.forward = _fwd

    def remove(self):
        self._m.forward = self._orig


# ── generic video decode (non-egocentric control) ────────────────────────────
def _decode_video_clip(path, img_size, n_frames):
    """Decode ``n_frames`` evenly-spanning frames from an arbitrary MP4.

    Returns a (C, T, H, W) uint8 tensor matching the EGTEA dataset layout, so the
    same normalization + encoder path applies. Even spacing over the whole clip
    approximates the ego pipeline's ~8 fps for 10 s / ~80-frame clips.
    """
    from decord import VideoReader, cpu

    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1, width=img_size, height=img_size)
    L = len(vr)
    idx = np.linspace(0, max(0, L - 1), n_frames).round().astype(int)
    frames = vr.get_batch(idx.tolist()).asnumpy()          # (T, H, W, C)
    del vr
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).contiguous()


# ── model builders ────────────────────────────────────────────────────────────
def build_finetuned_20(device, *, max_frames, fps, img_size, checkpoint, enc_lora, pred_lora, parent_ckpt):
    """Load the finetuned EGTEA model EXACTLY as the pruning eval does, so the
    captured score is the score the pruning method uses (base + ll enc-LoRA)."""
    base = T.build_model(device, max_frames, fps, img_size, str(checkpoint))
    for p in base.encoder.parameters():
        p.requires_grad = False
    T.load_lora_sidecars(base, str(enc_lora) if enc_lora else None, str(pred_lora) if pred_lora else None)
    gp = int(base.grid_size ** 2)
    enlarge_predictor_budget(base, (int(max_frames) // int(base.tubelet_size)) * gp, gp)
    model = T.PrunedAnticipativeModel(base, None, prune_threshold=10 ** 9).to(device)
    if parent_ckpt and Path(parent_ckpt).is_file():
        ck = torch.load(parent_ckpt, map_location="cpu", weights_only=False)
        miss, unexp = model.load_state_dict(ck["model"], strict=False)
        print(f"[finetuned] parent load: {len(miss)} missing, {len(unexp)} unexpected", flush=True)
        del ck
    base.eval()
    return base, "20"


def build_base_20(device, *, max_frames, fps, img_size, checkpoint):
    base = T.build_model(device, max_frames, fps, img_size, str(checkpoint))
    for p in base.parameters():
        p.requires_grad = False
    base.eval()
    return base, "20"


def _enable_vjepa_2_1_imports():
    """Make ``app.vjepa_2_1`` (under vjepa2/app) importable.

    The main-tree ``app`` is a regular package that shadows the namespace
    ``vjepa2/app``; extend its search path so ``app.vjepa_2_1`` resolves without
    editing the checked-in ``app/__init__.py``.
    """
    import app as _app_pkg
    vj_app = os.path.join(CODE_ROOT, "vjepa2", "app")
    if vj_app not in list(_app_pkg.__path__):
        _app_pkg.__path__.append(vj_app)


def build_base_21(device, *, max_frames, fps, img_size, checkpoint):
    """Base V-JEPA 2.1 via the same concat_ar path the 2.1 evals use."""
    _enable_vjepa_2_1_imports()
    model_kwargs = {
        "use_v2_1": True,
        "encoder": {
            "model_name": "vit_large", "checkpoint_key": "ema_encoder",
            "img_temporal_dim_size": 1, "tubelet_size": 2, "patch_size": 16,
            "uniform_power": True, "use_rope": True,
        },
        "predictor": {
            "model_name": "vit_predictor", "checkpoint_key": "predictor",
            "num_frames": 64, "depth": 12, "num_heads": 12, "predictor_embed_dim": 384,
            "teacher_embed_dim": 1664, "num_mask_tokens": 8, "n_output_distillation": 1,
            "return_all_tokens": True, "img_temporal_dim_size": 1, "uniform_power": True,
            "use_mask_tokens": True, "use_sdpa": True, "use_silu": False, "wide_silu": False,
            "use_rope": True,
        },
    }
    wrapper_kwargs = {"no_predictor": False, "num_output_frames": 2, "num_steps": 1}
    model = T.init_anticipative_module(
        frames_per_clip=max_frames, frames_per_second=fps, resolution=img_size,
        checkpoint=str(checkpoint), model_kwargs=model_kwargs, wrapper_kwargs=wrapper_kwargs,
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, "21"


# ── corner / concentration metrics (grid-aware) ───────────────────────────────
def corner_metrics(spatial: np.ndarray) -> dict:
    """spatial: (grid, grid), normalized so sum == 1. Returns corner concentration."""
    g = spatial.shape[0]
    total = float(spatial.sum()) + 1e-12
    uniform = 1.0 / (g * g)
    corners1 = spatial[0, 0] + spatial[0, -1] + spatial[-1, 0] + spatial[-1, -1]
    b = max(1, g // 8)  # 2x2 block for g=16, 3x3 for g=24
    cb = (spatial[:b, :b].sum() + spatial[:b, -b:].sum()
          + spatial[-b:, :b].sum() + spatial[-b:, -b:].sum())
    inner = spatial[b:-b, b:-b].mean() if g > 2 * b else 0.0
    edge_ring = (spatial.sum() - spatial[1:-1, 1:-1].sum()) / (g * g - (g - 2) * (g - 2))
    center = spatial[g // 2 - g // 4: g // 2 + g // 4, g // 2 - g // 4: g // 2 + g // 4].mean()
    return {
        "corner4_over_uniform": float(corners1 / (4 * uniform) / total),
        "cornerblk_frac": float(cb / total),
        "cornerblk_over_uniform": float((cb / (4 * b * b)) / uniform / total),
        "center_over_edge": float(center / (edge_ring + 1e-12)),
        "peak_over_uniform": float(spatial.max() / uniform / total),
    }


# ── main analysis loop ────────────────────────────────────────────────────────
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  variant={args.variant}  res={args.img_size}  frames={args.max_frames}", flush=True)

    if args.variant == "finetuned20":
        base, kind = build_finetuned_20(
            device, max_frames=args.max_frames, fps=args.fps, img_size=args.img_size,
            checkpoint=args.checkpoint, enc_lora=args.encoder_lora, pred_lora=args.predictor_lora,
            parent_ckpt=args.init_from_ckpt)
    elif args.variant == "base20":
        base, kind = build_base_20(device, max_frames=args.max_frames, fps=args.fps,
                                   img_size=args.img_size, checkpoint=args.checkpoint)
    elif args.variant == "base21":
        base, kind = build_base_21(device, max_frames=args.max_frames, fps=args.fps,
                                   img_size=args.img_size, checkpoint=args.checkpoint)
    else:
        raise SystemExit(f"unknown variant {args.variant}")

    encoder = base.encoder
    block_idx = args.block if args.block >= 0 else (len(encoder.blocks) + args.block)
    attn = encoder.blocks[block_idx].attn
    cap = (HeadAttnCapture21 if kind == "21" else HeadAttnCapture20)(attn, chunk_size=args.chunk_size)
    n_heads = int(attn.num_heads)
    grid = int(attn.grid_size)
    tub = int(base.tubelet_size)
    expected_slots = int(args.max_frames // tub)
    n_reg = int(getattr(attn, "n_registers", 0))
    n_cls = 1 if getattr(attn, "has_cls_first", False) else 0
    print(f"[model] block={block_idx} heads={n_heads} grid={grid} slots={expected_slots} "
          f"registers={n_reg} cls={n_cls}", flush=True)

    # data source: either EGTEA stream-MTP CSV (egocentric) or a directory of
    # arbitrary MP4s (e.g. Kinetics third-person) via --video-dir.
    use_video_dir = args.video_dir is not None
    mean = T.IMAGENET_MEAN.to(device)
    std = T.IMAGENET_STD.to(device)
    if use_video_dir:
        import glob
        files = sorted(glob.glob(os.path.join(str(args.video_dir), "*.mp4"))
                       + glob.glob(os.path.join(str(args.video_dir), "*.MP4")))
        picked = list(range(len(files)))[: args.n_samples] if args.n_samples > 0 else list(range(len(files)))
        print(f"[data] {len(picked)} videos from {args.video_dir}", flush=True)

        def get_clip(i):
            clip = _decode_video_clip(files[i], args.img_size, args.max_frames)  # (C,T,H,W) uint8
            c = clip.unsqueeze(0).to(device).float().div_(255.0)
            return c.sub_(mean).div_(std)
        val_ds = None
    else:
        val_ds = FpsSubsampledStreamMTPDataset(
            args.val_csv, args.video_root, args.img_size, src_fps=args.src_fps, fps=args.fps)
        import hashlib
        def _rk(r):
            return hashlib.md5(f"{args.val_subset_seed}|{r['video_id']}|{r.get('tick_frame', r.get('frame_indices',''))}".encode()).hexdigest()
        rows = list(range(len(val_ds.rows)))
        if args.only_context_sec > 0:
            rows = [i for i in rows if abs(float(val_ds.rows[i]["context_sec"]) - args.only_context_sec) < 1e-6]
        order = sorted(rows, key=lambda i: _rk(val_ds.rows[i]))
        picked = order[: args.n_samples]
        print(f"[data] {len(picked)} samples (context={args.only_context_sec}s, of {len(rows)} matched)", flush=True)

        def get_clip(i):
            batch = T.collate_stream([val_ds[i]])
            c = batch["clip"].to(device).float().div_(255.0)
            return c.sub_(mean).div_(std)

    # correctness: head-summed capture must equal the production TokenPruner score (2.0 only)
    if args.verify and kind == "20" and not use_video_dir:
        s0 = T.collate_stream([val_ds[picked[0]]])
        c0 = s0["clip"].to(device).float().div_(255.0).sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            encoder(c0)
        mine = cap.importance[0].sum(dim=0).float().cpu()          # sum over heads -> (N,)
        cap.remove()
        pruner = T.TokenPruner(encoder, keep_count=(mine.shape[0] // 2), gp=grid * grid)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            encoder(c0)
        ref = pruner._importance[0].float().cpu()
        pruner.remove()
        rel = (mine - ref).abs().max() / (ref.abs().max() + 1e-9)
        print(f"[verify] head-summed vs TokenPruner: max|Δ|={float((mine-ref).abs().max()):.4g} "
              f"rel={float(rel):.4g}  {'PASS' if rel < 1e-3 else 'FAIL'}", flush=True)
        cap = HeadAttnCapture20(encoder.blocks[block_idx].attn, chunk_size=args.chunk_size)

    n_ctx = expected_slots * grid * grid
    acc = np.zeros((n_heads, expected_slots, grid, grid), dtype=np.float64)  # sum of per-sample normalized maps
    reg_mass = np.zeros(n_heads, dtype=np.float64)   # frac of received attn on register/cls cols
    per_sample_spatial = []   # for cross-video stability (ll-style): [(heads, grid, grid) per clip]
    n_used = 0
    for idx in picked:
        try:
            clips = get_clip(idx)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] decode failed idx={idx}: {e}", flush=True)
            continue
        if args.hflip:
            clips = torch.flip(clips, dims=[-1])   # mirror width (position vs content test)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            encoder(clips)
        imp = cap.importance  # (B, heads, N)
        if imp is None:
            continue
        imp = imp[0].float().cpu().numpy()  # (heads, N)
        N = imp.shape[1]
        ctx = imp[:, n_cls:N - n_reg] if (n_cls or n_reg) else imp
        if ctx.shape[1] != n_ctx:
            continue  # non-target length; skip to keep the (slots,grid,grid) tensor uniform
        per_head_total = imp.sum(axis=1) + 1e-12                       # over ALL keys incl reg/cls
        if n_cls or n_reg:
            reg_mass += (imp[:, :n_cls].sum(axis=1) + imp[:, N - n_reg:].sum(axis=1)) / per_head_total
        maps = ctx.reshape(n_heads, expected_slots, grid, grid)
        maps = maps / per_head_total[:, None, None, None]              # normalize each head to sum≈1
        acc += maps
        if args.dump_per_sample:
            per_sample_spatial.append(maps.sum(axis=1).astype(np.float32))  # (heads, grid, grid) this clip
        n_used += 1
        if n_used % 25 == 0:
            print(f"  {n_used}/{len(picked)}", flush=True)

    if n_used == 0:
        raise SystemExit("no usable samples (context/length filter too strict?)")
    per_head_thw = (acc / n_used)                     # (heads, slots, grid, grid), sums to ~1/head
    reg_mass = reg_mass / n_used

    # per-head spatial (time-avg) + metrics
    spatial = per_head_thw.sum(axis=1)                # (heads, grid, grid)
    spatial = spatial / (spatial.sum(axis=(1, 2), keepdims=True) + 1e-12)
    metrics = [corner_metrics(spatial[h]) for h in range(n_heads)]
    order_by_corner = sorted(range(n_heads), key=lambda h: metrics[h]["cornerblk_over_uniform"], reverse=True)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "per_head_thw.npy", per_head_thw.astype(np.float32))
    if args.dump_per_sample and per_sample_spatial:
        np.save(outdir / "per_sample_spatial.npy", np.stack(per_sample_spatial))  # (n, heads, grid, grid)
    summary = {
        "variant": args.variant, "img_size": args.img_size, "grid": grid, "slots": expected_slots,
        "block": block_idx, "n_heads": n_heads, "n_used": n_used,
        "n_registers": n_reg, "n_cls": n_cls,
        "register_cls_mass_per_head": [round(float(x), 4) for x in reg_mass],
        "per_head_corner": {str(h): {k: round(v, 4) for k, v in metrics[h].items()} for h in range(n_heads)},
        "heads_ranked_by_cornerblk": order_by_corner,
        "top_corner_heads": order_by_corner[:6],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("variant", "grid", "slots", "n_used",
          "heads_ranked_by_cornerblk", "register_cls_mass_per_head")}, indent=2), flush=True)

    _make_figures(outdir, per_head_thw, spatial, metrics, order_by_corner, args)
    return summary


def _make_figures(outdir, per_head_thw, spatial, metrics, order_by_corner, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plots] matplotlib unavailable ({e}); arrays saved", flush=True)
        return
    n_heads, slots, grid, _ = per_head_thw.shape

    # Fig #1 -- all heads, time-averaged spatial map (reproduce the phenomenon)
    ncol = 4
    nrow = int(np.ceil(n_heads / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
    for h in range(nrow * ncol):
        ax = axes.flat[h]
        if h < n_heads:
            im = ax.imshow(spatial[h], cmap="magma")
            cb = metrics[h]["cornerblk_over_uniform"]
            ax.set_title(f"head {h}  corner×{cb:.1f}", fontsize=8,
                         color=("red" if cb > 2.0 else "black"))
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{args.variant} @{args.img_size}px  per-head received attn (time-avg, block {args.block})")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_per_head_spatial.png", dpi=120)
    plt.close(fig)

    # Fig #2 -- top corner heads: per-slot small multiples (3D / non-time-averaged)
    top = order_by_corner[:min(4, n_heads)]
    show_slots = np.linspace(0, slots - 1, min(slots, 10)).round().astype(int)
    fig, axes = plt.subplots(len(top), len(show_slots),
                             figsize=(1.6 * len(show_slots), 1.8 * len(top)), squeeze=False)
    for r, h in enumerate(top):
        vmax = per_head_thw[h].max()
        for c, s in enumerate(show_slots):
            ax = axes[r][c]
            ax.imshow(per_head_thw[h, s], cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={s}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"head {h}", fontsize=9)
    fig.suptitle(f"{args.variant} @{args.img_size}px  top corner heads across time-slots (block {args.block})")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_corner_heads_per_slot.png", dpi=120)
    plt.close(fig)
    print(f"[plots] saved fig1/fig2 to {outdir}", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description="[ATTN-CORNER-SINK] per-head encoder corner attention analysis")
    ap.add_argument("--variant", required=True, choices=["finetuned20", "base20", "base21"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    ap.add_argument("--init-from-ckpt", type=Path, default=None)
    ap.add_argument("--val-csv", type=Path, default=None)
    ap.add_argument("--video-root", type=Path, default=None)
    ap.add_argument("--video-dir", type=Path, default=None,
                    help="dir of arbitrary MP4s (non-ego control); overrides EGTEA CSV path")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--block", type=int, default=-1, help="encoder block to read (-1 = last)")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--only-context-sec", type=float, default=10.0)
    ap.add_argument("--val-subset-seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--verify", action="store_true", help="check head-summed capture == TokenPruner score (2.0)")
    ap.add_argument("--hflip", action="store_true", help="horizontally mirror input (position-vs-content test)")
    ap.add_argument("--dump-per-sample", action="store_true", help="save per-clip spatial maps for cross-video stability (ll-style)")
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
