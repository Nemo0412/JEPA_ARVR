#!/usr/bin/env python3
"""FastGen-style attention-head profiling for V-JEPA (arXiv:2310.01801).

Ge et al. assign each attention head the cheapest KV-keep policy that recovers
a fraction T of its attention map. LLM policies (special / punct / frequency /
local / full) are mapped onto V-JEPA spatiotemporal tokens:

  recent   last r_recent temporal slots          (sequence-anchor / <s> analog)
  center   inner spatial patches
  gaze     patches overlapping rasterized gaze   (multimodal special)
  imu      keep_aux IMU prefix in the predictor  (special tokens)
  hitter   top-r_f keys by column-sum            (frequency / heavy-hitter)
  local    spatiotemporal neighborhood of each query
  full     retain every key

Profiles two checkpoints on the same HD-EPIC P01 val clips:
  * video-only joint (encoder + predictor)
  * concat+cross-attention v2 (5ch gaze+pose encoder, IMU fusion CA, keep_aux predictor)

Outputs a JSON summary consumed by the analysis canvas.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

from app.hdepic_lora_action_anticipation.binary_input_adapter import BinaryMapInputAdapter  # noqa: E402
from app.hdepic_lora_action_anticipation.concat_plus_cross_attn import (  # noqa: E402
    ConcatPlusCrossAttnAdaptedModel,
)
from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate  # noqa: E402
from app.hdepic_lora_action_anticipation.pose_map_builder import GazePoseInputMapBuilder  # noqa: E402
from app.hdepic_lora_action_anticipation.prefill_clip_frame_cache import (  # noqa: E402
    _indices_for_row,
    _video_path,
)
from app.hdepic_lora_action_anticipation.clip_frame_cache import load_or_decode_clip  # noqa: E402
from app.hdepic_lora_action_anticipation.train_stream_mtp import build_model, load_lora_sidecars  # noqa: E402
from app.hdepic_lora_action_anticipation.tri_modal_fusion import (  # noqa: E402
    ImuTemporalEncoder,
    ImuTrajectoryLoader,
    ProjectedTriModalCrossAttention,
    compute_token_budgets,
    load_tri_modal_fusion_checkpoint,
)
from evals.action_anticipation_frozen.dataloader import VideoTransform  # noqa: E402
from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

logger = logging.getLogger("fastgen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Paper defaults (Sec. 3.4 / 5.1).
T_RECOVER = 0.95
R_RECENT = 0.30
R_FREQ = 0.30
R_LOCAL_T = 0.30
R_LOCAL_S = 0.30
CENTER_FRAC = 0.50


# ── token geometry ───────────────────────────────────────────────────────────
def token_coords(n_tok: int, grid: int, n_aux: int = 0):
    n_vid = n_tok - n_aux
    gp = grid * grid
    if n_vid <= 0 or n_vid % gp != 0:
        t = h = w = torch.zeros(n_tok, dtype=torch.long)
        return t, h, w, n_vid // gp if gp else 0
    t_slots = n_vid // gp
    vid = torch.arange(n_vid)
    t = torch.cat([torch.full((n_aux,), -1, dtype=torch.long), vid // gp])
    within = vid % gp
    h = torch.cat([torch.zeros(n_aux, dtype=torch.long), within // grid])
    w = torch.cat([torch.zeros(n_aux, dtype=torch.long), within % grid])
    return t, h, w, t_slots


def build_key_masks(n_tok: int, grid: int, n_aux: int = 0, gaze_flat: torch.Tensor | None = None):
    """Boolean key masks [N] plus local pairwise mask [N, N]."""
    t, h, w, t_slots = token_coords(n_tok, grid, n_aux)
    gp = grid * grid
    is_aux = torch.zeros(n_tok, dtype=torch.bool)
    if n_aux > 0:
        is_aux[:n_aux] = True
    is_vid = ~is_aux

    recent = torch.zeros(n_tok, dtype=torch.bool)
    if t_slots > 0:
        t_cut = int((1.0 - R_RECENT) * t_slots)
        recent = is_vid & (t >= t_cut)

    half = CENTER_FRAC / 2.0
    lo = int(round((0.5 - half) * grid))
    hi = int(round((0.5 + half) * grid))
    center = is_vid & (h >= lo) & (h < hi) & (w >= lo) & (w < hi)

    gaze = torch.zeros(n_tok, dtype=torch.bool)
    if gaze_flat is not None and int(gaze_flat.numel()) == int(is_vid.sum()):
        gaze[n_aux:] = gaze_flat.bool().cpu()

    t_win = max(1, int(round(R_LOCAL_T * max(t_slots, 1))))
    s_win = max(1, int(round(R_LOCAL_S * grid)))
    # Pairwise local: aux tokens are never "local neighbors" of video queries
    # (they are the special-token bucket). Video-video Chebyshev+temporal.
    tt = t.view(-1, 1)
    hh = h.view(-1, 1)
    ww = w.view(-1, 1)
    dt = (tt - tt.T).abs()
    dh = (hh - hh.T).abs()
    dw = (ww - ww.T).abs()
    both_vid = is_vid.view(-1, 1) & is_vid.view(1, -1)
    local = both_vid & (dt <= t_win) & (dh <= s_win) & (dw <= s_win)
    # Always keep self.
    local.fill_diagonal_(True)

    return {
        "recent": recent,
        "center": center,
        "gaze": gaze,
        "imu": is_aux,
        "local": local,
        "t_slots": t_slots,
        "gp": gp,
        "n_aux": n_aux,
        "n_tok": n_tok,
    }


def gaze_token_mask(gaze_map: torch.Tensor, tubelet: int, grid: int) -> torch.Tensor:
    """``gaze_map`` [1,T,H,W] or [T,H,W] -> flat [T_slots * grid * grid] bool."""
    if gaze_map.dim() == 4:
        gaze_map = gaze_map[0]
    t_frames, hp, wp = gaze_map.shape
    pad = (tubelet - (t_frames % tubelet)) % tubelet
    if pad:
        gaze_map = F.pad(gaze_map, (0, 0, 0, 0, 0, pad))
        t_frames = gaze_map.shape[0]
    slots = t_frames // tubelet
    g = gaze_map.view(slots, tubelet, hp, wp).mean(dim=1)
    pooled = F.adaptive_max_pool2d(g.unsqueeze(1), (grid, grid)).squeeze(1)
    return (pooled > 0).reshape(-1)


# ── RoPE attention stats ─────────────────────────────────────────────────────
def _rope_qk(module, x, mask=None, T=None, H_patches=None, W_patches=None):
    m = module
    B, N, C = x.size()
    grid_depth = int(N // (m.grid_size * m.grid_size)) if m.grid_size else 1
    qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if mask is not None:
        mask_p = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
        d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)
    else:
        if T is None or H_patches is None or W_patches is None:
            mask_p = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=x.device)
        else:
            mask_p = torch.arange(int(T * H_patches * W_patches), device=x.device)
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
        q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
        k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
    else:
        q = torch.cat([qd, qh, qw], dim=-1)
        k = torch.cat([kd, kh, kw], dim=-1)
    return q, k, v


def _union_catalog(masks: dict) -> list[tuple[str, torch.Tensor]]:
    """Cheapest-to-dearest static key sets (paper Eq. 2 analog)."""
    has_gaze = bool(masks["gaze"].any())
    has_imu = bool(masks["imu"].any())
    if has_imu:
        items = [
            ("imu", masks["imu"]),
            ("imu+recent", masks["imu"] | masks["recent"]),
        ]
        if has_gaze:
            items.append(("imu+recent+gaze", masks["imu"] | masks["recent"] | masks["gaze"]))
        return items
    items = [
        ("recent", masks["recent"]),
        ("recent+center", masks["recent"] | masks["center"]),
    ]
    if has_gaze:
        items.append(("recent+center+gaze", masks["recent"] | masks["center"] | masks["gaze"]))
    return items


def profile_qk(q, k, scale: float, masks: dict, chunk: int = 128, recover_t: float = T_RECOVER):
    """Return per-head masses + greedy FastGen policy. q,k: [B,H,N,D]."""
    B, H, N, _ = q.shape
    device = q.device
    unions = _union_catalog(masks)
    acc = {name: torch.zeros(B, H, device=device) for name, _ in unions}
    for n in ("recent", "center", "gaze", "imu", "local"):
        acc.setdefault(n, torch.zeros(B, H, device=device))
    col = torch.zeros(B, H, N, device=device, dtype=torch.float32)
    entropy = torch.zeros(B, H, device=device)
    n_q = 0
    local_mask = masks["local"].to(device=device)
    key_dev = {n: masks[n].to(device=device) for n in ("recent", "center", "gaze", "imu")}
    union_dev = [(n, m.to(device=device)) for n, m in unions]

    for i in range(0, N, chunk):
        q_c = q[:, :, i : i + chunk]
        logits = (q_c.float() @ k.float().transpose(-2, -1)) * scale
        attn = logits.softmax(dim=-1)
        n_q += attn.shape[2]
        col += attn.sum(dim=2)
        entropy += (-attn.clamp_min(1e-12).log() * attn).sum(dim=-1).sum(dim=-1)
        for n, km in key_dev.items():
            acc[n] += attn[..., km].sum(dim=-1).sum(dim=-1)
        for n, km in union_dev:
            if n in key_dev:
                continue
            acc[n] += attn[..., km].sum(dim=-1).sum(dim=-1)
        loc = local_mask[i : i + chunk].to(dtype=attn.dtype)
        acc["local"] += (attn * loc).sum(dim=-1).sum(dim=-1)
        del logits, attn

    n_q = max(n_q, 1)
    masses = {n: (acc[n] / n_q).mean(dim=0) for n in acc}
    entropy = (entropy / n_q).mean(dim=0)

    k_keep = max(1, int(round(R_FREQ * N)))
    top_idx = col.mean(dim=0).topk(k_keep, dim=-1).indices
    hitter_masks = torch.zeros(H, N, device=device, dtype=torch.bool)
    hitter_masks.scatter_(1, top_idx, True)
    base_mask = unions[-1][1].to(device=device) if unions else torch.zeros(N, dtype=torch.bool, device=device)
    base_name = unions[-1][0] if unions else "hitter"
    hitter_mass = torch.zeros(H, device=device)
    union_hitter_mass = torch.zeros(H, device=device)
    n_q2 = 0
    for i in range(0, N, chunk):
        q_c = q[:, :, i : i + chunk]
        logits = (q_c.float() @ k.float().transpose(-2, -1)) * scale
        attn = logits.softmax(dim=-1)
        n_q2 += attn.shape[2]
        for h in range(H):
            hm = hitter_masks[h]
            hitter_mass[h] += attn[0, h][..., hm].sum()
            union_hitter_mass[h] += attn[0, h][..., hm | base_mask].sum()
        del logits, attn
    masses["hitter"] = hitter_mass / max(n_q2, 1)
    masses[base_name + "+hitter"] = union_hitter_mass / max(n_q2, 1)

    assigned = []
    for h in range(H):
        chosen = "full"
        for name, _ in unions:
            if float(masses[name][h]) >= recover_t:
                chosen = name
                break
        else:
            hit_name = base_name + "+hitter"
            if float(masses[hit_name][h]) >= recover_t:
                chosen = hit_name
            elif float(masses["local"][h]) >= recover_t:
                chosen = hit_name + "+local"
            else:
                chosen = "full"
        assigned.append(chosen)

    return {
        "masses": {n: masses[n].detach().float().cpu().tolist() for n in masses},
        "entropy": entropy.detach().float().cpu().tolist(),
        "policy": assigned,
        "n_tok": N,
        "k_hitter": k_keep,
    }


def patch_rope_modules(blocks, bucket: str, layer_stats: dict, mask_fn, chunk: int):
    """Replace RoPEAttention.forward with profile-then-SDPA."""
    originals = []
    for li, block in enumerate(blocks):
        m = block.attn
        orig = m.forward
        originals.append((m, orig))

        def make(layer_idx, attn_mod, orig_fwd):
            def wrapped(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
                q, k, v = _rope_qk(attn_mod, x, mask, T, H_patches, W_patches)
                masks = mask_fn(x.shape[1], attn_mod)
                stats = profile_qk(q, k, attn_mod.scale, masks, chunk=chunk)
                layer_stats[bucket].append((layer_idx, stats))
                with torch.backends.cuda.sdp_kernel():
                    y = F.scaled_dot_product_attention(
                        q, k, v, dropout_p=0.0, is_causal=attn_mod.is_causal, attn_mask=attn_mask
                    )
                y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2])
                y = attn_mod.proj(y)
                y = attn_mod.proj_drop(y)
                return y

            return wrapped

        m.forward = make(li, m, orig)
    return originals


def restore_forwards(originals):
    for mod, orig in originals:
        mod.forward = orig


# ── fusion CA ────────────────────────────────────────────────────────────────
def patch_fusion(fusion, fusion_stats: list, recover_t: float = T_RECOVER):
    originals = []
    for li, layer in enumerate(fusion.layers):
        orig = layer.attn.forward

        def make(layer_idx, orig_fwd):
            def wrapped(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                out, w = orig_fwd(*args, **kwargs)
                # w: [B, heads, Nq, Nk]  (same-slot IMU keys)
                if w is not None:
                    fusion_stats.append(_profile_fusion_attn(layer_idx, w, recover_t))
                return out, w

            return wrapped

        layer.attn.forward = make(li, orig)
        originals.append((layer.attn, orig))
    return originals


def _profile_fusion_attn(layer_idx: int, w: torch.Tensor, recover_t: float):
    """w [B,H,Q,K] — keys are IMU tokens of the same temporal slot."""
    if w.dim() == 3:
        w = w.unsqueeze(1)
    B, H, Q, K = w.shape
    col = w.sum(dim=2)  # [B,H,K]
    k_keep = max(1, int(round(R_FREQ * K)))
    policies = []
    masses_hitter = []
    entropy = []
    for h in range(H):
        attn = w[:, h]  # [B,Q,K]
        ent = float((-(attn.clamp_min(1e-12).log() * attn).sum(dim=-1)).mean())
        entropy.append(ent)
        idx = col[:, h].mean(dim=0).topk(k_keep).indices
        mask = torch.zeros(K, dtype=torch.bool, device=w.device)
        mask[idx] = True
        mass = float(attn[..., mask].sum(dim=-1).mean())
        masses_hitter.append(mass)
        policies.append("hitter" if mass >= recover_t else "full")
    return {
        "layer": layer_idx,
        "n_heads": H,
        "n_keys": K,
        "policy": policies,
        "hitter_mass": masses_hitter,
        "entropy": entropy,
        "mean_key_mass": col.mean(dim=(0, 1)).detach().float().cpu().tolist(),
    }


# ── data ─────────────────────────────────────────────────────────────────────
def load_val_clips(args):
    import pandas as pd

    df = pd.read_csv(args.val_csv)
    transform = VideoTransform(training=False, crop_size=args.img_size)
    samples = []
    video_root = Path(args.video_root)
    n = 0
    for _, row in df.iterrows():
        if n >= args.n_sample:
            break
        video_id = str(row["video_id"])
        path = _video_path(video_root, video_id, file_format=1)
        if not path.is_file():
            # staged layouts sometimes flatten
            alt = video_root / f"{video_id}.MP4"
            path = alt if alt.is_file() else path
        if not path.is_file():
            continue
        try:
            vr = VideoReader(str(path), ctx=cpu(0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip %s: %s", video_id, exc)
            continue
        indices = _indices_for_row(
            vr,
            int(row["start_frame"]),
            int(row["stop_frame"]),
            args.frames,
            args.fps,
            args.anticipation,
            anticipation_point=1.0,
        )
        buf = load_or_decode_clip(
            video_id=video_id,
            indices=indices,
            decode_fn=lambda vr=vr, indices=indices: vr.get_batch(indices).asnumpy().copy(),
        )
        frames = [buf[t] for t in range(buf.shape[0])] if getattr(buf, "ndim", 0) == 4 else buf
        clip = transform(frames)
        if not torch.is_tensor(clip):
            clip = torch.as_tensor(clip)
        meta = {
            "video_id": video_id,
            "original_video_id": str(row.get("original_video_id", video_id)),
            "start_frame": int(row["start_frame"]),
            "stop_frame": int(row["stop_frame"]),
            "frame_indices": indices.tolist(),
            "vfps": float(vr.get_avg_fps()),
            "participant_id": str(row.get("participant_id", "P01")),
        }
        samples.append((clip, meta))
        n += 1
        logger.info("loaded sample %d/%d %s T=%d", n, args.n_sample, video_id, clip.shape[1])
    if not samples:
        raise RuntimeError("No val clips loaded")
    return samples


def gaze_cfg(args) -> dict:
    return {
        "mode": "concat_plus_cross_attn",
        "crop_size": args.img_size,
        "gaze_root": args.gaze_root,
        "extract_root": args.gaze_extract,
        "sync_root": args.gaze_sync,
        "binary_radius_px": 64,
        "pose_map": {"patch_height": 128, "patch_width": 9, "layout": "topleft", "normalize": "none"},
        "pose": {
            "enabled": True,
            "slam_root": args.pose_slam,
            "mapping_json": str(Path(args.pose_slam) / "vrs_to_multi_slam.json"),
            "feature_set": "pose_6d",
            "quality_min": 0.0,
            "history_sec": 0,
            "interframe_k_max": 128,
            "cache_sessions": True,
        },
    }


# ── models ───────────────────────────────────────────────────────────────────
def load_video_model(args, device):
    model = build_model(device, args.frames, args.fps, args.img_size, args.vjepa_ckpt)
    load_lora_sidecars(model, args.video_enc_lora, args.video_pred_lora)
    if hasattr(model.encoder, "use_activation_checkpointing"):
        model.encoder.use_activation_checkpointing = False
    model.eval()
    return model


def load_concat_ca_model(args, device):
    base = build_model(device, args.frames, args.fps, args.img_size, args.vjepa_ckpt)
    load_lora_sidecars(base, args.ca_enc_lora, args.ca_pred_lora)
    if hasattr(base.encoder, "use_activation_checkpointing"):
        base.encoder.use_activation_checkpointing = False
    embed_dim = int(base.embed_dim)
    grid = int(base.grid_size)
    _, _, n_i = compute_token_budgets(grid * grid, gaze_grid_size=10, imu_token_ratio=0.1)
    adapter = BinaryMapInputAdapter(
        hidden_dim=8, scale=1.0, temporal_kernel=3, binary_center=0.0, residual_clamp=1.0, in_channels=5
    ).to(device)
    payload = torch.load(args.ca_adapter, map_location="cpu", weights_only=False)
    key = "input_adapter" if "input_adapter" in payload else None
    adapter.load_state_dict(payload[key] if key else payload, strict=False)
    fusion_cfg = {
        "use_gaze_branch": False,
        "use_imu_branch": True,
        "keep_aux_tokens_in_predictor": True,
        "gaze_grid_size": 10,
        "imu_token_ratio": 0.1,
        "imu_encoder_type": "gru",
        "imu_hidden_dim": 128,
        "fusion_num_heads": 4,
        "fusion_num_layers": 3,
        "use_gated_residual": True,
        "gate_bias_init": -2.0,
        "dropout": 0.0,
    }
    fusion = ProjectedTriModalCrossAttention(
        embed_dim=embed_dim,
        attn_dim=embed_dim,
        num_heads=4,
        num_layers=3,
        dropout=0.0,
        use_gated_residual=True,
        use_gaze_branch=False,
        use_imu_branch=True,
        gate_bias_init=-2.0,
    ).to(device)
    imu_encoder = ImuTemporalEncoder(
        embed_dim=embed_dim, input_dim=6, hidden_dim=128, num_imu_tokens=n_i, encoder_type="gru"
    ).to(device)
    wrapped = ConcatPlusCrossAttnAdaptedModel(
        base, input_adapter=adapter, fusion=fusion, imu_encoder=imu_encoder, fusion_cfg=fusion_cfg, ca_aux="imu"
    )
    load_tri_modal_fusion_checkpoint(wrapped, args.ca_fusion)
    wrapped.eval()
    return wrapped


# ── aggregation ──────────────────────────────────────────────────────────────
POLICY_ORDER = [
    "imu",
    "imu+recent",
    "imu+recent+gaze",
    "imu+recent+hitter",
    "imu+recent+hitter+local",
    "recent",
    "recent+center",
    "recent+center+gaze",
    "recent+center+hitter",
    "recent+center+gaze+hitter",
    "recent+center+hitter+local",
    "hitter",
    "local",
    "full",
]


def collapse_policy(name: str) -> str:
    """Map hybrid names onto FastGen's five display buckets."""
    if name == "full":
        return "full"
    if "local" in name and "hitter" in name:
        return "hitter+local"
    if name.endswith("+local") or name == "local":
        return "local"
    if "hitter" in name or name == "hitter":
        return "hitter"
    if "gaze" in name:
        return "gaze"
    if name.startswith("imu"):
        return "special_imu"
    if name in {"recent", "recent+center"}:
        return "special_recent"
    return name


def reduce_layer_stats(entries):
    """entries: list of (layer_idx, stats) across samples."""
    by_layer = defaultdict(list)
    for li, st in entries:
        by_layer[int(li)].append(st)
    layers = []
    for li in sorted(by_layer):
        sts = by_layer[li]
        n_heads = len(sts[0]["policy"])
        # majority policy per head across samples
        policies = []
        mean_mass = defaultdict(list)
        mean_ent = []
        for h in range(n_heads):
            votes = [s["policy"][h] for s in sts]
            # majority
            pol = max(set(votes), key=votes.count)
            policies.append(pol)
            mean_ent.append(float(np.mean([s["entropy"][h] for s in sts])))
            for k, vals in sts[0]["masses"].items():
                mean_mass[k].append(float(np.mean([s["masses"][k][h] for s in sts])))
        bucket_counts = defaultdict(int)
        for p in policies:
            bucket_counts[collapse_policy(p)] += 1
        layers.append(
            {
                "layer": li,
                "n_heads": n_heads,
                "policy": policies,
                "bucket_counts": dict(bucket_counts),
                "mean_mass": {k: v for k, v in mean_mass.items()},
                "mean_entropy": mean_ent,
                "n_tok": sts[0]["n_tok"],
            }
        )
    return layers


def summarize_model(name, enc_layers, pred_layers, fusion_runs, n_sample):
    def stack_counts(layers):
        buckets = ["special_imu", "special_recent", "gaze", "hitter", "local", "hitter+local", "full"]
        out = {b: [] for b in buckets}
        for ly in layers:
            tot = max(ly["n_heads"], 1)
            cc = ly["bucket_counts"]
            for b in buckets:
                out[b].append(100.0 * cc.get(b, 0) / tot)
        return {"categories": [f"L{ly['layer']}" for ly in layers], "series": out}

    def global_share(layers):
        cc = defaultdict(int)
        n = 0
        for ly in layers:
            for p in ly["policy"]:
                cc[collapse_policy(p)] += 1
                n += 1
        return {k: round(100.0 * v / max(n, 1), 2) for k, v in cc.items()}, n

    enc_share, n_enc = global_share(enc_layers)
    pred_share, n_pred = global_share(pred_layers) if pred_layers else ({}, 0)

    # Which heads need full cache (structurally important / incompressible)
    full_enc = []
    for ly in enc_layers:
        for h, p in enumerate(ly["policy"]):
            if collapse_policy(p) == "full":
                full_enc.append({"layer": ly["layer"], "head": h, "entropy": ly["mean_entropy"][h]})

    mass_summary = {}
    if enc_layers:
        keys = enc_layers[0]["mean_mass"].keys()
        for k in keys:
            vals = [np.mean(ly["mean_mass"][k]) for ly in enc_layers]
            mass_summary[k] = {
                "mean": float(np.mean(vals)),
                "early": float(np.mean(vals[: max(len(vals) // 3, 1)])),
                "mid": float(np.mean(vals[len(vals) // 3 : 2 * len(vals) // 3])),
                "late": float(np.mean(vals[2 * len(vals) // 3 :])),
            }

    fusion_sum = None
    if fusion_runs:
        # majority per (layer, head)
        by_lh = defaultdict(list)
        for run in fusion_runs:
            by_lh[(run["layer"],)].append(run)
        # flatten all captured layer dumps
        layer_heads = defaultdict(list)
        for run in fusion_runs:
            layer_heads[run["layer"]].append(run)
        f_layers = []
        for li in sorted(layer_heads):
            runs = layer_heads[li]
            n_h = runs[0]["n_heads"]
            pols = []
            hit = []
            ent = []
            for h in range(n_h):
                votes = [r["policy"][h] for r in runs]
                pols.append(max(set(votes), key=votes.count))
                hit.append(float(np.mean([r["hitter_mass"][h] for r in runs])))
                ent.append(float(np.mean([r["entropy"][h] for r in runs])))
            f_layers.append({"layer": li, "policy": pols, "hitter_mass": hit, "entropy": ent, "n_keys": runs[0]["n_keys"]})
        n_full = sum(p == "full" for ly in f_layers for p in ly["policy"])
        n_hit = sum(p == "hitter" for ly in f_layers for p in ly["policy"])
        n_tot = n_full + n_hit
        fusion_sum = {
            "layers": f_layers,
            "share": {
                "hitter": round(100.0 * n_hit / max(n_tot, 1), 2),
                "full": round(100.0 * n_full / max(n_tot, 1), 2),
            },
            "n_heads": n_tot,
        }

    return {
        "name": name,
        "n_sample": n_sample,
        "encoder": {
            "layers": enc_layers,
            "share": enc_share,
            "n_heads": n_enc,
            "stack_pct": stack_counts(enc_layers),
            "full_heads": full_enc,
            "mass": mass_summary,
        },
        "predictor": {
            "layers": pred_layers,
            "share": pred_share,
            "n_heads": n_pred,
            "stack_pct": stack_counts(pred_layers) if pred_layers else None,
        },
        "fusion": fusion_sum,
    }


# ── run one model ────────────────────────────────────────────────────────────
def run_model(tag, model, samples, device, args, aux_builders=None):
    if hasattr(model, "tri"):
        core = model.tri.base_model
        fusion_host = model.tri
    elif hasattr(model, "base_model") and hasattr(model.base_model, "encoder"):
        core = model.base_model
        fusion_host = model
    else:
        core = model
        fusion_host = model
    encoder = core.encoder
    predictor = core.predictor
    grid = int(getattr(core, "grid_size", 16))
    tubelet = int(getattr(core, "tubelet_size", 2))

    enc_bucket: dict = defaultdict(list)
    pred_bucket: dict = defaultdict(list)
    fusion_stats: list = []

    gaze_holder = {"mask": None}

    def enc_mask_fn(n_tok, attn_mod):
        g = int(getattr(attn_mod, "grid_size", grid) or grid)
        return build_key_masks(n_tok, g, n_aux=0, gaze_flat=gaze_holder["mask"])

    def pred_mask_fn(n_tok, attn_mod):
        g = int(getattr(attn_mod, "grid_size", grid) or grid)
        n_aux = int(getattr(fusion_host, "_n_aux_context_tokens", 0) or 0)
        return build_key_masks(n_tok, g, n_aux=n_aux, gaze_flat=None)

    enc_orig = patch_rope_modules(list(encoder.blocks), "enc", enc_bucket, enc_mask_fn, args.chunk)
    pred_blocks = getattr(predictor, "predictor_blocks", None) or getattr(predictor, "blocks")
    pred_orig = patch_rope_modules(list(pred_blocks), "pred", pred_bucket, pred_mask_fn, args.chunk)
    fus_orig = []
    if hasattr(model, "fusion") and model.fusion is not None:
        fus_orig = patch_fusion(model.fusion, fusion_stats)

    n_ok = 0
    for i, (clip, meta) in enumerate(samples):
        clip = clip.unsqueeze(0).to(device=device, dtype=torch.float32)
        ant = torch.full((1,), float(args.anticipation), device=device)
        gaze_holder["mask"] = None
        aux_map = None
        imu_batch = None
        if aux_builders is not None:
            builder, imu_loader = aux_builders
            try:
                aux_map = builder.build(clip, [meta])
                gaze_ch = aux_map[:, 0]
                gaze_holder["mask"] = gaze_token_mask(gaze_ch[0], tubelet, grid)
                imu_batch = imu_loader.load_batch([meta], device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("aux failed on %s: %s", meta.get("video_id"), exc)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if aux_builders is not None:
                out = model(clip, ant, binary_map=aux_map, imu_batch=imu_batch)
            else:
                out = model(clip, ant)
        n_aux_now = int(getattr(fusion_host, "_n_aux_context_tokens", 0) or 0)
        if out is None:
            logger.warning("non-finite / None output on sample %d", i)
            continue
        n_ok += 1
        logger.info("%s sample %d/%d ok n_aux=%d", tag, i + 1, len(samples), n_aux_now)

    restore_forwards(enc_orig)
    restore_forwards(pred_orig)
    restore_forwards(fus_orig)

    enc_layers = reduce_layer_stats(enc_bucket["enc"])
    pred_layers = reduce_layer_stats(pred_bucket["pred"])
    return summarize_model(tag, enc_layers, pred_layers, fusion_stats, n_ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sample", type=int, default=24)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--anticipation", type=float, default=1.0)
    p.add_argument("--val-csv", default="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv")
    p.add_argument("--video-root", default="/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_videos")
    p.add_argument("--vjepa-ckpt", default="/scratch/ll5914/models/vjepa2/vitl.pt")
    p.add_argument(
        "--video-enc-lora",
        default="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/encoder_lora_best.pt",
    )
    p.add_argument(
        "--video-pred-lora",
        default="/scratch/ll5914/experiments/p01_video_pred_joint_heads_clip/action_anticipation_frozen/p01-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt",
    )
    ca = "/scratch/ll5914/experiments/concat_plus_cross_attn_v2/action_anticipation_frozen/concat-plus-ca-v2-L3-keepaux-softgate-vitl16-256-12ep-1xh100"
    p.add_argument("--ca-enc-lora", default=f"{ca}/encoder_lora_best.pt")
    p.add_argument("--ca-pred-lora", default=f"{ca}/predictor_lora_best.pt")
    p.add_argument("--ca-adapter", default=f"{ca}/binary_input_adapter_best.pt")
    p.add_argument("--ca-fusion", default=f"{ca}/tri_modal_fusion_best.pt")
    p.add_argument("--gaze-root", default="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze")
    p.add_argument("--gaze-extract", default="/scratch/ll5914/datasets/HD-EPIC/_gaze_extract")
    p.add_argument("--gaze-sync", default="/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos")
    p.add_argument("--pose-slam", default="/scratch/ll5914/datasets/HD-EPIC/SLAM-and-Gaze/P01/SLAM/multi")
    p.add_argument("--out", default="/scratch/ll5914/experiments/fastgen_head_profile/summary.json")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-ca", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("TRI_MODAL_FRAME_CACHE", "/scratch/ll5914/datasets/HD-EPIC/_clip_frame_cache/p01_f32_at1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s n_sample=%d recover_T=%.2f", device, args.n_sample, T_RECOVER)
    samples = load_val_clips(args)
    logger.info("loaded %d clips", len(samples))

    report = {
        "paper": "arXiv:2310.01801 FastGen",
        "recover_T": T_RECOVER,
        "r_recent": R_RECENT,
        "r_freq": R_FREQ,
        "r_local_t": R_LOCAL_T,
        "r_local_s": R_LOCAL_S,
        "center_frac": CENTER_FRAC,
        "n_clips_loaded": len(samples),
        "models": {},
    }

    if not args.skip_video:
        logger.info("=== video-only ===")
        video = load_video_model(args, device)
        report["models"]["video_only"] = run_model("video_only", video, samples, device, args, aux_builders=None)
        del video
        torch.cuda.empty_cache()

    if not args.skip_ca:
        logger.info("=== concat+CA (gaze+IMU) ===")
        ca_model = load_concat_ca_model(args, device)
        cfg = gaze_cfg(args)
        gate = GazeTokenGate({**cfg, "mode": "token_gate"})
        builder = GazePoseInputMapBuilder(cfg, gate=gate)
        imu_loader = ImuTrajectoryLoader(cfg, gate=gate)
        report["models"]["gaze_imu"] = run_model(
            "gaze_imu", ca_model, samples, device, args, aux_builders=(builder, imu_loader)
        )
        del ca_model
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
