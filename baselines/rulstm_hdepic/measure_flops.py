#!/usr/bin/env python3
"""FLOPs gap: Streaming RU-LSTM vs V-JEPA ViT-L (same HD-EPIC stream protocol).

RU-LSTM / TSN: measured with torch FlopCounterMode.
V-JEPA ViT-L: closed-form ViT FLOPs (avoids slow CPU forward on 80-frame clips).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rulstm_hdepic"))
sys.path.insert(0, str(ROOT / "rulstm" / "RULSTM"))

from train_stream_rulstm import StreamingRULSTM  # noqa: E402


def gflops(n: float) -> float:
    return float(n) / 1e9


def count_flops(fn, *args, **kwargs) -> int:
    with FlopCounterMode(display=False) as mode:
        fn(*args, **kwargs)
    return int(mode.get_total_flops())


def build_rulstm(variant: str, num_verb=82, num_noun=150, num_action=1063) -> StreamingRULSTM:
    kw = dict(
        num_verb=num_verb, num_noun=num_noun, num_action=num_action,
        feat_in=1024, use_layernorm=True, trunk_layers=0,
    )
    if variant == "original":
        return StreamingRULSTM(**kw, hidden=1024, depth=1, input_proj=False, mlp_head=False).eval()
    if variant == "v1":
        return StreamingRULSTM(**kw, hidden=2048, depth=4, input_proj=True, mlp_head=True).eval()
    if variant == "v2":
        return StreamingRULSTM(**kw, hidden=4096, depth=1, input_proj=True, mlp_head=True).eval()
    raise ValueError(variant)


def rulstm_flops(model: StreamingRULSTM, context_sec: float, alpha: float = 0.25) -> dict:
    T = max(1, int(round(context_sec / alpha)))
    feats = torch.randn(1, T, 1024)
    lengths = torch.tensor([T], dtype=torch.long)
    fl = count_flops(model, feats, lengths)
    return {
        "context_sec": context_sec,
        "T_feat": T,
        "params_m": model.count_parameters()["total_m"],
        "flops": fl,
        "gflops": gflops(fl),
    }


def tsn_flops_per_frame() -> int:
    from pretrainedmodels import bninception

    model = bninception(pretrained=None)
    model.last_linear = nn.Identity()
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.eval()
    x = torch.randn(1, 3, 256, 454)
    return count_flops(model, x)


def vit_encoder_flops(
    *,
    T: int,
    img_size: int = 256,
    patch: int = 16,
    tubelet: int = 2,
    dim: int = 1024,
    depth: int = 24,
    mlp_ratio: float = 4.0,
    in_chans: int = 3,
) -> dict:
    """Standard ViT FLOPs (MAC×2 counted as FLOPs via 2*MAC convention of many profilers).

    Per block (approx, matching common fvcore/thop style for matmuls):
      attn: QKV 3 N D^2 + proj N D^2 + scores N^2 D + attn@V N^2 D
          = 4 N D^2 + 2 N^2 D
      mlp:  2 N D (mlp_ratio D) = 2 * mlp_ratio * N D^2
    Patch embed: in_chans * tubelet * patch^2 * dim * N
    """
    assert T % tubelet == 0
    gh = gw = img_size // patch
    gt = T // tubelet
    N = gt * gh * gw
    mlp_hidden = int(dim * mlp_ratio)

    patch_embed = in_chans * tubelet * patch * patch * dim * N
    # use 2 FLOPs per MAC for matmul-heavy terms (FLOPs = 2*MACs)
    def mm(a, b, c):
        return 2 * a * b * c

    per_block = (
        mm(N, dim, 3 * dim)  # qkv
        + mm(N, N, dim)      # q@k^T  (heads folded into dim)
        + mm(N, N, dim)      # attn@v
        + mm(N, dim, dim)    # out proj
        + mm(N, dim, mlp_hidden)
        + mm(N, mlp_hidden, dim)
    )
    # LayerNorm ~ 5 N D (minor); include for completeness
    per_block += 2 * (5 * N * dim)

    total = patch_embed + depth * per_block
    # final norm
    total += 5 * N * dim
    return {
        "T_frames": T,
        "n_tokens": N,
        "params_m": 304.0,  # ViT-L/16 nominal
        "flops": total,
        "gflops": gflops(total),
        "method": "analytical_vit_matmul_x2",
    }


def predictor_flops(
    n_ctx_tokens: int,
    keep_count: int = 4096,
    *,
    dim: int = 384,
    depth: int = 12,
    mlp_ratio: float = 4.0,
    n_pred_tokens: int = 256,  # 1 tubelet × 16×16 for num_output_frames=2
) -> dict:
    N = min(n_ctx_tokens, keep_count) + n_pred_tokens
    mlp_hidden = int(dim * mlp_ratio)

    def mm(a, b, c):
        return 2 * a * b * c

    per_block = (
        mm(N, dim, 3 * dim)
        + mm(N, N, dim)
        + mm(N, N, dim)
        + mm(N, dim, dim)
        + mm(N, dim, mlp_hidden)
        + mm(N, mlp_hidden, dim)
        + 2 * (5 * N * dim)
    )
    total = depth * per_block
    return {
        "n_tokens": N,
        "n_ctx_kept": min(n_ctx_tokens, keep_count),
        "flops": total,
        "gflops": gflops(total),
        "method": "analytical_vit_predictor",
    }


def main():
    contexts = [4.0, 6.0, 8.0, 10.0]
    out = {
        "protocol": {
            "rulstm_feat_fps": 4.0,
            "jepa_fps": 8,
            "jepa_img": 256,
            "horizons_sec": [2, 4, 6],
            "keep_count": 4096,
            "note": (
                "Per-tick batch=1. RU-LSTM temporal uses TSN features; "
                "full RU pipeline = TSN(all ctx frames)+temporal. "
                "JEPA = ViT-L encoder on pixels (+ predictor on pruned tokens)."
            ),
        },
        "rulstm_temporal": {},
        "tsn": {},
        "vjepa_vitl_encoder": [],
        "comparison_at_10s": {},
    }

    print("=" * 72, flush=True)
    print("1) RU-LSTM temporal head FLOPs", flush=True)
    for variant in ("original", "v1", "v2"):
        model = build_rulstm(variant)
        rows = []
        for c in contexts:
            r = rulstm_flops(model, c)
            rows.append(r)
            print(
                f"  {variant:8s} ctx={c:4.1f}s T={r['T_feat']:3d} "
                f"params={r['params_m']:7.1f}M  FLOPs={r['gflops']:8.3f} GFLOPs",
                flush=True,
            )
        out["rulstm_temporal"][variant] = rows

    print("=" * 72, flush=True)
    print("2) TSN-BNInception frontend", flush=True)
    fl_frame = tsn_flops_per_frame()
    out["tsn"] = {"per_frame_flops": fl_frame, "per_frame_gflops": gflops(fl_frame), "by_context": []}
    print(f"  per frame 256x454: {gflops(fl_frame):.3f} GFLOPs", flush=True)
    for c in contexts:
        n = int(round(c / 0.25))
        total = fl_frame * n
        row = {
            "context_sec": c,
            "n_frames": n,
            "flops": total,
            "gflops": gflops(total),
            "incremental_2s_gflops": gflops(fl_frame * 8),
        }
        out["tsn"]["by_context"].append(row)
        print(
            f"  ctx={c:4.1f}s ×{n:3d} = {gflops(total):8.1f} GFLOPs "
            f"(+2s incremental ≈ {gflops(fl_frame*8):.1f})",
            flush=True,
        )

    print("=" * 72, flush=True)
    print("3) V-JEPA ViT-L encoder (+ predictor approx)", flush=True)
    for c in contexts:
        T = int(round(c * 8))
        if T % 2:
            T += 1
        enc = vit_encoder_flops(T=T)
        enc["context_sec"] = c
        pred = predictor_flops(enc["n_tokens"])
        enc["predictor"] = pred
        enc["encoder_plus_predictor_gflops"] = enc["gflops"] + pred["gflops"]
        out["vjepa_vitl_encoder"].append(enc)
        print(
            f"  ctx={c:4.1f}s T={T:3d} tokens={enc['n_tokens']:5d} "
            f"enc={enc['gflops']:8.1f}  pred≈{pred['gflops']:7.1f}  "
            f"sum={enc['encoder_plus_predictor_gflops']:8.1f} GFLOPs",
            flush=True,
        )

    r10_orig = out["rulstm_temporal"]["original"][-1]
    r10_v2 = out["rulstm_temporal"]["v2"][-1]
    tsn10 = out["tsn"]["by_context"][-1]
    j10 = out["vjepa_vitl_encoder"][-1]

    full_v2 = tsn10["gflops"] + r10_v2["gflops"]
    full_orig = tsn10["gflops"] + r10_orig["gflops"]
    jepa = j10["encoder_plus_predictor_gflops"]

    out["comparison_at_10s"] = {
        "rulstm_original_temporal_gflops": r10_orig["gflops"],
        "rulstm_v2_temporal_gflops": r10_v2["gflops"],
        "tsn_10s_gflops": tsn10["gflops"],
        "rulstm_original_full_gflops": full_orig,
        "rulstm_v2_full_gflops": full_v2,
        "vjepa_encoder_gflops": j10["gflops"],
        "vjepa_predictor_gflops": j10["predictor"]["gflops"],
        "vjepa_full_gflops": jepa,
        "ratio_vitl_enc_over_v2_temporal": j10["gflops"] / max(r10_v2["gflops"], 1e-12),
        "ratio_jepa_full_over_v2_full": jepa / max(full_v2, 1e-12),
        "ratio_jepa_full_over_orig_full": jepa / max(full_orig, 1e-12),
        "ratio_v2_temporal_over_orig_temporal": r10_v2["gflops"] / max(r10_orig["gflops"], 1e-12),
    }

    print("=" * 72, flush=True)
    print("4) Gap @ 10s context", flush=True)
    c10 = out["comparison_at_10s"]
    for k, v in c10.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4g}", flush=True)

    out_path = Path(
        "/scratch/ll5914/experiments/rulstm_hdepic_p01_stream_vitl_aligned_v2/flops_comparison.json"
    )
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    local = ROOT / "rulstm_hdepic" / "flops_comparison.json"
    local.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}\nWrote {local}", flush=True)


if __name__ == "__main__":
    main()
