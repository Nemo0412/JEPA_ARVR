#!/usr/bin/env python3
"""Pure-Python FLOPs estimate (no torch) for RU-LSTM vs V-JEPA ViT-L."""
from __future__ import annotations

import json
from pathlib import Path


def gflops(n: float) -> float:
    return n / 1e9


def mm(a: float, b: float, c: float) -> float:
    """FLOPs for matmul (A,B)@(B,C); count 2 per MAC."""
    return 2.0 * a * b * c


def layernorm(n: float, d: float) -> float:
    return 5.0 * n * d


def linear(n: float, din: float, dout: float) -> float:
    return mm(n, din, dout) + n * dout  # +bias


def gelu(n: float) -> float:
    return 8.0 * n  # rough


def lstm_step(feat_in: float, hidden: float, depth: int = 1) -> float:
    """One LSTM timestep, `depth` layers. Input size = feat_in for layer0, else hidden."""
    total = 0.0
    din = feat_in
    for _ in range(depth):
        # 4 gates: ih + hh (+bias negligible in mm term; add 4*hidden)
        total += mm(1, din, 4 * hidden) + mm(1, hidden, 4 * hidden) + 4 * hidden
        # elementwise gate nonlinearities / cell ~ 10*hidden
        total += 10.0 * hidden
        din = hidden
    return total


def rulstm_temporal_flops(
    *,
    context_sec: float,
    alpha: float = 0.25,
    feat_in: int = 1024,
    hidden: int = 1024,
    depth: int = 1,
    input_proj: bool = False,
    mlp_head: bool = False,
    num_verb: int = 82,
    num_noun: int = 150,
    num_action: int = 1063,
    horizons=(2.0, 4.0, 6.0),
    use_layernorm: bool = True,
) -> dict:
    T = max(1, int(round(context_sec / alpha)))
    fl = 0.0
    # feat LN
    if use_layernorm:
        fl += layernorm(T, feat_in)
    din = feat_in
    if input_proj:
        fl += linear(T, feat_in, hidden) + gelu(T * hidden)
        din = hidden
    # rolling over T
    fl += T * lstm_step(din, hidden, depth)
    # unrolling per horizon
    for h in horizons:
        steps = max(1, int(round(h / alpha)))
        fl += steps * lstm_step(din, hidden, depth)
        # out LN + 3 heads on last state
        if use_layernorm:
            fl += layernorm(1, hidden)
        if mlp_head:
            for ncls in (num_verb, num_noun, num_action):
                fl += linear(1, hidden, hidden) + gelu(hidden) + linear(1, hidden, ncls)
        else:
            for ncls in (num_verb, num_noun, num_action):
                fl += linear(1, hidden, ncls)
    return {"T_feat": T, "flops": fl, "gflops": gflops(fl)}


def param_count_rulstm(
    *,
    feat_in=1024,
    hidden=1024,
    depth=1,
    input_proj=False,
    mlp_head=False,
    num_verb=82,
    num_noun=150,
    num_action=1063,
    use_layernorm=True,
) -> float:
    p = 0
    if use_layernorm:
        p += 2 * feat_in  # feat LN
        p += 2 * hidden   # out LN
    if input_proj:
        p += feat_in * hidden + hidden
    # LSTM params per layer: 4*((din*h)+(h*h)+2*h) for PyTorch (bias_ih+bias_hh)
    din = hidden if input_proj else feat_in
    for layer in range(depth):
        d = din if layer == 0 else hidden
        one = 4 * (d * hidden + hidden * hidden + 2 * hidden)
        p += 2 * one  # rolling + unrolling
    for ncls in (num_verb, num_noun, num_action):
        if mlp_head:
            p += hidden * hidden + hidden + hidden * ncls + ncls
        else:
            p += hidden * ncls + ncls
    return p / 1e6


def vit_encoder_flops(T: int, img=256, patch=16, tubelet=2, dim=1024, depth=24, mlp_ratio=4.0, in_chans=3):
    gh = gw = img // patch
    gt = T // tubelet
    N = gt * gh * gw
    mlp_h = int(dim * mlp_ratio)
    patch_embed = in_chans * tubelet * patch * patch * dim * N  # MACs≈FLOPs here (1 mul)
    # use 2/MAC for dense matmuls
    per = (
        mm(N, dim, 3 * dim)
        + mm(N, N, dim)
        + mm(N, N, dim)
        + mm(N, dim, dim)
        + mm(N, dim, mlp_h)
        + mm(N, mlp_h, dim)
        + 2 * layernorm(N, dim)
    )
    total = patch_embed + depth * per + layernorm(N, dim)
    return {"T_frames": T, "n_tokens": N, "flops": total, "gflops": gflops(total)}


def predictor_flops(n_ctx: int, keep=4096, dim=384, depth=12, mlp_ratio=4.0, n_pred=256):
    N = min(n_ctx, keep) + n_pred
    mlp_h = int(dim * mlp_ratio)
    per = (
        mm(N, dim, 3 * dim)
        + mm(N, N, dim)
        + mm(N, N, dim)
        + mm(N, dim, dim)
        + mm(N, dim, mlp_h)
        + mm(N, mlp_h, dim)
        + 2 * layernorm(N, dim)
    )
    total = depth * per
    return {"n_tokens": N, "flops": total, "gflops": gflops(total)}


def bninception_flops(h=256, w=454) -> float:
    """Scale known BN-Inception ~2.0 GFLOPs @224^2 by spatial area.

    Literature: BN-Inception ≈ 2.0–2.1 GFLOPs at 224×224 (multiply-adds×2 style varies;
    we use 2.05e9 FLOPs @224^2 as a mid estimate, then scale by HxW).
    """
    base = 2.05e9
    return base * (h * w) / (224 * 224)


def main():
    contexts = [4.0, 6.0, 8.0, 10.0]
    variants = {
        "original": dict(hidden=1024, depth=1, input_proj=False, mlp_head=False),
        "v1": dict(hidden=2048, depth=4, input_proj=True, mlp_head=True),
        "v2": dict(hidden=4096, depth=1, input_proj=True, mlp_head=True),
    }

    out = {"rulstm_temporal": {}, "tsn": {}, "vjepa": [], "comparison_at_10s": {}}

    print("=" * 72)
    print("1) RU-LSTM temporal head")
    for name, cfg in variants.items():
        pm = param_count_rulstm(**cfg)
        rows = []
        print(f"  [{name}] ~{pm:.1f}M params")
        for c in contexts:
            r = rulstm_temporal_flops(context_sec=c, **cfg)
            r["params_m"] = pm
            r["context_sec"] = c
            rows.append(r)
            print(f"    ctx={c:4.1f}s T={r['T_feat']:3d}  {r['gflops']:.3f} GFLOPs")
        out["rulstm_temporal"][name] = rows

    print("=" * 72)
    print("2) TSN BN-Inception frontend")
    per = bninception_flops()
    out["tsn"] = {"per_frame_gflops": gflops(per), "by_context": [], "method": "scaled_from_224"}
    print(f"  per frame 256x454 ≈ {gflops(per):.2f} GFLOPs")
    for c in contexts:
        n = int(round(c / 0.25))
        row = {
            "context_sec": c,
            "n_frames": n,
            "gflops": gflops(per * n),
            "incremental_2s_gflops": gflops(per * 8),
        }
        out["tsn"]["by_context"].append(row)
        print(f"    ctx={c:4.1f}s ×{n} = {row['gflops']:.1f} GFLOPs  (+2s tick ≈ {row['incremental_2s_gflops']:.1f})")

    print("=" * 72)
    print("3) V-JEPA ViT-L (fps=8, 256^2)")
    for c in contexts:
        T = int(round(c * 8))
        if T % 2:
            T += 1
        enc = vit_encoder_flops(T)
        pred = predictor_flops(enc["n_tokens"])
        row = {
            "context_sec": c,
            **enc,
            "predictor_gflops": pred["gflops"],
            "full_gflops": enc["gflops"] + pred["gflops"],
        }
        out["vjepa"].append(row)
        print(
            f"  ctx={c:4.1f}s T={T} tokens={enc['n_tokens']}  "
            f"enc={enc['gflops']:.0f}  pred={pred['gflops']:.0f}  sum={row['full_gflops']:.0f} GFLOPs"
        )

    o = out["rulstm_temporal"]["original"][-1]
    v2 = out["rulstm_temporal"]["v2"][-1]
    tsn = out["tsn"]["by_context"][-1]
    j = out["vjepa"][-1]
    full_o = tsn["gflops"] + o["gflops"]
    full_v2 = tsn["gflops"] + v2["gflops"]

    out["comparison_at_10s"] = {
        "rulstm_orig_temporal_gflops": o["gflops"],
        "rulstm_v2_temporal_gflops": v2["gflops"],
        "rulstm_v2_over_orig_temporal": v2["gflops"] / o["gflops"],
        "tsn_10s_gflops": tsn["gflops"],
        "rulstm_orig_full_gflops": full_o,
        "rulstm_v2_full_gflops": full_v2,
        "vjepa_encoder_gflops": j["gflops"],
        "vjepa_full_gflops": j["full_gflops"],
        "vitl_enc_over_v2_temporal": j["gflops"] / v2["gflops"],
        "jepa_full_over_v2_full": j["full_gflops"] / full_v2,
        "jepa_full_over_orig_full": j["full_gflops"] / full_o,
        "note": (
            "Params were aligned (~328M vs ~304M) but compute was not: "
            "v2 temporal head is still << ViT-L pixel encoder; "
            "even full RU-LSTM+TSN pipeline is far below JEPA at 10s context."
        ),
    }

    print("=" * 72)
    print("4) Gap @ 10s (val-dominant)")
    c10 = out["comparison_at_10s"]
    print(f"  RU-LSTM orig temporal     : {c10['rulstm_orig_temporal_gflops']:.3f} GFLOPs")
    print(f"  RU-LSTM v2 temporal       : {c10['rulstm_v2_temporal_gflops']:.3f} GFLOPs  ({c10['rulstm_v2_over_orig_temporal']:.1f}× orig)")
    print(f"  TSN 40 frames             : {c10['tsn_10s_gflops']:.1f} GFLOPs")
    print(f"  RU-LSTM v2 FULL           : {c10['rulstm_v2_full_gflops']:.1f} GFLOPs")
    print(f"  V-JEPA ViT-L encoder      : {c10['vjepa_encoder_gflops']:.0f} GFLOPs")
    print(f"  V-JEPA enc+pred           : {c10['vjepa_full_gflops']:.0f} GFLOPs")
    print(f"  ViT-L enc / v2 temporal   : {c10['vitl_enc_over_v2_temporal']:.0f}×")
    print(f"  JEPA full / v2 full       : {c10['jepa_full_over_v2_full']:.1f}×")

    paths = [
        Path("/home/ll5914/Jepa_baseline/rulstm_hdepic/flops_comparison.json"),
        Path("/scratch/ll5914/experiments/rulstm_hdepic_p01_stream_vitl_aligned_v2/flops_comparison.json"),
    ]
    text = json.dumps(out, indent=2)
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
