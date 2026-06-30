#!/usr/bin/env python
"""Comprehensive per-module FLOPs re-measurement for all B12 comparison methods.

Covers every method in the 4s and 1min comparison (V-JEPA2 + Qwen2.5-VL-3B).
All numbers measured fresh in a single job for consistency.

V-JEPA2 module breakdown (encoder | predictor | probe):
  - 4s  (32f,  keep=4096, no-prune)
  - 1min (480f, keep=4096, rebase)    — prune4096
  - 1min (480f, keep=4096, true-pos)  — prune4096 true
  - 1min (480f, keep=61440, true-pos) — full-context, no-prune
  - 1min (480f, mid next_attn 9:0.5,17:4096, true-pos)
  - 1min (480f, mid feature_norm 8:0.5,16:4096, true-pos)
  - 7s (56f, keep=7168, true-pos) — full-context, no-prune
  - 7s (56f, mid next_attn 9:0.5,17:4096, true-pos)

Qwen2.5-VL-3B module breakdown (vision tower | LLM):
  - 4s  (32f,  keep=1.0)
  - 4s  (32f,  keep=0.15)
  - 1min (480f, keep=1.0)
  - 1min (480f, keep=0.0667)
  - 7s (56f, keep=1.0)

For each config reports:
  forward_TF / trainstep_TF for each module, total, and op-level breakdown.

Outputs (all to OUT_DIR = outputs/b12_flops_profile/):
  vjepa_rerun_4s.json
  vjepa_rerun_1min_prune4096_rebase.json
  vjepa_rerun_1min_prune4096_true.json
  vjepa_rerun_1min_fullctx61440_true.json
  qwen_rerun_32f_kr1p0.json
  qwen_rerun_32f_kr0p15.json
  qwen_rerun_480f_kr1p0.json
  qwen_rerun_480f_kr0p0667.json
  module_breakdown_summary.json      — consolidated table
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/path/to/VJEPA2-EXP")
OUT_DIR = Path(os.environ.get("OUT_DIR", f"{SHARED}/outputs/b12_flops_profile"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TF = 1e12


# ---------------------------------------------------------------------------
# GQA SDPA patch (required for Qwen)
# ---------------------------------------------------------------------------

def _patch_sdpa_flops_for_gqa():
    import torch.utils.flop_counter as fc
    bmm_flop = fc.bmm_flop

    def sdpa_flop_count(query_shape, key_shape, value_shape):
        b, h, s_q, d_q = query_shape
        _b2, _h2, s_k, _d2 = key_shape
        _b3, _h3, _s3, d_v = value_shape
        return (bmm_flop((b*h, s_q, d_q), (b*h, d_q, s_k))
                + bmm_flop((b*h, s_q, s_k), (b*h, s_k, d_v)))

    def sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape):
        b, h, s_q, d_q = query_shape
        _b2, _h2, s_k, _d2 = key_shape
        _b3, _h3, _s3, d_v = value_shape
        return (bmm_flop((b*h, s_q, d_q), (b*h, d_q, s_k))
                + bmm_flop((b*h, s_q, d_v), (b*h, d_v, s_k))
                + bmm_flop((b*h, s_k, s_q), (b*h, s_q, d_v))
                + bmm_flop((b*h, s_q, s_k), (b*h, s_k, d_q))
                + bmm_flop((b*h, d_q, s_q), (b*h, s_q, s_k)))

    fc.sdpa_flop_count = sdpa_flop_count
    fc.sdpa_backward_flop_count = sdpa_backward_flop_count


_patch_sdpa_flops_for_gqa()


def _count(fn):
    """Run fn under FlopCounterMode; return (total_flops_int, breakdown_dict_TF)."""
    c = FlopCounterMode(display=False)
    with c:
        fn()
    torch.cuda.synchronize()
    g = c.get_flop_counts().get("Global", {})
    bd = {str(op): round(int(f) / TF, 5)
          for op, f in sorted(g.items(), key=lambda kv: -kv[1])}
    return int(c.get_total_flops()), bd


def _write(name: str, payload: dict):
    p = OUT_DIR / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[profile] wrote {p}", flush=True)
    return payload


# ===========================================================================
# V-JEPA2 profiling
# ===========================================================================

def _build_vjepa(num_frames: int, dev: torch.device):
    import app.hdepic_lora_action_anticipation.train_vjepa_prune_anticipation as M
    ckpt = f"{SHARED}/checkpoints/vitl.pt"
    model = M.build_model(dev, num_frames, 8, 256, ckpt)
    M.apply_lora_to_predictor(model.predictor, rank=8, alpha=16.0)
    gp = int(model.grid_size) ** 2
    return model, M, gp


def _sanitize_tag(s: str) -> str:
    return s.replace(":", "x").replace(",", "_").replace(".", "p").replace(" ", "")


def profile_vjepa(
    num_frames: int,
    keep_count: int,
    position_mode: str,
    warmup: int = 1,
    encoder_prune_schedule: str = "",
    encoder_prune_metric: str = "",
):
    """Measure encoder | predictor | probe FLOPs for one V-JEPA2 config."""
    import app.hdepic_lora_action_anticipation.train_vjepa_prune_anticipation as M

    mid_suffix = ""
    if encoder_prune_schedule:
        mid_suffix = f"_mid{encoder_prune_metric}_{_sanitize_tag(encoder_prune_schedule)}"
    tag = f"vjepa_rerun_nf{num_frames}_keep{keep_count}_{position_mode}{mid_suffix}"
    print(f"\n{'='*64}", flush=True)
    print(f"[profile] V-JEPA2  nf={num_frames}  keep={keep_count}  pos={position_mode}"
          f"  mid={encoder_prune_metric or '-'} {encoder_prune_schedule or ''}", flush=True)

    dev = torch.device("cuda")
    model, M, gp = _build_vjepa(num_frames, dev)
    n_full = (num_frames // int(model.tubelet_size)) * gp
    model._position_mode = position_mode
    model._full_n = n_full

    if position_mode == "true":
        model.predictor.num_patches = ((n_full // gp) + 8) * gp

    no_prune = keep_count >= n_full and not encoder_prune_schedule
    pruner = None
    mid_pruner = None
    if encoder_prune_schedule:
        schedule = M.parse_encoder_prune_schedule(encoder_prune_schedule, n_full, gp)
        mid_pruner = M.MidEncoderPruner(
            model.encoder,
            schedule=schedule,
            gp=gp,
            metric=encoder_prune_metric,
            chunk_size=256,
        )
        keep_count = mid_pruner.keep_count
    elif not no_prune:
        pruner = M.TokenPruner(model.encoder, keep_count=keep_count, gp=gp, chunk_size=256)

    probe = M.HDEpicProbe(model.embed_dim, 106, 300, 1700).to(dev)
    clips = torch.randn(1, 3, num_frames, 256, 256, device=dev)
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    # ---- (A) Encoder only ----
    def enc_only():
        with torch.no_grad(), ac():
            model.encoder(clips)

    print("  [A] encoder only …", flush=True)
    for _ in range(warmup):
        enc_only()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    enc_f, enc_bd = _count(enc_only)
    enc_wall = time.perf_counter() - t0
    print(f"      encoder fwd = {enc_f/TF:.3f} TF  ({enc_wall:.1f}s to count)", flush=True)

    # ---- (B) Encode + prune ----
    prune_overhead_f = 0
    if mid_pruner is not None:
        encprune_f = enc_f
        encprune_bd = enc_bd
        ctx_tokens = torch.randn(1, keep_count, model.embed_dim, device=dev)
        ctx_idx = torch.arange(keep_count, device=dev).unsqueeze(0)
        print(f"      mid-encoder encode fwd = {encprune_f/TF:.3f} TF  "
              f"(output tokens {keep_count})", flush=True)
    elif pruner is not None:
        def enc_prune():
            with torch.no_grad(), ac():
                M.encode_and_prune(model, pruner, clips)

        for _ in range(warmup):
            enc_prune()
        torch.cuda.synchronize()
        encprune_f, encprune_bd = _count(enc_prune)
        prune_overhead_f = encprune_f - enc_f
        print(f"      encode+prune fwd = {encprune_f/TF:.3f} TF  (prune overhead {prune_overhead_f/TF:.3f} TF)", flush=True)
        ctx_tokens = torch.randn(1, keep_count, model.embed_dim, device=dev)
        ctx_idx = torch.arange(keep_count, device=dev).unsqueeze(0)
    else:
        encprune_f = enc_f
        encprune_bd = enc_bd
        ctx_tokens = torch.randn(1, n_full, model.embed_dim, device=dev)
        ctx_idx = torch.arange(n_full, device=dev).unsqueeze(0)

    # ---- (C) Predictor only ----
    lora_params = [p for p in model.predictor.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(list(probe.parameters()) + lora_params, lr=1e-4)
    crit = nn.CrossEntropyLoss()
    lab = torch.zeros(1, dtype=torch.long, device=dev)

    def pred_only():
        with ac():
            M.anticipate_from_ctx(model, ctx_tokens, 1.0, ctx_idx=ctx_idx)

    print("  [C] predictor only …", flush=True)
    for _ in range(warmup):
        pred_only()
    torch.cuda.synchronize()
    pred_f, pred_bd = _count(pred_only)
    print(f"      predictor fwd = {pred_f/TF:.3f} TF", flush=True)

    # ---- (D) Probe only ----
    n_pred_slots = round(1.0 * 8 / 2)
    n_pred_tokens = n_pred_slots * gp  # 1024 for 1s anticipation
    dummy_pred_out = torch.randn(1, n_pred_tokens, model.embed_dim, device=dev)

    def probe_only():
        with ac():
            probe(dummy_pred_out.float())

    print("  [D] probe only …", flush=True)
    probe_f, probe_bd = _count(probe_only)
    print(f"      probe fwd = {probe_f/TF:.4f} TF", flush=True)

    # ---- (E) Predictor+probe combined fwd ----
    def pred_probe_fwd():
        with ac():
            feats = M.anticipate_from_ctx(model, ctx_tokens, 1.0, ctx_idx=ctx_idx)
            probe(feats.float())

    torch.cuda.empty_cache()
    pred_probe_fwd_f, pred_probe_fwd_bd = _count(pred_probe_fwd)

    # ---- (F) Predictor+probe train step ----
    def pred_probe_step():
        opt.zero_grad(set_to_none=True)
        with ac():
            feats = M.anticipate_from_ctx(model, ctx_tokens, 1.0, ctx_idx=ctx_idx)
            v, n, a = probe(feats.float())
            loss = crit(v, lab) + crit(n, lab) + crit(a, lab)
        loss.backward()
        opt.step()

    print("  [F] predictor+probe train step …", flush=True)
    for _ in range(warmup):
        pred_probe_step()
    torch.cuda.synchronize()
    step_f, step_bd = _count(pred_probe_step)
    print(f"      pred+probe trainstep = {step_f/TF:.3f} TF", flush=True)

    # Wall-time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred_probe_step()
    torch.cuda.synchronize()
    wall_step = time.perf_counter() - t0

    result = {
        "config": {
            "num_frames": num_frames,
            "keep_count": keep_count,
            "n_tokens_full": n_full,
            "position_mode": position_mode,
            "no_prune": no_prune,
            "encoder_prune_schedule": encoder_prune_schedule,
            "encoder_prune_metric": encoder_prune_metric,
        },
        "encoder_fwd_TF": enc_f / TF,
        "prune_overhead_TF": prune_overhead_f / TF,
        "encode_plus_prune_fwd_TF": encprune_f / TF,
        "predictor_fwd_TF": pred_f / TF,
        "probe_fwd_TF": probe_f / TF,
        "predictor_probe_combined_fwd_TF": pred_probe_fwd_f / TF,
        "predictor_probe_trainstep_TF": step_f / TF,
        "total_fwd_TF (encode+prune+predictor+probe)": (encprune_f + pred_probe_fwd_f) / TF,
        "wall_step_sec": wall_step,
        "encoder_breakdown_TF": enc_bd,
        "predictor_breakdown_TF": pred_bd,
        "probe_breakdown_TF": probe_bd,
        "predictor_probe_step_breakdown_TF": step_bd,
    }
    _write(tag, result)
    return tag, result


# ===========================================================================
# Qwen profiling
# ===========================================================================

def _build_qwen_model(n_frames: int, keep_ratio: float, dtype=torch.bfloat16):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import app.hdepic_lora_action_anticipation.train_vlm_probe_lora as train_mod
    from app.hdepic_lora_action_anticipation.train_vlm_probe_lora import (
        DEFAULT_MODEL_IDS, VLMProbe, apply_lora_to_llm,
    )
    from app.hdepic_lora_action_anticipation.qwen_token_pruning import install_qwen_video_token_pruner

    model_id = DEFAULT_MODEL_IDS["qwen25vl"]
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    train_mod._QWEN_FRAME_SIZE = 256

    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, local_files_only=True,
    ).to("cuda")
    hidden_size = (
        getattr(backbone.config, "hidden_size", None)
        or backbone.config.text_config.hidden_size
    )
    for p in backbone.parameters():
        p.requires_grad = False
    apply_lora_to_llm(backbone, rank=16, alpha=32.0)

    visual_module = (
        dict(backbone.named_modules()).get("visual")
        or dict(backbone.named_modules()).get("model.visual")
    )
    orig_visual_fwd = visual_module.forward

    @torch.no_grad()
    def visual_no_grad(*a, **kw):
        return orig_visual_fwd(*a, **kw)
    visual_module.forward = visual_no_grad

    backbone.gradient_checkpointing_enable()
    backbone.config.use_cache = False

    if keep_ratio < 1.0:
        install_qwen_video_token_pruner(backbone, keep_ratio=keep_ratio, chunk_size=0)

    model = VLMProbe(backbone, hidden_size, 106, 303, 1680).to("cuda")
    return processor, model, visual_module, orig_visual_fwd


def profile_qwen(n_frames: int, keep_ratio: float, warmup: int = 1):
    from app.hdepic_lora_action_anticipation.train_vlm_probe_lora import build_qwen_inputs_batch

    tag = f"qwen_rerun_nf{n_frames}_kr{str(keep_ratio).replace('.', 'p')}"
    print(f"\n{'='*64}", flush=True)
    print(f"[profile] Qwen  nf={n_frames}  keep_ratio={keep_ratio}", flush=True)

    dtype = torch.bfloat16
    processor, model, visual_module, orig_visual_fwd = _build_qwen_model(n_frames, keep_ratio, dtype)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    frames = np.zeros((n_frames, 256, 256, 3), dtype=np.uint8)
    inputs_cpu = build_qwen_inputs_batch(processor, [frames])
    inputs_gpu = {k: v.to("cuda") for k, v in inputs_cpu.items()}

    # Token count info
    grid_thw_key = "video_grid_thw" if "video_grid_thw" in inputs_cpu else "image_grid_thw"
    grid_thw = inputs_cpu[grid_thw_key]
    t, h, w = grid_thw[0].tolist()
    n_merged_pre_prune = int(t * h * w)
    n_merged_post_prune = int(n_merged_pre_prune * keep_ratio) if keep_ratio < 1.0 else n_merged_pre_prune
    print(f"  grid_thw={grid_thw[0].tolist()}  merged_tokens={n_merged_pre_prune}  "
          f"after_prune={n_merged_post_prune}", flush=True)

    labels = {k: torch.tensor([0], device="cuda") for k in ["verb", "noun", "action"]}

    def full_fwd():
        with torch.no_grad():
            model(**inputs_gpu)

    def full_step():
        opt.zero_grad(set_to_none=True)
        v, n, a = model(**inputs_gpu)
        loss = (criterion(v, labels["verb"])
                + criterion(n, labels["noun"])
                + criterion(a, labels["action"]))
        loss.backward()
        opt.step()

    # Warmup
    print("  warmup …", flush=True)
    for _ in range(warmup):
        full_step()
    torch.cuda.synchronize()

    # ---- (A) Vision tower only — capture args via pre-hook, replay under FlopCounterMode ----
    _vis_args, _vis_kwargs = [], []

    def _capture_hook(mod, args, kwargs):
        if not _vis_args:
            _vis_args.append(args)
            _vis_kwargs.append(dict(kwargs))

    h_cap = visual_module.register_forward_pre_hook(_capture_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(**inputs_gpu)
    finally:
        h_cap.remove()

    captured_args = _vis_args[0]
    captured_kwargs = _vis_kwargs[0]

    # Restore original (unwrapped) forward for profiling
    visual_module.forward = orig_visual_fwd

    def vis_fwd():
        with torch.no_grad():
            orig_visual_fwd(*captured_args, **captured_kwargs)

    print("  [A] vision tower only …", flush=True)
    torch.cuda.empty_cache()
    vis_f, vis_bd = _count(vis_fwd)
    print(f"      vision_tower fwd = {vis_f/TF:.3f} TF", flush=True)

    # Re-wrap with no_grad
    @torch.no_grad()
    def visual_no_grad(*a, **kw):
        return orig_visual_fwd(*a, **kw)
    visual_module.forward = visual_no_grad

    # ---- (B) Full forward ----
    print("  [B] full model forward …", flush=True)
    torch.cuda.empty_cache()
    fwd_f, fwd_bd = _count(full_fwd)
    print(f"      total fwd = {fwd_f/TF:.3f} TF", flush=True)

    # ---- (C) Full train step ----
    print("  [C] full train step (fwd+bwd) …", flush=True)
    torch.cuda.empty_cache()
    step_f, step_bd = _count(full_step)
    print(f"      total trainstep = {step_f/TF:.3f} TF", flush=True)

    # Wall-time + peak memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    full_step()
    torch.cuda.synchronize()
    wall_step = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated()

    llm_fwd_f = fwd_f - vis_f
    # ViT is frozen/no_grad → no backward for ViT; backward is LLM only
    # train_step = ViT_fwd + LLM_fwd + LLM_bwd(incl GC recompute)
    llm_step_f = step_f - vis_f

    result = {
        "config": {
            "n_frames": n_frames,
            "keep_ratio": keep_ratio,
            "n_merged_tokens_pre_prune": n_merged_pre_prune,
            "n_merged_tokens_post_prune": n_merged_post_prune,
        },
        "vision_tower_fwd_TF": vis_f / TF,
        "llm_fwd_TF": llm_fwd_f / TF,
        "total_fwd_TF": fwd_f / TF,
        "llm_trainstep_TF": llm_step_f / TF,
        "total_trainstep_TF": step_f / TF,
        "wall_step_sec": wall_step,
        "peak_mem_GB": peak_mem / 1e9,
        "vision_tower_op_breakdown_TF": vis_bd,
        "fwd_op_breakdown_TF": fwd_bd,
        "step_op_breakdown_TF": step_bd,
    }
    _write(tag, result)
    return tag, result


# ===========================================================================
# Summary table
# ===========================================================================

def print_and_save_summary(vjepa_results: dict, qwen_results: dict):
    summary = {"vjepa": vjepa_results, "qwen": qwen_results}
    _write("module_breakdown_summary", summary)

    print("\n" + "="*100, flush=True)
    print("B12 FLOPs BREAKDOWN — per-module, all methods re-measured", flush=True)
    print("="*100, flush=True)

    # V-JEPA2 table
    print("\n--- V-JEPA2 (encoder | predictor | probe split) ---", flush=True)
    hdr = (f"{'Config':<45} {'enc_fwd':>10} {'pred_fwd':>9} {'probe_fwd':>10} "
           f"{'pred+probe_step':>16} {'total_fwd':>11}")
    print(hdr, flush=True)
    print("-"*105, flush=True)
    for tag, r in vjepa_results.items():
        cfg = r.get("config", {})
        lbl = f"nf{cfg.get('num_frames')} keep{cfg.get('keep_count')} {cfg.get('position_mode')}"
        ef = f"{r.get('encoder_fwd_TF', 0):.2f}"
        prf = f"{r.get('predictor_fwd_TF', 0):.4f}"
        pbf = f"{r.get('probe_fwd_TF', 0):.5f}"
        cs = f"{r.get('predictor_probe_trainstep_TF', 0):.4f}"
        tf = f"{r.get('total_fwd_TF (encode+prune+predictor+probe)', 0):.2f}"
        print(f"  {lbl:<43} {ef:>10} {prf:>9} {pbf:>10} {cs:>16} {tf:>11}", flush=True)

    # Qwen table
    print("\n--- Qwen2.5-VL-3B (vision tower | LLM split) ---", flush=True)
    hdr = (f"{'Config':<35} {'tokens(post)':>13} {'vis_fwd':>9} {'llm_fwd':>9} "
           f"{'total_fwd':>10} {'llm_step':>10} {'total_step':>11} {'peak_GB':>8}")
    print(hdr, flush=True)
    print("-"*110, flush=True)
    for tag, r in qwen_results.items():
        cfg = r.get("config", {})
        lbl = f"nf{cfg.get('n_frames')} kr{cfg.get('keep_ratio')}"
        mt = str(cfg.get("n_merged_tokens_post_prune", "?"))
        vt = f"{r.get('vision_tower_fwd_TF', 0):.3f}"
        lf = f"{r.get('llm_fwd_TF', 0):.3f}"
        tf = f"{r.get('total_fwd_TF', 0):.3f}"
        ls = f"{r.get('llm_trainstep_TF', 0):.3f}"
        ts = f"{r.get('total_trainstep_TF', 0):.3f}"
        pm = f"{r.get('peak_mem_GB', 0):.1f}"
        print(f"  {lbl:<33} {mt:>13} {vt:>9} {lf:>9} {tf:>10} {ls:>10} {ts:>11} {pm:>8}", flush=True)

    print("\n(all TF = TeraFLOPs/sample; FlopCounterMode analytic counts; batch=1)", flush=True)
    print("="*100, flush=True)


# ===========================================================================
# Main
# ===========================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--skip-vjepa", action="store_true")
    ap.add_argument("--skip-qwen", action="store_true")
    ap.add_argument("--vjepa-configs", default="all",
                    help=("comma-separated: 4s,1min_prune_rebase,1min_prune_true,1min_fullctx,"
                          "1min_mid_nextattn,1min_mid_featurenorm,7s_fullctx,7s_mid_nextattn"))
    ap.add_argument("--qwen-configs", default="all",
                    help="comma-separated: 32f_kr1p0,32f_kr0p15,480f_kr1p0,480f_kr0p0667,56f_kr1p0")
    a = ap.parse_args()

    torch.manual_seed(0)
    vjepa_results, qwen_results = {}, {}

    # V-JEPA2
    if not a.skip_vjepa:
        vjepa_all = {
            "4s":                  (32,  4096,  "rebased", "", ""),
            "1min_prune_rebase":   (480, 4096,  "rebased", "", ""),
            "1min_prune_true":     (480, 4096,  "true", "", ""),
            "1min_fullctx":        (480, 61440, "true", "", ""),
            "1min_mid_nextattn":   (480, 4096,  "true", "9:0.5,17:4096", "next_attn"),
            "1min_mid_featurenorm":(480, 4096,  "true", "8:0.5,16:4096", "feature_norm"),
            "7s_fullctx":          (56,  7168,  "true", "", ""),
            "7s_mid_nextattn":     (56,  4096,  "true", "9:0.5,17:4096", "next_attn"),
        }
        wanted = set(vjepa_all.keys()) if a.vjepa_configs == "all" \
                 else set(a.vjepa_configs.split(","))
        for key, (nf, kc, pm, eps, epm) in vjepa_all.items():
            if key in wanted:
                tag, result = profile_vjepa(
                    nf,
                    kc,
                    pm,
                    warmup=a.warmup,
                    encoder_prune_schedule=eps,
                    encoder_prune_metric=epm,
                )
                vjepa_results[tag] = result
                torch.cuda.empty_cache()

    # Qwen
    if not a.skip_qwen:
        qwen_all = {
            "32f_kr1p0":   (32,  1.0),
            "32f_kr0p15":  (32,  0.15),
            "480f_kr1p0":  (480, 1.0),
            "480f_kr0p0667":(480, 0.0667),
            "56f_kr1p0":   (56,  1.0),
        }
        wanted = set(qwen_all.keys()) if a.qwen_configs == "all" \
                 else set(a.qwen_configs.split(","))
        for key, (nf, kr) in qwen_all.items():
            if key in wanted:
                tag, result = profile_qwen(nf, kr, warmup=a.warmup)
                qwen_results[tag] = result
                torch.cuda.empty_cache()

    print_and_save_summary(vjepa_results, qwen_results)


if __name__ == "__main__":
    main()
