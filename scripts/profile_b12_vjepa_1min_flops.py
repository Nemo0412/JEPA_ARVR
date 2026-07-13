#!/usr/bin/env python
"""Real per-sample FLOPs (FlopCounterMode) for the 1-min V-JEPA2 method.

Measures the ACTUAL pipeline of train_vjepa_prune_anticipation.py:
  (A) encode+prune  : encoder(480 frames -> 61440 tokens) + attention-importance pruning to 4096.
                      Forward-only, no grad (= the one-time, cached, frozen-encoder cost per sample).
  (A0) encoder-only : same encoder forward WITHOUT the pruning-importance overhead (pruner detached).
  (B) predictor+probe train-step : on cached 4096 tokens, fwd+bwd through predictor-LoRA + probe
                      (= the per-optimizer-step training cost, repeated every epoch).
  (B0) predictor+probe forward-only.

FlopCounterMode has correct analytic SDPA fwd/bwd formulas, so these are "real" FLOPs (not the
undercounted torch.profiler(with_flops) number). bf16 autocast (matches training).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/path/to/VJEPA2-EXP")
import app.hdepic_lora_action_anticipation.train_vjepa_prune_anticipation as M  # noqa: E402


def count(fn):
    c = FlopCounterMode(display=False)
    with c:
        fn()
    torch.cuda.synchronize()
    g = c.get_flop_counts().get("Global", {})
    bd = {str(op): int(f) for op, f in sorted(g.items(), key=lambda kv: -kv[1])}
    return int(c.get_total_flops()), bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-frames", type=int, default=480)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument("--position-mode", choices=["rebased", "true"], default="rebased")
    ap.add_argument("--out-dir", default=f"{SHARED}/outputs/b12_flops_profile")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    model = M.build_model(dev, a.num_frames, 8, 256, f"{SHARED}/checkpoints/vitl.pt")
    gp = int(model.grid_size) ** 2
    pruner = M.TokenPruner(model.encoder, keep_count=a.keep_count, gp=gp, chunk_size=256)
    M.apply_lora_to_predictor(model.predictor, rank=8, alpha=16.0)
    model._position_mode = a.position_mode
    model._full_n = (a.num_frames // int(model.tubelet_size)) * gp
    if a.position_mode == "true":
        model.predictor.num_patches = ((model._full_n // gp) + 8) * gp
    probe = M.HDEpicProbe(model.embed_dim, 106, 300, 1700).to(dev)

    clips = torch.randn(1, 3, a.num_frames, 256, 256, device=dev)
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    # (A) encode + prune (forward only)
    def enc_prune():
        with torch.no_grad(), ac():
            M.encode_and_prune(model, pruner, clips)
    enc_prune(); torch.cuda.synchronize()
    f_encprune, bd_encprune = count(enc_prune)

    # (A0) encoder only (detach pruner monkey-patch)
    pruner.remove()
    def enc_only():
        with torch.no_grad(), ac():
            model.encoder(clips)
    enc_only(); torch.cuda.synchronize()
    f_enc, _ = count(enc_only)

    # (B) predictor + probe train-step on cached tokens
    ctx = torch.randn(1, a.keep_count, model.embed_dim, device=dev)
    idx = torch.arange(a.keep_count, device=dev).unsqueeze(0)
    lora = [p for p in model.predictor.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(list(probe.parameters()) + lora, lr=1e-4)
    crit = nn.CrossEntropyLoss()
    lab = torch.zeros(1, dtype=torch.long, device=dev)

    def pred_fwd():
        with ac():
            feats = M.anticipate_from_ctx(model, ctx, 1.0, ctx_idx=idx)
            probe(feats.float())

    def pred_step():
        opt.zero_grad(set_to_none=True)
        with ac():
            feats = M.anticipate_from_ctx(model, ctx, 1.0, ctx_idx=idx)
            v, n, act = probe(feats.float())
            loss = crit(v, lab) + crit(n, lab) + crit(act, lab)
        loss.backward(); opt.step()

    pred_step(); torch.cuda.synchronize()
    f_pred_fwd, _ = count(pred_fwd)
    f_pred_step, bd_pred = count(pred_step)

    TF = 1e12
    result = {
        "config": {"num_frames": a.num_frames, "keep_count": a.keep_count,
                   "n_tokens_full": model._full_n, "position_mode": a.position_mode,
                   "embed_dim": model.embed_dim, "gp": gp},
        "encode_only_fwd_TF": f_enc / TF,
        "encode_plus_prune_fwd_TF": f_encprune / TF,
        "prune_overhead_TF": (f_encprune - f_enc) / TF,
        "predictor_probe_fwd_TF": f_pred_fwd / TF,
        "predictor_probe_trainstep_TF": f_pred_step / TF,
        "per_sample_total_fwd_TF (encode+prune + predictor)": (f_encprune + f_pred_fwd) / TF,
        "encode_plus_prune_breakdown": {k: round(v / TF, 4) for k, v in list(bd_encprune.items())[:8]},
        "predictor_step_breakdown": {k: round(v / TF, 4) for k, v in list(bd_pred.items())[:8]},
    }
    tag = f"vjepa_1min_method_nf{a.num_frames}_keep{a.keep_count}_{a.position_mode}"
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"[profile] wrote {out / (tag + '.json')}", flush=True)


if __name__ == "__main__":
    main()
