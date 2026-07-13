#!/usr/bin/env python
"""Does the V-JEPA2 predictor have a hard input-token limit? Prove it by experiment.

Unlike the encoder (RoPE, no cap -- see probe_vjepa_max_frames.py), the predictor builds a
`pred_tokens` buffer of size `self.num_patches` (fixed at init from `num_frames=64` =>
32*16*16 = 8192) and then `apply_masks(pred_tokens, masks_y)` = `torch.gather` at the TARGET
position indices. The anticipative wrapper sets target indices ~ N + 1279 (N = (T/2)*256 ctxt
tokens, +256*anticipation_steps skip, +256 N_pred). So once N+1279 >= num_patches the gather
indexes out of range and crashes -> a hard cap at ~T=54 frames (with 1s anticipation).

We test two modes:
  - stock : num_patches left at 8192          -> expect a hard break ~T=54.
  - lifted: num_patches raised to fit the indices -> expect it to scale like the encoder
            (compute/memory-bound only), proving the cap is a config buffer, not architecture.

A CUDA out-of-range gather is an async device-side assert (unrecoverable in-process), so the
orchestrator runs each frame count in its OWN subprocess (`--single`). Random weights: the cap
is purely structural (index vs buffer size), independent of trained values, and this keeps each
worker fast (no 5GB checkpoint load).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---- config matching the real V-JEPA2 ViT-L predictor (see profile_b12_flops.py) ----
RES = 256
GRID = RES // 16          # 16
TUBELET = 2
ENC_EMBED = 1024
ANTICIPATION_SEC = 1.0
FPS = 8
NUM_OUTPUT_FRAMES = 2
STOCK_NUM_PATCHES = (64 // TUBELET) * GRID * GRID   # 8192


def _positions(T: int):
    import torch
    N = (T // TUBELET) * GRID * GRID
    ctxt = torch.arange(N)
    steps = int(ANTICIPATION_SEC * FPS / TUBELET)          # 4
    skip = N + GRID * GRID * steps                          # N + 1024
    n_pred = GRID * GRID * (NUM_OUTPUT_FRAMES // TUBELET)   # 256
    tgt = torch.arange(n_pred) + skip
    return N, ctxt, tgt, int(tgt.max().item())


def _build_predictor(num_patches_override=None):
    import torch
    from src.models.predictor import vit_predictor

    pred = vit_predictor(
        img_size=(RES, RES),
        patch_size=16,
        num_frames=64,
        tubelet_size=TUBELET,
        embed_dim=ENC_EMBED,
        predictor_embed_dim=384,
        depth=12,
        num_heads=12,
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=10,
        use_rope=True,
    ).to("cuda")
    pred.eval()
    if num_patches_override is not None:
        pred.num_patches = int(num_patches_override)
    return pred


def run_single(T: int, mode: str) -> dict:
    import torch

    N, ctxt, tgt, max_idx = _positions(T)
    rec = {"frames": T, "ctxt_tokens": N, "max_target_index": max_idx,
           "stock_num_patches": STOCK_NUM_PATCHES, "seconds_at_8fps": T / FPS}
    num_patches = None if mode == "stock" else max_idx + 1
    pred = _build_predictor(num_patches_override=num_patches)
    rec["num_patches_used"] = int(pred.num_patches)

    x = torch.randn(1, N, ENC_EMBED, device="cuda")
    masks_x = ctxt.unsqueeze(0).to("cuda")
    masks_y = tgt.unsqueeze(0).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = pred(x, masks_x=masks_x, masks_y=masks_y)
    torch.cuda.synchronize()
    rec.update({
        "ok": True,
        "out_shape": list(out.shape),
        "fwd_sec": round(time.perf_counter() - t0, 3),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    })
    return rec


def orchestrate(out_dir: Path, slow_limit_sec: float) -> None:
    import torch
    sweeps = {
        "stock": [32, 48, 54, 56, 64, 96, 128],
        "lifted": [56, 64, 128, 256, 512, 1024, 2048],
    }
    summary = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": {"res": RES, "anticipation_sec": ANTICIPATION_SEC, "fps": FPS,
                   "num_output_frames": NUM_OUTPUT_FRAMES, "stock_num_patches": STOCK_NUM_PATCHES,
                   "predicted_stock_max_frames": "max_target_index < num_patches => N+1279 < 8192 => T<=54"},
        "runs": {},
    }
    for mode, grid in sweeps.items():
        print(f"\n===== predictor sweep mode={mode} =====", flush=True)
        runs = []
        for T in grid:
            proc = subprocess.run(
                [sys.executable, __file__, "--single", str(T), "--mode", mode],
                capture_output=True, text=True,
            )
            line = next((l for l in proc.stdout.splitlines() if l.startswith("{")), None)
            if proc.returncode == 0 and line:
                rec = json.loads(line)
                print(f"[{mode}] frames={T:5d} ctxt={rec['ctxt_tokens']:7d} max_idx={rec['max_target_index']:7d} "
                      f"num_patches={rec['num_patches_used']:7d} OK mem={rec.get('peak_mem_gb')}GB fwd={rec.get('fwd_sec')}s",
                      flush=True)
            else:
                err_tail = " | ".join(proc.stderr.strip().splitlines()[-3:])[:300]
                rec = {"frames": T, "ok": False, "returncode": proc.returncode, "error_tail": err_tail}
                N, _, _, max_idx = _positions(T)
                rec.update({"ctxt_tokens": N, "max_target_index": max_idx})
                print(f"[{mode}] frames={T:5d} ctxt={N:7d} max_idx={max_idx:7d} FAILED rc={proc.returncode}: {err_tail[:160]}",
                      flush=True)
            runs.append(rec)
            ok = rec.get("ok")
            if mode == "stock" and not ok:
                print(f"[{mode}] first failure at T={T} -> hard cap is just below this. Stopping stock sweep.", flush=True)
                break
            if ok and rec.get("fwd_sec", 0) > slow_limit_sec:
                rec["stop_reason"] = "too_slow"
                print(f"[{mode}] forward exceeded {slow_limit_sec}s -> stopping (compute-bound).", flush=True)
                break
        summary["runs"][mode] = runs
        ok_frames = [r["frames"] for r in runs if r.get("ok")]
        summary[f"max_ok_frames_{mode}"] = max(ok_frames, default=None)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "vjepa2_predictor_max_tokens.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2), flush=True)
    print(f"[probe] wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/scratch/yh6416/VJEPA2-EXP/outputs/vjepa_max_frames")
    parser.add_argument("--slow-limit-sec", type=float, default=120.0)
    parser.add_argument("--single", type=int, default=None, help="worker: run one frame count")
    parser.add_argument("--mode", choices=["stock", "lifted"], default="stock")
    args = parser.parse_args()

    if args.single is not None:
        rec = run_single(args.single, args.mode)
        print(json.dumps(rec), flush=True)
        return
    orchestrate(Path(args.out_dir), args.slow_limit_sec)


if __name__ == "__main__":
    main()
