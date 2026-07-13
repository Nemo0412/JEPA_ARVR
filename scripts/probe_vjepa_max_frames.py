#!/usr/bin/env python
"""Find the max number of frames the V-JEPA2 ViT-L encoder can take in one forward.

No temporal downsampling -- we literally feed T frames (T even, tubelet_size=2) at 256px and
run the FROZEN encoder forward under no_grad (exactly how it is used in the probe/LoRA pipeline,
where the backbone never gets gradients). The encoder uses RoPE (positions built on the fly,
no learned pos-embed table) so there is no architectural frame cap; the limit is GPU memory and
N^2 attention compute. SDPA's memory-efficient backend keeps activation memory ~O(N), so the
ceiling is high -- this sweep finds where it actually breaks (OOM) or gets too slow on THIS GPU.

Reports, per frame count: token count, peak CUDA memory, forward wall-time, pass/fail.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _gpu_name() -> str:
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def _gpu_total_gb() -> float:
    try:
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        return float("nan")


def build_encoder(dtype: torch.dtype):
    from evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar import init_module

    model_kwargs = {
        "encoder": {
            "model_name": "vit_large",
            "checkpoint_key": "target_encoder",
            "tubelet_size": 2,
            "patch_size": 16,
            "uniform_power": True,
            "use_rope": True,
        },
        "predictor": {
            "model_name": "vit_predictor",
            "checkpoint_key": "predictor",
            "num_frames": 64,
            "depth": 12,
            "num_heads": 12,
            "predictor_embed_dim": 384,
            "num_mask_tokens": 10,
            "uniform_power": True,
            "use_mask_tokens": True,
            "use_sdpa": True,
            "use_silu": False,
            "wide_silu": False,
            "use_rope": True,
        },
    }
    wrapper_kwargs = {"no_predictor": False, "num_output_frames": 2, "num_steps": 1}
    backbone = init_module(
        frames_per_clip=32,
        frames_per_second=8,
        resolution=256,
        checkpoint="/scratch/yh6416/VJEPA2-EXP/checkpoints/vitl.pt",
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )
    encoder = backbone.encoder.to("cuda").to(dtype)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def sweep(dtype: torch.dtype, frame_grid, res: int, slow_limit_sec: float) -> list[dict]:
    encoder = build_encoder(dtype)
    results = []
    for T in frame_grid:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tokens = (T // 2) * (res // 16) * (res // 16)
        rec = {"frames": T, "tokens": tokens, "seconds_at_8fps": T / 8.0}
        try:
            x = torch.randn(1, 3, T, res, res, device="cuda", dtype=dtype)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = encoder(x)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rec.update(
                {
                    "ok": True,
                    "out_shape": list(out.shape),
                    "fwd_sec": round(dt, 3),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                }
            )
            del x, out
            print(f"[{dtype}] frames={T:5d} tokens={tokens:7d} OK  "
                  f"mem={rec['peak_mem_gb']:.2f}GB  fwd={dt:.2f}s  out={rec['out_shape']}", flush=True)
            if dt > slow_limit_sec:
                rec["stop_reason"] = "too_slow"
                results.append(rec)
                print(f"[{dtype}] forward exceeded {slow_limit_sec}s -> stopping sweep (compute-bound).", flush=True)
                break
        except torch.cuda.OutOfMemoryError as e:
            rec.update({"ok": False, "error": "OOM", "detail": str(e)[:200]})
            print(f"[{dtype}] frames={T:5d} tokens={tokens:7d} OOM", flush=True)
            results.append(rec)
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            msg = str(e)
            is_oom = "out of memory" in msg.lower()
            rec.update({"ok": False, "error": "OOM" if is_oom else "RuntimeError", "detail": msg[:200]})
            print(f"[{dtype}] frames={T:5d} tokens={tokens:7d} {'OOM' if is_oom else 'ERR'}: {msg[:120]}", flush=True)
            results.append(rec)
            torch.cuda.empty_cache()
            if is_oom:
                break
            else:
                continue
        results.append(rec)
    del encoder
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/scratch/yh6416/VJEPA2-EXP/outputs/b12_flops_profile")
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--slow-limit-sec", type=float, default=180.0)
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument(
        "--frames",
        default="32,64,128,256,384,512,768,1024,1536,2048,3072,4096",
    )
    args = parser.parse_args()

    frame_grid = [int(t) for t in args.frames.split(",")]
    assert all(t % 2 == 0 for t in frame_grid), "tubelet_size=2 requires even frame counts"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "gpu": _gpu_name(),
        "gpu_total_mem_gb": round(_gpu_total_gb(), 1),
        "resolution": args.res,
        "tubelet_size": 2,
        "tokens_per_frame": (args.res // 16) * (args.res // 16) // 2,
        "note": "frozen encoder, no_grad, no downsampling; RoPE => no architectural frame cap; SDPA mem-efficient backend",
        "runs": {},
    }
    for name in args.dtypes.split(","):
        dtype = dtype_map[name.strip()]
        print(f"\n===== sweep dtype={name} =====", flush=True)
        runs = sweep(dtype, frame_grid, args.res, args.slow_limit_sec)
        summary["runs"][name] = runs
        ok = [r for r in runs if r.get("ok")]
        summary[f"max_ok_frames_{name}"] = max((r["frames"] for r in ok), default=None)

    path = out_dir / "vjepa2_encoder_max_frames.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2), flush=True)
    print(f"[probe] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
