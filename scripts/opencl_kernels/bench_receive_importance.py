#!/usr/bin/env python3
"""OpenCL bench for JEPA postfuse attention-receive importance.

Mirrors:
  JEPA_ARVR/app/.../train_stream_mtp_concat_ca.py :: _postfuse_attn_importance

Usage (login / POCL CPU):
  /scratch/ll5914/conda_envs/jepa_opencl/bin/python \\
    scripts/opencl_kernels/bench_receive_importance.py --N 512 --D 64

Optional Torch reference (SVD env has torch):
  /scratch/ll5914/conda_envs/SVD/bin/python \\
    scripts/opencl_kernels/bench_receive_importance.py --N 512 --backend numpy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import pyopencl as cl
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"pyopencl required: {e}") from e

CL_PATH = Path(__file__).with_name("receive_importance.cl")


def ref_receive_importance(x: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    """NumPy reference. x: [B,N,D] float32 (raw tokens)."""
    B, N, D = x.shape
    # L2 normalize
    norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)
    xn = x / norms
    scale = D**-0.5
    imp = np.zeros((B, N), dtype=np.float32)
    for ci in range(0, N, chunk_size):
        q = xn[:, ci : ci + chunk_size]  # [B,C,D]
        logits = np.matmul(q, xn.transpose(0, 2, 1)) * scale  # [B,C,N]
        # softmax last dim
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp(logits - m)
        attn = e / e.sum(axis=-1, keepdims=True)
        imp += attn.sum(axis=1)
    return imp


def torch_receive_importance(x: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    video = torch.from_numpy(x)
    B, N, D = video.shape
    xf = F.normalize(video.float(), dim=-1)
    scale = D**-0.5
    imp = torch.zeros(B, N, dtype=torch.float32)
    for ci in range(0, N, chunk_size):
        q = xf[:, ci : ci + chunk_size]
        logits = torch.bmm(q, xf.transpose(1, 2)) * scale
        imp += logits.softmax(dim=-1).sum(dim=1)
    return imp.numpy()


class OpenCLReceiveImportance:
    def __init__(self, platform_idx: int = 0, device_idx: int = 0):
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms")
        platform = platforms[platform_idx]
        devices = platform.get_devices()
        if not devices:
            raise RuntimeError(f"No devices on platform {platform.name}")
        self.device = devices[device_idx]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(
            self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
        )
        src = CL_PATH.read_text()
        self.prg = cl.Program(self.ctx, src).build(options="-cl-std=CL1.2")
        self.k_norm = cl.Kernel(self.prg, "l2_normalize_rows")
        self.k_qk = cl.Kernel(self.prg, "qk_logits_chunk")
        self.k_sm = cl.Kernel(self.prg, "softmax_rows")
        self.k_red = cl.Kernel(self.prg, "reduce_chunk_to_imp")
        print(f"OpenCL: {platform.name} | {self.device.name}")

    def run(self, x: np.ndarray, chunk_size: int = 256) -> tuple[np.ndarray, dict]:
        """x [B,N,D] float32 -> imp [B,N], timings ms."""
        assert x.dtype == np.float32 and x.ndim == 3
        B, N, D = x.shape
        scale = np.float32(D**-0.5)
        mf = cl.mem_flags

        buf_in = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        buf_x = cl.Buffer(self.ctx, mf.READ_WRITE, size=x.nbytes)
        buf_imp = cl.Buffer(self.ctx, mf.READ_WRITE, size=B * N * 4)

        timings = {"normalize_ms": 0.0, "qk_ms": 0.0, "softmax_ms": 0.0, "reduce_ms": 0.0}

        # normalize
        e = self.k_norm(
            self.queue, (B * N,), None, buf_in, buf_x, np.int32(B * N), np.int32(D)
        )
        e.wait()
        timings["normalize_ms"] += 1e-6 * (e.profile.end - e.profile.start)

        cl.enqueue_fill_buffer(self.queue, buf_imp, np.float32(0), 0, B * N * 4).wait()

        # reuse max chunk buffers
        max_C = min(chunk_size, N)
        logits_host_bytes = B * max_C * N * 4
        buf_logits = cl.Buffer(self.ctx, mf.READ_WRITE, size=logits_host_bytes)
        buf_attn = cl.Buffer(self.ctx, mf.READ_WRITE, size=logits_host_bytes)

        for q_start in range(0, N, chunk_size):
            C = min(chunk_size, N - q_start)
            total = B * C * N

            e1 = self.k_qk(
                self.queue,
                (total,),
                None,
                buf_x,
                buf_logits,
                np.int32(B),
                np.int32(N),
                np.int32(D),
                np.int32(q_start),
                np.int32(C),
                scale,
            )
            e1.wait()
            timings["qk_ms"] += 1e-6 * (e1.profile.end - e1.profile.start)

            e2 = self.k_sm(
                self.queue,
                (B * C,),
                None,
                buf_logits,
                buf_attn,
                np.int32(B * C),
                np.int32(N),
            )
            e2.wait()
            timings["softmax_ms"] += 1e-6 * (e2.profile.end - e2.profile.start)

            e3 = self.k_red(
                self.queue,
                (B * N,),
                None,
                buf_attn,
                buf_imp,
                np.int32(B),
                np.int32(N),
                np.int32(C),
            )
            e3.wait()
            timings["reduce_ms"] += 1e-6 * (e3.profile.end - e3.profile.start)

        imp = np.empty((B, N), dtype=np.float32)
        cl.enqueue_copy(self.queue, imp, buf_imp).wait()
        timings["total_ms"] = sum(timings.values())
        return imp, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--N", type=int, default=512, help="num tokens (try 512/1024/2048)")
    ap.add_argument("--D", type=int, default=64, help="embed dim (ViT head uses smaller; full D=1024 later)")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--platform", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--check-torch", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.B, args.N, args.D), dtype=np.float32)

    t0 = time.perf_counter()
    ref = ref_receive_importance(x, chunk_size=args.chunk)
    ref_ms = (time.perf_counter() - t0) * 1e3
    print(f"NumPy ref: shape={ref.shape} sum={ref.sum():.4f}  wall={ref_ms:.2f} ms")

    if args.check_torch:
        t0 = time.perf_counter()
        tref = torch_receive_importance(x, chunk_size=args.chunk)
        print(
            f"Torch ref: max_abs_diff_vs_numpy={np.max(np.abs(tref - ref)):.3e}  "
            f"wall={(time.perf_counter() - t0) * 1e3:.2f} ms"
        )

    ocl = OpenCLReceiveImportance(args.platform, args.device)
    for _ in range(args.warmup):
        ocl.run(x, chunk_size=args.chunk)

    best = None
    for i in range(args.iters):
        out, timings = ocl.run(x, chunk_size=args.chunk)
        diff = np.max(np.abs(out - ref))
        print(
            f"iter{i}: max_abs_diff={diff:.3e}  "
            f"ocl_profile_total={timings['total_ms']:.2f} ms  "
            f"(norm={timings['normalize_ms']:.2f} qk={timings['qk_ms']:.2f} "
            f"sm={timings['softmax_ms']:.2f} red={timings['reduce_ms']:.2f})"
        )
        if best is None or timings["total_ms"] < best["total_ms"]:
            best = timings

    rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-6)
    ok = np.max(np.abs(out - ref)) < 1e-4 * max(1.0, np.max(np.abs(ref)))
    print(f"best_ocl_total={best['total_ms']:.2f} ms  numpy_wall={ref_ms:.2f} ms  PASS={ok} rel={rel:.3e}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
