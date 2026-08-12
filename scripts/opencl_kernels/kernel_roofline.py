"""Per-kernel FLOPs / Bytes (as implemented) for roofline:

  AI          = FLOPs / Bytes
  BW_eff      = Bytes / T          (reported as GB/s)
  Compute_eff = FLOPs / T          (reported as GFLOP/s)

Bytes count issued global load/store traffic in the OpenCL loops
(re-reads included). That is the right model for these naive kernels.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np
import pyopencl as cl

F32 = 4

PROBE_SRC = r"""
__kernel void memcpy_f32(__global const float *src, __global float *dst, const int n) {
    const int i = get_global_id(0);
    if (i < n) dst[i] = src[i];
}
__kernel void fma_peak(__global float *x, const int n, const int reps) {
    const int i = get_global_id(0);
    if (i >= n) return;
    float a = x[i];
    float b = 1.000000119f;
    float c = 0.000000119f;
    for (int r = 0; r < reps; ++r) {
        a = fma(a, b, c);
        a = fma(a, b, c);
        a = fma(a, b, c);
        a = fma(a, b, c);
    }
    x[i] = a;
}
"""


def _fma2(n: int) -> int:
    """n fused multiply-adds counted as 2 FLOPs each."""
    return 2 * int(n)


def kernel_cost(name: str, **kw: Any) -> tuple[int, int]:
    """Return (FLOPs, Bytes) for one launch of `name`."""
    fn = _COSTS.get(name)
    if fn is None:
        # generic D2D copy: pass nbytes=
        if name.startswith("d2d_") or name in ("memcpy_f32", "empty_launch"):
            return _COSTS[name](**kw) if name in _COSTS else (0, int(kw.get("nbytes", 0)) * 2)
        return 0, 0
    return fn(**kw)


def layernorm_rows(rows: int, D: int, **_) -> tuple[int, int]:
    # mean D add + div; var 3D + div; rsqrt; affine 4D
    flops = int(rows) * (8 * int(D) + 3)
    bytes_ = int(rows) * int(D) * F32 * 6  # x*3 + y + gamma + beta
    return flops, bytes_


def gemm_nn(M: int, N: int, K: int, beta: float = 0.0, **_) -> tuple[int, int]:
    mn = int(M) * int(N)
    k = int(K)
    extra = 2 if abs(float(beta)) > 0 else 1  # alpha, optional beta
    flops = mn * (2 * k + extra)
    bytes_ = mn * (2 * k + 1 + (1 if abs(float(beta)) > 0 else 0)) * F32
    return flops, bytes_


def rope_apply(rows: int, Dh: int, **_) -> tuple[int, int]:
    rows, Dh = int(rows), int(Dh)
    dh2 = Dh // 2
    # per i: div, pow, mul, cos, sin ≈ 5 specials; mix 3*Dh
    flops = rows * (5 * dh2 + 3 * Dh)
    bytes_ = rows * (1 + 3 * Dh) * F32  # pos + x read*2 + x write
    return flops, bytes_


def sdpa_rows(B: int, H: int, N: int, Dh: int, **_) -> tuple[int, int]:
    q = int(B) * int(H) * int(N)
    N, Dh = int(N), int(Dh)
    # two QK dots (2*2N Dh) + AV (2 N Dh) + scale/max/exp/add ~5N + Dh scale
    flops = q * (6 * N * Dh + 5 * N + Dh + 1)
    bytes_ = q * (6 * N + 2) * Dh * F32
    return flops, bytes_


def sdpa_cross_rows(B: int, H: int, Nq: int, Nk: int, Dh: int, **_) -> tuple[int, int]:
    q = int(B) * int(H) * int(Nq)
    Nk, Dh = int(Nk), int(Dh)
    flops = q * (6 * Nk * Dh + 5 * Nk + Dh + 1)
    bytes_ = q * (6 * Nk + 2) * Dh * F32
    return flops, bytes_


def gelu(n: int, **_) -> tuple[int, int]:
    n = int(n)
    return n * 10, n * 2 * F32


def gated_residual(n: int, **_) -> tuple[int, int]:
    n = int(n)
    return n * 5, n * 4 * F32  # z, delta, gate, out


def l2_normalize_rows(rows: int, D: int, **_) -> tuple[int, int]:
    rows, D = int(rows), int(D)
    flops = rows * (2 * D + 1 + D)
    bytes_ = rows * D * F32 * 3
    return flops, bytes_


def qk_logits_chunk(B: int, N: int, D: int, C: int, **_) -> tuple[int, int]:
    n = int(B) * int(C) * int(N)
    D = int(D)
    flops = n * (2 * D + 1)
    bytes_ = n * (2 * D + 1) * F32
    return flops, bytes_


def softmax_rows(rows: int, N: int, **_) -> tuple[int, int]:
    rows, N = int(rows), int(N)
    flops = rows * (5 * N)
    bytes_ = rows * N * F32 * 4
    return flops, bytes_


def reduce_chunk_to_imp(B: int, N: int, C: int, **_) -> tuple[int, int]:
    bn, C = int(B) * int(N), int(C)
    flops = bn * C
    bytes_ = bn * (C + 2) * F32
    return flops, bytes_


def fill_recency(N: int, **_) -> tuple[int, int]:
    N = int(N)
    return N * 2, N * F32


def recency_reweight(B: int, N: int, **_) -> tuple[int, int]:
    bn = int(B) * int(N)
    return bn * 3, bn * 2 * F32 + int(N) * F32


def topk_indices_row(B: int, N: int, K: int, **_) -> tuple[int, int]:
    B, N, K = int(B), int(N), int(K)
    # for k in 0..K-1: scan N, each with up to k taken-checks
    flops = B * (K * N + N * K * (K - 1) // 2)
    bytes_ = B * (K * N * F32 + K * F32)
    return flops, bytes_


def residual_add(n: int, **_) -> tuple[int, int]:
    n = int(n)
    return n, n * 3 * F32  # a, b, out


def gather_tokens(B: int, K: int, D: int, **_) -> tuple[int, int]:
    B, K, D = int(B), int(K), int(D)
    flops = 0
    bytes_ = B * K * F32 + B * K * D * F32 * 2  # idx + in + out
    return flops, bytes_


def memcpy_f32(n: int | None = None, nbytes: int = 0, **_) -> tuple[int, int]:
    if n is None:
        n = int(nbytes) // F32
    n = int(n)
    return 0, n * 2 * F32


def empty_launch(**_) -> tuple[int, int]:
    return 0, 0


def d2d_copy(nbytes: int, **_) -> tuple[int, int]:
    return 0, int(nbytes) * 2


_COSTS = {
    "layernorm_rows": layernorm_rows,
    "gemm_nn": gemm_nn,
    "rope_apply": rope_apply,
    "sdpa_rows": sdpa_rows,
    "sdpa_cross_rows": sdpa_cross_rows,
    "gelu": gelu,
    "gated_residual": gated_residual,
    "l2_normalize_rows": l2_normalize_rows,
    "qk_logits_chunk": qk_logits_chunk,
    "softmax_rows": softmax_rows,
    "reduce_chunk_to_imp": reduce_chunk_to_imp,
    "fill_recency": fill_recency,
    "recency_reweight": recency_reweight,
    "topk_indices_row": topk_indices_row,
    "gather_tokens": gather_tokens,
    "residual_add": residual_add,
    "memcpy_f32": memcpy_f32,
    "empty_launch": empty_launch,
    "d2d_copy_Q": d2d_copy,
    "d2d_copy_fuse": d2d_copy,
    "d2d_zero_imp": d2d_copy,
}


def metrics(flops: int, bytes_: int, t_ms: float) -> dict[str, float]:
    t_s = max(float(t_ms) * 1e-3, 1e-12)
    ai = (float(flops) / float(bytes_)) if bytes_ else 0.0
    bw_eff = (float(bytes_) / 1e9) / t_s  # GB/s
    compute_eff = (float(flops) / 1e9) / t_s  # GFLOP/s
    return {"AI": ai, "BW_eff_GBs": bw_eff, "Compute_eff_GFLOPs": compute_eff}


def classify(
    ai: float,
    ridge: float,
    flops: int,
    bytes_: int,
    bw_eff: float = 0.0,
    compute_eff: float = 0.0,
    peak_bw: float = 0.0,
    peak_gflops: float = 0.0,
) -> str:
    if flops == 0 and bytes_ == 0:
        return "overhead"
    if flops == 0:
        return "memory-bound"
    if bytes_ == 0:
        return "compute-bound"
    bw_util = (bw_eff / peak_bw) if peak_bw > 0 else 0.0
    flop_util = (compute_eff / peak_gflops) if peak_gflops > 0 else 0.0
    # Achieved-util first (works even if ridge is noisy on PoCL).
    if bw_util >= 0.25 and flop_util < 0.15:
        return "memory-bound"
    if flop_util >= 0.25 and bw_util < 0.25:
        return "compute-bound"
    if ridge > 0:
        if ai < 0.5 * ridge and bw_util >= 0.10:
            return "memory-bound"
        if ai > 2.0 * ridge and flop_util >= 0.10:
            return "compute-bound"
    if max(bw_util, flop_util) < 0.10:
        return "latency-bound (low util)"
    return "mixed"


def attach(
    flops: int,
    bytes_: int,
    t_ms: float,
    ridge: float,
    peak_bw: float = 0.0,
    peak_gflops: float = 0.0,
) -> dict[str, Any]:
    m = metrics(flops, bytes_, t_ms)
    m["flops"] = int(flops)
    m["bytes"] = int(bytes_)
    m["ridge_AI"] = float(ridge)
    m["BW_util"] = (m["BW_eff_GBs"] / peak_bw) if peak_bw > 0 else 0.0
    m["FLOP_util"] = (m["Compute_eff_GFLOPs"] / peak_gflops) if peak_gflops > 0 else 0.0
    m["bound"] = classify(
        m["AI"],
        ridge,
        flops,
        bytes_,
        bw_eff=m["BW_eff_GBs"],
        compute_eff=m["Compute_eff_GFLOPs"],
        peak_bw=peak_bw,
        peak_gflops=peak_gflops,
    )
    return m


def measure_peaks(ctx: cl.Context, queue: cl.CommandQueue, nbytes: int = 32 << 20) -> dict[str, float]:
    """On-device memcpy BW peak + FMA compute peak (PoCL-CUDA / CPU)."""
    prg = cl.Program(ctx, PROBE_SRC).build(options="-cl-std=CL1.2")
    nfloat = max(nbytes // 4, 1)
    src = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=nfloat * 4)
    dst = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=nfloat * 4)
    host = np.zeros(nfloat, dtype=np.float32)
    cl.enqueue_copy(queue, src, host).wait()

    bw_times = []
    kn = cl.Kernel(prg, "memcpy_f32")
    for i in range(8):
        queue.finish()
        ev = kn(queue, (nfloat,), None, src, dst, np.int32(nfloat))
        ev.wait()
        t = 1e-6 * (ev.profile.end - ev.profile.start)
        if i >= 2:
            bw_times.append(t)
    t_bw = statistics.median(bw_times)
    bytes_moved = nfloat * 8
    peak_bw_gbs = (bytes_moved / 1e9) / max(t_bw * 1e-3, 1e-12)

    n_fma = 1 << 22
    reps = 128
    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=n_fma * 4)
    cl.enqueue_copy(queue, buf, np.ones(n_fma, dtype=np.float32)).wait()
    fk = cl.Kernel(prg, "fma_peak")
    fma_times = []
    for i in range(6):
        queue.finish()
        ev = fk(queue, (n_fma,), None, buf, np.int32(n_fma), np.int32(reps))
        ev.wait()
        t = 1e-6 * (ev.profile.end - ev.profile.start)
        if i >= 2:
            fma_times.append(t)
    t_fma = statistics.median(fma_times)
    flops = n_fma * reps * 8  # 4 fma * 2 FLOPs
    peak_gflops = (flops / 1e9) / max(t_fma * 1e-3, 1e-12)
    ridge = peak_gflops / peak_bw_gbs if peak_bw_gbs > 0 else 0.0
    return {
        "peak_BW_GBs": float(peak_bw_gbs),
        "peak_GFLOPs": float(peak_gflops),
        "ridge_AI": float(ridge),
        "memcpy_probe_bytes": int(bytes_moved),
        "fma_probe_flops": int(flops),
    }
