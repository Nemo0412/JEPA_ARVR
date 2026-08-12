#!/usr/bin/env python3
"""Per-kernel latency breakdown: launch vs device compute vs H2D/D2H IO.

Uses OpenCL event profiling:
  launch_path_ms  = start - queued   (queued→submit→start on device timeline)
  submit_ms       = submit - queued
  device_ms       = end - start      (kernel body: arithmetic + on-device mem access)
  h2d_ms / d2h_ms = buffer transfer events (host ↔ device)

Host enqueue wall-clock (ocl_call_ms) is also reported: queue.finish + enqueue + wait.

Run:
  # CPU (login / any PoCL host)
  /scratch/ll5914/conda_envs/jepa_opencl/bin/python \\
    scripts/opencl_kernels/profile_latency_breakdown.py --cpu

  # GPU node (prefer NVIDIA via PoCL-CUDA)
  .../python scripts/opencl_kernels/profile_latency_breakdown.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyopencl as cl

HERE = Path(__file__).resolve().parent
import sys

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from kernel_roofline import attach, kernel_cost, measure_peaks


def ns_to_ms(x: int | float) -> float:
    return 1e-6 * float(x)


def event_times_ms(ev: cl.Event) -> dict[str, float]:
    ev.wait()
    q = ev.profile.queued
    s = ev.profile.submit
    a = ev.profile.start
    e = ev.profile.end
    return {
        "queued_to_submit_ms": ns_to_ms(s - q),
        "submit_to_start_ms": ns_to_ms(a - s),
        "launch_path_ms": ns_to_ms(a - q),  # queued → start
        "device_ms": ns_to_ms(e - a),  # start → end (compute + on-device IO)
        "total_event_ms": ns_to_ms(e - q),
    }


@dataclass
class KernelStat:
    name: str
    bytes_h2d: int = 0
    bytes_d2h: int = 0
    # medians over iters
    h2d_ms: float = 0.0
    d2h_ms: float = 0.0
    launch_path_ms: float = 0.0
    submit_to_start_ms: float = 0.0
    device_ms: float = 0.0
    ocl_call_ms: float = 0.0  # host wall: enqueue+wait (one shot, avg)
    h2d_gbs: float = 0.0
    d2h_gbs: float = 0.0
    notes: str = ""
    flops: int = 0
    bytes_traffic: int = 0
    AI: float = 0.0
    BW_eff_GBs: float = 0.0
    Compute_eff_GFLOPs: float = 0.0
    bound: str = ""


@dataclass
class OCL:
    ctx: cl.Context
    queue: cl.CommandQueue
    device: cl.Device
    platform: cl.Platform
    kernels: dict
    device_label: str

    @classmethod
    def create(
        cls,
        sources: list[Path],
        platform: int | None = None,
        device: int | None = None,
        prefer_gpu: bool = True,
    ) -> "OCL":
        plats = cl.get_platforms()
        if not plats:
            raise RuntimeError("No OpenCL platforms")

        if platform is not None:
            plat = plats[platform]
            devs = plat.get_devices()
            if device is not None:
                dev = devs[device]
            elif prefer_gpu:
                gpus = [d for d in devs if d.type & cl.device_type.GPU]
                dev = gpus[0] if gpus else devs[0]
            else:
                # prefer CPU if available
                cpus = [d for d in devs if d.type & cl.device_type.CPU]
                dev = cpus[0] if cpus else devs[0]
        else:
            chosen = None
            if prefer_gpu:
                for p in plats:
                    for d in p.get_devices():
                        is_gpu = bool(d.type & cl.device_type.GPU)
                        name_l = d.name.lower()
                        if is_gpu or any(k in name_l for k in ("nvidia", "cuda", "h100", "a100", "l40")):
                            chosen = (p, d)
                            break
                    if chosen:
                        break
            else:
                for p in plats:
                    for d in p.get_devices():
                        if d.type & cl.device_type.CPU or "cpu" in d.name.lower():
                            chosen = (p, d)
                            break
                    if chosen:
                        break
            if chosen is None:
                plat = plats[0]
                dev = plat.get_devices()[0]
            else:
                plat, dev = chosen

        ctx = cl.Context([dev])
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        src = "\n".join(p.read_text() for p in sources)
        # tiny IO probe kernels appended
        src = src + """
__kernel void memcpy_f32(__global const float *src, __global float *dst, const int n) {
    const int i = get_global_id(0);
    if (i < n) dst[i] = src[i];
}
__kernel void empty_launch(const int n) {
    (void)n;
}
"""
        prg = cl.Program(ctx, src).build(options="-cl-std=CL1.2")
        names = [
            "layernorm_rows",
            "gemm_nn",
            "rope_apply",
            "sdpa_rows",
            "sdpa_cross_rows",
            "residual_add",
            "gelu",
            "gated_residual",
            "l2_normalize_rows",
            "qk_logits_chunk",
            "softmax_rows",
            "reduce_chunk_to_imp",
            "recency_reweight",
            "fill_recency",
            "gather_tokens",
            "topk_indices_row",
            "memcpy_f32",
            "empty_launch",
        ]
        kernels = {n: cl.Kernel(prg, n) for n in names}
        dtype = cl.device_type.to_string(dev.type)
        label = f"{plat.name} | {dev.name} | type={dtype}"
        print(f"OpenCL: {label}")
        return cls(ctx, queue, dev, plat, kernels, label)

    def empty_buf(self, nbytes: int) -> cl.Buffer:
        return cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=max(nbytes, 4))


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def profile_transfers(
    ocl: OCL,
    host_in: np.ndarray | None,
    host_out_shape: tuple[int, ...] | None,
    host_out_dtype,
    iters: int,
    warmup: int,
) -> tuple[float, float, int, int, cl.Buffer | None, np.ndarray | None]:
    """Return (h2d_ms_med, d2h_ms_med, bytes_h2d, bytes_d2h, device_buf_or_None, host_out)."""
    mf = cl.mem_flags
    h2d_times: list[float] = []
    d2h_times: list[float] = []
    bytes_h2d = int(host_in.nbytes) if host_in is not None else 0
    bytes_d2h = 0
    dev_buf = None
    host_out = None

    if host_in is not None:
        # allocate once, re-write for timing
        dev_buf = cl.Buffer(ocl.ctx, mf.READ_WRITE, size=host_in.nbytes)
        for i in range(warmup + iters):
            ev = cl.enqueue_copy(ocl.queue, dev_buf, host_in, is_blocking=False)
            t = event_times_ms(ev)["device_ms"]
            if i >= warmup:
                h2d_times.append(t)

    if host_out_shape is not None:
        host_out = np.empty(host_out_shape, dtype=host_out_dtype)
        bytes_d2h = int(host_out.nbytes)
        if dev_buf is None:
            # only D2H: empty device buffer of same size
            dev_buf = ocl.empty_buf(host_out.nbytes)
        for i in range(warmup + iters):
            ev = cl.enqueue_copy(ocl.queue, host_out, dev_buf, is_blocking=False)
            t = event_times_ms(ev)["device_ms"]
            if i >= warmup:
                d2h_times.append(t)

    return median(h2d_times), median(d2h_times), bytes_h2d, bytes_d2h, dev_buf, host_out


def time_kernel(
    ocl: OCL,
    name: str,
    gsize: tuple[int, ...],
    args: tuple,
    bytes_h2d_est: int,
    bytes_d2h_est: int,
    h2d_src: np.ndarray | None,
    d2h_shape: tuple[int, ...] | None,
    d2h_dtype=np.float32,
    warmup: int = 3,
    iters: int = 10,
    notes: str = "",
) -> KernelStat:
    """Time H2D (of representative inputs), kernel launch/device, D2H of outputs."""
    # --- IO: H2D of primary host buffers (if any). For multi-buffer kernels, use packed estimate.
    h2d_ms, d2h_ms, b_h2d, b_d2h, _, _ = profile_transfers(
        ocl, h2d_src, d2h_shape, d2h_dtype, iters=iters, warmup=warmup
    )
    # If caller estimated multi-buffer bytes, scale H2D/D2H times linearly from measured 1-buffer rate.
    # We measure a single bulk transfer of max relevant buffer size for bandwidth, then scale.
    # Better: separate total-bytes timing below.

    # Re-measure IO at full byte sizes with one big buffer each direction (bandwidth proxy),
    # then latency for small kernels is the transfer event for that size.
    def transfer_size_ms(nbytes: int, direction: str) -> float:
        if nbytes <= 0:
            return 0.0
        host = np.empty(nbytes // 4, dtype=np.float32)
        if nbytes % 4:
            host = np.empty((nbytes + 3) // 4, dtype=np.float32)
        nbytes = int(host.nbytes)
        times = []
        buf = cl.Buffer(ocl.ctx, cl.mem_flags.READ_WRITE, size=nbytes)
        for i in range(warmup + iters):
            if direction == "h2d":
                # fill host with small pattern each time would be slow; leave zeros
                ev = cl.enqueue_copy(ocl.queue, buf, host, is_blocking=False)
            else:
                ev = cl.enqueue_copy(ocl.queue, host, buf, is_blocking=False)
            t = event_times_ms(ev)["device_ms"]
            if i >= warmup:
                times.append(t)
        return median(times)

    h2d_ms = transfer_size_ms(bytes_h2d_est, "h2d")
    d2h_ms = transfer_size_ms(bytes_d2h_est, "d2h")
    b_h2d, b_d2h = bytes_h2d_est, bytes_d2h_est

    # --- Kernel launch + device
    launch_path, submit_start, device, ocl_call = [], [], [], []
    kn = ocl.kernels[name]
    for i in range(warmup + iters):
        ocl.queue.finish()
        t0 = time.perf_counter()
        ev = kn(ocl.queue, gsize, None, *args)
        # ensure completion for host ocl_call
        ev.wait()
        t1 = time.perf_counter()
        ets = event_times_ms(ev)
        if i >= warmup:
            launch_path.append(ets["launch_path_ms"])
            submit_start.append(ets["submit_to_start_ms"])
            device.append(ets["device_ms"])
            ocl_call.append(1e3 * (t1 - t0))

    h2d_gbs = (b_h2d / 1e9) / (h2d_ms / 1e3) if h2d_ms > 0 else 0.0
    d2h_gbs = (b_d2h / 1e9) / (d2h_ms / 1e3) if d2h_ms > 0 else 0.0

    return KernelStat(
        name=name,
        bytes_h2d=b_h2d,
        bytes_d2h=b_d2h,
        h2d_ms=h2d_ms,
        d2h_ms=d2h_ms,
        launch_path_ms=median(launch_path),
        submit_to_start_ms=median(submit_start),
        device_ms=median(device),
        ocl_call_ms=mean(ocl_call),
        h2d_gbs=h2d_gbs,
        d2h_gbs=d2h_gbs,
        notes=notes,
    )


def run(N=64, D=64, H=4, Nk=16, Kkeep=16, chunk=16, warmup=3, iters=10, prefer_gpu=True):
    B = 1
    Dh = D // H
    assert D % H == 0 and Dh % 2 == 0

    ocl = OCL.create(
        [HERE / "encoder_block.cl", HERE / "receive_importance.cl", HERE / "prune_ops.cl"],
        prefer_gpu=prefer_gpu,
    )
    peaks = measure_peaks(ocl.ctx, ocl.queue)
    ridge = peaks["ridge_AI"]
    print(
        f"peaks: BW={peaks['peak_BW_GBs']:.1f} GB/s  "
        f"FLOP={peaks['peak_GFLOPs']:.1f} GFLOP/s  ridge_AI={ridge:.3f} FLOP/Byte"
    )
    rng = np.random.default_rng(0)
    f4 = np.float32
    i4 = np.int32
    stats: list[KernelStat] = []

    # Shared random payloads
    x_nd = rng.standard_normal((B * N, D), dtype=f4)
    gamma = rng.standard_normal((D,), dtype=f4)
    beta = rng.standard_normal((D,), dtype=f4)
    A = rng.standard_normal((N, D), dtype=f4)
    W = rng.standard_normal((D, D), dtype=f4)
    q_rope = rng.standard_normal((B * H * N, Dh), dtype=f4)
    pos = np.tile(np.arange(N, dtype=f4), B * H)
    Q = rng.standard_normal((B, H, N, Dh), dtype=f4)
    K = rng.standard_normal((B, H, N, Dh), dtype=f4)
    V = rng.standard_normal((B, H, N, Dh), dtype=f4)
    Qc = rng.standard_normal((B, H, N, Dh), dtype=f4)
    Kc = rng.standard_normal((B, H, Nk, Dh), dtype=f4)
    Vc = rng.standard_normal((B, H, Nk, Dh), dtype=f4)
    t_vec = rng.standard_normal((N * D,), dtype=f4)
    z = rng.standard_normal((N * D,), dtype=f4)
    delta = rng.standard_normal((N * D,), dtype=f4)
    glog = rng.standard_normal((N * D,), dtype=f4)
    tokens = rng.standard_normal((B, N, D), dtype=f4)
    scores = rng.standard_normal((B, N), dtype=f4)
    idx = np.sort(rng.integers(0, N, size=(B, Kkeep), dtype=np.int32), axis=1)

    mf = cl.mem_flags

    def to_dev(arr, ro=False):
        flags = mf.READ_ONLY | mf.COPY_HOST_PTR if ro else mf.READ_WRITE | mf.COPY_HOST_PTR
        return cl.Buffer(ocl.ctx, flags, hostbuf=arr)

    # ---- Baseline: empty launch + pure device memcpy ----
    stats.append(
        time_kernel(
            ocl,
            "empty_launch",
            (1,),
            (i4(0),),
            bytes_h2d_est=0,
            bytes_d2h_est=0,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
            notes="pure launch overhead (empty NDRange)",
        )
    )
    n_copy = max(N * D, 1)
    src_c = to_dev(rng.standard_normal((n_copy,), dtype=f4))
    dst_c = ocl.empty_buf(n_copy * 4)
    stats.append(
        time_kernel(
            ocl,
            "memcpy_f32",
            (n_copy,),
            (src_c, dst_c, i4(n_copy)),
            bytes_h2d_est=n_copy * 4,
            bytes_d2h_est=n_copy * 4,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
            notes=f"on-device copy {n_copy} floats; H2D/D2H sized to that buffer",
        )
    )

    # ---- layernorm ----
    bx, bg, bb = to_dev(x_nd), to_dev(gamma, True), to_dev(beta, True)
    by = ocl.empty_buf(x_nd.nbytes)
    stats.append(
        time_kernel(
            ocl,
            "layernorm_rows",
            (B * N,),
            (bx, by, bg, bb, i4(B * N), i4(D), f4(1e-6)),
            bytes_h2d_est=x_nd.nbytes + gamma.nbytes + beta.nbytes,
            bytes_d2h_est=x_nd.nbytes,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )

    # ---- gemm ----
    ba, bw = to_dev(A), to_dev(W, True)
    bc = to_dev(np.zeros((N, D), dtype=f4))
    stats.append(
        time_kernel(
            ocl,
            "gemm_nn",
            (N * D,),
            (ba, bw, bc, i4(N), i4(D), i4(D), f4(1), f4(0)),
            bytes_h2d_est=A.nbytes + W.nbytes,
            bytes_d2h_est=A.nbytes,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
            notes="QKV/MLP linear piece [N,D]@[D,D]",
        )
    )

    # ---- rope ----
    bq = to_dev(q_rope.copy())
    bp = to_dev(pos, True)
    stats.append(
        time_kernel(
            ocl,
            "rope_apply",
            (B * H * N,),
            (bq, bp, i4(B * H * N), i4(Dh)),
            bytes_h2d_est=q_rope.nbytes + pos.nbytes,
            bytes_d2h_est=q_rope.nbytes,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )

    # ---- sdpa self ----
    scale = f4(Dh**-0.5)
    bQ, bK, bV = to_dev(Q), to_dev(K), to_dev(V)
    bO = ocl.empty_buf(Q.nbytes)
    stats.append(
        time_kernel(
            ocl,
            "sdpa_rows",
            (B * H * N,),
            (bQ, bK, bV, bO, i4(B), i4(H), i4(N), i4(Dh), scale),
            bytes_h2d_est=Q.nbytes * 3,
            bytes_d2h_est=Q.nbytes,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
            notes="self-attn",
        )
    )

    # ---- sdpa cross ----
    bQc, bKc, bVc = to_dev(Qc), to_dev(Kc), to_dev(Vc)
    bOc = ocl.empty_buf(Qc.nbytes)
    stats.append(
        time_kernel(
            ocl,
            "sdpa_cross_rows",
            (B * H * N,),
            (bQc, bKc, bVc, bOc, i4(B), i4(H), i4(N), i4(Nk), i4(Dh), scale),
            bytes_h2d_est=Qc.nbytes + Kc.nbytes + Vc.nbytes,
            bytes_d2h_est=Qc.nbytes,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
            notes="cross-attn Nk",
        )
    )

    # ---- gelu ----
    bt, by_g = to_dev(t_vec), ocl.empty_buf(t_vec.nbytes)
    stats.append(
        time_kernel(
            ocl,
            "gelu",
            (t_vec.size,),
            (bt, by_g, i4(t_vec.size)),
            bytes_h2d_est=t_vec.nbytes,
            bytes_d2h_est=t_vec.nbytes,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )

    # ---- gated residual ----
    bz, bd, bg2 = to_dev(z), to_dev(delta), to_dev(glog)
    bo = ocl.empty_buf(z.nbytes)
    stats.append(
        time_kernel(
            ocl,
            "gated_residual",
            (z.size,),
            (bz, bd, bg2, bo, i4(z.size)),
            bytes_h2d_est=z.nbytes * 3,
            bytes_d2h_est=z.nbytes,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )

    # ---- l2 norm ----
    bin_ = to_dev(tokens)
    bxn = ocl.empty_buf(tokens.nbytes)
    stats.append(
        time_kernel(
            ocl,
            "l2_normalize_rows",
            (B * N,),
            (bin_, bxn, i4(B * N), i4(D)),
            bytes_h2d_est=tokens.nbytes,
            bytes_d2h_est=tokens.nbytes,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )

    # ---- receive_importance chunks summed ----
    # Time representative inner kernels; also report sum as pipeline.
    max_C = min(chunk, N)
    blogits = ocl.empty_buf(B * max_C * N * 4)
    battn = ocl.empty_buf(B * max_C * N * 4)
    bimp = to_dev(np.zeros((B, N), dtype=f4))
    sc = f4(D**-0.5)
    # Use first chunk only for detailed breakdown; scale note
    C = max_C
    qs = 0
    stats.append(
        time_kernel(
            ocl,
            "qk_logits_chunk",
            (B * C * N,),
            (bxn, blogits, i4(B), i4(N), i4(D), i4(qs), i4(C), sc),
            bytes_h2d_est=tokens.nbytes,
            bytes_d2h_est=B * C * N * 4,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
            notes=f"one chunk C={C}; full receive ~N/C chunks",
        )
    )
    stats.append(
        time_kernel(
            ocl,
            "softmax_rows",
            (B * C,),
            (blogits, battn, i4(B * C), i4(N)),
            bytes_h2d_est=B * C * N * 4,
            bytes_d2h_est=B * C * N * 4,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )
    stats.append(
        time_kernel(
            ocl,
            "reduce_chunk_to_imp",
            (B * N,),
            (battn, bimp, i4(B), i4(N), i4(C)),
            bytes_h2d_est=B * C * N * 4,
            bytes_d2h_est=B * N * 4,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )

    # full receive_importance: device only (buffers resident)
    n_chunks = (N + chunk - 1) // chunk
    # measure multi-chunk device total once with events sum
    recv_device = []
    for it in range(warmup + iters):
        ocl.queue.finish()
        bimp2 = to_dev(np.zeros((B, N), dtype=f4))
        tsum = 0.0
        for qs in range(0, N, chunk):
            C = min(chunk, N - qs)
            e1 = ocl.kernels["qk_logits_chunk"](
                ocl.queue, (B * C * N,), None, bxn, blogits, i4(B), i4(N), i4(D), i4(qs), i4(C), sc
            )
            e1.wait()
            tsum += event_times_ms(e1)["device_ms"]
            e2 = ocl.kernels["softmax_rows"](ocl.queue, (B * C,), None, blogits, battn, i4(B * C), i4(N))
            e2.wait()
            tsum += event_times_ms(e2)["device_ms"]
            e3 = ocl.kernels["reduce_chunk_to_imp"](
                ocl.queue, (B * N,), None, battn, bimp2, i4(B), i4(N), i4(C)
            )
            e3.wait()
            tsum += event_times_ms(e3)["device_ms"]
        if it >= warmup:
            recv_device.append(tsum)
    # IO for tokens in + imp out
    h2d_tok = transfer_size_ms_helper(ocl, tokens.nbytes, "h2d", warmup, iters)
    d2h_imp = transfer_size_ms_helper(ocl, B * N * 4, "d2h", warmup, iters)
    stats.append(
        KernelStat(
            name="receive_importance_full",
            bytes_h2d=tokens.nbytes,
            bytes_d2h=B * N * 4,
            h2d_ms=h2d_tok,
            d2h_ms=d2h_imp,
            launch_path_ms=0.0,
            submit_to_start_ms=0.0,
            device_ms=median(recv_device),
            ocl_call_ms=0.0,
            h2d_gbs=(tokens.nbytes / 1e9) / (h2d_tok / 1e3) if h2d_tok > 0 else 0.0,
            d2h_gbs=(B * N * 4 / 1e9) / (d2h_imp / 1e3) if d2h_imp > 0 else 0.0,
            notes=f"sum device_ms over {n_chunks} chunks; launch is multi-kernel",
        )
    )

    # ---- prune helpers ----
    brec = ocl.empty_buf(N * 4)
    stats.append(
        time_kernel(
            ocl,
            "fill_recency",
            (N,),
            (brec, i4(N), i4(8)),
            bytes_h2d_est=0,
            bytes_d2h_est=N * 4,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )
    bscores = to_dev(scores.copy())
    stats.append(
        time_kernel(
            ocl,
            "recency_reweight",
            (B * N,),
            (bscores, brec, i4(B), i4(N), f4(0.5)),
            bytes_h2d_est=scores.nbytes + N * 4,
            bytes_d2h_est=scores.nbytes,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )
    bidx = ocl.empty_buf(B * Kkeep * 4)
    stats.append(
        time_kernel(
            ocl,
            "topk_indices_row",
            (B,),
            (bscores, bidx, i4(B), i4(N), i4(Kkeep)),
            bytes_h2d_est=scores.nbytes,
            bytes_d2h_est=B * Kkeep * 4,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )
    btok, bids = to_dev(tokens), to_dev(idx)
    bout = ocl.empty_buf(B * Kkeep * D * 4)
    stats.append(
        time_kernel(
            ocl,
            "gather_tokens",
            (B * Kkeep * D,),
            (btok, bids, bout, i4(B), i4(N), i4(Kkeep), i4(D)),
            bytes_h2d_est=tokens.nbytes + idx.nbytes,
            bytes_d2h_est=B * Kkeep * D * 4,
            h2d_src=None,
            d2h_shape=None,
            warmup=warmup,
            iters=iters,
        )
    )

    cost_kw = {
        "empty_launch": {},
        "memcpy_f32": {"n": n_copy},
        "layernorm_rows": {"rows": B * N, "D": D},
        "gemm_nn": {"M": N, "N": D, "K": D, "beta": 0.0},
        "rope_apply": {"rows": B * H * N, "Dh": Dh},
        "sdpa_rows": {"B": B, "H": H, "N": N, "Dh": Dh},
        "sdpa_cross_rows": {"B": B, "H": H, "Nq": N, "Nk": Nk, "Dh": Dh},
        "gelu": {"n": int(t_vec.size)},
        "gated_residual": {"n": int(z.size)},
        "l2_normalize_rows": {"rows": B * N, "D": D},
        "qk_logits_chunk": {"B": B, "N": N, "D": D, "C": max_C},
        "softmax_rows": {"rows": B * max_C, "N": N},
        "reduce_chunk_to_imp": {"B": B, "N": N, "C": max_C},
        "fill_recency": {"N": N},
        "recency_reweight": {"B": B, "N": N},
        "topk_indices_row": {"B": B, "N": N, "K": Kkeep},
        "gather_tokens": {"B": B, "K": Kkeep, "D": D},
    }
    recv_flops = recv_bytes = 0
    for qs in range(0, N, chunk):
        C = min(chunk, N - qs)
        f, b = kernel_cost("qk_logits_chunk", B=B, N=N, D=D, C=C)
        recv_flops += f
        recv_bytes += b
        f, b = kernel_cost("softmax_rows", rows=B * C, N=N)
        recv_flops += f
        recv_bytes += b
        f, b = kernel_cost("reduce_chunk_to_imp", B=B, N=N, C=C)
        recv_flops += f
        recv_bytes += b
    f, b = kernel_cost("l2_normalize_rows", rows=B * N, D=D)
    recv_flops += f
    recv_bytes += b

    for s in stats:
        if s.name == "receive_importance_full":
            flops, nbytes = recv_flops, recv_bytes
        else:
            flops, nbytes = kernel_cost(s.name, **cost_kw.get(s.name, {}))
        m = attach(
            flops,
            nbytes,
            s.device_ms,
            ridge,
            peak_bw=peaks["peak_BW_GBs"],
            peak_gflops=peaks["peak_GFLOPs"],
        )
        s.flops = m["flops"]
        s.bytes_traffic = m["bytes"]
        s.AI = m["AI"]
        s.BW_eff_GBs = m["BW_eff_GBs"]
        s.Compute_eff_GFLOPs = m["Compute_eff_GFLOPs"]
        s.bound = m["bound"]

    # print table
    print()
    print(f"shapes: N={N} D={D} H={H} Nk={Nk} K={Kkeep} chunk={chunk}  warmup={warmup} iters={iters}")
    print(
        f"{'kernel':28s} {'device':>8s} {'FLOPs':>12s} {'Bytes':>12s} "
        f"{'AI':>8s} {'BW_eff':>8s} {'GFLOP/s':>8s}  bound"
    )
    print("-" * 120)
    rows = []
    for s in stats:
        io_ms = s.h2d_ms + s.d2h_ms
        print(
            f"{s.name:28s} {s.device_ms:8.3f} {s.flops:12d} {s.bytes_traffic:12d} "
            f"{s.AI:8.3f} {s.BW_eff_GBs:8.2f} {s.Compute_eff_GFLOPs:8.2f}  {s.bound}"
        )
        rows.append({**asdict(s), "io_ms": io_ms, "peaks": peaks})
    print()
    print(
        "columns:\n"
        f"  ridge_AI={ridge:.3f} FLOP/Byte  (peak {peaks['peak_GFLOPs']:.1f} GFLOP/s, {peaks['peak_BW_GBs']:.1f} GB/s)\n"
        "  AI = FLOPs / Bytes (issued global ld/st, re-reads counted)\n"
        "  BW_eff GB/s = Bytes / T_device ;  Compute_eff GFLOP/s = FLOPs / T_device\n"
        "  memory-bound if AI < 0.5*ridge ; compute-bound if AI > 2*ridge\n"
    )
    return {
        "device": ocl.device_label,
        "prefer_gpu": prefer_gpu,
        "peaks": peaks,
        "shapes": {"N": N, "D": D, "H": H, "Nk": Nk, "K": Kkeep, "chunk": chunk},
        "warmup": warmup,
        "iters": iters,
        "kernels": rows,
    }


def transfer_size_ms_helper(ocl: OCL, nbytes: int, direction: str, warmup: int, iters: int) -> float:
    if nbytes <= 0:
        return 0.0
    nfloat = max((nbytes + 3) // 4, 1)
    host = np.zeros(nfloat, dtype=np.float32)
    nbytes = int(host.nbytes)
    times = []
    buf = cl.Buffer(ocl.ctx, cl.mem_flags.READ_WRITE, size=nbytes)
    for i in range(warmup + iters):
        if direction == "h2d":
            ev = cl.enqueue_copy(ocl.queue, buf, host, is_blocking=False)
        else:
            ev = cl.enqueue_copy(ocl.queue, host, buf, is_blocking=False)
        t = event_times_ms(ev)["device_ms"]
        if i >= warmup:
            times.append(t)
    return median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--Nk", type=int, default=16)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cpu", action="store_true", help="Force CPU device")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    rep = run(
        N=args.N,
        D=args.D,
        H=args.H,
        Nk=args.Nk,
        Kkeep=args.K,
        chunk=args.chunk,
        warmup=args.warmup,
        iters=args.iters,
        prefer_gpu=not args.cpu,
    )
    out = Path(args.out) if args.out else HERE / (
        "profile_breakdown_cpu.json" if args.cpu else "profile_breakdown_gpu.json"
    )
    out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
