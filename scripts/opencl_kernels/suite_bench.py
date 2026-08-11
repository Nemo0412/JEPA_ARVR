#!/usr/bin/env python3
"""Unified OpenCL smoke-bench for V-JEPA / HD-EPIC streaming kernels.

Maps to the concat+CA stream path:
  adapter → patch_embed → encoder blocks (LN/QKV/RoPE/SDPA/MLP)
  → IMU CA (+ gate residual) → receive-importance → recency/topk/gather
  → predictor blocks (same attn primitives @ Dp)

These are research baselines for system-level ideation (fusion, scheduling,
memory layout), not production FlashAttention replacements.

Run:
  /scratch/ll5914/conda_envs/jepa_opencl/bin/python \\
    scripts/opencl_kernels/suite_bench.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyopencl as cl

HERE = Path(__file__).resolve().parent


def gelu_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layernorm_np(x, gamma, beta, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def rope_np(x, pos):
    """x [rows, Dh], pos [rows] — Meta V-JEPA2 pretrained-compatible freqs."""
    rows, Dh = x.shape
    assert Dh % 2 == 0
    half = Dh // 2
    omega = np.arange(half, dtype=np.float32) / float(half)
    omega = (10000.0 ** (-omega)).astype(np.float32)
    freq = pos[:, None] * omega[None, :]  # [rows, half]
    emb_sin = np.sin(freq)
    emb_cos = np.cos(freq)
    # duplicate freqs (pretrained bug-compatible)
    emb_sin = np.concatenate([emb_sin, emb_sin], axis=-1)
    emb_cos = np.concatenate([emb_cos, emb_cos], axis=-1)
    y = x.reshape(rows, half, 2)
    y1, y2 = y[..., 0], y[..., 1]
    y_rot = np.stack([-y2, y1], axis=-1).reshape(rows, Dh)
    return x * emb_cos + y_rot * emb_sin


def sdpa_np(Q, K, V, scale):
    # Q,K,V: [B,H,N,Dh]
    logits = np.einsum("bhnd,bhmd->bhnm", Q, K) * scale
    m = logits.max(axis=-1, keepdims=True)
    attn = np.exp(logits - m)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return np.einsum("bhnm,bhmd->bhnd", attn, V)


def sdpa_cross_np(Q, K, V, scale):
    logits = np.einsum("bhnd,bhmd->bhnm", Q, K) * scale
    m = logits.max(axis=-1, keepdims=True)
    attn = np.exp(logits - m)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return np.einsum("bhnm,bhmd->bhnd", attn, V)


@dataclass
class OCL:
    ctx: cl.Context
    queue: cl.CommandQueue
    kernels: dict

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

        # Explicit selection, else prefer first GPU-capable device across platforms.
        if platform is not None:
            plat = plats[platform]
            devs = plat.get_devices()
            if device is not None:
                dev = devs[device]
            elif prefer_gpu:
                gpus = [d for d in devs if d.type & cl.device_type.GPU]
                dev = gpus[0] if gpus else devs[0]
            else:
                dev = devs[0]
        else:
            chosen = None
            if prefer_gpu:
                for p in plats:
                    for d in p.get_devices():
                        # POCL CUDA backend often reports GPU; also match name.
                        is_gpu = bool(d.type & cl.device_type.GPU)
                        name_l = d.name.lower()
                        if is_gpu or "nvidia" in name_l or "cuda" in name_l or "h100" in name_l or "a100" in name_l:
                            chosen = (p, d)
                            break
                    if chosen:
                        break
            if chosen is None:
                plat = plats[0 if platform is None else platform]
                dev = plat.get_devices()[0 if device is None else device]
            else:
                plat, dev = chosen

        ctx = cl.Context([dev])
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        src = "\n".join(p.read_text() for p in sources)
        # Avoid duplicate symbols if receive + encoder both define nothing conflicting
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
        ]
        kernels = {n: cl.Kernel(prg, n) for n in names}
        dtype = cl.device_type.to_string(dev.type)
        print(f"OpenCL: {plat.name} | {dev.name} | type={dtype}")
        return cls(ctx, queue, kernels)

    def buf(self, arr: np.ndarray, write=True):
        mf = cl.mem_flags
        if write:
            return cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr)
        return cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)

    def empty(self, nbytes: int):
        return cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=nbytes)

    def read(self, buf, shape, dtype=np.float32):
        out = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self.queue, out, buf).wait()
        return out

    def profile_ms(self, event) -> float:
        event.wait()
        return 1e-6 * (event.profile.end - event.profile.start)


def check(name, got, ref, tol=1e-4):
    diff = float(np.max(np.abs(got - ref)))
    scale = float(max(np.max(np.abs(ref)), 1.0))
    ok = diff < tol * scale
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:28s} max_abs={diff:.3e}")
    return ok


def run_suite(
    N=64,
    D=64,
    H=4,
    Nk=16,
    Kkeep=16,
    chunk=16,
    platform: int | None = None,
    device: int | None = None,
    prefer_gpu: bool = True,
):
    B = 1
    Dh = D // H
    assert D % H == 0 and Dh % 2 == 0

    ocl = OCL.create(
        [
            HERE / "encoder_block.cl",
            HERE / "receive_importance.cl",
            HERE / "prune_ops.cl",
        ],
        platform=platform,
        device=device,
        prefer_gpu=prefer_gpu,
    )
    rng = np.random.default_rng(0)
    ok_all = True
    timings = {}

    # ---- LayerNorm ----
    x = rng.standard_normal((B * N, D), dtype=np.float32)
    gamma = rng.standard_normal((D,), dtype=np.float32)
    beta = rng.standard_normal((D,), dtype=np.float32)
    ref = layernorm_np(x.reshape(B * N, D), gamma, beta)
    bx, bg, bb = ocl.buf(x), ocl.buf(gamma), ocl.buf(beta)
    by = ocl.empty(x.nbytes)
    e = ocl.kernels["layernorm_rows"](
        ocl.queue, (B * N,), None, bx, by, bg, bb, np.int32(B * N), np.int32(D), np.float32(1e-6)
    )
    timings["layernorm"] = ocl.profile_ms(e)
    ok_all &= check("layernorm", ocl.read(by, x.shape), ref)

    # ---- GEMM (QKV piece): [N,D] @ [D,D] ----
    A = rng.standard_normal((N, D), dtype=np.float32)
    W = rng.standard_normal((D, D), dtype=np.float32)
    ref = A @ W
    ba, bw = ocl.buf(A), ocl.buf(W)
    bc = ocl.buf(np.zeros_like(ref))
    e = ocl.kernels["gemm_nn"](
        ocl.queue, (N * D,), None, ba, bw, bc, np.int32(N), np.int32(D), np.int32(D), np.float32(1), np.float32(0)
    )
    timings["gemm_nn"] = ocl.profile_ms(e)
    ok_all &= check("gemm_nn (attn/MLP linear)", ocl.read(bc, ref.shape), ref, tol=2e-3)

    # ---- RoPE ----
    q = rng.standard_normal((B * H * N, Dh), dtype=np.float32)
    # axial positions simplified: use token index
    pos = np.tile(np.arange(N, dtype=np.float32), B * H)
    ref = rope_np(q.copy(), pos)
    bq = ocl.buf(q.copy())
    bp = ocl.buf(pos)
    e = ocl.kernels["rope_apply"](ocl.queue, (B * H * N,), None, bq, bp, np.int32(B * H * N), np.int32(Dh))
    timings["rope_apply"] = ocl.profile_ms(e)
    ok_all &= check("rope_apply (V-JEPA)", ocl.read(bq, q.shape), ref, tol=2e-4)

    # ---- SDPA self ----
    Q = rng.standard_normal((B, H, N, Dh), dtype=np.float32)
    K = rng.standard_normal((B, H, N, Dh), dtype=np.float32)
    V = rng.standard_normal((B, H, N, Dh), dtype=np.float32)
    scale = Dh**-0.5
    ref = sdpa_np(Q, K, V, scale)
    bQ, bK, bV = ocl.buf(Q), ocl.buf(K), ocl.buf(V)
    bO = ocl.empty(Q.nbytes)
    e = ocl.kernels["sdpa_rows"](
        ocl.queue,
        (B * H * N,),
        None,
        bQ,
        bK,
        bV,
        bO,
        np.int32(B),
        np.int32(H),
        np.int32(N),
        np.int32(Dh),
        np.float32(scale),
    )
    timings["sdpa_self"] = ocl.profile_ms(e)
    ok_all &= check("sdpa_self (encoder/pred)", ocl.read(bO, Q.shape), ref, tol=2e-4)

    # ---- SDPA cross (fusion CA) ----
    Qc = rng.standard_normal((B, H, N, Dh), dtype=np.float32)
    Kc = rng.standard_normal((B, H, Nk, Dh), dtype=np.float32)
    Vc = rng.standard_normal((B, H, Nk, Dh), dtype=np.float32)
    ref = sdpa_cross_np(Qc, Kc, Vc, scale)
    bQc, bKc, bVc = ocl.buf(Qc), ocl.buf(Kc), ocl.buf(Vc)
    bOc = ocl.empty(Qc.nbytes)
    e = ocl.kernels["sdpa_cross_rows"](
        ocl.queue,
        (B * H * N,),
        None,
        bQc,
        bKc,
        bVc,
        bOc,
        np.int32(B),
        np.int32(H),
        np.int32(N),
        np.int32(Nk),
        np.int32(Dh),
        np.float32(scale),
    )
    timings["sdpa_cross"] = ocl.profile_ms(e)
    ok_all &= check("sdpa_cross (IMU CA)", ocl.read(bOc, Qc.shape), ref, tol=2e-4)

    # ---- GELU + gated residual ----
    t = rng.standard_normal((N * D,), dtype=np.float32)
    ref = gelu_np(t)
    bt, by = ocl.buf(t), ocl.empty(t.nbytes)
    e = ocl.kernels["gelu"](ocl.queue, (t.size,), None, bt, by, np.int32(t.size))
    timings["gelu"] = ocl.profile_ms(e)
    ok_all &= check("gelu (MLP)", ocl.read(by, t.shape), ref, tol=2e-4)

    z = rng.standard_normal((N * D,), dtype=np.float32)
    delta = rng.standard_normal((N * D,), dtype=np.float32)
    glog = rng.standard_normal((N * D,), dtype=np.float32)
    ref = z + (1.0 / (1.0 + np.exp(-glog))) * delta
    bz, bd, bg, bo = ocl.buf(z), ocl.buf(delta), ocl.buf(glog), ocl.empty(z.nbytes)
    e = ocl.kernels["gated_residual"](ocl.queue, (z.size,), None, bz, bd, bg, bo, np.int32(z.size))
    timings["gated_residual"] = ocl.profile_ms(e)
    ok_all &= check("gated_residual (fusion)", ocl.read(bo, z.shape), ref, tol=2e-4)

    # ---- Receive importance (existing) ----
    tokens = rng.standard_normal((B, N, D), dtype=np.float32)
    # numpy ref
    xn = tokens / np.linalg.norm(tokens, axis=-1, keepdims=True).clip(1e-12)
    sc = D**-0.5
    imp_ref = np.zeros((B, N), np.float32)
    for ci in range(0, N, chunk):
        qch = xn[:, ci : ci + chunk]
        logits = np.matmul(qch, xn.transpose(0, 2, 1)) * sc
        m = logits.max(-1, keepdims=True)
        attn = np.exp(logits - m)
        attn /= attn.sum(-1, keepdims=True)
        imp_ref += attn.sum(1)

    bin_ = ocl.buf(tokens)
    bxn = ocl.empty(tokens.nbytes)
    bimp = ocl.buf(np.zeros((B, N), np.float32))
    e = ocl.kernels["l2_normalize_rows"](
        ocl.queue, (B * N,), None, bin_, bxn, np.int32(B * N), np.int32(D)
    )
    timings["l2_norm"] = ocl.profile_ms(e)
    max_C = min(chunk, N)
    blogits = ocl.empty(B * max_C * N * 4)
    battn = ocl.empty(B * max_C * N * 4)
    t_qk = t_sm = t_rd = 0.0
    for qs in range(0, N, chunk):
        C = min(chunk, N - qs)
        e1 = ocl.kernels["qk_logits_chunk"](
            ocl.queue,
            (B * C * N,),
            None,
            bxn,
            blogits,
            np.int32(B),
            np.int32(N),
            np.int32(D),
            np.int32(qs),
            np.int32(C),
            np.float32(sc),
        )
        t_qk += ocl.profile_ms(e1)
        e2 = ocl.kernels["softmax_rows"](
            ocl.queue, (B * C,), None, blogits, battn, np.int32(B * C), np.int32(N)
        )
        t_sm += ocl.profile_ms(e2)
        e3 = ocl.kernels["reduce_chunk_to_imp"](
            ocl.queue, (B * N,), None, battn, bimp, np.int32(B), np.int32(N), np.int32(C)
        )
        t_rd += ocl.profile_ms(e3)
    timings["receive_importance"] = t_qk + t_sm + t_rd
    ok_all &= check("receive_importance", ocl.read(bimp, (B, N)), imp_ref, tol=2e-4)

    # ---- Recency + topk + gather ----
    scores = imp_ref.copy()
    gp = 8
    rec = np.empty((N,), np.float32)
    brec = ocl.empty(N * 4)
    e = ocl.kernels["fill_recency"](ocl.queue, (N,), None, brec, np.int32(N), np.int32(gp))
    timings["fill_recency"] = ocl.profile_ms(e)
    rec = ocl.read(brec, (N,))
    strength = 0.5
    bscores = ocl.buf(scores.copy())
    e = ocl.kernels["recency_reweight"](
        ocl.queue, (B * N,), None, bscores, brec, np.int32(B), np.int32(N), np.float32(strength)
    )
    timings["recency_reweight"] = ocl.profile_ms(e)
    scores_ref = scores * (1.0 + strength * rec)[None, :]
    ok_all &= check("recency_reweight", ocl.read(bscores, scores.shape), scores_ref)

    bidx = ocl.empty(B * Kkeep * 4)
    e = ocl.kernels["topk_indices_row"](
        ocl.queue, (B,), None, bscores, bidx, np.int32(B), np.int32(N), np.int32(Kkeep)
    )
    timings["topk_indices"] = ocl.profile_ms(e)
    idx = ocl.read(bidx, (B, Kkeep), dtype=np.int32)
    # verify set equality with numpy argpartition
    ref_idx = np.argpartition(-scores_ref, Kkeep, axis=1)[:, :Kkeep]
    set_ok = set(idx[0].tolist()) == set(ref_idx[0].tolist())
    print(f"  [{'PASS' if set_ok else 'FAIL'}] {'topk_indices':28s} set_equal={set_ok}")
    ok_all &= set_ok

    # gather in original order: sort idx
    idx_sorted = np.sort(idx, axis=1)
    tokens_g = tokens
    ref_g = np.take_along_axis(tokens_g, idx_sorted[..., None].repeat(D, axis=2), axis=1)
    btok, bids, bout = ocl.buf(tokens_g), ocl.buf(idx_sorted.astype(np.int32)), ocl.empty(B * Kkeep * D * 4)
    e = ocl.kernels["gather_tokens"](
        ocl.queue,
        (B * Kkeep * D,),
        None,
        btok,
        bids,
        bout,
        np.int32(B),
        np.int32(N),
        np.int32(Kkeep),
        np.int32(D),
    )
    timings["gather_tokens"] = ocl.profile_ms(e)
    ok_all &= check("gather_tokens", ocl.read(bout, (B, Kkeep, D)), ref_g)

    print("\n=== Profile (ms, OpenCL event timestamps) ===")
    for k, v in timings.items():
        print(f"  {k:28s} {v:8.3f}")
    print(f"\nALL_PASS={ok_all}  shapes: N={N} D={D} H={H} Nk={Nk} K={Kkeep}")

    print(
        """
Pipeline ↔ kernels
  1 encoder Block:     layernorm | gemm_nn(QKV/proj/MLP) | rope_apply | sdpa_self | gelu | residual
  2 fusion CA:         gemm_nn(Wqkv/Wo) | sdpa_cross | gated_residual
  3 prune:             receive_importance | fill_recency | recency_reweight | topk | gather
  4 predictor Block:   same as encoder @ Dp (reuse sdpa_self/rope/gemm/gelu)
System ideas to explore next: fuse rope+sdpa; fuse receive-imp without materializing logits;
fuse CA+gate epilogue; streaming KV cache for AR predictor; prune+gather in one pass.
"""
    )
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--Nk", type=int, default=16)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--platform", type=int, default=None)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--cpu", action="store_true", help="Prefer/force CPU (do not auto-pick GPU)")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()
    if args.list_devices:
        for i, p in enumerate(cl.get_platforms()):
            print(f"[{i}] {p.name}")
            for j, d in enumerate(p.get_devices()):
                print(f"  [{j}] {d.name} type={cl.device_type.to_string(d.type)}")
        raise SystemExit(0)
    ok = run_suite(
        N=args.N,
        D=args.D,
        H=args.H,
        Nk=args.Nk,
        Kkeep=args.K,
        chunk=args.chunk,
        platform=args.platform,
        device=args.device,
        prefer_gpu=not args.cpu,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
