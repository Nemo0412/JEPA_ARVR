#!/usr/bin/env python3
"""Pre-profiling for unique JEPA ideas:
  (1) Fused postfuse receive-importance prune
  (2) Fusion-aware prune (CA coupled with receive scores)

Measures current OpenCL baseline cost / scaling / bandwidth proxies on the
login-node POCL CPU device. Absolute ms are not GPU truth; slopes and
stage ratios are what matter for deciding what to fuse next.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyopencl as cl

HERE = Path(__file__).resolve().parent


def build():
    plat = cl.get_platforms()[0]
    dev = plat.get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    src = (HERE / "receive_importance.cl").read_text() + "\n" + (HERE / "encoder_block.cl").read_text()
    prg = cl.Program(ctx, src).build(options="-cl-std=CL1.2")
    ks = {
        n: cl.Kernel(prg, n)
        for n in [
            "l2_normalize_rows",
            "qk_logits_chunk",
            "softmax_rows",
            "reduce_chunk_to_imp",
            "sdpa_cross_rows",
            "gemm_nn",
            "gated_residual",
        ]
    }
    print(f"device: {plat.name} | {dev.name}")
    return ctx, q, ks


def ms(ev):
    ev.wait()
    return 1e-6 * (ev.profile.end - ev.profile.start)


def profile_receive(ctx, q, ks, B, N, D, chunk, warmup=1, iters=3):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, N, D), dtype=np.float32)
    mf = cl.mem_flags
    scale = np.float32(D**-0.5)

    def once():
        bin_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        bx = cl.Buffer(ctx, mf.READ_WRITE, size=x.nbytes)
        bimp = cl.Buffer(ctx, mf.READ_WRITE, size=B * N * 4)
        cl.enqueue_fill_buffer(q, bimp, np.float32(0), 0, B * N * 4).wait()
        t = {"norm": 0.0, "qk": 0.0, "softmax": 0.0, "reduce": 0.0}
        t["norm"] += ms(ks["l2_normalize_rows"](q, (B * N,), None, bin_, bx, np.int32(B * N), np.int32(D)))
        max_C = min(chunk, N)
        blogits = cl.Buffer(ctx, mf.READ_WRITE, size=B * max_C * N * 4)
        battn = cl.Buffer(ctx, mf.READ_WRITE, size=B * max_C * N * 4)
        nbytes_logits_peak = B * max_C * N * 4
        for qs in range(0, N, chunk):
            C = min(chunk, N - qs)
            t["qk"] += ms(
                ks["qk_logits_chunk"](
                    q,
                    (B * C * N,),
                    None,
                    bx,
                    blogits,
                    np.int32(B),
                    np.int32(N),
                    np.int32(D),
                    np.int32(qs),
                    np.int32(C),
                    scale,
                )
            )
            t["softmax"] += ms(
                ks["softmax_rows"](q, (B * C,), None, blogits, battn, np.int32(B * C), np.int32(N))
            )
            t["reduce"] += ms(
                ks["reduce_chunk_to_imp"](
                    q, (B * N,), None, battn, bimp, np.int32(B), np.int32(N), np.int32(C)
                )
            )
        t["total"] = sum(t.values())
        t["peak_logits_MiB"] = nbytes_logits_peak / (1024**2)
        # bytes touched approx: norm read+write x; each chunk read x for QK ~ 2*C*N*D + write C*N
        # rough traffic proxy for QK alone (dominant):
        n_chunks = (N + chunk - 1) // chunk
        t["qk_bytes_proxy_GiB"] = n_chunks * (2.0 * chunk * N * D * 4) / (1024**3)  # upper-ish
        return t

    for _ in range(warmup):
        once()
    runs = [once() for _ in range(iters)]
    best = min(runs, key=lambda r: r["total"])
    return best


def profile_fusion_stub(ctx, q, ks, B, N_v, N_imu, D, H, warmup=1, iters=3):
    """Cheap stand-in for projected CA: gemm-less QKV already assumed; time sdpa_cross + gate."""
    rng = np.random.default_rng(1)
    Dh = D // H
    Q = rng.standard_normal((B, H, N_v, Dh), dtype=np.float32)
    K = rng.standard_normal((B, H, N_imu, Dh), dtype=np.float32)
    V = rng.standard_normal((B, H, N_imu, Dh), dtype=np.float32)
    z = rng.standard_normal((B * N_v * D,), dtype=np.float32)
    delta = rng.standard_normal(z.shape, dtype=np.float32)
    g = rng.standard_normal(z.shape, dtype=np.float32)
    scale = np.float32(Dh**-0.5)
    mf = cl.mem_flags

    def once():
        bQ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q)
        bK = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K)
        bV = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V)
        bO = cl.Buffer(ctx, mf.READ_WRITE, size=Q.nbytes)
        t = {}
        t["sdpa_cross"] = ms(
            ks["sdpa_cross_rows"](
                q,
                (B * H * N_v,),
                None,
                bQ,
                bK,
                bV,
                bO,
                np.int32(B),
                np.int32(H),
                np.int32(N_v),
                np.int32(N_imu),
                np.int32(Dh),
                scale,
            )
        )
        bz = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z)
        bd = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta)
        bg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=g)
        bo = cl.Buffer(ctx, mf.READ_WRITE, size=z.nbytes)
        t["gated_residual"] = ms(
            ks["gated_residual"](q, (z.size,), None, bz, bd, bg, bo, np.int32(z.size))
        )
        t["total"] = t["sdpa_cross"] + t["gated_residual"]
        return t

    for _ in range(warmup):
        once()
    runs = [once() for _ in range(iters)]
    return min(runs, key=lambda r: r["total"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=64, help="profile D (use 64/128 on CPU; 1024 on GPU later)")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    ctx, q, ks = build()
    Ns = [128, 256, 512, 1024, 2048]
    rows = []
    print("\n=== Idea1: receive-importance scaling ===")
    print(f"{'N':>6} {'total':>10} {'qk':>10} {'softmax':>10} {'reduce':>8} {'norm':>8} {'qk%':>6} {'peakMiB':>8}")
    for N in Ns:
        t = profile_receive(ctx, q, ks, B=1, N=N, D=args.D, chunk=args.chunk)
        t["N"] = N
        t["D"] = args.D
        t["chunk"] = args.chunk
        rows.append({"kind": "receive", **t})
        print(
            f"{N:6d} {t['total']:10.2f} {t['qk']:10.2f} {t['softmax']:10.2f} {t['reduce']:8.2f} "
            f"{t['norm']:8.2f} {100*t['qk']/t['total']:5.1f}% {t['peak_logits_MiB']:8.2f}"
        )

    # slope estimate total vs N^2
    if len(rows) >= 2:
        n0, n1 = rows[0]["N"], rows[-1]["N"]
        t0, t1 = rows[0]["total"], rows[-1]["total"]
        ratio_t = t1 / max(t0, 1e-9)
        ratio_n2 = (n1 / n0) ** 2
        print(f"\nscaling: N {n0}->{n1} (N^2 x{ratio_n2:.1f}), time x{ratio_t:.1f}  (ideal mem-bound QK ~N^2)")

    print("\n=== Idea2 context: fusion CA stub (video<-IMU) ===")
    print(f"{'Nv':>6} {'Nimu':>6} {'sdpa_x':>10} {'gate':>10} {'total':>10} {'vs_recv@Nv':>12}")
    fusion_rows = []
    recv_by_n = {r["N"]: r["total"] for r in rows}
    for Nv, Nimu in [(256, 26), (512, 26), (1024, 26), (2048, 26), (512, 64)]:
        # Use D divisible by H=4
        D = args.D
        H = 4
        if D % H:
            continue
        f = profile_fusion_stub(ctx, q, ks, B=1, N_v=Nv, N_imu=Nimu, D=D, H=H)
        f.update({"Nv": Nv, "Nimu": Nimu, "D": D})
        fusion_rows.append({"kind": "fusion", **f})
        recv = recv_by_n.get(Nv, float("nan"))
        print(
            f"{Nv:6d} {Nimu:6d} {f['sdpa_cross']:10.2f} {f['gated_residual']:10.2f} {f['total']:10.2f} "
            f"{recv:12.2f}"
        )

    print(
        """
Interpretation (idea1/2):
  - If qk% dominates receive-importance, materializing [C,N] logits is the main tax.
  - If softmax+reduce together are large, need online column-accumulate (no attn buffer).
  - If fusion total << receive@same Nv, idea2 should REUSE CA intermediates for scoring
    rather than paying a second full self-attn receive pass.
"""
    )

    out = {
        "device": "POCL CPU (login)",
        "note": "Relative scaling for design; re-run on NVIDIA OpenCL for absolute GPU numbers.",
        "receive": rows,
        "fusion": fusion_rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    else:
        out_path = HERE / "profile_idea12_latest.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
