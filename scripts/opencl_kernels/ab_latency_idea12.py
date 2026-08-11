#!/usr/bin/env python3
"""A/B latency for Idea1 (fused receive) and Idea2 (fusion-aware scores).

Methods timed end-to-end (score tensor ready for topk):
  baseline      — chunked qk+softmax+reduce (current OpenCL)
  idea1         — fused receive, no logits buffers (math ≈ baseline)
  idea2_replace — only CA path scores (||O||^2 over heads); NO self-receive
  idea1+2_hybrid— alpha*idea1 + beta*ca_score (pays both; accuracy-oriented)
  idea1+2_light — CA scores + strided fused self (stride>1)

Device: POCL CPU on login node. Numbers are relative; re-run on GPU for serving nums.
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
    src = "\n".join(
        [
            (HERE / "receive_importance.cl").read_text(),
            (HERE / "encoder_block.cl").read_text(),
            (HERE / "idea12_fused.cl").read_text(),
            (HERE / "prune_ops.cl").read_text(),
        ]
    )
    prg = cl.Program(ctx, src).build(options="-cl-std=CL1.2")
    names = [
        "l2_normalize_rows",
        "qk_logits_chunk",
        "softmax_rows",
        "reduce_chunk_to_imp",
        "softmax_rows_inplace",
        "receive_importance_fused",
        "receive_importance_fused_strided",
        "sdpa_cross_rows",
        "ca_output_l2_score",
        "mix_scores",
        "fill_recency",
        "apply_recency",
        "gated_residual",
    ]
    ks = {n: cl.Kernel(prg, n) for n in names}
    print(f"device: {plat.name} | {dev.name}")
    return ctx, q, ks


def ms(ev):
    ev.wait()
    return 1e-6 * (ev.profile.end - ev.profile.start)


class Runner:
    def __init__(self, ctx, q, ks, B, N, D, Nimu, H, chunk, recency_strength=1.0, gp=8):
        self.ctx, self.q, self.ks = ctx, q, ks
        self.B, self.N, self.D = B, N, D
        self.Nimu, self.H = Nimu, H
        self.Dh = D // H
        assert D % H == 0
        self.chunk = chunk
        self.recency_strength = recency_strength
        self.gp = gp
        self.mf = cl.mem_flags
        rng = np.random.default_rng(0)
        self.x = rng.standard_normal((B, N, D), dtype=np.float32)
        self.Q = rng.standard_normal((B, H, N, self.Dh), dtype=np.float32)
        self.K = rng.standard_normal((B, H, Nimu, self.Dh), dtype=np.float32)
        self.V = rng.standard_normal((B, H, Nimu, self.Dh), dtype=np.float32)
        self.scale_d = np.float32(D**-0.5)
        self.scale_h = np.float32(self.Dh**-0.5)

    def _bufs_x(self):
        bin_ = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.x)
        bx = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=self.x.nbytes)
        return bin_, bx

    def normalize(self, bin_, bx):
        return ms(
            self.ks["l2_normalize_rows"](
                self.q, (self.B * self.N,), None, bin_, bx, np.int32(self.B * self.N), np.int32(self.D)
            )
        )

    def recency_ms(self, bscore):
        brec = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=self.N * 4)
        t0 = ms(
            self.ks["fill_recency"](
                self.q, (self.N,), None, brec, np.int32(self.N), np.int32(self.gp)
            )
        )
        t1 = ms(
            self.ks["apply_recency"](
                self.q,
                (self.B * self.N,),
                None,
                bscore,
                brec,
                np.int32(self.B),
                np.int32(self.N),
                np.float32(self.recency_strength),
            )
        )
        return t0 + t1

    def baseline(self):
        B, N, D, chunk = self.B, self.N, self.D, self.chunk
        bin_, bx = self._bufs_x()
        bimp = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * N * 4)
        cl.enqueue_fill_buffer(self.q, bimp, np.float32(0), 0, B * N * 4).wait()
        t = {"norm": self.normalize(bin_, bx), "qk": 0.0, "sm": 0.0, "red": 0.0}
        max_C = min(chunk, N)
        blogits = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * max_C * N * 4)
        battn = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * max_C * N * 4)
        for qs in range(0, N, chunk):
            C = min(chunk, N - qs)
            t["qk"] += ms(
                self.ks["qk_logits_chunk"](
                    self.q,
                    (B * C * N,),
                    None,
                    bx,
                    blogits,
                    np.int32(B),
                    np.int32(N),
                    np.int32(D),
                    np.int32(qs),
                    np.int32(C),
                    self.scale_d,
                )
            )
            t["sm"] += ms(
                self.ks["softmax_rows"](
                    self.q, (B * C,), None, blogits, battn, np.int32(B * C), np.int32(N)
                )
            )
            t["red"] += ms(
                self.ks["reduce_chunk_to_imp"](
                    self.q, (B * N,), None, battn, bimp, np.int32(B), np.int32(N), np.int32(C)
                )
            )
        t["recency"] = self.recency_ms(bimp)
        t["total"] = sum(t.values())
        return t

    def idea1(self):
        """Idea1 memory fusion: single C×N buffer (in-place softmax) + reduce; no attn buffer.
        Math identical to baseline. Avoids writing a second C×N attn tensor.
        """
        B, N, D, chunk = self.B, self.N, self.D, self.chunk
        bin_, bx = self._bufs_x()
        bimp = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * N * 4)
        cl.enqueue_fill_buffer(self.q, bimp, np.float32(0), 0, B * N * 4).wait()
        t = {"norm": self.normalize(bin_, bx), "qk": 0.0, "sm": 0.0, "red": 0.0}
        max_C = min(chunk, N)
        blogits = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * max_C * N * 4)
        for qs in range(0, N, chunk):
            C = min(chunk, N - qs)
            t["qk"] += ms(
                self.ks["qk_logits_chunk"](
                    self.q,
                    (B * C * N,),
                    None,
                    bx,
                    blogits,
                    np.int32(B),
                    np.int32(N),
                    np.int32(D),
                    np.int32(qs),
                    np.int32(C),
                    self.scale_d,
                )
            )
            t["sm"] += ms(
                self.ks["softmax_rows_inplace"](
                    self.q, (B * C,), None, blogits, np.int32(B * C), np.int32(N)
                )
            )
            t["red"] += ms(
                self.ks["reduce_chunk_to_imp"](
                    self.q, (B * N,), None, blogits, bimp, np.int32(B), np.int32(N), np.int32(C)
                )
            )
        t["recency"] = self.recency_ms(bimp)
        t["total"] = sum(t.values())
        return t

    def idea1_naive_fused(self):
        """Fully fused no-buffer path (often slower on CPU due to atomics + recompute)."""
        B, N, D = self.B, self.N, self.D
        bin_, bx = self._bufs_x()
        bimp = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * N * 4)
        cl.enqueue_fill_buffer(self.q, bimp, np.float32(0), 0, B * N * 4).wait()
        t = {"norm": self.normalize(bin_, bx)}
        t["fused"] = ms(
            self.ks["receive_importance_fused"](
                self.q,
                (B * N,),
                None,
                bx,
                bimp,
                np.int32(B),
                np.int32(N),
                np.int32(D),
                self.scale_d,
            )
        )
        t["recency"] = self.recency_ms(bimp)
        t["total"] = sum(t.values())
        return t

    def _ca_score(self):
        B, H, N, Nimu, Dh = self.B, self.H, self.N, self.Nimu, self.Dh
        bQ = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.Q)
        bK = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.K)
        bV = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.V)
        bO = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=self.Q.nbytes)
        t_ca = ms(
            self.ks["sdpa_cross_rows"](
                self.q,
                (B * H * N,),
                None,
                bQ,
                bK,
                bV,
                bO,
                np.int32(B),
                np.int32(H),
                np.int32(N),
                np.int32(Nimu),
                np.int32(Dh),
                self.scale_h,
            )
        )
        bscore = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * N * 4)
        cl.enqueue_fill_buffer(self.q, bscore, np.float32(0), 0, B * N * 4).wait()
        t_sc = ms(
            self.ks["ca_output_l2_score"](
                self.q,
                (B * H * N,),
                None,
                bO,
                bscore,
                np.int32(B),
                np.int32(H),
                np.int32(N),
                np.int32(Dh),
            )
        )
        return bscore, t_ca + t_sc, {"ca": t_ca, "ca_score": t_sc}

    def idea2_replace(self):
        """Replace self-receive with CA-derived scores (+recency)."""
        bscore, t_main, parts = self._ca_score()
        t = dict(parts)
        t["recency"] = self.recency_ms(bscore)
        t["total"] = t_main + t["recency"]
        return t

    def idea12_hybrid(self, alpha=1.0, beta=1.0):
        """Idea1 single-buffer self + CA scores mixed (pays nearly full self + CA)."""
        t = self.idea1()
        # idea1 already applied recency on self only; redo with mix
        B, N, D, chunk = self.B, self.N, self.D, self.chunk
        bin_, bx = self._bufs_x()
        bself = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * N * 4)
        cl.enqueue_fill_buffer(self.q, bself, np.float32(0), 0, B * N * 4).wait()
        t = {"norm": self.normalize(bin_, bx), "qk": 0.0, "sm": 0.0, "red": 0.0}
        max_C = min(chunk, N)
        blogits = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * max_C * N * 4)
        for qs in range(0, N, chunk):
            C = min(chunk, N - qs)
            t["qk"] += ms(
                self.ks["qk_logits_chunk"](
                    self.q,
                    (B * C * N,),
                    None,
                    bx,
                    blogits,
                    np.int32(B),
                    np.int32(N),
                    np.int32(D),
                    np.int32(qs),
                    np.int32(C),
                    self.scale_d,
                )
            )
            t["sm"] += ms(
                self.ks["softmax_rows_inplace"](
                    self.q, (B * C,), None, blogits, np.int32(B * C), np.int32(N)
                )
            )
            t["red"] += ms(
                self.ks["reduce_chunk_to_imp"](
                    self.q, (B * N,), None, blogits, bself, np.int32(B), np.int32(N), np.int32(C)
                )
            )
        bca, _, parts = self._ca_score()
        t.update(parts)
        bout = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * N * 4)
        t["mix"] = ms(
            self.ks["mix_scores"](
                self.q,
                (B * N,),
                None,
                bself,
                bca,
                bout,
                np.int32(B * N),
                np.float32(alpha),
                np.float32(beta),
            )
        )
        t["recency"] = self.recency_ms(bout)
        t["total"] = sum(v for k, v in t.items() if k != "total")
        return t

    def idea12_light(self, stride=4, alpha=1.0, beta=1.0):
        """CA scores + strided self (query subsample via chunked qk, math approx)."""
        B, N, D = self.B, self.N, self.D
        bin_, bx = self._bufs_x()
        bself = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * N * 4)
        cl.enqueue_fill_buffer(self.q, bself, np.float32(0), 0, B * N * 4).wait()
        t = {"norm": self.normalize(bin_, bx), "qk": 0.0, "sm": 0.0, "red": 0.0}
        blogits = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * 1 * N * 4)
        for qs in range(0, N, stride):
            C = 1
            t["qk"] += ms(
                self.ks["qk_logits_chunk"](
                    self.q,
                    (B * C * N,),
                    None,
                    bx,
                    blogits,
                    np.int32(B),
                    np.int32(N),
                    np.int32(D),
                    np.int32(qs),
                    np.int32(C),
                    self.scale_d,
                )
            )
            t["sm"] += ms(
                self.ks["softmax_rows_inplace"](
                    self.q, (B * C,), None, blogits, np.int32(B * C), np.int32(N)
                )
            )
            t["red"] += ms(
                self.ks["reduce_chunk_to_imp"](
                    self.q, (B * N,), None, blogits, bself, np.int32(B), np.int32(N), np.int32(C)
                )
            )
        # scale mass ~ stride so total sum ~ N
        # optional host side would multiply; cheap mix handles alpha
        bca, _, parts = self._ca_score()
        t.update(parts)
        bout = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * N * 4)
        t["mix"] = ms(
            self.ks["mix_scores"](
                self.q,
                (B * N,),
                None,
                bself,
                bca,
                bout,
                np.int32(B * N),
                np.float32(alpha * float(stride)),
                np.float32(beta),
            )
        )
        t["recency"] = self.recency_ms(bout)
        t["total"] = sum(v for k, v in t.items() if k != "total")
        return t

    def check_idea1_correct(self):
        """Compare idea1 single-buffer path vs numpy reference."""
        B, N, D = self.B, self.N, self.D
        xn = self.x / np.linalg.norm(self.x, axis=-1, keepdims=True).clip(1e-12)
        sc = D**-0.5
        imp_ref = np.zeros((B, N), np.float32)
        for i in range(N):
            logits = (xn[:, i : i + 1] @ xn.transpose(0, 2, 1))[:, 0] * sc
            m = logits.max(-1, keepdims=True)
            a = np.exp(logits - m)
            a /= a.sum(-1, keepdims=True)
            imp_ref += a

        # run idea1 into buffer
        chunk = self.chunk
        bin_, bx = self._bufs_x()
        bimp = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * N * 4)
        cl.enqueue_fill_buffer(self.q, bimp, np.float32(0), 0, B * N * 4).wait()
        self.normalize(bin_, bx)
        max_C = min(chunk, N)
        blogits = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=B * max_C * N * 4)
        for qs in range(0, N, chunk):
            C = min(chunk, N - qs)
            self.ks["qk_logits_chunk"](
                self.q,
                (B * C * N,),
                None,
                bx,
                blogits,
                np.int32(B),
                np.int32(N),
                np.int32(D),
                np.int32(qs),
                np.int32(C),
                self.scale_d,
            ).wait()
            self.ks["softmax_rows_inplace"](
                self.q, (B * C,), None, blogits, np.int32(B * C), np.int32(N)
            ).wait()
            self.ks["reduce_chunk_to_imp"](
                self.q, (B * N,), None, blogits, bimp, np.int32(B), np.int32(N), np.int32(C)
            ).wait()
        got = np.empty((B, N), np.float32)
        cl.enqueue_copy(self.q, got, bimp).wait()
        return float(np.max(np.abs(got - imp_ref)))


def best_of(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    runs = [fn() for _ in range(iters)]
    return min(runs, key=lambda r: r["total"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--Nimu", type=int, default=26)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--Ns", type=str, default="256,512,1024,2048")
    ap.add_argument("--out", type=str, default=str(HERE / "ab_idea12_latency.json"))
    args = ap.parse_args()

    ctx, q, ks = build()
    Ns = [int(x) for x in args.Ns.split(",") if x.strip()]
    results = []

    # correctness once
    r = Runner(ctx, q, ks, B=1, N=128, D=args.D, Nimu=args.Nimu, H=args.H, chunk=args.chunk)
    err = r.check_idea1_correct()
    print(f"idea1 fused vs numpy max_abs_err @N=128: {err:.3e}")

    header = f"{'N':>6} {'baseline':>10} {'idea1':>10} {'i2_repl':>10} {'i12_hyb':>10} {'i12_light':>10} | speedups vs baseline"
    print(header)
    print("-" * len(header))

    for N in Ns:
        R = Runner(ctx, q, ks, B=1, N=N, D=args.D, Nimu=args.Nimu, H=args.H, chunk=args.chunk)
        base = best_of(R.baseline, args.iters, args.warmup)
        i1 = best_of(R.idea1, args.iters, args.warmup)
        i2 = best_of(R.idea2_replace, args.iters, args.warmup)
        hy = best_of(lambda: R.idea12_hybrid(1.0, 1.0), args.iters, args.warmup)
        lt = best_of(lambda: R.idea12_light(args.stride, 1.0, 1.0), args.iters, args.warmup)

        def sp(t):
            return base["total"] / max(t["total"], 1e-9)

        row = {
            "N": N,
            "D": args.D,
            "Nimu": args.Nimu,
            "baseline_ms": base["total"],
            "idea1_ms": i1["total"],
            "idea2_replace_ms": i2["total"],
            "idea12_hybrid_ms": hy["total"],
            "idea12_light_ms": lt["total"],
            "speedup_idea1": sp(i1),
            "speedup_idea2_replace": sp(i2),
            "speedup_idea12_hybrid": sp(hy),
            "speedup_idea12_light": sp(lt),
            "detail": {"baseline": base, "idea1": i1, "idea2": i2, "hybrid": hy, "light": lt},
        }
        results.append(row)
        print(
            f"{N:6d} {base['total']:10.2f} {i1['total']:10.2f} {i2['total']:10.2f} "
            f"{hy['total']:10.2f} {lt['total']:10.2f} | "
            f"i1={sp(i1):5.2f}x i2={sp(i2):5.2f}x hyb={sp(hy):5.2f}x light={sp(lt):5.2f}x"
        )

    out = {
        "device": "POCL CPU login node",
        "disclaimer": "Relative latency only. idea2_replace changes scorer (accuracy may change). "
        "idea1 is math-equivalent (fp error). GPU numbers will differ.",
        "idea1_correctness_err_N128": err,
        "rows": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}")

    # summary for humans
    if results:
        r = results[-1]
        print("\n=== Summary @ largest N ===")
        print(f"N={r['N']} D={r['D']}")
        print(f"  baseline (chunked self receive):     {r['baseline_ms']:.2f} ms")
        print(f"  Idea1 fused (same scores):           {r['idea1_ms']:.2f} ms  ({r['speedup_idea1']:.2f}x)")
        print(f"  Idea2 replace (CA scores only):      {r['idea2_replace_ms']:.2f} ms  ({r['speedup_idea2_replace']:.2f}x)  [accuracy may change]")
        print(f"  Idea1+2 hybrid (full self+CA):       {r['idea12_hybrid_ms']:.2f} ms  ({r['speedup_idea12_hybrid']:.2f}x)")
        print(f"  Idea1+2 light (CA+self/stride):      {r['idea12_light_ms']:.2f} ms  ({r['speedup_idea12_light']:.2f}x)")


if __name__ == "__main__":
    main()
