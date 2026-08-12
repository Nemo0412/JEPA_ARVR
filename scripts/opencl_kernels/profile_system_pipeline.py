#!/usr/bin/env python3
"""System-level OpenCL pipeline for V-JEPA fusion (HD-EPIC tri-modal / Ego4D bi-modal).

Matches ``ProjectedTriModalCrossAttention`` in
``app/hdepic_lora_action_anticipation/tri_modal_fusion.py``:

  --dataset hdepic (default, video/gaze/IMU):
    gaze:  Q=Z_g, KV=concat(Z_v, Z_i)
    imu:   Q=Z_i, KV=concat(Z_v, Z_g)
    video: Q=Z_v, KV=concat(Z_g, Z_i)

  --dataset ego4d (no IMU; gaze-only aux):
    gaze:  Q=Z_g, KV=Z_v
    video: Q=Z_v, KV=Z_g   # Gaze as K/V; do NOT concat IMU

  each update = QKV GEMMs + sdpa_cross + Wo + gate MLP + gated residual
               (+ optional FFN: LN → Linear→GELU→Linear)

Plus a V-JEPA encoder-block stub on video tokens and postfuse prune.

Token budgets default to compute_token_budgets() ratios
(n_gaze=grid², n_imu≈0.1·n_video). Use --vitl-tokens for ViT-L spatial
(Nv=256, Ng=100, Ni=26; Ego4D drops Ni). D/H are scaled-down OpenCL
stand-ins (not 1024/16).

Reports (steady-state, buffers resident):
  - sum launch (queued→start) across all kernel launches
  - sum device (start→end) across all kernels
  - system H2D / D2H once per tick (cold) vs 0 for steady
  - bound label: IO-bound / launch-bound / device-bound (+ rough AI note)

Run:
  .../python scripts/opencl_kernels/profile_system_pipeline.py --cpu
  .../python scripts/opencl_kernels/profile_system_pipeline.py --dataset ego4d --vitl-tokens
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyopencl as cl

HERE = Path(__file__).resolve().parent
import sys

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from kernel_roofline import attach, kernel_cost, measure_peaks


def ns_ms(x: float) -> float:
    return 1e-6 * float(x)


def ev_parts(ev: cl.Event) -> tuple[float, float]:
    """Return (launch_path_ms, device_ms)."""
    ev.wait()
    return ns_ms(ev.profile.start - ev.profile.queued), ns_ms(ev.profile.end - ev.profile.start)


def med(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


@dataclass
class LaunchRec:
    name: str
    launch_ms: float
    device_ms: float
    flops: int = 0
    bytes: int = 0


@dataclass
class TickReport:
    mode: str  # cold_io | steady
    wall_ms: float
    h2d_ms: float
    d2h_ms: float
    io_ms: float
    launch_ms: float
    device_ms: float
    n_launches: int
    launches: list[dict]
    bound: str
    io_frac: float
    launch_frac: float
    device_frac: float
    bytes_h2d: int
    bytes_d2h: int
    h2d_gbs: float
    d2h_gbs: float


def classify(io_ms: float, launch_ms: float, device_ms: float) -> tuple[str, float, float, float]:
    tot = max(io_ms + launch_ms + device_ms, 1e-12)
    io_f, la_f, de_f = io_ms / tot, launch_ms / tot, device_ms / tot
    # Dominant bucket; also flag near-ties.
    order = sorted(
        [("IO-bound (host↔device)", io_f), ("launch-bound", la_f), ("device-bound (compute+on-device mem)", de_f)],
        key=lambda x: -x[1],
    )
    bound = order[0][0]
    if order[0][1] < 0.45:
        bound = f"mixed ({order[0][0].split()[0]}/{order[1][0].split()[0]})"
    return bound, io_f, la_f, de_f


class Pipeline:
    def __init__(
        self,
        prefer_gpu: bool,
        Nv: int,
        Ng: int,
        Ni: int,
        D: int,
        H: int,
        Kkeep: int,
        chunk: int,
        layers: int = 1,
        use_ffn: bool = False,
        ffn_mult: int = 4,
        dataset: str = "hdepic",
    ):
        # N = video tokens (encoder + postfuse prune).
        # hdepic: Nv / Ng / Ni ; ego4d: Nv / Ng (no IMU).
        self.dataset = str(dataset).lower().strip()
        if self.dataset not in ("hdepic", "ego4d"):
            raise ValueError(f"dataset must be 'hdepic' or 'ego4d', got {dataset!r}")
        self.use_imu = self.dataset == "hdepic"
        self.Nv, self.Ng = int(Nv), int(Ng)
        self.Ni = int(Ni) if self.use_imu else 0
        self.N = self.Nv
        self.D, self.H = D, H
        self.K, self.chunk = Kkeep, chunk
        self.layers = int(layers)
        self.use_ffn = bool(use_ffn)
        self.ffn_mult = int(ffn_mult)
        self.B = 1
        self.Dh = D // H
        assert D % H == 0 and self.Dh % 2 == 0
        assert self.Ng >= 1 and self.Nv >= 1
        if self.use_imu:
            assert self.Ni >= 1

        plats = cl.get_platforms()
        chosen = None
        if prefer_gpu:
            for p in plats:
                for d in p.get_devices():
                    name = d.name.lower()
                    if (d.type & cl.device_type.GPU) or any(k in name for k in ("nvidia", "cuda", "a100", "h100", "l40")):
                        chosen = (p, d)
                        break
                if chosen:
                    break
        else:
            for p in plats:
                for d in p.get_devices():
                    if (d.type & cl.device_type.CPU) or "cpu" in d.name.lower():
                        chosen = (p, d)
                        break
                if chosen:
                    break
        if chosen is None:
            chosen = (plats[0], plats[0].get_devices()[0])
        plat, dev = chosen
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        src = "\n".join(
            (HERE / f).read_text()
            for f in ("encoder_block.cl", "receive_importance.cl", "prune_ops.cl")
        )
        prg = cl.Program(self.ctx, src).build(options="-cl-std=CL1.2")
        names = [
            "layernorm_rows",
            "gemm_nn",
            "rope_apply",
            "sdpa_rows",
            "sdpa_cross_rows",
            "gelu",
            "gated_residual",
            "residual_add",
            "l2_normalize_rows",
            "qk_logits_chunk",
            "softmax_rows",
            "reduce_chunk_to_imp",
            "fill_recency",
            "recency_reweight",
            "topk_indices_row",
            "gather_tokens",
        ]
        self.k = {n: cl.Kernel(prg, n) for n in names}
        self.device_label = f"{plat.name} | {dev.name} | {cl.device_type.to_string(dev.type)}"
        print(f"OpenCL: {self.device_label}")
        if self.use_imu:
            path = "ProjectedTriModalCrossAttention video/gaze/IMU"
            print(
                f"V-JEPA tri-modal ({self.dataset}): Nv={self.Nv} Ng={self.Ng} Ni={self.Ni} "
                f"D={D} H={H} layers={self.layers} ffn={self.use_ffn} ({path})"
            )
        else:
            path = "ProjectedTriModalCrossAttention video/gaze (Ego4D; KV=gaze only)"
            print(
                f"V-JEPA bi-modal ({self.dataset}): Nv={self.Nv} Ng={self.Ng} Ni=0 "
                f"D={D} H={H} layers={self.layers} ffn={self.use_ffn} ({path})"
            )
        self.path_label = path

        rng = np.random.default_rng(0)
        f4 = np.float32
        B, N, D, H, Dh, K = self.B, self.N, D, H, self.Dh, Kkeep
        Nv, Ng, Ni = self.Nv, self.Ng, self.Ni

        self.host = {
            "z_v": rng.standard_normal((Nv, D), dtype=f4),
            "z_g": rng.standard_normal((Ng, D), dtype=f4),
            "gamma": rng.standard_normal((D,), dtype=f4),
            "beta": rng.standard_normal((D,), dtype=f4),
            "W_qkv": rng.standard_normal((D, D), dtype=f4),
            "W_out": rng.standard_normal((D, D), dtype=f4),
            "W_mlp": rng.standard_normal((D, D), dtype=f4),
            "Wq": rng.standard_normal((D, D), dtype=f4),
            "Wk": rng.standard_normal((D, D), dtype=f4),
            "Wv": rng.standard_normal((D, D), dtype=f4),
            "Wo": rng.standard_normal((D, D), dtype=f4),
            "Wg1": rng.standard_normal((D, D), dtype=f4),
            "Wg2": rng.standard_normal((D, D), dtype=f4),
            "Wff1": rng.standard_normal((D, D * self.ffn_mult), dtype=f4),
            "Wff2": rng.standard_normal((D * self.ffn_mult, D), dtype=f4),
            "Q": rng.standard_normal((B, H, N, Dh), dtype=f4),
            "K": rng.standard_normal((B, H, N, Dh), dtype=f4),
            "V": rng.standard_normal((B, H, N, Dh), dtype=f4),
            "pos": np.tile(np.arange(N, dtype=f4), B * H),
        }
        if self.use_imu:
            self.host["z_i"] = rng.standard_normal((Ni, D), dtype=f4)
        self.bytes_h2d = int(sum(a.nbytes for a in self.host.values()))
        self.bytes_d2h = int(B * K * D * 4 + B * K * 4)

        mf = cl.mem_flags
        self.dev = {}
        for name, arr in self.host.items():
            self.dev[name] = cl.Buffer(self.ctx, mf.READ_WRITE, size=arr.nbytes)

        if self.use_imu:
            max_nk = max(Nv + Ni, Nv + Ng, Ng + Ni)
            max_nq = max(Nv, Ng, Ni)
        else:
            max_nk = max(Nv, Ng)
            max_nq = max(Nv, Ng)
        hid = D * self.ffn_mult
        self.dev["x"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=Nv * D * 4)
        self.dev["y"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max(Nv, max_nq) * D * 4)
        self.dev["tmp"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max(Nv, max_nq) * max(D, hid) * 4)
        self.dev["q_flat"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=B * H * N * Dh * 4)
        self.dev["O"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=B * H * N * Dh * 4)
        self.dev["Qb"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nq * D * 4)
        self.dev["Ob"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nq * D * 4)
        self.dev["k_tmp"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nk * D * 4)
        self.dev["v_tmp"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nk * D * 4)
        self.dev["k_cat"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nk * D * 4)
        self.dev["v_cat"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nk * D * 4)
        self.dev["fused"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nq * D * 4)
        self.dev["t1"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nq * D * 4)
        self.dev["t2"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nq * D * 4)
        self.dev["h"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nq * hid * 4)
        self.dev["glog"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=max_nq * D * 4)
        self.dev["xn"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=N * D * 4)
        max_C = min(chunk, N)
        self.dev["logits"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=B * max_C * N * 4)
        self.dev["attn"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=B * max_C * N * 4)
        self.dev["imp"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=B * N * 4)
        self.dev["rec"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=N * 4)
        self.dev["idx"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=B * K * 4)
        self.dev["gathered"] = cl.Buffer(self.ctx, mf.READ_WRITE, size=B * K * D * 4)

        self.host_out_tok = np.empty((B, K, D), dtype=f4)
        self.host_out_idx = np.empty((B, K), dtype=np.int32)
        self.dev["zeros_imp"] = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros((B, N), dtype=f4)
        )

    def _h2d_all(self) -> float:
        t = 0.0
        for name, arr in self.host.items():
            ev = cl.enqueue_copy(self.queue, self.dev[name], arr, is_blocking=False)
            ev.wait()
            t += ns_ms(ev.profile.end - ev.profile.start)
        ev = cl.enqueue_copy(self.queue, self.dev["x"], self.host["z_v"], is_blocking=False)
        ev.wait()
        t += ns_ms(ev.profile.end - ev.profile.start)
        return t

    def _d2h_outs(self) -> float:
        t = 0.0
        ev = cl.enqueue_copy(self.queue, self.host_out_tok, self.dev["gathered"], is_blocking=False)
        ev.wait()
        t += ns_ms(ev.profile.end - ev.profile.start)
        ev = cl.enqueue_copy(self.queue, self.host_out_idx, self.dev["idx"], is_blocking=False)
        ev.wait()
        t += ns_ms(ev.profile.end - ev.profile.start)
        return t

    def _run_kernels(self) -> list[LaunchRec]:
        """Full on-device pipeline; returns per-launch timing."""
        B, N, D, H, Dh, K, chunk = self.B, self.N, self.D, self.H, self.Dh, self.K, self.chunk
        Nv, Ng, Ni = self.Nv, self.Ng, self.Ni
        i4, f4 = np.int32, np.float32
        scale = f4(Dh**-0.5)
        sc = f4(D**-0.5)
        recs: list[LaunchRec] = []
        k = self.k
        d = self.dev

        def launch(name, gsize, args, **cost_kw):
            ev = k[name](self.queue, gsize, None, *args)
            la, de = ev_parts(ev)
            flops, nbytes = kernel_cost(name, **cost_kw)
            recs.append(LaunchRec(name, la, de, flops, nbytes))

        def d2d(tag, dst, src, nbytes, dst_offset=0, src_offset=0):
            ev = cl.enqueue_copy(
                self.queue,
                dst,
                src,
                dst_offset=int(dst_offset),
                src_offset=int(src_offset),
                byte_count=int(nbytes),
            )
            la, de = ev_parts(ev)
            flops, nb = kernel_cost(tag, nbytes=nbytes)
            recs.append(LaunchRec(tag, la, de, flops, nb))

        def gemm(a, w, c, m, n, kk, beta=0.0):
            launch(
                "gemm_nn",
                (m * n,),
                (a, w, c, i4(m), i4(n), i4(kk), f4(1), f4(beta)),
                M=m,
                N=n,
                K=kk,
                beta=beta,
            )

        def branch_update(z_q, nq, auxs):
            """One ProjectedCrossAttentionUpdate: Q=z_q, KV=concat(auxs)."""
            gemm(z_q, d["Wq"], d["Qb"], nq, D, D)
            off = 0
            nk = 0
            for z_aux, na in auxs:
                gemm(z_aux, d["Wk"], d["k_tmp"], na, D, D)
                d2d("d2d_copy_fuse", d["k_cat"], d["k_tmp"], na * D * 4, dst_offset=off)
                gemm(z_aux, d["Wv"], d["v_tmp"], na, D, D)
                d2d("d2d_copy_fuse", d["v_cat"], d["v_tmp"], na * D * 4, dst_offset=off)
                off += na * D * 4
                nk += na
            launch(
                "sdpa_cross_rows",
                (B * H * nq,),
                (d["Qb"], d["k_cat"], d["v_cat"], d["Ob"], i4(B), i4(H), i4(nq), i4(nk), i4(Dh), scale),
                B=B,
                H=H,
                Nq=nq,
                Nk=nk,
                Dh=Dh,
            )
            gemm(d["Ob"], d["Wo"], d["fused"], nq, D, D)
            # gate MLP: GELU(z@Wg1 + fused@Wg1) @ Wg2 → logits; residual
            gemm(z_q, d["Wg1"], d["t1"], nq, D, D)
            gemm(d["fused"], d["Wg1"], d["t2"], nq, D, D)
            launch("residual_add", (nq * D,), (d["t1"], d["t2"], d["h"], i4(nq * D)), n=nq * D)
            launch("gelu", (nq * D,), (d["h"], d["t1"], i4(nq * D)), n=nq * D)
            gemm(d["t1"], d["Wg2"], d["glog"], nq, D, D)
            launch(
                "gated_residual",
                (nq * D,),
                (z_q, d["fused"], d["glog"], d["y"], i4(nq * D)),
                n=nq * D,
            )
            d2d("d2d_copy_fuse", z_q, d["y"], nq * D * 4)
            if self.use_ffn:
                hid = D * self.ffn_mult
                launch(
                    "layernorm_rows",
                    (nq,),
                    (z_q, d["t1"], d["gamma"], d["beta"], i4(nq), i4(D), f4(1e-6)),
                    rows=nq,
                    D=D,
                )
                gemm(d["t1"], d["Wff1"], d["h"], nq, hid, D)
                launch("gelu", (nq * hid,), (d["h"], d["tmp"], i4(nq * hid)), n=nq * hid)
                gemm(d["tmp"], d["Wff2"], d["t2"], nq, D, hid)
                launch("residual_add", (nq * D,), (z_q, d["t2"], d["y"], i4(nq * D)), n=nq * D)
                d2d("d2d_copy_fuse", z_q, d["y"], nq * D * 4)

        # --- encoder-ish block ---
        launch(
            "layernorm_rows",
            (B * N,),
            (d["x"], d["y"], d["gamma"], d["beta"], i4(B * N), i4(D), f4(1e-6)),
            rows=B * N,
            D=D,
        )
        # QKV GEMM stand-in: y @ W_qkv -> tmp  (N,D)@(D,D)
        launch(
            "gemm_nn",
            (N * D,),
            (d["y"], d["W_qkv"], d["tmp"], i4(N), i4(D), i4(D), f4(1), f4(0)),
            M=N,
            N=D,
            K=D,
            beta=0.0,
        )
        d2d("d2d_copy_Q", d["q_flat"], d["Q"], B * H * N * Dh * 4)
        launch(
            "rope_apply",
            (B * H * N,),
            (d["q_flat"], d["pos"], i4(B * H * N), i4(Dh)),
            rows=B * H * N,
            Dh=Dh,
        )
        launch(
            "sdpa_rows",
            (B * H * N,),
            (d["Q"], d["K"], d["V"], d["O"], i4(B), i4(H), i4(N), i4(Dh), scale),
            B=B,
            H=H,
            N=N,
            Dh=Dh,
        )
        launch(
            "gemm_nn",
            (N * D,),
            (d["O"], d["W_out"], d["tmp"], i4(N), i4(D), i4(D), f4(1), f4(0)),
            M=N,
            N=D,
            K=D,
            beta=0.0,
        )
        launch("gelu", (B * N * D,), (d["tmp"], d["y"], i4(B * N * D)), n=B * N * D)
        launch(
            "gemm_nn",
            (N * D,),
            (d["y"], d["W_mlp"], d["x"], i4(N), i4(D), i4(D), f4(1), f4(0)),
            M=N,
            N=D,
            K=D,
            beta=0.0,
        )

        # --- ProjectedTriModalCrossAttention ---
        # x is encoder video tokens; copy into z_v working buffer
        d2d("d2d_copy_fuse", d["z_v"], d["x"], Nv * D * 4)
        for _ in range(self.layers):
            if self.use_imu:
                # gaze: Q=Z_g, KV=concat(Z_v, Z_i)
                branch_update(d["z_g"], Ng, [(d["z_v"], Nv), (d["z_i"], Ni)])
                # imu: Q=Z_i, KV=concat(Z_v, Z_g)
                branch_update(d["z_i"], Ni, [(d["z_v"], Nv), (d["z_g"], Ng)])
                # video: Q=Z_v, KV=concat(Z_g, Z_i)
                branch_update(d["z_v"], Nv, [(d["z_g"], Ng), (d["z_i"], Ni)])
            else:
                # Ego4D: gaze-only aux — no IMU concat
                # gaze: Q=Z_g, KV=Z_v
                branch_update(d["z_g"], Ng, [(d["z_v"], Nv)])
                # video: Q=Z_v, KV=Z_g
                branch_update(d["z_v"], Nv, [(d["z_g"], Ng)])
        d2d("d2d_copy_fuse", d["x"], d["z_v"], Nv * D * 4)

        # --- postfuse prune on fused video tokens ---
        d2d("d2d_zero_imp", d["imp"], d["zeros_imp"], B * N * 4)
        launch(
            "l2_normalize_rows",
            (B * N,),
            (d["x"], d["xn"], i4(B * N), i4(D)),
            rows=B * N,
            D=D,
        )
        for qs in range(0, N, chunk):
            C = min(chunk, N - qs)
            launch(
                "qk_logits_chunk",
                (B * C * N,),
                (d["xn"], d["logits"], i4(B), i4(N), i4(D), i4(qs), i4(C), sc),
                B=B,
                N=N,
                D=D,
                C=C,
            )
            launch(
                "softmax_rows",
                (B * C,),
                (d["logits"], d["attn"], i4(B * C), i4(N)),
                rows=B * C,
                N=N,
            )
            launch(
                "reduce_chunk_to_imp",
                (B * N,),
                (d["attn"], d["imp"], i4(B), i4(N), i4(C)),
                B=B,
                N=N,
                C=C,
            )
        launch("fill_recency", (N,), (d["rec"], i4(N), i4(8)), N=N)
        launch(
            "recency_reweight",
            (B * N,),
            (d["imp"], d["rec"], i4(B), i4(N), f4(0.5)),
            B=B,
            N=N,
        )
        launch(
            "topk_indices_row",
            (B,),
            (d["imp"], d["idx"], i4(B), i4(N), i4(K)),
            B=B,
            N=N,
            K=K,
        )
        launch(
            "gather_tokens",
            (B * K * D,),
            (d["x"], d["idx"], d["gathered"], i4(B), i4(N), i4(K), i4(D)),
            B=B,
            K=K,
            D=D,
        )
        return recs

def run_tick(pipe: Pipeline, mode: str) -> TickReport:
    """mode: cold_io | steady (no H2D; kernels + D2H) | device_only (no host IO)."""
    pipe.queue.finish()
    h2d = 0.0
    d2h = 0.0
    t0 = time.perf_counter()
    if mode == "cold_io":
        h2d = pipe._h2d_all()
    recs = pipe._run_kernels()
    if mode in ("cold_io", "steady"):
        d2h = pipe._d2h_outs()
    pipe.queue.finish()
    wall = 1e3 * (time.perf_counter() - t0)

    launch_ms = sum(r.launch_ms for r in recs)
    device_ms = sum(r.device_ms for r in recs)
    io_ms = h2d + d2h
    bound, io_f, la_f, de_f = classify(io_ms, launch_ms, device_ms)
    h2d_gbs = (pipe.bytes_h2d / 1e9) / (h2d / 1e3) if h2d > 0 else 0.0
    d2h_gbs = (pipe.bytes_d2h / 1e9) / (d2h / 1e3) if d2h > 0 else 0.0
    return TickReport(
        mode=mode,
        wall_ms=wall,
        h2d_ms=h2d,
        d2h_ms=d2h,
        io_ms=io_ms,
        launch_ms=launch_ms,
        device_ms=device_ms,
        n_launches=len(recs),
        launches=[asdict(r) for r in recs],
        bound=bound,
        io_frac=io_f,
        launch_frac=la_f,
        device_frac=de_f,
        bytes_h2d=pipe.bytes_h2d if mode == "cold_io" else 0,
        bytes_d2h=pipe.bytes_d2h if mode in ("cold_io", "steady") else 0,
        h2d_gbs=h2d_gbs,
        d2h_gbs=d2h_gbs,
    )


def aggregate_launches(
    launches: list[dict],
    ridge: float = 0.0,
    peak_bw: float = 0.0,
    peak_gflops: float = 0.0,
) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in launches:
        by.setdefault(r["name"], []).append(r)
    out = []
    for name, rs in by.items():
        flops = int(sum(r.get("flops", 0) for r in rs))
        nbytes = int(sum(r.get("bytes", 0) for r in rs))
        device_ms = sum(r["device_ms"] for r in rs)
        rec = {
            "name": name,
            "count": len(rs),
            "launch_ms": sum(r["launch_ms"] for r in rs),
            "device_ms": device_ms,
            **attach(flops, nbytes, device_ms, ridge, peak_bw=peak_bw, peak_gflops=peak_gflops),
        }
        out.append(rec)
    out.sort(key=lambda x: -x["device_ms"])
    return out


def median_kernels(
    reps: list[TickReport],
    ridge: float,
    peak_bw: float,
    peak_gflops: float,
) -> list[dict]:
    per = [aggregate_launches(r.launches, ridge, peak_bw, peak_gflops) for r in reps]
    names = [x["name"] for x in per[0]]
    out = []
    for name in names:
        series = [next(x for x in tick if x["name"] == name) for tick in per]
        flops = series[0]["flops"]
        nbytes = series[0]["bytes"]
        device_ms = med([s["device_ms"] for s in series])
        rec = {
            "name": name,
            "count": series[0]["count"],
            "launch_ms": med([s["launch_ms"] for s in series]),
            "device_ms": device_ms,
            **attach(flops, nbytes, device_ms, ridge, peak_bw=peak_bw, peak_gflops=peak_gflops),
        }
        out.append(rec)
    out.sort(key=lambda x: -x["device_ms"])
    return out


def run(
    prefer_gpu: bool,
    Nv: int,
    Ng: int,
    Ni: int,
    D: int,
    H: int,
    K: int,
    chunk: int,
    warmup: int,
    iters: int,
    layers: int = 1,
    use_ffn: bool = False,
    dataset: str = "hdepic",
):
    pipe = Pipeline(
        prefer_gpu,
        Nv=Nv,
        Ng=Ng,
        Ni=Ni,
        D=D,
        H=H,
        Kkeep=K,
        chunk=chunk,
        layers=layers,
        use_ffn=use_ffn,
        dataset=dataset,
    )
    peaks = measure_peaks(pipe.ctx, pipe.queue)
    ridge = peaks["ridge_AI"]
    print(
        f"peaks: BW={peaks['peak_BW_GBs']:.1f} GB/s  "
        f"FLOP={peaks['peak_GFLOPs']:.1f} GFLOP/s  ridge_AI={ridge:.3f} FLOP/Byte"
    )
    # cold once
    cold = run_tick(pipe, "cold_io")

    # warmup steady
    for _ in range(warmup):
        run_tick(pipe, "device_only")
        run_tick(pipe, "steady")

    steady_reps = [run_tick(pipe, "steady") for _ in range(iters)]
    device_reps = [run_tick(pipe, "device_only") for _ in range(iters)]

    def pack(reps: list[TickReport]) -> dict:
        return {
            "wall_ms": med([r.wall_ms for r in reps]),
            "h2d_ms": med([r.h2d_ms for r in reps]),
            "d2h_ms": med([r.d2h_ms for r in reps]),
            "io_ms": med([r.io_ms for r in reps]),
            "launch_ms": med([r.launch_ms for r in reps]),
            "device_ms": med([r.device_ms for r in reps]),
            "n_launches": reps[0].n_launches,
            "bound": classify(
                med([r.io_ms for r in reps]),
                med([r.launch_ms for r in reps]),
                med([r.device_ms for r in reps]),
            )[0],
            "io_frac": mean([r.io_frac for r in reps]),
            "launch_frac": mean([r.launch_frac for r in reps]),
            "device_frac": mean([r.device_frac for r in reps]),
            "per_kernel": median_kernels(reps, ridge=ridge, peak_bw=peaks["peak_BW_GBs"], peak_gflops=peaks["peak_GFLOPs"]),
        }

    steady = pack(steady_reps)
    device_only = pack(device_reps)

    print()
    print(f"shapes: Nv={Nv} Ng={Ng} Ni={Ni} D={D} H={H} K={K} chunk={chunk} layers={layers} ffn={use_ffn}")
    print(f"pipeline launches/tick: {cold.n_launches}")
    print()
    print("=== SYSTEM tick breakdown (ms) ===")
    print(f"{'mode':14s} {'wall':>8s} {'H2D':>8s} {'D2H':>8s} {'IO':>8s} {'launch':>8s} {'device':>8s}  bound")
    print("-" * 100)

    def row(tag, r):
        if isinstance(r, TickReport):
            print(
                f"{tag:14s} {r.wall_ms:8.3f} {r.h2d_ms:8.3f} {r.d2h_ms:8.3f} {r.io_ms:8.3f} "
                f"{r.launch_ms:8.3f} {r.device_ms:8.3f}  {r.bound}"
            )
        else:
            print(
                f"{tag:14s} {r['wall_ms']:8.3f} {r['h2d_ms']:8.3f} {r['d2h_ms']:8.3f} {r['io_ms']:8.3f} "
                f"{r['launch_ms']:8.3f} {r['device_ms']:8.3f}  {r['bound']}"
            )

    row("cold+IO", cold)
    row("steady+D2H", steady)
    row("device_only", device_only)

    print()
    print("=== per-kernel roofline (steady tick)  launch=queued→start  T=device ===")
    print(
        f"{'kernel':24s} {'#':>3s} {'launch':>8s} {'device':>8s} {'FLOPs':>12s} {'Bytes':>12s} "
        f"{'AI':>8s} {'BW_eff':>8s} {'GFLOP/s':>8s}  bound"
    )
    for r in steady["per_kernel"]:
        print(
            f"{r['name']:24s} {r['count']:3d} {r['launch_ms']:8.3f} {r['device_ms']:8.3f} "
            f"{r['flops']:12d} {r['bytes']:12d} "
            f"{r['AI']:8.3f} {r['BW_eff_GBs']:8.2f} {r['Compute_eff_GFLOPs']:8.2f}  {r['bound']}"
        )

    print()
    print(
        "How to read:\n"
        f"  ridge_AI = peak_FLOP/s / peak_BW = {ridge:.3f} FLOP/Byte "
        f"(peak {peaks['peak_GFLOPs']:.1f} GFLOP/s, {peaks['peak_BW_GBs']:.1f} GB/s)\n"
        "  memory-bound if BW_util>=25% (or AI<<ridge); compute-bound if FLOP_util>=25%\n"
        "  otherwise latency-bound (kernel too small to fill the device)\n"
        "  BW_eff = Bytes/T ; Compute_eff = FLOPs/T\n"
    )
    # heuristic tip
    s = steady
    if s["device_frac"] >= 0.5:
        tip = "System (steady) looks DEVICE-bound: optimize SDPA/topk/GEMM kernels (tiling/fuse), not PCIe."
    elif s["io_frac"] >= 0.5:
        tip = "System (steady) looks IO-bound: cut D2H, keep tokens on device, fuse host sync points."
    else:
        tip = "System (steady) looks LAUNCH/mixed: fuse kernels, reduce enqueue count, larger tiles."
    print("TIP:", tip)

    return {
        "device": pipe.device_label,
        "prefer_gpu": prefer_gpu,
        "shapes": {
            "Nv": Nv,
            "Ng": Ng,
            "Ni": Ni,
            "D": D,
            "H": H,
            "K": K,
            "chunk": chunk,
            "layers": layers,
            "use_ffn": use_ffn,
            "dataset": pipe.dataset,
            "path": pipe.path_label,
        },
        "peaks": peaks,
        "bytes_h2d_cold": pipe.bytes_h2d,
        "bytes_d2h": pipe.bytes_d2h,
        "cold": asdict(cold),
        "steady": steady,
        "device_only": device_only,
        "tip": tip,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nv", type=int, default=64, help="video tokens per slot")
    ap.add_argument("--Ng", type=int, default=16, help="gaze tokens (grid^2; vitl default 100)")
    ap.add_argument("--Ni", type=int, default=8, help="IMU tokens (~0.1*Nv; vitl default 26)")
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--layers", type=int, default=1, help="fusion_num_layers")
    ap.add_argument("--ffn", action="store_true", help="enable fusion FFN residual")
    ap.add_argument(
        "--vitl-tokens",
        action="store_true",
        help="ViT-L spatial budgets: Nv=256 Ng=100 Ni=26 (compute_token_budgets)",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="hdepic",
        choices=("hdepic", "ego4d"),
        help="hdepic=video/gaze/IMU; ego4d=video/gaze only (KV=gaze, no IMU concat)",
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    nv, ng, ni = args.Nv, args.Ng, args.Ni
    if args.vitl_tokens:
        nv, ng, ni = 256, 100, 26
        if args.dataset == "ego4d":
            ni = 0
    rep = run(
        prefer_gpu=not args.cpu,
        Nv=nv,
        Ng=ng,
        Ni=ni,
        D=args.D,
        H=args.H,
        K=args.K,
        chunk=args.chunk,
        warmup=args.warmup,
        iters=args.iters,
        layers=args.layers,
        use_ffn=args.ffn,
        dataset=args.dataset,
    )
    out = Path(args.out) if args.out else HERE / (
        "profile_system_cpu.json" if args.cpu else "profile_system_gpu.json"
    )
    # launches list can be large; keep aggregates in steady/device_only
    cold = rep["cold"]
    ridge = float(rep["peaks"]["ridge_AI"])
    cold["launches"] = aggregate_launches(
        cold["launches"],
        ridge=ridge,
        peak_bw=float(rep["peaks"]["peak_BW_GBs"]),
        peak_gflops=float(rep["peaks"]["peak_GFLOPs"]),
    )
    out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {out}")
    try:
        from plot_roofline import plot as plot_roofline

        fig_path = out.with_name(out.stem + "_roofline.png")
        plot_roofline(rep, fig_path)
    except Exception as exc:  # noqa: BLE001
        print(f"roofline figure skipped: {exc}")


if __name__ == "__main__":
    main()
