#!/usr/bin/env python3
"""Roofline + launch/device figure from profile_system_pipeline JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def plot(rep: dict, out: Path) -> None:
    peaks = rep["peaks"]
    pk_bw = float(peaks["peak_BW_GBs"])
    pk_flop = float(peaks["peak_GFLOPs"])
    ridge = float(peaks["ridge_AI"])
    kernels = rep["steady"]["per_kernel"]
    shapes = rep.get("shapes", {})
    device = rep.get("device", "")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=160)
    ax = axes[0]

    # Roof: min(BW * I, peak_flop). I in FLOP/Byte; BW GB/s → GFLOP/s = BW * I
    i_min, i_max = 1e-3, max(40.0, ridge * 8)
    xs = np.logspace(np.log10(i_min), np.log10(i_max), 400)
    roof = np.minimum(pk_bw * xs, pk_flop)
    ax.plot(xs, roof, color="#222222", lw=1.6, label="roofline")
    ax.axvline(ridge, color="#888888", ls="--", lw=0.9, label=f"ridge AI={ridge:.2f}")
    ax.text(ridge * 1.08, pk_flop * 0.55, "compute\nbound", fontsize=8, color="#555")
    ax.text(ridge * 0.04, pk_flop * 0.55, "memory\nbound", fontsize=8, color="#555")

    colors = {
        "memory-bound": "#c44e52",
        "compute-bound": "#4c72b0",
        "latency-bound (low util)": "#dd8452",
        "mixed": "#55a868",
        "overhead": "#8c8c8c",
    }
    for k in kernels:
        ai = max(float(k["AI"]), 3e-3) if float(k["flops"]) == 0 else max(float(k["AI"]), 1e-4)
        y = max(float(k["Compute_eff_GFLOPs"]), 1e-3)
        c = colors.get(k.get("bound", ""), "#4c72b0")
        ax.scatter(ai, y, s=36 + 4 * int(k.get("count", 1)), c=c, zorder=3, edgecolors="k", lw=0.3)
        label = k["name"].replace("_rows", "").replace("_chunk", "")
        if k["device_ms"] >= 0.2 or k["name"] in ("gemm_nn", "sdpa_cross_rows", "sdpa_rows", "topk_indices_row"):
            ax.annotate(label, (ai, y), textcoords="offset points", xytext=(4, 3), fontsize=6.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(i_min, i_max)
    ax.set_ylim(1e-3, pk_flop * 3)
    ax.set_xlabel("Arithmetic intensity AI = FLOPs / Bytes  (FLOP/Byte)")
    ax.set_ylabel("Compute_eff = FLOPs / T_device  (GFLOP/s)")
    ax.set_title("Roofline (device time T)")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.7)
    ax.legend(loc="lower right", fontsize=7, frameon=False)

    # Launch vs device
    axb = axes[1]
    ks = sorted(kernels, key=lambda x: -(x["launch_ms"] + x["device_ms"]))
    names = [k["name"].replace("_rows", "").replace("_chunk", "").replace("_indices", "") for k in ks]
    launch = np.array([k["launch_ms"] for k in ks])
    device_t = np.array([k["device_ms"] for k in ks])
    y = np.arange(len(ks))
    axb.barh(y, launch, color="#8172b3", label="launch (queued→start)", height=0.72)
    axb.barh(y, device_t, left=launch, color="#4c72b0", label="device (start→end)", height=0.72)
    axb.set_yticks(y)
    axb.set_yticklabels(names, fontsize=7)
    axb.invert_yaxis()
    axb.set_xlabel("time per tick (ms)")
    axb.set_title("Launch vs device latency")
    axb.legend(loc="lower right", fontsize=7, frameon=False)
    axb.grid(True, axis="x", ls=":", lw=0.4, alpha=0.7)

    tot_l = float(rep["steady"]["launch_ms"])
    tot_d = float(rep["steady"]["device_ms"])
    cap = (
        f"{device}\n"
        f"Nv={shapes.get('Nv')} Ng={shapes.get('Ng')} Ni={shapes.get('Ni')} "
        f"D={shapes.get('D')} H={shapes.get('H')}  |  "
        f"peak {pk_flop:.0f} GFLOP/s, {pk_bw:.0f} GB/s  |  "
        f"steady launch {tot_l:.2f} ms + device {tot_d:.2f} ms"
    )
    fig.suptitle("V-JEPA tri-modal fusion OpenCL  ·  video/gaze/IMU", fontsize=11, y=1.02)
    fig.text(0.01, -0.04, cap, fontsize=7, color="#444")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=str(HERE / "profile_system_gpu_N256.json"))
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    src = Path(args.json)
    out = Path(args.out) if args.out else HERE / "figures" / (src.stem + "_roofline.png")
    plot(load(src), out)


if __name__ == "__main__":
    main()
