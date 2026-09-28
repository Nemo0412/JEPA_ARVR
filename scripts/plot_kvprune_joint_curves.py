#!/usr/bin/env python3
"""Plot kvprune joint curves (step loss / epoch train loss / val Top-5).

Mirrors the local 2s figure layout. Reads each run's loss_steps.csv +
loss_epoch.csv (written by run_local_kvprune_joint_2s.py), optionally syncs
copies under the repo, and writes a 1x3 PNG.

Example (6s norope + rope_all on cluster):

  python scripts/plot_kvprune_joint_curves.py \\
    --runs norope:/scratch/ll5914/experiments/kvprune_joint_h6s/joint_6s_norope \\
           rope_all:/scratch/ll5914/experiments/kvprune_joint_h6s/joint_6s_rope_all \\
    --title 'kvprune joint 6s — norope / rope_all' \\
    --out /home/ll5914/Jepa/JEPA_ARVR/kvprune_joint_6s_loss_curves.png \\
    --sync-dir /home/ll5914/Jepa/JEPA_ARVR/plots/kvprune_joint_6s
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_run(spec: str) -> tuple[str, Path, float]:
    """label:path or label:path:global_batch. global_batch aligns x to 1-GPU samples."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"expected label:path[:global_batch], got {spec!r}")
    label = parts[0].strip()
    if len(parts) == 2:
        return label, Path(parts[1].strip()), 1.0
    try:
        gbatch = float(parts[-1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad global_batch in {spec!r}") from exc
    path = ":".join(parts[1:-1]).strip()
    return label, Path(path), gbatch


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) == 0:
        return x
    w = max(1, min(int(w), len(x)))
    ker = np.ones(w, dtype=np.float64) / w
    return np.convolve(x, ker, mode="same")


def _load_steps(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "loss_steps.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return df
    df = df[df["valid"].astype(float) > 0].copy()
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["loss"])
    return df


def _load_epoch(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "loss_epoch.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return df
    for c in ("epoch", "train_loss", "val_top5"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["epoch"])


def _latest_iter_pass(df: pd.DataFrame) -> pd.DataFrame:
    """One point per epoch-local iter, using the most recent visit (max global_step).

    Wall restarts inflate global_step; plotting by iter with last-write-wins shows
    true progress through the epoch (≈0..5696) without replay inflation.
    """
    if df.empty or "iter" not in df.columns:
        return df
    out = df.copy()
    out["iter"] = pd.to_numeric(out["iter"], errors="coerce")
    if "global_step" in out.columns:
        out["global_step"] = pd.to_numeric(out["global_step"], errors="coerce")
    else:
        out["global_step"] = np.arange(len(out), dtype=np.float64)
    out = out.dropna(subset=["iter", "loss"])
    if out.empty:
        return out
    # Last write wins per iter.
    idx = out.groupby(out["iter"].astype(int), sort=True)["global_step"].idxmax()
    return out.loc[idx].sort_values("iter").reset_index(drop=True)


def _epoch_iter_series(
    df: pd.DataFrame,
    iters_per_epoch: int | None = None,
    global_batch: float = 1.0,
) -> tuple[pd.DataFrame, int]:
    """Concatenate epochs on x: ep0 iters, then ep1 iters, ... (no global_step inflation).

    Last-write-wins per (epoch, iter). x = epoch * N + iter * global_batch so a
    2-GPU run (half as many iters) lines up with a 1-GPU run over the same samples.
    """
    if df.empty or "iter" not in df.columns or "epoch" not in df.columns:
        return df, int(iters_per_epoch or 0)
    out = df.copy()
    out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
    out["iter"] = pd.to_numeric(out["iter"], errors="coerce")
    if "global_step" in out.columns:
        out["global_step"] = pd.to_numeric(out["global_step"], errors="coerce")
    else:
        out["global_step"] = np.arange(len(out), dtype=np.float64)
    out = out.dropna(subset=["epoch", "iter", "loss"])
    if out.empty:
        return out, int(iters_per_epoch or 0)
    idx = out.groupby(
        [out["epoch"].astype(int), out["iter"].astype(int)], sort=True
    )["global_step"].idxmax()
    out = out.loc[idx].sort_values(["epoch", "iter"]).reset_index(drop=True)
    inferred = int(out["iter"].max()) + 1
    n = int(iters_per_epoch) if iters_per_epoch and iters_per_epoch > 0 else inferred
    n = max(n, inferred)
    g = float(global_batch) if global_batch and global_batch > 0 else 1.0
    out["x"] = out["epoch"].astype(int) * n + out["iter"].astype(float) * g
    return out, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        type=_parse_run,
        required=True,
        help="label:path[:global_batch]. global_batch>1 stretches iter so x matches 1-GPU samples.",
    )
    ap.add_argument("--title", default="kvprune joint")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--sync-dir",
        type=Path,
        default=None,
        help="If set, copy each run's CSVs here as <label>_loss_{steps,epoch}.csv",
    )
    ap.add_argument("--ma", type=int, default=100, help="step-loss moving average window")
    ap.add_argument(
        "--align-batch",
        type=float,
        default=2.0,
        help="Align step axis to this global batch (step *= world*bs / align_batch). "
        "Cluster 1-GPU bs=1 → multiply steps by 0.5 vs local 2-GPU bs=1.",
    )
    ap.add_argument(
        "--run-global-batch",
        type=float,
        default=1.0,
        help="This run's global batch (n_gpu * batch_size). Default 1 for single-GPU cluster.",
    )
    ap.add_argument(
        "--x-mode",
        choices=("epoch_iter", "global_step", "iter_latest"),
        default="epoch_iter",
        help="epoch_iter = ep0 then ep1 … on x (epoch*N+iter; default, no restart "
        "inflation). global_step = raw counter. iter_latest = only latest epoch, x=iter.",
    )
    ap.add_argument(
        "--iters-per-epoch",
        type=int,
        default=5696,
        help="Epoch length for epoch_iter x-axis (train kept size). 0 = infer from CSV.",
    )
    args = ap.parse_args()

    if args.sync_dir is not None:
        args.sync_dir.mkdir(parents=True, exist_ok=True)

    series = []
    for label, run_dir, gbatch in args.runs:
        steps = _load_steps(run_dir)
        epoch = _load_epoch(run_dir)
        if args.sync_dir is not None:
            for name in ("loss_steps.csv", "loss_epoch.csv"):
                src = run_dir / name
                if src.is_file():
                    shutil.copy2(src, args.sync_dir / f"{label}_{name}")
        scale = float(args.run_global_batch) / float(args.align_batch)
        series.append({
            "label": label, "steps": steps, "epoch": epoch, "scale": scale, "gbatch": gbatch,
        })

    # Match 2s figure: C0 blue / C2 green (skip C1 orange used by rope_blk0 there).
    default_cycle = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    label_color = {
        "norope": "#1f77b4",
        "rope_all": "#2ca02c",
        "rope": "#2ca02c",
        "rope_relpred": "#ff7f0e",
        "rope_blk0": "#9467bd",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    def _color(i: int, label: str) -> str:
        return label_color.get(label, default_cycle[i % len(default_cycle)])

    def _legend(ax, loc: str) -> None:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=loc, fontsize=8)

    # 1) step loss
    ax = axes[0]
    tips = []
    epoch_ns: list[int] = []
    xmax = 0.0
    for i, s in enumerate(series):
        df = s["steps"]
        if df.empty:
            continue
        c = _color(i, s["label"])
        if args.x_mode == "iter_latest":
            df = _latest_iter_pass(df)
            if df.empty:
                continue
            x = df["iter"].to_numpy(dtype=np.float64)
            y = df["loss"].to_numpy(dtype=np.float64)
            tips.append(f"{s['label']}: iter {int(x[0])}→{int(x[-1])}/5696")
        elif args.x_mode == "epoch_iter":
            n_arg = None if args.iters_per_epoch <= 0 else int(args.iters_per_epoch)
            df, n = _epoch_iter_series(
                df, iters_per_epoch=n_arg, global_batch=float(s["gbatch"]),
            )
            if df.empty:
                continue
            epoch_ns.append(n)
            x = df["x"].to_numpy(dtype=np.float64)
            y = df["loss"].to_numpy(dtype=np.float64)
            ep_hi = int(df["epoch"].max())
            it_hi = int(df.loc[df["epoch"] == ep_hi, "iter"].max())
            tips.append(
                f"{s['label']}: through ep{ep_hi} it{it_hi}  "
                f"(x={int(x[0])}→{int(x[-1])}, N={n}/ep, gbatch={s['gbatch']:g})"
            )
        else:
            x = df["global_step"].to_numpy(dtype=np.float64) * s["scale"]
            y = df["loss"].to_numpy(dtype=np.float64)
            tips.append(f"{s['label']}: steps={len(df)}")
        if len(x):
            xmax = max(xmax, float(x[-1]))
        ax.plot(x, _rolling_mean(y, args.ma), color=c, lw=1.8, label=s["label"])
    if args.x_mode == "iter_latest":
        ax.set_title("Step loss (latest epoch pass)")
        ax.set_xlabel("iter within epoch (bs=1 → 5696 / epoch)")
    elif args.x_mode == "epoch_iter":
        n = max(epoch_ns) if epoch_ns else int(args.iters_per_epoch or 5696)
        ax.set_title("Step loss (by epoch·iter)")
        ax.set_xlabel(f"samples, 1-GPU aligned  (ep0: 0…{n - 1}, ep1: {n}…)")
        # Epoch boundaries for quick visual.
        if n > 0 and xmax > 0:
            for b in range(n, int(xmax) + n, n):
                ax.axvline(b, color="0.7", lw=0.8, ls="--", zorder=0)
    else:
        use_align = abs(float(args.run_global_batch) / float(args.align_batch) - 1.0) > 1e-9
        if use_align:
            ax.set_title("Step loss (batch-aligned)")
            ax.set_xlabel(f"aligned step (equiv. @ global batch={args.align_batch:g})")
        else:
            ax.set_title("Step loss")
            ax.set_xlabel("global step")
    ax.set_ylabel(f"train loss ({args.ma}-step MA)")
    ax.grid(True, alpha=0.3)
    _legend(ax, "upper right")

    # 2) epoch train loss
    ax = axes[1]
    for i, s in enumerate(series):
        df = s["epoch"]
        if df.empty or "train_loss" not in df.columns:
            continue
        ax.plot(
            df["epoch"], df["train_loss"],
            color=_color(i, s["label"]), marker="o", lw=1.8, label=s["label"],
        )
    ax.set_title("Epoch train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean train loss")
    ax.grid(True, alpha=0.3)
    _legend(ax, "upper right")

    # 3) val top-5
    ax = axes[2]
    for i, s in enumerate(series):
        df = s["epoch"]
        if df.empty or "val_top5" not in df.columns:
            continue
        ax.plot(
            df["epoch"], df["val_top5"],
            color=_color(i, s["label"]), marker="o", lw=1.8, label=s["label"],
        )
    ax.set_title("Val Top-5")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val Top-5 (%)")
    ax.grid(True, alpha=0.3)
    _legend(ax, "lower right")

    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    pdf = args.out.with_suffix(".pdf")
    fig.savefig(pdf)
    print(f"wrote {args.out}")
    print(f"wrote {pdf}")
    for tip in tips:
        print(f"  {tip}")
    for s in series:
        n_e = len(s["epoch"])
        tip = ""
        if n_e:
            last = s["epoch"].iloc[-1]
            tip = f"  last ep={int(last['epoch'])} loss={last.get('train_loss', float('nan')):.3f} top5={last.get('val_top5', float('nan')):.2f}"
        print(f"  {s['label']}: epochs={n_e}{tip}")


if __name__ == "__main__":
    main()
