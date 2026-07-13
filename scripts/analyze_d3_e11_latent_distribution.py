"""D3 E11: direct_rope vs oracle target latent distribution (probe-free).

Collects per-sample mean-pooled target latents (encoder oracle vs predictor direct_rope),
writes summary stats + histogram/PCA figures. Oracle chunk norms can optionally be merged
from an existing failure-modes ``latent_stats.csv`` for comparison plots.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation.future_latent_compare import (
    FutureOracleDataset,
    _build_samples,
    _collate,
    _last_layer,
    _load_encoder_predictor,
    _predict_direct,
)
from evals.action_anticipation_frozen.dataloader import filter_annotations

logger = logging.getLogger("d3_e11_latent_distribution")


def _spatial_tokens_per_chunk(cfg: dict, encoder) -> int:
    data_cfg = cfg["experiment"]["data"]
    wrapper_cfg = cfg["model_kwargs"].get("wrapper_kwargs", {})
    grid = int(data_cfg["resolution"] // encoder.patch_size)
    tubelet = int(encoder.tubelet_size)
    num_output_frames = max(int(wrapper_cfg.get("num_output_frames", 2)), tubelet)
    return int(grid * grid * (num_output_frames // tubelet))


def _pool_chunk(tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool spatial tokens -> [B, D]."""
    return tokens.float().mean(dim=1)


def _chunk_norm(tokens: torch.Tensor) -> np.ndarray:
    """L2 norm of flattened chunk per sample."""
    flat = tokens.float().reshape(tokens.size(0), -1)
    return torch.linalg.vector_norm(flat, dim=1).detach().cpu().numpy()


def _read_oracle_norms_from_csv(csv_path: Path) -> np.ndarray | None:
    import csv

    if not csv_path.is_file():
        return None
    norms = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("oracle_target_norm", "")
            if val != "":
                norms.append(float(val))
    return np.asarray(norms, dtype=np.float64) if norms else None


@torch.no_grad()
def collect_latents(args, cfg, device):
    data_cfg = cfg["experiment"]["data"]
    annotations = filter_annotations(
        data_cfg["dataset"],
        data_cfg["base_path"],
        data_cfg["dataset_train"],
        data_cfg["dataset_val"],
        file_format=data_cfg.get("file_format", 1),
    )
    samples = _build_samples(annotations["val"])
    encoder, predictor = _load_encoder_predictor(cfg, device)
    spatial = _spatial_tokens_per_chunk(cfg, encoder)

    ds = FutureOracleDataset(
        samples=samples,
        horizon_sec=args.horizon,
        frames_per_clip=data_cfg["frames_per_clip"],
        fps=float(data_cfg["frames_per_second"]),
        anticipation_point=tuple(data_cfg.get("val_anticipation_point", [0.0, 0.0])),
        resolution=data_cfg["resolution"],
        drop_incomplete_history=args.drop_incomplete_history,
        max_samples=args.max_samples or None,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )
    use_bf16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", False)) and device.type == "cuda"

    oracle_vecs: list[np.ndarray] = []
    direct_vecs: list[np.ndarray] = []
    oracle_norms: list[np.ndarray] = []
    direct_norms: list[np.ndarray] = []
    cos_list: list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        observed = batch["observed"].to(device, non_blocking=True)
        oracle_clip = batch["oracle"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
            obs_tok = encoder(observed)
            ora_tok = encoder(oracle_clip)
            ora_last = _last_layer(ora_tok, encoder.embed_dim)
            oracle_target = ora_last[:, -spatial:, :]
            direct_target, info = _predict_direct(
                encoder,
                predictor,
                obs_tok,
                args.horizon,
                cfg,
                device,
                dense=False,
                rope_scale_mode="ntk_temporal",
            )
            if direct_target is None:
                raise RuntimeError(f"direct_rope failed: {info}")

            o_pool = _pool_chunk(oracle_target)
            d_pool = _pool_chunk(direct_target)
            cos = torch.nn.functional.cosine_similarity(o_pool, d_pool, dim=1)

            oracle_vecs.append(o_pool.detach().cpu().numpy())
            direct_vecs.append(d_pool.detach().cpu().numpy())
            oracle_norms.append(_chunk_norm(oracle_target))
            direct_norms.append(_chunk_norm(direct_target))
            cos_list.append(cos.detach().cpu().numpy())

        if batch_idx % args.log_every == 0:
            logger.info("batch=%d", batch_idx)

    oracle_x = np.concatenate(oracle_vecs, axis=0)
    direct_x = np.concatenate(direct_vecs, axis=0)
    return {
        "oracle_x": oracle_x,
        "direct_x": direct_x,
        "oracle_norm": np.concatenate(oracle_norms, axis=0),
        "direct_norm": np.concatenate(direct_norms, axis=0),
        "cos": np.concatenate(cos_list, axis=0),
        "spatial": spatial,
        "samples": int(oracle_x.shape[0]),
        "embed_dim": int(oracle_x.shape[1]),
    }


def _summary_stats(oracle_x: np.ndarray, direct_x: np.ndarray, oracle_norm: np.ndarray, direct_norm: np.ndarray, cos: np.ndarray):
    o_std = oracle_x.std(axis=0)
    d_std = direct_x.std(axis=0)
    o_mean = oracle_x.mean(axis=0)
    d_mean = direct_x.mean(axis=0)
    return {
        "samples": int(oracle_x.shape[0]),
        "embed_dim": int(oracle_x.shape[1]),
        "oracle_norm_mean": float(oracle_norm.mean()),
        "oracle_norm_std": float(oracle_norm.std()),
        "direct_norm_mean": float(direct_norm.mean()),
        "direct_norm_std": float(direct_norm.std()),
        "norm_ratio_mean": float((direct_norm / np.maximum(oracle_norm, 1e-8)).mean()),
        "norm_ratio_median": float(np.median(direct_norm / np.maximum(oracle_norm, 1e-8))),
        "cos_mean": float(cos.mean()),
        "cos_median": float(np.median(cos)),
        "per_dim_mean_l2_diff": float(np.linalg.norm(d_mean - o_mean)),
        "per_dim_std_l2_ratio": float(np.linalg.norm(d_std) / max(np.linalg.norm(o_std), 1e-8)),
        "per_dim_std_pearson": float(np.corrcoef(o_std, d_std)[0, 1]) if oracle_x.shape[1] > 1 else 1.0,
    }


def _pca_2d(oracle_x: np.ndarray, direct_x: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([oracle_x, direct_x], axis=0).astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    n, d = x.shape
    if n < 3:
        return np.zeros((n, 2)), np.zeros(2)
    # Randomized SVD for wide embeddings
    k = min(32, d, n - 1)
    idx = rng.choice(d, size=k, replace=False) if d > k else np.arange(d)
    xs = x[:, idx]
    _, _, vt = np.linalg.svd(xs, full_matrices=False)
    comp = vt[:2]
    # Project full centered x using least-squares fit from subsample
    if d > k:
        w, _, _, _ = np.linalg.lstsq(x[:, idx], x, rcond=None)
        comp = (comp @ w)
    comp = comp / (np.linalg.norm(comp, axis=1, keepdims=True) + 1e-8)
    proj = x @ comp.T
    return proj, comp


def _save_plots(out_dir: Path, oracle_norm: np.ndarray, direct_norm: np.ndarray, oracle_x, direct_x, oracle_norm_csv: np.ndarray | None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Norm histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(oracle_norm.min(), direct_norm.min()),
        max(oracle_norm.max(), direct_norm.max()),
        50,
    )
    ax.hist(oracle_norm, bins=bins, alpha=0.55, density=True, label="oracle (encoder)")
    ax.hist(direct_norm, bins=bins, alpha=0.55, density=True, label="direct_rope (predictor)")
    if oracle_norm_csv is not None and oracle_norm_csv.size:
        ax.hist(oracle_norm_csv, bins=bins, alpha=0.35, density=True, histtype="step", linewidth=2, label="oracle (cached csv)")
    ax.set_xlabel("chunk L2 norm")
    ax.set_ylabel("density")
    ax.set_title("E11: target chunk norm distribution @ 10s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "norm_histogram.png", dpi=150)
    plt.close(fig)

    # Per-dimension std scatter
    o_std = oracle_x.std(axis=0)
    d_std = direct_x.std(axis=0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(o_std, d_std, s=4, alpha=0.35)
    lim = max(o_std.max(), d_std.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("oracle per-dim std")
    ax.set_ylabel("direct_rope per-dim std")
    ax.set_title("Per-dimension std (mean-pooled samples)")
    fig.tight_layout()
    fig.savefig(fig_dir / "per_dim_std_scatter.png", dpi=150)
    plt.close(fig)

    # Per-dimension mean diff
    o_mean = oracle_x.mean(axis=0)
    d_mean = direct_x.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(d_mean - o_mean, lw=0.8)
    ax.set_xlabel("dimension")
    ax.set_ylabel("mean(direct) - mean(oracle)")
    ax.set_title("Per-dimension mean shift")
    fig.tight_layout()
    fig.savefig(fig_dir / "per_dim_mean_shift.png", dpi=150)
    plt.close(fig)

    # PCA
    proj, _ = _pca_2d(oracle_x, direct_x)
    n = oracle_x.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(proj[:n, 0], proj[:n, 1], s=8, alpha=0.45, label="oracle")
    ax.scatter(proj[n:, 0], proj[n:, 1], s=8, alpha=0.45, label="direct_rope")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA on mean-pooled target vectors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_scatter.png", dpi=150)
    plt.close(fig)

    # Norm ratio histogram
    ratio = direct_norm / np.maximum(oracle_norm, 1e-8)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratio, bins=50, alpha=0.8, color="steelblue")
    ax.axvline(ratio.mean(), color="crimson", ls="--", label=f"mean={ratio.mean():.3f}")
    ax.set_xlabel("||direct|| / ||oracle||")
    ax.set_ylabel("count")
    ax.set_title("Per-sample norm ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "norm_ratio_histogram.png", dpi=150)
    plt.close(fig)


def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    oracle_norm_csv = _read_oracle_norms_from_csv(args.oracle_latent_stats_csv) if args.oracle_latent_stats_csv else None

    if args.latent_npz and args.latent_npz.is_file():
        logger.info("Loading cached latents from %s", args.latent_npz)
        data = np.load(args.latent_npz)
        bundle = {k: data[k] for k in data.files}
    else:
        with args.config.open("r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bundle = collect_latents(args, cfg, device)
        np.savez_compressed(
            args.out_dir / "latent_vectors_mean_pooled.npz",
            oracle_x=bundle["oracle_x"],
            direct_x=bundle["direct_x"],
            oracle_norm=bundle["oracle_norm"],
            direct_norm=bundle["direct_norm"],
            cos=bundle["cos"],
        )

    summary = _summary_stats(
        bundle["oracle_x"],
        bundle["direct_x"],
        bundle["oracle_norm"],
        bundle["direct_norm"],
        bundle["cos"],
    )
    summary["experiment"] = "D3-E11-latent-distribution"
    summary["horizon_sec"] = args.horizon
    if oracle_norm_csv is not None:
        summary["cached_oracle_norm_mean"] = float(oracle_norm_csv.mean())
        summary["cached_vs_recomputed_oracle_norm_delta"] = float(
            abs(oracle_norm_csv.mean() - summary["oracle_norm_mean"])
        )

    o_std = bundle["oracle_x"].std(axis=0)
    d_std = bundle["direct_x"].std(axis=0)
    np.save(args.out_dir / "per_dim_mean_oracle.npy", bundle["oracle_x"].mean(axis=0))
    np.save(args.out_dir / "per_dim_mean_direct.npy", bundle["direct_x"].mean(axis=0))
    np.save(args.out_dir / "per_dim_std_oracle.npy", o_std)
    np.save(args.out_dir / "per_dim_std_direct.npy", d_std)

    _save_plots(
        args.out_dir,
        bundle["oracle_norm"],
        bundle["direct_norm"],
        bundle["oracle_x"],
        bundle["direct_x"],
        oracle_norm_csv,
    )

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("E11 summary: %s", json.dumps(summary, indent=2))
    logger.info("Wrote figures under %s", args.out_dir / "figures")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--horizon", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--drop-incomplete-history", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument(
        "--oracle-latent-stats-csv",
        type=Path,
        default=None,
        help="Optional existing latent_stats.csv for oracle norm overlay.",
    )
    p.add_argument(
        "--latent-npz",
        type=Path,
        default=None,
        help="Reuse cached npz from a prior E11 run (plots only, no GPU).",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
