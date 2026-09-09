#!/usr/bin/env python3
"""Descriptive relative-lag analysis of frozen pooled B18 attention; CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

PROTOCOL = "b18-predictor-prune/egtea-ctx16-position-lag-v1"
SOURCE_PROTOCOL = "b18-predictor-prune/egtea-ctx16-target-crossslot-v1"
LAGS = np.arange(-63, 64)
Q, K = np.indices((64, 64))
DELTA = K - Q
PAIRS = 64 - np.abs(LAGS)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def rownorm(a):
    mass = a.sum(-1, keepdims=True)
    require(np.isfinite(a).all() and (a >= 0).all() and (mass > 0).all(), "Invalid/zero source row")
    return a / mass


def errors(actual, reconstructed, offslot):
    support = DELTA != 0 if offslot else np.ones((64, 64), dtype=bool)
    a, r = actual[:, support], reconstructed[:, support]
    sq = ((a - r) ** 2).sum(-1)
    energy = (a ** 2).sum(-1)
    centered = ((a - a.mean(-1, keepdims=True)) ** 2).sum(-1)
    require((energy > 0).all() and (centered > 0).all(), "Degenerate reconstruction denominator")
    return {"per_head_rmse": np.sqrt(sq / support.sum()).tolist(),
            "per_head_relative_frobenius_error": np.sqrt(sq / energy).tolist(),
            "per_head_uncentered_energy_explained": (1 - sq / energy).tolist(),
            "per_head_centered_r2": (1 - sq / centered).tolist(),
            "overall_relative_frobenius_error": float(np.sqrt(sq.sum() / energy.sum())),
            "overall_uncentered_energy_explained": float(1 - sq.sum() / energy.sum()),
            "overall_centered_r2_within_head": float(1 - sq.sum() / centered.sum()),
            "support": "off-diagonal entries only" if offslot else "all context matrix entries",
            "per_head_source_row_l1_error": np.abs(actual - reconstructed).sum(-1).tolist(),
            "per_head_received_column_l1_error": np.abs(actual - reconstructed).sum(-2).tolist()}


def analyze(matrix, offslot):
    # Fit each head separately. The diagonal mean is the least-squares Toeplitz
    # projection for this same matrix, not an independent predictive test.
    conditional = rownorm(matrix)
    results, arrays = {}, {}
    for label, a in (("raw", matrix), ("context_row_conditional", conditional)):
        lag_sum = np.stack([a[:, DELTA == d].sum(-1) for d in LAGS], axis=-1)
        kernel = lag_sum / PAIRS
        fit = kernel[:, DELTA + 63]
        require(np.allclose(lag_sum.sum(-1), a.sum((1, 2)), rtol=1e-12, atol=1e-12), "Lag mass conservation")
        require(np.allclose(fit.sum((1, 2)), a.sum((1, 2)), rtol=1e-12, atol=1e-12), "Toeplitz global mass conservation")
        if offslot:
            require(np.count_nonzero(fit[:, Q == K]) == 0, "Offslot projection reintroduced diagonal")
        projected_conditional = rownorm(fit)
        source = {}
        for name, mask in (("same_slot", DELTA == 0), ("adjacent_only", np.abs(DELTA) == 1),
                           ("distance_2_to_7", (np.abs(DELTA) >= 2) & (np.abs(DELTA) <= 7)),
                           ("distance_ge8", np.abs(DELTA) >= 8), ("older_keys", DELTA < 0),
                           ("newer_keys", DELTA > 0)):
            source[name] = (a * mask).sum(-1).tolist()
        boundaries = {}
        for name, start, stop in (("oldest8_sources", 0, 8), ("middle8_sources", 28, 36), ("newest8_sources", 56, 64)):
            counts = np.array([((DELTA[start:stop]) == d).sum() for d in LAGS])
            sums = np.stack([a[:, start:stop, :][:, DELTA[start:stop] == d].sum(-1) for d in LAGS], -1)
            corrected = np.divide(sums, counts, out=np.zeros_like(sums), where=counts[None, :] > 0)
            boundaries[name] = {"source_slots": [start, stop], "available_pair_counts": counts.tolist(),
                                "per_head_lag_sum_div_source_count": (sums / (stop - start)).tolist(),
                                "per_head_pair_corrected_lag_mean": corrected.tolist(),
                                "unavailable_lags": LAGS[counts == 0].tolist()}
        results[label] = {"per_head_lag_total_mass": lag_sum.tolist(),
                          "per_head_lag_mass_per_source_row": (lag_sum / 64).tolist(),
                          "per_head_pair_corrected_lag_kernel": kernel.tolist(),
                          "per_head_actual_row_mass": a.sum(-1).tolist(),
                          "per_head_projected_row_mass": fit.sum(-1).tolist(),
                          "per_head_actual_received_columns_per_source": (a.sum(-2) / 64).tolist(),
                          "per_head_projected_received_columns_per_source": (fit.sum(-2) / 64).tolist(),
                          "toeplitz_errors": errors(a, fit, offslot),
                          "source_row_bands": source, "boundary_strata": boundaries}
        if label == "context_row_conditional":
            results[label]["finite_support_row_renormalized_projection_errors"] = errors(a, projected_conditional, offslot)
            results[label]["per_head_row_renormalized_received_columns_per_source"] = (projected_conditional.sum(-2) / 64).tolist()
        arrays[label + "_actual"] = a
        arrays[label + "_toeplitz"] = fit
        arrays[label + "_toeplitz_then_row_normalized"] = projected_conditional
    return results, arrays


def figures(out, blocks, stored):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for scale in ("raw", "context_row_conditional"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for bi, block in enumerate((0, 11)):
            for ki, kind in enumerate(("original", "offslot")):
                ax = axes[bi, ki]
                result = blocks[f"L{block}"][kind][scale]
                kernel = np.asarray(result["per_head_pair_corrected_lag_kernel"])
                for row in kernel:
                    ax.plot(LAGS, np.where(row > 0, row, np.nan), color="tab:blue", alpha=.20, linewidth=.8)
                ax.plot(LAGS, np.where(kernel.mean(0) > 0, kernel.mean(0), np.nan), color="black", label="Mean of 12 heads")
                ax.set(title=f"L{block}: {kind}", xlabel="lag = key slot − query slot", ylabel="Mean mass per available slot pair", yscale="log")
                ax.legend(fontsize=8)
        fig.suptitle(f"{scale}: lag sums divided by 64−|lag|, each head separately\nOriginal softmax readout; examples and within-slot queries were pooled in source. Zero diagonal omitted on log scale.", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out / f"lag_pair_corrected_{scale}.png", dpi=170); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for bi, block in enumerate((0, 11)):
            for ki, kind in enumerate(("original", "offslot")):
                res = blocks[f"L{block}"][kind][scale]
                ax = axes[bi, ki]
                for field, label in (("per_head_actual_received_columns_per_source", "Actual received column mass"),
                                     ("per_head_projected_received_columns_per_source", "Finite-support Toeplitz projection (before row renorm)")):
                    ax.plot(np.asarray(res[field]).mean(0), label=label)
                if scale == "context_row_conditional":
                    ax.plot(np.asarray(res["per_head_row_renormalized_received_columns_per_source"]).mean(0), label="Projection, THEN row normalized", linestyle="--")
                ax.set(title=f"L{block}: {kind}", xlabel="Context key slot", ylabel="Mass / 64 source slots"); ax.legend(fontsize=7)
        fig.suptitle(f"{scale}: actual versus same-flow fitted lag projection\nEach head fitted independently; projection is descriptive, not independent mechanism validation.", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out / f"received_reconstruction_{scale}.png", dpi=170); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for bi, block in enumerate((0, 11)):
            for ki, kind in enumerate(("original", "offslot")):
                res = blocks[f"L{block}"][kind][scale]
                ax = axes[bi, ki]
                ax.plot(np.asarray(res["per_head_actual_row_mass"]).mean(0), label="Actual source-row mass")
                ax.plot(np.asarray(res["per_head_projected_row_mass"]).mean(0), label="Finite-support lag projection")
                ax.set(title=f"L{block}: {kind}", xlabel="Context source query slot", ylabel="Mass over context keys")
                ax.legend(fontsize=8)
        fig.suptitle(f"{scale}: finite support and source-row mass\nThe raw Toeplitz fit preserves total matrix mass but need not preserve each source-row total.", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out / f"source_row_reconstruction_{scale}.png", dpi=170); plt.close(fig)

    for block in (0, 11):
        for kind in ("original", "offslot"):
            a = stored[f"L{block}_{kind}_raw_actual"]
            shifted = np.full((12, 64, 127), np.nan)
            for q in range(64):
                shifted[:, q, np.arange(64) - q + 63] = a[:, q, :]
            fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
            vmax = float(np.nanquantile(shifted, .995))
            for h, ax in enumerate(axes.flat):
                im = ax.imshow(shifted[h], origin="lower", aspect="auto", extent=(-63.5, 63.5, -.5, 63.5), vmin=0, vmax=vmax)
                ax.set(title=f"Head {h}", xlabel="key slot − query slot", ylabel="Source query slot")
            fig.colorbar(im, ax=axes.ravel().tolist(), label="Raw probability per source query, clipped at shared 99.5th percentile", shrink=.7)
            fig.suptitle(f"L{block} {kind}: source slot × relative lag, separate heads\nBlank cells are unavailable boundary pairs; source samples and within-slot queries are already pooled.", fontsize=11)
            fig.savefig(out / f"source_lag_heads_L{block}_{kind}.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for bi, block in enumerate((0, 11)):
        for ki, scale in enumerate(("raw", "context_row_conditional")):
            ax = axes[bi, ki]
            for kind, line in (("original", "-"), ("offslot", "--")):
                bands = blocks[f"L{block}"][kind][scale]["source_row_bands"]
                for field, color in (("adjacent_only", "tab:orange"), ("distance_ge8", "tab:blue")):
                    ax.plot(np.asarray(bands[field]).mean(0), linestyle=line, color=color, label=f"{kind}: {field}")
            ax.set(title=f"L{block}: {scale}", xlabel="Source context query slot", ylabel="Band mass per source query"); ax.legend(fontsize=7)
    fig.suptitle("Source-row bands retain boundary effects\nRaw off-diagonal mass is unchanged by diagonal deletion; row conditioning changes its denominator.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out / "source_row_near_far.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for bi, block in enumerate((0, 11)):
        target = blocks[f"L{block}"]["target_separate"]
        for col, field, title in ((0, "per_head_context_query_to_target_key_by_source_slot", "Context query → target key"),
                                  (1, "per_head_target_query_to_context_key", "Target query → context key")):
            values = np.asarray(target[field]); ax = axes[bi, col]
            for row in values:
                ax.plot(row, alpha=.25, linewidth=.8, color="tab:blue")
            ax.plot(values.mean(0), color="black", label="Mean of heads")
            ax.set(title=f"L{block}: {title}", xlabel="Context source slot" if col == 0 else "Context key slot", ylabel="Raw original-query probability")
            ax.legend(fontsize=8)
    fig.suptitle("Target rows and columns, excluded from context lag fit\nTarget array index 64 uses RoPE slot 72; within-source-slot query and sample pooling precedes head display.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out / "target_rows_columns_separate.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir
    meta = json.loads((source / "metadata.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    require(meta["evaluation_protocol"] == SOURCE_PROTOCOL and str(meta["job_id"]) == "17026550", "Unexpected source identity")
    require(summary["n_rows"] == 4000, "Source not complete 4000")
    require(meta["token_layout"]["target_rope_start_slot"] == 72, "Unexpected target coordinates")
    profile_path = source / "crossslot_profiles.npz"
    with np.load(profile_path) as arrays:
        flow = arrays["mean_flow_raw"].astype(np.float64)
        off = arrays["mean_flow_offslot"].astype(np.float64)
        group_mean = arrays["raw_group_mass"].mean(0, dtype=np.float64)
    require(flow.shape == (2, 12, 65, 65) and off.shape == (2, 12, 64, 65), "Unexpected flow shape")
    require(np.isfinite(flow).all() and (flow >= 0).all(), "Invalid original flow")
    require(np.allclose(flow.sum(-1), 1, rtol=.01, atol=.001), "Original raw row sums")
    expected = flow[:, :, :64].copy()
    expected[:, :, np.arange(64), np.arange(64)] = 0
    require(np.allclose(off, expected, rtol=1e-6, atol=1e-8), "Stored offslot mismatch")
    require(np.allclose(flow[:, :, :64].sum(2) * 256, group_mean[:, 0], rtol=1e-5, atol=.005), "Context group/flow mismatch")
    require(np.allclose(flow[:, :, 64] * 256, group_mean[:, 1], rtol=1e-5, atol=.005), "Target group/flow mismatch")
    blocks, stored = {}, {}
    for bi, block in enumerate((0, 11)):
        entry = {}
        for kind, matrix in (("original", flow[bi, :, :64, :64]), ("offslot", off[bi, :, :, :64])):
            entry[kind], values = analyze(matrix, kind == "offslot")
            stored.update({f"L{block}_{kind}_{key}": value for key, value in values.items()})
        entry["target_separate"] = {
            "target_array_index": 64, "target_rope_start_slot": 72,
            "per_head_context_query_to_target_key_by_source_slot": flow[bi, :, :64, 64].tolist(),
            "per_head_target_query_to_context_key": flow[bi, :, 64, :64].tolist(),
            "per_head_target_query_to_context_key_conditioned": rownorm(flow[bi, :, 64, :64]).tolist(),
            "per_head_target_query_to_target_key": flow[bi, :, 64, 64].tolist(),
            "target_query_context_key_rope_lags": (np.arange(64) - 72).tolist(),
            "context_query_target_key_rope_lags": (72 - np.arange(64)).tolist()}
        blocks[f"L{block}"] = entry
    result = {"evaluation_protocol": PROTOCOL, "source_evaluation_protocol": SOURCE_PROTOCOL,
              "run_tag": "b18-q1-relative-lag", "job_id": os.environ.get("SLURM_JOB_ID"),
              "source_job_id": "17026550", "metric_scope": "diagnostic-pooled-attention",
              "eval_path": "Frozen full4000 mean per-head query-slot/key-slot flow; no model rerun",
              "coverage": {"rows": 4000, "sessions": 86, "participant_ids": 32},
              "source_dir": str(source), "source_sha256": {str(p): sha(p) for p in (profile_path, source / "metadata.json", source / "summary.json")},
              "analysis_source_sha256": sha(__file__), "lag_definition": "key_slot - query_slot; negative is older key relative to source query",
              "lags": LAGS.tolist(), "available_pair_counts": PAIRS.tolist(),
              "normalization": "Stored flow already averages examples and 256 queries per source slot. Raw has original all-key query mass denominator; context_row_conditional normalizes each head/source-slot mean row over context keys after sample/query pooling.",
              "stationary_projection": "Per-head least-squares Toeplitz diagonal mean, fitted separately to original/offslot and raw/context-row-conditioned matrices. Raw finite truncation changes row mass. Conditional projection is also reported after a SECOND row normalization on finite support, an explicitly different boundary hypothesis.",
              "limits": ["Descriptive projection fits the same matrix; agreement is not independent evidence of stationary mechanisms.", "Stored pooled flow cannot recover per-example or per-token-query lag distributions or uncertainty.", "Target rows/columns are separate and excluded from context lag fit; target array index64 has RoPE slot72.", "Offslot means different temporal slots, including neighbors; not synonymous with long-range.", "Reported residual fractions describe pooled matrix variation, not task utility or variance of model outputs."],
              "blocks": blocks}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out_dir / "lag_reconstructions.npz", **stored)
    figures(args.out_dir, blocks, stored)
    brief = {"job_id": result["job_id"], "gates": "passed", "blocks": {
        block: {kind: {scale: entry[kind][scale]["toeplitz_errors"] for scale in ("raw", "context_row_conditional")}
                for kind in ("original", "offslot")} for block, entry in blocks.items()}}
    (args.out_dir / "brief.json").write_text(json.dumps(brief, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"job_id": result["job_id"], "gates": "passed", "output": str(args.out_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
