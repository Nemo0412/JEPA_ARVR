#!/usr/bin/env python3
"""Pooled same-slot deletion from existing B18 raw mean flows; CPU Slurm only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

PROTOCOL = "b18-predictor-prune/egtea-ctx16-target-crossslot-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalized(value):
    total = value.sum(-1, keepdims=True)
    require(np.isfinite(value).all() and (value >= 0).all() and (total > 0).all(),
            "Invalid or zero pooled attention distribution")
    return value / total


def temporal(received, queries):
    # received: head,key slot; already averaged over all 4000 examples.
    heads = received.shape[0]
    absolute = received.sum(0) / (heads * queries)
    conditional = normalized(received[:, :64].sum(0))
    per_head = normalized(received[:, :64])
    entropy = -(conditional * np.log(np.maximum(conditional, 1e-30))).sum() / np.log(64)
    return {"source_queries": queries, "absolute_context_probability": float(absolute[:64].sum()),
            "absolute_target_probability": float(absolute[64:].sum()),
            "absolute_context_profile_per_original_query_head": absolute[:64].tolist(),
            "context_conditional_profile": conditional.tolist(),
            "context_conditional_entropy_div_log64": float(entropy),
            "context_conditional_recent8_fraction": float(conditional[-8:].sum()),
            "context_conditional_recent16_fraction": float(conditional[-16:].sum()),
            "context_conditional_oldest8_fraction": float(conditional[:8].sum()),
            "context_conditional_max_slot_probability": float(conditional.max()),
            "per_head_context_conditional_profile": per_head.tolist(),
            "order": "Average examples first (stored raw flow), sum queries then heads, normalize over context keys last. Per-head profiles normalize before head pooling and are separately labeled."}


def figures(out, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode, field in (("absolute", "head_pooled_raw_context_matrix"),
                        ("conditional", "head_pooled_row_context_conditional_matrix")):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for bi, block in enumerate((0, 11)):
            matrices = [np.asarray(results[f"L{block}"]["flow"][kind][field])
                        for kind in ("original", "offslot")]
            vmax = max(float(x.max()) for x in matrices)
            for col, (label, matrix) in enumerate(zip(("Original context queries", "Same-slot readout removed"), matrices)):
                ax = axes[bi, col]
                im = ax.imshow(matrix, origin="lower", vmin=0, vmax=vmax, aspect="auto")
                ax.set(xlabel="Context key slot", ylabel="Context query slot", title=f"L{block}: {label}")
                fig.colorbar(im, ax=ax, fraction=.046)
        label = ("Original probability; mean over samples, heads and within-slot queries" if mode == "absolute"
                 else "Rows conditioned on context keys AFTER sample and head pooling")
        fig.suptitle(label + "\nEach block uses a shared color scale. Posthoc readout only; model unchanged.", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(out / f"pooled_flow_{mode}.png", dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for bi, block in enumerate((0, 11)):
        for col, (field, title) in enumerate((("absolute_context_profile_per_original_query_head", "Absolute mass"),
                                             ("context_conditional_profile", "Context-conditioned received shape"))):
            for kind, label in (("context_original", "Original context queries"), ("context_offslot", "Same-slot removed")):
                axes[bi, col].plot(results[f"L{block}"]["temporal"][kind][field], label=label)
            axes[bi, col].set(xlabel="Context key slot: 0 oldest, 63 newest", ylabel="Probability", title=f"L{block}: {title}")
            axes[bi, col].legend(fontsize=8)
    fig.suptitle("Received temporal profile: pooled context-query source held fixed\nConditioning occurs after pooling; no per-query renormalization or per-sample inference.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(out / "pooled_received_temporal.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir
    metadata = json.loads((source / "metadata.json").read_text())
    source_summary = json.loads((source / "summary.json").read_text())
    profiles_path = source / "attention_profiles.npz"
    arrays = np.load(profiles_path)
    groups = arrays["group_mass"].astype(np.float64)
    require(source_summary["n_rows"] == groups.shape[0] == 4000, "Expected complete original 4000 rows")
    require(metadata["evaluation_protocol"] == "b18-predictor-prune/egtea-ctx16-paired-hybrid-v1", "Unexpected source protocol")
    require(str(metadata["job_id"]) == "17007365", "Unexpected source GPU job")
    require(groups.shape[1:3] == (2, 2) and groups.shape[-1] == 65, "Unexpected source group layout")
    require(np.isfinite(groups).all() and (groups >= 0).all(), "Invalid source group mass")
    results, saved = {}, {}
    for bi, block in enumerate((0, 11)):
        flow = arrays[f"mean_queryslot_keyslot_L{block}"].astype(np.float64)
        heads = groups.shape[3]
        require(flow.shape == (heads, 65, 65) and np.isfinite(flow).all() and (flow >= 0).all(), "Invalid raw source flow")
        context = flow[:, :64, :]
        target = flow[:, 64:, :]
        c_received = context.sum(1) * 256
        t_received = target.sum(1) * 256
        require(np.allclose(c_received, groups[:, bi, 0].mean(0), rtol=1e-5, atol=.005), "Context flow/group mean mismatch")
        require(np.allclose(t_received, groups[:, bi, 1].mean(0), rtol=1e-5, atol=.005), "Target flow/group mean mismatch")
        require(np.allclose(context.sum(-1), 1, rtol=.01, atol=.001), "Raw context query-row mass invalid")
        require(np.allclose(target.sum(-1), 1, rtol=.01, atol=.001), "Raw target query-row mass invalid")
        off = context.copy()
        ix = np.arange(64)
        diagonal = context[:, ix, ix].copy()
        off[:, ix, ix] = 0
        require(np.array_equal(off[..., 64:], context[..., 64:]), "Target-key column changed")
        require(np.count_nonzero(off[:, ix, ix]) == 0 and (off <= context).all(), "Invalid same-slot deletion")
        o_received = off.sum(1) * 256
        require(np.allclose(c_received[:, :64] - o_received[:, :64], diagonal * 256, rtol=1e-9, atol=1e-9), "Received-mass deletion mismatch")
        original_sum = float(context.sum())
        removed_sum = float(diagonal.sum())
        remaining_sum = float(off.sum())
        require(np.isclose(original_sum, removed_sum + remaining_sum, rtol=1e-12), "Raw mass conservation failed")
        block_result = {"mass": {
            "original_probability_per_context_query_head": original_sum / (heads * 64),
            "same_slot_removed_share_of_original_context_query_allkey_mass": removed_sum / original_sum,
            "same_slot_removed_share_of_original_context_key_mass": removed_sum / float(context[..., :64].sum()),
            "remaining_allkey_share_of_original_context_query_allkey_mass": remaining_sum / original_sum,
            "remaining_context_key_share_of_original_context_query_allkey_mass": float(off[..., :64].sum()) / original_sum,
            "unchanged_target_key_share_of_original_context_query_allkey_mass": float(off[..., 64:].sum()) / original_sum,
            "target_query_share_of_original_context_received_mass": float(t_received[:, :64].sum() / (c_received[:, :64].sum() + t_received[:, :64].sum())),
            "target_query_share_of_residual_context_received_mass": float(t_received[:, :64].sum() / (o_received[:, :64].sum() + t_received[:, :64].sum()))},
            "temporal": {name: temporal(value, queries) for name, value, queries in (
                ("context_original", c_received, 16384), ("context_offslot", o_received, 16384),
                ("allquery_original", c_received + t_received, 16640), ("allquery_residual", o_received + t_received, 16640),
                ("targetquery_unchanged", t_received, 256))}, "flow": {}}
        for name, value in (("original", context), ("offslot", off)):
            pooled = value.mean(0)[:, :64]
            conditional = normalized(pooled)
            distance = abs(ix[:, None] - ix[None, :])
            block_result["flow"][name] = {
                "head_pooled_raw_context_matrix": pooled.tolist(),
                "head_pooled_row_context_conditional_matrix": conditional.tolist(),
                "adjacent_only_absolute_per_source_query_head": float((pooled * (distance == 1)).sum(-1).mean()),
                "adjacent_only_after_pooled_row_context_conditioning": float((conditional * (distance == 1)).sum(-1).mean()),
                "order": "Stored flow first averages samples and 256 queries per source slot; then average heads; conditional matrix normalizes each resulting row over context keys. It cannot reconstruct per-token-query renormalization."}
        results[f"L{block}"] = block_result
        saved[f"original_raw_flow_L{block}"] = flow
        saved[f"context_offslot_raw_flow_L{block}"] = off
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = {"evaluation_protocol": PROTOCOL, "scope": "existing-artifact pooled diagnostic; no accuracy evaluation",
        "analysis_job_id": os.environ.get("SLURM_JOB_ID"), "metric_scope": "native",
        "eval_path": "Full4000 EGTEA split1 ctx16 existing unconditioned mean flow, post-softmax same-slot readout deletion",
        "source_gpu_job_id": metadata["job_id"], "source_analysis_job_id": "17007430",
        "source_dir": str(source), "source_sha256": {str(source / name): sha(source / name) for name in ("metadata.json", "summary.json", "attention_profiles.npz")},
        "analysis_source_sha256": {str(Path(__file__).resolve()): sha(__file__)},
        "manifest_sha256": metadata["manifest_sha256"], "n_source_rows": 4000,
        "parent_checkpoint_paths": metadata["checkpoint_paths"], "gates": "All source layout, row mass, source group/flow agreement, diagonal deletion, unchanged target column and raw conservation checks passed.",
        "blocks": results, "limits": ["Pooled descriptive result only: no per-sample residual distributions, uncertainty intervals, spatial token rankings, or self-token split recoverable.",
            "No model attention change; deleting readout terms does not establish causal utility.",
            "Aggregate conditional normalization is not per-token-query renormalization.",
            "This validation-pooled profile is not a training-calibrated offline selector."]}
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    brief = {"analysis_job_id": report["analysis_job_id"], "scope": report["scope"], "gates": report["gates"], "blocks": {
        name: {"mass": value["mass"], "temporal": {kind: {key: item[key] for key in (
            "context_conditional_entropy_div_log64", "context_conditional_recent8_fraction", "context_conditional_recent16_fraction", "context_conditional_oldest8_fraction")}
            for kind, item in value["temporal"].items()}} for name, value in results.items()}}
    (args.out_dir / "brief.json").write_text(json.dumps(brief, indent=2) + "\n")
    np.savez_compressed(args.out_dir / "pooled_flow_readouts.npz", **saved)
    figures(args.out_dir, results)
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
