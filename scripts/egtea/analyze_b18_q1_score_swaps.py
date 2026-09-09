#!/usr/bin/env python3
"""Independent B18 score-cohort and swap-utility analysis, CPU Slurm only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_b18_q1_position_policy import read, require, sha, source_manifest

PROTOCOL = "b18-predictor-prune/egtea-ctx16-score-swaps-v1"
BLOCKS = [0, 11]
HORIZONS = ["2s", "4s", "6s"]
SOURCES = ["self", "same_slot_other", "cross_slot_context", "target_query"]
SEEDS = [1701, 1702, 1703]
BOOTSTRAP_SEED = 2026090613
N, KEEP, GP = 16384, 4096, 256
COHORTS = ["incoming_online_only", "outgoing_offline_only", "shared", "neither"]
RANK_GRID = np.array([1, 16, 64, 256, 512, 1024, 2048, 3072, 3584, 4096,
                      4608, 5120, 6144, 8192, 12288, 16384])
GAP_GRID = np.r_[-np.geomspace(10, .001, 100), 0., np.geomspace(.001, 10, 100)]
NEAR_BANDS = [.02, .05, .10]
VARIANTS = ["offline", "online", "near", "strong"] + [f"random_seed{s}" for s in SEEDS]
ARMS = [f"original__L{b}__{v}" for b in BLOCKS for v in VARIANTS]
ARMS += [f"packed__L{b}__{v}" for b in BLOCKS for v in ["offline", "online"]]
ARMS += ["original__recent"]


def ranking(scores):
    """Exact tie intervals and deterministic descending-score/index ordinal."""
    s = np.asarray(scores, dtype=np.float64)
    require(s.shape == (N,) and np.isfinite(s).all() and (s >= 0).all(), "Invalid rank score vector")
    order = np.argsort(-s, kind="stable")
    ordinal = np.empty(N, dtype=np.int32); ordinal[order] = np.arange(1, N + 1)
    ascending = np.sort(s)
    rank_min = N - np.searchsorted(ascending, s, side="right") + 1
    rank_max = N - np.searchsorted(ascending, s, side="left")
    cut = float((s[order[KEEP - 1]] + s[order[KEEP]]) / 2)
    mean = float(s.mean())
    require(cut > 0 and mean > 0, "Zero cutoff/mean: relative margin undefined")
    gap = s - cut
    return {"score": s, "order": order, "ordinal": ordinal, "rank_min": rank_min,
            "rank_max": rank_max, "gap_raw": gap, "gap_relative_cutoff": gap / cut,
            "gap_mean_normalized": gap / mean, "cutoff": cut, "mean_score": mean,
            "score_k": float(s[order[KEEP - 1]]), "score_k_plus_1": float(s[order[KEEP]]),
            "boundary_tied": bool(s[order[KEEP - 1]] == s[order[KEEP]])}


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if not x.size:
        return None
    return {"mean": float(x.mean()), "min": float(x.min()), "p05": float(np.quantile(x, .05)),
            "median": float(np.median(x)), "p95": float(np.quantile(x, .95)), "max": float(x.max())}


def rank_summary(info, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return {"tokens": 0}
    result = {"tokens": int(len(indices)), "cutoff": info["cutoff"],
              "mean_score": info["mean_score"], "boundary_tied": info["boundary_tied"]}
    for name in ("score", "rank_min", "rank_max", "ordinal", "gap_raw", "gap_relative_cutoff", "gap_mean_normalized"):
        result[name] = stats(info[name][indices])
    rel = info["gap_relative_cutoff"][indices]
    result["near_cutoff_fraction"] = [float((np.abs(rel) <= band).mean()) for band in NEAR_BANDS]
    result["ordinal_cdf"] = (np.searchsorted(np.sort(info["ordinal"][indices]), RANK_GRID, side="right") / len(indices)).tolist()
    result["relative_gap_cdf"] = (np.searchsorted(np.sort(rel), GAP_GRID, side="right") / len(indices)).tolist()
    result["tied_rank_fraction"] = float((info["rank_min"][indices] != info["rank_max"][indices]).mean())
    return result


def plots(out, diagnostic, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode in ["online", "offline"]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for bi, block in enumerate(BLOCKS):
            for cohort in COHORTS:
                r = diagnostic["rank"][f"L{block}/{cohort}/{mode}"]
                if not r["nonempty_rows"]:
                    continue
                axes[bi, 0].plot(RANK_GRID, r["equal_nonempty_row_ordinal_cdf"], label=cohort)
                axes[bi, 1].plot(GAP_GRID, r["equal_nonempty_row_relative_gap_cdf"], label=cohort)
            axes[bi, 0].axvline(4096, color="gray", linewidth=1)
            axes[bi, 0].set(title=f"L{block}: {mode} ranks", xlabel="Deterministic ordinal, 1 highest", ylabel="Within-cohort CDF, equal receiver mean")
            axes[bi, 1].axvline(0, color="gray", linewidth=1)
            axes[bi, 1].set(title=f"L{block}: {mode} cutoff distance", xlabel="(score − own midpoint cutoff) / cutoff", ylabel="Within-cohort CDF, equal receiver mean", xscale="symlog")
            for ax in axes[bi]:
                ax.legend(fontsize=7)
        fig.suptitle(f"Online/offline full top4096 disagreement cohorts, scored by {mode}\nEach nonempty receiver cohort normalized first; four-source quantities are a separate current-sample readout.", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out / f"cohort_rank_margin_cdf_{mode}.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for bi, block in enumerate(BLOCKS):
        for ci, cohort in enumerate(COHORTS[:2]):
            values = np.asarray(diagnostic["sources"][f"L{block}/{cohort}"]["mean_receiver_component_mass_per_slot"])
            ax = axes[bi, ci]
            ax.stackplot(np.arange(64), values, labels=SOURCES)
            ax.set(title=f"L{block}: {cohort}", xlabel="Original context key slot", ylabel="Raw received mass, receiver mean")
            ax.legend(fontsize=7)
    fig.suptitle("Four current-validation source components on exchanged-candidate cohorts\nMass summed over cohort tokens and heads; no claim about historical training-map components.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out / "incoming_outgoing_sources_by_slot.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for bi, block in enumerate(BLOCKS):
        for ci, cohort in enumerate(COHORTS[:2]):
            r = diagnostic["sources"][f"L{block}/{cohort}"]["online_cutoff_signed_means"]
            ax = axes[bi, ci]
            if r["nonempty_receivers"]:
                values = r["equal_nonempty_receiver_mean"]
                ax.bar(range(5), values, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "gray"])
                ax.axhline(0, color="black", linewidth=.8)
            ax.set_xticks(range(5), ["self", "same-slot\nother", "cross-slot\ncontext", "target", "reduction\nresidual"])
            ax.set(title=f"L{block}: {cohort}", ylabel="Signed score distance from online midpoint cutoff")
    fig.suptitle("Online cutoff distance decomposed into four current-sample sources plus reduction residual\nBoundary reference averages components at online ranks4096/4097; receiver means weighted equally. Offline cutoff is not decomposed.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out / "cohort_online_cutoff_sources.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = {"near": "tab:blue", "strong": "tab:orange", "random_seed1701": "tab:green", "random_seed1702": "tab:red", "random_seed1703": "tab:purple"}
    for bi, block in enumerate(BLOCKS):
        for side, style in [("incoming", "-"), ("outgoing", "--")]:
            for variant in VARIANTS[2:]:
                r = diagnostic["rank"][f"L{block}/{variant}/{side}/online"]
                if r["nonempty_rows"]:
                    axes[bi, 0].plot(RANK_GRID, r["equal_nonempty_row_ordinal_cdf"], label=f"{variant}: {side}", color=colors[variant], linestyle=style)
                    axes[bi, 1].plot(GAP_GRID, r["equal_nonempty_row_relative_gap_cdf"], label=f"{variant}: {side}", color=colors[variant], linestyle=style)
        axes[bi, 0].axvline(4096, color="gray", linewidth=1)
        axes[bi, 0].set(title=f"L{block} actual swap ranks", xlabel="Online deterministic ordinal", ylabel="CDF, equal nonempty receiver mean")
        axes[bi, 1].axvline(0, color="gray", linewidth=1)
        axes[bi, 1].set(title=f"L{block} actual swap margins", xlabel="Online relative midpoint-cutoff gap", ylabel="CDF, equal nonempty receiver mean", xscale="symlog")
        axes[bi, 1].legend(fontsize=6, ncol=2)
    fig.suptitle("Near is the eligible pool's low-preference-gap end, not assumed globally near cutoff\nSolid=incoming, dashed=outgoing; every arm has identical per-slot swap quotas.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out / "swap_rank_margin_cdf.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for block in BLOCKS:
        cov = diagnostic["coverage"][f"L{block}"]
        axes[0].plot(cov["mean_capacity_per_slot"], label=f"L{block}: eligible capacity")
        axes[0].plot(cov["mean_quota_per_slot"], "--", label=f"L{block}: actual swaps")
    axes[0].set(xlabel="Original slot", ylabel="Token count, receiver mean", title="Matched candidate coverage"); axes[0].legend(fontsize=8)
    names = paired["primary"] + ["L0__strong_minus_near", "L0__strong_minus_offline"]
    for i, name in enumerate(names):
        r = paired["contrasts"][name]["2s"]; ci = r["video_id"]["top5_95ci_pp"]
        axes[1].hlines(i, *ci, color="tab:blue", linewidth=3)
        if name in paired["primary"]:
            axes[1].hlines(i, *r["video_id"]["top5_97_5ci_pp_bonferroni_two_primary"], color="tab:blue", linewidth=1)
        axes[1].plot(r["top5_delta_pp"], i, "o", color="tab:blue")
    axes[1].set_yticks(range(len(names)), names); axes[1].invert_yaxis(); axes[1].axvline(0, color="gray", linewidth=1)
    axes[1].set(xlabel="Action Top-5 difference, percentage points", title="+2 s paired session intervals\nThick95%; thin97.5% for two L11 primaries")
    fig.tight_layout(); fig.savefig(out / "swap_coverage_and_utility.png", dpi=170); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--bootstrap-reps", type=int, default=9999)
    args = p.parse_args()
    metas, summaries, rows, parts, offline, correct, loss, valid, coverage, numeric = load(args.runs, args.allow_partial)
    require((valid.sum(0) > 0).all(), "No valid primary/supporting labels")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    diagnostic = diagnostics(rows, parts, offline, args.out_dir)
    weights, primary = definitions()
    paired = paired_bootstrap(rows, correct, loss, valid, ARMS, weights, primary, args.bootstrap_reps)
    results = {a: {h: {"n": int(valid[:, hi].sum()), "correct": int(correct[:, hi, ai].sum()),
                       "action_top5": float(correct[:, hi, ai].sum() / valid[:, hi].sum()),
                       "mean_ce": float(loss[:, hi, ai].sum() / valid[:, hi].sum())} for hi, h in enumerate(HORIZONS)} for ai, a in enumerate(ARMS)}
    random = {f"L{b}": {h: {"per_seed_action_top5": [results[f"original__L{b}__random_seed{s}"][h]["action_top5"] for s in SEEDS],
                           "mean_action_top5": float(np.mean([results[f"original__L{b}__random_seed{s}"][h]["action_top5"] for s in SEEDS])),
                           "seed_sd_pp": float(100 * np.std([results[f"original__L{b}__random_seed{s}"][h]["action_top5"] for s in SEEDS], ddof=1))}
                      for h in HORIZONS} for b in BLOCKS}
    result = {"evaluation_protocol": PROTOCOL, "metric_scope": "native", "eval_path": metas[0]["eval_path"],
              "analysis_job_id": os.environ.get("SLURM_JOB_ID"), "input_jobs": [m["job_id"] for m in metas],
              "input_dirs": [str(x) for x in args.runs], "analysis_source_sha256": {str(Path(__file__)): sha(__file__),
                  str(Path(__file__).with_name("analyze_b18_q1_position_policy.py")): sha(Path(__file__).with_name("analyze_b18_q1_position_policy.py"))},
              "smoke_only": args.allow_partial, "coverage": coverage, "gates": "passed", "numeric_checks": numeric,
              "producer_gates": [s["gates"] for s in summaries], "results": results, "paired_inference": paired,
              "observed_capture_dtypes": [s.get("observed_capture_dtypes", {"status": "not_recorded_in_this_producer_run"}) for s in summaries],
              "random_seed_summary": random, "diagnostics": diagnostic,
              "exclusions": {h: dict(Counter(r["label_validity"][h] for r in rows)) for h in HORIZONS},
              "limits": ["Current VAL receiver source components do not reconstruct offline TRAIN calibration components.",
                         "Relative margins use each scorer's own scale and midpoint cutoff; zero denominators fail explicitly.",
                         "Near/strong/random swaps preserve F temporal quotas and use only matched within-slot disagreements.",
                         "Reduction sensitivity uses the same softmax probabilities at their observed dtype; legacy and FP32-cast reductions can be identical. It does not estimate all numerical/stochastic score uncertainty.",
                         "No score-utility correlation is treated as an independent causal mechanism; actual mask interventions and primary contrasts are reported separately."]}
    np.savez_compressed(args.out_dir / "paired_outcomes.npz", correct=correct, ce=loss, valid=valid, arms=np.array(ARMS))
    (args.out_dir / "merged_execution_order.json").write_text(json.dumps([{k: r[k] for k in ["sample_id", "video_id", "participant_id", "source_index", "selection_index"]} for r in rows], indent=2) + "\n")
    plots(args.out_dir, diagnostic, paired)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    brief = {"analysis_job_id": result["analysis_job_id"], "smoke_only": args.allow_partial, "coverage": coverage, "gates": "passed",
             "swap_coverage": diagnostic["coverage"], "reduction_sensitivity": diagnostic["reduction_sensitivity"],
             "primary_2s": {name: paired["contrasts"][name]["2s"] for name in primary}, "results": results, "numeric_checks": numeric}
    (args.out_dir / "brief.json").write_text(json.dumps(brief, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"analysis_job_id": result["analysis_job_id"], "gates": "passed", "smoke_only": args.allow_partial, "coverage": coverage, "output": str(args.out_dir)}, indent=2), flush=True)


class CohortAccumulator:
    """Equal-example summaries and CDFs; empty cohorts are explicitly counted."""
    def __init__(self):
        self.rows = 0; self.nonempty = 0; self.tokens = 0
        self.counts = []; self.fields = defaultdict(list)
        self.rank_cdf = np.zeros(len(RANK_GRID)); self.gap_cdf = np.zeros(len(GAP_GRID))
        self.bands = np.zeros(len(NEAR_BANDS))

    def add(self, summary):
        self.rows += 1; self.tokens += summary["tokens"]; self.counts.append(summary["tokens"])
        if not summary["tokens"]:
            return
        self.nonempty += 1
        for name in ("score", "rank_min", "rank_max", "ordinal", "gap_raw", "gap_relative_cutoff", "gap_mean_normalized"):
            for metric, value in summary[name].items():
                self.fields[name + "/" + metric].append(value)
        self.fields["tied_rank_fraction"].append(summary["tied_rank_fraction"])
        self.rank_cdf += summary["ordinal_cdf"]; self.gap_cdf += summary["relative_gap_cdf"]
        self.bands += summary["near_cutoff_fraction"]

    def result(self):
        return {"rows": self.rows, "nonempty_rows": self.nonempty, "empty_rows": self.rows - self.nonempty,
                "total_tokens": self.tokens, "per_row_token_count": stats(self.counts),
                "per_row_stat_distributions": {k: stats(v) for k, v in self.fields.items()},
                "equal_nonempty_row_ordinal_cdf": (self.rank_cdf / self.nonempty).tolist() if self.nonempty else None,
                "equal_nonempty_row_relative_gap_cdf": (self.gap_cdf / self.nonempty).tolist() if self.nonempty else None,
                "equal_nonempty_row_near_cutoff_fractions": (self.bands / self.nonempty).tolist() if self.nonempty else None,
                "weighting": "Each nonempty receiver cohort contributes one normalized CDF, then receivers are averaged equally. Quantiles describe each receiver's statistic, not pooled token quantiles. Empty receiver cohorts are counted and omitted only for undefined within-cohort statistics."}


def source_summary(components, indices):
    indices = np.asarray(indices, dtype=np.int64)
    selected = components[:, indices]
    totals = selected.sum(-1, dtype=np.float64)
    temporal = np.stack([np.bincount(np.asarray(indices) // GP, weights=selected[i], minlength=64) for i in range(4)])
    return {"tokens": int(len(indices)), "component_sum": totals.tolist(),
            "component_mean_per_token": (totals / len(indices)).tolist() if len(indices) else None,
            "component_fraction_of_cohort_mass": (totals / totals.sum()).tolist() if totals.sum() > 0 else None,
            "component_mass_per_slot": temporal.tolist()}


def reduction_diagnostic(legacy, total_fp32):
    precise = ranking(total_fp32)
    a, b = legacy["order"][:KEEP], precise["order"][:KEEP]
    overlap = int(np.intersect1d(a, b, assume_unique=True).size)
    delta = np.asarray(total_fp32, dtype=np.float64) - legacy["score"]
    return {"same_probability_top4096_overlap": overlap,
            "same_probability_top4096_jaccard": overlap / (2 * KEEP - overlap),
            "legacy_cutoff": legacy["cutoff"], "fp32_sum_cutoff": precise["cutoff"],
            "cutoff_difference_relative_to_legacy": (precise["cutoff"] - legacy["cutoff"]) / legacy["cutoff"],
            "fp32_minus_legacy_raw": stats(delta), "mean_absolute_rank_change": float(np.abs(precise["ordinal"] - legacy["ordinal"]).mean()),
            "legacy_only_indices": np.setdiff1d(a, b, assume_unique=True).tolist(),
            "fp32_only_indices": np.setdiff1d(b, a, assume_unique=True).tolist(),
            "scope": "Same softmax values at the dtype recorded by the producer; explicit FP32-cast total versus legacy query reduction. These can use the same actual precision. This is conditional reduction sensitivity, not total score noise and not a separate utility arm."}


def paired_bootstrap(rows, correct, losses, valid, arms, definitions, primary, reps):
    names = list(definitions)
    weights = np.array([[definitions[name].get(a, 0.) for a in arms] for name in names])
    values = np.stack([correct @ weights.T, losses @ weights.T], -1)
    point = values.sum(0) / valid.sum(0)[:, None, None]
    result = {name: {h: {"n": int(valid[:, hi].sum()), "top5_delta_pp": float(100 * point[hi, ci, 0]),
                           "ce_delta": float(point[hi, ci, 1])} for hi, h in enumerate(HORIZONS)} for ci, name in enumerate(names)}
    for key, offset in (("video_id", 0), ("participant_id", 1)):
        clusters = sorted({r[key] for r in rows}); index = {x: i for i, x in enumerate(clusters)}
        ids = np.array([index[r[key]] for r in rows]); g = len(clusters)
        sums = np.zeros((g,) + values.shape[1:]); denom = np.zeros((g, 3))
        np.add.at(sums, ids, values); np.add.at(denom, ids, valid)
        rng = np.random.default_rng(BOOTSTRAP_SEED + offset); draws = np.empty((reps,) + point.shape)
        for start in range(0, reps, 256):
            stop = min(start + 256, reps)
            multiplicity = rng.multinomial(g, np.full(g, 1 / g), size=stop - start)
            d = multiplicity @ denom; require((d > 0).all(), "Bootstrap empty valid horizon")
            draws[start:stop] = (multiplicity @ sums.reshape(g, -1)).reshape((stop - start,) + point.shape) / d[:, :, None, None]
        ci = np.quantile(draws, [.025, .975], axis=0)
        wide = np.quantile(draws, [.0125, .9875], axis=0) if key == "video_id" else None
        for ni, name in enumerate(names):
            for hi, h in enumerate(HORIZONS):
                entry = {"cluster_n": g, "top5_95ci_pp": (100 * ci[:, hi, ni, 0]).tolist(), "ce_95ci": ci[:, hi, ni, 1].tolist()}
                if key == "video_id" and name in primary and h == "2s":
                    entry["top5_97_5ci_pp_bonferroni_two_primary"] = (100 * wide[:, hi, ni, 0]).tolist()
                result[name][h][key] = entry
    return {"primary": primary, "primary_horizon": "2s", "seed": BOOTSTRAP_SEED, "reps": reps,
            "definitions": definitions, "contrasts": result,
            "method": "Paired session-cluster percentile bootstrap of ratio-of-sums, participant sensitivity; each receiver has equal outcome weight, all mask outcomes stay paired. Random seeds averaged within receiver.",
            "multiplicity": "95% marginal intervals throughout; two prespecified L11 Top5@2 primary contrasts additionally receive two-sided97.5% session intervals as Bonferroni support. Bootstrap coverage is approximate; secondary comparisons are not familywise-adjusted.",
            "limits": ["Repeated exploratory reuse of validation, not independent confirmation.", "Strong minus near may mean less harm; strong minus offline is separately required to support actual benefit.", "Swaps preserve offline temporal counts and cover only within-slot matched disagreements; conclusions do not automatically apply to full online versus offline allocation.", "Offline calibration map remains fixed; bootstrap omits calibration-set uncertainty.", "Rank stability, component residuals and task utility are distinct estimands."]}


def definitions():
    result = {}
    for b in [11, 0]:
        prefix = f"original__L{b}__"
        for name, lhs, rhs in (("strong_minus_near", "strong", "near"),
                               ("strong_minus_offline", "strong", "offline"),
                               ("near_minus_offline", "near", "offline"),
                               ("online_minus_offline", "online", "offline")):
            result[f"L{b}__{name}"] = {prefix + lhs: 1., prefix + rhs: -1.}
        for variant in ["strong", "near", "offline"]:
            result[f"L{b}__{variant}_minus_random_mean3"] = {prefix + variant: 1., **{prefix + f"random_seed{s}": -1 / 3 for s in SEEDS}}
        result[f"L{b}__packed_online_minus_offline"] = {f"packed__L{b}__online": 1., f"packed__L{b}__offline": -1.}
    return result, ["L11__strong_minus_near", "L11__strong_minus_offline"]


def load(runs, partial):
    runs = sorted(runs, key=lambda p: read(p / "metadata.json")["start"])
    metas = [read(p / "metadata.json") for p in runs]; common = metas[0]
    for m in metas:
        require(m["evaluation_protocol"] == PROTOCOL and m["arms"] == ARMS and m["blocks"] == BLOCKS and m["sources"] == SOURCES, "Protocol/axes mismatch")
        require(m["selection"]["keep"] == KEEP and m["selection"]["anchor_slots"] == 0, "Expected full-domain top4096, not hybrid")
        require(m["selection"]["tie_rule"] == "stable descending score, original index ascending", "Unexpected chosen ordinal rule")
        require(m["random_seeds"] == SEEDS, "Random seed drift")
        for key in ("manifest_sha256", "train_csv_sha256", "val_csv_sha256", "checkpoint_paths", "checkpoint_file_identity", "source_sha256", "calibration_sha256", "selection", "token_layout", "reference_score_sha256"):
            require(m[key] == common[key], f"Cross-shard {key} mismatch")
    layout = common["token_layout"]
    for key, value in {"n_context_tokens": N, "n_target_tokens": GP, "tokens_per_slot": GP,
                       "target_array_start_slot": 64, "original_target_rope_start_slot": 72,
                       "packed_target_rope_start_slot": 24}.items():
        require(layout[key] == value, f"Unexpected token layout: {key}")
    heads_by_block = layout["num_heads"]
    require(set(heads_by_block) == {str(b) for b in BLOCKS}, "Head metadata block mismatch")
    num_heads = heads_by_block[str(BLOCKS[0])]
    require(isinstance(num_heads, int) and num_heads > 0 and
            all(heads_by_block[str(b)] == num_heads for b in BLOCKS), "Inconsistent block head counts")
    for path, value in common["source_sha256"].items():
        require(sha(path) == value, f"Changed frozen source {path}")
    require(common["checkpoint_paths"] == [common["arguments"][k] for k in ("checkpoint", "init_from_ckpt", "encoder_lora", "predictor_lora")], "Actual checkpoint path mismatch")
    for path, identity in common["checkpoint_file_identity"].items():
        st = Path(path).stat()
        require(identity == {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}, "Checkpoint file identity drift")
    expected, source, vocabulary = source_manifest(common)
    reference_dir = Path(common["reference_dir"])
    reference_meta = read(reference_dir / "metadata.json")
    require(reference_meta["evaluation_protocol"] == "b18-predictor-prune/egtea-ctx16-paired-hybrid-v1", "Historical source protocol mismatch")
    reference = {r["sample_id"]: r for r in (json.loads(x) for x in (reference_dir / "predictions.jsonl").read_text().splitlines())}
    reference_scores = np.load(reference_dir / "scores.npy", mmap_mode="r")
    require(sha(reference_dir / "scores.npy") == common["reference_score_sha256"], "Historical score hash changed")
    offline = []
    for block, key in zip(BLOCKS, ["calib_l0", "calib_l11"]):
        path = Path(common["arguments"][key]); digest = sha(path)
        require(digest == common["calibration_sha256"][str(path)] == reference_meta["calibration_sha256"][str(path)], "Offline calibration map changed")
        cm = read(path.with_name(path.name.replace("_map_64x256.npy", "_meta.json")))
        require(cm["block"] == block and cm["n_used"] == 512 and cm["calib_csv"] == common["arguments"]["train_csv"], "Offline train calibration identity mismatch")
        offline.append(np.load(path).reshape(-1))
    offline = np.stack(offline)
    all_rows, parts, corrects, losses, valids, positions, summaries = [], [], [], [], [], [], []
    numeric = {"ce_max_abs_reconstruction": 0., "ambiguous_class_top5_ties": 0, "component_total_max_abs": 0.,
               "historical_score_max_abs": 0., "historical_packed": {}}
    for path, meta in zip(runs, metas):
        summary = read(path / "summary.json"); order = read(path / "execution_order.json")
        rows = [json.loads(x) for x in (path / "predictions.jsonl").read_text().splitlines()]
        start, stop = meta["start"], meta["stop"]; n = stop - start
        require(start % 4 == stop % 4 == 0 and n > 0 and len(rows) == len(order) == summary["n_rows"] == n, "Incomplete/split shard")
        require(summary["evaluation_protocol"] == PROTOCOL, "Summary protocol mismatch")
        observed_dtypes = summary.get("observed_capture_dtypes")
        if not partial:
            require(observed_dtypes is not None and set(observed_dtypes) == {str(b) for b in BLOCKS}, "Full run must record observed capture dtypes")
            require(all(all(k in observed_dtypes[str(b)] for k in ["prob", "legacy_query_reduction", "fp32_query_reduction", "legacy_accumulator", "fp32_accumulator"]) for b in BLOCKS), "Incomplete observed reduction dtypes")
        if summaries:
            require(observed_dtypes == summaries[0].get("observed_capture_dtypes"), "Cross-shard capture dtype mismatch")
        data = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in summary["artifact_sha256"]}
        require(all(sha(path / f"{name}.npy") == digest for name, digest in summary["artifact_sha256"].items()), "Saved artifact hash mismatch")
        shapes = {"legacy_online_scores": (n, 2, N), "source_total_fp32": (n, 2, N),
                  "source_components_fp32": (n, 2, 4, N), "mask_indices": (n, 19, KEEP),
                  "swap_quotas": (n, 2, 64), "swap_capacities": (n, 2, 64),
                  "action_logits": (n, 19, 3, len(vocabulary)), "source_head_temporal": (n, 2, 4, num_heads, 64)}
        for name, shape in shapes.items():
            require(data[name].shape == shape, f"Bad {name} shape")
            dtype = np.int32 if name == "mask_indices" else np.int16 if name.startswith("swap_") else np.float32
            require(data[name].dtype == dtype, f"Bad {name} dtype")
        require(np.array_equal(data["offline_scores"], offline), "Offline stored score mismatch")
        require(np.array_equal(data["target_positions"], np.arange(256)[None, :] + np.array([18432, 6144])[:, None]), "Target coordinates mismatch")
        if meta["arguments"]["parity_check"]:
            require(all(summary["gates"][key] is True for key in ["capture_output_exact", "legacy_score_exact", "packed_helper_exact"]), "Missing same-device smoke parity")
        correct = np.zeros((n, 3, 19)); loss = np.zeros_like(correct); valid = np.zeros((n, 3), dtype=bool)
        for ri, (r, o) in enumerate(zip(rows, order)):
            require(all(r[k] == v for k, v in o.items()) and all(r[k] == v for k, v in expected[start + ri].items()), "Execution/manifest identity mismatch")
            require(set(r["arms"]) == set(ARMS), "Missing/extra arm")
            comps = np.asarray(data["source_components_fp32"][ri], dtype=np.float64)
            total = np.asarray(data["source_total_fp32"][ri], dtype=np.float64)
            legacy = np.asarray(data["legacy_online_scores"][ri], dtype=np.float64)
            heads = np.asarray(data["source_head_temporal"][ri], dtype=np.float64)
            require(all(np.isfinite(x).all() and (x >= 0).all() for x in [comps, total, legacy, heads]), "Invalid attention readout")
            require(np.allclose(comps.sum(1), total, rtol=3e-6, atol=3e-5), "FP32 sources do not sum to FP32 total")
            require(np.allclose(comps.reshape(2, 4, 64, GP).sum(-1), heads.sum(2), rtol=3e-6, atol=3e-4), "Head/slot source layout mismatch")
            numeric["component_total_max_abs"] = max(numeric["component_total_max_abs"], float(np.abs(comps.sum(1) - total).max()))
            numeric["historical_score_max_abs"] = max(numeric["historical_score_max_abs"], float(np.abs(legacy - reference_scores[start + ri]).max()))
            src = source[r["source_index"]]
            for hi, h in enumerate(HORIZONS):
                pair = (int(src["mtp_verbs"].split(",")[hi]), int(src["mtp_nouns"].split(",")[hi]))
                state = "masked" if float(src["mtp_mask"].split(",")[hi]) <= .5 else "valid" if pair in vocabulary else "out_of_training_vocabulary"
                require(r["label_validity"][h] == state, "Source label validity mismatch")
                valid[ri, hi] = state == "valid"
            logits = np.asarray(data["action_logits"][ri], dtype=np.float64)
            require(np.isfinite(logits).all(), "Nonfinite class logits")
            for ai, arm in enumerate(ARMS):
                index = data["mask_indices"][ri, ai]
                require((np.diff(index) > 0).all() and index[0] >= 0 and index[-1] < N, "Unsorted/out-of-range mask")
                item = r["arms"][arm]
                require(item["keep_per_slot"] == np.bincount(index // GP, minlength=64).tolist(), "Recorded temporal allocation differs from actual indices")
                packed = arm.startswith("packed")
                require(item["context_position_first"] == (0 if packed else int(index[0])) and item["context_position_last"] == (4095 if packed else int(index[-1])), "Context coordinates mismatch")
                require(item["target_position_first"] == (6144 if packed else 18432) and item["target_position_last"] == (6399 if packed else 18687), "Prediction target coordinates mismatch")
                for hi, h in enumerate(HORIZONS):
                    metric = item["metrics"].get(h)
                    require((metric is not None) == valid[ri, hi], "Arm exclusion differs")
                    if metric is None:
                        continue
                    label = vocabulary[(int(src["mtp_verbs"].split(",")[hi]), int(src["mtp_nouns"].split(",")[hi]))]
                    require(metric["label"] == label and isinstance(metric["top5"], bool), "Invalid metric label/type")
                    v = logits[ai, hi]; top = v.max(); ce = np.log(np.exp(v - top).sum()) + top - v[label]
                    error = abs(float(ce) - metric["ce"])
                    require(np.isfinite(metric["ce"]) and error <= 1e-5, "Saved CE/logit mismatch")
                    numeric["ce_max_abs_reconstruction"] = max(numeric["ce_max_abs_reconstruction"], error)
                    greater, equal = int((v > v[label]).sum()), int((v == v[label]).sum())
                    if greater >= 5:
                        require(not metric["top5"], "Impossible Top5 success")
                    elif greater + equal <= 5:
                        require(metric["top5"], "Missing certain Top5 success")
                    else:
                        numeric["ambiguous_class_top5_ties"] += 1
                    correct[ri, hi, ai] = metric["top5"]; loss[ri, hi, ai] = metric["ce"]
                    if packed:
                        _, layer, variant = arm.split("__")
                        prior = reference[r["sample_id"]]["arms"][f"{variant}_{layer}"]["metrics"][h]
                        audit = numeric["historical_packed"].setdefault(arm, {}).setdefault(h, {"n": 0, "top5_disagreements": 0, "ce_max_abs": 0.})
                        audit["n"] += 1; audit["top5_disagreements"] += metric["top5"] != prior["top5"]
                        audit["ce_max_abs"] = max(audit["ce_max_abs"], abs(metric["ce"] - prior["ce"]))
        for ai, arm in enumerate(ARMS):
            for hi, h in enumerate(HORIZONS):
                s = summary["results"][arm]
                require(s.get(f"n@{h}", 0) == valid[:, hi].sum() and s.get(f"correct@{h}", 0) == correct[:, hi, ai].sum(), "GPU aggregate count mismatch")
                require(np.isclose(s.get(f"ce_sum@{h}", 0), loss[:, hi, ai].sum(), rtol=1e-7, atol=1e-5), "GPU aggregate CE mismatch")
        positions.extend(range(start, stop)); all_rows.extend(rows); parts.append(data); summaries.append(summary)
        corrects.append(correct); losses.append(loss); valids.append(valid)
    require(len(set(positions)) == len(all_rows) == len({r["sample_id"] for r in all_rows}), "Overlapping rows")
    if not partial:
        require(positions == list(range(4000)), "Incomplete full4000 coverage")
    coverage = {"rows": len(all_rows), "manifest_rows": 4000, "complete": positions == list(range(4000)),
                "sessions": len({r["video_id"] for r in all_rows}), "participant_ids": len({r["participant_id"] for r in all_rows}),
                "ranges": [[m["start"], m["stop"]] for m in metas], "manifest_sha256": common["manifest_sha256"],
                "calibration_sha256": common["calibration_sha256"]}
    return metas, summaries, all_rows, parts, offline, np.concatenate(corrects), np.concatenate(losses), np.concatenate(valids), coverage, numeric


def expected_swaps(online, offline, sample, block):
    fm = np.zeros(N, dtype=bool); om = np.zeros(N, dtype=bool)
    fm[offline["order"][:KEEP]] = True; om[online["order"][:KEEP]] = True
    incoming, outgoing = np.flatnonzero(om & ~fm), np.flatnonzero(fm & ~om)
    cohorts = [incoming, outgoing, np.flatnonzero(fm & om), np.flatnonzero(~fm & ~om)]
    require(len(incoming) == len(outgoing), "Unequal full-set disagreement sides")
    additions = {v: [] for v in VARIANTS[2:]}; removals = {v: [] for v in VARIANTS[2:]}
    rngs = {s: np.random.default_rng(int(hashlib.sha256(f"score-swaps|{s}|{block}|{sample['sample_id']}".encode()).hexdigest()[:16], 16)) for s in SEEDS}
    quotas, capacities = [], []
    for slot in range(64):
        ii = incoming[incoming // GP == slot]; ii = ii[np.argsort(online["ordinal"][ii])]
        ee = outgoing[outgoing // GP == slot]; ee = ee[np.argsort(online["ordinal"][ee])]
        cap = min(len(ii), len(ee)); q = cap // 2
        quotas.append(q); capacities.append(cap)
        if not q:
            continue
        additions["near"].extend(ii[-q:]); removals["near"].extend(ee[:q])
        additions["strong"].extend(ii[:q]); removals["strong"].extend(ee[-q:])
        for seed in SEEDS:
            v = f"random_seed{seed}"
            additions[v].extend(ii[rngs[seed].choice(len(ii), q, replace=False)])
            removals[v].extend(ee[rngs[seed].choice(len(ee), q, replace=False)])
    result = {"offline": np.flatnonzero(fm), "online": np.flatnonzero(om)}
    sides = {}
    for v in VARIANTS[2:]:
        add = np.asarray(additions[v], dtype=np.int64); remove = np.asarray(removals[v], dtype=np.int64)
        mask = fm.copy(); mask[remove] = False; mask[add] = True
        result[v] = np.flatnonzero(mask); sides[v] = (add, remove)
    require(not np.intersect1d(sides["near"][0], sides["strong"][0]).size and not np.intersect1d(sides["near"][1], sides["strong"][1]).size, "Near/strong endpoints overlap")
    return result, cohorts, sides, np.asarray(quotas), np.asarray(capacities)


def diagnostics(rows, parts, offline_scores, out):
    off_ranks = [ranking(x) for x in offline_scores]
    rank_acc = defaultdict(CohortAccumulator); source_acc = defaultdict(list)
    source_slots = defaultdict(lambda: np.zeros((4, 64)))
    cutoff_deltas = defaultdict(list)
    coverage_values = defaultdict(list); reductions = defaultdict(list); swap_deltas = defaultdict(list)
    head_sums = np.zeros(parts[0]["source_head_temporal"].shape[1:], dtype=np.float64); token_examples = []
    row_cursor = 0
    with (out / "row_score_diagnostics.jsonl").open("x") as handle:
        for data in parts:
            for ri in range(len(data["legacy_online_scores"])):
                row = rows[row_cursor]; row_cursor += 1
                head_sums += data["source_head_temporal"][ri]
                for bi, block in enumerate(BLOCKS):
                    online = ranking(data["legacy_online_scores"][ri, bi]); offline = off_ranks[bi]
                    comps = np.asarray(data["source_components_fp32"][ri, bi], dtype=np.float64)
                    total = np.asarray(data["source_total_fp32"][ri, bi], dtype=np.float64)
                    expected, cohorts, sides, quota, capacity = expected_swaps(online, offline, row, block)
                    require(np.array_equal(quota, data["swap_quotas"][ri, bi]) and np.array_equal(capacity, data["swap_capacities"][ri, bi]), "GPU quota/capacity differs from independent candidates")
                    require(quota.tolist() == row["swap_quotas"][bi] and capacity.tolist() == row["swap_capacities"][bi], "Quota row artifact mismatch")
                    for variant, idx in expected.items():
                        require(np.array_equal(idx, data["mask_indices"][ri, ARMS.index(f"original__L{block}__{variant}")]), f"GPU mask differs from stable deterministic/random reconstruction: {variant}")
                        if variant in ["offline", "online"]:
                            require(np.array_equal(idx, data["mask_indices"][ri, ARMS.index(f"packed__L{block}__{variant}")]), "Packed/reference masks differ")
                    require(np.array_equal(data["mask_indices"][ri, -1], np.arange(12288, 16384)), "Recent mask mismatch")
                    for si, info in enumerate([online, offline]):
                        observed = row["cutoffs_online_offline"][bi][si]
                        val = info["score_k"]; greater = int((info["score"] > val).sum()); equal = int((info["score"] == val).sum())
                        require(observed["value"] == val and observed["greater"] == greater and observed["equal"] == equal and observed["selected_equal"] == KEEP - greater, "GPU Kth-score/tie audit mismatch")
                        require(observed["cross_boundary_tie"] == info["boundary_tied"], "GPU cutoff boundary tie flag mismatch")
                    m, d, cap = int(quota.sum()), len(cohorts[0]), int(capacity.sum())
                    if m == 0:
                        baseline_logits = data["action_logits"][ri, ARMS.index(f"original__L{block}__offline")]
                        require(all(np.array_equal(baseline_logits, data["action_logits"][ri, ARMS.index(f"original__L{block}__{v}")]) for v in VARIANTS[2:]), "Zero-swap masks produced different logits")
                    cov = {"m": m, "zero_swap": int(m == 0), "incoming_outgoing_each": d, "eligible_matched_capacity": cap,
                           "full_symmetric_difference": 2 * d, "exchanged_symmetric_difference": 2 * m,
                           "disagreement_fraction_covered": m / d if d else None, "quota_per_slot": quota.tolist(), "capacity_per_slot": capacity.tolist()}
                    coverage_values[f"L{block}"].append(cov)
                    boundary = online["order"][[KEEP - 1, KEEP]]
                    component_cutoff = comps[:, boundary].mean(-1)
                    reduction = online["score"] - total
                    reduction_cutoff = reduction[boundary].mean()
                    record = {"sample_id": row["sample_id"], "video_id": row["video_id"], "block": block, "coverage": cov,
                              "online_cutoff_component_reference": component_cutoff.tolist(), "online_cutoff_reduction_reference": float(reduction_cutoff),
                              "cohorts": {}, "swaps": {}}
                    for cohort, idx in zip(COHORTS, cohorts):
                        item = {}
                        for mode, info in [("online", online), ("offline", offline)]:
                            rs = rank_summary(info, idx); rank_acc[f"L{block}/{cohort}/{mode}"].add(rs)
                            item[mode] = {k: v for k, v in rs.items() if not k.endswith("_cdf")}
                        ss = source_summary(comps, idx); source_acc[f"L{block}/{cohort}"].append(ss["component_sum"])
                        source_slots[f"L{block}/{cohort}"] += ss["component_mass_per_slot"]
                        delta = comps[:, idx] - component_cutoff[:, None]
                        residual = reduction[idx] - reduction_cutoff
                        require(np.allclose(delta.sum(0) + residual, online["gap_raw"][idx], rtol=1e-5, atol=5e-5), "Four sources plus reduction residual fail cutoff margin reconstruction")
                        item["sources"] = {k: v for k, v in ss.items() if k != "component_mass_per_slot"}
                        item["online_cutoff_signed_component_mean"] = delta.mean(-1).tolist() if len(idx) else None
                        item["online_cutoff_reduction_residual_mean"] = float(residual.mean()) if len(idx) else None
                        if len(idx):
                            cutoff_deltas[f"L{block}/{cohort}"].append(np.r_[delta.mean(-1), residual.mean()])
                        record["cohorts"][cohort] = item
                    for variant, (incoming, outgoing) in sides.items():
                        item = {}
                        for side, idx in [("incoming", incoming), ("outgoing", outgoing)]:
                            item[side] = {}
                            for mode, info in [("online", online), ("offline", offline)]:
                                rs = rank_summary(info, idx); rank_acc[f"L{block}/{variant}/{side}/{mode}"].add(rs)
                                item[side][mode] = {k: v for k, v in rs.items() if not k.endswith("_cdf")}
                        source_delta = comps[:, incoming].sum(-1) - comps[:, outgoing].sum(-1)
                        legacy_delta = float(online["score"][incoming].sum() - online["score"][outgoing].sum())
                        precise_delta = float(total[incoming].sum() - total[outgoing].sum())
                        require(legacy_delta >= -1e-5, "Online preference swap has negative aggregate score gain")
                        item["delta"] = {"m": m, "source_component_sum": source_delta.tolist(), "fp32_total_sum": precise_delta,
                                         "legacy_score_sum": legacy_delta, "legacy_minus_fp32_reduction_sum": legacy_delta - precise_delta,
                                         "source_component_mean_per_swap": (source_delta / m).tolist() if m else None,
                                         "legacy_score_mean_per_swap": legacy_delta / m if m else None}
                        swap_deltas[f"L{block}/{variant}"].append(item["delta"]); record["swaps"][variant] = item
                        if row_cursor <= 2 and variant in ["near", "strong"]:
                            # Two prospectively fixed first execution rows, a small
                            # audit view; aggregate conclusions use every row.
                            for side, idx in [("incoming", incoming), ("outgoing", outgoing)]:
                                for token in idx[:8]:
                                    token_examples.append({"sample_id": row["sample_id"], "block": block, "variant": variant, "side": side,
                                        "token_index": int(token), "slot": int(token // GP), "sources": comps[:, token].tolist(),
                                        **{f"{mode}_{field}": float(info[field][token]) for mode, info in [("online", online), ("offline", offline)]
                                           for field in ["rank_min", "rank_max", "ordinal", "score", "gap_raw", "gap_relative_cutoff", "gap_mean_normalized"]}})
                    rd = reduction_diagnostic(online, total); reductions[f"L{block}"].append(rd)
                    record["reduction_sensitivity"] = rd
                    handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    result = {"rank": {k: v.result() for k, v in rank_acc.items()}, "sources": {}, "swap_deltas": {}, "coverage": {}, "reduction_sensitivity": {}}
    for key, values in source_acc.items():
        x = np.asarray(values); summed = x.sum(0); masses = x.sum(-1)
        result["sources"][key] = {"component_sum_all_receivers": summed.tolist(), "component_mean_receiver_sum": x.mean(0).tolist(),
            "global_mass_weighted_component_fraction": (summed / summed.sum()).tolist() if summed.sum() else None,
            "equal_nonzero_receiver_component_fraction": (x[masses > 0] / masses[masses > 0, None]).mean(0).tolist() if (masses > 0).any() else None,
            "nonzero_mass_receivers": int((masses > 0).sum()), "mean_receiver_component_mass_per_slot": (source_slots[key] / len(values)).tolist()}
        signed = np.asarray(cutoff_deltas[key])
        result["sources"][key]["online_cutoff_signed_means"] = {
            "order": SOURCES + ["legacy_minus_fp32_reduction_residual"],
            "nonempty_receivers": len(signed),
            "equal_nonempty_receiver_mean": signed.mean(0).tolist() if len(signed) else None,
            "per_receiver_mean_distributions": [stats(signed[:, i]) for i in range(5)] if len(signed) else None,
            "scope": "Components centered at ONLINE midpoint boundary component values, plus separate reduction residual. Their sum reconstructs the online raw cutoff margin; not an offline calibration-map decomposition."}
    for key, values in swap_deltas.items():
        comp = np.asarray([v["source_component_sum"] for v in values]); total = np.array([v["fp32_total_sum"] for v in values])
        legacy = np.array([v["legacy_score_sum"] for v in values]); m = np.array([v["m"] for v in values])
        result["swap_deltas"][key] = {"mean_receiver_source_component_sum": comp.mean(0).tolist(),
            "aggregate_source_component_fraction_of_fp32_gap": (comp.sum(0) / total.sum()).tolist() if total.sum() > 1e-8 else None,
            "aggregate_fp32_gap_denominator": float(total.sum()), "mean_receiver_legacy_gap": float(legacy.mean()),
            "mean_nonzero_receiver_gap_per_swap": float((legacy[m > 0] / m[m > 0]).mean()) if (m > 0).any() else None,
            "per_receiver_legacy_score_gain": stats(legacy), "zero_swap_receivers": int((m == 0).sum()),
            "interpretation": "Signed component differences may be negative. Fraction uses summed component gaps divided by summed FP32 total gap, not mean of unstable individual ratios."}
    for key, values in coverage_values.items():
        result["coverage"][key] = {name: stats([v[name] for v in values if v[name] is not None]) for name in ["m", "incoming_outgoing_each", "eligible_matched_capacity", "disagreement_fraction_covered"]}
        result["coverage"][key].update(zero_swap_rows=sum(v["zero_swap"] for v in values), rows=len(values),
            mean_quota_per_slot=np.mean([v["quota_per_slot"] for v in values], 0).tolist(),
            mean_capacity_per_slot=np.mean([v["capacity_per_slot"] for v in values], 0).tolist())
    for key, values in reductions.items():
        result["reduction_sensitivity"][key] = {name: stats([v[name] for v in values]) for name in ["same_probability_top4096_overlap", "same_probability_top4096_jaccard", "mean_absolute_rank_change", "cutoff_difference_relative_to_legacy"]}
    result.update(source_order=SOURCES, rank_grid=RANK_GRID.tolist(), relative_gap_grid=GAP_GRID.tolist(), near_bands=NEAR_BANDS,
                  mean_source_head_temporal=(head_sums / len(rows)).tolist(),
                  source_scope="All four-source quantities describe current validation receivers, never the offline training-map components.",
                  cutoff_definition="Midpoint of online/offline own K and K+1 scores. Four-source cutoff decomposition uses the ONLINE boundary tokens only, plus separately retained legacy-versus-FP32 reduction residual.")
    (out / "fixed_first2_token_examples.json").write_text(json.dumps(token_examples, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    main()
