#!/usr/bin/env python3
"""Independent CPU reconstruction of B18 Q3 exact self-return statistics."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from app.hdepic_lora_action_anticipation.probe_b18_q3_token_self_return import (
    PROTOCOL, GP, NS, NH, SUFFIXES, ROW_FIELDS, digest, write_json, bytehash, compare_scores)
from scripts.egtea.analyze_b18_q3_oldest_mechanism import mask_stats

GROUPS = {"slot0": [0], "slot1": [1], "slot2": [2], "slot3": [3], "oldest0_1": [0, 1],
          "remaining2_63": list(range(2, 64)), "middle16_47": list(range(16, 48)), "all_context": list(range(64))}
METRIC_DEPENDENCY = Path(__file__).with_name("analyze_b18_q3_oldest_mechanism.py")
METRIC_HASH = "5d7726c5eb7a0365757e54ef0284cc62718e26578acd988d2a808b1c3a00700f"


def ratio_stats(mass):
    all_mass, context, same = mass["all"], mass["context"], mass["same_slot"]
    result = {f + "_fraction_all": mass[f] / all_mass for f in ("self", "same_slot", "same_slot_other", "other_context", "target")}
    result.update({f + "_fraction_context": mass[f] / context for f in ("self", "same_slot", "same_slot_other", "other_context")})
    result["self_fraction_same_slot"] = mass["self"] / same
    result["mass_sums"] = mass
    return result


def measure(arrays, suffix, indices, heads=None):
    mass = {}
    for f in ROW_FIELDS:
        a = arrays["row_" + f + "_" + suffix]
        if heads is not None: a = a[heads]
        mass[f] = float(a[:, indices].sum(dtype=np.float64))
    return ratio_stats(mass)


def summarize(pooled):
    out = {}
    for suffix in SUFFIXES:
        out[suffix] = dict(groups={name: measure(pooled, suffix, ids) for name, ids in GROUPS.items()},
            slots=[measure(pooled, suffix, [i]) for i in range(64)],
            heads={str(h): {name: measure(pooled, suffix, ids, [h]) for name, ids in GROUPS.items()} for h in range(12)},
            head_slot_self_fraction_all=[[measure(pooled, suffix, [i], [h])["self_fraction_all"] for i in range(64)] for h in range(12)],
            selection={route: mask_stats(pooled["score_" + route + "_" + suffix])[0] for route in ("legacy", "fp32", "self_excluded_fp32")})
        a, ma = mask_stats(pooled["score_fp32_" + suffix])
        b, mb = mask_stats(pooled["score_self_excluded_fp32_" + suffix])
        out[suffix]["selection_fp32_self_exclusion"] = dict(topk_jaccard=float(np.count_nonzero(ma & mb) / np.count_nonzero(ma | mb)),
            changed_out=int(np.count_nonzero(ma & ~mb)), count_delta=(mb.sum(1) - ma.sum(1)).tolist(),
            interpretation="zero exact diagonal; all queries retained; no row renormalization; readout only")
        flow = pooled["flow_fp32_" + suffix].sum(0)[:, :64]
        columns = pooled["head_fp32_" + suffix][:, :64 * GP].reshape(12, 64, GP).sum((0, 2))
        same = np.diag(flow[:64])
        exact = pooled["row_self_" + suffix].sum((0, 2))
        competitors = flow.copy(); competitors[np.arange(64), np.arange(64)] = -np.inf
        largest = same > competitors.max(0)
        out[suffix]["received_source"] = dict(
            denominator="all query sources, including future target; matched FP32 received mass per context-key slot",
            flow_column_normalized=(flow / flow.sum(0)).tolist(), same_slot_share=(same / columns).tolist(),
            exact_self_share=(exact / columns).tolist(), same_slot_other_share=((same - exact) / columns).tolist(),
            other_sources_share=((columns - same) / columns).tolist(),
            same_slot_largest_single_source_count=int(largest.sum()), same_slot_largest_slots=np.flatnonzero(largest).tolist(),
            same_slot_above_half_count=int((same / columns > .5).sum()), same_slot_above_half_slots=np.flatnonzero(same / columns > .5).tolist())
    return out


def figures(out, summary, pooled, receiver):
    x = np.arange(64)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fields = ("self_fraction_all", "same_slot_other_fraction_all", "other_context_fraction_all", "target_fraction_all",
              "self_fraction_same_slot", "self_fraction_context")
    titles = ("Exact token self / all", "Same-slot OTHER tokens / all", "Other context slots / all", "Future target destinations / all",
              "Exact self / whole same-slot block", "Exact self / context destinations")
    for ax, field, title in zip(axs.ravel(), fields, titles):
        for suffix, label in (("actual", "Actual L0"), ("no_rope_readout", "Same-QK no-RoPE readout")):
            ax.plot(x, [100 * z[field] for z in summary["population"][suffix]["slots"]], label=label)
        ax.set_title(title); ax.set_ylabel("mass ratio (%)"); ax.axvspan(-.4, 1.4, color="orange", alpha=.12)
        ax.set_xlabel("temporal slot (0 = oldest)"); ax.legend(fontsize=8)
    fig.suptitle(f"Continuous16s; {summary['n_rows']} training videos; ratios of summed masses")
    fig.tight_layout(); fig.savefig(out / "self_and_same_slot_temporal.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for ax, suffix in zip(axs[0], SUFFIXES):
        heat = 100 * np.asarray(summary["population"][suffix]["head_slot_self_fraction_all"])
        im = ax.imshow(heat, aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax, label="exact self / all (%)"); ax.set_title(suffix); ax.set_xlabel("temporal slot"); ax.set_ylabel("head")
    for i, half in enumerate(summary["calibration_halves"]):
        axs[1, 0].plot(x, [100 * z["self_fraction_all"] for z in half["actual"]["slots"]], label=f"half{i}, n={summary['half_n'][i]}")
    axs[1, 0].set_title("Original halves: actual exact self / all"); axs[1, 0].legend(); axs[1, 0].set_ylabel("mass ratio (%)")
    a = [100 * r["actual"]["oldest0_1"]["self_fraction_all"] for r in receiver]
    b = [100 * r["actual"]["middle16_47"]["self_fraction_all"] for r in receiver]
    axs[1, 1].scatter(b, a, s=18, alpha=.7)
    lo, hi = min(a + b), max(a + b); axs[1, 1].plot([lo, hi], [lo, hi], "k:")
    axs[1, 1].set(xlabel="Middle slots16:48 self / all (%)", ylabel="Oldest slots0/1 self / all (%)", title="Paired receivers; descriptive, no CI")
    fig.tight_layout(); fig.savefig(out / "heads_halves_receivers.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for col, suffix in enumerate(SUFFIXES):
        for route, label in (("legacy", "Legacy chunked received score"), ("fp32", "Matched FP32 original"), ("self_excluded_fp32", "Matched FP32 self excluded")):
            stats = summary["population"][suffix]["selection"][route]
            axs[0, col].plot(x, stats["count"], label=label)
            axs[1, col].plot(x, stats["mass_over_uniform"], label=label)
        axs[0, col].set_title(suffix); axs[0, col].axhline(64, c="gray", ls=":")
        axs[1, col].axhline(1, c="gray", ls=":")
        for ax in axs[:, col]: ax.legend(fontsize=8); ax.set_xlabel("temporal slot")
    axs[0, 0].set_ylabel("retained / 256; K4096"); axs[1, 0].set_ylabel("received mass / uniform")
    fig.suptitle("Self exclusion changes only the diagnostic readout; no query renormalization")
    fig.tight_layout(); fig.savefig(out / "self_exclusion_readout.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), gridspec_kw={"height_ratios": [2, 1]})
    for col, suffix in enumerate(SUFFIXES):
        z = summary["population"][suffix]["received_source"]
        im = axs[0, col].imshow(100 * np.asarray(z["flow_column_normalized"]), aspect="auto", origin="upper", vmin=0, vmax=100)
        fig.colorbar(im, ax=axs[0, col], label="share of RECEIVED mass (%); each column sums100")
        axs[0, col].set(title=suffix, xlabel="context KEY slot receiving attention", ylabel="QUERY source slot (row64 = target at pos72)")
        for field, label in (("same_slot_share", "Whole same-slot source"), ("exact_self_share", "Exact token self"),
                             ("same_slot_other_share", "Other tokens within same slot")):
            axs[1, col].plot(x, 100 * np.asarray(z[field]), label=label)
        axs[1, col].axhline(50, c="black", ls=":", label="Majority threshold50%")
        axs[1, col].set(xlabel="context KEY slot", ylabel="share of RECEIVED mass (%)",
            title=f"Same slot largest single source: {z['same_slot_largest_single_source_count']}/64; above50%: {z['same_slot_above_half_count']}/64")
        axs[1, col].legend(fontsize=8)
    fig.suptitle("Incoming source decomposition; column normalization, no color clipping")
    fig.tight_layout(); fig.savefig(out / "received_source_columns.png", dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    assert not args.out.exists(); args.out.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text()); source = manifest["source"]
    assert manifest["protocol"] == PROTOCOL and digest(METRIC_DEPENDENCY) == METRIC_HASH
    assert digest(manifest["source_manifest"]) == manifest["source_manifest_sha256"]
    assert digest(manifest["prior_manifest"]) == manifest["prior_manifest_sha256"]
    for path, sha in manifest["code_sha256"].items(): assert digest(path) == sha, path
    for model in source["model"].values(): assert digest(model["path"]) == model["sha256"]
    files, runs = {}, []
    for path in args.runs:
        meta = json.loads((path / "metadata.json").read_text()); done = json.loads((path / "summary.json").read_text())
        assert done["run_status"] == "completed" and meta["protocol"] == PROTOCOL
        assert meta["manifest_sha256"] == digest(args.manifest)
        assert meta["is_causal"] is False and meta["attn_mask"] is None
        for gate in done["gates"].get("hardware", {}).values(): assert gate["passed"]
        assert len(done["artifacts"]) == done["stop"] - done["start"]
        for item in done["artifacts"]:
            i = item["index"]; assert i not in files and done["start"] <= i < done["stop"]
            assert digest(item["path"]) == item["sha256"]
            files[i] = Path(item["path"])
        runs.append(dict(path=str(path), metadata=meta, summary=done))
    indices = sorted(files)
    if not args.allow_partial: assert indices == list(range(86))
    pooled, halves = {}, [{}, {}]
    half_n = [0, 0]; receivers, archive_gates, masks = [], {}, {}
    errors = dict(partition=0., same_slot_partition=0., context_partition=0., fp32_column_diagonal=0.,
                  fp32_flow_column=0., legacy_flow_column=0., flow_row=0., probability_row_sum=0.)
    for i in indices:
        row, sample = manifest["rows"][i], source["samples"][i]
        assert digest(sample["cache"]["path"]) == sample["cache"]["sha256"]
        assert bytehash(np.load(sample["cache"]["path"], mmap_mode="r")[-128:]) == row["input_sha256"]
        assert digest(row["archive"]) == row["archive_sha256"] and digest(row["reference"]) == row["reference_sha256"]
        with np.load(files[i]) as raw: a = {k: raw[k].astype(np.float64) for k in raw.files}
        assert int(a.pop("sample_index")) == i and all(np.isfinite(z).all() for z in a.values())
        assert np.array_equal(a.pop("position_ids"), np.r_[np.arange(NS * GP), np.arange(72 * GP, 73 * GP)])
        with np.load(row["reference"]) as ref:
            gate = compare_scores(a["score_legacy_actual"], ref["score_actual"].astype(np.float64)); assert gate["passed"]
            gate["flow_exact"] = bool(np.array_equal(a["flow_legacy_actual"], ref["flow_actual"]))
            assert np.allclose(a["flow_legacy_actual"], ref["flow_actual"], rtol=.005, atol=.03)
            archive_gates[str(i)] = gate
        receiver = dict(index=i, sample_id=sample["sample_id"], video_id=sample["video_id"], participant=sample["participant"], half=sample["calibration_half"])
        for suffix in SUFFIXES:
            r = {f: a["row_" + f + "_" + suffix] for f in ROW_FIELDS}
            assert all(z.shape == (NH, NS, GP) for z in r.values())
            assert all((z >= 0).all() for z in r.values())
            assert (r["self"] <= r["same_slot"] + 2e-6).all()
            assert (r["same_slot"] <= r["context"] + 2e-6).all() and (r["context"] <= r["all"] + 2e-6).all()
            checks = dict(partition=r["self"] + r["same_slot_other"] + r["other_context"] + r["target"] - r["all"],
                          same_slot_partition=r["self"] + r["same_slot_other"] - r["same_slot"],
                          context_partition=r["same_slot"] + r["other_context"] - r["context"])
            for key, delta in checks.items():
                err = float(np.max(np.abs(delta))); errors[key] = max(errors[key], err); assert err <= 2e-6
            errors["probability_row_sum"] = max(errors["probability_row_sum"], float(np.max(np.abs(r["all"] - 1))))
            assert errors["probability_row_sum"] < .01
            diag = a["diagonal_all_queries_" + suffix]
            assert diag.shape == (NH, 65, GP) and np.array_equal(diag[:, :64], r["self"])
            new, excluded = a["head_fp32_" + suffix], a["head_self_excluded_fp32_" + suffix]
            err = float(np.max(np.abs(new - excluded - diag.reshape(NH, -1))))
            errors["fp32_column_diagonal"] = max(errors["fp32_column_diagonal"], err); assert err <= .01
            for route in ("legacy", "fp32"):
                flow = a["flow_" + route + "_" + suffix]; head = a["head_" + route + "_" + suffix]
                assert flow.shape == (12, 65, 65) and head.shape == (12, 65 * GP)
                err = float(np.max(np.abs(flow.sum(1) - head.reshape(12, 65, GP).sum(-1))))
                errors[route + "_flow_column"] = max(errors[route + "_flow_column"], err)
                assert err <= (.03 if route == "legacy" else .01)
            err = float(np.max(np.abs(a["flow_fp32_" + suffix][:, :64].sum(-1) - r["all"].sum(-1))))
            errors["flow_row"] = max(errors["flow_row"], err); assert err <= .01
            for route in ("legacy", "fp32", "self_excluded_fp32"):
                reconstructed = a["head_" + route + "_" + suffix][:, :NS * GP].sum(0).reshape(NS, GP)
                assert np.allclose(reconstructed, a["score_" + route + "_" + suffix], rtol=1e-6, atol=1e-5)
                stats, mask = mask_stats(a["score_" + route + "_" + suffix]); assert mask.sum() == 4096
                masks[f"q3_{i:03d}__{route}__{suffix}"] = mask
            receiver[suffix] = {name: measure(a, suffix, ids) for name, ids in GROUPS.items()}
            receiver[suffix]["selection"] = {route: mask_stats(a["score_" + route + "_" + suffix])[0] for route in ("legacy", "fp32", "self_excluded_fp32")}
        receivers.append(receiver)
        half = sample["calibration_half"]; half_n[half] += 1
        for k, v in a.items():
            if k not in pooled: pooled[k] = np.zeros_like(v)
            if k not in halves[half]: halves[half][k] = np.zeros_like(v)
            pooled[k] += v; halves[half][k] += v
    assert min(half_n) > 0
    population = summarize(pooled)
    half_results = [summarize(h) for h in halves]
    if not args.allow_partial:
        assert half_n == [43, 43]
        assert population["actual"]["selection"]["legacy"]["count"][:2] == [144, 139]
    paired = {}
    for suffix in SUFFIXES:
        paired[suffix] = {}
        for field in ("self_fraction_all", "self_fraction_context", "self_fraction_same_slot", "same_slot_other_fraction_all"):
            delta = np.asarray([r[suffix]["oldest0_1"][field] - r[suffix]["middle16_47"][field] for r in receivers])
            paired[suffix][field] = dict(mean_percentage_points=float(delta.mean() * 100), median_percentage_points=float(np.median(delta) * 100),
                q10_q90_percentage_points=(np.quantile(delta, [.1, .9]) * 100).tolist(), positive=int((delta > 0).sum()), n=len(delta))
    summary = dict(protocol=PROTOCOL, run_status="completed", partial=bool(args.allow_partial), n_rows=len(indices), half_n=half_n,
        manifest=str(args.manifest), manifest_sha256=digest(args.manifest), analysis_sha256=digest(__file__),
        metric_dependency=dict(path=str(METRIC_DEPENDENCY), sha256=METRIC_HASH), metric_scope="selection diagnostic; no action accuracy",
        aggregation="ratio of summed actual FP32 softmax probability masses under BF16 autocast; CPU accumulation FP64; no mean-per-query-ratio metric",
        population=population, calibration_halves=half_results, paired_oldest_minus_middle=paired, max_errors=errors, archive_gates=archive_gates,
        source_audit=dict(rgb_cache_hashes=True, exact_suffix_hashes=True, original_archives=True, compact_references=True, source_model_code=True),
        limitations=["Only predictor L0 measured; previous stage norms do not locate onset of exact self attention",
                     "No-RoPE removes all axes from same-QK readout, not model forward", "Self exclusion is unnormalized readout only",
                     "Receiver/half results descriptive; do not infer classification utility or causal layer origin"])
    write_json(args.out / "summary.json", summary); write_json(args.out / "receivers.json", receivers)
    write_json(args.out / "input_runs.json", runs)
    np.savez_compressed(args.out / "pooled_sums.npz", **pooled)
    np.savez_compressed(args.out / "individual_masks.npz", **masks)
    with (args.out / "per_slot.csv").open("w") as f:
        fields = [k for k in population["actual"]["slots"][0] if k != "mass_sums"]
        w = csv.DictWriter(f, fieldnames=["readout", "slot", *fields]); w.writeheader()
        for suffix in SUFFIXES:
            for i, z in enumerate(population[suffix]["slots"]): w.writerow(dict(readout=suffix, slot=i, **{k: z[k] for k in fields}))
    figures(args.out, summary, pooled, receivers)
    print("SELF_ANALYSIS", json.dumps(dict(n_rows=len(indices), partial=args.allow_partial, max_errors=errors,
          actual_groups=population["actual"]["groups"], actual_counts={k: v["count"][:4] for k, v in population["actual"]["selection"].items()})), flush=True)


if __name__ == "__main__": main()
