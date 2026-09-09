#!/usr/bin/env python3
"""CPU-only aggregation of the complete frozen oldest-slot interventions."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from app.hdepic_lora_action_anticipation.probe_b18_q3_oldest_mechanism import (
    PROTOCOL, CONDITIONS, digest, write_json, replacements, construct,
)


def mask_stats(score):
    score = np.asarray(score, dtype=np.float64)
    assert score.shape == (64, 256) and np.isfinite(score).all()
    ids = np.argsort(-score.ravel(), kind="stable")[:4096]
    mask = np.zeros(score.size, dtype=bool)
    mask[ids] = True
    mask = mask.reshape(score.shape)
    tau = score.ravel()[ids[-1]]
    strict, tied = score > tau, score == tau
    remaining = 4096 - int(strict.sum())
    strict_n, tied_n = strict.sum(1), tied.sum(1)
    lower = strict_n + np.maximum(0, remaining - (int(tied.sum()) - tied_n))
    upper = strict_n + np.minimum(remaining, tied_n)
    return dict(count=mask.sum(1).tolist(), mass_over_uniform=(score.sum(1) / score.sum() * 64).tolist(),
                cutoff=float(tau), cutoff_over_mean=float(tau / score.mean()),
                quantiles_over_mean=(np.quantile(score, [.1, .5, .9, .99], axis=1) / score.mean()).T.tolist(),
                strict_above=int(strict.sum()), tied_at_cutoff=int(tied.sum()), selected_ties=remaining,
                tie_allocation_ambiguous=bool(np.any(lower != upper)),
                count_lower=lower.tolist(), count_upper=upper.tolist()), mask


def mean2(a, ix=(0, 1)):
    return float(np.asarray(a)[list(ix)].mean())


def scientific_summary(pooled):
    out = {}
    for condition in CONDITIONS:
        stats, _ = mask_stats(pooled["score_actual__" + condition])
        no, _ = mask_stats(pooled["score_no_rope_readout__" + condition])
        flow = pooled["flow_actual__" + condition]
        headmass = flow[:, :, :64].sum(1)
        excess = flow[:, :, :2].sum(-1) - flow[:, :, :64].sum(-1) / 32
        qgroups = {"oldest_context_queries": [0, 1], "remaining_context_queries": list(range(2, 64)), "target_queries": [64]}
        out[condition] = dict(actual=stats, no_rope_readout=no,
            start_count=stats["count"][:2], next_real_count=stats["count"][2:4], middle_count=stats["count"][32:34],
            start_mass_uniform=mean2(stats["mass_over_uniform"]),
            head_start_mass_uniform=(headmass[:, :2].mean(1) / headmass.mean(1)).tolist(),
            head_oldest_excess=excess.sum(1).tolist(),
            querygroup_oldest_excess={k: float(excess[:, v].sum()) for k, v in qgroups.items()},
            querygroup_oldest_received={k: float(flow[:, v, :2].sum()) for k, v in qgroups.items()},
            querygroup_oldest_mass_uniform={k: float(flow[:, v, :2].sum() / (flow[:, v, :64].sum() / 32)) for k, v in qgroups.items()},
            without_oldest_queries_start_mass_uniform=float(flow[:, 2:, :2].sum() / (flow[:, 2:, :64].sum() / 32)),
            total_target_key_mass=float(flow[:, :, 64].sum()),
            total_context_key_mass=float(headmass.sum()))
        trace = {}
        for key, value in pooled.items():
            if not key.endswith("__" + condition):
                continue
            name = key.split("__")[0]
            if name.startswith("stage_"):
                trace[name] = dict(start_norm=float(value[:2, 0].mean()), interior_norm=float(value[16:48, 0].mean()),
                                   start_over_middle_norm=float(value[:2, 0].mean() / value[16:48, 0].mean()),
                                   start_alignment=float(value[:2, 5].mean()), interior_alignment=float(value[16:48, 5].mean()))
            elif name in ("q_pre_norm", "k_pre_norm", "q_post_norm", "k_post_norm"):
                trace[name] = dict(start=float(value[:, :2, 0].mean()), interior=float(value[:, 16:48, 0].mean()),
                                   per_head_start_over_middle=(value[:, :2, 0].mean(1) / value[:, 16:48, 0].mean(1)).tolist())
            elif name.startswith(("key_logit_", "qmean_kmean_cos_")):
                trace[name] = dict(start=float(value[:, :2].mean()), interior=float(value[:, 16:48].mean()),
                                   per_head_start=value[:, :2].mean(1).tolist(),
                                   per_head_interior=value[:, 16:48].mean(1).tolist())
        out[condition]["trace"] = trace
    return out


def plot_all(args, manifest, pooled, summary, masks, representative):
    x = np.arange(64)
    fig, axs = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    groups = [CONDITIONS[1:4], CONDITIONS[4:6], CONDITIONS[6:]]
    for row, group in enumerate(groups):
        for col, field in enumerate(("count", "mass_over_uniform")):
            ax = axs[row, col]
            ax.plot(x, summary["conditions"]["continuous"]["actual"][field], c="black", label="continuous", lw=2)
            for condition in group:
                ax.plot(x, summary["conditions"][condition]["actual"][field], label=condition, alpha=.85)
            ax.axhline(64 if col == 0 else 1, ls=":", c="gray")
            ax.axvspan(-.4, 1.4, color="orange", alpha=.1)
            ax.axvspan(31.6, 33.4, color="blue", alpha=.06)
            ax.set_ylabel("retained / 256" if col == 0 else "received mass / uniform")
            ax.legend(fontsize=8)
    axs[-1, 0].set_xlabel("temporal slot (0 = oldest)")
    axs[-1, 1].set_xlabel("temporal slot (0 = oldest)")
    fig.suptitle(f"16s / fixed 4096-token budget; independently pooled {summary['n_rows']} training videos")
    fig.tight_layout(); fig.savefig(args.out / "fixed4s_interventions.png", dpi=150); plt.close(fig)

    stages = ["patch", "encoder_block00", "encoder_block11", "encoder_block23", "encoder_final_norm", "predictor_embed", "predictor_l0_norm1"]
    fig, axs = plt.subplots(4, 2, figsize=(14, 13))
    for ax, stage in zip(axs.ravel(), stages):
        for cond in ("continuous", "start_noise_20260907", "start_gray", "start_donor", "middle_noise_20260907"):
            z = pooled["stage_" + stage + "__" + cond][:64, 0]
            ax.plot(x, z / z[16:48].mean(), label=cond)
        ax.set_title(stage + ": mean token norm / middle-half mean")
        ax.axhline(1, c="gray", ls=":"); ax.set_xlim(-1, 64)
    axs.ravel()[-1].axis("off")
    axs.ravel()[-1].legend(*axs[0, 0].get_legend_handles_labels(), loc="center")
    fig.suptitle("Measured representation norms: descriptive stages, not causal attribution")
    fig.tight_layout(); fig.savefig(args.out / "stage_traces.png", dpi=140); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for j, suffix in enumerate(("actual", "no_rope_readout")):
        flow = pooled["flow_" + suffix + "__continuous"].sum(0)
        norm = flow[:, :64] / flow[:, :64].sum(-1, keepdims=True) * 64
        im = axs[0, j].imshow(norm, aspect="auto", origin="lower", cmap="viridis", vmin=.5, vmax=1.7)
        axs[0, j].set_title(suffix + ": received / uniform per query slot")
        axs[0, j].set_xlabel("key slot"); axs[0, j].set_ylabel("query slot (64 = target)")
        fig.colorbar(im, ax=axs[0, j])
    for c in ("continuous", "start_noise_20260907", "start_gray", "start_donor"):
        axs[0, 2].plot(np.arange(12), summary["conditions"][c]["head_start_mass_uniform"], marker="o", label=c)
    axs[0, 2].axhline(1, c="gray", ls=":"); axs[0, 2].set_title("Oldest-two received / uniform, by head"); axs[0, 2].legend(fontsize=8)
    for j, name in enumerate(("key_logit_mean", "key_logit_rms_dispersion", "qmean_kmean_cos")):
        for suffix in ("actual", "no_rope_readout"):
            axs[1, j].plot(x, pooled[name + "_" + suffix + "__continuous"][:, :64].mean(0), label=suffix)
        axs[1, j].set_title(name + ": head mean"); axs[1, j].legend(fontsize=8)
    fig.suptitle("Continuous baseline L0; no-RoPE is a same-Q/K readout, not a model intervention")
    fig.tight_layout(); fig.savefig(args.out / "heads_queries_logits.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    for condition in ("continuous", "start_noise_20260907", "start_gray", "start_donor"):
        flow = pooled["flow_actual__" + condition].sum(0)
        oldest = flow[:, :2].sum(-1)
        uniform = flow[:, :64].sum(-1) / 32
        axs[0, 0].plot(np.arange(65), oldest / uniform, label=condition)
        axs[0, 1].plot(np.arange(65), oldest - uniform, label=condition)
    axs[0, 0].set_title("Oldest-two mass / uniform, by query source"); axs[0, 0].axhline(1, ls=":", c="gray")
    axs[0, 1].set_title("Oldest-two excess mass, by query source"); axs[0, 1].axhline(0, ls=":", c="gray")
    for ax in axs[0]:
        ax.legend(fontsize=8); ax.set_xlabel("query slot (64 = prediction target)")
    flow = pooled["flow_actual__continuous"].sum(0)
    for name, positions in (("all queries", slice(None)), ("exclude oldest queries0/1", slice(2, None)), ("target query only", slice(64, 65))):
        z = flow[positions, :64].sum(0)
        axs[1, 0].plot(x, z / z.mean(), label=name)
    axs[1, 0].axhline(1, ls=":", c="gray"); axs[1, 0].legend(fontsize=8)
    axs[1, 0].set_title("Continuous baseline received mass by key slot / uniform")
    keys = ("oldest_context_queries", "remaining_context_queries", "target_queries")
    for j, c in enumerate(("continuous", "start_noise_20260907", "start_gray", "start_donor")):
        z = summary["conditions"][c]["querygroup_oldest_excess"]
        axs[1, 1].bar(np.arange(3) + (j - 1.5) * .19, [z[k] for k in keys], width=.19, label=c)
    axs[1, 1].set_xticks(np.arange(3), ["oldest queries0/1", "context queries2:64", "target query"])
    axs[1, 1].axhline(0, ls=":", c="gray"); axs[1, 1].set_title("Signed oldest-two excess over matched uniform mass")
    fig.suptitle("Query-source decomposition of actual attention; removing terms is a diagnostic readout")
    fig.tight_layout(); fig.savefig(args.out / "query_source_contributions.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for j, condition in enumerate(("continuous", "start_noise_20260907", "start_gray", "start_donor")):
        ax = axs.ravel()[j]
        z = summary["conditions"][condition]["actual"]
        quant = np.asarray(z["quantiles_over_mean"])
        ax.fill_between(x, quant[:, 0], quant[:, 2], alpha=.25, label="token q10–q90")
        ax.plot(x, quant[:, 1], label="median")
        ax.axhline(z["cutoff_over_mean"], ls="--", c="red", label="global top4096 cutoff")
        ax.plot(x, z["mass_over_uniform"], label="mean")
        ax.set_title(condition); ax.set_ylabel("score / global mean"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out / "score_thresholds.png", dpi=150); plt.close(fig)

    sample = manifest["source"]["samples"][representative]
    donor_sample = next(s for s in manifest["source"]["samples"] if s["sample_id"] == sample["donor_sample_id"])
    receiver = np.load(sample["cache"]["path"], mmap_mode="r")
    donor = np.load(donor_sample["cache"]["path"], mmap_mode="r")
    tiles = replacements(sample, donor)
    rows = ["continuous", "start_noise_20260907", "start_gray", "start_donor", "middle_noise_20260907", "middle_gray", "middle_donor"]
    slots = [0, 1, 2, 32, 33, 63]
    fig, axs = plt.subplots(len(rows), len(slots), figsize=(13, 15))
    for i, condition in enumerate(rows):
        rgb = construct(receiver, tiles, condition)
        for j, slot in enumerate(slots):
            ax = axs[i, j]
            ax.imshow(rgb[slot * 2])
            kept = masks[condition][slot].reshape(16, 16)
            overlay = np.zeros((16, 16, 4)); overlay[..., 1] = 1; overlay[..., 3] = kept * .42
            ax.imshow(overlay, extent=(-.5, 255.5, 255.5, -.5), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"slot {slot}: {int(kept.sum())}/256", fontsize=9)
            if j == 0:
                ax.set_ylabel(condition, fontsize=8)
    fig.suptitle(f"Frozen representative {sample['sample_id']} / {sample['video_id']}; green = pooled mask")
    fig.tight_layout(); fig.savefig(args.out / "rgb_pooled_masks.png", dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    assert manifest["protocol"] == PROTOCOL
    source = manifest["source"]
    seen, sums, halves, individuals, run_records = set(), {}, [{}, {}], {c: [] for c in CONDITIONS}, []
    half_n = [0, 0]
    baseline_gates = []
    closure_max = 0.
    for run in args.runs:
        meta = json.loads((run / "metadata.json").read_text())
        status = json.loads((run / "summary.json").read_text())
        assert meta["protocol"] == PROTOCOL and status["run_status"] == "completed"
        assert meta["manifest_sha256"] == digest(args.manifest)
        run_records.append(dict(path=str(run), metadata=meta, summary=status))
        files = sorted(run.glob("q3_*.npz"))
        assert len(files) == status["n_rows"] == status["stop"] - status["start"]
        for path in files:
            with np.load(path) as arrays:
                index = int(arrays["sample_index"])
                assert index not in seen and meta["start"] <= index < meta["stop"]
                seen.add(index)
                half = source["samples"][index]["calibration_half"]
                half_n[half] += 1
                for key in arrays.files:
                    if key == "sample_index":
                        continue
                    value = arrays[key].astype(np.float64)
                    assert np.isfinite(value).all()
                    if key not in sums:
                        sums[key] = np.zeros_like(value)
                    sums[key] += value
                    if key not in halves[half]:
                        halves[half][key] = np.zeros_like(value)
                    halves[half][key] += value
                for c in CONDITIONS:
                    z = arrays["score_actual__" + c]
                    stats, mask = mask_stats(z)
                    assert int(mask.sum()) == 4096
                    individuals[c].append(dict(index=index, count=stats["count"], mass_over_uniform=stats["mass_over_uniform"]))
                    for suffix in ("actual", "no_rope_readout"):
                        flow = arrays["flow_" + suffix + "__" + c]
                        head = arrays["head_importance_" + suffix + "__" + c].reshape(12, 65, 256)
                        assert flow.shape == (12, 65, 65)
                        error = np.max(np.abs(flow.sum(1) - head.sum(-1)))
                        closure_max = max(closure_max, float(error))
                        assert np.allclose(flow.sum(1), head.sum(-1), rtol=1e-5, atol=.03)
                        assert abs(head.sum() / (12 * 65 * 256) - 1) < .01
                original_run = "b18-q3-boundary-full-a-17059417" if index < 43 else "b18-q3-boundary-full-b-17059470"
                archive = Path(source["samples"][index]["cache"]["path"]).parents[2] / original_run / "capture" / path.name
                with np.load(archive) as old:
                    z, expected = arrays["score_actual__continuous"], old["score__continuous_16s"]
                    rel = float(np.linalg.norm(z - expected) / np.linalg.norm(expected))
                    _, m1 = mask_stats(z); _, m2 = mask_stats(expected)
                    jac = float((m1 & m2).sum() / (m1 | m2).sum())
                    assert rel <= .005 and jac >= .98
                    baseline_gates.append(dict(index=index, relative_l2=rel, topk_jaccard=jac))
    assert args.allow_partial or seen == set(range(86))
    n = len(seen)
    pooled = {k: v / n for k, v in sums.items()}
    pooled_halves = [{k: v / half_n[h] for k, v in halves[h].items()} for h in range(2)]
    result = scientific_summary(pooled)
    masks = {c: mask_stats(pooled["score_actual__" + c])[1] for c in CONDITIONS}
    half_result = [scientific_summary(v) if v else {} for v in pooled_halves]
    contrasts = {}
    b = result["continuous"]["actual"]
    for c in CONDITIONS[1:]:
        z = result[c]["actual"]
        changed = [0, 1] if c.startswith("start_") else [32, 33]
        outside = np.ones(64, dtype=bool); outside[changed] = False
        score_delta = pooled["score_actual__" + c] - pooled["score_actual__continuous"]
        contrasts[c] = dict(start_count_delta=(np.asarray(z["count"][:2]) - b["count"][:2]).tolist(),
            manipulated_count_delta=(np.asarray(z["count"])[changed] - np.asarray(b["count"])[changed]).tolist(),
            kept_jaccard=float((masks[c] & masks["continuous"]).sum() / (masks[c] | masks["continuous"]).sum()),
            outside_changed_score_relative_l2=float(np.linalg.norm(score_delta[outside]) / np.linalg.norm(pooled["score_actual__continuous"][outside])),
            outside_changed_mean_absolute_count_delta=float(np.abs(np.asarray(z["count"])[outside] - np.asarray(b["count"])[outside]).mean()))
    summary = dict(protocol=PROTOCOL, metric_scope="selection-and-representation-diagnostic", run_status="completed",
        n_rows=n, n_conditions=9, sample_indices=sorted(seen), half_n=half_n, partial=args.allow_partial,
        manifest=str(args.manifest), manifest_sha256=digest(args.manifest), analysis_code_sha256=digest(__file__),
        conditions=result, calibration_halves=half_result, contrasts=contrasts,
        baseline_gates=baseline_gates, closure_max_abs=closure_max,
        limitations=["No downstream accuracy", "Noise is OOD", "No-RoPE readout is not a full model intervention",
                     "Donor-linked rows and halves are descriptive, not independent-sample confidence intervals",
                     "Measured stage statistics do not identify a causal origin"])
    summary["same_tile_position_controls"] = {
        kind: dict(start_replacement_count=result["start_" + kind]["actual"]["count"][:2],
                   middle_replacement_count=result["middle_" + kind]["actual"]["count"][32:34],
                   start_replacement_mass_uniform=mean2(result["start_" + kind]["actual"]["mass_over_uniform"]),
                   middle_replacement_mass_uniform=mean2(result["middle_" + kind]["actual"]["mass_over_uniform"], (32, 33)))
        for kind in ("noise_20260907", "gray", "donor")}
    for c in CONDITIONS:
        a = np.asarray([r["count"] for r in individuals[c]])
        result[c]["individual_count_mean"] = a.mean(0).tolist()
        result[c]["individual_oldest_above_uniform"] = int((a[:, :2].mean(1) > 64).sum())
        if c != "continuous":
            ref = np.asarray([r["count"] for r in individuals["continuous"]])
            result[c]["individual_start_delta_mean"] = float((a[:, :2] - ref[:, :2]).mean())
            result[c]["individual_start_delta_positive"] = int(((a[:, :2] - ref[:, :2]).mean(1) > 0).sum())
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "input_runs.json", run_records)
    write_json(args.out / "individual_masks_summary.json", individuals)
    np.savez_compressed(args.out / "pooled_traces_scores_masks.npz", **pooled,
                        **{"mask__" + c: masks[c] for c in CONDITIONS})
    # All individual exact masks, independent of pooling, remain directly reconstructible
    # from archived complete per-token scores; save them explicitly for review too.
    indiv_masks = {}
    for run in args.runs:
        for path in sorted(run.glob("q3_*.npz")):
            with np.load(path) as a:
                for c in CONDITIONS:
                    indiv_masks[path.stem + "__" + c] = mask_stats(a["score_actual__" + c])[1]
    np.savez_compressed(args.out / "individual_masks.npz", **indiv_masks)
    with (args.out / "per_slot.csv").open("w") as f:
        w = csv.writer(f); w.writerow(["condition", "slot", "count", "mass_uniform", "median_score_over_mean", "cutoff_over_mean"])
        for c in CONDITIONS:
            z = result[c]["actual"]
            for t in range(64):
                w.writerow([c, t, z["count"][t], z["mass_over_uniform"][t], z["quantiles_over_mean"][t][1], z["cutoff_over_mean"]])
    brief = {c: {k: result[c][k] for k in ("start_count", "next_real_count", "middle_count", "start_mass_uniform", "individual_oldest_above_uniform")} for c in CONDITIONS}
    write_json(args.out / "brief.json", brief)
    plot_all(args, manifest, pooled, summary, masks, min(seen))
    print(json.dumps(dict(n_rows=n, n_conditions=9, half_n=half_n, baseline_max_relative_l2=max(x["relative_l2"] for x in baseline_gates),
                         baseline_min_jaccard=min(x["topk_jaccard"] for x in baseline_gates), brief=brief)), flush=True)


if __name__ == "__main__":
    main()
