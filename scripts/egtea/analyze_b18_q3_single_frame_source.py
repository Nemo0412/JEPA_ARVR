#!/usr/bin/env python3
"""Independent CPU audit/aggregation for the frozen single-frame source study."""
from __future__ import annotations
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from app.hdepic_lora_action_anticipation.probe_b18_q3_single_frame_source import (
    PROTOCOL, CONDITIONS, KINDS, SEEDS, bytehash, digest, write_json, construct, seeded_rng, compare_scores,
)
from scripts.egtea.analyze_b18_q3_oldest_mechanism import mask_stats


def audit_hd(item):
    from decord import VideoReader, cpu
    metadata, draws = item
    path = Path(metadata["path"])
    assert path.stat().st_size == metadata["size_bytes"] and path.stat().st_mtime_ns == metadata["mtime_ns"]
    assert digest(path) == metadata["sha256"]
    if not draws: return dict(video_id=metadata["video_id"], n_images=0)
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1, width=256, height=256)
    assert len(reader) == metadata["frame_count"]
    assert abs(reader.get_avg_fps() / metadata["source_fps"] - 1) < 1e-6
    indices = sorted(set(d["native_frame"] for d in draws))
    frames = reader.get_batch(indices).asnumpy()
    for index, image in zip(indices, frames):
        for draw in draws:
            if draw["native_frame"] == index:
                assert bytehash(image) == draw["image_sha256"]
                assert np.array_equal(image, np.load(draw["image_path"]))
    return dict(video_id=metadata["video_id"], n_images=len(indices), source_hash=True, repeated_decode_exact=True)


def audit_inputs(manifest, selected):
    assert digest(manifest["hdepic_train"]) == manifest["hdepic_train_sha256"]
    samples = manifest["source"]["samples"]
    lookup = {s["sample_id"]: s for s in samples}
    hd_meta = manifest["hdepic_videos"]
    assert len(hd_meta) == 20 and len(samples) == 86
    rows = [manifest["rows"][i] for i in sorted(selected)]
    hd_draws = [d for r in rows for d in r["draws"].values() if d["source_kind"] == "hdepic"]
    with ProcessPoolExecutor(max_workers=4) as pool:
        hd_audit = list(pool.map(audit_hd, [(m, [d for d in hd_draws if d["video_id"] == m["video_id"]]) for m in hd_meta]))
    checked = set()
    for index in sorted(selected):
        sample, row = samples[index], manifest["rows"][index]
        receiver = np.load(sample["cache"]["path"], mmap_mode="r")
        assert digest(row["image_archive"]["path"]) == row["image_archive"]["sha256"]
        images = np.load(row["image_archive"]["path"])
        assert np.array_equal(images["self_first_repeat"], receiver[128])
        for key, draw in row["draws"].items():
            kind, seed = draw["source_kind"], draw["seed"]
            rng, value = seeded_rng(seed, sample["sample_id"], kind)
            assert value == draw["derived_seed"]
            if kind == "hdepic":
                donor = hd_meta[int(rng.integers(20))]
                assert donor["video_id"] == draw["video_id"]
                assert int(rng.integers(donor["frame_count"])) == draw["native_frame"]
                source_image = np.load(draw["image_path"])
            else:
                candidates = [s for s in samples if s["participant"] != sample["participant"]]
                donor = sample if kind == "same_window" else candidates[int(rng.integers(len(candidates)))]
                assert donor["sample_id"] == draw["sample_id"]
                assert int(rng.integers(4 if kind == "same_window" else 0, 128)) == draw["local_rgb_index"]
                assert donor["frame_indices"][draw["cache_index"]] == draw["native_frame"]
                if donor["sample_id"] not in checked:
                    assert digest(donor["cache"]["path"]) == donor["cache"]["sha256"]
                    checked.add(donor["sample_id"])
                source_image = np.load(donor["cache"]["path"], mmap_mode="r")[draw["cache_index"]]
            assert np.array_equal(source_image, images[key])
            assert bytehash(images[key]) == draw["image_sha256"]
            assert bytehash(np.repeat(images[key][None], 4, axis=0)) == draw["tile_sha256"]
        for c in CONDITIONS:
            assert bytehash(construct(receiver, images, c)) == row["input_sha256"][c]
        for kind in KINDS:
            assert np.array_equal(construct(receiver, images, f"start_{kind}_{SEEDS[0]}")[:4], construct(receiver, images, f"middle_{kind}_{SEEDS[0]}")[64:68])
    return dict(n_rows=len(selected), n_inputs=len(selected) * 14, deterministic_draws_exact=True,
                native_image_and_repeated_tile_exact=True, unchanged_input_regions_exact=True, hdepic=hd_audit)


def summarize(pooled):
    out = {}
    for c in CONDITIONS:
        stats, _ = mask_stats(pooled["score_actual__" + c])
        no, _ = mask_stats(pooled["score_no_rope_readout__" + c])
        flow = pooled["flow_actual__" + c]
        headmass = flow[:, :, :64].sum(1)
        excess = flow[:, :, :2].sum(-1) - flow[:, :, :64].sum(-1) / 32
        groups = {"oldest_queries_0_1": [0, 1], "other_context_queries_2_63": list(range(2, 64)), "target_query_row64_position72": [64]}
        result = dict(actual=stats, no_rope_readout=no, start_count=stats["count"][:2], next_count=stats["count"][2:4],
            middle_count=stats["count"][32:34], start_mass_uniform=float(np.mean(stats["mass_over_uniform"][:2])),
            head_start_mass_uniform=(headmass[:, :2].mean(1) / headmass.mean(1)).tolist(),
            head_oldest_excess=excess.sum(1).tolist(), query_sources={name: dict(
                mass_uniform=float(flow[:, ids, :2].sum() / (flow[:, ids, :64].sum() / 32)),
                signed_excess=float(excess[:, ids].sum()), received_mass=float(flow[:, ids, :2].sum())) for name, ids in groups.items()},
            without_oldest_queries_mass_uniform=float(flow[:, 2:, :2].sum() / (flow[:, 2:, :64].sum() / 32)))
        trace = {}
        for key, value in pooled.items():
            if not key.endswith("__" + c): continue
            name = key.split("__")[0]
            if name.startswith("stage_"):
                trace[name] = dict(start_norm=float(value[:2, 0].mean()), middle_norm=float(value[16:48, 0].mean()),
                    start_middle_norm_ratio=float(value[:2, 0].mean() / value[16:48, 0].mean()),
                    start_alignment=float(value[:2, 5].mean()), middle_alignment=float(value[16:48, 5].mean()))
            elif name in ("q_pre_norm", "k_pre_norm", "q_post_norm", "k_post_norm"):
                trace[name] = dict(start=float(value[:, :2, 0].mean()), middle=float(value[:, 16:48, 0].mean()))
            elif name.startswith(("key_logit_", "qmean_kmean_cos_")):
                trace[name] = dict(start=float(value[:, :2].mean()), middle=float(value[:, 16:48].mean()))
        result["trace"] = trace
        out[c] = result
    return out


def figures(out, manifest, summary, pooled, masks):
    results, x = summary["conditions"], np.arange(64)
    fig, axs = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    groups = [[f"start_{kind}_{seed}" for seed in SEEDS] for kind in KINDS] + [[f"middle_{kind}_{SEEDS[0]}" for kind in KINDS]]
    for i, group in enumerate(groups):
        for j, field in enumerate(("count", "mass_over_uniform")):
            ax = axs[i, j]
            for c in ["continuous", "start_self_first_repeat"] + group:
                ax.plot(x, results[c]["actual"][field], label=c, lw=2 if c == "continuous" else 1.2)
            ax.axhline(64 if j == 0 else 1, ls=":", c="gray")
            ax.set_ylabel("kept / 256" if j == 0 else "received mass / uniform"); ax.legend(fontsize=7)
    for ax in axs[-1]: ax.set_xlabel("temporal slot (0 = oldest)")
    fig.suptitle(f"Single-image repetitions: all14 conditions; pooled {summary['n_rows']} videos, K4096")
    fig.tight_layout(); fig.savefig(out / "all_sources_temporal.png", dpi=140); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    for j, metric in enumerate(("start_count", "start_mass_uniform")):
        for half in range(2):
            vals = [np.mean(summary["calibration_halves"][half][c][metric]) for c in CONDITIONS]
            axs[0, j].plot(np.arange(14), vals, "o-", label=f"half{half} n={summary['half_n'][half]}")
        vals = [np.mean(results[c][metric]) for c in CONDITIONS]
        axs[0, j].plot(np.arange(14), vals, "k.--", label="pooled86")
        axs[0, j].set_xticks(np.arange(14), [c.replace("202609", "s") for c in CONDITIONS], rotation=80, fontsize=7)
        axs[0, j].set_title(metric); axs[0, j].legend(fontsize=8)
    for i, kind in enumerate(KINDS):
        vals = [results[f"start_{kind}_{seed}"]["start_mass_uniform"] for seed in SEEDS]
        axs[1, 0].plot(SEEDS, vals, "o-", label=kind)
    axs[1, 0].legend(); axs[1, 0].set_title("All three source draws retained; descriptive seeds")
    vals = [results[c]["individual_oldest_above_uniform"] for c in CONDITIONS]
    axs[1, 1].bar(np.arange(14), vals); axs[1, 1].set_ylim(0, summary["n_rows"])
    axs[1, 1].set_xticks(np.arange(14), [c.replace("202609", "s") for c in CONDITIONS], rotation=80, fontsize=7)
    axs[1, 1].set_title("Individual topK: videos with oldest mean keep >64")
    fig.tight_layout(); fig.savefig(out / "seed_half_stability.png", dpi=140); plt.close(fig)

    shown = ["continuous", "start_self_first_repeat"] + [f"start_{kind}_{SEEDS[0]}" for kind in KINDS]
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    for c in shown:
        flow = pooled["flow_actual__" + c].sum(0)
        z = flow[:, :2].sum(-1) / (flow[:, :64].sum(-1) / 32)
        axs[0, 0].plot(np.arange(65), z, label=c)
        axs[0, 1].plot(np.arange(12), results[c]["head_start_mass_uniform"], "o-", label=c)
        for j, name in enumerate(("key_logit_mean_actual", "qmean_kmean_cos_actual")):
            axs[1, j].plot(x, pooled[name + "__" + c][:, :64].mean(0), label=c)
    for ax in axs.ravel(): ax.legend(fontsize=7)
    axs[0, 0].set_title("Oldest-key mass by query source / within-source uniform")
    axs[0, 0].set_xlabel("query row (64 has actual target position72)")
    axs[0, 1].set_title("Oldest mass / uniform by head")
    axs[1, 0].set_title("Actual key-logit mean by temporal slot")
    axs[1, 1].set_title("Actual mean-Q/mean-K cosine by temporal slot")
    fig.tight_layout(); fig.savefig(out / "query_heads_logits.png", dpi=150); plt.close(fig)

    stages = ["patch", "encoder_block00", "encoder_block11", "encoder_block23", "encoder_final_norm", "predictor_embed", "predictor_l0_norm1"]
    fig, axs = plt.subplots(4, 2, figsize=(15, 12))
    for ax, stage in zip(axs.ravel(), stages):
        for c in shown:
            z = pooled["stage_" + stage + "__" + c][:64, 0]
            ax.plot(x, z / z[16:48].mean(), label=c)
        ax.set_title(stage + ": norm / middle-half mean"); ax.axhline(1, ls=":", c="gray")
    axs.ravel()[-1].axis("off"); axs.ravel()[-1].legend(*axs[0, 0].get_legend_handles_labels(), loc="center")
    fig.suptitle("Measured stages; these summaries do not establish a causal origin")
    fig.tight_layout(); fig.savefig(out / "stage_traces.png", dpi=140); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    for ax, c in zip(axs.ravel(), shown):
        a = results[c]["actual"]; q = np.array(a["quantiles_over_mean"])
        ax.fill_between(x, q[:, 0], q[:, 2], alpha=.2, label="q10–q90")
        ax.plot(x, q[:, 1], label="median"); ax.axhline(a["cutoff_over_mean"], ls="--", color="red", label="K4096 cutoff")
        ax.set_title(c); ax.legend(fontsize=8)
    axs.ravel()[-1].axis("off")
    fig.tight_layout(); fig.savefig(out / "scores_cutoffs.png", dpi=140); plt.close(fig)

    # Preselected first two receivers; all source draws shown as raw images.
    for index in (0, 1):
        sample, row = manifest["source"]["samples"][index], manifest["rows"][index]
        receiver = np.load(sample["cache"]["path"], mmap_mode="r")
        images = np.load(row["image_archive"]["path"])
        fig, axs = plt.subplots(3, 3, figsize=(11, 11))
        for i, kind in enumerate(KINDS):
            for j, seed in enumerate(SEEDS):
                key = f"{kind}_{seed}"; d = row["draws"][key]
                axs[i, j].imshow(images[key]); axs[i, j].axis("off")
                axs[i, j].set_title(f"{key}\n{d['video_id']} native frame{d['native_frame']}", fontsize=8)
        fig.suptitle(f"Frozen receiver{sample['sample_id']}: all nine selected single images, no output selection")
        fig.tight_layout(); fig.savefig(out / f"source_images_{sample['sample_id']}.png", dpi=130); plt.close(fig)
        rows = shown + [f"middle_{kind}_{SEEDS[0]}" for kind in KINDS]
        slots = (0, 1, 2, 32, 33)
        fig, axs = plt.subplots(len(rows), len(slots), figsize=(12, 17))
        for i, c in enumerate(rows):
            clip = construct(receiver, images, c)
            for j, t in enumerate(slots):
                ax = axs[i, j]; ax.imshow(clip[t * 2])
                kept = masks[c][t].reshape(16, 16)
                rgba = np.zeros((16, 16, 4)); rgba[..., 1] = 1; rgba[..., 3] = kept * .42
                ax.imshow(rgba, extent=(-.5, 255.5, 255.5, -.5), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"slot{t}: {kept.sum()}/256", fontsize=9)
                if j == 0: ax.set_ylabel(c, fontsize=7)
        fig.suptitle(f"{sample['sample_id']} / {sample['video_id']}; green = pooled mask, K4096")
        fig.tight_layout(); fig.savefig(out / f"rgb_masks_{sample['sample_id']}.png", dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    for name in ("manifest", "out"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--runs", nargs="+", type=Path, required=True); p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text()); assert manifest["protocol"] == PROTOCOL
    metric_dependency = Path(__file__).with_name("analyze_b18_q3_oldest_mechanism.py")
    assert digest(metric_dependency) == "5d7726c5eb7a0365757e54ef0284cc62718e26578acd988d2a808b1c3a00700f"
    source = manifest["source"]; selected, sums, half_sums = set(), {}, [{}, {}]
    half_n, runs, baseline_gates, indiv, indiv_masks = [0, 0], [], [], {c: [] for c in CONDITIONS}, {}
    max_closure = 0.
    for run in args.runs:
        meta = json.loads((run / "metadata.json").read_text()); status = json.loads((run / "summary.json").read_text())
        assert meta["protocol"] == PROTOCOL and status["run_status"] == "completed"
        assert meta["manifest_sha256"] == digest(args.manifest)
        if not args.allow_partial:
            assert len(meta["parity"]["smoke_all28_conditions"]) == 28
            assert all(v["passed"] for v in meta["parity"]["smoke_all28_conditions"].values())
        runs.append(dict(path=str(run), metadata=meta, summary=status))
        files = sorted(run.glob("q3_*.npz")); assert len(files) == status["n_rows"] == meta["stop"] - meta["start"]
        for path in files:
            with np.load(path) as a:
                index = int(a["sample_index"]); assert index not in selected and meta["start"] <= index < meta["stop"]
                selected.add(index); half = source["samples"][index]["calibration_half"]; half_n[half] += 1
                for key in a.files:
                    if key == "sample_index": continue
                    z = a[key].astype(np.float64); assert np.isfinite(z).all()
                    if key not in sums: sums[key] = np.zeros_like(z)
                    if key not in half_sums[half]: half_sums[half][key] = np.zeros_like(z)
                    sums[key] += z; half_sums[half][key] += z
                for c in CONDITIONS:
                    stats, mask = mask_stats(a["score_actual__" + c]); assert mask.sum() == 4096
                    indiv[c].append(dict(index=index, count=stats["count"], mass_over_uniform=stats["mass_over_uniform"]))
                    indiv_masks[path.stem + "__" + c] = mask
                    assert np.array_equal(a["positions__" + c], np.r_[np.arange(64), 72] * 256)
                    for suffix in ("actual", "no_rope_readout"):
                        flow = a["flow_" + suffix + "__" + c]; head = a["head_importance_" + suffix + "__" + c].reshape(12, 65, 256)
                        assert flow.shape == (12, 65, 65)
                        err = float(np.max(np.abs(flow.sum(1) - head.sum(-1)))); max_closure = max(max_closure, err)
                        assert np.allclose(flow.sum(1), head.sum(-1), rtol=1e-5, atol=.03)
                        assert abs(head.sum() / (12 * 65 * 256) - 1) < .01
                prior = "b18-q3-mechanism-full-a-17092450" if index < 43 else "b18-q3-mechanism-full-b-17092454"
                with np.load(args.manifest.parents[2] / "q3_oldest_mechanism" / prior / "capture" / path.name) as b:
                    gate = compare_scores(a["score_actual__continuous"], b["score_actual__continuous"])
                    assert gate["passed"]; baseline_gates.append(dict(index=index, **gate))
    assert args.allow_partial or selected == set(range(86))
    input_audit = audit_inputs(manifest, selected)
    pooled = {k: v / len(selected) for k, v in sums.items()}
    result = summarize(pooled)
    half_results = [summarize({k: v / half_n[h] for k, v in half_sums[h].items()}) for h in range(2)]
    masks = {c: mask_stats(pooled["score_actual__" + c])[1] for c in CONDITIONS}
    ref = np.asarray([r["count"] for r in indiv["continuous"]])
    for c in CONDITIONS:
        z = np.asarray([r["count"] for r in indiv[c]])
        result[c]["individual_count_mean"] = z.mean(0).tolist()
        result[c]["individual_oldest_above_uniform"] = int((z[:, :2].mean(1) > 64).sum())
        result[c]["individual_oldest_delta_mean"] = float((z[:, :2] - ref[:, :2]).mean())
        result[c]["individual_oldest_delta_positive"] = int(((z[:, :2] - ref[:, :2]).mean(1) > 0).sum())
    same_image = {}
    for kind in KINDS:
        a, b = result[f"start_{kind}_{SEEDS[0]}"]["actual"], result[f"middle_{kind}_{SEEDS[0]}"]["actual"]
        same_image[kind] = dict(start_count=a["count"][:2], middle_count=b["count"][32:34],
            start_mass_uniform=float(np.mean(a["mass_over_uniform"][:2])), middle_mass_uniform=float(np.mean(b["mass_over_uniform"][32:34])))
    old_path = args.manifest.parents[2] / "q3_oldest_mechanism" / "analysis-17092458" / "summary.json"
    old = json.loads(old_path.read_text())
    historical = dict(protocol=old["protocol"], source=str(old_path), source_sha256=digest(old_path),
        note="Historical independent four-frame/noise/gray controls; not pooled into the new population",
        conditions={c: {k: old["conditions"][c][k] for k in ("start_count", "start_mass_uniform")} for c in ("continuous", "start_noise_20260907", "start_gray", "start_donor")})
    summary = dict(protocol=PROTOCOL, metric_scope="selection-and-representation-diagnostic", run_status="completed", partial=args.allow_partial,
        n_rows=len(selected), n_conditions=14, sample_indices=sorted(selected), half_n=half_n, conditions=result,
        calibration_halves=half_results, same_image_position_controls=same_image, historical_controls=historical,
        source_coverage=manifest["source_coverage"], input_audit=input_audit, baseline_gates=baseline_gates, max_closure_abs=max_closure,
        manifest=str(args.manifest), manifest_sha256=digest(args.manifest), analysis_sha256=digest(__file__),
        metric_dependency=dict(path=str(metric_dependency), sha256=digest(metric_dependency)),
        limitations=manifest["limitations"] + ["no downstream accuracy", "no-RoPE is a same-QK readout removing all axes", "oldest-query source is not exact self-token attention"])
    write_json(args.out / "summary.json", summary); write_json(args.out / "input_runs.json", runs)
    write_json(args.out / "input_audit.json", input_audit); write_json(args.out / "individual_mask_summary.json", indiv)
    np.savez_compressed(args.out / "pooled_traces_scores_masks.npz", **pooled, **{"mask__" + c: v for c, v in masks.items()})
    np.savez_compressed(args.out / "individual_masks.npz", **indiv_masks)
    with (args.out / "source_draws.csv").open("w") as f:
        names = ["receiver", "key", "source_kind", "seed", "video_id", "participant", "local_rgb_index", "native_frame", "source_seconds", "image_sha256", "tile_sha256"]
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore"); writer.writeheader()
        for row in manifest["rows"]:
            for key, d in row["draws"].items(): writer.writerow(dict(receiver=row["sample_id"], key=key, **d))
    with (args.out / "per_slot.csv").open("w") as f:
        writer = csv.writer(f); writer.writerow(["condition", "slot", "count", "mass_uniform", "median_over_mean", "cutoff_over_mean"])
        for c in CONDITIONS:
            a = result[c]["actual"]
            for t in range(64): writer.writerow([c, t, a["count"][t], a["mass_over_uniform"][t], a["quantiles_over_mean"][t][1], a["cutoff_over_mean"]])
    brief = {c: {k: result[c][k] for k in ("start_count", "next_count", "middle_count", "start_mass_uniform", "individual_oldest_above_uniform")} for c in CONDITIONS}
    write_json(args.out / "brief.json", brief)
    figures(args.out, manifest, summary, pooled, masks)
    print("COMPLETE", json.dumps(dict(n_rows=len(selected), n_conditions=14, half_n=half_n, brief=brief)), flush=True)


if __name__ == "__main__": main()
