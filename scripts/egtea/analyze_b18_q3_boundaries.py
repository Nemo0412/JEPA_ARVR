#!/usr/bin/env python3
"""CPU-only analysis of frozen B18 Q3 captures; no video/model inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from app.hdepic_lora_action_anticipation.probe_b18_q3_boundaries import (
    GP, PROTOCOL, conditions, construct, digest, write_json,
)


def selection(score, k=None):
    """Exact global top-K with auditable threshold/tie allocation bounds."""
    flat = np.asarray(score, dtype=np.float64).reshape(-1)
    assert np.isfinite(flat).all() and (flat >= 0).all() and flat.sum() > 0
    k = flat.size // 4 if k is None else k
    assert 0 < k <= flat.size
    order = np.argsort(-flat, kind="stable")
    threshold = flat[order[k-1]]
    mask = np.zeros(flat.size, dtype=bool)
    mask[order[:k]] = True
    strict = (flat > threshold).reshape(-1, GP).sum(1)
    ties = (flat == threshold).reshape(-1, GP).sum(1)
    left = k - int(strict.sum())
    lower = strict + np.maximum(0, left - (ties.sum() - ties))
    upper = strict + np.minimum(ties, left)
    kept = mask.reshape(-1, GP).sum(1)
    assert kept.sum() == k and np.all(lower <= kept) and np.all(kept <= upper)
    return dict(k=k, threshold=float(threshold), threshold_over_mean=float(threshold/flat.mean()),
                kept=kept.tolist(), strict_above=strict.tolist(), threshold_ties=ties.tolist(),
                tie_count_lower=lower.tolist(), tie_count_upper=upper.tolist(),
                ties_cross_boundary=bool(0 < left < ties.sum())), mask.reshape(-1, GP)


def tie_self_check():
    # An all-tied map exposes the full allocation ambiguity, including slot 0.
    stats, _ = selection(np.ones((4, GP)))
    assert stats["ties_cross_boundary"] and stats["tie_count_lower"] == [0]*4
    assert stats["tie_count_upper"] == [256]*4 and sum(stats["kept"]) == 256
    distinct = np.arange(1, 4*GP+1).reshape(4, GP)
    stats, _ = selection(distinct)
    assert not stats["ties_cross_boundary"] and stats["kept"] == [0, 0, 0, 256]


def load_captures(manifest, manifest_path, runs, partial):
    collected, metadata, timing, gpu = {}, [], [], []
    for run in runs:
        meta = json.loads((run / "metadata.json").read_text())
        summary = json.loads((run / "summary.json").read_text())
        assert meta["protocol"] == summary["protocol"] == PROTOCOL
        assert meta["manifest_sha256"] == digest(manifest_path)
        assert meta["code_sha256"] == manifest["code_sha256"]
        assert meta["model"] == manifest["model"] and meta["dtype"] == "bf16"
        assert meta["query_chunk"] == 256 and summary["run_status"] == "completed"
        assert summary["n_conditions"] == len(conditions())
        assert meta["start"] == summary["start"] and meta["stop"] == summary["stop"]
        files = sorted(p for p in run.glob("q3_*.npz"))
        assert len(files) == summary["n_rows"] == meta["stop"] - meta["start"]
        seen_run = []
        for file in files:
            with np.load(file) as saved:
                i = int(saved["sample_index"])
                assert meta["start"] <= i < meta["stop"] and i not in collected
                assert file.stem == manifest["samples"][i]["sample_id"]
                expected = {"sample_index"} | {p + "__" + c["name"] for c in conditions()
                                              for p in ("score", "head_time", "total", "target")}
                assert set(saved.files) == expected
                data = {}
                for c in conditions():
                    name, slots = c["name"], c["seconds"]*4
                    score = saved["score__" + name]
                    head = saved["head_time__" + name]
                    total, target = saved["total__" + name], saved["target__" + name]
                    assert score.shape == (slots, GP) and head.shape == (12, slots)
                    assert total.shape == target.shape == (12,)
                    for value in (score, head, total, target):
                        assert np.isfinite(value).all() and (value >= 0).all()
                    assert np.allclose(score.sum(1), head.sum(0), rtol=1e-5, atol=1e-3)
                    assert np.allclose(head.sum(1)+target, total, rtol=1e-5, atol=1e-3)
                    assert abs(total.sum()/(12*(slots+1)*GP)-1) < .01
                    data[name] = score.copy()
                collected[i] = data
                seen_run.append(i)
        assert sorted(seen_run) == list(range(meta["start"], meta["stop"]))
        metadata.append(meta)
        timing.append(summary)
        monitor = run.parent / "gpu.csv"
        if monitor.is_file():
            samples = []
            with monitor.open() as f:
                for row in csv.reader(f):
                    if len(row) >= 3:
                        try:
                            samples.append(float(row[2]))
                        except ValueError:
                            continue
            gpu.append(dict(path=str(monitor), samples=len(samples),
                            whole_job_mean=float(np.mean(samples)) if samples else None))
    indices = sorted(collected)
    if not partial:
        assert indices == list(range(len(manifest["samples"]))), "incomplete full coverage"
    assert len(indices) > 0
    scores = {c["name"]: np.stack([collected[i][c["name"]] for i in indices]) for c in conditions()}
    return indices, scores, metadata, timing, gpu


def analyze(args):
    tie_self_check()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["protocol"] == PROTOCOL and manifest["conditions"] == conditions()
    indices, scores, metadata, timing, gpu = load_captures(manifest, args.manifest, args.runs, args.allow_partial)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "input_runs.json", [str(p) for p in args.runs])
    halves = np.asarray([manifest["samples"][i]["calibration_half"] for i in indices])
    summaries, maps, masks, curves = {}, {}, {}, {}
    for cond in conditions():
        name = cond["name"]
        raw = scores[name]
        pooled = raw.mean(0, dtype=np.float64)
        stats, mask = selection(pooled)
        online = np.asarray([selection(row)[0]["kept"] for row in raw])
        mass = pooled.sum(1)/pooled.sum()
        sample_mass = raw.astype(np.float64).sum(2)
        sample_mass /= sample_mass.sum(1, keepdims=True)
        half_stats = {}
        for half in (0, 1):
            if np.any(halves == half):
                half_stats[str(half)] = selection(raw[halves == half].mean(0, dtype=np.float64))[0]
        stats.update(condition=cond, n_samples=len(raw), mass_fraction=mass.tolist(),
                     mass_over_uniform=(mass*len(mass)).tolist(),
                     online_kept_mean=online.mean(0).tolist(),
                     online_start2_over64_fraction=float((online[:, :2].mean(1)>64).mean()),
                     sample_mass_mean=sample_mass.mean(0).tolist(),
                     calibration_halves=half_stats,
                     per_slot_mean_over_global=(pooled.mean(1)/pooled.mean()).tolist(),
                     per_slot_std_over_global=(pooled.std(1)/pooled.mean()).tolist(),
                     per_slot_quantiles_over_global=(np.quantile(pooled, [.1,.25,.5,.75,.9], axis=1)/pooled.mean()).T.tolist())
        if pooled.size >= 8192:
            stats["secondary_fixed4096"] = selection(pooled, 4096)[0]
        maps[name], masks[name] = pooled, mask
        curves[name] = dict(mass=mass, online=online, sample_mass=sample_mass)
        summaries[name] = stats
    contrasts = {}
    for cond in conditions():
        name, slots = cond["name"], cond["seconds"]*4
        kept = np.asarray(summaries[name]["kept"])
        values = dict(start0=int(kept[0]), start1=int(kept[1]), start2_mean=float(kept[:2].mean()),
                      middle_half_mean=float(kept[slots//4:3*slots//4].mean()),
                      start2_mass_over_uniform=float(np.mean(summaries[name]["mass_over_uniform"][:2])))
        if cond["cut"] is not None:
            boundary = int(slots*cond["cut"])
            base = f"continuous_{cond['seconds']}s"
            baseline = np.asarray(summaries[base]["kept"])
            values.update(cut_slot=boundary, cut_first2=kept[boundary:boundary+2].tolist(),
                          continuous_same_slots=baseline[boundary:boundary+2].tolist(),
                          pooled_cut_first2_delta=float((kept-baseline)[boundary:boundary+2].mean()),
                          online_cut_first2_delta=float((curves[name]["online"]-curves[base]["online"])[:, boundary:boundary+2].mean()),
                          cut_mass_over_uniform_delta=float(np.mean(np.asarray(summaries[name]["mass_over_uniform"])[boundary:boundary+2] - np.asarray(summaries[base]["mass_over_uniform"])[boundary:boundary+2])))
        if cond["shift"]:
            original = cond["shift"]*4
            values.update(original_first_source_slots=[original, original+1],
                          original_first_source_kept=kept[original:original+2].tolist(),
                          new_start_minus_original_source=float(kept[:2].mean()-kept[original:original+2].mean()))
        contrasts[name] = values
    summary = dict(protocol=PROTOCOL, metric_scope="selection-diagnostic", is_partial=args.allow_partial,
                   n_samples=len(indices), n_videos=len(set(manifest["samples"][i]["video_id"] for i in indices)),
                   n_participants=len(set(manifest["samples"][i]["participant"] for i in indices)),
                   conditions=summaries, boundary_contrasts=contrasts, gpu_monitor=gpu,
                   input_metadata=metadata, capture_timings=timing,
                   interpretation_limits=["finite lengths", "artificial cross-video cuts", "no action accuracy or causal utility",
                                          "donor-linked rows; disjoint-half stability is descriptive", "64/128 context slots exceed documented 32-slot predictor training extent"])
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "brief.json", dict(n_samples=len(indices), partial=args.allow_partial,
                                             boundaries=contrasts, gpu_monitor=gpu))
    np.savez_compressed(args.out / "pooled_scores_and_masks.npz",
                        **{f"score__{k}":v for k,v in maps.items()}, **{f"mask__{k}":v for k,v in masks.items()})
    with (args.out / "per_slot.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "slot", "mass_over_uniform", "offline_kept", "online_mean_kept", "strict_above", "ties", "tie_lower", "tie_upper"])
        for name, stats in summaries.items():
            for t in range(len(stats["kept"])):
                writer.writerow([name, t] + [stats[key][t] for key in ["mass_over_uniform", "kept", "online_kept_mean", "strict_above", "threshold_ties", "tie_count_lower", "tie_count_upper"]])
    plot(args.out, summaries, maps, masks, manifest, indices, args.allow_partial)
    print(json.dumps(dict(n_samples=len(indices), partial=args.allow_partial, boundary_contrasts=contrasts)), flush=True)


def plot(out, summaries, maps, masks, manifest, indices, partial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prefix = "SMOKE ONLY | " if partial else ""
    fig, ax = plt.subplots(2, 4, figsize=(18, 7))
    for col, seconds in enumerate((4, 8, 16, 32)):
        stats = summaries[f"continuous_{seconds}s"]
        slots = len(stats["kept"])
        x = np.arange(slots)
        ax[0,col].plot(x, stats["kept"], color="darkorange", label="offline topK(mean)")
        for half, values in stats["calibration_halves"].items():
            ax[0,col].plot(x, values["kept"], linestyle="--", alpha=.5, label=f"calibration half {half}")
        ax[0,col].axhline(64, color="gray", linestyle=":", label="uniform budget")
        ax[0,col].set(title=f"{seconds}s / {slots} slots", ylim=(0,270), ylabel="Retained tokens / slot")
        ax[1,col].plot(x, stats["mass_over_uniform"], color="royalblue")
        ax[1,col].axhline(1, color="gray", linestyle=":")
        ax[1,col].set(xlabel="Local time slot (0 = oldest)", ylabel="Received mass / uniform")
    ax[0,0].legend(fontsize=8)
    fig.suptitle(prefix+"Frozen predictor L0 | continuous context | 25% retained; separate score-mass and selection readouts")
    fig.tight_layout(); fig.savefig(out/"window_lengths.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(15,9))
    for axis, seconds in zip(axes.flat, (4,8,16,32)):
        axis.plot(summaries[f"continuous_{seconds}s"]["kept"], label="continuous", color="black")
        for frac, color in ((25,"tab:blue"),(50,"tab:orange"),(75,"tab:green")):
            name = f"cut{frac}_{seconds}s"
            if name in summaries:
                stats = summaries[name]
                axis.plot(stats["kept"], label=f"A to B cut at {frac}%", color=color)
                boundary = seconds*4*frac//100
                axis.axvline(boundary, color=color, linestyle=":", alpha=.6)
                axis.scatter([boundary,boundary+1], np.asarray(stats["kept"])[boundary:boundary+2], color=color, zorder=4)
        axis.axhline(64,color="gray",linestyle=":")
        axis.set(title=f"{seconds}s | each condition independently recalibrated", xlabel="Local slot; dotted lines = cut", ylabel="Offline retained tokens / slot", ylim=(0,270))
        axis.legend(fontsize=8)
    fig.suptitle(prefix+"Scene-cut test | raw-video concatenation; continuous position indices | top-25%")
    fig.tight_layout(); fig.savefig(out/"scene_cuts.png",dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(15,5))
    for name, shift, color in (("continuous_16s",0,"black"),("shift_m4_16s",16,"tab:blue"),("shift_m8_16s",32,"tab:orange")):
        y = summaries[name]["kept"]
        axes[0].plot(np.arange(64),y,label=name,color=color)
        axes[0].scatter([shift,shift+1],np.asarray(y)[shift:shift+2],color=color)
        axes[1].plot(np.arange(64)-shift,y,label=name,color=color)
        axes[1].axvline(-shift,linestyle=":",color=color)
    for axis in axes:
        axis.axhline(64,color="gray",linestyle=":"); axis.legend(fontsize=8)
        axis.set(ylabel="Offline retained tokens / slot",ylim=(0,270))
    axes[0].set(xlabel="Local slot (all windows begin at 0)",title="Dots track original first source frames")
    axes[1].set(xlabel="Source-time slot relative to original window start",title="Same source frames align; dotted lines = new starts")
    fig.suptitle(prefix+"Moving window-start test | 16s windows with exact shared RGB | independently recalibrated")
    fig.tight_layout(); fig.savefig(out/"moving_starts.png",dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,4,figsize=(18,4))
    for axis, seconds in zip(axes,(4,8,16,32)):
        name = f"continuous_{seconds}s"
        m = maps[name]/maps[name].mean()
        for slot in (0,1,len(m)//2):
            axis.step(np.sort(m[slot]),np.arange(1,257)/256,where="post",label=f"slot {slot}")
        axis.axvline(summaries[name]["threshold_over_mean"],color="black",linestyle=":",label="global top-K threshold")
        axis.set(title=f"{seconds}s",xlabel="Token score / global mean",ylabel="Fraction of slot tokens below score",xlim=(0,3))
        axis.legend(fontsize=7)
    fig.suptitle(prefix+"Within-slot score distributions | similar means need not imply equal threshold exceedance")
    fig.tight_layout(); fig.savefig(out/"score_distributions.png",dpi=160); plt.close(fig)
    sample = manifest["samples"][indices[0]]
    donor_sample = next(s for s in manifest["samples"] if s["sample_id"]==sample["donor_sample_id"])
    receiver, donor = np.load(sample["cache"]["path"],mmap_mode="r"),np.load(donor_sample["cache"]["path"],mmap_mode="r")
    fig, axes = plt.subplots(3,8,figsize=(20,8))
    for row, name in enumerate(("continuous_16s","cut50_16s","shift_m4_16s")):
        cond = next(c for c in conditions() if c["name"]==name)
        rgb = construct(receiver,donor,cond)
        for axis, slot in zip(axes[row],(0,1,16,31,32,33,48,63)):
            pixel_mask = np.repeat(np.repeat(masks[name][slot].reshape(16,16),16,0),16,1)
            axis.imshow(rgb[slot*2]*pixel_mask[...,None])
            axis.set_title(f"{name}\nslot {slot}; kept {masks[name][slot].sum()}",fontsize=8)
            axis.axis("off")
    fig.suptitle(prefix+f"Black = dropped | fixed pooled masks illustrated on receiver {sample['video_id']}\nDonor prefix {donor_sample['video_id']}; cut50 changes scene before slot 32; no position reset",fontsize=11)
    fig.tight_layout(); fig.savefig(out/"boundary_rgb_masks.png",dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--runs",type=Path,nargs="+",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--allow-partial",action="store_true")
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
