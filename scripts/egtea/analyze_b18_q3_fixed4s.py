#!/usr/bin/env python3
"""Fixed four-second token-budget reanalysis of immutable Q3 captures (CPU)."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from analyze_b18_q3_boundaries import load_captures, selection, tie_self_check
from app.hdepic_lora_action_anticipation.probe_b18_q3_boundaries import (
    PROTOCOL, conditions, construct, digest, write_json,
)

K = 4096
SLOTS_PER_SECOND = 4


def mass(score):
    return score.mean(1) / score.mean()


def contrast(kept, baseline, score_mass, baseline_mass, boundary):
    delta = np.asarray(kept) - np.asarray(baseline)
    dm = np.asarray(score_mass) - np.asarray(baseline_mass)
    offsets = np.arange(-8, 9)
    offsets = offsets[(boundary + offsets >= 0) & (boundary + offsets < len(delta))]
    near = np.arange(boundary - 4, boundary + 4)
    return dict(
        boundary_slot=boundary, offsets_slots=offsets.tolist(),
        kept_delta=delta[boundary + offsets].tolist(),
        mass_over_uniform_delta=dm[boundary + offsets].tolist(),
        pre1s_mean_delta=float(delta[boundary-4:boundary].mean()),
        post1s_mean_delta=float(delta[boundary:boundary+4].mean()),
        post2_slots_mean_delta=float(delta[boundary:boundary+2].mean()),
        neighborhood2s_mean_delta=float(delta[near].mean()),
        neighborhood2s_peak_delta=int(delta[near].max()),
        neighborhood2s_peak_offset=int(near[np.argmax(delta[near])] - boundary),
        pre1s_mass_delta=float(dm[boundary-4:boundary].mean()),
        post1s_mass_delta=float(dm[boundary:boundary+4].mean()),
    )


def mosaic(mask):
    """Render all spatial 16x16 masks in temporal row-major order."""
    n = len(mask)
    rows = (n + 15) // 16
    canvas = np.full((rows*18-2, 16*18-2), .5)
    for t, tile in enumerate(mask):
        r, c = divmod(t, 16)
        canvas[r*18:r*18+16, c*18:c*18+16] = tile.reshape(16, 16)
    return canvas


def plot(out, summaries, masks, cuts, old, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(out / name, dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for col, seconds in enumerate((4, 8, 16, 32)):
        name = f"continuous_{seconds}s"
        s = summaries[name]
        x = np.arange(len(s["kept"])) / SLOTS_PER_SECOND
        ax = axes[0, col]
        ax.plot(x, s["kept"], color="darkorange", label="fixed 4096 tokens")
        ax.plot(x, old["conditions"][name]["kept"], color="black", ls="--", alpha=.55, label="25% comparison")
        for h, v in s["calibration_halves"].items():
            ax.plot(x, v["kept"], color="darkorange", ls=":" if h == "0" else "--", alpha=.4)
        ax.axhline(K/len(x), color="gray", ls=":", label="uniform fixed-budget allocation")
        ax.set(title=f"{seconds}s input -> 4s budget ({100*4/seconds:g}%)", ylim=(0, 270), ylabel="Retained tokens / slot")
        ax = axes[1, col]
        ax.plot(x, s["mass_over_uniform"], color="royalblue", label="mean score / global mean")
        ax.axhline(1, color="gray", ls=":")
        ax.axhline(s["threshold_over_mean"], color="darkorange", ls="--", label="fixed-budget threshold")
        ax.set(xlabel="Seconds from window start (oldest at 0)", ylabel="Normalized score")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Frozen L0 offline selection | 86 videos | fixed 4096 context tokens (4s equivalent, not contiguous 4s)\nFaint orange curves: disjoint 43-video calibrations; score curves do not depend on K")
    save(fig, "fixed4s_windows.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, seconds in zip(axes.flat, (4, 8, 16, 32)):
        n = seconds * SLOTS_PER_SECOND
        x = np.arange(n) / SLOTS_PER_SECOND
        ax.plot(x, summaries[f"continuous_{seconds}s"]["kept"], color="black", label="continuous")
        for pct, color in ((25, "tab:blue"), (50, "tab:orange"), (75, "tab:green")):
            name = f"cut{pct}_{seconds}s"
            if name in summaries:
                ax.plot(x, summaries[name]["kept"], color=color, label=f"cut at {pct}%")
                ax.axvline(seconds*pct/100, color=color, ls=":", alpha=.7)
        ax.axhline(K/n, color="gray", ls=":")
        ax.set(title=f"{seconds}s input -> fixed 4s token budget", xlabel="Seconds from window start", ylabel="Retained tokens / slot", ylim=(0, 270))
        ax.legend(fontsize=8)
    fig.suptitle("Artificial scene cuts | independently recalibrated offline masks | fixed 4096 tokens\nOriginal continuous position indices; 4s input retains every token")
    save(fig, "fixed4s_scene_cuts.png")

    fig, axes = plt.subplots(4, 2, figsize=(17, 14))
    for ax, (name, c) in zip(axes.flat, cuts.items()):
        x = np.asarray(c["offsets_slots"]) / SLOTS_PER_SECOND
        y = np.asarray(c["kept_delta"])
        ax.axhline(0, color="gray", lw=.7)
        ax.axvline(0, color="gray", ls=":")
        ax.plot(x, y, "o-", ms=3, color="tab:blue", label="retained-token delta")
        ax.fill_between(x, 0, y, where=y>=0, color="tab:blue", alpha=.1)
        ax.fill_between(x, 0, y, where=y<0, color="tab:red", alpha=.1)
        for h, v in c["calibration_halves"].items():
            ax.plot(x, v["kept_delta"], color="tab:blue", ls=":" if h == "0" else "--", alpha=.35, label=f"calibration half {h}")
        ax.set(title=name, xlabel="Seconds relative to cut (0 = first post-cut slot)", ylabel="Cut - continuous: tokens / slot", xlim=(-2, 2))
        right = ax.twinx()
        right.plot(x, c["mass_over_uniform_delta"], "s-", ms=3, color="darkorange", label="normalized mean-score delta")
        right.set_ylabel("Cut - continuous: normalized score", color="darkorange")
        # Align both zero baselines; limits include all half-calibration curves.
        yl = max(1, max(abs(v) for v in c["kept_delta"]),
                 max(abs(v) for h in c["calibration_halves"].values() for v in h["kept_delta"])) * 1.15
        ml = max(.001, max(abs(v) for v in c["mass_over_uniform_delta"])) * 1.15
        ax.set_ylim(-yl, yl)
        right.set_ylim(-ml, ml)
        if name == next(iter(cuts)):
            ax.legend(loc="upper left", fontsize=7)
            right.legend(loc="lower left", fontsize=7)
    fig.suptitle("Cut-aligned differences at fixed 4096 tokens | +/-2s exploratory view\nEach curve subtracts its matched continuous length; axes rescale per panel; halves describe stability, not confidence intervals")
    save(fig, "fixed4s_cut_deltas.png")

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    for ax, seconds in zip(axes.flat, (4, 8, 16, 32)):
        m = masks[f"continuous_{seconds}s"]
        ax.imshow(mosaic(m), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set(title=f"{seconds}s input | {len(m)} slots | exactly {int(m.sum())} retained tokens",
               xticks=np.arange(16)*18+7.5, xticklabels=np.arange(16),
               yticks=np.arange((len(m)+15)//16)*18+7.5,
               yticklabels=np.arange(0, len(m), 16), xlabel="Slot offset within row", ylabel="First slot in row")
    fig.suptitle("All spatial masks at fixed 4s budget | each tile = 16x16 spatial tokens\nWhite = retained; black = dropped; gray = tile separators; temporal order runs left to right, then down")
    save(fig, "fixed4s_spatial_masks.png")

    mask_dir = out / "spatial_masks"
    mask_dir.mkdir()
    for name, m in masks.items():
        plt.imsave(mask_dir / f"{name}.png", mosaic(m), cmap="gray", vmin=0, vmax=1)

    fig, axes = plt.subplots(8, 3, figsize=(18, 14))
    for row, (name, c) in enumerate(cuts.items()):
        seconds = summaries[name]["condition"]["seconds"]
        base_name = f"continuous_{seconds}s"
        b = c["boundary_slot"]
        selected = list(range(b-4, b+4))
        for col, this_name in enumerate((base_name, name)):
            tiles = np.concatenate([masks[this_name][t].reshape(16, 16) for t in selected], axis=1)
            axes[row, col].imshow(tiles, cmap="gray", vmin=0, vmax=1)
        a = np.concatenate([masks[base_name][t].reshape(16, 16) for t in selected], axis=1)
        z = np.concatenate([masks[name][t].reshape(16, 16) for t in selected], axis=1)
        rgb = np.full((*a.shape, 3), .85)
        rgb[z & ~a] = (0.1, .65, .3)
        rgb[a & ~z] = (.85, .15, .15)
        axes[row, 2].imshow(rgb)
        for col, ax in enumerate(axes[row]):
            ax.axvline(63.5, color="royalblue", lw=1)
            ax.set(xticks=np.arange(8)*16+7.5, xticklabels=np.arange(-4, 4), yticks=[])
            ax.set_title(f"{name}: " + ("continuous", "cut", "added green / removed red")[col], fontsize=9)
    fig.suptitle("Spatial selection around every cut | offsets -4..+3 slots (-1s..+0.75s)\nBlue divider = scene cut; each tile is 16x16; fixed 4096 globally, so changes can redistribute budget outside this view")
    save(fig, "fixed4s_cut_spatial_changes.png")

    sample = manifest["samples"][0]
    donor_sample = next(s for s in manifest["samples"] if s["sample_id"] == sample["donor_sample_id"])
    for item in (sample, donor_sample):
        assert digest(item["cache"]["path"]) == item["cache"]["sha256"]
    receiver = np.load(sample["cache"]["path"], mmap_mode="r")
    donor = np.load(donor_sample["cache"]["path"], mmap_mode="r")
    fig, axes = plt.subplots(4, 8, figsize=(20, 11))
    for row, seconds in enumerate((4, 8, 16, 32)):
        n, b = seconds*4, seconds*2
        choices = (0, 1, b-1, b, b+1, b+2, n-2, n-1)
        for col, t in enumerate(choices):
            name = f"continuous_{seconds}s"
            cond = summaries[name]["condition"]
            rgb = construct(receiver, donor, cond)
            pm = np.repeat(np.repeat(masks[name][t].reshape(16,16), 16, 0), 16, 1)
            axes[row, col].imshow(rgb[t*2] * pm[..., None])
            axes[row, col].set_title(f"{seconds}s | slot {t}\nkept {int(masks[name][t].sum())}", fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle(f"Fixed pooled masks illustrated on frozen sample 0: {sample['video_id']}\nBlack = dropped; four lengths share the same endpoint; this is an illustration, not per-example selection")
    save(fig, "fixed4s_rgb_masks.png")


def analyze(args):
    tie_self_check()
    manifest = json.loads(args.manifest.read_text())
    old = json.loads((args.source / "summary.json").read_text())
    runs = [Path(p) for p in json.loads((args.source / "input_runs.json").read_text())]
    assert manifest["protocol"] == old["protocol"] == PROTOCOL
    assert not old["is_partial"] and old["n_samples"] == 86
    indices, scores, metadata, _, _ = load_captures(manifest, args.manifest, runs, False)
    assert indices == list(range(86))
    halves = np.asarray([manifest["samples"][i]["calibration_half"] for i in indices])
    assert np.bincount(halves).tolist() == [43, 43]
    summaries, maps, masks, per_example = {}, {}, {}, {}
    with np.load(args.source / "pooled_scores_and_masks.npz") as original:
        for cond in conditions():
            name = cond["name"]
            raw = scores[name]
            pooled = raw.mean(0, dtype=np.float64)
            assert np.array_equal(pooled, original["score__" + name])
            primary, primary_mask = selection(pooled)
            assert primary["kept"] == old["conditions"][name]["kept"]
            assert np.array_equal(primary_mask, original["mask__" + name])
            s, m = selection(pooled, K)
            if "secondary_fixed4096" in old["conditions"][name]:
                assert s == old["conditions"][name]["secondary_fixed4096"]
            if len(m) == 16:
                assert m.all() and not s["ties_cross_boundary"]
            hs = {}
            for h in (0, 1):
                hmap = raw[halves == h].mean(0, dtype=np.float64)
                hs[str(h)] = selection(hmap, K)[0]
                hs[str(h)]["mass_over_uniform"] = mass(hmap).tolist()
            counts = np.asarray([selection(r, K)[0]["kept"] for r in raw])
            s.update(condition=cond, n_samples=len(indices), calibration_halves=hs,
                     mass_over_uniform=mass(pooled).tolist(),
                     per_example_kept_mean=counts.mean(0).tolist(),
                     per_example_start2_above_uniform_fraction=float((counts[:, :2].mean(1) > K/len(m)).mean()),
                     start2=s["kept"][:2], end2=s["kept"][-2:],
                     uniform_tokens_per_slot=K/len(m),
                     oldest4s_budget_fraction=float(m[:16].sum()/K),
                     recent4s_budget_fraction=float(m[-16:].sum()/K),
                     peak_slot=int(np.argmax(s["kept"])), peak_kept=int(max(s["kept"])))
            summaries[name], maps[name], masks[name], per_example[name] = s, pooled, m, counts
    cuts = {}
    for seconds in (4, 8, 16, 32):
        for pct in (25, 50, 75):
            name = f"cut{pct}_{seconds}s"
            if name not in summaries:
                continue
            s, base = summaries[name], summaries[f"continuous_{seconds}s"]
            b = seconds*4*pct//100
            c = contrast(s["kept"], base["kept"], s["mass_over_uniform"], base["mass_over_uniform"], b)
            c["calibration_halves"] = {h: contrast(s["calibration_halves"][h]["kept"], base["calibration_halves"][h]["kept"],
                                                  s["calibration_halves"][h]["mass_over_uniform"], base["calibration_halves"][h]["mass_over_uniform"], b) for h in ("0", "1")}
            d = per_example[name] - per_example[f"continuous_{seconds}s"]
            c["per_example_neighborhood2s_mean_delta"] = float(d[:, b-4:b+4].mean())
            c["per_example_neighborhood2s_positive_fraction"] = float((d[:, b-4:b+4].mean(1) > 0).mean())
            cuts[name] = c
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = {str(p): digest(p) for p in [args.manifest, args.source/"summary.json", args.source/"pooled_scores_and_masks.npz", Path(__file__), Path(__file__).with_name("analyze_b18_q3_boundaries.py"), Path(__file__).with_name("run_b18_q3_fixed4s_cpu.slurm")]}
    for run in runs:
        for p in sorted(run.glob("*.npz")):
            provenance[str(p)] = digest(p)
    write_json(args.out/"provenance.json", dict(inputs=provenance, input_runs=[str(p) for p in runs], capture_metadata=metadata))
    summary = dict(protocol=PROTOCOL, analysis_id="fixed4096-cut-neighborhood-v1", metric_scope="selection-diagnostic",
                   eval_path="CPU reanalysis of frozen full86 L0 captures", job_id=os.environ.get("SLURM_JOB_ID"),
                   n_samples=86, n_conditions=len(summaries), token_budget=K,
                   interpretation_limits=["Fixed token count equals four seconds, not contiguous footage", "Original primary top25 results unchanged; fixed4096 was an existing secondary readout", "Cut-neighborhood summaries are exploratory after inspection", "Donor-linked rows and calibration halves are descriptive, not confidence intervals", "No downstream task accuracy or natural-cut generalization"],
                   conditions=summaries, cut_contrasts=cuts)
    write_json(args.out/"summary.json", summary)
    np.savez_compressed(args.out/"fixed4s_masks.npz", **masks)
    np.savez_compressed(args.out/"per_example_keep_counts.npz", **per_example)
    with (args.out/"per_slot.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["condition", "slot", "seconds_from_start", "fixed4096_kept", "top25_kept", "mass_over_uniform", "half0_kept", "half1_kept"])
        for name, s in summaries.items():
            for t, count in enumerate(s["kept"]):
                w.writerow([name, t, t/4, count, old["conditions"][name]["kept"][t], s["mass_over_uniform"][t], s["calibration_halves"]["0"]["kept"][t], s["calibration_halves"]["1"]["kept"][t]])
    plot(args.out, summaries, masks, cuts, old, manifest)
    write_json(args.out/"completion.json", dict(run_status="completed", job_id=os.environ.get("SLURM_JOB_ID"), n_samples=86, n_conditions=len(summaries), all_masks_exactly4096=all(int(m.sum()) == K for m in masks.values()), primary_reproduced=True, cached_fixed4096_reproduced=True, png_count=len(list(args.out.rglob("*.png")))))
    print(json.dumps(dict(run_status="completed", out=str(args.out), n_samples=86, n_conditions=len(summaries))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args())
