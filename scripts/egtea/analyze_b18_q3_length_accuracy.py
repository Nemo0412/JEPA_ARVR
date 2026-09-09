#!/usr/bin/env python3
"""Independent CPU artifact audit and session-cluster paired uncertainty."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from app.hdepic_lora_action_anticipation.eval_b18_q3_length_accuracy import (
    PROTOCOL, LENGTHS, STRATEGIES, HORIZONS, KEEP, sha, write_json, model_args, make_masks, CALIB_MANIFEST,
)
from app.hdepic_lora_action_anticipation import train_stream_mtp as T


def monitor_stats(run, timing):
    records = []
    with (run.parent / "gpu.csv").open() as f:
        for line in f:
            fields = [v.strip() for v in line.split(",")]
            if len(fields) != 5:
                continue
            records.append((datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(fields[2])))
    arr = np.asarray(records)
    begin, end = timing[1]["start"] if len(timing) > 1 else timing[0]["start"], timing[-1]["compute_end"]
    selected = arr[(arr[:, 0] >= begin) & (arr[:, 0] <= end), 1]
    return dict(path=str(run.parent / "gpu.csv"), interval_seconds=5, whole_samples=len(arr),
                whole_mean=float(arr[:, 1].mean()), steady_samples=len(selected),
                steady_mean=float(selected.mean()) if len(selected) else None,
                steady_start_epoch=begin, steady_end_epoch=end,
                steady_definition="second measured batch start through last compute end; excludes startup/anchors/first measured batch")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--partial", action="store_true")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--failed-anchors", nargs="*", type=Path, default=[])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    assert manifest["protocol"] == PROTOCOL
    masks, mask_audit = make_masks()
    with np.load(args.manifest.parent / "fixed_masks.npz") as frozen:
        assert all(np.array_equal(frozen[k], v) for k, v in masks.items())
    arms = [f"{s}s__{a}" for s in LENGTHS for a in STRATEGIES]
    metrics = ["top1", "top3", "top5", "ce"]
    records, job_reports, source_versions = [], [], []
    maps = T.load_action_maps(model_args().train_csv)
    calibration_manifest = json.loads(CALIB_MANIFEST.read_text())
    assert calibration_manifest["train_csv_sha256"] == manifest["source_sha256"][str(model_args().train_csv)]
    for entry in calibration_manifest["model"].values():
        assert manifest["source_sha256"][entry["path"]] == entry["sha256"], "calibration/evaluation model mismatch"
    frozen_samples = manifest["samples"]
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        assert summary["protocol"] == PROTOCOL and summary["manifest_sha256"] == sha(args.manifest)
        assert summary["class_counts"] == [len(v) for v in maps]
        assert summary["target_positions"]["first_target"] == 6144
        assert summary["target_positions"]["n_target"] == 256
        assert summary["target_positions"]["context"] == list(range(KEEP))
        source_versions.append(summary["source_sha256"])
        rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
        assert len(rows) == summary["rows"] == summary["stop"] - summary["start"]
        expected = frozen_samples[summary["start"]:summary["stop"]]
        assert [r["sample_id"] for r in rows] == [s["sample_id"] for s in expected]
        for r, s in zip(rows, expected):
            for key in ["common_index", "q1_execution_index", "selection_index", "source_index", "video_id", "participant_id"]:
                assert r[key] == s[key], key
            assert set(r["arms"]) == set(arms)
            sr = s["row"]
            vs, ns, ms = [[float(v) for v in sr[key].split(",")] for key in ["mtp_verbs", "mtp_nouns", "mtp_mask"]]
            for hi, horizon in enumerate(HORIZONS):
                h = f"{horizon:g}s"
                label = r["labels"][h]
                valid = ms[hi] > 0.5 and (int(vs[hi]), int(ns[hi])) in maps[2]
                assert label["valid"] == valid
                if valid:
                    assert label["label"] == maps[2][(int(vs[hi]), int(ns[hi]))]
            assert r["arms"][arms[0]] == r["arms"][arms[1]] == r["arms"][arms[2]], "4s alias identity"
        # Reconstruct correctness and CE independently from all saved FP32 logits.
        cursor = 0
        for path in sorted(run.glob("logits_*.npz")):
            with np.load(path) as z:
                ids = z["sample_ids"].tolist()
                batch = rows[cursor:cursor + len(ids)]
                assert ids == [r["sample_id"] for r in batch]
                for arm in arms:
                    logits = z[arm]
                    assert logits.shape == (len(ids), 3, len(maps[2])) and np.isfinite(logits).all()
                    for bi, r in enumerate(batch):
                        for hi, horizon in enumerate(HORIZONS):
                            h = f"{horizon:g}s"
                            x = torch.from_numpy(logits[bi, hi])
                            top = x.topk(5).indices.tolist()
                            saved = r["arms"][arm][h]
                            assert top == saved["top5_indices"]
                            tie = {}
                            for k in [1, 3, 5]:
                                threshold = float(x.topk(k).values[-1])
                                above = int((x > threshold).sum())
                                equal = int((x == threshold).sum())
                                tie[f"top{k}"] = dict(boundary_tie=above < k < above + equal,
                                                       n_equal=equal, threshold=threshold)
                            if r["labels"][h]["valid"]:
                                y = r["labels"][h]["label"]
                                assert saved["label"] == y
                                for k in [1, 3, 5]:
                                    assert saved[f"top{k}"] == (y in top[:k])
                                    kt = tie[f"top{k}"]
                                    yscore = float(x[y])
                                    kt.update(lower=yscore > kt["threshold"] or (yscore == kt["threshold"] and not kt["boundary_tie"]),
                                              upper=yscore >= kt["threshold"])
                                ce = float(torch.nn.functional.cross_entropy(x[None], torch.tensor([y])))
                                assert abs(ce - saved["ce"]) < 1e-6
                            r.setdefault("tie_audit", {}).setdefault(arm, {})[h] = tie
            cursor += len(ids)
        assert cursor == len(rows)
        timing = [json.loads(v) for v in (run / "timings.jsonl").read_text().splitlines()]
        assert sum(t["rows"] for t in timing) == len(rows)
        job_reports.append(dict(run=str(run), job_id=summary["job_id"], n=len(rows),
            start=summary["start"], stop=summary["stop"], gpu=summary["gpu"],
            seconds=summary["seconds"], row_seconds=summary["seconds"] / len(rows),
            peak_cuda_bytes=summary["peak_cuda_bytes"], monitor=monitor_stats(run, timing),
            hardware_comparison=summary.get("hardware_comparison"), parity=summary["parity"],
            summary_sha256=sha(run / "summary.json"), predictions_sha256=sha(run / "predictions.jsonl")))
        records.extend(rows)
    assert all(v == source_versions[0] for v in source_versions), "different execution source versions across shards"
    records.sort(key=lambda r: r["common_index"])
    assert len({r["sample_id"] for r in records}) == len(records), "overlapping shards"
    if not args.partial:
        assert [r["common_index"] for r in records] == list(range(manifest["common_n"])), "incomplete final union"
    sessions = sorted({r["video_id"] for r in records})
    si = {s: i for i, s in enumerate(sessions)}
    values = np.full((len(records), len(arms), 3, 4), np.nan)
    tie_lower = np.full((len(records), len(arms), 3, 3), np.nan)
    tie_upper = np.full_like(tie_lower, np.nan)
    tie_boundary = np.zeros_like(tie_lower, dtype=bool)
    valid = np.zeros((len(records), 3), dtype=bool)
    for ri, r in enumerate(records):
        for hi, horizon in enumerate(HORIZONS):
            h = f"{horizon:g}s"
            valid[ri, hi] = r["labels"][h]["valid"]
            if valid[ri, hi]:
                for ai, arm in enumerate(arms):
                    values[ri, ai, hi] = [r["arms"][arm][h][m] for m in metrics]
                    tie_lower[ri, ai, hi] = [r["tie_audit"][arm][h][m]["lower"] for m in metrics[:3]]
                    tie_upper[ri, ai, hi] = [r["tie_audit"][arm][h][m]["upper"] for m in metrics[:3]]
                    tie_boundary[ri, ai, hi] = [r["tie_audit"][arm][h][m]["boundary_tie"] for m in metrics[:3]]
    denom = np.zeros((len(sessions), 3))
    sums = np.zeros((len(sessions), len(arms), 3, 4))
    for ri, r in enumerate(records):
        idx = si[r["video_id"]]
        denom[idx] += valid[ri]
        sums[idx] += np.nan_to_num(values[ri])
    rng = np.random.default_rng(20260907)
    weights = rng.multinomial(len(sessions), np.full(len(sessions), 1 / len(sessions)), size=args.bootstrap)
    bdenom = weights @ denom
    bsums = np.einsum("bs,sahm->bahm", weights, sums)
    boot = bsums / bdenom[:, None, :, None]
    mean = sums.sum(0) / denom.sum(0)[None, :, None]
    estimates = []
    for ai, arm in enumerate(arms):
        for hi, horizon in enumerate(HORIZONS):
            for mi, metric in enumerate(metrics):
                scale = 1 if metric == "ce" else 100
                estimates.append(dict(arm=arm, horizon=horizon, metric=metric,
                    n=int(denom[:, hi].sum()), estimate=float(mean[ai, hi, mi] * scale),
                    ci95=(np.quantile(boot[:, ai, hi, mi], [0.025, 0.975]) * scale).tolist(),
                    units="nats" if metric == "ce" else "percent"))
                if metric != "ce":
                    lower, upper = tie_lower[:, ai, hi, mi], tie_upper[:, ai, hi, mi]
                    estimates[-1].update(tie_accuracy_bounds=[float(np.nanmean(lower) * 100), float(np.nanmean(upper) * 100)],
                        boundary_tie_rows=int(tie_boundary[:, ai, hi, mi].sum()),
                        true_label_ambiguous_rows=int((upper > lower).sum()))
    comparisons = []
    def add(kind, a, b):
        comparisons.append((kind, arms.index(a), arms.index(b)))
    for strategy in STRATEGIES:
        for older, newer in zip(LENGTHS[:-1], LENGTHS[1:]):
            add("adjacent_length", f"{newer}s__{strategy}", f"{older}s__{strategy}")
        for sec in LENGTHS[1:]:
            add("versus_encoded4s", f"{sec}s__{strategy}", "4s__recent4s")
    for sec in LENGTHS:
        for strategy in [STRATEGIES[0], STRATEGIES[2]]:
            add("matched_length_vs_recent", f"{sec}s__{strategy}", f"{sec}s__recent4s")
    contrasts = []
    for kind, a, b in comparisons:
        for hi, horizon in enumerate(HORIZONS):
            for mi, metric in enumerate(metrics):
                scale = 1 if metric == "ce" else 100
                contrast = dict(kind=kind, a=arms[a], b=arms[b], horizon=horizon, metric=metric,
                    n=int(denom[:, hi].sum()), delta=float((mean[a, hi, mi] - mean[b, hi, mi]) * scale),
                    ci95=(np.quantile(boot[:, a, hi, mi] - boot[:, b, hi, mi], [0.025, 0.975]) * scale).tolist(),
                    units="nats" if metric == "ce" else "percentage_points")
                if metric != "ce":
                    d = values[:, a, hi, mi] - values[:, b, hi, mi]
                    contrast.update(wins=int((d > 0).sum()), losses=int((d < 0).sum()))
                    contrast["tie_delta_bounds"] = [
                        float(np.nanmean(tie_lower[:, a, hi, mi] - tie_upper[:, b, hi, mi]) * 100),
                        float(np.nanmean(tie_upper[:, a, hi, mi] - tie_lower[:, b, hi, mi]) * 100)]
                    if arms[a].startswith("4s__") and arms[b].startswith("4s__"):
                        contrast["tie_delta_bounds"] = [0.0, 0.0]  # identical aliased logits/ranking
                contrasts.append(contrast)
    report = dict(protocol=PROTOCOL, metric_scope="native", is_partial=args.partial,
        population=dict(source_n=4000, common_n=manifest["common_n"], evaluated_n=len(records),
                        excluded_n=manifest["excluded_n"], sessions=len(sessions),
                        participants=len({r["participant_id"] for r in records}),
                        valid_denominators=denom.sum(0).astype(int).tolist()),
        bootstrap=dict(unit="session/video", n_clusters=len(sessions), draws=args.bootstrap, seed=20260907,
                       estimator="ratio of summed row numerators to summed valid denominators in sampled sessions",
                       interval="paired percentile95%; exploratory multiple contrasts; reused validation"),
        class_ranking="CPU torch.topk on FP32 exported BF16 logits; arbitrary backend cutoff-tie order. Bounds report possible label membership; historical Q1 CUDA ranking is not asserted equal.",
        manifest=str(args.manifest), manifest_sha256=sha(args.manifest), masks=mask_audit,
        input_runs=job_reports, estimates=estimates, contrasts=contrasts,
        gates=dict(exact_final_union=not args.partial, unique_rows=True, logits_recomputed=True,
                   four_second_alias_identity=True, frozen_masks_exact=True, labels_native_exact=True,
                   constant_context_and_target_positions=True, source_versions_identical=True))
    failed_audits = []
    for failed_path in args.failed_anchors:
        reference_path = args.manifest.parent.parent / "b18-q3-length-smoke-17091977/eval/hardware_anchor.npz"
        audit = dict(path=str(failed_path), reference=str(reference_path), arrays={})
        with np.load(reference_path) as ref, np.load(failed_path) as failed:
            assert set(ref.files) == set(failed.files)
            for key in ref.files:
                if key.startswith("indices") or key.startswith("sample"):
                    assert np.array_equal(ref[key], failed[key])
                    continue
                a, b = failed[key].astype(np.float64), ref[key].astype(np.float64)
                delta = a - b
                audit["arrays"][key] = dict(relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(b)),
                    max_abs=float(abs(delta).max()),
                    top1_agreement=float((a.argmax(-1) == b.argmax(-1)).mean()))
        failed_audits.append(audit)
    report["excluded_hardware_anchor_audits"] = failed_audits
    write_json(args.out / "summary.json", report)
    write_json(args.out / "primary.json", dict(population=report["population"],
        estimates=[e for e in estimates if e["horizon"] == 2 and e["metric"] == "top5"],
        contrasts=[c for c in contrasts if c["horizon"] == 2 and c["metric"] == "top5"],
        jobs=job_reports))
    np.savez_compressed(args.out / "paired_metrics.npz", values=values, valid=valid,
                        tie_lower=tie_lower, tie_upper=tie_upper, tie_boundary=tie_boundary,
                        sample_ids=np.asarray([r["sample_id"] for r in records]), arms=np.asarray(arms),
                        sessions=np.asarray([r["video_id"] for r in records]),
                        metrics=np.asarray(metrics), horizons=np.asarray(HORIZONS))
    colors = ["#c96823", "#2575b6", "#34874a"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for hi, h in enumerate(HORIZONS):
        for strategy, color in zip(STRATEGIES, colors):
            ix = [arms.index(f"{sec}s__{strategy}") for sec in LENGTHS]
            axes[hi].plot(LENGTHS, mean[ix, hi, 2] * 100, "o-", color=color, label=strategy)
        axes[hi].set(title=f"Action Top5 @ {h:g}s", xlabel="Full encoder input (seconds)", xticks=LENGTHS)
        axes[hi].grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Fixed 4096 predictor context tokens · {len(records)} paired rows · {len(sessions)} sessions")
    fig.tight_layout(); fig.savefig(args.out / "length_accuracy.png", dpi=180); plt.close(fig)
    selected = [c for c in contrasts if c["horizon"] == 2 and c["metric"] == "top5"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, kind in zip(axes, ["adjacent_length", "matched_length_vs_recent", "versus_encoded4s"]):
        chosen = [c for c in selected if c["kind"] == kind]
        for i, c in enumerate(chosen):
            ci = c["ci95"]
            ax.plot(ci, [i, i], color="#333333")
            ax.scatter(c["delta"], i, color=colors[STRATEGIES.index(c["a"].split("__")[1])])
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.set(yticks=range(len(chosen)), yticklabels=[c["a"].replace("__", " ") + " − " + c["b"].replace("__", " ") for c in chosen],
               xlabel="Action Top5 @2s delta (pp); paired session CI95%", title=kind.replace("_", " "))
        ax.tick_params(axis="y", labelsize=7)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(args.out / "paired_deltas.png", dpi=180); plt.close(fig)
    print(json.dumps(dict(population=report["population"], gates=report["gates"], out=str(args.out))), flush=True)


if __name__ == "__main__":
    main()
