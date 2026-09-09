#!/usr/bin/env python3
"""Independent CPU validation and paired analysis of frozen B18 position policies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

PROTOCOL = "b18-predictor-prune/egtea-ctx16-position-lag-v1"
POLICIES = ["packed", "packed_shift48", "original"]
SELECTORS = ["recent", "hybrid_allquery_L11", "hybrid_target_L11"] + [f"hybrid_target_random_seed{s}" for s in (1701, 1702, 1703)]
ARMS = [f"{p}__{s}" for p in POLICIES for s in SELECTORS]
HORIZONS = ["2s", "4s", "6s"]
SEED = 2026090611


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def source_manifest(meta):
    manifest = Path(meta["manifest"])
    manifest_meta = read(manifest.with_suffix(".meta.json"))
    require(sha(manifest) == meta["manifest_sha256"] == manifest_meta["manifest_sha256"], "Manifest hash mismatch")
    with manifest.open() as f:
        entries = list(csv.DictReader(f))
    require(len(entries) == 4000, "Unexpected manifest size")
    for key in ("train_csv", "val_csv"):
        require(sha(meta["arguments"][key]) == meta[key + "_sha256"], f"Changed {key}")
    require(meta["val_csv_sha256"] == manifest_meta["source_csv_sha256"], "Manifest source CSV mismatch")
    with Path(meta["arguments"]["val_csv"]).open() as f:
        source = list(csv.DictReader(f))
    vocabulary = {}
    with Path(meta["arguments"]["train_csv"]).open() as f:
        for row in csv.DictReader(f):
            for v, n, m in zip(row["mtp_verbs"].split(","), row["mtp_nouns"].split(","), row["mtp_mask"].split(",")):
                pair = (int(v), int(n))
                if float(m) >= .5 and min(pair) >= 0 and pair not in vocabulary:
                    vocabulary[pair] = len(vocabulary)
    expected = []
    for e in entries:
        i = int(e["original_csv_row_index"]); row = source[i]
        require(hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == e["row_sha256"], "CSV row hash mismatch")
        expected.append({"sample_id": hashlib.sha256((row["video_id"] + "|" + row["frame_indices"]).encode()).hexdigest(),
                         "source_index": i, "selection_index": int(e["selection_index"]),
                         "video_id": row["video_id"], "participant_id": e["participant_id"]})
    ordered = sorted(expected, key=lambda r: (r["video_id"], r["selection_index"]))
    paired = []
    for i in range(2000):
        a, b = dict(ordered[i]), dict(ordered[i + 2000])
        a.update(donor_sample_id=b["sample_id"], pair_id=i)
        b.update(donor_sample_id=a["sample_id"], pair_id=i)
        paired.extend((a, b))
    return paired, source, vocabulary


def validate_masks(meta, expected, reference):
    path = Path(meta["mask_path"]); mask_meta = read(path.with_suffix(".meta.json"))
    require(mask_meta == meta["mask_metadata"], "Mask metadata differs from frozen snapshot")
    require(mask_meta["evaluation_protocol"] == PROTOCOL and mask_meta["selectors"] == SELECTORS, "Mask protocol/selectors mismatch")
    require(mask_meta["random_seeds"] == [1701, 1702, 1703], "Random seed drift")
    require(mask_meta["source_run"] == "17026550" and mask_meta["manifest_sha256"] == meta["manifest_sha256"], "Mask source mismatch")
    require(sha(path) == meta["mask_sha256"] == mask_meta["mask_sha256"], "Mask bytes changed")
    require(sha(mask_meta["source_score_path"]) == mask_meta["source_score_sha256"], "Archived selector score bytes changed")
    require(sha(mask_meta["boundary_tie_path"]) == mask_meta["boundary_tie_sha256"], "Tie artifact changed")
    require(sha(Path(meta["reference_dir"]) / "metadata.json") == mask_meta["source_metadata_sha256"], "Reference metadata changed")
    selector_sources = [p for p in meta["source_sha256"] if p.endswith("/eval_b18_q1_target_crossslot.py")]
    require(len(selector_sources) == 1 and mask_meta["selector_source_sha256"] == meta["source_sha256"][selector_sources[0]], "Mask selector source differs from evaluator source")
    masks = np.load(path, mmap_mode="r")
    require(masks.shape == (4000, 6, 4096) and masks.dtype == np.int32, "Mask shape/dtype mismatch")
    counts = np.zeros((4000, 6, 64), dtype=np.int16)
    for ri, row in enumerate(expected):
        for si, selector in enumerate(SELECTORS):
            indices = masks[ri, si]
            require((np.diff(indices) > 0).all() and indices[0] >= 0 and indices[-1] < 16384, "Invalid mask index ordering/range")
            count = np.bincount(indices // 256, minlength=64)
            counts[ri, si] = count
            require(count.tolist() == reference[row["sample_id"]]["arms"][selector]["keep_per_slot"], "Mask temporal counts differ from source")
            if si == 0:
                require(np.array_equal(indices, np.arange(48 * 256, 64 * 256)), "Recent mask not last16 slots")
            else:
                require(np.array_equal(indices[-3072:], np.arange(52 * 256, 64 * 256)) and (indices[:1024] < 52 * 256).all(), "Hybrid anchor/history mask mismatch")
        require(np.array_equal(counts[ri, 3:], np.broadcast_to(counts[ri, 2], (3, 64))), "Random counts not target-matched")
    ties = [json.loads(line) for line in Path(mask_meta["boundary_tie_path"]).read_text().splitlines()]
    require(len(ties) == 8000, "Missing cutoff tie records")
    seen = set()
    known = {x["sample_id"] for x in expected}
    for t in ties:
        identity = (t["sample_id"], t["selector"])
        require(identity not in seen and identity[0] in known and identity[1] in SELECTORS[1:3], "Invalid/duplicate tie identity")
        seen.add(identity)
        g, e, selected = t["strictly_greater"], t["equal_to_cutoff"], t["selected_at_cutoff"]
        require(0 <= g < 1024 and e >= 1 and selected == 1024 - g and selected <= e, "Invalid cutoff tie counts")
        require(t["cross_boundary_tie"] == (g + e > 1024), "Tie indicator inconsistent")
    tie_n = sum(t["cross_boundary_tie"] for t in ties)
    require(tie_n == mask_meta["boundary_tie_cases"], "Tie count summary mismatch")
    return masks, counts, {"cases": 8000, "cross_boundary_tie_cases": tie_n,
                           "by_selector": {s: sum(t["cross_boundary_tie"] for t in ties if t["selector"] == s) for s in SELECTORS[1:3]},
                           "scope": "Original GPU indices were not archived. Current reconstructed masks are frozen across all policies; source slot counts and seeds do not establish exact historical index identity at cutoff ties."}


def validate(runs, partial):
    runs = sorted(runs, key=lambda p: read(p / "metadata.json")["start"])
    metas = [read(p / "metadata.json") for p in runs]
    common = metas[0]
    for m in metas:
        require(m["evaluation_protocol"] == PROTOCOL and m["metric_scope"] == "native", "Evaluation identity mismatch")
        require(m["policies"] == POLICIES and m["selectors"] == SELECTORS and m["arms"] == ARMS, "Arm axes mismatch")
        for key in ("manifest_sha256", "train_csv_sha256", "val_csv_sha256", "checkpoint_paths", "checkpoint_file_identity", "reference_dir", "mask_sha256", "mask_metadata", "source_sha256", "target_starts", "action_logits_axes"):
            require(m[key] == common[key], f"Cross-shard {key} mismatch")
        require(m["num_steps"] == 1 and m["target_starts"] == [6144, 18432, 18432], "Unexpected target layout")
    for p, digest in common["source_sha256"].items():
        require(sha(p) == digest, f"Frozen source changed: {p}")
    require(common["checkpoint_paths"] == [common["arguments"][k] for k in ("checkpoint", "init_from_ckpt", "encoder_lora", "predictor_lora")], "Actual checkpoint args differ from metadata")
    for path, identity in common["checkpoint_file_identity"].items():
        stat = Path(path).stat()
        require(identity == {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}, "Checkpoint attributes changed")
    expected, source, vocabulary = source_manifest(common)
    refdir = Path(common["reference_dir"])
    reference = {r["sample_id"]: r for r in (json.loads(line) for line in (refdir / "predictions.jsonl").read_text().splitlines())}
    require(read(refdir / "metadata.json")["evaluation_protocol"] == "b18-predictor-prune/egtea-ctx16-target-crossslot-v1", "Reference protocol mismatch")
    reforder = read(refdir / "execution_order.json")
    require(len(reference) == len(reforder) == 4000 and all(all(r[k] == v for k, v in e.items()) for r, e in zip(reforder, expected)), "Reference execution order mismatch")
    masks, counts, ties = validate_masks(common, expected, reference)
    all_rows, correct_parts, loss_parts, valid_parts, positions = [], [], [], [], []
    numeric = {"logit_finite": True, "ce_reconstruction_max_abs": 0., "ambiguous_top5_cutoff_tie_cases": 0,
               "recent_original_shift48_exact_rows": 0, "historical_packed": {}, "fp32_smoke_probes": []}
    historical = {s: {h: {"n": 0, "top5_disagreements": 0, "ce_max_abs": 0.} for h in HORIZONS} for s in SELECTORS}
    shift_squares = np.zeros(3); packed_squares = np.zeros(3); shift_max = np.zeros(3); shift_elements = 0
    for path, meta in zip(runs, metas):
        summary = read(path / "summary.json"); order = read(path / "execution_order.json")
        rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
        start, stop = meta["start"], meta["stop"]; n = stop - start
        require(start % 4 == stop % 4 == 0 and len(rows) == len(order) == summary["n_rows"] == n, "Shard row count/range mismatch")
        require(summary["evaluation_protocol"] == PROTOCOL and summary["recent_original_shift48_exact_batches"] == n // 4, "Missing same-coordinate GPU parity gate")
        target = np.load(path / "target_positions.npy")
        require(np.array_equal(target, np.arange(256)[None, :] + np.array([6144, 18432, 18432])[:, None]), "Stored target coordinates mismatch")
        require(np.array_equal(np.load(path / "packed_context_positions.npy"), np.arange(4096)), "Packed coordinates mismatch")
        require(np.array_equal(np.load(path / "packed_shift48_context_positions.npy"), np.arange(4096) + 12288), "Shift48 coordinates mismatch")
        logits = np.load(path / "action_logits.npy", mmap_mode="r")
        require(logits.shape == (n, 3, 6, 3, len(vocabulary)) and logits.dtype == np.float32, "Formal logit layout/dtype mismatch")
        correct = np.zeros((n, 3, 18)); losses = np.zeros_like(correct); valid = np.zeros((n, 3), dtype=bool)
        for ri, (r, o) in enumerate(zip(rows, order)):
            index = start + ri
            require(all(r[k] == v for k, v in o.items()) and all(r[k] == v for k, v in expected[index].items()), "Prediction/execution/manifest identity mismatch")
            require(set(r["arms"]) == set(ARMS), "Missing/extra arm")
            logit = np.asarray(logits[ri], dtype=np.float64)
            require(np.isfinite(logit).all(), "Nonfinite formal logits")
            require(np.array_equal(logits[ri, 1, 0], logits[ri, 2, 0]), "Recent same-coordinate logits differ")
            numeric["recent_original_shift48_exact_rows"] += 1
            delta = logit[1, 0] - logit[0, 0]
            shift_squares += (delta ** 2).sum(-1); packed_squares += (logit[0, 0] ** 2).sum(-1)
            shift_max = np.maximum(shift_max, np.abs(delta).max(-1)); shift_elements += delta.shape[-1]
            src = source[r["source_index"]]
            for hi, h in enumerate(HORIZONS):
                pair = (int(src["mtp_verbs"].split(",")[hi]), int(src["mtp_nouns"].split(",")[hi]))
                state = "masked" if float(src["mtp_mask"].split(",")[hi]) <= .5 else "valid" if pair in vocabulary else "out_of_training_vocabulary"
                require(r["label_validity"][h] == state == reference[r["sample_id"]]["label_validity"][h], "Source validity mismatch")
                valid[ri, hi] = state == "valid"
            for pi, policy in enumerate(POLICIES):
                for si, selector in enumerate(SELECTORS):
                    arm = f"{policy}__{selector}"; item = r["arms"][arm]; ai = pi * 6 + si
                    require(item["keep_per_slot"] == counts[index, si].tolist(), "Different masks/counts across policy")
                    context = np.arange(4096) if pi == 0 else np.arange(4096) + 12288 if pi == 1 else masks[index, si]
                    require(item["context_position_first"] == int(context[0]) and item["context_position_last"] == int(context[-1]), "Context coordinate endpoints mismatch")
                    require(item["target_position_first"] == int(target[pi, 0]) and item["target_position_last"] == int(target[pi, -1]), "Target coordinate endpoints mismatch")
                    if pi > 0:
                        require(np.array_equal(context[-3072:], masks[index, si, -3072:]), "Shift/original recent12 anchor coordinates differ")
                    for hi, h in enumerate(HORIZONS):
                        metric = item["metrics"].get(h)
                        require((metric is not None) == valid[ri, hi], "Arm label exclusion mismatch")
                        if metric is None:
                            continue
                        label = vocabulary[(int(src["mtp_verbs"].split(",")[hi]), int(src["mtp_nouns"].split(",")[hi]))]
                        require(metric["label"] == label and isinstance(metric["top5"], bool), "Invalid label/top5")
                        v = logit[pi, si, hi]; maximum = v.max()
                        ce = float(np.log(np.exp(v - maximum).sum()) + maximum - v[label])
                        err = abs(ce - metric["ce"])
                        numeric["ce_reconstruction_max_abs"] = max(numeric["ce_reconstruction_max_abs"], err)
                        require(np.isfinite(metric["ce"]) and err <= 1e-5, "CE does not match saved logits")
                        greater, equal = int((v > v[label]).sum()), int((v == v[label]).sum())
                        if greater >= 5:
                            require(metric["top5"] is False, "Top5 impossible given label rank")
                        elif greater + equal <= 5:
                            require(metric["top5"] is True, "Top5 missing certain label")
                        else:
                            numeric["ambiguous_top5_cutoff_tie_cases"] += 1
                        correct[ri, hi, ai] = metric["top5"]; losses[ri, hi, ai] = metric["ce"]
                        if pi == 0:
                            old = reference[r["sample_id"]]["arms"][selector]["metrics"][h]
                            stats = historical[selector][h]; stats["n"] += 1
                            stats["top5_disagreements"] += metric["top5"] != old["top5"]
                            stats["ce_max_abs"] = max(stats["ce_max_abs"], abs(metric["ce"] - old["ce"]))
            require(r["arms"]["packed_shift48__recent"]["metrics"] == r["arms"]["original__recent"]["metrics"], "Same-coordinate recent metrics differ")
        for ai, arm in enumerate(ARMS):
            reported = summary["results"][arm]
            for hi, h in enumerate(HORIZONS):
                require(reported.get(f"n@{h}", 0) == valid[:, hi].sum() and reported.get(f"correct@{h}", 0) == correct[:, hi, ai].sum(), "GPU aggregate counts mismatch")
                require(np.isclose(reported.get(f"ce_sum@{h}", 0), losses[:, hi, ai].sum(), rtol=1e-7, atol=1e-5), "GPU aggregate CE mismatch")
        if meta["arguments"]["smoke_probes"]:
            probes = read(path / "smoke_probes.json")
            require(probes["same_device_packed_helper_exact"] and probes["original_full_helper_exact"] and all(probes["tf32_restored"]), "Missing same-device helper parity")
            probe_logits = np.load(path / "fp32_recent_probe_logits.npy")
            require(probe_logits.shape == (4, 2, 3, len(vocabulary)) and probe_logits.dtype == np.float32 and np.isfinite(probe_logits).all(), "Invalid FP32 smoke logits")
            delta = probe_logits[:, 0].astype(np.float64) - probe_logits[:, 1]
            require(np.isclose(np.abs(delta).max(), probes["fp32_logits_shift_difference"]["max_abs"], rtol=1e-5, atol=1e-6), "FP32 measured difference summary mismatch")
            ce_differences = {}
            for hi, h in enumerate(HORIZONS):
                deltas = []
                for ri in range(4):
                    metric = rows[ri]["arms"]["packed__recent"]["metrics"].get(h)
                    if metric is None:
                        continue
                    v = probe_logits[ri, :, hi].astype(np.float64); maximum = v.max(-1)
                    ce = np.log(np.exp(v - maximum[:, None]).sum(-1)) + maximum - v[:, metric["label"]]
                    deltas.append(float(ce[1] - ce[0]))
                ce_differences[h] = {"valid_rows": len(deltas), "per_row_shift_minus_packed_ce": deltas,
                                     "mean_shift_minus_packed_ce": float(np.mean(deltas)) if deltas else None}
            numeric["fp32_smoke_probes"].append({"job_id": meta["job_id"], **probes, "ce_shift_measurements": ce_differences})
        positions.extend(range(start, stop)); all_rows.extend(rows)
        correct_parts.append(correct); loss_parts.append(losses); valid_parts.append(valid)
    require(len(set(positions)) == len(all_rows) == len({r["sample_id"] for r in all_rows}), "Overlap/repeated sample")
    if not partial:
        require(positions == list(range(4000)), "Incomplete full coverage")
    numeric["historical_packed"] = historical
    numeric["recent_shift48_minus_packed_formal_logits"] = {h: {
        "max_abs": float(shift_max[hi]), "rms": float(np.sqrt(shift_squares[hi] / shift_elements)),
        "relative_l2_to_packed": float(np.sqrt(shift_squares[hi] / packed_squares[hi])),
        "n_logit_elements": shift_elements} for hi, h in enumerate(HORIZONS)}
    coverage = {"rows": len(all_rows), "manifest_rows": 4000, "complete": positions == list(range(4000)),
                "sessions": len({r["video_id"] for r in all_rows}), "participant_ids": len({r["participant_id"] for r in all_rows}),
                "ranges": [[m["start"], m["stop"]] for m in metas], "manifest_sha256": common["manifest_sha256"],
                "mask_sha256": common["mask_sha256"], "checkpoint_identity": "Path/byte-size/mtime_ns checked; no checkpoint content hash."}
    return metas, all_rows, np.concatenate(correct_parts), np.concatenate(loss_parts), np.concatenate(valid_parts), counts[positions], coverage, ties, numeric


def contrasts():
    result = {}
    for hi, lo in (("original", "packed"), ("packed_shift48", "packed"), ("original", "packed_shift48")):
        for comparator in ("random_mean3", "allquery"):
            weights = {f"{hi}__hybrid_target_L11": 1., f"{lo}__hybrid_target_L11": -1.}
            others = SELECTORS[3:] if comparator == "random_mean3" else [SELECTORS[1]]
            for s in others:
                weights[f"{hi}__{s}"] = -1 / len(others); weights[f"{lo}__{s}"] = 1 / len(others)
            result[f"interaction_{hi}_minus_{lo}__target_minus_{comparator}"] = weights
    for p in POLICIES:
        for name, others in (("random_mean3", SELECTORS[3:]), ("allquery", [SELECTORS[1]]), ("recent", [SELECTORS[0]])):
            result[f"{p}__target_minus_{name}"] = {f"{p}__hybrid_target_L11": 1., **{f"{p}__{s}": -1 / len(others) for s in others}}
    result["recent_common_shift48_minus_packed_secondary"] = {"packed_shift48__recent": 1., "packed__recent": -1.}
    return result


def bootstrap(rows, correct, losses, valid, reps):
    definitions = contrasts(); names = list(definitions)
    weights = np.array([[definitions[name].get(a, 0.) for a in ARMS] for name in names])
    values = np.stack([correct @ weights.T, losses @ weights.T], -1)
    point = values.sum(0) / valid.sum(0)[:, None, None]
    result = {name: {h: {"n": int(valid[:, hi].sum()), "top5_delta_pp": float(100 * point[hi, ci, 0]),
                           "ce_delta": float(point[hi, ci, 1])} for hi, h in enumerate(HORIZONS)} for ci, name in enumerate(names)}
    for key, offset in (("video_id", 0), ("participant_id", 1)):
        clusters = sorted({r[key] for r in rows}); index = {x: i for i, x in enumerate(clusters)}
        ids = np.array([index[r[key]] for r in rows]); g = len(clusters)
        sums = np.zeros((g,) + values.shape[1:]); denom = np.zeros((g, 3))
        np.add.at(sums, ids, values); np.add.at(denom, ids, valid)
        rng = np.random.default_rng(SEED + offset); draws = np.empty((reps,) + point.shape)
        for start in range(0, reps, 256):
            stop = min(start + 256, reps)
            multiplicity = rng.multinomial(g, np.full(g, 1 / g), size=stop - start)
            d = multiplicity @ denom; require((d > 0).all(), "Bootstrap empty valid horizon")
            draws[start:stop] = (multiplicity @ sums.reshape(g, -1)).reshape((stop - start,) + point.shape) / d[:, :, None, None]
        ci = np.quantile(draws, [.025, .975], axis=0)
        for ni, name in enumerate(names):
            for hi, h in enumerate(HORIZONS):
                result[name][h][key] = {"cluster_n": g, "top5_95ci_pp": (100 * ci[:, hi, ni, 0]).tolist(), "ce_95ci": ci[:, hi, ni, 1].tolist()}
    return {"primary": names[0], "primary_horizon": "2s", "seed": SEED, "reps": reps, "definitions": definitions, "contrasts": result,
            "method": "Paired session-cluster percentile bootstrap of ratio-of-sums; participant-cluster sensitivity. Shared draws across policies/selectors, random seeds averaged within sample.",
            "limits": ["One prospectively declared primary Top5 interaction; remaining contrasts, horizons and CE are secondary/exploratory.", "Marginal intervals do not provide simultaneous coverage across multiple reported comparisons.", "Same repeatedly inspected validation manifest; not independent confirmation.", "Original minus packed changes absolute origin and history coordinates under the checkpoint-compatible RoPE implementation. Shift48 controls separate these policy components; no translation invariance is assumed.", "Current frozen masks are reconstructed from historical scores; original exact indices were not archived."]}


def plots(out, results, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(6)
    for pi, p in enumerate(POLICIES):
        values = [100 * results[f"{p}__{s}"]["2s"]["action_top5"] for s in SELECTORS]
        ax.plot(x, values, marker="o", label=p)
    ax.set_xticks(x, ["recent", "all-query", "target-query", "random 1701", "random 1702", "random 1703"])
    ax.set(ylabel="Action Top-5 at +2 s (%)", title="Same frozen masks and encoder, three predictor coordinate policies")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "accuracy_position_policies.png", dpi=170); plt.close(fig)
    names = list(paired["contrasts"])[:6]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(names):
        m = paired["contrasts"][name]["2s"]; lo, hi = m["video_id"]["top5_95ci_pp"]
        ax.hlines(i, lo, hi, color="tab:blue")
        ax.plot(m["top5_delta_pp"], i, "o", color="tab:blue")
    ax.set_yticks(range(len(names)), [n.replace("interaction_", "").replace("__", ": ") for n in names])
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=1)
    ax.set(xlabel="Difference in selector contrast, percentage points", title="+2 s paired session-cluster 95% intervals\nFirst row primary; remaining rows secondary, marginal intervals")
    fig.tight_layout(); fig.savefig(out / "position_interactions_session_ci.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=9999)
    args = parser.parse_args()
    metas, rows, correct, losses, valid, counts, coverage, ties, numeric = validate(args.runs, args.allow_partial)
    require((valid.sum(0) > 0).all(), "No valid horizon")
    results = {a: {h: {"n": int(valid[:, hi].sum()), "correct": int(correct[:, hi, ai].sum()),
                       "action_top5": float(correct[:, hi, ai].sum() / valid[:, hi].sum()),
                       "mean_ce": float(losses[:, hi, ai].sum() / valid[:, hi].sum())} for hi, h in enumerate(HORIZONS)} for ai, a in enumerate(ARMS)}
    paired = bootstrap(rows, correct, losses, valid, args.bootstrap_reps)
    random = {p: {h: {"per_seed_action_top5": [results[f"{p}__{s}"][h]["action_top5"] for s in SELECTORS[3:]],
                       "mean_action_top5": float(np.mean([results[f"{p}__{s}"][h]["action_top5"] for s in SELECTORS[3:]])),
                       "seed_sd_pp": float(100 * np.std([results[f"{p}__{s}"][h]["action_top5"] for s in SELECTORS[3:]], ddof=1))}
                   for h in HORIZONS} for p in POLICIES}
    result = {"evaluation_protocol": PROTOCOL, "metric_scope": "native", "eval_path": metas[0]["eval_path"],
              "analysis_job_id": os.environ.get("SLURM_JOB_ID"), "input_jobs": [m["job_id"] for m in metas],
              "input_dirs": [str(p) for p in args.runs], "analysis_source_sha256": sha(__file__),
              "smoke_only": args.allow_partial, "coverage": coverage, "gates": "passed", "mask_reconstruction": ties,
              "numeric_checks": numeric, "results": results, "paired_inference": paired, "random_seed_summary": random,
              "exclusions": {h: dict(Counter(r["label_validity"][h] for r in rows)) for h in HORIZONS},
              "selection": {s: {"mean_keep_per_slot": counts[:, si].mean(0).tolist()} for si, s in enumerate(SELECTORS)},
              "precision": "Formal GPU forward used BF16 autocast; action logits were cast/stored float32. CPU CE is checked with float64 log-sum-exp. FP32 no-TF32 translation probes are separate measurements, not equivalence gates.",
              "top5_validation": "CUDA Top5 outcomes are checked against strict rank bounds from saved logits; cutoff ties are counted rather than resolved with potentially different CPU tie-breaking."}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "merged_execution_order.json").write_text(json.dumps([{k: r[k] for k in ("sample_id", "video_id", "participant_id", "source_index", "selection_index", "pair_id")} for r in rows], indent=2) + "\n")
    np.savez_compressed(args.out_dir / "paired_outcomes.npz", correct=correct, ce=losses, valid=valid, arms=np.array(ARMS))
    plots(args.out_dir, results, paired)
    brief = {"analysis_job_id": result["analysis_job_id"], "smoke_only": args.allow_partial, "coverage": coverage,
             "gates": "passed", "mask_reconstruction": ties, "numeric_checks": numeric,
             "results": results, "primary_2s": paired["contrasts"][paired["primary"]]["2s"]}
    (args.out_dir / "brief.json").write_text(json.dumps(brief, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"analysis_job_id": result["analysis_job_id"], "gates": "passed", "smoke_only": args.allow_partial, "coverage": coverage, "output": str(args.out_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
