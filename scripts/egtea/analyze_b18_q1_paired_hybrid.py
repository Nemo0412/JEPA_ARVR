#!/usr/bin/env python3
"""Validate and analyze B18 paired hybrid shards in a Slurm CPU container.

Bootstrap units are receiver sessions, with participant sensitivity. All arm
outcomes from one receiver row remain paired. Donor intervals condition on the
fixed donor assignment and do not capture dependence induced by shared donors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROTOCOL = "b18-predictor-prune/egtea-ctx16-paired-hybrid-v1"
HORIZONS = ["2s", "4s", "6s"]
RANDOM = [f"hybrid_random_matched_L11_seed{x}" for x in (1701, 1702, 1703)]
MAIN = "hybrid_online_L11"
CONTRASTS = {
    "hybrid_L11_minus_recent": {MAIN: 1, "recent": -1},
    "hybrid_L11_minus_random_mean3": {MAIN: 1, **{x: -1 / 3 for x in RANDOM}},
    "hybrid_L11_minus_donor": {MAIN: 1, "hybrid_donor_matched_L11": -1},
    "depth_by_online_offline_interaction": {"online_L11": 1, "online_L0": -1,
                                             "offline_L11": -1, "offline_L0": 1},
    "online_L11_minus_L0": {"online_L11": 1, "online_L0": -1},
    "offline_L11_minus_L0": {"offline_L11": 1, "offline_L0": -1},
    "hybrid_online_L11_minus_L0": {"hybrid_online_L11": 1, "hybrid_online_L0": -1},
    "hybrid_offline_L11_minus_L0": {"hybrid_offline_L11": 1, "hybrid_offline_L0": -1},
    **{f"hybrid_L11_minus_random_seed{x}": {MAIN: 1, f"hybrid_random_matched_L11_seed{x}": -1}
       for x in (1701, 1702, 1703)},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalized(x):
    denominator = x.sum(-1, keepdims=True)
    require(bool((denominator > 0).all()), "Zero attention normalizer")
    return x / denominator


def entropy(x):
    p = normalized(x)
    return -(p * np.log(np.maximum(p, 1e-30))).sum(-1) / np.log(p.shape[-1])


def concentration(x):
    p = normalized(x)
    return {"entropy_div_log64": float(entropy(p).mean()),
            "sum_squared_probabilities": float(np.square(p).sum(-1).mean()),
            "max_slot_probability": float(p.max(-1).mean()),
            "recent8_fraction": float(p[..., -8:].sum(-1).mean()),
            "recent16_fraction": float(p[..., -16:].sum(-1).mean()),
            "recent32_fraction": float(p[..., -32:].sum(-1).mean())}


def pooling_decomposition(context_mass):
    """Weighted Jensen gaps for raw sample/head pooling, all in H/log(64) units."""
    raw = np.asarray(context_mass, dtype=np.float64)  # sample,head,context_slot
    head_profiles = normalized(raw)
    head_weights = normalized(raw.sum(-1))
    sample_raw = raw.sum(1)
    sample_profiles = normalized(sample_raw)
    sample_weights = sample_raw.sum(-1) / sample_raw.sum()
    pooled_raw = normalized(sample_raw.sum(0))
    sample_entropy = entropy(sample_profiles)
    weighted_sample_entropy = float(sample_weights @ sample_entropy)
    within_sample_head_entropy = (head_weights * entropy(head_profiles)).sum(1)
    head_gap = sample_entropy - within_sample_head_entropy
    sample_gap = float(entropy(pooled_raw) - weighted_sample_entropy)
    return {
        "units": "All entropy and Jensen-Shannon gaps divided by log(64). Uniform profile entropy is 1.",
        "actual_raw_pool_profile": pooled_raw.tolist(),
        "equal_sample_profile": sample_profiles.mean(0).tolist(),
        "actual_raw_pool_concentration": concentration(pooled_raw),
        "equal_sample_pool_concentration": concentration(sample_profiles.mean(0)),
        "sample_entropy_unweighted_mean": float(sample_entropy.mean()),
        "sample_entropy_raw_mass_weighted_mean": weighted_sample_entropy,
        "sample_pooling_js_raw_mass_weighted": sample_gap,
        "sample_pooling_js_equal_sample": float(entropy(sample_profiles.mean(0)) - sample_entropy.mean()),
        "head_pooling_js_raw_mass_weighted_mean": float(sample_weights @ head_gap),
        "head_pooling_js_raw_mass_weighted_per_sample_mean": float(head_gap.mean()),
        "head_pooling_js_equal_head_per_sample_mean": float((entropy(head_profiles.mean(1))-entropy(head_profiles).mean(1)).mean()),
        "total_sample_head_pooling_js_raw_mass_weighted": float(entropy(pooled_raw) - sample_weights @ within_sample_head_entropy),
        "sample_context_mass_weight_relative_to_uniform": distribution(sample_weights * len(sample_weights)),
        "mean_sample_L1_distance_to_actual_raw_pool": float(np.abs(sample_profiles-pooled_raw).sum(-1).mean()),
        "weighting": "Actual pooling normalizes summed raw scores after pooling; samples and heads are weighted by their received context-key mass. Equal-sample/head alternatives normalize first and are distinct descriptive summaries.",
    }


def distribution(values):
    x = np.asarray(values)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "p05": float(np.quantile(x, .05)), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95))}


def load_shards(paths, allow_partial):
    paths = sorted(paths, key=lambda p: json.loads((p / "metadata.json").read_text())["start"])
    metadatas, summaries, records, group_masses, flow_parts, score_parts = [], [], [], [], [], []
    for path in paths:
        for file in ("metadata.json", "summary.json", "execution_order.json", "predictions.jsonl",
                     "attention_profiles.npz", "scores.npy"):
            require((path / file).is_file(), f"Missing completed artifact {path / file}")
        metadata = json.loads((path / "metadata.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        order = json.loads((path / "execution_order.json").read_text())
        lines = [json.loads(x) for x in (path / "predictions.jsonl").read_text().splitlines()]
        n = metadata["stop"] - metadata["start"]
        require(n == len(lines) == len(order) == summary["n_rows"], f"Shard count mismatch: {path}")
        require(metadata["start"] % 2 == metadata["stop"] % 2 == 0, "Shard split donor pair")
        require(summary["evaluation_protocol"] == metadata["evaluation_protocol"] == PROTOCOL,
                "Protocol mismatch")
        require(summary["score_block_order"] == [0, 11] and summary["query_group_order"] == ["context", "target"],
                "Unexpected attention block/group ordering")
        for i, (entry, line) in enumerate(zip(order, lines)):
            for key in ("sample_id", "video_id", "participant_id", "source_index", "selection_index",
                        "donor_sample_id", "pair_id"):
                require(entry[key] == line[key], f"Execution/prediction identity mismatch {path}:{i}:{key}")
        profiles = np.load(path / "attention_profiles.npz")
        mass = profiles["group_mass"]
        require(mass.ndim == 5 and mass.shape[:3] == (n, 2, 2) and mass.shape[-1] == 65,
                f"Unexpected group_mass layout {mass.shape}; require 64 context slots + one target block")
        require(np.isfinite(mass).all() and (mass >= 0).all(), "Invalid attention mass")
        if "token_layout" in metadata:
            layout = metadata["token_layout"]
            for key, value in {"n_context_tokens":16384, "n_target_tokens":256, "tokens_per_slot":256,
                               "target_rope_start_slot":72, "target_array_start_slot":64}.items():
                require(layout[key] == value, f"Unexpected token layout: {key}")
            require(all(layout["num_heads"][str(b)] == mass.shape[3] for b in (0,11)), "Head-count mismatch")
        scores = np.load(path / "scores.npy", mmap_mode="r")
        require(scores.shape == (n, 2, 16384) and scores.dtype == np.float32, "Score layout mismatch")
        require(np.isfinite(scores).all() and (scores >= 0).all(), "Invalid scores")
        score_temporal = scores.reshape(n, 2, 64, 256).sum(-1, dtype=np.float64)
        profile_temporal = mass[..., :64].sum((2, 3), dtype=np.float64)
        require(np.allclose(score_temporal, profile_temporal, rtol=.002, atol=.02),
                "Scores/profiles row or block alignment mismatch")
        query_sums = mass.sum(-1, dtype=np.float64)
        require(np.allclose(query_sums[:, :, 0], 16384, rtol=.01), "Context query mass is not 16384/head")
        require(np.allclose(query_sums[:, :, 1], 256, rtol=.01), "Target query mass is not 256/head")
        flow = [profiles[f"mean_queryslot_keyslot_L{b}"] for b in (0, 11)]
        for f in flow:
            require(f.shape == (mass.shape[3], 65, 65), "Unexpected flow layout")
            require(np.isfinite(f).all() and (f >= 0).all(), "Invalid flow")
            require(np.allclose(f.sum(-1), 1, rtol=.01), "Flow query rows not normalized")
        metadatas.append(metadata); summaries.append(summary); records.extend(lines)
        group_masses.append(mass); flow_parts.append((n, flow)); score_parts.append(score_temporal)
    common = metadatas[0]
    common_keys = ("evaluation_protocol", "manifest", "manifest_sha256", "arms", "primary_horizon",
                   "metric_scope", "eval_path", "source_sha256", "train_csv_sha256", "calibration_sha256",
                   "checkpoint_paths", "checkpoint_file_identity", "selection")
    for metadata in metadatas[1:]:
        for key in common_keys:
            require(metadata[key] == common[key], f"Cross-shard mismatch: {key}")
        for key in ("train_csv", "val_csv", "seed", "n_samples"):
            require(metadata["arguments"][key] == common["arguments"][key], f"Argument mismatch: {key}")
        if "token_layout" in common or "token_layout" in metadata:
            require(common.get("token_layout") == metadata.get("token_layout"), "Token layout mismatch")
    require(sha(common["arguments"]["train_csv"]) == common["train_csv_sha256"], "Training vocabulary CSV changed")
    vocabulary = {}
    with Path(common["arguments"]["train_csv"]).open() as handle:
        for train_row in csv.DictReader(handle):
            for v, n, m in zip(train_row["mtp_verbs"].split(","), train_row["mtp_nouns"].split(","), train_row["mtp_mask"].split(",")):
                pair = (int(v), int(n))
                if float(m) >= .5 and pair[0] >= 0 and pair[1] >= 0 and pair not in vocabulary:
                    vocabulary[pair] = len(vocabulary)
    manifest_path = Path(common["manifest"])
    require(sha(manifest_path) == common["manifest_sha256"], "Manifest bytes changed")
    manifest_meta = json.loads(manifest_path.with_suffix(".meta.json").read_text())
    require(manifest_meta["evaluation_protocol"] == PROTOCOL, "Manifest protocol mismatch")
    source_path = Path(manifest_meta["source_csv"])
    require(sha(source_path) == manifest_meta["source_csv_sha256"], "Source CSV bytes changed")
    with manifest_path.open() as handle:
        manifest_rows = list(csv.DictReader(handle))
    with source_path.open() as handle:
        source_rows = list(csv.DictReader(handle))
    identities = []
    for row in manifest_rows:
        source = source_rows[int(row["original_csv_row_index"])]
        require(hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                == row["row_sha256"], "Manifest row hash mismatch")
        identities.append({"sample_id": hashlib.sha256((source["video_id"] + "|" + source["frame_indices"]).encode()).hexdigest(),
                           "selection_index": int(row["selection_index"]), "video_id": row["video_id"],
                           "participant_id": row["participant_id"], "source_index": int(row["original_csv_row_index"])})
    ordered = sorted(identities, key=lambda s: (s["video_id"], s["selection_index"]))
    half = len(ordered) // 2
    expected = []
    for i in range(half):
        a, b = dict(ordered[i]), dict(ordered[i + half])
        a.update(donor_sample_id=b["sample_id"], pair_id=i)
        b.update(donor_sample_id=a["sample_id"], pair_id=i)
        expected.extend((a, b))
    expected_positions = [j for m in metadatas for j in range(m["start"], m["stop"])]
    require(len(expected_positions) == len(set(expected_positions)), "Overlapping shard ranges")
    require(len({x["sample_id"] for x in records}) == len(records), "Duplicate sample rows")
    if not allow_partial:
        require(expected_positions == list(range(len(expected))), "Missing full manifest coverage")
    for i, record in zip(expected_positions, records):
        for key, value in expected[i].items():
            require(record[key] == value, f"Unexpected donor/manifest identity at execution row {i}: {key}")
        source = source_rows[record["source_index"]]
        for hi, horizon in enumerate(HORIZONS):
            pair = (int(source["mtp_verbs"].split(",")[hi]), int(source["mtp_nouns"].split(",")[hi]))
            state = "masked" if float(source["mtp_mask"].split(",")[hi]) <= .5 else (
                "valid" if pair in vocabulary else "out_of_training_vocabulary")
            require(record["label_validity"][horizon] == state, "CSV prediction label-validity mismatch")
            if state == "valid":
                require(all(record["arms"][arm]["metrics"][horizon]["label"] == vocabulary[pair]
                            for arm in common["arms"]), "Prediction label differs from source CSV vocabulary")
    for i in range(0, len(records), 2):
        a, b = records[i:i + 2]
        require(a["donor_sample_id"] == b["sample_id"] and b["donor_sample_id"] == a["sample_id"], "Broken donor pair")
        require(a["video_id"] != b["video_id"] and a["pair_id"] == b["pair_id"], "Invalid donor pair")
    for path, digest in common["calibration_sha256"].items():
        require(sha(path) == digest, f"Calibration map changed: {path}")
    checkpoint_evidence = {}
    for path in common["checkpoint_paths"]:
        stat = Path(path).stat()
        require(common["checkpoint_file_identity"][path] == {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                f"Checkpoint file identity changed: {path}")
        checkpoint_evidence[path] = {"size_bytes_at_analysis": stat.st_size,
                                     "mtime_ns_at_analysis": stat.st_mtime_ns}
    total = len(records)
    flows = [sum(n * flow[bi].astype(np.float64) for n, flow in flow_parts) / total for bi in range(2)]
    coverage = {"n_rows": total, "full_manifest_rows": len(expected), "complete": total == len(expected),
                "n_sessions": len({x["video_id"] for x in records}),
                "n_participant_identifiers": len({x["participant_id"] for x in records}),
                "n_donor_pairs": total // 2, "ranges": [[m["start"], m["stop"]] for m in metadatas],
                "manifest_sha256": common["manifest_sha256"], "source_csv_sha256": manifest_meta["source_csv_sha256"],
                "checkpoint_file_evidence": checkpoint_evidence,
                "checkpoint_identity_caveat": "Paths, byte sizes and mtime_ns agree at run and analysis. Checkpoint file content hashes were not recorded, so identity is based on these file attributes.",
                "source_hashes_at_run": common["source_sha256"]}
    return common, metadatas, summaries, records, np.concatenate(group_masses), flows, np.concatenate(score_parts), coverage


def outcomes(records, arms, shard_summaries, metadata):
    n, a = len(records), len(arms)
    correct = np.zeros((n, 3, a), dtype=np.float64)
    loss = np.zeros_like(correct)
    valid = np.zeros((n, 3), dtype=bool)
    keep = np.zeros((n, a, 64), dtype=np.int32)
    labels = np.full((n, 3), -1, dtype=np.int32)
    exclusion = {h: Counter() for h in HORIZONS}
    for i, record in enumerate(records):
        require(set(record["arms"]) == set(arms), "Arm coverage mismatch")
        for ai, arm in enumerate(arms):
            item = record["arms"][arm]
            counts = np.array(item["keep_per_slot"])
            require(counts.shape == (64,) and np.issubdtype(counts.dtype, np.integer), "Invalid keep count array")
            require(counts.sum() == 4096 and (counts >= 0).all() and (counts <= 256).all(), "Invalid token budget")
            keep[i, ai] = counts
            if arm == "recent":
                require((counts[:48] == 0).all() and (counts[48:] == 256).all(), "Recent mask mismatch")
            if arm.startswith("hybrid"):
                require((counts[52:] == 256).all() and counts[:52].sum() == 1024, "Hybrid anchor/budget mismatch")
            for hi, horizon in enumerate(HORIZONS):
                state = record["label_validity"][horizon]
                metric = item["metrics"].get(horizon)
                require((metric is not None) == (state == "valid"), "Arm label validity mismatch")
                if ai == 0:
                    valid[i, hi] = state == "valid"
                    exclusion[horizon][state] += 1
                    if metric is not None:
                        labels[i, hi] = metric["label"]
                if metric is not None:
                    require(metric["label"] == labels[i, hi], "Arm target label mismatch")
                    require(isinstance(metric["top5"], bool) and np.isfinite(metric["ce"]) and metric["ce"] >= 0,
                            "Invalid classifier outcome")
                    correct[i, hi, ai] = metric["top5"]
                    loss[i, hi, ai] = metric["ce"]
        main = keep[i, arms.index(MAIN)]
        for arm in RANDOM + ["hybrid_donor_matched_L11"]:
            require(np.array_equal(keep[i, arms.index(arm)], main), "Matched-control temporal count mismatch")
    cursor = 0
    for meta, summary in zip(metadata, shard_summaries):
        count = meta["stop"] - meta["start"]
        for ai, arm in enumerate(arms):
            for hi, horizon in enumerate(HORIZONS):
                res = summary["results"][arm]
                require(res.get(f"n@{horizon}", 0) == valid[cursor:cursor + count, hi].sum(), "Summary denominator mismatch")
                require(res.get(f"correct@{horizon}", 0) == correct[cursor:cursor + count, hi, ai].sum(), "Summary accuracy mismatch")
                require(np.isclose(res.get(f"ce_sum@{horizon}", 0), loss[cursor:cursor + count, hi, ai].sum(), atol=1e-5), "Summary CE mismatch")
        cursor += count
    result = {}
    for ai, arm in enumerate(arms):
        result[arm] = {h: {"n": int(valid[:, hi].sum()), "correct": int(correct[:, hi, ai].sum()),
                           "action_top5": float(correct[:, hi, ai].sum() / max(1, valid[:, hi].sum())),
                           "mean_ce": float(loss[:, hi, ai].sum() / max(1, valid[:, hi].sum()))}
                       for hi, h in enumerate(HORIZONS)}
    return correct, loss, valid, keep, result, {h: dict(x) for h, x in exclusion.items()}


def bootstrap(records, arms, correct, loss, valid, reps, seed):
    names = list(CONTRASTS)
    weights = np.array([[CONTRASTS[name].get(arm, 0) for arm in arms] for name in names])
    values = np.stack([correct @ weights.T, loss @ weights.T], -1)  # row,horizon,contrast,metric
    points = values.sum(0) / valid.sum(0)[:, None, None]
    answer = {name: {h: {"top5_delta_pp": float(points[hi, ci, 0] * 100),
                         "ce_delta": float(points[hi, ci, 1]), "n": int(valid[:, hi].sum())}
                     for hi, h in enumerate(HORIZONS)} for ci, name in enumerate(names)}
    cluster_details = {}
    for cluster_key in ("video_id", "participant_id"):
        clusters = sorted({r[cluster_key] for r in records})
        index = {name: i for i, name in enumerate(clusters)}
        membership = np.array([index[r[cluster_key]] for r in records])
        sums = np.zeros((len(clusters),) + values.shape[1:])
        denom = np.zeros((len(clusters), 3))
        np.add.at(sums, membership, values)
        np.add.at(denom, membership, valid)
        rng = np.random.default_rng(seed + (1 if cluster_key == "participant_id" else 0))
        samples = np.empty((reps,) + points.shape)
        for start in range(0, reps, 256):
            stop = min(reps, start + 256)
            draws = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=stop-start)
            numerator = (draws @ sums.reshape(len(clusters), -1)).reshape((stop-start,) + points.shape)
            denominator = draws @ denom
            require((denominator > 0).all(), "Bootstrap replicate has no valid horizon rows")
            samples[start:stop] = numerator / denominator[:, :, None, None]
        ci = np.quantile(samples, [.025, .975], axis=0)
        for ci_index, name in enumerate(names):
            for hi, horizon in enumerate(HORIZONS):
                answer[name][horizon][cluster_key] = {
                    "cluster_n": len(clusters), "bootstrap_reps": reps,
                    "top5_delta_pp_95ci": (100 * ci[:, hi, ci_index, 0]).tolist(),
                    "ce_delta_95ci": ci[:, hi, ci_index, 1].tolist(),
                    "top5_bootstrap_se_pp": float(samples[:, hi, ci_index, 0].std(ddof=1) * 100),
                }
        cluster_details[cluster_key] = {"cluster_n": len(clusters), "row_counts": dict(Counter(r[cluster_key] for r in records))}
    return {"contrasts": answer, "cluster_details": cluster_details,
            "method": "Paired cluster nonparametric percentile bootstrap, ratio of resampled outcome sums to resampled valid-row counts; same draws across every arm and contrast.",
            "seed": seed, "repetitions": reps, "confidence_level": .95,
            "primary": list(CONTRASTS)[:4], "primary_horizon": "2s",
            "multiplicity": "Intervals are marginal, not simultaneous; four primary contrasts and secondary horizons/depth checks require joint interpretation.",
            "donor_limitations": "Receiver-video and participant intervals for donor contrast condition on the one fixed cross-video donor pairing. They omit sampling uncertainty of donor reassignment and dependence linking one video's receiver outcomes to other receivers that use it as donor. They are conditional descriptive intervals, not a full dependency-aware donor test.",
            "calibration_limitations": "Offline maps fixed across all bootstrap draws. Validation bootstrap does not include calibration sample/map estimation variability.",
            "random_limitations": "Three fixed draws per receiver are averaged within receiver before bootstrap. They are not independent validation rows; three-seed SD is descriptive, not a confidence interval for arbitrary future random seeds."}


def keep_analysis(keep, arms):
    out = {}
    for ai, arm in enumerate(arms):
        counts = keep[:, ai]
        out[arm] = {"mean_keep_per_slot": counts.mean(0).tolist(),
                    **{f"{name}_count": distribution(counts[:, sl].sum(1))
                       for name, sl in {"recent8": slice(56,64), "recent16": slice(48,64),
                                        "recent32": slice(32,64), "older_than_recent16": slice(0,48),
                                        "displaced_recent4": slice(48,52), "fixed_recent12": slice(52,64)}.items()},
                    "whole_slots_per_sample": distribution((counts == 256).sum(1)),
                    "historical_count_definition": "0–47 is outside pure recent's 16 slots. 48–51 is recent material competing for the hybrid's 1024 flexible tokens. 52–63 is the fixed 12-slot anchor."}
        if "offline" in arm:
            require(np.equal(counts, counts[0]).all(), f"Offline mask changes between samples: {arm}")
    return out


def attention_analysis(mass, flows, temporal_scores):
    out = {"layout": {"context_key_slots": [0,63], "target_key_array_index": 64,
                      "context_query_count": 16384, "target_query_count": 256,
                      "gp": 256, "heads": mass.shape[3],
                      "caveat": "Array index 64 is the appended target block, not context time slot 64. Its actual RoPE target position is separated by the anticipation gap."}}
    for bi, block in enumerate((0, 11)):
        raw = mass[:, bi].astype(np.float64)  # sample,group,head,key
        result = {}
        for gi, group in enumerate(("context_queries", "target_queries")):
            profile = raw[:, gi]
            per_head_context = normalized(profile[..., :64])
            per_sample_head_pool = normalized(profile[..., :64].sum(1))
            pooled = normalized(profile[..., :64].sum((0,1)))
            allkey_per_query = profile / (16384 if gi == 0 else 256)
            result[group] = {
                "per_sample_per_head_context_conditional": concentration(per_head_context),
                "per_sample_head_pooled_context_conditional": concentration(per_sample_head_pool),
                "pooled_context_conditional": concentration(pooled),
                "pooled_context_profile": pooled.tolist(),
                "sample_average_of_normalized_profiles": per_sample_head_pool.mean(0).tolist(),
                "entropy_sample_pooling_gap": float(entropy(per_sample_head_pool.mean(0)) - entropy(per_sample_head_pool).mean()),
                "entropy_head_pooling_gap_mean": float((entropy(per_head_context.mean(1)) - entropy(per_head_context).mean(1)).mean()),
                "absolute_probability_context_keys_per_query": distribution(allkey_per_query[..., :64].sum(-1)),
                "absolute_probability_target_keys_per_query": distribution(allkey_per_query[..., 64:].sum(-1)),
                "normalization": "Context-conditional profiles renormalize within 64 context key slots; absolute masses divide raw received mass by the group's query count and retain target-key allocation.",
                "pooling_decomposition": pooling_decomposition(profile[..., :64]),
            }
        allquery = raw.sum(1)  # sample,head,key
        contextcontrib = raw[:, 0, :, :64].sum((1,2))
        targetcontrib = raw[:, 1, :, :64].sum((1,2))
        profile = normalized(allquery[..., :64].sum(1))
        result["legacy_all_query_head_sum"] = {
            "per_sample_context_conditional": concentration(profile),
            "actual_raw_pooled_context_conditional": concentration(allquery[..., :64].sum((0,1))),
            "equal_sample_pooled_context_conditional": concentration(profile.mean(0)),
            "pooled_raw_mass_profile": normalized(temporal_scores[:,bi].sum(0)).tolist(),
            "entropy_sample_pooling_gap": float(entropy(profile.mean(0))-entropy(profile).mean()),
            "target_query_share_of_context_received_mass": distribution(targetcontrib / (contextcontrib+targetcontrib)),
            "target_query_fraction_before_key_conditioning": 256 / (16384 + 256),
            "pooling_decomposition": pooling_decomposition(allquery[..., :64]),
            "interpretation_limit": "Near-uniform marginal received mass does not imply uniform attention per query; head/query-group/sample pooling can hide selectivity."}
        flow = flows[bi]
        context_flow = flow[:, :64, :64]
        conditional = normalized(context_flow)
        q, k = np.indices((64,64))
        result["queryslot_keyslot_flow"] = {
            "context_conditional_same_slot_probability": float((conditional * (q==k)).sum(-1).mean()),
            "context_conditional_within_one_slot_probability": float((conditional * (abs(q-k)<=1)).sum(-1).mean()),
            "context_conditional_within_four_slots_probability": float((conditional * (abs(q-k)<=4)).sum(-1).mean()),
            "context_query_absolute_target_key_probability": float(flow[:, :64, 64:].sum(-1).mean()),
            "target_query_context_conditional_profile": normalized(flow[:, 64:, :64].sum((0,1))).tolist(),
            "mean_context_flow_head_pooled": conditional.mean(0).tolist(),
            "caveat": "Flow is already averaged across examples and across 256 queries per slot. It cannot recover per-query/per-example attention entropy or attribute individual query causality."}
        out[f"L{block}"] = result
    return out


def score_map_analysis(shards, metadata):
    """Joint spatial/temporal pooling diagnostics; validation means never select tokens."""
    dimension = 16384
    arrays = [np.load(path / "scores.npy", mmap_mode="r") for path in shards]
    totals = np.zeros((2, dimension), dtype=np.float64)
    normalized_sums = np.zeros_like(totals)
    row_mass, row_entropy, spatial_conditional_entropy = [], [], []
    for scores in arrays:
        for start in range(0, len(scores), 32):
            chunk = np.array(scores[start:start+32], dtype=np.float64)
            total = chunk.sum(-1)
            p = chunk / total[..., None]
            hnats = -(p * np.log(np.maximum(p, 1e-30))).sum(-1)
            tp = p.reshape(-1, 2, 64, 256).sum(-1)
            temporal_hnats = -(tp * np.log(np.maximum(tp, 1e-30))).sum(-1)
            totals += chunk.sum(0)
            normalized_sums += p.sum(0)
            row_mass.append(total)
            row_entropy.append(hnats / np.log(dimension))
            spatial_conditional_entropy.append((hnats-temporal_hnats) / np.log(256))
    masses = np.concatenate(row_mass)
    entropies = np.concatenate(row_entropy)
    spatial_entropies = np.concatenate(spatial_conditional_entropy)
    refs = {"validation_raw_mean":totals / len(masses),
            "training_calibration":np.stack([np.load(metadata["arguments"][key]).reshape(-1)
                                              for key in ("calib_l0","calib_l11")]).astype(np.float64)}
    correlations = {name:{"cosine":[],"pearson":[]} for name in refs}
    for scores in arrays:
        for start in range(0, len(scores), 32):
            x = np.array(scores[start:start+32], dtype=np.float64)
            square = np.square(x).sum(-1)
            centered_norm = np.sqrt(np.maximum(square-np.square(x.sum(-1))/dimension,0))
            for name, ref in refs.items():
                centered_ref = ref-ref.mean(-1,keepdims=True)
                norm_ref = np.sqrt(np.square(ref).sum(-1))
                centered_ref_norm = np.sqrt(np.square(centered_ref).sum(-1))
                require((centered_norm>0).all() and (centered_ref_norm>0).all(),"Constant score map cannot define Pearson correlation")
                correlations[name]["cosine"].append((x*ref).sum(-1)/(np.sqrt(square)*norm_ref))
                correlations[name]["pearson"].append((x*centered_ref).sum(-1)/(centered_norm*centered_ref_norm))
    correlations = {k:{m:np.concatenate(v) for m,v in values.items()} for k,values in correlations.items()}
    result = {"scope":"Joint 64 temporal slots × 256 spatial positions, received scores summed over all queries and heads.",
              "warning":"Temporal flatness alone does not imply spatial sample-specific information survives pooling. Validation raw mean is diagnostic only; it is never fitted/evaluated as a new pruning arm.",
              "sample_count":len(masses)}
    for bi,block in enumerate((0,11)):
        actual_pool = normalized(totals[bi])
        equal_pool = normalized(normalized_sums[bi])
        weights = masses[:,bi]/masses[:,bi].sum()
        raw_gap = float(entropy(actual_pool)-weights@entropies[:,bi])
        va,ca = refs["validation_raw_mean"][bi],refs["training_calibration"][bi]
        result[f"L{block}"]={
            "entropy_units":"Joint entropy and Jensen gaps divided by log(16384); conditional spatial entropy divided by log(256).",
            "per_sample_joint_entropy":distribution(entropies[:,bi]),
            "per_sample_conditional_spatial_entropy_given_time":distribution(spatial_entropies[:,bi]),
            "actual_raw_pooled_joint_entropy":float(entropy(actual_pool)),
            "equal_sample_pooled_joint_entropy":float(entropy(equal_pool)),
            "actual_raw_weighted_sample_pooling_js":raw_gap,
            "actual_raw_weighted_sample_pooling_js_nats":raw_gap*np.log(dimension),
            "equal_sample_pooling_js":float(entropy(equal_pool)-entropies[:,bi].mean()),
            "sample_similarity_to_maps":{name:{metric:distribution(np.clip(values[metric][:,bi],-1,1))
                                                   for metric in ("cosine","pearson")}
                                            for name,values in correlations.items()},
            "validation_raw_mean_vs_training_calibration_pearson":float(np.corrcoef(va,ca)[0,1]),
            "validation_raw_mean_vs_training_calibration_cosine":float(va@ca/(np.linalg.norm(va)*np.linalg.norm(ca))),
            "actual_raw_pooled_spatial_marginal":actual_pool.reshape(64,256).sum(0).tolist(),
            "interpretation_limit":"Score-map heterogeneity and pooling gaps measure lost distinctions, not whether those distinctions improve the prediction task."
        }
    return result


def figures(out_dir, keep, attention):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for block in (0,11):
        for kind, style in (("online", "-"),("offline", "--")):
            arm = f"{kind}_L{block}"
            axes[0].plot(keep[arm]["mean_keep_per_slot"], style, label=arm)
        for group, style in (("context_queries", "-"),("target_queries", "--")):
            axes[1].plot(attention[f"L{block}"][group]["pooled_context_profile"], style, label=f"L{block} {group}")
    axes[0].set(xlabel="Context slot (old → recent)",ylabel="Mean retained tokens",title="Actual top-K allocation")
    axes[1].set(xlabel="Context key slot",ylabel="Context-conditioned attention",title="Query-group profiles")
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=7)
    matrix = attention["L11"]["queryslot_keyslot_flow"]["mean_context_flow_head_pooled"]
    im=axes[2].imshow(matrix,origin="lower",aspect="auto")
    axes[2].set(xlabel="Context key slot",ylabel="Context query slot",title="L11 mean context flow")
    fig.colorbar(im,ax=axes[2],fraction=.045)
    fig.tight_layout();fig.savefig(out_dir/"selection_and_attention.png",dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shards",nargs="+",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--allow-partial",action="store_true",help="Smoke only; validates the exact declared shard union")
    p.add_argument("--bootstrap-reps",type=int,default=9999)
    p.add_argument("--bootstrap-seed",type=int,default=202609051)
    p.add_argument("--no-plots",action="store_true")
    args=p.parse_args()
    source_digest_at_start=sha(__file__)
    require(args.bootstrap_reps>=199,"Need at least 199 bootstrap replicates")
    common, metas, summaries, rows, mass, flows, scores, coverage=load_shards(args.shards,args.allow_partial)
    arms=common["arms"]
    correct,loss,valid,counts,result,exclusions=outcomes(rows,arms,summaries,metas)
    paired=bootstrap(rows,arms,correct,loss,valid,args.bootstrap_reps,args.bootstrap_seed)
    selections=keep_analysis(counts,arms)
    attention=attention_analysis(mass,flows,scores)
    score_maps=score_map_analysis(args.shards,common)
    random_summary={h:{"seed_top5":[result[a][h]["action_top5"] for a in RANDOM],
                       "seed_mean_top5":float(np.mean([result[a][h]["action_top5"] for a in RANDOM])),
                       "seed_sd_top5_pp":float(np.std([result[a][h]["action_top5"] for a in RANDOM],ddof=1)*100)} for h in HORIZONS}
    report={"evaluation_protocol":PROTOCOL,"run_kind":"diagnostic","metric_scope":"native",
            "eval_path":common["eval_path"],"analysis_job_id":os.environ.get("SLURM_JOB_ID"),
            "analysis_source_sha256":source_digest_at_start,"input_dirs":[str(p) for p in args.shards],
            "input_jobs":[m["job_id"] for m in metas],"input_run_tags":[m["run_tag"] for m in metas],
            "coverage":coverage,"exclusions":exclusions,"results":result,
            "paired_inference":paired,"random_seed_summary":random_summary,
            "selection_diagnostics":selections,"attention_diagnostics":attention,
            "joint_score_map_diagnostics":score_maps,
            "smoke_only":args.allow_partial,
            "limits":["No validation confidence interval includes calibration-map uncertainty.",
                      "The bootstrap does not measure GPU repeatability; capture parity is a separate first-batch check.",
                      "Fixed 12+4 allocation tests one intervention, not an optimized recency/history frontier.",
                      "History outside the 12-slot anchor includes four slots that pure recent retains; inspect 0–47 vs48–51 allocation before claiming older-history utility."]}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    target=args.out_dir/"summary.json"
    require(not target.exists(),"Refusing to overwrite existing analysis summary")
    target.write_text(json.dumps(report,indent=2)+"\n")
    (args.out_dir/"merged_execution_order.json").write_text(json.dumps([{k:r[k] for k in ("sample_id","video_id","participant_id","source_index","selection_index","donor_sample_id","pair_id")} for r in rows],indent=2)+"\n")
    if not args.no_plots:
        figures(args.out_dir,selections,attention)
    display={"summary":str(target),"coverage":coverage,"results_2s":{a:r["2s"] for a,r in result.items()},
             "primary_2s":{k:v["2s"] for k,v in paired["contrasts"].items() if k in paired["primary"]},
             "mechanism_brief":{f"L{block}":{
                 "temporal_sample_entropy":attention[f"L{block}"]["legacy_all_query_head_sum"]["pooling_decomposition"]["sample_entropy_unweighted_mean"],
                 "temporal_raw_pool_entropy":attention[f"L{block}"]["legacy_all_query_head_sum"]["pooling_decomposition"]["actual_raw_pool_concentration"]["entropy_div_log64"],
                 "temporal_raw_sample_pooling_js":attention[f"L{block}"]["legacy_all_query_head_sum"]["pooling_decomposition"]["sample_pooling_js_raw_mass_weighted"],
                 "joint_sample_entropy":score_maps[f"L{block}"]["per_sample_joint_entropy"]["mean"],
                 "joint_raw_pool_entropy":score_maps[f"L{block}"]["actual_raw_pooled_joint_entropy"],
                 "joint_raw_sample_pooling_js":score_maps[f"L{block}"]["actual_raw_weighted_sample_pooling_js"],
                 "context_same_slot_probability":attention[f"L{block}"]["queryslot_keyslot_flow"]["context_conditional_same_slot_probability"],
                 "context_within_one_slot_probability":attention[f"L{block}"]["queryslot_keyslot_flow"]["context_conditional_within_one_slot_probability"],
                 "online_recent16_mean_count":selections[f"online_L{block}"]["recent16_count"]["mean"],
                 "hybrid_outside_recent16_mean_count":selections[f"hybrid_online_L{block}"]["older_than_recent16_count"]["mean"],
             } for block in (0,11)},
             "smoke_only":args.allow_partial}
    (args.out_dir/"brief.json").write_text(json.dumps(display,indent=2)+"\n")
    print(json.dumps(display,indent=2))


if __name__=="__main__":
    main()
