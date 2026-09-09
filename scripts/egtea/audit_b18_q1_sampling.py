#!/usr/bin/env python3
"""CPU-only B18 Q1 artifact audit and immutable session-proportional manifest.

Run inside the project container on a Slurm CPU allocation. No model inference.
Rows use zero-based indices into the complete source CSV (excluding its header).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_once(path, text):
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"Refusing to overwrite differing immutable artifact: {path}")
    else:
        path.write_text(text)


def summarize_rows(rows, source_indices, vocabulary):
    chosen = [rows[i] for i in source_indices]
    videos = Counter(r["video_id"] for r in chosen)
    participants = Counter(r["video_id"].split("-")[0] for r in chosen)
    by_video = defaultdict(list)
    for row in chosen:
        by_video[row["video_id"]].append(row)
    tick_gaps, overlap, same_label, exact_frame_overlap = [], [], [], []
    nonoverlap_greedy = 0
    for video_rows in by_video.values():
        video_rows.sort(key=lambda r: int(r["tick_frame"]))
        last_end = -1
        for row in video_rows:
            if int(row["start_frame"]) >= last_end:
                nonoverlap_greedy += 1
                last_end = int(row["tick_frame"])
        for a, b in zip(video_rows, video_rows[1:]):
            fps = float(a["vfps"])
            tick_gaps.append((int(b["tick_frame"]) - int(a["tick_frame"])) / fps)
            intersection = max(0, min(int(a["tick_frame"]), int(b["tick_frame"])) -
                               max(int(a["start_frame"]), int(b["start_frame"])))
            overlap.append(intersection / (int(a["tick_frame"]) - int(a["start_frame"])))
            fa, fb = set(a["frame_indices"].split(",")), set(b["frame_indices"].split(","))
            exact_frame_overlap.append(len(fa & fb) / len(fa))
            la = (a["mtp_verbs"].split(",")[0], a["mtp_nouns"].split(",")[0])
            lb = (b["mtp_verbs"].split(",")[0], b["mtp_nouns"].split(",")[0])
            same_label.append(la == lb)
    denominators = {}
    for hi, horizon in enumerate((2, 4, 6)):
        valid = missing = oov = 0
        for row in chosen:
            mask = float(row["mtp_mask"].split(",")[hi])
            verb = int(row["mtp_verbs"].split(",")[hi])
            noun = int(row["mtp_nouns"].split(",")[hi])
            if mask <= 0.5:
                missing += 1
            elif (verb, noun) not in vocabulary:
                oov += 1
            else:
                valid += 1
        denominators[f"{horizon}s"] = dict(valid=valid, missing_label=missing, out_of_vocabulary=oov)
    def quantiles(values):
        return dict(zip(("min", "q25", "median", "q75", "max"),
                        np.quantile(values, [0, .25, .5, .75, 1]).tolist())) if values else None
    return {
        "n_rows": len(chosen), "n_videos": len(videos), "n_participants": len(participants),
        "videos": dict(sorted(videos.items())), "participants": dict(sorted(participants.items())),
        "valid_denominators": denominators, "adjacent_within_video_pairs": len(tick_gaps),
        "tick_gap_sec": quantiles(tick_gaps), "context_interval_overlap_fraction": quantiles(overlap),
        "decoded_frame_set_overlap_fraction": quantiles(exact_frame_overlap),
        "adjacent_same_2s_action_fraction": float(np.mean(same_label)) if same_label else None,
        "greedy_disjoint_context_count": nonoverlap_greedy,
        "caveat": "Disjoint-window count is descriptive, not an estimated effective sample size. "
                  "Video and participant clusters can remain dependent even after context overlap removal.",
    }


def proportional_manifest(rows, eligible, n, seed):
    groups = defaultdict(list)
    for i in eligible:
        groups[rows[i]["video_id"]].append(i)
    total = len(eligible)
    quotas = {video: n * len(indices) // total for video, indices in groups.items()}
    remaining = n - sum(quotas.values())
    priority = sorted(groups, key=lambda video: (-(n * len(groups[video]) % total), video))
    for video in priority[:remaining]:
        quotas[video] += 1
    if min(quotas.values()) < 1:
        raise RuntimeError("Requested sample cannot cover every session under proportional allocation")
    selected = []
    for video, indices in groups.items():
        indices.sort(key=lambda i: (hashlib.sha256(
            f"{seed}|{video}|{rows[i]['frame_indices']}".encode()).hexdigest(), i))
        selected.extend(indices[:quotas[video]])
    return sorted(selected), dict(sorted(quotas.items()))


def map_summary(path, k):
    array = np.load(path).astype(np.float32)
    assert array.shape == (64, 256) and np.isfinite(array).all()
    flat = array.reshape(-1)
    indices = np.argsort(-flat, kind="stable")[:k]
    cutoff = float(flat[indices[-1]])
    at_cutoff = np.flatnonzero(flat == cutoff)
    counts = np.bincount(indices // 256, minlength=64)
    mass = array.sum(axis=1, dtype=np.float64)
    mass /= mass.sum()
    return {
        "map_path": str(path), "map_sha256": sha256(path), "keep_count": k,
        "topk_tie_boundary_total": len(at_cutoff),
        "topk_tie_boundary_selected": int(np.count_nonzero(flat[indices] == cutoff)),
        "unique_mask_if_no_split_boundary_tie": bool(len(at_cutoff) == np.count_nonzero(flat[indices] == cutoff)),
        "kept_count_by_slot": counts.tolist(), "attention_mass_by_slot": mass.tolist(),
        "recent8_keep_fraction": float(counts[-8:].sum() / k),
        "recent16_keep_fraction": float(counts[-16:].sum() / k),
        "recent32_keep_fraction": float(counts[-32:].sum() / k),
        "recent8_attention_mass_fraction": float(mass[-8:].sum()),
        "whole_slots": np.flatnonzero(counts == 256).tolist(),
        "caveat": "Offline fixed-mask counts only. Online mask allocation cannot be recovered from calibration mass.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sample-size", type=int, default=4000)
    ap.add_argument("--seed", default="b18-q1-paired-v1")
    args = ap.parse_args()
    ann = args.data_root / "data/egtea/vjepa_annotations/stream_16s_split/split1"
    train_path, val_path = ann / "EGTEA_train_stream_mtp.csv", ann / "EGTEA_val_stream_mtp.csv"
    train, val = load_rows(train_path), load_rows(val_path)
    vocabulary = set()
    for row in train:
        for v, n, m in zip(row["mtp_verbs"].split(","), row["mtp_nouns"].split(","), row["mtp_mask"].split(",")):
            if float(m) >= .5 and int(v) >= 0 and int(n) >= 0:
                vocabulary.add((int(v), int(n)))
    eligible = [i for i, row in enumerate(val) if abs(float(row["context_sec"]) - 16) < 1e-6]
    assert {int(val[i]["n_model_frames"]) for i in eligible} == {128}
    old_indices = eligible[:4000]
    selected, quotas = proportional_manifest(val, eligible, args.sample_size, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["original_csv_row_index", "selection_index", "video_id", "participant_id", "tick_frame",
              "context_sec", "frame_indices", "row_sha256"]
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for selection_index, i in enumerate(selected):
        row = val[i]
        writer.writerow({"original_csv_row_index": i, "selection_index": selection_index,
                         "participant_id": row["video_id"].split("-")[0], "row_sha256": row_hash(row),
                         **{k: row[k] for k in ("video_id", "tick_frame", "context_sec", "frame_indices")}})
    manifest_path = args.out_dir / f"paired{args.sample_size}_manifest.csv"
    write_once(manifest_path, buffer.getvalue())
    train_eligible = [i for i, row in enumerate(train) if abs(float(row["context_sec"]) - 16) < 1e-6]
    calibration_indices = sorted(train_eligible, key=lambda i: hashlib.md5(
        f"0|{train[i]['video_id']}|{train[i]['frame_indices']}".encode()).hexdigest())[:512]
    out_base = args.data_root / "outputs/attn_corner_sink"
    metadata = {
        "evaluation_protocol": "b18-predictor-prune/egtea-ctx16-paired-hybrid-v1",
        "selection": "Proportional per-session largest-remainder quotas; SHA256 within session; source CSV order afterward",
        "seed": args.seed, "source_csv": str(val_path), "source_csv_sha256": sha256(val_path),
        "source_index_definition": "zero-based data row excluding header, before context filtering",
        "row_sha256_definition": "sha256(json.dumps(csv_dict, sort_keys=True, separators=(',', ':')).encode())",
        "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "eligible_rows": len(eligible), "sample_size": len(selected), "per_session_quotas": quotas,
        "old4000_overlap_rows": len(set(old_indices) & set(selected)),
    }
    write_once(manifest_path.with_suffix(".meta.json"), json.dumps(metadata, indent=2) + "\n")
    results = {}
    for path in sorted((out_base / "multi_strategy").glob("*-sb*-nb1000.json")):
        record = json.loads(path.read_text())
        results[path.name] = {k: record[k] for k in ("prune_strategy", "action_top5", "n")}
    audit = {
        "source_csv_sha256": sha256(val_path), "train_csv_sha256": sha256(train_path),
        "old4000": summarize_rows(val, old_indices, vocabulary),
        "full_ctx16": summarize_rows(val, eligible, vocabulary),
        "new4000": summarize_rows(val, selected, vocabulary),
        "calibration512_reconstructed_seed0": summarize_rows(train, calibration_indices, vocabulary),
        "calibration_reconstruction_caveat": "Selection reconstructed from current source and default seed 0; old metadata omitted seed and selected row IDs. n_used=512 confirms count only.",
        "offline_masks": {str(block): map_summary(out_base / "pred_offline_calib" /
                          f"calib_predblk{block}_map_64x256.npy", 4096) for block in (0, 6, 11)},
        "old_result_aggregates": results, "manifest": metadata,
        "paired_prediction_artifacts": "Not saved by old multi_strategy evaluator; neither paired test nor cluster confidence interval recoverable from aggregate JSONs.",
        "inference_vs_sampling": "Frozen eval uses deterministic row/frame order and eval mode, but repeated GPU predictions were not compared. Dataset sampling and calibration sampling uncertainty are separate from GPU numerical repeatability.",
    }
    write_once(args.out_dir / "audit.json", json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"audit": str(args.out_dir / "audit.json"), "manifest": metadata,
                      "units": {k: {q: audit[k][q] for q in ("n_rows", "n_videos", "n_participants", "valid_denominators", "tick_gap_sec", "context_interval_overlap_fraction", "greedy_disjoint_context_count")}
                                for k in ("old4000", "full_ctx16", "new4000")},
                      "offline": {k: {q: v[q] for q in ("recent8_keep_fraction", "recent16_keep_fraction", "recent32_keep_fraction", "recent8_attention_mass_fraction", "whole_slots", "unique_mask_if_no_split_boundary_tie")}
                                  for k, v in audit["offline_masks"].items()}}, indent=2))


if __name__ == "__main__":
    main()
