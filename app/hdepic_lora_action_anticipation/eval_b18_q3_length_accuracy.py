#!/usr/bin/env python3
"""B18 Q3 fixed-4096 predictor budget versus full encoder context length.

Prepare/cache and evaluation run only within Slurm/container. Frozen Q1 rows,
independent training-calibrated masks; no validation-conditioned selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.hdepic_lora_action_anticipation import eval_b18_q1_paired_hybrid as P
from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.eval_stream_mtp_multi_strategy import predict_from_encoded

PROTOCOL = "b18-predictor-prune/egtea-q3-fixed4s-length-accuracy-v1"
ROOT = Path("/scratch/yh6416/VJEPA2-EXP")
OUTROOT = ROOT / "outputs/attn_corner_sink/q3_length_accuracy"
ANN = ROOT / "data/egtea/vjepa_annotations/stream_16s_split/split1"
Q1MANIFEST = ROOT / "outputs/attn_corner_sink/q1_evidence_audit_20260905/paired4000_manifest.csv"
CALIBRATION = ROOT / "outputs/attn_corner_sink/q3_boundaries/analysis-full86/pooled_scores_and_masks.npz"
CALIB_MANIFEST = ROOT / "outputs/attn_corner_sink/q3_boundaries/prepared/manifest.json"
ARCHIVED_MASKS = ROOT / "outputs/attn_corner_sink/q3_boundaries/analysis-fixed4s-17069720/fixed4s_masks.npz"
LENGTHS = [4, 8, 16, 32]
STRATEGIES = ["offline_L0", "recent4s", "hybrid_recent3s_offline1s"]
HORIZONS = [2.0, 4.0, 6.0]
KEEP = 4096


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def model_args():
    return argparse.Namespace(train_csv=ANN / "EGTEA_train_stream_mtp.csv",
        val_csv=ANN / "EGTEA_val_stream_mtp.csv",
        video_root=ROOT / "data/egtea/stream_session_videos",
        checkpoint=ROOT / "checkpoints/vitl.pt",
        init_from_ckpt=ROOT / "outputs/egtea_stream_mtp/action_anticipation_frozen/egtea-split1-stream-mtp-2-4-6-vitl256-llwarm/best.pt",
        encoder_lora=ROOT / "outputs/egtea_split1_video_enc_ll_exact/action_anticipation_frozen/egtea-split1-video-enc-fixed-vitl16-256-10ep/encoder_lora_best.pt",
        predictor_lora=ROOT / "outputs/egtea_split1_video_pred_joint_heads_ll_exact/action_anticipation_frozen/egtea-split1-video-pred-joint-heads-vitl16-256-10ep/predictor_lora_best.pt")


def make_masks():
    masks, audit = {}, {}
    with np.load(CALIBRATION) as z, np.load(ARCHIVED_MASKS) as archived:
        for sec in LENGTHS:
            score = z[f"score__continuous_{sec}s"].reshape(-1)
            assert len(score) == sec * 1024 and score.dtype == np.float64 and np.isfinite(score).all()
            chosen = np.argsort(-score, kind="stable")[:KEEP]
            offline = np.sort(chosen)
            old = archived[f"continuous_{sec}s"].reshape(-1)
            assert np.array_equal(np.flatnonzero(old), offline), "archived fixed4096 mask mismatch"
            anchor_start = len(score) - 3072
            anchor = np.arange(anchor_start, len(score))
            history = np.argsort(-score[:anchor_start], kind="stable")[:1024]
            selections = [offline, np.arange(len(score) - KEEP, len(score)), np.sort(np.r_[history, anchor])]
            for name, idx in zip(STRATEGIES, selections):
                assert len(idx) == KEEP and (np.diff(idx) > 0).all() and idx.min() >= 0 and idx.max() < len(score)
                masks[f"{sec}s__{name}"] = idx
            assert np.array_equal(selections[1][-3072:], selections[2][-3072:])
            if sec == 4:
                assert all(np.array_equal(selections[0], idx) for idx in selections)
            threshold = float(score[chosen[-1]])
            audit[f"{sec}s"] = dict(score_dtype=str(score.dtype), global_threshold=threshold,
                n_equal_threshold=int((score == threshold).sum()),
                archived_mask_exact=True, keep_per_slot={name: np.bincount(idx // 256, minlength=sec * 4).tolist()
                                                        for name, idx in zip(STRATEGIES, selections)},
                recent4s_fraction={name: float((idx >= len(score) - KEEP).sum() / KEEP)
                                   for name, idx in zip(STRATEGIES, selections)})
    return masks, audit


def stage_row(item):
    from decord import VideoReader, cpu
    sample, cache = item
    path = Path(cache) / f"{sample['sample_id']}.npy"
    # Retries may retain completed immutable rows. A sidecar is written only
    # after shape, suffix, decode and content hashing have succeeded.
    side = path.with_suffix(".json")
    if side.exists():
        result = json.loads(side.read_text())
        assert path.stat().st_size == result["bytes"]
        return result
    t0 = time.time()
    vr = VideoReader(sample["video_path"], ctx=cpu(0), num_threads=1, width=256, height=256)
    idx = np.asarray(sample["frame_indices_32s"], dtype=np.int64)
    assert int(idx.min()) >= sample["origin_frame"] and int(idx.max()) < len(vr)
    assert len(vr) == sample["n_frames"] and abs(vr.get_avg_fps() / sample["vfps"] - 1) < 0.005
    rgb = vr.get_batch(idx.tolist()).asnumpy()
    assert rgb.shape == (256, 256, 256, 3) and rgb.dtype == np.uint8
    if sample["common_index"] < 4:
        original = vr.get_batch(sample["frame_indices_16s"]).asnumpy()
        assert np.array_equal(rgb[-128:], original), "direct Q1 RGB decode differs"
    del vr
    np.save(path, rgb)
    result = dict(sample_id=sample["sample_id"], common_index=sample["common_index"],
                  path=str(path), sha256=sha(path), bytes=path.stat().st_size, seconds=time.time() - t0)
    write_json(side, result)
    return result


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    ma = model_args()
    assert sha(Q1MANIFEST) == "c23bdf34bafe75a59e05e16a32e7e8aaf4717a04655982daebd70b079e3b8bf2"
    q1 = P.load_manifest(ma.val_csv, Q1MANIFEST, 4000)
    samples, excluded, audits = [], [], []
    for execution_index, sample in enumerate(q1["samples"]):
        row = sample["row"]
        original = np.asarray([int(v) for v in row["frame_indices"].split(",")])
        tick, fps, origin = int(row["tick_frame"]), float(row["vfps"]), int(row["origin_frame"])
        assert row["split"] == "val" and origin == int(row["n_frames"]) // 2
        assert original.shape == (128,) and (np.diff(original) > 0).all()
        assert original[0] == int(row["start_frame"]) and original[-1] == tick - 1
        generator32 = np.rint(np.linspace(tick - int(round(32 * fps)), tick - 1, 256)).astype(np.int64)
        frames = np.r_[generator32[:128], original]
        identity = {k: v for k, v in sample.items() if k != "row"}
        identity["q1_execution_index"] = execution_index
        if frames[0] < origin:
            excluded.append(dict(**identity, reason="32s_start_before_validation_half_origin",
                                 proposed_start=int(frames[0]), origin_frame=origin, missing_frames=int(origin - frames[0])))
            continue
        assert (np.diff(frames) > 0).all() and frames[-1] < tick and frames[-1] < int(row["n_frames"])
        video = ma.video_root / row["video_id"] / (row["video_id"] + ".MP4")
        assert video.is_file(), video
        s = dict(**identity, common_index=len(samples), video_path=str(video), origin_frame=origin,
                 n_frames=int(row["n_frames"]), vfps=fps, row=row,
                 frame_indices_16s=original.tolist(), frame_indices_32s=frames.tolist())
        samples.append(s)
        q3_grid = np.rint(original[-1] - np.arange(255, -1, -1) * fps / 8).astype(np.int64)
        audits.append(dict(sample_id=s["sample_id"], max_abs_frame_difference_q3=int(abs(frames - q3_grid).max()),
            mismatch_frames_q3=int((frames != q3_grid).sum()), boundary_gap=int(frames[128] - frames[127]),
            generator32_suffix_differences=int((generator32[128:] != original).sum()),
            spans_sec={f"{sec}s": float((frames[-1] - frames[-sec * 8]) / fps) for sec in LENGTHS}))
    masks, mask_audit = make_masks()
    np.savez_compressed(args.out / "fixed_masks.npz", **masks)
    paths = [Q1MANIFEST, Q1MANIFEST.with_suffix(".meta.json"), CALIBRATION, CALIB_MANIFEST, ARCHIVED_MASKS,
             *[getattr(ma, key) for key in ["train_csv", "val_csv", "checkpoint", "init_from_ckpt", "encoder_lora", "predictor_lora"]],
             Path(__file__), Path(P.__file__), Path(T.__file__), Path(__file__).with_name("eval_stream_mtp_multi_strategy.py"),
             Path(__file__).resolve().parents[2] / "scripts/egtea/make_b13_egtea_stream_half_split.py"]
    meta = dict(protocol=PROTOCOL, source_n=4000, common_n=len(samples), excluded_n=len(excluded),
        sessions=len({s["video_id"] for s in samples}), participants=len({s["participant_id"] for s in samples}),
        samples=samples, excluded=excluded, frame_audit=audits, masks=mask_audit,
        source_sha256={str(p): sha(p) for p in paths}, masks_sha256=sha(args.out / "fixed_masks.npz"),
        frame_rule="32s: source-generator 256-point linspace first128 + exact Q1 source128 suffix; 4/8/16s exact suffixes; no clipping, padding, duplicates, future or pre-val-origin frames",
        calibration_sampling_difference="Q3 calibration uses backward native-fps/8 grid; validation preserves original Q1 linspace suffix. These are not bitwise frame-grid equivalent.",
        execution="full encoder per length, shared by strategies; packed 4096 predictor context and target position slot24 at anticipation2s",
        cache_dir=str(args.out / "rgb"), created=time.time())
    write_json(args.out / "manifest.json", meta)
    print(json.dumps({k: meta[k] for k in ["source_n", "common_n", "excluded_n", "sessions", "participants"]}), flush=True)
    cache(argparse.Namespace(manifest=args.out / "manifest.json", start=0, stop=args.cache_stop,
                             workers=args.workers, report=args.out / "cache-smoke.json"))


def cache(args):
    manifest = json.loads(args.manifest.read_text())
    path = Path(manifest["cache_dir"])
    path.mkdir(exist_ok=True)
    stop = args.stop or manifest["common_n"]
    samples = manifest["samples"][args.start:stop]
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(stage_row, [(s, path) for s in samples])):
            results.append(result)
            if (i + 1) % 32 == 0 or i + 1 == len(samples):
                print(f"[cache] {i+1}/{len(samples)} elapsed={time.time()-started:.1f}s", flush=True)
    write_json(args.report, dict(protocol=PROTOCOL, manifest_sha256=sha(args.manifest), start=args.start, stop=stop,
                                seconds=time.time() - started, workers=args.workers, rows=results))


class CachedRows(Dataset):
    def __init__(self, samples, cache_dir):
        self.samples, self.cache_dir = samples, Path(cache_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = self.cache_dir / f"{sample['sample_id']}.npy"
        side = json.loads(path.with_suffix(".json").read_text())
        assert side["sample_id"] == sample["sample_id"] and path.stat().st_size == side["bytes"]
        rgb = np.load(path)
        assert rgb.shape == (256, 256, 256, 3) and rgb.dtype == np.uint8
        return torch.from_numpy(rgb).permute(3, 0, 1, 2).contiguous(), sample


def collate(batch):
    return torch.stack([v[0] for v in batch]), [v[1] for v in batch]


def compare_anchor(current, reference):
    """Fail-closed cross-device gate; BF16 logits may round differently."""
    report = {}
    assert current.keys() == reference.keys()
    for key, value in current.items():
        ref = reference[key]
        if key.startswith("indices") or key.startswith("sample"):
            assert np.array_equal(value, ref), key
            continue
        diff = value.astype(np.float64) - ref.astype(np.float64)
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12))
        max_abs = float(abs(diff).max())
        # Logit gate fixed before smoke: relative L2<=0.005 and max abs<=0.15.
        assert rel <= 0.005 and max_abs <= 0.15, (key, rel, max_abs)
        report[key] = dict(relative_l2=rel, max_abs=max_abs)
    return report


def evaluate(args):
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    assert manifest["protocol"] == PROTOCOL
    assert sha(args.manifest.parent / "fixed_masks.npz") == manifest["masks_sha256"]
    ma = model_args()
    for path in [ma.train_csv, ma.val_csv, ma.checkpoint, ma.init_from_ckpt, ma.encoder_lora, ma.predictor_lora,
                 CALIBRATION, CALIB_MANIFEST, ARCHIVED_MASKS]:
        assert sha(path) == manifest["source_sha256"][str(path)], path
    torch.manual_seed(20260907); np.random.seed(20260907)
    assert torch.cuda.is_bf16_supported(), "BF16-capable GPU required; no precision fallback"
    device = torch.device("cuda")
    base, mtp, maps = P.build(ma, device)
    # RoPE encoder accepts dynamic T; no learned state or angle scaling changes.
    masks, mask_audit = make_masks()
    with np.load(args.manifest.parent / "fixed_masks.npz") as frozen:
        assert all(np.array_equal(idx, frozen[key]) for key, idx in masks.items())
    gpu_masks = {key: torch.as_tensor(idx, device=device) for key, idx in masks.items()}
    stop = args.stop or manifest["common_n"]
    samples = manifest["samples"][args.start:stop]
    assert samples and 0 <= args.start < stop <= manifest["common_n"]
    metadata = dict(protocol=PROTOCOL, run_tag=args.tag, job_id=os.environ.get("SLURM_JOB_ID"),
        metric_scope="native", eval_path="EGTEA split1 Q1 common eligible subset; full encoder per length; packed fixed4096 predictor + native MTP head",
        manifest=str(args.manifest), manifest_sha256=sha(args.manifest), start=args.start, stop=stop,
        strategies=STRATEGIES, lengths=LENGTHS, horizons=HORIZONS, batch_size=args.batch, workers=args.workers,
        dtype="BF16 autocast; logits exported FP32", masks=mask_audit,
        class_ranking="CPU torch.topk on exported FP32 action logits; ties backend-dependent; CPU analysis reports tie bounds",
        class_counts=[len(v) for v in maps],
        source_sha256={str(Path(p)): sha(p) for p in [__file__, P.__file__, T.__file__,
                     Path(__file__).with_name("eval_stream_mtp_multi_strategy.py"),
                     Path(__file__).resolve().parents[2] / "vjepa2/src/models/utils/modules.py",
                     Path(__file__).resolve().parents[2] / "vjepa2/src/models/predictor.py"]},
        gpu=dict(name=torch.cuda.get_device_name(), memory=torch.cuda.get_device_properties(0).total_memory,
                 torch=torch.__version__, cuda=torch.version.cuda),
        target_positions=dict(context=list(range(KEEP)), anticipation_steps=8, first_target=6144,
                              n_target=256, rebased=True),
        hardware_gate=dict(relative_l2_max=0.005, logit_max_abs=0.15),
        model_load="P.build exact missing/unexpected zero, frozen eval; unchanged checkpoint-compatible RoPE")
    write_json(args.out / "metadata.json", metadata)
    write_json(args.out / "execution_order.json", [{k: v for k, v in s.items() if k != "row"} for s in samples])

    def forward_rgb(rgb, identities, parity=False):
        B = len(identities)
        ant = torch.full((B,), 2.0, device=device)
        rgb = rgb.to(device).float().div_(255).sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
        results, anchors = {}, {"sample_ids": np.asarray([s["sample_id"] for s in identities])}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for sec in LENGTHS:
                clip = rgb[:, :, -sec * 8:]
                encoded = base.encoder(clip)
                assert encoded.shape[1] == sec * 1024
                arms = STRATEGIES if sec > 4 else STRATEGIES[:1]
                for strategy in arms:
                    name = f"{sec}s__{strategy}"
                    idx = gpu_masks[name].expand(B, -1)
                    kept = encoded.gather(1, idx.unsqueeze(-1).expand(-1, -1, encoded.shape[-1]))
                    out = mtp(predict_from_encoded(base, kept, ant))
                    results[name] = np.stack([out[h]["action"].float().cpu().numpy() for h in HORIZONS], axis=1)
                    assert results[name].shape == (B, 3, len(maps[2])) and np.isfinite(results[name]).all()
                    if sec in [4, 16, 32]:
                        anchors["logits__" + name] = results[name]
                        anchors["indices__" + name] = masks[name]
                    if parity and sec == 4:
                        # Compute all three once for exact same-execution identity,
                        # then deduplicate the 4s arms in the representative loop.
                        for alias in STRATEGIES[1:]:
                            ai = gpu_masks[f"4s__{alias}"].expand(B, -1)
                            again = mtp(predict_from_encoded(base, encoded.gather(1, ai.unsqueeze(-1).expand_as(kept)), ant))
                            assert all(torch.equal(out[h]["action"], again[h]["action"]) for h in HORIZONS)
                    if parity and sec == 16 and strategy == "recent4s":
                        # Q1 path literally uses same x.gather + imported helper;
                        # independently express its recent index and require logits.
                        recent_idx = torch.arange(16384 - KEEP, 16384, device=device).expand(B, -1)
                        assert torch.equal(idx, recent_idx)
                        legacy = mtp(P.predict_from_encoded(base, encoded.gather(1, recent_idx.unsqueeze(-1).expand_as(kept)), ant))
                        assert all(torch.equal(out[h]["action"], legacy[h]["action"]) for h in HORIZONS)
                if sec == 4:
                    for alias in STRATEGIES[1:]:
                        results[f"4s__{alias}"] = results[f"4s__{STRATEGIES[0]}"]
        return results, anchors

    # Identical first two common rows, B=2, on every backend/shard. Decode path
    # is checked independently against the Q1 loader in the smoke.
    fixed = manifest["samples"][:2]
    ar_rgb, ar_samples = collate([CachedRows(fixed, manifest["cache_dir"])[i] for i in range(2)])
    if args.parity:
        direct = P.IdentifiedDataset(ma, fixed)
        for i in range(2):
            assert torch.equal(direct[i]["clip"], ar_rgb[i, :, -128:]), "Q1 loader RGB suffix parity"
    _, anchor = forward_rgb(ar_rgb, ar_samples, parity=args.parity)
    np.savez_compressed(args.out / "hardware_anchor.npz", **anchor)
    metadata["parity"] = dict(q1_rgb_suffix_exact=args.parity, four_second_arms_exact=args.parity,
                              q1_recent16_logits_exact=args.parity)
    if args.reference:
        with np.load(args.reference) as reference:
            metadata["hardware_comparison"] = compare_anchor(anchor, dict(reference))
    write_json(args.out / "metadata.json", metadata)
    del ar_rgb
    loader = DataLoader(CachedRows(samples, manifest["cache_dir"]), batch_size=args.batch,
        shuffle=False, num_workers=args.workers, collate_fn=collate, pin_memory=True,
        **({"prefetch_factor": 2} if args.workers else {}))
    output = (args.out / "predictions.jsonl").open("x")
    timings = (args.out / "timings.jsonl").open("x")
    started = previous = time.time()
    cursor = 0
    for it, (rgb, identities) in enumerate(loader):
        t0 = time.time()
        results, _ = forward_rgb(rgb, identities)
        torch.cuda.synchronize()
        compute_end = time.time()
        valid_labels = []
        for s in identities:
            row = s["row"]
            labels = {}
            verbs = [int(v) for v in row["mtp_verbs"].split(",")]
            nouns = [int(v) for v in row["mtp_nouns"].split(",")]
            validity = [float(v) for v in row["mtp_mask"].split(",")]
            for hi, h in enumerate(HORIZONS):
                key = f"{h:g}s"
                if validity[hi] <= 0.5:
                    labels[key] = dict(valid=False, reason="masked")
                else:
                    _, _, action, keep = T.map_labels(torch.tensor([verbs[hi]]), torch.tensor([nouns[hi]]), *maps, torch.device("cpu"))
                    labels[key] = (dict(valid=True, label=int(action[0])) if len(keep) else
                                   dict(valid=False, reason="out_of_training_vocabulary"))
            valid_labels.append(labels)
        for bi, sample in enumerate(identities):
            record = {k: sample[k] for k in ["sample_id", "common_index", "q1_execution_index", "selection_index", "source_index", "video_id", "participant_id"]}
            record["labels"], record["arms"] = valid_labels[bi], {}
            for name, logits_batch in results.items():
                arm = {}
                for hi, h in enumerate(HORIZONS):
                    key = f"{h:g}s"
                    logits = torch.from_numpy(logits_batch[bi, hi])
                    top5 = logits.topk(5).indices.tolist()
                    label = valid_labels[bi][key]
                    entry = dict(top5_indices=top5)
                    if label["valid"]:
                        y = label["label"]
                        entry.update(label=y, top1=y == top5[0], top3=y in top5[:3], top5=y in top5,
                            ce=float(torch.nn.functional.cross_entropy(logits[None], torch.tensor([y]))))
                    arm[key] = entry
                record["arms"][name] = arm
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
        # Full FP32 action logits, all arms/horizons, immutable batch files.
        np.savez_compressed(args.out / f"logits_{cursor:05d}.npz", sample_ids=np.asarray([s["sample_id"] for s in identities]), **results)
        cursor += len(identities)
        output.flush()
        end = time.time()
        timing = dict(iteration=it, start=t0, compute_end=compute_end, end=end,
                      rows=len(identities), data_wait=t0 - previous, compute_seconds=compute_end - t0,
                      seconds=end - t0, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        timings.write(json.dumps(timing) + "\n"); timings.flush()
        if it % 8 == 0 or cursor == len(samples):
            print(f"[eval] n={cursor}/{len(samples)} compute={compute_end-t0:.3f}s data={t0-previous:.3f}s elapsed={end-started:.1f}s", flush=True)
        previous = end
    output.close(); timings.close()
    assert cursor == stop - args.start
    summary = dict(**metadata, rows=cursor, compute_start=started, compute_end=time.time(), seconds=time.time()-started,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated())
    write_json(args.out / "summary.json", summary)
    print("[done] " + json.dumps(dict(rows=cursor, seconds=summary["seconds"], out=str(args.out))), flush=True)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="mode", required=True)
    prep = sp.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--cache-stop", type=int, default=32)
    prep.add_argument("--workers", type=int, default=4)
    cp = sp.add_parser("cache")
    cp.add_argument("--manifest", type=Path, required=True)
    cp.add_argument("--start", type=int, default=0)
    cp.add_argument("--stop", type=int, default=0)
    cp.add_argument("--workers", type=int, default=8)
    cp.add_argument("--report", type=Path, required=True)
    ev = sp.add_parser("eval")
    ev.add_argument("--manifest", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--tag", required=True)
    ev.add_argument("--start", type=int, default=0)
    ev.add_argument("--stop", type=int, default=32)
    ev.add_argument("--batch", type=int, default=2)
    ev.add_argument("--workers", type=int, default=2)
    ev.add_argument("--parity", action="store_true")
    ev.add_argument("--reference", type=Path)
    args = p.parse_args()
    {"prepare": prepare, "cache": cache, "eval": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
