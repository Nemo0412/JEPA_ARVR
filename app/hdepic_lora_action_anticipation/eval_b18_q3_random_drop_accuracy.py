#!/usr/bin/env python3
"""B18 Q3 fixed global-random 4096/16384 predictor-token accuracy.

Run only inside the project Slurm/container route.  The three random masks are
fixed across validation examples; this estimates fixed random position pruning,
not per-example resampling.
"""
from __future__ import annotations
from app.hdepic_lora_action_anticipation.share_paths import DATA_ROOT as SHARE_DATA_ROOT, VJEPA_ROOT as SHARE_VJEPA_ROOT

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation import eval_b18_q1_paired_hybrid as P
from app.hdepic_lora_action_anticipation import eval_b18_q3_length_accuracy as L
from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.eval_stream_mtp_multi_strategy import predict_from_encoded

PROTOCOL = "b18-predictor-prune/egtea-q3-random-drop-accuracy-v1"
ROOT = Path(str(SHARE_DATA_ROOT))
OUTROOT = ROOT / "outputs/attn_corner_sink/q3_random_drop_accuracy"
ANN = ROOT / "data/egtea/vjepa_annotations/stream_16s_split/split1"
Q1MANIFEST = ROOT / "outputs/attn_corner_sink/q1_evidence_audit_20260905/paired4000_manifest.csv"
RANDOM_SEEDS = [20260908, 20260909, 20260910]
ANCHORS = ["recent4s", "offline_L0", "hybrid_recent3s_offline1s"]
RANDOM_ARMS = [f"random_seed{s}" for s in RANDOM_SEEDS]
ARMS = ANCHORS + RANDOM_ARMS
HORIZONS = [2.0, 4.0, 6.0]
KEEP = 4096
TOTAL = 16384


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def model_args():
    return L.model_args()


def make_masks():
    length_masks, length_audit = L.make_masks()
    masks = {
        "recent4s": length_masks["16s__recent4s"],
        "offline_L0": length_masks["16s__offline_L0"],
        "hybrid_recent3s_offline1s": length_masks["16s__hybrid_recent3s_offline1s"],
    }
    for seed, arm in zip(RANDOM_SEEDS, RANDOM_ARMS):
        masks[arm] = np.sort(np.random.default_rng(seed).choice(TOTAL, KEEP, replace=False))
    audit = {}
    for arm, idx in masks.items():
        idx = np.asarray(idx, dtype=np.int64)
        assert idx.shape == (KEEP,) and (np.diff(idx) > 0).all() and idx[0] >= 0 and idx[-1] < TOTAL
        payload = idx.astype("<i8", copy=False).tobytes()
        audit[arm] = dict(index_sha256=hashlib.sha256(payload).hexdigest(), keep=KEEP,
                          temporal_counts=np.bincount(idx // 256, minlength=64).tolist(),
                          spatial_counts=np.bincount(idx % 256, minlength=256).tolist(),
                          oldest_slot_count=int((idx < 256).sum()), recent16_slots_count=int((idx >= 48 * 256).sum()))
    audit["source_length_mask_audit"] = length_audit["16s"]
    return masks, audit


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    ma = model_args()
    assert sha(Q1MANIFEST) == "c23bdf34bafe75a59e05e16a32e7e8aaf4717a04655982daebd70b079e3b8bf2"
    q1 = P.load_manifest(ma.val_csv, Q1MANIFEST, 4000)
    samples = q1["samples"]
    assert len(samples) == 4000 and len({s["sample_id"] for s in samples}) == 4000
    assert len({s["video_id"] for s in samples}) == 86 and len({s["participant_id"] for s in samples}) == 32
    masks, audit = make_masks()
    np.savez_compressed(args.out / "fixed_masks.npz", **masks)
    source_paths = [Q1MANIFEST, Q1MANIFEST.with_suffix(".meta.json"), L.CALIBRATION, L.CALIB_MANIFEST,
                    L.ARCHIVED_MASKS, ma.train_csv, ma.val_csv, ma.checkpoint, ma.init_from_ckpt,
                    ma.encoder_lora, ma.predictor_lora, Path(__file__), Path(P.__file__), Path(L.__file__),
                    Path(T.__file__), Path(__file__).with_name("eval_stream_mtp_multi_strategy.py")]
    execution = [{k: v for k, v in s.items() if k != "row"} for s in samples]
    write_json(args.out / "execution_order.json", execution)
    meta = dict(protocol=PROTOCOL, source_n=4000, evaluated_n=4000, sessions=86, participants=32,
                q1_manifest=str(Q1MANIFEST), q1_manifest_sha256=sha(Q1MANIFEST),
                q1_manifest_meta_sha256=sha(Q1MANIFEST.with_suffix(".meta.json")),
                execution_order_sha256=sha(args.out / "execution_order.json"), samples=samples,
                masks=audit, masks_npz_sha256=sha(args.out / "fixed_masks.npz"),
                random_definition="uniform without replacement from positions 0..16383; one fixed shared mask per seed across all samples",
                inference_definition="16s encoder; sorted fixed indices; packed/rebased arange(4096) predictor context; unchanged native MTP head",
                source_sha256={str(p): sha(p) for p in source_paths}, created=time.time())
    write_json(args.out / "manifest.json", meta)
    print(json.dumps({k: meta[k] for k in ["protocol", "evaluated_n", "sessions", "participants", "masks_npz_sha256"]}), flush=True)


def compare_anchor(current, reference):
    report = {}
    assert np.array_equal(current["sample_ids"], reference["sample_ids"])
    for arm in ANCHORS:
        key = "logits__" + arm
        rkey = key if key in reference else "logits__16s__" + arm
        a, b = current[key].astype(np.float64), reference[rkey].astype(np.float64)
        diff = a - b
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-12))
        max_abs = float(abs(diff).max())
        assert rel <= 0.005 and max_abs <= 0.15, (arm, rel, max_abs)
        ikey = "indices__" + arm
        rikey = ikey if ikey in reference else "indices__16s__" + arm
        assert np.array_equal(current[ikey], reference[rikey])
        report[arm] = dict(relative_l2=rel, max_abs=max_abs,
                           top1_agreement=float((a.argmax(-1) == b.argmax(-1)).mean()))
    # Once the random protocol has an accepted smoke anchor, full shards also
    # compare all three newly defined random arms on the same fixed rows.
    for arm in RANDOM_ARMS:
        key = "logits__" + arm
        if key not in reference:
            continue
        a, b = current[key].astype(np.float64), reference[key].astype(np.float64)
        diff = a - b
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-12)); max_abs = float(abs(diff).max())
        assert rel <= 0.005 and max_abs <= 0.15, (arm, rel, max_abs)
        assert np.array_equal(current["indices__" + arm], reference["indices__" + arm])
        report[arm] = dict(relative_l2=rel, max_abs=max_abs,
                           top1_agreement=float((a.argmax(-1) == b.argmax(-1)).mean()))
    return report


def evaluate(args):
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    assert manifest["protocol"] == PROTOCOL and manifest["evaluated_n"] == 4000
    assert sha(args.manifest.parent / "fixed_masks.npz") == manifest["masks_npz_sha256"]
    ma = model_args()
    for path in [ma.train_csv, ma.val_csv, ma.checkpoint, ma.init_from_ckpt, ma.encoder_lora,
                 ma.predictor_lora, L.CALIBRATION, L.CALIB_MANIFEST, L.ARCHIVED_MASKS]:
        assert sha(path) == manifest["source_sha256"][str(path)], path
    torch.manual_seed(20260908); np.random.seed(20260908)
    assert torch.cuda.is_bf16_supported(), "BF16-capable GPU required; no precision fallback"
    device = torch.device("cuda")
    base, mtp, maps = P.build(ma, device)
    masks, audit = make_masks()
    with np.load(args.manifest.parent / "fixed_masks.npz") as frozen:
        assert set(frozen.files) == set(ARMS)
        assert all(np.array_equal(frozen[k], v) for k, v in masks.items())
    gpu_masks = {k: torch.as_tensor(v, device=device) for k, v in masks.items()}
    stop = args.stop or manifest["evaluated_n"]
    assert 0 <= args.start < stop <= manifest["evaluated_n"]
    samples = manifest["samples"][args.start:stop]
    ds = P.IdentifiedDataset(ma, samples)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=P.collate, pin_memory=True,
                        **({"prefetch_factor": 2} if args.workers else {}))
    metadata = dict(protocol=PROTOCOL, run_tag=args.tag, job_id=os.environ.get("SLURM_JOB_ID"),
        metric_scope="native", manifest=str(args.manifest), manifest_sha256=sha(args.manifest),
        start=args.start, stop=stop, arms=ARMS, random_seeds=RANDOM_SEEDS, horizons=HORIZONS,
        batch_size=args.batch, workers=args.workers, class_counts=[len(x) for x in maps], masks=audit,
        dtype="BF16 autocast forward; exported action logits FP32",
        selection="all fixed indices sorted in original 0..16383 coordinates; predictor rebase unchanged",
        random_scope="fixed shared global position masks; temporal and spatial allocations both vary",
        target_positions=dict(context=list(range(KEEP)), anticipation_steps=8, first_target=6144, n_target=256, rebased=True),
        hardware_gate=dict(relative_l2_max=0.005, logit_max_abs=0.15),
        source_sha256={str(Path(p)): sha(Path(p)) for p in [__file__, P.__file__, L.__file__, T.__file__,
            Path(__file__).with_name("eval_stream_mtp_multi_strategy.py"),
            SHARE_VJEPA_ROOT / "src/models/utils/modules.py",
            SHARE_VJEPA_ROOT / "src/models/predictor.py"]},
        gpu=dict(name=torch.cuda.get_device_name(), memory=torch.cuda.get_device_properties(0).total_memory,
                 torch=torch.__version__, cuda=torch.version.cuda), model_load="P.build strict missing/unexpected zero")
    write_json(args.out / "metadata.json", metadata)
    write_json(args.out / "execution_order.json", [{k: v for k, v in s.items() if k != "row"} for s in samples])

    def forward(clip, identities):
        clip = clip.to(device).float().div_(255).sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
        ant = torch.full((len(identities),), 2.0, device=device)
        result = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = base.encoder(clip)
            assert encoded.shape[1] == TOTAL
            for arm in ARMS:
                idx = gpu_masks[arm].expand(len(identities), -1)
                kept = encoded.gather(1, idx.unsqueeze(-1).expand(-1, -1, encoded.shape[-1]))
                out = mtp(predict_from_encoded(base, kept, ant))
                result[arm] = np.stack([out[h]["action"].float().cpu().numpy() for h in HORIZONS], axis=1)
                assert result[arm].shape == (len(identities), 3, len(maps[2])) and np.isfinite(result[arm]).all()
        return result

    fixed_samples = manifest["samples"][:2]
    if args.reference:
        with np.load(args.reference) as reference:
            reference_ids = reference["sample_ids"].tolist()
        by_id = {s["sample_id"]: s for s in manifest["samples"]}
        assert len(reference_ids) == 2 and all(i in by_id for i in reference_ids)
        fixed_samples = [by_id[i] for i in reference_ids]
    fixed_ds = P.IdentifiedDataset(ma, fixed_samples)
    fixed_batch = P.collate([fixed_ds[0], fixed_ds[1]])
    fixed_logits = forward(fixed_batch["clip"], fixed_batch["samples"])
    anchor = {"sample_ids": np.asarray([s["sample_id"] for s in fixed_batch["samples"]])}
    for arm in ARMS:
        anchor["logits__" + arm] = fixed_logits[arm]
        anchor["indices__" + arm] = masks[arm]
    np.savez_compressed(args.out / "hardware_anchor.npz", **anchor)
    metadata["hardware_comparison"] = None
    if args.reference:
        with np.load(args.reference) as reference:
            metadata["hardware_comparison"] = compare_anchor(anchor, dict(reference))
    write_json(args.out / "metadata.json", metadata)

    output = (args.out / "predictions.jsonl").open("x")
    timings = (args.out / "timings.jsonl").open("x")
    started = previous = time.time(); cursor = 0
    for it, batch in enumerate(loader):
        t0 = time.time(); logits_by_arm = forward(batch["clip"], batch["samples"])
        torch.cuda.synchronize(); compute_end = time.time()
        valid_labels = []
        for bi in range(len(batch["samples"])):
            labels = {}
            for hi, horizon in enumerate(HORIZONS):
                h = f"{horizon:g}s"
                if float(batch["mtp_mask"][bi, hi]) <= 0.5:
                    labels[h] = dict(valid=False, reason="masked"); continue
                _, _, action, keep = T.map_labels(batch["mtp_verbs"][bi, hi:hi+1], batch["mtp_nouns"][bi, hi:hi+1], *maps, torch.device("cpu"))
                labels[h] = dict(valid=True, label=int(action[0])) if len(keep) else dict(valid=False, reason="out_of_training_vocabulary")
            valid_labels.append(labels)
        for bi, sample in enumerate(batch["samples"]):
            record = {k: sample[k] for k in ["sample_id", "selection_index", "source_index", "video_id", "participant_id"]}
            record["labels"], record["arms"] = valid_labels[bi], {}
            for arm, logits_batch in logits_by_arm.items():
                record["arms"][arm] = {}
                for hi, horizon in enumerate(HORIZONS):
                    h = f"{horizon:g}s"; x = torch.from_numpy(logits_batch[bi, hi]); top = x.topk(5).indices.tolist()
                    e = dict(top5_indices=top)
                    if valid_labels[bi][h]["valid"]:
                        y = valid_labels[bi][h]["label"]
                        e.update(label=y, top1=y == top[0], top3=y in top[:3], top5=y in top,
                                 ce=float(torch.nn.functional.cross_entropy(x[None], torch.tensor([y]))))
                    record["arms"][arm][h] = e
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
        np.savez_compressed(args.out / f"logits_{cursor:05d}.npz",
                            sample_ids=np.asarray([s["sample_id"] for s in batch["samples"]]), **logits_by_arm)
        cursor += len(batch["samples"]); output.flush()
        end = time.time(); timings.write(json.dumps(dict(iteration=it, start=t0, compute_end=compute_end, end=end,
            rows=len(batch["samples"]), data_wait=t0-previous, compute_seconds=compute_end-t0,
            seconds=end-t0, peak_cuda_bytes=torch.cuda.max_memory_allocated())) + "\n"); timings.flush()
        if it % 16 == 0 or cursor == len(samples): print(f"[eval] {cursor}/{len(samples)} compute={compute_end-t0:.3f}s data={t0-previous:.3f}s", flush=True)
        previous = end
    output.close(); timings.close()
    assert cursor == stop - args.start
    write_json(args.out / "summary.json", dict(**metadata, rows=cursor, seconds=time.time()-started,
                                               peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    print(json.dumps(dict(rows=cursor, seconds=time.time()-started, out=str(args.out))), flush=True)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="mode", required=True)
    pp = sub.add_parser("prepare"); pp.add_argument("--out", type=Path, required=True)
    ev = sub.add_parser("eval"); ev.add_argument("--manifest", type=Path, required=True); ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--tag", required=True); ev.add_argument("--start", type=int, default=0); ev.add_argument("--stop", type=int, default=32)
    ev.add_argument("--batch", type=int, default=4); ev.add_argument("--workers", type=int, default=4); ev.add_argument("--reference", type=Path)
    args = p.parse_args(); {"prepare": prepare, "eval": evaluate}[args.mode](args)


if __name__ == "__main__": main()
