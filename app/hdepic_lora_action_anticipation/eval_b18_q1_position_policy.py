#!/usr/bin/env python3
"""Paired packed/original predictor coordinates on immutable B18 masks (Slurm).

The checkpoint-compatible RoPE implementation is unchanged. Common translation
is measured, not assumed invariant: upstream preserves a frequency-repeat bug.
"""
from __future__ import annotations
from app.hdepic_lora_action_anticipation.share_paths import VJEPA_ROOT as SHARE_VJEPA_ROOT

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation import eval_b18_q1_target_crossslot as Q

P = Q.P
PROTOCOL = "b18-predictor-prune/egtea-ctx16-position-lag-v1"
SELECTORS = Q.ARMS[:-1]
POLICIES = ["packed", "packed_shift48", "original"]
ARMS = [f"{policy}__{selector}" for policy in POLICIES for selector in SELECTORS]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_records(path):
    records = {}
    with (path / "predictions.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            records[row["sample_id"]] = {"arms": {s: row["arms"][s] for s in SELECTORS},
                                         "label_validity": row["label_validity"]}
    return records


def fixed_masks(args, manifest, reference, device):
    """Build once from archived scores, then only load this exact mask artifact.

    Uses the frozen previous GPU selector and seeds, verifies every archived
    per-slot allocation, and hashes exact indices. Packed/original never select.
    """
    meta_path = args.mask_path.with_suffix(".meta.json")
    source_path = args.reference_dir / "token_scores.npy"
    source_sha = file_sha(source_path)
    if args.mask_path.exists():
        metadata = json.loads(meta_path.read_text())
        assert metadata["source_score_sha256"] == source_sha
        assert metadata["manifest_sha256"] == P.sha(args.manifest)
        assert metadata["evaluation_protocol"] == PROTOCOL
        assert metadata["selectors"] == SELECTORS
        assert metadata["selector_source_sha256"] == P.sha(Q.__file__)
        assert metadata["mask_sha256"] == file_sha(args.mask_path)
        assert metadata["boundary_tie_sha256"] == file_sha(metadata["boundary_tie_path"])
        loaded = np.load(args.mask_path, mmap_mode="r")
        assert loaded.shape == (4000, len(SELECTORS), 4096) and loaded.dtype == np.int32
        return loaded, metadata
    scores = np.load(source_path, mmap_mode="r")
    assert scores.shape == (4000, 2, 3, 16384)
    masks = np.empty((4000, len(SELECTORS), 4096), dtype=np.int32)
    boundary_ties = []
    started = time.time()
    for start in range(0, 4000, 4):
        samples = manifest["samples"][start:start + 4]
        sc = torch.from_numpy(np.array(scores[start:start + 4, 1])).to(device)
        with torch.inference_mode():
            selected = Q.selection({kind: sc[:, ki] for ki, kind in enumerate(Q.SCORE_KINDS)}, samples, device)
            for ki, selector in [(0, "hybrid_allquery_L11"), (1, "hybrid_target_L11")]:
                history = sc[:, ki, :52 * 256]
                cutoff = history.topk(1024, 1).values[:, -1]
                greater = (history > cutoff[:, None]).sum(1).cpu().tolist()
                equal = (history == cutoff[:, None]).sum(1).cpu().tolist()
                for bi, sample in enumerate(samples):
                    boundary_ties.append({"sample_id": sample["sample_id"], "selector": selector,
                        "cutoff": float(cutoff[bi]), "strictly_greater": greater[bi], "equal_to_cutoff": equal[bi],
                        "selected_at_cutoff": 1024 - greater[bi], "cross_boundary_tie": greater[bi] + equal[bi] > 1024})
        for si, selector in enumerate(SELECTORS):
            indices = selected[selector].cpu().numpy()
            masks[start:start + 4, si] = indices
            for bi, sample in enumerate(samples):
                counts = np.bincount(indices[bi] // 256, minlength=64)
                assert counts.tolist() == reference[sample["sample_id"]]["arms"][selector]["keep_per_slot"]
    args.mask_path.parent.mkdir(parents=True, exist_ok=True)
    # Single smoke builds, full run reuses; fail closed rather than overwrite.
    with args.mask_path.open("xb") as f:
        np.save(f, masks)
    tie_path = args.mask_path.with_suffix(".ties.jsonl")
    with tie_path.open("x") as f:
        for row in boundary_ties: f.write(json.dumps(row) + "\n")
    metadata = {"evaluation_protocol": PROTOCOL, "source_run": "17026550", "source_score_path": str(source_path),
        "source_score_sha256": source_sha, "source_metadata_sha256": P.sha(args.reference_dir / "metadata.json"),
        "manifest_sha256": P.sha(args.manifest), "selectors": SELECTORS, "random_seeds": list(P.RANDOM_SEEDS),
        "mask_sha256": file_sha(args.mask_path), "shape": list(masks.shape), "dtype": "int32",
        "selector_source_sha256": P.sha(Q.__file__), "creation_gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__, "boundary_tie_path": str(tie_path), "boundary_tie_sha256": file_sha(tie_path),
        "boundary_tie_cases": sum(x["cross_boundary_tie"] for x in boundary_ties),
        "reconstruction_caveat": "Original indices were not archived. Counts alone do not prove index identity when cutoff ties exist; reconstructed masks are frozen identically for all three policies.",
        "creation_job": os.environ.get("SLURM_JOB_ID"), "creation_seconds": time.time()-started,
        "archived_slot_count_checks": 4000 * len(SELECTORS)}
    with meta_path.open("x") as f:
        json.dump(metadata, f, indent=2)
    return masks, metadata


def explicit_predict(core, feats, context_positions, target_positions):
    assert core.num_steps == 1
    B, N, D = feats.shape
    assert context_positions.shape == (B, N) and target_positions.shape == (B, 256)
    assert bool((context_positions[:, 1:] > context_positions[:, :-1]).all())
    assert bool((target_positions[:, 0] > context_positions[:, -1]).all())
    embedding = int(core.encoder.embed_dim)
    context = feats[:, :, -embedding:] if D != embedding else feats
    pred = core.predictor(feats, masks_x=context_positions, masks_y=target_positions)
    pred = pred[0] if isinstance(pred, tuple) else pred
    pred = pred[:, :, -embedding:] if pred.shape[-1] != embedding else pred
    return torch.cat([context.clone(), pred], 1), pred


def difference(a, b):
    d = a.float() - b.float()
    return {"max_abs": float(d.abs().max()), "rms": float(d.square().mean().sqrt()),
            "relative_l2": float(d.norm() / a.float().norm().clamp_min(1e-12))}


def run_smoke_probes(core, mtp, x, recent_idx, ant, target_coordinates):
    B = x.shape[0]
    packed_context = torch.arange(4096, device=x.device).expand(B, -1)
    original_context = recent_idx
    assert torch.equal(original_context - packed_context, torch.full_like(original_context, 48 * 256))
    assert torch.equal(target_coordinates[1] - target_coordinates[0], torch.full_like(target_coordinates[0], 48 * 256))
    assert torch.equal(original_context % 256, packed_context % 256)
    recent_features = x.gather(1, recent_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        packed, _ = explicit_predict(core, recent_features, packed_context, target_coordinates[0].expand(B, -1))
        legacy = P.predict_from_encoded(core, recent_features, ant)
        assert torch.equal(packed, legacy), "same-device packed route differs from previous helper"
        full_context = torch.arange(16384, device=x.device).unsqueeze(0)
        full_explicit, _ = explicit_predict(core, x[:1], full_context, target_coordinates[1].unsqueeze(0))
        full_legacy = P.predict_from_encoded(core, x[:1], ant[:1])
        assert torch.equal(full_explicit, full_legacy), "original full-coordinate route differs from full helper"
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            pk, pred_pk = explicit_predict(core, recent_features.float(), packed_context, target_coordinates[0].expand(B, -1))
            ori, pred_ori = explicit_predict(core, recent_features.float(), original_context, target_coordinates[1].expand(B, -1))
            logits_pk, logits_ori = mtp(pk), mtp(ori)
            logits_a = torch.stack([logits_pk[h]["action"] for h in P.HORIZONS], 1)
            logits_b = torch.stack([logits_ori[h]["action"] for h in P.HORIZONS], 1)
            assert bool(torch.isfinite(logits_a).all() & torch.isfinite(logits_b).all())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
    return ({"same_device_packed_helper_exact": True, "original_full_helper_exact": True,
        "recent_translation_tokens": 12288, "translation_invariance_is_not_a_gate": True,
        "reason": "checkpoint-compatible upstream RoPE frequency-repeat expansion need not be translation invariant",
        "fp32_predictor_shift_difference": difference(pred_pk, pred_ori),
        "fp32_logits_shift_difference": difference(logits_a, logits_b),
        "tf32_restored": [torch.backends.cuda.matmul.allow_tf32 == old_matmul, torch.backends.cudnn.allow_tf32 == old_cudnn]},
        torch.stack([logits_a, logits_b], 1).cpu().numpy())


def main():
    p = argparse.ArgumentParser()
    for key in ["train-csv", "val-csv", "video-root", "checkpoint", "init-from-ckpt", "encoder-lora", "predictor-lora",
                "out-dir", "manifest", "reference-dir", "mask-path"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--tag", required=True); p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--start", type=int, default=0); p.add_argument("--stop", type=int, default=128)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--smoke-probes", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert args.start % 4 == args.stop % 4 == 0 and 0 <= args.start < args.stop <= 4000
    manifest = P.load_manifest(args.val_csv, args.manifest, 4000)
    samples = manifest["samples"][args.start:args.stop]
    reference = reference_records(args.reference_dir)
    ref_meta = json.loads((args.reference_dir / "metadata.json").read_text())
    ref_summary = json.loads((args.reference_dir / "summary.json").read_text())
    assert ref_summary["n_rows"] == 4000 and ref_meta["evaluation_protocol"] == Q.PROTOCOL
    assert ref_meta["score_kind_order"] == Q.SCORE_KINDS and ref_meta["score_block_order"] == [0, 11]
    ref_order = json.loads((args.reference_dir / "execution_order.json").read_text())
    assert [x["sample_id"] for x in manifest["samples"]] == [x["sample_id"] for x in ref_order]
    actual_paths = [str(x) for x in [args.checkpoint, args.init_from_ckpt, args.encoder_lora, args.predictor_lora]]
    assert actual_paths == ref_meta["checkpoint_paths"]
    for path, identity in ref_meta["checkpoint_file_identity"].items():
        stat = Path(path).stat(); assert (stat.st_size, stat.st_mtime_ns) == (identity["bytes"], identity["mtime_ns"])
    assert P.sha(args.manifest) == ref_meta["manifest_sha256"] and P.sha(args.train_csv) == ref_meta["train_csv_sha256"]
    device = torch.device("cuda"); torch.manual_seed(args.seed); np.random.seed(args.seed)
    masks, mask_meta = fixed_masks(args, manifest, reference, device)
    base, mtp, maps = P.build(args, device)
    assert base.num_steps == 1 and base.grid_size ** 2 == 256
    assert base.frames_per_second == 8 and base.tubelet_size == 2
    assert int(base.grid_size ** 2 * (base.num_output_frames // base.tubelet_size)) == 256
    target_coords = torch.stack([torch.arange(256, device=device) + 4096 + 2048,
                                 torch.arange(256, device=device) + 16384 + 2048,
                                 torch.arange(256, device=device) + 16384 + 2048])
    np.save(args.out_dir / "target_positions.npy", target_coords.cpu().numpy())
    np.save(args.out_dir / "packed_context_positions.npy", np.arange(4096, dtype=np.int64))
    np.save(args.out_dir / "packed_shift48_context_positions.npy", np.arange(4096, dtype=np.int64) + 12288)
    code_root = Path(__file__).resolve().parents[2]
    from src.models.utils import modules as imported_modules
    source_paths = [Path(__file__), Path(P.__file__), Path(Q.__file__), Path(P.T.__file__),
        code_root / "app/hdepic_lora_action_anticipation/eval_stream_mtp_multi_strategy.py",
        code_root / "app/hdepic_lora_action_anticipation/eval_stream_mtp_kvcache_prune.py",
        Path(imported_modules.__file__), SHARE_VJEPA_ROOT / "src/models/predictor.py",
        code_root / "scripts/egtea/run_b18_q1_position_policy.slurm"]
    metadata = {"evaluation_protocol": PROTOCOL, "run_tag": args.tag, "job_id": os.environ.get("SLURM_JOB_ID"),
        "metric_scope": "native", "eval_path": "EGTEA fixed4000; shared encoder; frozen masks; paired predictor coordinates",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "start": args.start, "stop": args.stop, "arms": ARMS, "selectors": SELECTORS, "policies": POLICIES,
        "manifest": str(args.manifest), "manifest_sha256": P.sha(args.manifest),
        "val_csv_sha256": P.sha(args.val_csv), "train_csv_sha256": P.sha(args.train_csv),
        "checkpoint_paths": actual_paths, "checkpoint_file_identity": ref_meta["checkpoint_file_identity"],
        "reference_dir": str(args.reference_dir), "mask_path": str(args.mask_path), "mask_metadata": mask_meta,
        "mask_sha256": mask_meta["mask_sha256"], "mask_axes": "full_manifest_execution_row,selector,kept_original_token_index",
        "original_context_positions": "exact stored mask indices; same kept features as packed",
        "target_positions_axes": "policy,target_token", "target_starts": [6144, 18432, 18432], "num_steps": 1,
        "action_logits_axes": "sample,policy,selector,horizon(2/4/6),action_class",
        "random_seeds": list(P.RANDOM_SEEDS), "source_sha256": {str(x): P.sha(x) for x in source_paths},
        "actual_imported_rope_module": str(Path(imported_modules.__file__).resolve()),
        "rope_compatibility": "upstream frequency-repeat bug preserved; FP32 common-shift difference is a measurement, not a numerical-equivalence gate",
        "gpu": {"name": torch.cuda.get_device_name(), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}}
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.out_dir / "execution_order.json").write_text(json.dumps([{k: v for k, v in x.items() if k != "row"} for x in samples], indent=2))
    ds = P.IdentifiedDataset(args, samples)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=args.workers, collate_fn=P.collate,
                        **({"prefetch_factor": 2} if args.workers else {}))
    logits_file = np.lib.format.open_memmap(args.out_dir / "action_logits.npy", mode="w+", dtype=np.float32,
        shape=(len(samples), len(POLICIES), len(SELECTORS), 3, len(maps[2])))
    records_file = (args.out_dir / "predictions.jsonl").open("x")
    totals = {a: defaultdict(float) for a in ARMS}
    historical = {s: defaultdict(float) for s in SELECTORS}
    probes = None; recent_exact_batches = 0; cursor = 0; started = previous_end = time.time()
    for it, batch in enumerate(loader):
        begin = time.time(); data_wait = begin - previous_end
        clips = batch["clip"].to(device).float().div_(255).sub_(P.T.IMAGENET_MEAN.to(device)).div_(P.T.IMAGENET_STD.to(device))
        B = len(batch["samples"]); ant = torch.full((B,), 2.0, device=device)
        batch_masks = torch.from_numpy(np.array(masks[args.start + cursor:args.start + cursor + B])).to(device).long()
        records = [{**s, "arms": {}, "label_validity": reference[s["sample_id"]]["label_validity"]} for s in batch["samples"]]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = base.encoder(clips); assert x.shape[1] == 16384
        if it == 0 and args.smoke_probes:
            probes, probe_logits = run_smoke_probes(base, mtp, x, batch_masks[:, 0], ant, target_coords)
            np.save(args.out_dir / "fp32_recent_probe_logits.npy", probe_logits)
            (args.out_dir / "smoke_probes.json").write_text(json.dumps(probes, indent=2))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for si, selector in enumerate(SELECTORS):
                indices = batch_masks[:, si]
                assert indices.shape == (B, 4096) and bool((indices[:, 1:] > indices[:, :-1]).all())
                kept = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                counts = torch.stack([torch.bincount(ix // 256, minlength=64) for ix in indices]).cpu().tolist()
                for pi, policy in enumerate(POLICIES):
                    packed_positions = torch.arange(4096, device=device).expand(B, -1)
                    positions = packed_positions if policy == "packed" else packed_positions + 12288 if policy == "packed_shift48" else indices
                    tgt = target_coords[pi].expand(B, -1)
                    assert torch.equal(positions, packed_positions if pi == 0 else packed_positions + 12288 if pi == 1 else batch_masks[:, si])
                    if pi > 0: assert torch.equal(positions[:, -3072:], indices[:, -3072:])
                    assert int(tgt[0, 0]) == (4096 if pi == 0 else 16384) + 2048
                    tokens, _ = explicit_predict(base, kept, positions, tgt)
                    output = mtp(tokens); arm = f"{policy}__{selector}"
                    if selector == "recent" and policy == "packed_shift48":
                        recent_shift_tokens = tokens.clone()
                        recent_shift_logits = {h: output[h]["action"].clone() for h in P.HORIZONS}
                    if selector == "recent" and policy == "original":
                        assert torch.equal(tokens, recent_shift_tokens)
                        assert all(torch.equal(output[h]["action"], recent_shift_logits[h]) for h in P.HORIZONS)
                        recent_exact_batches += 1
                    for bi in range(B):
                        records[bi]["arms"][arm] = {"keep_per_slot": counts[bi], "metrics": {},
                            "context_position_first": int(positions[bi, 0]), "context_position_last": int(positions[bi, -1]),
                            "target_position_first": int(tgt[bi, 0]), "target_position_last": int(tgt[bi, -1])}
                    for hi, h in enumerate(P.HORIZONS):
                        logits = output[h]["action"].float()
                        assert bool(torch.isfinite(logits).all()), "nonfinite formal action logits"
                        logits_file[cursor:cursor + B, pi, si, hi] = logits.cpu().numpy()
                        for bi, sample in enumerate(batch["samples"]):
                            old = reference[sample["sample_id"]]["arms"][selector]["metrics"].get(f"{h:g}s")
                            if old is None: continue
                            label = int(old["label"])
                            ce = float(torch.nn.functional.cross_entropy(logits[bi:bi + 1], torch.tensor([label], device=device)))
                            top5 = bool((logits[bi].topk(5).indices == label).any())
                            records[bi]["arms"][arm]["metrics"][f"{h:g}s"] = {"label": label, "top5": top5, "ce": ce}
                            totals[arm][f"n@{h:g}s"] += 1; totals[arm][f"correct@{h:g}s"] += top5; totals[arm][f"ce_sum@{h:g}s"] += ce
                            if pi == 0:
                                historical[selector][f"n@{h:g}s"] += 1
                                historical[selector][f"top5_disagreements@{h:g}s"] += top5 != old["top5"]
                                historical[selector][f"ce_max_abs@{h:g}s"] = max(historical[selector][f"ce_max_abs@{h:g}s"], abs(ce-old["ce"]))
            for row in records: records_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            records_file.flush(); logits_file.flush(); cursor += B
        torch.cuda.synchronize(); previous_end = time.time()
        print(f"itr={it+1}/{len(loader)} n={cursor} data_wait={data_wait:.2f}s compute_wall={previous_end-begin:.2f}s elapsed={previous_end-started:.1f}s", flush=True)
    records_file.close(); assert cursor == len(samples)
    assert file_sha(args.mask_path) == mask_meta["mask_sha256"]
    assert all(P.sha(path) == value for path, value in metadata["source_sha256"].items())
    summary = {**metadata, "n_rows": cursor, "seconds": time.time()-started, "results": totals,
               "historical_packed_comparison": historical, "smoke_probes": probes,
               "recent_original_shift48_exact_batches": recent_exact_batches}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps({"rows": cursor, "seconds": summary["seconds"], "output": str(args.out_dir)}), flush=True)


if __name__ == "__main__": main()
