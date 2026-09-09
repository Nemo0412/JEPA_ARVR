#!/usr/bin/env python3
"""B18 same-slot online/offline score swaps; execute only in Slurm/container.

Selection uses a new same-device legacy attention readout. Four disjoint
post-softmax source sums explain current validation scores, not the training
calibration map. All main swap predictions use original token coordinates.
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

from app.hdepic_lora_action_anticipation import eval_b18_q1_paired_hybrid as P
from app.hdepic_lora_action_anticipation import eval_b18_q1_position_policy as R

PROTOCOL = "b18-predictor-prune/egtea-ctx16-score-swaps-v1"
BLOCKS = [0, 11]
SOURCES = ["self", "same_slot_other", "cross_slot_context", "target_query"]
VARIANTS = ["offline", "online", "near", "strong"] + [f"random_seed{s}" for s in P.RANDOM_SEEDS]
ARMS = [f"original__L{b}__{v}" for b in BLOCKS for v in VARIANTS]
ARMS += [f"packed__L{b}__{v}" for b in BLOCKS for v in ["offline", "online"]]
ARMS += ["original__recent"]


class SourceCapture:
    """Return the unmodified attention output and capture disjoint readouts."""
    def __init__(self, module, n_context=16384, gp=256):
        from src.models.utils.modules import rotate_queries_or_keys
        self.module, self.original = module, module.forward
        cap = self

        def forward(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            if attn_mask is not None or module.is_causal or module.proj_drop_prob != 0:
                raise RuntimeError("legacy readout requires unmasked noncausal attention without dropout")
            B, N, _ = x.shape
            H = module.num_heads
            assert N == n_context + gp and mask is not None
            assert torch.equal(mask[:, :n_context], torch.arange(n_context, device=x.device).expand(B, -1))
            assert torch.equal(mask[:, n_context:], (torch.arange(gp, device=x.device) + 18432).expand(B, -1))
            output = cap.original(x, mask=mask, attn_mask=attn_mask, T=T,
                                  H_patches=H_patches, W_patches=W_patches)
            qkv = module.qkv(x).unflatten(-1, (3, H, -1)).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            positions = module.separate_positions(mask.unsqueeze(1).repeat(1, H, 1), H_patches, W_patches)
            qs, ks, start = [], [], 0
            for dim, pos in zip([module.d_dim, module.h_dim, module.w_dim], positions):
                qs.append(rotate_queries_or_keys(q[..., start:start + dim], pos=pos))
                ks.append(rotate_queries_or_keys(k[..., start:start + dim], pos=pos))
                start += dim
            if start < module.head_dim:
                qs.append(q[..., start:]); ks.append(k[..., start:])
            q, k = torch.cat(qs, -1), torch.cat(ks, -1)
            legacy = torch.zeros(B, H, n_context, device=x.device)
            total = torch.zeros_like(legacy)
            components = torch.zeros(B, 4, H, n_context, device=x.device)
            for ci in range(0, N, gp):
                logits = (q[:, :, ci:ci + gp] @ k.transpose(-2, -1)) * module.scale
                prob = logits.softmax(-1)
                legacy_received = prob.sum(2)
                legacy += legacy_received.float()[:, :, :n_context]
                received = prob.sum(2, dtype=torch.float32)[:, :, :n_context]
                if ci == 0:
                    cap.observed_dtypes = {"input": str(x.dtype), "qkv": str(qkv.dtype),
                        "rotated_q": str(q.dtype), "rotated_k": str(k.dtype), "logits": str(logits.dtype),
                        "prob": str(prob.dtype), "legacy_query_reduction": str(legacy_received.dtype),
                        "fp32_query_reduction": str(received.dtype), "legacy_accumulator": str(legacy.dtype),
                        "fp32_accumulator": str(total.dtype)}
                total += received
                if ci == n_context:
                    components[:, 3] += received
                else:
                    same = prob[:, :, :, ci:ci + gp].float()
                    diagonal = same.diagonal(dim1=-2, dim2=-1)
                    components[:, 0, :, ci:ci + gp] = diagonal.clone()
                    diagonal.zero_()
                    components[:, 1, :, ci:ci + gp] = same.sum(2)
                    # Columns in own key slot are excluded, avoiding subtraction
                    # from a potentially self-dominated accumulated score.
                    received[:, :, ci:ci + gp] = 0
                    components[:, 2] += received
            cap.legacy = legacy.sum(1)
            cap.total = total.sum(1)
            cap.components = components.sum(2)
            cap.head_temporal = components.reshape(B, 4, H, 64, gp).sum(-1)
            assert bool(torch.isfinite(cap.components).all() & (cap.components >= 0).all())
            assert torch.allclose(cap.components.sum(1), cap.total, rtol=3e-6, atol=3e-5), "source conservation"
            return output

        module.forward = forward

    def remove(self):
        self.module.forward = self.original


def ordered(score, indices=None):
    """Descending score, ascending original index at ties; no label input."""
    if indices is None:
        indices = torch.arange(score.numel(), device=score.device)
    indices = indices.sort().values
    return indices[torch.argsort(score[indices], descending=True, stable=True)]


def make_masks(scores, offline, samples):
    B = len(samples)
    masks = {a: [] for a in ARMS}
    quotas, capacities, cutoffs = [], [], []
    for bi, sample in enumerate(samples):
        sample_q, sample_cap, sample_cut = [], [], []
        for li, block in enumerate(BLOCKS):
            score = scores[bi, li]
            f = ordered(offline[li])[:4096].sort().values
            o = ordered(score)[:4096].sort().values
            fm = torch.zeros(16384, dtype=torch.bool, device=score.device); fm[f] = True
            om = torch.zeros_like(fm); om[o] = True
            incoming = (om & ~fm).nonzero().flatten()
            outgoing = (fm & ~om).nonzero().flatten()
            assert len(incoming) == len(outgoing)
            chosen_in = {v: [] for v in VARIANTS[2:]}
            chosen_out = {v: [] for v in VARIANTS[2:]}
            rngs = {s: np.random.default_rng(int(hashlib.sha256(
                f"score-swaps|{s}|{block}|{sample['sample_id']}".encode()).hexdigest()[:16], 16)) for s in P.RANDOM_SEEDS}
            qq, cc = [], []
            for slot in range(64):
                ii = ordered(score, incoming[incoming // 256 == slot])
                ee = ordered(score, outgoing[outgoing // 256 == slot])
                capacity = min(len(ii), len(ee)); q = capacity // 2
                qq.append(q); cc.append(capacity)
                if not q: continue
                chosen_in["near"].append(ii[-q:]); chosen_out["near"].append(ee[:q])
                chosen_in["strong"].append(ii[:q]); chosen_out["strong"].append(ee[-q:])
                assert not bool(torch.isin(ii[:q], ii[-q:]).any())
                assert not bool(torch.isin(ee[:q], ee[-q:]).any())
                for seed in P.RANDOM_SEEDS:
                    name = f"random_seed{seed}"
                    chosen_in[name].append(ii[torch.as_tensor(rngs[seed].choice(len(ii), q, replace=False), device=score.device)])
                    chosen_out[name].append(ee[torch.as_tensor(rngs[seed].choice(len(ee), q, replace=False), device=score.device)])
            local = {"offline": f, "online": o}
            for variant in VARIANTS[2:]:
                added = torch.cat(chosen_in[variant]) if chosen_in[variant] else incoming[:0]
                removed = torch.cat(chosen_out[variant]) if chosen_out[variant] else outgoing[:0]
                keep = fm.clone(); keep[removed] = False; keep[added] = True
                index = keep.nonzero().flatten()
                assert len(added) == len(removed) == sum(qq) and len(index) == 4096
                assert torch.equal(torch.bincount(index // 256, minlength=64), torch.bincount(f // 256, minlength=64))
                assert bool(om[added].all() & (~fm[added]).all() & fm[removed].all() & (~om[removed]).all())
                local[variant] = index
            for variant, index in local.items():
                masks[f"original__L{block}__{variant}"].append(index)
                if variant in ["offline", "online"]:
                    masks[f"packed__L{block}__{variant}"].append(index)
            cuts = []
            for vector, index in [(score, o), (offline[li], f)]:
                cutoff = vector[index].min()
                greater, equal = int((vector > cutoff).sum()), int((vector == cutoff).sum())
                cuts.append({"value": float(cutoff), "greater": greater, "equal": equal,
                             "selected_equal": 4096 - greater, "cross_boundary_tie": greater < 4096 < greater + equal})
            sample_q.append(qq); sample_cap.append(cc); sample_cut.append(cuts)
        masks["original__recent"].append(torch.arange(12288, 16384, device=scores.device))
        quotas.append(sample_q); capacities.append(sample_cap); cutoffs.append(sample_cut)
    result = {a: torch.stack(masks[a]) for a in ARMS}
    for index in result.values():
        assert index.shape == (B, 4096) and bool((index[:, 1:] > index[:, :-1]).all())
        assert int(index.min()) >= 0 and int(index.max()) < 16384
    return result, np.asarray(quotas, dtype=np.int16), np.asarray(capacities, dtype=np.int16), cutoffs


def main():
    p = argparse.ArgumentParser()
    for key in ["train-csv", "val-csv", "video-root", "checkpoint", "init-from-ckpt", "encoder-lora",
                "predictor-lora", "calib-l0", "calib-l11", "out-dir", "manifest", "reference-dir"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--tag", required=True); p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--start", type=int, default=0); p.add_argument("--stop", type=int, default=128)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--parity-check", action="store_true")
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    assert args.start % 4 == args.stop % 4 == 0 and 0 <= args.start < args.stop <= 4000
    manifest = P.load_manifest(args.val_csv, args.manifest, 4000)
    samples = manifest["samples"][args.start:args.stop]
    ref_meta = json.loads((args.reference_dir / "metadata.json").read_text())
    ref_order = json.loads((args.reference_dir / "execution_order.json").read_text())
    assert ref_meta["evaluation_protocol"] == P.PROTOCOL
    assert [s["sample_id"] for s in manifest["samples"]] == [s["sample_id"] for s in ref_order]
    assert P.sha(args.manifest) == ref_meta["manifest_sha256"]
    actual_paths = [str(x) for x in [args.checkpoint, args.init_from_ckpt, args.encoder_lora, args.predictor_lora]]
    assert actual_paths == ref_meta["checkpoint_paths"]
    for path, ident in ref_meta["checkpoint_file_identity"].items():
        stat = Path(path).stat(); assert (stat.st_size, stat.st_mtime_ns) == (ident["bytes"], ident["mtime_ns"])
    reference = {}
    for line in (args.reference_dir / "predictions.jsonl").read_text().splitlines():
        record = json.loads(line); reference[record["sample_id"]] = record
    old_scores = np.load(args.reference_dir / "scores.npy", mmap_mode="r")
    assert old_scores.shape == (4000, 2, 16384)
    device = torch.device("cuda"); torch.manual_seed(args.seed); np.random.seed(args.seed)
    base, mtp, maps = P.build(args, device)
    assert base.num_steps == 1 and base.grid_size ** 2 == 256 and base.frames_per_second == 8 and base.tubelet_size == 2
    assert base.grid_size ** 2 * (base.num_output_frames // base.tubelet_size) == 256
    offline = torch.stack([torch.from_numpy(np.load(x)).flatten() for x in [args.calib_l0, args.calib_l11]]).to(device)
    assert offline.shape == (2, 16384) and bool(torch.isfinite(offline).all() & (offline >= 0).all())
    calib_paths = [args.calib_l0, args.calib_l11]
    for block, path in zip(BLOCKS, calib_paths):
        calmeta = json.loads(path.with_name(path.name.replace("_map_64x256.npy", "_meta.json")).read_text())
        assert calmeta["block"] == block and calmeta["n_used"] == 512 and calmeta["calib_csv"] == str(args.train_csv)
        assert P.sha(path) == ref_meta["calibration_sha256"][str(path)]
    from src.models.utils import modules as imported_modules
    root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__), Path(P.__file__), Path(R.__file__), Path(R.Q.__file__), Path(P.T.__file__),
        root / "app/hdepic_lora_action_anticipation/eval_stream_mtp_multi_strategy.py",
        root / "app/hdepic_lora_action_anticipation/eval_stream_mtp_kvcache_prune.py",
        Path(imported_modules.__file__), SHARE_VJEPA_ROOT / "src/models/predictor.py", root / "scripts/egtea/run_b18_q1_score_swaps.slurm"]
    metadata = {"evaluation_protocol": PROTOCOL, "sample_manifest_protocol": P.PROTOCOL,
        "run_tag": args.tag, "job_id": os.environ.get("SLURM_JOB_ID"), "metric_scope": "native",
        "eval_path": "EGTEA fixed4000; original-coordinate same-slot score swaps; native communicating MTP",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "start": args.start, "stop": args.stop, "arms": ARMS, "blocks": BLOCKS, "sources": SOURCES,
        "primary_block": 11, "primary_horizon": 2, "primary_contrasts": ["strong_minus_near", "strong_minus_offline"],
        "manifest": str(args.manifest), "manifest_sha256": P.sha(args.manifest),
        "val_csv_sha256": P.sha(args.val_csv), "train_csv_sha256": P.sha(args.train_csv),
        "checkpoint_paths": actual_paths, "checkpoint_file_identity": ref_meta["checkpoint_file_identity"],
        "calibration_sha256": {str(x): P.sha(x) for x in calib_paths}, "calibration_n": 512,
        "reference_dir": str(args.reference_dir), "reference_score_sha256": R.file_sha(args.reference_dir / "scores.npy"),
        "source_sha256": {str(x): P.sha(x) for x in source_paths}, "random_seeds": list(P.RANDOM_SEEDS),
        "mask_axes": "sample,arm,sorted_original_context_token_index", "mask_dtype": "int32",
        "legacy_online_scores_axes": "sample,block,context_key_token", "source_components_fp32_axes": "sample,block,source,context_key_token",
        "source_total_fp32_axes": "sample,block,context_key_token", "source_head_temporal_axes": "sample,block,source,head,key_slot",
        "swap_quotas_axes": "sample,block,slot", "action_logits_axes": "sample,arm,horizon(2/4/6),action_class",
        "token_layout": {"n_context_tokens": 16384, "n_target_tokens": 256, "tokens_per_slot": 256,
            "target_array_start_slot": 64, "original_target_rope_start_slot": 72, "packed_target_rope_start_slot": 24,
            "num_heads": {str(b): base.predictor.predictor_blocks[b].attn.num_heads for b in BLOCKS}},
        "selection": {"keep": 4096, "anchor_slots": 0, "tie_rule": "stable descending score, original index ascending",
            "quota": "floor(min(incoming_count_slot,outgoing_count_slot)/2)",
            "near": "lowest-score incoming and highest-score outgoing in eligible same-slot disagreement pools; not necessarily near global cutoff",
            "strong": "highest-score incoming and lowest-score outgoing, same per-slot quota",
            "random_seed_derivation": "sha256(score-swaps|seed|block|sample_id) first16hex; numpy default_rng per sample/block/seed; slot order0..63, incoming then outgoing draw"},
        "readout": "BF16 autocast forward; legacy softmax/default query reduction then float32 chunk/head accumulation; four disjoint post-softmax sources sum in float32; actual intermediate dtypes recorded in summary, no normalization or attention intervention",
        "source_scope": "current validation receiver; does not decompose training-calibration offline map",
        "actual_imported_rope_module": str(Path(imported_modules.__file__).resolve()),
        "gpu": {"name": torch.cuda.get_device_name(), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "torch_version": torch.__version__}
    assert metadata["train_csv_sha256"] == ref_meta["train_csv_sha256"]
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.out_dir / "execution_order.json").write_text(json.dumps([{k: v for k, v in s.items() if k != "row"} for s in samples], indent=2))
    np.save(args.out_dir / "offline_scores.npy", offline.cpu().numpy())
    np.save(args.out_dir / "target_positions.npy", np.stack([np.arange(256) + 18432, np.arange(256) + 6144]))
    shape_specs = {"legacy_online_scores": (2, 16384), "source_total_fp32": (2, 16384),
        "source_components_fp32": (2, 4, 16384), "mask_indices": (len(ARMS), 4096),
        "swap_quotas": (2, 64), "swap_capacities": (2, 64), "action_logits": (len(ARMS), 3, len(maps[2]))}
    arrays = {key: np.lib.format.open_memmap(args.out_dir / f"{key}.npy", mode="w+",
        dtype=np.int32 if key == "mask_indices" else np.int16 if key.startswith("swap_") else np.float32,
        shape=(len(samples), *shape)) for key, shape in shape_specs.items()}
    rows_file = (args.out_dir / "predictions.jsonl").open("x")
    loader = DataLoader(P.IdentifiedDataset(args, samples), batch_size=4, shuffle=False, num_workers=args.workers,
        collate_fn=P.collate, **({"prefetch_factor": 2} if args.workers else {}))
    totals = {a: defaultdict(float) for a in ARMS}; historical = defaultdict(lambda: defaultdict(float))
    head_temporal = []; observed_dtypes = None; gates = {"capture_output_exact": None, "legacy_score_exact": None, "packed_helper_exact": None,
        "source_conservation_max_abs": 0.0, "legacy_fp32_max_abs": 0.0}
    cursor = 0; started = previous_end = time.time()
    for it, batch in enumerate(loader):
        begin = time.time(); data_wait = begin - previous_end
        clip = batch["clip"].to(device).float().div_(255).sub_(P.T.IMAGENET_MEAN.to(device)).div_(P.T.IMAGENET_STD.to(device))
        B = len(batch["samples"]); ant = torch.full((B,), 2.0, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = base.encoder(clip); assert x.shape[1] == 16384
            context = torch.arange(16384, device=device).expand(B, -1)
            target = (torch.arange(256, device=device) + 18432).expand(B, -1)
            caps = {b: SourceCapture(base.predictor.predictor_blocks[b].attn) for b in BLOCKS}
            try: captured = base.predictor(x, masks_x=context, masks_y=target)
            finally:
                for cap in caps.values(): cap.remove()
            scores = torch.stack([caps[b].legacy for b in BLOCKS], 1)
            components = torch.stack([caps[b].components for b in BLOCKS], 1)
            total = torch.stack([caps[b].total for b in BLOCKS], 1)
            current_dtypes = {str(b): caps[b].observed_dtypes for b in BLOCKS}
            if observed_dtypes is None: observed_dtypes = current_dtypes
            assert observed_dtypes == current_dtypes, "capture dtype changed between batches"
            if it == 0 and args.parity_check:
                old_caps = {b: P.QueryGroupCapture(base.predictor.predictor_blocks[b].attn, 16384) for b in BLOCKS}
                try: unchanged = base.predictor(x, masks_x=context, masks_y=target)
                finally:
                    for cap in old_caps.values(): cap.remove()
                assert torch.equal(captured, unchanged)
                old = torch.stack([old_caps[b].importance[:, :, :16384].sum(1) for b in BLOCKS], 1)
                assert torch.equal(scores, old)
                gates["capture_output_exact"] = gates["legacy_score_exact"] = True
            gates["source_conservation_max_abs"] = max(gates["source_conservation_max_abs"], float((components.sum(2) - total).abs().max()))
            gates["legacy_fp32_max_abs"] = max(gates["legacy_fp32_max_abs"], float((scores - total).abs().max()))
            indices, quota, capacity, cutoffs = make_masks(scores, offline, batch["samples"])
            arrays["legacy_online_scores"][cursor:cursor+B] = scores.cpu().numpy()
            arrays["source_components_fp32"][cursor:cursor+B] = components.cpu().numpy()
            arrays["source_total_fp32"][cursor:cursor+B] = total.cpu().numpy()
            arrays["mask_indices"][cursor:cursor+B] = torch.stack([indices[a] for a in ARMS], 1).cpu().numpy()
            arrays["swap_quotas"][cursor:cursor+B] = quota; arrays["swap_capacities"][cursor:cursor+B] = capacity
            head_temporal.append(torch.stack([caps[b].head_temporal for b in BLOCKS], 1).cpu().numpy())
            bridge_score = torch.from_numpy(np.array(old_scores[args.start+cursor:args.start+cursor+B])).to(device)
            historical["legacy_scores"]["max_abs"] = max(historical["legacy_scores"]["max_abs"], float((scores-bridge_score).abs().max()))
            rows = [{**s, "arms": {}, "label_validity": {}, "cutoffs_online_offline": cutoffs[i],
                     "swap_quotas": quota[i].tolist(), "swap_capacities": capacity[i].tolist()} for i, s in enumerate(batch["samples"])]
            labels_by_horizon = {}
            for hi, h in enumerate(P.HORIZONS):
                valid = batch["mtp_mask"][:, hi].to(device) > .5
                _, _, labels, keep = P.T.map_labels(batch["mtp_verbs"][:, hi][valid.cpu()].to(device),
                    batch["mtp_nouns"][:, hi][valid.cpu()].to(device), *maps, device)
                vp = valid.nonzero().flatten()[keep]; labels_by_horizon[h] = vp, labels
                valid_set = set(vp.cpu().tolist())
                for bi in range(B): rows[bi]["label_validity"][f"{h:g}s"] = "valid" if bi in valid_set else "masked" if not bool(valid[bi]) else "out_of_training_vocabulary"
            for ai, arm in enumerate(ARMS):
                index = indices[arm]; kept = x.gather(1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                packed = arm.startswith("packed")
                pos = torch.arange(4096, device=device).expand(B, -1) if packed else index
                tgt = (torch.arange(256, device=device) + (6144 if packed else 18432)).expand(B, -1)
                tokens, _ = R.explicit_predict(base, kept, pos, tgt)
                if it == 0 and args.parity_check and arm == "packed__L0__offline":
                    assert torch.equal(tokens, P.predict_from_encoded(base, kept, ant)); gates["packed_helper_exact"] = True
                outputs = mtp(tokens)
                counts = torch.stack([torch.bincount(z // 256, minlength=64) for z in index]).cpu().tolist()
                for bi in range(B): rows[bi]["arms"][arm] = {"metrics": {}, "keep_per_slot": counts[bi],
                    "context_position_first": int(pos[bi, 0]), "context_position_last": int(pos[bi, -1]),
                    "target_position_first": int(tgt[bi, 0]), "target_position_last": int(tgt[bi, -1])}
                for hi, h in enumerate(P.HORIZONS):
                    logits = outputs[h]["action"].float(); assert bool(torch.isfinite(logits).all())
                    arrays["action_logits"][cursor:cursor+B, ai, hi] = logits.cpu().numpy()
                    vp, labels = labels_by_horizon[h]
                    if not len(vp): continue
                    correct = (logits[vp].topk(5, -1).indices == labels[:, None]).any(-1)
                    losses = torch.nn.functional.cross_entropy(logits[vp], labels, reduction="none")
                    for bi, label, right, ce in zip(vp.cpu().tolist(), labels.cpu().tolist(), correct.cpu().tolist(), losses.cpu().tolist()):
                        rows[bi]["arms"][arm]["metrics"][f"{h:g}s"] = {"label": label, "top5": right, "ce": ce}
                        totals[arm][f"n@{h:g}s"] += 1; totals[arm][f"correct@{h:g}s"] += right; totals[arm][f"ce_sum@{h:g}s"] += ce
                        if packed:
                            _, layer, variant = arm.split("__"); old_arm = f"{variant}_{layer}"
                            prior = reference[rows[bi]["sample_id"]]["arms"][old_arm]["metrics"][f"{h:g}s"]
                            assert label == prior["label"]
                            historical[arm][f"n@{h:g}s"] += 1
                            historical[arm][f"top5_disagreements@{h:g}s"] += right != prior["top5"]
                            historical[arm][f"ce_max_abs@{h:g}s"] = max(historical[arm][f"ce_max_abs@{h:g}s"], abs(ce-prior["ce"]))
        for row in rows: rows_file.write(json.dumps(row, separators=(",", ":")) + "\n")
        rows_file.flush()
        for array in arrays.values(): array.flush()
        cursor += B; torch.cuda.synchronize(); previous_end = time.time()
        print(f"itr={it+1}/{len(loader)} n={cursor} data_wait={data_wait:.2f}s compute_wall={previous_end-begin:.2f}s elapsed={previous_end-started:.1f}s", flush=True)
    rows_file.close(); assert cursor == len(samples)
    np.save(args.out_dir / "source_head_temporal.npy", np.concatenate(head_temporal))
    assert all(P.sha(path) == value for path, value in metadata["source_sha256"].items())
    assert all(P.sha(path) == value for path, value in metadata["calibration_sha256"].items())
    hashes = {name: R.file_sha(args.out_dir / f"{name}.npy") for name in [*arrays, "offline_scores", "source_head_temporal", "target_positions"]}
    summary = {**metadata, "n_rows": cursor, "results": totals, "gates": gates,
               "observed_capture_dtypes": observed_dtypes,
               "historical_bridge": historical, "artifact_sha256": hashes, "seconds": time.time()-started}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps({"rows": cursor, "seconds": summary["seconds"], "output": str(args.out_dir)}), flush=True)


if __name__ == "__main__": main()
