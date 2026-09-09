#!/usr/bin/env python3
"""B18 target-query selection and same-slot readout diagnostics (Slurm only).

All model attention outputs remain unchanged. Diagnostic deletion is after
softmax and removes context-query -> same-time-slot context-key contributions.
Target queries/keys are never confused with the latest observed context slot.
"""
from __future__ import annotations
from app.hdepic_lora_action_anticipation.share_paths import VJEPA_ROOT as SHARE_VJEPA_ROOT

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation import eval_b18_q1_paired_hybrid as P

PROTOCOL = "b18-predictor-prune/egtea-ctx16-target-crossslot-v1"
ARMS = ["recent", "hybrid_allquery_L11", "hybrid_target_L11"]
ARMS += [f"hybrid_target_random_seed{s}" for s in P.RANDOM_SEEDS]
ARMS += ["hybrid_target_donor"]
SCORE_KINDS = ["all_query", "target_query", "context_offslot"]
DIAGNOSTIC_KINDS = ["context_original", "context_offslot", "allquery_original", "allquery_residual", "targetquery"]
NEAR_ZERO = 1e-8


class CrossSlotCapture:
    """Chunked legacy scores plus exact per-token-query conditional readout.

    All/target received scores retain the legacy BF16 softmax/sum convention.
    Conditional readout uses float32 probabilities and explicitly measured
    surviving row sums, including target keys. Nothing changes model outputs.
    """
    def __init__(self, module, n_context=16384, gp=256):
        from src.models.utils.modules import rotate_queries_or_keys
        self.module, self.original = module, module.forward
        cap = self

        def forward(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            if attn_mask is not None or module.is_causal or module.proj_drop_prob != 0:
                raise RuntimeError("score parity requires unmasked noncausal attention without dropout")
            B, N, _ = x.shape
            H, S, C = module.num_heads, N // gp, n_context // gp
            assert N % gp == 0 and n_context % gp == 0
            assert torch.equal(mask[:, :n_context], torch.arange(n_context, device=x.device).expand(B, -1))
            assert bool((mask[:, n_context:] >= n_context).all())
            out = cap.original(x, mask=mask, attn_mask=attn_mask, T=T,
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
            total = torch.zeros(B, H, N, device=x.device)
            target = torch.zeros_like(total)
            offslot = torch.zeros_like(total)
            raw_group = torch.zeros(B, 2, H, S, device=x.device)
            diagonal = torch.zeros(B, H, C, device=x.device)
            diagonal_fp32 = torch.zeros(B, H, C, device=x.device)
            self_token = torch.zeros(B, H, C, device=x.device)
            conditional = torch.zeros(B, H, S, device=x.device)
            denominator = {key: torch.zeros(B, H, C, device=x.device) for key in
                           ["survival_mean", "survival_min", "survival_zero", "survival_nearzero",
                            "original_row_sum_mean", "original_row_sum_min"]}
            flows = {key: torch.zeros(H, S if key == "raw" else C, S, device=x.device)
                     for key in ["raw", "offslot", "conditional"]}
            for ci in range(0, N, gp):
                logits = (q[:, :, ci:ci + gp] @ k.transpose(-2, -1)) * module.scale
                prob = logits.softmax(-1)
                received = prob.sum(2).float()  # exact legacy dtype and reduction
                total += received
                temporal = received.reshape(B, H, S, gp).sum(-1)
                raw_group[:, int(ci >= n_context)] += temporal
                flows["raw"][:, ci // gp] = temporal.sum(0) / gp
                if ci >= n_context:
                    target += received
                    continue
                slot = ci // gp
                diagonal[:, :, slot] = received[:, :, ci:ci + gp].sum(-1)
                diagonal_fp32[:, :, slot] = prob[:, :, :, ci:ci + gp].float().sum((-1, -2))
                self_token[:, :, slot] = prob[:, :, :, ci:ci + gp].float().diagonal(dim1=-2, dim2=-1).sum(-1)
                # Whole chunk consists of queries from this same source slot.
                # Zeroing its key-slot columns in received is exactly equivalent
                # to deleting this block of post-softmax probabilities before sum.
                removed = received.clone()
                removed[:, :, ci:ci + gp] = 0
                offslot += removed
                flows["offslot"][:, slot] = removed.reshape(B, H, S, gp).sum(-1).sum(0) / gp
                pr = prob.float()
                original_sum = pr.sum(-1)
                pr[:, :, :, ci:ci + gp] = 0
                survival = pr.sum(-1)
                denominator["survival_mean"][:, :, slot] = survival.mean(-1)
                denominator["survival_min"][:, :, slot] = survival.min(-1).values
                denominator["survival_zero"][:, :, slot] = (survival == 0).sum(-1)
                denominator["survival_nearzero"][:, :, slot] = (survival <= NEAR_ZERO).sum(-1)
                denominator["original_row_sum_mean"][:, :, slot] = original_sum.mean(-1)
                denominator["original_row_sum_min"][:, :, slot] = original_sum.min(-1).values
                pr /= torch.where(survival > 0, survival, torch.ones_like(survival)).unsqueeze(-1)
                cond = pr.sum(2).reshape(B, H, S, gp).sum(-1)
                conditional += cond
                flows["conditional"][:, slot] = cond.sum(0) / gp
            cap.token_scores = torch.stack([total[:, :, :n_context], target[:, :, :n_context],
                                             offslot[:, :, :n_context]], 1)
            cap.raw_group = raw_group
            cap.offslot_context = offslot.reshape(B, H, S, gp).sum(-1)
            cap.conditional_context = conditional
            cap.diagonal_context = diagonal
            cap.diagonal_context_fp32 = diagonal_fp32
            cap.self_token_context = self_token
            cap.denominator = denominator
            cap.flows = flows
            return out

        module.forward = forward

    def remove(self):
        self.module.forward = self.original


def selection(scores, samples, device):
    B, N = scores["all_query"].shape
    recent = torch.arange(N - 4096, N, device=device).expand(B, -1)
    anchor = torch.arange(52 * 256, N, device=device).expand(B, -1)
    result = {"recent": recent}
    for kind, arm in [("all_query", "hybrid_allquery_L11"), ("target_query", "hybrid_target_L11")]:
        history = scores[kind][:, :52 * 256].topk(1024, 1).indices
        result[arm] = torch.cat([history, anchor], 1).sort(1).values
    ref = result["hybrid_target_L11"]
    counts = torch.stack([torch.bincount(x // 256, minlength=64) for x in ref]).cpu()
    for seed in P.RANDOM_SEEDS:
        randomized = []
        for bi, sample in enumerate(samples):
            rng_seed = int(P.hashlib.sha256(f"{seed}|{sample['sample_id']}".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(rng_seed)
            history = [rng.choice(256, int(counts[bi, slot]), replace=False) + slot * 256 for slot in range(52)]
            randomized.append(torch.cat([torch.as_tensor(np.concatenate(history), device=device), anchor[bi]]).sort().values)
        result[f"hybrid_target_random_seed{seed}"] = torch.stack(randomized)
    donated = []
    assert B % 2 == 0
    for bi, sample in enumerate(samples):
        donor = bi ^ 1
        assert sample["donor_sample_id"] == samples[donor]["sample_id"]
        assert sample["video_id"] != samples[donor]["video_id"]
        history = [scores["target_query"][donor, slot * 256:(slot + 1) * 256].topk(int(counts[bi, slot])).indices + slot * 256
                   for slot in range(52)]
        donated.append(torch.cat([*history, anchor[bi]]).sort().values)
    result["hybrid_target_donor"] = torch.stack(donated)
    for arm, idx in result.items():
        assert idx.shape == (B, 4096) and bool((idx[:, 1:] > idx[:, :-1]).all())
        assert int(idx.min()) >= 0 and int(idx.max()) < N
        if arm != "recent":
            assert torch.equal(idx[:, -3072:], anchor)
        if "random" in arm or "donor" in arm:
            actual = torch.stack([torch.bincount(x // 256, minlength=64) for x in idx]).cpu()
            assert torch.equal(actual, counts)
    return result


def load_reference(path):
    ref_meta = json.loads((path / "metadata.json").read_text())
    ref_summary = json.loads((path / "summary.json").read_text())
    assert ref_summary["n_rows"] == 4000
    predictions = {}
    with (path / "predictions.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            predictions[row["sample_id"]] = {a: row["arms"][a]["metrics"] for a in ["recent", "hybrid_online_L11"]}
    order = json.loads((path / "execution_order.json").read_text())
    return ref_meta, predictions, order, np.load(path / "scores.npy", mmap_mode="r")


def main():
    p = argparse.ArgumentParser()
    for key in ["train-csv", "val-csv", "video-root", "checkpoint", "init-from-ckpt", "encoder-lora",
                "predictor-lora", "out-dir", "manifest", "reference-dir"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=0)
    p.add_argument("--parity-check", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # The unchanged artifact retains its original manifest-generation protocol;
    # this run's new scientific evaluation protocol is independently emitted.
    manifest = P.load_manifest(args.val_csv, args.manifest, args.n_samples)
    stop = args.stop or args.n_samples
    assert 0 <= args.start < stop <= args.n_samples
    assert args.start % 4 == stop % 4 == 0 and args.batch_size == 4
    samples = manifest["samples"][args.start:stop]
    ref_meta, reference, ref_order, ref_scores = load_reference(args.reference_dir)
    assert [r["sample_id"] for r in samples] == [r["sample_id"] for r in ref_order[args.start:stop]]
    assert P.sha(args.manifest) == ref_meta["manifest_sha256"]
    supplied_checkpoint_paths = [str(x) for x in [args.checkpoint, args.init_from_ckpt, args.encoder_lora, args.predictor_lora]]
    assert supplied_checkpoint_paths == ref_meta["checkpoint_paths"], "supplied checkpoint paths differ from reference"
    device = torch.device("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    base, mtp, maps = P.build(args, device)
    for path, identity in ref_meta["checkpoint_file_identity"].items():
        stat = Path(path).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (identity["bytes"], identity["mtime_ns"])
    n_pred = int(base.grid_size ** 2 * (base.num_output_frames // base.tubelet_size))
    assert n_pred == 256
    code_root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__), Path(P.__file__), Path(P.T.__file__),
                    code_root / "app/hdepic_lora_action_anticipation/eval_stream_mtp_multi_strategy.py",
                    code_root / "app/hdepic_lora_action_anticipation/eval_stream_mtp_kvcache_prune.py",
                    code_root / "scripts/egtea/run_b18_q1_target_crossslot.slurm",
                    SHARE_VJEPA_ROOT / "src/models/utils/modules.py", SHARE_VJEPA_ROOT / "src/models/predictor.py"]
    metadata = {"evaluation_protocol": PROTOCOL, "sample_manifest_protocol": P.PROTOCOL,
        "run_tag": args.tag, "job_id": os.environ.get("SLURM_JOB_ID"), "metric_scope": "native",
        "eval_path": "EGTEA split1 ctx16 fixed paired4000; native communicating MTP; target-query history selection",
        "manifest": str(args.manifest), "manifest_sha256": P.sha(args.manifest),
        "start": args.start, "stop": stop, "arms": ARMS, "primary_horizon": 2,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {str(x): P.sha(x) for x in source_paths},
        "train_csv_sha256": P.sha(args.train_csv), "val_csv_sha256": P.sha(args.val_csv),
        "checkpoint_paths": ref_meta["checkpoint_paths"], "checkpoint_file_identity": ref_meta["checkpoint_file_identity"],
        "reference_dir": str(args.reference_dir), "reference_metadata_sha256": P.sha(args.reference_dir / "metadata.json"),
        "gpu": {"name": torch.cuda.get_device_name(), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "token_layout": {"n_context_tokens": 16384, "n_target_tokens": n_pred, "tokens_per_slot": 256,
            "target_array_start_slot": 64, "target_rope_start_slot": 72,
            "num_heads": {str(b): base.predictor.predictor_blocks[b].attn.num_heads for b in [0, 11]}},
        "score_block_order": [0, 11], "score_kind_order": SCORE_KINDS,
        "diagnostic_kind_order": DIAGNOSTIC_KINDS, "diagnostic_topk": 4096,
        "diagnostic_keep_per_slot_axes": "sample,block,diagnostic_kind,context_slot; actual GPU topk4096",
        "token_scores_axes": "sample,block,score_kind,context_key_token (head-summed)",
        "query_group_order": ["context", "target"], "raw_group_mass_axes": "sample,block,query_group,head,key_slot",
        "conditional_definition": "post-softmax float32, zero contextQ same-slot contextK, divide each query by actual surviving sum over ALL keys including target; zero denominator returns zeros",
        "near_zero_threshold": NEAR_ZERO,
        "random_seeds": list(P.RANDOM_SEEDS),
        "diagnostic_dtypes": "legacy BF16 softmax/query sum then float32 accumulation for selectors/raw received; float32 post-softmax probability renormalization and self-token diagonal sum for diagnostics",
        "selection": {"keep": 4096, "recent_anchor_slots": 12, "history_tokens": 1024,
            "score_query_group": "256 target queries at L11 for target arms; all16640 queries for anchor",
            "positions": "original indices sorted then arange(4096) rebase", "diagnostic_intervention": "readout only"}}
    assert metadata["train_csv_sha256"] == ref_meta["train_csv_sha256"]
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.out_dir / "execution_order.json").write_text(json.dumps([{k: v for k, v in s.items() if k != "row"} for s in samples], indent=2))
    ds = P.IdentifiedDataset(args, samples)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=args.workers, collate_fn=P.collate,
                        **({"prefetch_factor": 2} if args.workers else {}))
    token_file = np.lib.format.open_memmap(args.out_dir / "token_scores.npy", mode="w+", dtype=np.float32,
                                          shape=(len(samples), 2, 3, 16384))
    predictions = (args.out_dir / "predictions.jsonl").open("x")
    sums = {arm: defaultdict(float) for arm in ARMS}
    profile_lists = defaultdict(list)
    flow_sums, token_map_sums = {}, None
    parity = {"score_exact_batches": 0, "anchor_top5_equal_examples": 0, "anchor_ce_max_abs": 0.0,
              "ce_tolerance": 1e-5, "same_mask_model_output_exact": None}
    cursor, started = 0, time.time()
    previous_end = started
    for it, batch in enumerate(loader):
        begin = time.time(); data_wait = begin - previous_end
        clip = batch["clip"].to(device).float().div_(255).sub_(P.T.IMAGENET_MEAN.to(device)).div_(P.T.IMAGENET_STD.to(device))
        B = len(batch["samples"]); ant = torch.full((B,), 2.0, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = base.encoder(clip)
            assert x.shape[1] == 16384
            ctxt = torch.arange(16384, device=device).expand(B, -1)
            anticipation_steps = (ant * base.frames_per_second / base.tubelet_size).to(torch.int64)
            tgt = torch.arange(n_pred, device=device).expand(B, -1) + 16384 + 256 * anticipation_steps[:, None]
            caps = {b: CrossSlotCapture(base.predictor.predictor_blocks[b].attn) for b in [0, 11]}
            try:
                captured_output = base.predictor(x, masks_x=ctxt, masks_y=tgt)
            finally:
                for cap in caps.values(): cap.remove()
            if it == 0 and args.parity_check:
                normal_output = base.predictor(x, masks_x=ctxt, masks_y=tgt)
                assert torch.equal(captured_output, normal_output), "readout modified model predictor output"
                parity["same_mask_model_output_exact"] = True
            packed = torch.stack([caps[b].token_scores.sum(2) for b in [0, 11]], 1).float()
            old_scores = torch.from_numpy(np.array(ref_scores[args.start + cursor:args.start + cursor + B])).to(device)
            assert torch.equal(packed[:, :, 0], old_scores), "all-query scores differ from original full4000"
            parity["score_exact_batches"] += 1
            diagnostic_counts = []
            for block_index in range(2):
                all_score, target_score, offslot_score = packed[:, block_index].unbind(1)
                readouts = [all_score - target_score, offslot_score, all_score, offslot_score + target_score, target_score]
                count_by_kind = []
                for readout in readouts:
                    chosen = readout.topk(4096, 1).indices
                    count_by_kind.append(torch.stack([torch.bincount(ix // 256, minlength=64) for ix in chosen]))
                diagnostic_counts.append(torch.stack(count_by_kind, 1))
            profile_lists["diagnostic_keep_per_slot"].append(torch.stack(diagnostic_counts, 1).cpu().numpy().astype(np.int16))
            idxs = selection({kind: packed[:, 1, ki] for ki, kind in enumerate(SCORE_KINDS)}, batch["samples"], device)
            records = [{**s, "arms": {}, "label_validity": {}} for s in batch["samples"]]
            labels_by_horizon = {}
            for hi, h in enumerate(P.HORIZONS):
                valid = batch["mtp_mask"][:, hi].to(device) > 0.5
                _, _, labels, keep = P.T.map_labels(batch["mtp_verbs"][:, hi][valid.cpu()].to(device),
                    batch["mtp_nouns"][:, hi][valid.cpu()].to(device), *maps, device)
                vp = valid.nonzero().flatten()[keep]
                labels_by_horizon[h] = vp, labels
                valid_set = set(vp.cpu().tolist())
                for bi, record in enumerate(records):
                    record["label_validity"][f"{h:g}s"] = "valid" if bi in valid_set else "masked" if not bool(valid[bi]) else "out_of_training_vocabulary"
            for arm in ARMS:
                idx = idxs[arm]
                output = mtp(P.predict_from_encoded(base, x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])), ant))
                counts = torch.stack([torch.bincount(v // 256, minlength=64) for v in idx]).cpu().tolist()
                for bi in range(B): records[bi]["arms"][arm] = {"keep_per_slot": counts[bi], "metrics": {}}
                for h in P.HORIZONS:
                    vp, labels = labels_by_horizon[h]
                    if not len(vp): continue
                    logits = output[h]["action"][vp].float()
                    correct = (logits.topk(5, -1).indices == labels[:, None]).any(-1).cpu().tolist()
                    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none").cpu().tolist()
                    for bi, y, c, loss in zip(vp.cpu().tolist(), labels.cpu().tolist(), correct, ce):
                        records[bi]["arms"][arm]["metrics"][f"{h:g}s"] = {"label": y, "top5": c, "ce": loss}
                        if arm in ["recent", "hybrid_allquery_L11"]:
                            old_arm = "recent" if arm == "recent" else "hybrid_online_L11"
                            old = reference[records[bi]["sample_id"]][old_arm][f"{h:g}s"]
                            assert old["label"] == y and old["top5"] == c
                            delta = abs(old["ce"] - loss)
                            assert delta <= parity["ce_tolerance"], f"anchor CE parity failure {delta}"
                            parity["anchor_ce_max_abs"] = max(parity["anchor_ce_max_abs"], delta)
                            parity["anchor_top5_equal_examples"] += 1
                    sums[arm][f"n@{h:g}s"] += len(vp)
                    sums[arm][f"correct@{h:g}s"] += sum(correct)
                    sums[arm][f"ce_sum@{h:g}s"] += sum(ce)
            token_file[cursor:cursor + B] = packed.cpu().numpy()
            for name, attr in [("raw_group_mass", "raw_group"), ("offslot_context_mass", "offslot_context"),
                               ("conditional_context_mass", "conditional_context"), ("diagonal_context_mass", "diagonal_context"),
                               ("diagonal_context_mass_fp32", "diagonal_context_fp32"),
                               ("self_token_context_mass", "self_token_context")]:
                profile_lists[name].append(torch.stack([getattr(caps[b], attr) for b in [0, 11]], 1).cpu().numpy())
            for name in caps[0].denominator:
                profile_lists[name].append(torch.stack([caps[b].denominator[name] for b in [0, 11]], 1).cpu().numpy())
            for name in ["raw", "offslot", "conditional"]:
                value = torch.stack([caps[b].flows[name] for b in [0, 11]]).cpu().numpy()
                flow_sums[name] = flow_sums.get(name, 0) + value
            token_map_batch = torch.stack([caps[b].token_scores.sum(0) for b in [0, 11]]).cpu().numpy()
            token_map_sums = token_map_batch if token_map_sums is None else token_map_sums + token_map_batch
            for record in records: predictions.write(json.dumps(record, separators=(",", ":")) + "\n")
            predictions.flush(); token_file.flush(); cursor += B
        torch.cuda.synchronize(); previous_end = time.time()
        print(f"itr={it + 1}/{len(loader)} n={cursor} data_wait={data_wait:.2f}s compute_wall={previous_end-begin:.2f}s elapsed={previous_end-started:.1f}s", flush=True)
    predictions.close(); assert cursor == len(samples)
    arrays = {name: np.concatenate(values) for name, values in profile_lists.items()}
    arrays.update({"mean_flow_" + name: value / cursor for name, value in flow_sums.items()})
    arrays["mean_head_token_maps"] = token_map_sums / cursor
    np.savez_compressed(args.out_dir / "crossslot_profiles.npz", **arrays)
    assert all(P.sha(path) == digest for path, digest in metadata["source_sha256"].items()), "source files changed during run"
    summary = {**metadata, "n_rows": cursor, "seconds": time.time()-started, "results": sums, "parity": parity}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps({"rows": cursor, "seconds": summary["seconds"], "parity": parity}), flush=True)


if __name__ == "__main__": main()
