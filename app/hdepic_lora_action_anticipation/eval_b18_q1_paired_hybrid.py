#!/usr/bin/env python3
"""B18 Q1 paired recent/history intervention; run only in a Slurm container.

Frozen full-context encoder is shared by every arm. All selections use exactly
4096 context tokens, sorted and position-rebased as in the existing B18 route.
The full-context predictor scoring pass is shared by L0 and L11. No labels are
used by selection. The 12+4 split and primary contrasts are fixed before eval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (
    FpsSubsampledStreamMTPDataset, enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.eval_stream_mtp_multi_strategy import predict_from_encoded

PROTOCOL = "b18-predictor-prune/egtea-ctx16-paired-hybrid-v1"
HORIZONS = [2.0, 4.0, 6.0]
RANDOM_SEEDS = [1701, 1702, 1703]
ARMS = ["recent", "online_L0", "online_L11", "offline_L0", "offline_L11",
        "hybrid_online_L0", "hybrid_online_L11", "hybrid_offline_L0", "hybrid_offline_L11"]
ARMS += [f"hybrid_random_matched_L11_seed{s}" for s in RANDOM_SEEDS]
ARMS += ["hybrid_donor_matched_L11"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_key(row):
    return hashlib.sha256((row["video_id"] + "|" + row["frame_indices"]).encode()).hexdigest()


def load_manifest(csv_path, manifest_path, n):
    with Path(csv_path).open() as f:
        source = list(csv.DictReader(f))
    meta = json.loads(manifest_path.with_suffix(".meta.json").read_text())
    assert meta["source_csv_sha256"] == sha(csv_path)
    assert meta["manifest_sha256"] == sha(manifest_path)
    assert meta["evaluation_protocol"] == PROTOCOL
    with manifest_path.open() as f:
        manifest_rows = list(csv.DictReader(f))
    assert len(manifest_rows) == n
    chosen = []
    for selection_index, entry in enumerate(manifest_rows):
        assert int(entry["selection_index"]) == selection_index
        source_index = int(entry["original_csv_row_index"])
        row = source[source_index]
        assert entry["row_sha256"] == hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert entry["video_id"] == row["video_id"] and entry["frame_indices"] == row["frame_indices"]
        assert float(row["context_sec"]) == 16 and int(row["n_model_frames"]) == 128
        chosen.append({"source_index": source_index, "selection_index": selection_index,
                       "sample_id": row_key(row), "video_id": row["video_id"],
                       "participant_id": entry["participant_id"], "tick_frame": entry["tick_frame"],
                       "row": row})
    assert len(chosen) == n and len({x["sample_id"] for x in chosen}) == n
    assert n % 2 == 0
    ordered = sorted(chosen, key=lambda s: (s["video_id"], s["selection_index"]))
    paired = []
    for i in range(n // 2):
        a, b = ordered[i], ordered[i + n // 2]
        assert a["video_id"] != b["video_id"], "deterministic donor pair shares a session"
        a["donor_sample_id"], b["donor_sample_id"] = b["sample_id"], a["sample_id"]
        a["pair_id"] = b["pair_id"] = i
        paired.extend([a, b])
    return {**meta, "n": n, "samples": paired}


class IdentifiedDataset(FpsSubsampledStreamMTPDataset):
    def __init__(self, args, samples):
        super().__init__(args.val_csv, args.video_root, 256, src_fps=8, fps=8)
        self.samples = samples
        self.rows = [s["row"] for s in samples]

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        result["sample"] = {k: v for k, v in self.samples[idx].items() if k != "row"}
        return result


def collate(batch):
    result = T.collate_stream(batch)
    result["samples"] = [b["sample"] for b in batch]
    return result


class QueryGroupCapture:
    """Mirror the existing BF16 score, retaining context/target and head summaries.

    The actual attention output is returned unchanged. Fail closed on nonempty
    attention masks/causal/dropout settings unsupported by legacy score formula.
    Context then target sequence order is checked from the actual position mask.
    Chunk size 256 matches HeadAttnCapture20, keeping its summation convention.
    """
    def __init__(self, module, n_context, gp=256):
        from src.models.utils.modules import rotate_queries_or_keys
        self.module, self.original = module, module.forward
        self.n_context, self.gp = n_context, gp
        self.importance = self.group_profiles = self.flow = None
        cap = self

        def forward(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            if attn_mask is not None or module.is_causal or module.proj_drop_prob != 0:
                raise RuntimeError("legacy score parity requires no mask, no causality, no dropout")
            B, N, _ = x.shape
            assert n_context % gp == 0 and N % gp == 0 and mask is not None
            assert torch.equal(mask[:, :n_context], torch.arange(n_context, device=x.device).expand(B, -1))
            assert bool((mask[:, n_context:] >= n_context).all())
            out = cap.original(x, mask=mask, attn_mask=attn_mask, T=T,
                               H_patches=H_patches, W_patches=W_patches)
            qkv = module.qkv(x).unflatten(-1, (3, module.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            positions = module.separate_positions(mask.unsqueeze(1).repeat(1, module.num_heads, 1),
                                                 H_patches, W_patches)
            qs, ks, start = [], [], 0
            for dim, pos in zip([module.d_dim, module.h_dim, module.w_dim], positions):
                qs.append(rotate_queries_or_keys(q[..., start:start + dim], pos=pos))
                ks.append(rotate_queries_or_keys(k[..., start:start + dim], pos=pos))
                start += dim
            if start < module.head_dim:
                qs.append(q[..., start:]); ks.append(k[..., start:])
            q, k = torch.cat(qs, -1), torch.cat(ks, -1)
            total = torch.zeros(B, module.num_heads, N, device=x.device, dtype=torch.float32)
            group = torch.zeros(B, 2, module.num_heads, N // gp, device=x.device, dtype=torch.float32)
            flow = torch.zeros(module.num_heads, N // gp, N // gp, device=x.device, dtype=torch.float32)
            for ci in range(0, N, gp):
                logits = (q[:, :, ci:ci + gp] @ k.transpose(-2, -1)) * module.scale
                received = logits.softmax(-1).sum(2).float()  # exactly legacy order/dtype
                total += received
                temporal = received.reshape(B, module.num_heads, N // gp, gp).sum(-1)
                group[:, int(ci >= n_context)] += temporal
                flow[:, ci // gp] = temporal.sum(0) / gp
            cap.importance = total
            # Raw query-summed received mass; consumers explicitly normalize over
            # all keys or condition on context keys, and divide by group query count.
            cap.group_profiles = group
            cap.flow = flow
            return out

        module.forward = forward

    def remove(self):
        self.module.forward = self.original


def select_indices(scores, offline, samples, keep=4096, anchor_slots=12, gp=256):
    B, N = scores[0].shape
    anchor_start = N - anchor_slots * gp
    anchor = torch.arange(anchor_start, N, device=scores[0].device).expand(B, -1)
    recent = torch.arange(N - keep, N, device=anchor.device).expand(B, -1)
    result = {"recent": recent}
    for block in [0, 11]:
        for mode, score in [("online", scores[block]), ("offline", offline[block].expand(B, -1))]:
            result[f"{mode}_L{block}"] = score.topk(keep, dim=1).indices.sort(1).values
            history = score[:, :anchor_start].topk(keep - anchor.shape[1], dim=1).indices
            result[f"hybrid_{mode}_L{block}"] = torch.cat([history, anchor], 1).sort(1).values
    reference = result["hybrid_online_L11"]
    temporal_counts = torch.stack([torch.bincount(idx // gp, minlength=N // gp) for idx in reference]).cpu()
    donor_indices = []
    assert B % 2 == 0
    for bi, sample in enumerate(samples):
        donor = bi ^ 1
        assert sample["donor_sample_id"] == samples[donor]["sample_id"]
        assert sample["video_id"] != samples[donor]["video_id"]
        hist = [scores[11][donor, slot * gp:(slot + 1) * gp].topk(int(temporal_counts[bi, slot])).indices + slot * gp
                for slot in range(anchor_start // gp)]
        donor_indices.append(torch.cat([*hist, anchor[bi]]).sort().values)
    result["hybrid_donor_matched_L11"] = torch.stack(donor_indices)
    for seed in RANDOM_SEEDS:
        randomized = []
        for bi, sample in enumerate(samples):
            rng_seed = int(hashlib.sha256(f"{seed}|{sample['sample_id']}".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(rng_seed)
            hist = [rng.choice(gp, size=int(temporal_counts[bi, slot]), replace=False) + slot * gp
                    for slot in range(anchor_start // gp)]
            idx = torch.as_tensor(np.concatenate(hist), device=anchor.device, dtype=torch.long)
            randomized.append(torch.cat([idx, anchor[bi]]).sort().values)
        name = f"hybrid_random_matched_L11_seed{seed}"
        result[name] = torch.stack(randomized)
    for name, idx in result.items():
        assert idx.shape == (B, keep) and bool((idx[:, 1:] > idx[:, :-1]).all())
        assert int(idx.min()) >= 0 and int(idx.max()) < N
        if name.startswith("hybrid"):
            assert torch.equal(idx[:, -anchor.shape[1]:], anchor)
        if name.startswith("hybrid_random") or name == "hybrid_donor_matched_L11":
            counts = torch.stack([torch.bincount(x // gp, minlength=N // gp) for x in idx]).cpu()
            assert torch.equal(counts, temporal_counts)
    return result


def build(args, device):
    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    base = T.build_model(device, 128, 8, 256, str(args.checkpoint))
    T.load_lora_sidecars(base, str(args.encoder_lora), str(args.predictor_lora))
    enlarge_predictor_budget(base, 64 * 256, 256)
    classifier = T.AttentiveClassifier(verb_classes=verb_map, noun_classes=noun_map,
        action_classes=action_map, embed_dim=int(base.encoder.embed_dim), num_heads=16,
        depth=4, use_activation_checkpointing=True).to(device)
    mtp = T.CommunicatingMLPMTPClassifier(classifier, horizons_sec=HORIZONS,
                                         comm_layers=2, comm_heads=4).to(device)
    ck = torch.load(args.init_from_ckpt, map_location="cpu", weights_only=False)
    wrap = T.PrunedAnticipativeModel(base, None, prune_threshold=10 ** 9)
    model_load = wrap.load_state_dict(ck["model"], strict=False)
    clf_load = mtp.load_state_dict(ck["mtp_classifier"], strict=False)
    print(f"[load] model={model_load} classifier={clf_load} best={ck.get('best')}", flush=True)
    if model_load.missing_keys or model_load.unexpected_keys or clf_load.missing_keys or clf_load.unexpected_keys:
        raise RuntimeError("unexpected checkpoint key mismatch")
    for module in [base, mtp]:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    return base, mtp, (verb_map, noun_map, action_map)


def main():
    p = argparse.ArgumentParser()
    for key in ["train-csv", "val-csv", "video-root", "checkpoint", "init-from-ckpt", "encoder-lora",
                "predictor-lora", "calib-l0", "calib-l11", "out-dir", "manifest"]:
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
    manifest = load_manifest(args.val_csv, args.manifest, args.n_samples)
    stop = args.stop or manifest["n"]
    assert 0 <= args.start < stop <= manifest["n"]
    assert args.start % 2 == stop % 2 == args.batch_size % 2 == 0
    samples = manifest["samples"][args.start:stop]
    device = torch.device("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    base, mtp, maps = build(args, device)
    offline = {0: torch.from_numpy(np.load(args.calib_l0)).flatten().to(device),
               11: torch.from_numpy(np.load(args.calib_l11)).flatten().to(device)}
    for score in offline.values():
        assert score.shape == (16384,) and bool(torch.isfinite(score).all())
    metadata = {"evaluation_protocol": PROTOCOL, "run_tag": args.tag,
        "job_id": os.environ.get("SLURM_JOB_ID"), "metric_scope": "native",
        "eval_path": "EGTEA split1 ctx16 proportional session sample; paired shared encoder; native Action Top5",
        "manifest": str(args.manifest), "manifest_sha256": sha(args.manifest),
        "start": args.start, "stop": stop, "arms": ARMS, "primary_horizon": 2,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {str(x): sha(x) for x in [Path(__file__), Path(T.__file__),
            Path(__file__).with_name("eval_stream_mtp_multi_strategy.py"),
            Path(__file__).with_name("eval_stream_mtp_kvcache_prune.py"),
            Path(__file__).resolve().parents[2] / "vjepa2/src/models/utils/modules.py",
            Path(__file__).resolve().parents[2] / "vjepa2/src/models/predictor.py"]},
        "train_csv_sha256": sha(args.train_csv),
        "calibration_sha256": {str(x): sha(x) for x in [args.calib_l0, args.calib_l11]},
        "checkpoint_paths": [str(x) for x in [args.checkpoint, args.init_from_ckpt,
                                                args.encoder_lora, args.predictor_lora]],
        "checkpoint_file_identity": {str(x): {"bytes": x.stat().st_size, "mtime_ns": x.stat().st_mtime_ns}
             for x in [args.checkpoint, args.init_from_ckpt, args.encoder_lora, args.predictor_lora]},
        "gpu": {"name": torch.cuda.get_device_name(), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "token_layout": {"n_context_tokens": 16384, "n_target_tokens": int(base.grid_size ** 2 * (base.num_output_frames // base.tubelet_size)),
                         "tokens_per_slot": 256, "target_rope_start_slot": 72,
                         "target_array_start_slot": 64,
                         "num_heads": {str(b): base.predictor.predictor_blocks[b].attn.num_heads for b in [0, 11]}},
        "selection": {"context_slots": 64, "gp": 256, "keep": 4096, "anchor_slots": 12,
                      "positions": "sort original indices then rebase arange(4096)",
                      "random": "3 fixed seed spatial draws matching L11 hybrid per-slot counts",
                      "donor": "video-sorted full manifest split in equal halves, pair offset n/2; donor ranks within receiver L11 per-slot historical counts"}}
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.out_dir / "execution_order.json").write_text(json.dumps([{k: v for k, v in s.items() if k != "row"} for s in samples], indent=2))
    ds = IdentifiedDataset(args, samples)
    lk = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    if args.workers:
        lk["prefetch_factor"] = 2
    loader = DataLoader(ds, **lk)
    score_file = np.lib.format.open_memmap(args.out_dir / "scores.npy", mode="w+", dtype=np.float32,
                                          shape=(len(samples), 2, 16384))
    predictions = (args.out_dir / "predictions.jsonl").open("x")
    profiles, flow_sums, summaries = [], {}, {arm: defaultdict(float) for arm in ARMS}
    cursor = 0
    started = time.time()
    previous_end = started
    for it, batch in enumerate(loader):
        iteration_start = time.time()
        data_wait = iteration_start - previous_end
        clip = batch["clip"].to(device).float().div_(255).sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
        B = len(batch["samples"])
        ant = torch.full((B,), 2.0, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = base.encoder(clip)
            assert x.shape[1] == 16384
            ctxt = torch.arange(16384, device=device).expand(B, -1)
            npred = int(base.grid_size ** 2 * (base.num_output_frames // base.tubelet_size))
            anticipation_steps = (ant * base.frames_per_second / base.tubelet_size).to(torch.int64)
            tgt = torch.arange(npred, device=device).expand(B, -1) + 16384 + 256 * anticipation_steps[:, None]
            caps = {b: QueryGroupCapture(base.predictor.predictor_blocks[b].attn, 16384) for b in [0, 11]}
            try:
                scored_output = base.predictor(x, masks_x=ctxt, masks_y=tgt)
            finally:
                for cap in caps.values():
                    cap.remove()
            scores = {b: cap.importance[:, :, :16384].sum(1).float() for b, cap in caps.items()}
            if it == 0 and args.parity_check:
                from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import HeadAttnCapture20
                legacy = {b: HeadAttnCapture20(base.predictor.predictor_blocks[b].attn) for b in [0, 11]}
                try:
                    parity_output = base.predictor(x, masks_x=ctxt, masks_y=tgt)
                finally:
                    for cap in legacy.values():
                        cap.remove()
                out_a = scored_output[0] if isinstance(scored_output, tuple) else scored_output
                out_b = parity_output[0] if isinstance(parity_output, tuple) else parity_output
                assert torch.equal(out_a, out_b), "capture changed predictor output"
                for b in [0, 11]:
                    old = legacy[b].importance[:, :, :16384].sum(1).float()
                    assert torch.equal(scores[b], old), f"score parity failed L{b}"
                print("[parity] legacy/new scores and predictor outputs exactly equal L0/L11", flush=True)
                metadata["legacy_capture_parity"] = "exact first batch L0/L11 scores and predictor output"
                (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            idxs = select_indices(scores, offline, batch["samples"])
            batch_records = [{**s, "arms": {}} for s in batch["samples"]]
            valid_labels = {}
            for hi, h in enumerate(HORIZONS):
                valid = batch["mtp_mask"][:, hi].to(device) > 0.5
                _, _, labels, keep = T.map_labels(batch["mtp_verbs"][:, hi][valid.cpu()].to(device),
                    batch["mtp_nouns"][:, hi][valid.cpu()].to(device), *maps, device)
                vp = valid.nonzero().flatten()[keep]
                valid_labels[h] = (vp, labels)
                valid_set = set(vp.cpu().tolist())
                for bi, record in enumerate(batch_records):
                    record.setdefault("label_validity", {})[f"{h:g}s"] = (
                        "valid" if bi in valid_set else "masked" if not bool(valid[bi]) else "out_of_training_vocabulary")
            for arm in ARMS:
                idx = idxs[arm]
                kept = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                output = mtp(predict_from_encoded(base, kept, ant))
                counts = torch.stack([torch.bincount(v // 256, minlength=64) for v in idx]).cpu().tolist()
                for bi in range(B):
                    batch_records[bi]["arms"][arm] = {"keep_per_slot": counts[bi], "metrics": {}}
                for h in HORIZONS:
                    vp, labels = valid_labels[h]
                    if not len(vp):
                        continue
                    logits = output[h]["action"][vp].float()
                    correct = (logits.topk(5, -1).indices == labels[:, None]).any(-1).cpu().tolist()
                    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none").cpu().tolist()
                    for bi, y, c, loss in zip(vp.cpu().tolist(), labels.cpu().tolist(), correct, ce):
                        batch_records[bi]["arms"][arm]["metrics"][f"{h:g}s"] = {"label": y, "top5": c, "ce": loss}
                    summaries[arm][f"n@{h:g}s"] += len(vp)
                    summaries[arm][f"correct@{h:g}s"] += sum(correct)
                    summaries[arm][f"ce_sum@{h:g}s"] += sum(ce)
                for b in [0, 11]:
                    selected = scores[b].gather(1, idx).cpu()
                    for bi in range(B):
                        is_history = idx[bi].cpu() < 52 * 256
                        history_score = selected[bi][is_history]
                        batch_records[bi]["arms"][arm][f"history_score_L{b}_mean"] = (
                            float(history_score.mean()) if len(history_score) else None)
            score_file[cursor:cursor + B] = torch.stack([scores[0], scores[11]], 1).cpu().numpy()
            profiles.append(torch.stack([caps[0].group_profiles, caps[11].group_profiles], 1).cpu().numpy())
            for b in [0, 11]:
                flow_sums[b] = flow_sums.get(b, 0) + caps[b].flow.cpu().numpy()
            for record in batch_records:
                predictions.write(json.dumps(record, separators=(",", ":")) + "\n")
            predictions.flush(); score_file.flush()
            cursor += B
        torch.cuda.synchronize()
        previous_end = time.time()
        print(f"itr={it + 1}/{len(loader)} n={cursor} data_wait={data_wait:.2f}s compute_wall={previous_end-iteration_start:.2f}s elapsed={previous_end-started:.1f}s", flush=True)
    predictions.close()
    assert cursor == len(samples)
    np.savez_compressed(args.out_dir / "attention_profiles.npz", group_mass=np.concatenate(profiles),
                        mean_queryslot_keyslot_L0=flow_sums[0] / cursor,
                        mean_queryslot_keyslot_L11=flow_sums[11] / cursor)
    report = {**metadata, "n_rows": cursor, "seconds": time.time()-started, "results": summaries,
              "query_group_order": ["context", "target"], "score_block_order": [0, 11],
              "group_mass_axes": "sample,block,query_group,head,key_slot; raw query-summed mass",
              "mean_flow_axes": "head,query_slot,key_slot; query-averaged then example-averaged mass"}
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print("[done] " + json.dumps({"rows": cursor, "seconds": report["seconds"], "output": str(args.out_dir)}), flush=True)


if __name__ == "__main__":
    main()
