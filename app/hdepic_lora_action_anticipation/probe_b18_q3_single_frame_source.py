#!/usr/bin/env python3
"""Frozen single-frame source interventions; CPU preparation, GPU L0 trace."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from app.hdepic_lora_action_anticipation.probe_b18_q3_oldest_mechanism import (
    SignalTrace, bytehash, digest, write_json, CaptureComplete, SEEDS, GP, NS,
)

PROTOCOL = "b18-predictor-prune/egtea-q3-single-frame-source-v1"
GROUP = "B18-predictor-prune-keep-pattern-causes"
KINDS = ("same_window", "other_egtea", "hdepic")
CONDITIONS = ["continuous", "start_self_first_repeat"] + [f"start_{kind}_{seed}" for kind in KINDS for seed in SEEDS] + [f"middle_{kind}_{SEEDS[0]}" for kind in KINDS]


def seeded_rng(seed, sample_id, kind):
    value = int.from_bytes(hashlib.sha256(f"{seed}|{sample_id}|{kind}".encode()).digest()[:8], "little")
    return np.random.default_rng(value), value


def construct(receiver, images, condition):
    clip = np.array(receiver[-128:], copy=True)
    if condition != "continuous":
        where, key = condition.split("_", 1)
        start = 0 if where == "start" else 64
        tile = np.repeat(images[key][None], 4, axis=0)
        clip[start:start + 4] = tile
        assert np.array_equal(clip[:start], receiver[-128:][:start])
        assert np.array_equal(clip[start + 4:], receiver[-128:][start + 4:])
        assert np.array_equal(clip[start:start + 4], tile)
    assert clip.shape == (128, 256, 256, 3) and clip.dtype == np.uint8
    return clip


def hd_metadata(item):
    from decord import VideoReader, cpu
    video_id, path = item
    stat = Path(path).stat()
    reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    native_shape = reader[0].asnumpy().shape
    out = dict(video_id=video_id, participant="P01", path=path, resolved_path=str(Path(path).resolve()),
               source_fps=float(reader.get_avg_fps()), frame_count=len(reader), native_shape=list(native_shape),
               size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest(path), split="p01_fixed_train")
    del reader
    return out


def decode_hd(item):
    from decord import VideoReader, cpu
    metadata, picks, out = item
    if not picks:
        return dict(video_id=metadata["video_id"], decoded_frames=0)
    reader = VideoReader(metadata["path"], ctx=cpu(0), num_threads=1, width=256, height=256)
    assert len(reader) == metadata["frame_count"]
    indices = sorted(set(p["native_frame"] for p in picks))
    frames = reader.get_batch(indices).asnumpy()
    assert frames.shape == (len(indices), 256, 256, 3) and frames.dtype == np.uint8
    for index, image in zip(indices, frames):
        np.save(Path(out) / f"{metadata['video_id']}_{index}.npy", image)
    return dict(video_id=metadata["video_id"], decoded_frames=len(indices))


def prepare(args):
    source = json.loads(args.source_manifest.read_text())
    assert source["n_videos"] == 86 and source["n_participants"] == 32
    args.out.mkdir(parents=True, exist_ok=False)
    hd_dir, image_dir = args.out / "hdepic_frames", args.out / "images"
    hd_dir.mkdir(); image_dir.mkdir()
    hd_ids = [x.strip() for x in args.hdepic_train.read_text().splitlines() if x.strip()]
    assert len(hd_ids) == len(set(hd_ids)) == 20 and all(x.startswith("P01-") for x in hd_ids)
    items = [(vid, str(args.hdepic_root / (vid.replace("P01-", "P01_", 1) + ".MP4"))) for vid in hd_ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        hd_meta = list(pool.map(hd_metadata, items))
    samples = source["samples"]
    lookup = {s["sample_id"]: s for s in samples}
    rows = []
    for sample in samples:
        draws = {}
        for kind in KINDS:
            for seed in SEEDS:
                rng, derived = seeded_rng(seed, sample["sample_id"], kind)
                key = f"{kind}_{seed}"
                if kind == "hdepic":
                    donor = hd_meta[int(rng.integers(20))]
                    native_frame = int(rng.integers(donor["frame_count"]))
                    draw = dict(source_kind=kind, seed=seed, derived_seed=derived, video_id=donor["video_id"],
                                participant="P01", native_frame=native_frame, source_path=donor["path"],
                                source_fps=donor["source_fps"], source_seconds=native_frame / donor["source_fps"],
                                image_path=str(hd_dir / f"{donor['video_id']}_{native_frame}.npy"))
                else:
                    candidates = [s for s in samples if s["participant"] != sample["participant"]]
                    donor = sample if kind == "same_window" else candidates[int(rng.integers(len(candidates)))]
                    local = int(rng.integers(4 if kind == "same_window" else 0, 128))
                    native_frame = donor["frame_indices"][128 + local]
                    draw = dict(source_kind=kind, seed=seed, derived_seed=derived, sample_id=donor["sample_id"],
                                video_id=donor["video_id"], participant=donor["participant"], cache_index=128 + local,
                                local_rgb_index=local, native_frame=native_frame, source_path=donor["video_path"],
                                source_fps=donor["source_fps"], source_seconds=native_frame / donor["source_fps"],
                                candidate_videos=1 if kind == "same_window" else len(candidates))
                    assert kind == "same_window" or donor["participant"] != sample["participant"]
                draws[key] = draw
        rows.append(dict(sample_id=sample["sample_id"], draws=draws,
                         self_first_repeat=dict(video_id=sample["video_id"], participant=sample["participant"],
                             local_rgb_index=0, cache_index=128, native_frame=sample["frame_indices"][128])))
    write_json(args.out / "selection_before_decode.json", dict(protocol=PROTOCOL, rows=rows, hdepic_videos=hd_meta))
    # Selection is immutable before any selected image is decoded; failures do not resample.
    decode_items = [(m, [d for r in rows for d in r["draws"].values() if d["source_kind"] == "hdepic" and d["video_id"] == m["video_id"]], str(hd_dir)) for m in hd_meta]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        decoded = list(pool.map(decode_hd, decode_items))
    for s in samples:
        assert digest(s["cache"]["path"]) == s["cache"]["sha256"]
    for sample, row in zip(samples, rows):
        receiver = np.load(sample["cache"]["path"], mmap_mode="r")
        images = {"self_first_repeat": np.array(receiver[128], copy=True)}
        for key, draw in row["draws"].items():
            if draw["source_kind"] == "hdepic":
                image = np.load(draw["image_path"])
            else:
                donor = lookup[draw["sample_id"]]
                image = np.array(np.load(donor["cache"]["path"], mmap_mode="r")[draw["cache_index"]], copy=True)
            assert image.shape == (256, 256, 3) and image.dtype == np.uint8
            images[key] = image
            draw["image_sha256"] = bytehash(image)
            draw["tile_sha256"] = bytehash(np.repeat(image[None], 4, axis=0))
        image_path = image_dir / f"{sample['sample_id']}.npz"
        np.savez_compressed(image_path, **images)
        row["image_archive"] = dict(path=str(image_path), sha256=digest(image_path))
        row["image_sha256"] = {k: bytehash(v) for k, v in images.items()}
        row["input_sha256"] = {c: bytehash(construct(receiver, images, c)) for c in CONDITIONS}
        for kind in KINDS:
            first = construct(receiver, images, f"start_{kind}_{SEEDS[0]}")[:4]
            mid = construct(receiver, images, f"middle_{kind}_{SEEDS[0]}")[64:68]
            assert np.array_equal(first, mid)
    for value in source["model"].values():
        assert digest(value["path"]) == value["sha256"]
    old_mechanism = args.source_manifest.parents[2] / "q3_oldest_mechanism" / "prepared" / "manifest.json"
    old_manifest = json.loads(old_mechanism.read_text())
    dependencies = dict(old_manifest["code_sha256"])
    for path, expected in dependencies.items():
        assert digest(path) == expected, f"old frozen dependency changed: {path}"
    dependencies[str(Path(__file__).resolve())] = digest(__file__)
    coverage = {}
    for kind in KINDS:
        draws = [d for r in rows for d in r["draws"].values() if d["source_kind"] == kind]
        keys = [(d["video_id"], d["native_frame"]) for d in draws]
        coverage[kind] = dict(n_draws=len(draws), n_videos=len(set(d["video_id"] for d in draws)),
                              n_participants=len(set(d["participant"] for d in draws)),
                              exact_image_repeat_draws=len(keys) - len(set(keys)),
                              videos=dict(Counter(d["video_id"] for d in draws)))
    write_json(args.out / "manifest.json", dict(protocol=PROTOCOL, group=GROUP, conditions=CONDITIONS,
        source_manifest=str(args.source_manifest), source_manifest_sha256=digest(args.source_manifest), source=source,
        previous_mechanism_manifest=str(old_mechanism), previous_mechanism_manifest_sha256=digest(old_mechanism),
        hdepic_train=str(args.hdepic_train), hdepic_train_sha256=digest(args.hdepic_train), hdepic_videos=hd_meta,
        rows=rows, seeds=SEEDS, code_sha256=dependencies, source_coverage=coverage, decoded=decoded,
        preprocessing="uint8 single RGB frame repeated4 times; original direct256 resize/ImageNet normalization; all otherRGB exact",
        limitations=["same_window is one continuous16s window, not manually annotated same scene", "HD-EPIC pool is P01 fixed train20 videos only", "three seeds/donor-linked rows are not independent inferential replicates"]))
    print("PREPARED", json.dumps(dict(n_rows=86, n_conditions=14, coverage=coverage)), flush=True)


def compare_scores(actual, expected):
    rel = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    masks = [set(np.argsort(-z.ravel(), kind="stable")[:4096].tolist()) for z in (actual, expected)]
    jac = len(masks[0] & masks[1]) / len(masks[0] | masks[1])
    return dict(relative_l2=rel, topk_jaccard=jac, passed=rel <= .005 and jac >= .98)


def gpu(args):
    import torch
    from app.hdepic_lora_action_anticipation import train_stream_mtp as T
    from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import HeadAttnCapture20, build_finetuned_20
    torch.set_num_threads(1); torch.manual_seed(SEEDS[0])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["protocol"] == PROTOCOL and manifest["conditions"] == CONDITIONS
    for path, expected in manifest["code_sha256"].items():
        assert digest(path) == expected, f"frozen capture dependency changed: {path}"
    source = manifest["source"]
    assert 0 <= args.start < args.stop <= 86
    args.out.mkdir(parents=True, exist_ok=False)
    paths = {k: v["path"] for k, v in source["model"].items()}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        base, _ = build_finetuned_20(torch.device("cuda"), max_frames=256, fps=8, img_size=256,
            checkpoint=paths["checkpoint"], enc_lora=paths["encoder_lora"], pred_lora=paths["predictor_lora"], parent_ckpt=paths["parent"])
    print(log.getvalue(), flush=True)
    assert "[finetuned] parent load: 0 missing, 0 unexpected" in log.getvalue()
    base.requires_grad_(False)
    assert base.num_steps == 1 and base.tubelet_size == 2
    model = T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).eval()
    trace = SignalTrace(base)
    ant = torch.full((1,), 2., device="cuda")
    mean, std = T.IMAGENET_MEAN.cuda(), T.IMAGENET_STD.cuda()
    def run(rgb, early=True):
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(3, 0, 1, 2).unsqueeze(0).cuda()
        x = x.float().div_(255).sub_(mean).div_(std)
        def abort(*unused):
            raise CaptureComplete()
        hook = trace.m.register_forward_hook(abort) if early else None
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, ant)
            assert not early
        except CaptureComplete:
            assert early
        finally:
            if hook is not None: hook.remove()
    anchor_rgb = np.array(np.load(source["samples"][0]["cache"]["path"], mmap_mode="r")[-128:])
    ref = HeadAttnCapture20(trace.m, chunk_size=GP)
    run(anchor_rgb); reference = ref.importance.clone()
    parity = {}
    if args.parity:
        run(anchor_rgb, early=False)
        assert torch.equal(reference, ref.importance)
        parity["reference_full_early_exact"] = True
    ref.remove(); trace.install(); trace.enabled = True
    run(anchor_rgb)
    assert torch.equal(reference[0], trace.data["head_importance_actual"])
    parity["instrumented_reference_exact"] = True
    if args.parity:
        traced = {k: v.clone() for k, v in trace.data.items()}
        run(anchor_rgb, early=False)
        assert all(torch.equal(v, trace.data[k]) for k, v in traced.items())
        parity["instrumented_full_early_exact"] = True
    actual = reference[0, :, :NS * GP].sum(0).reshape(NS, GP).cpu().numpy()
    archived = np.load(args.archive_anchor)["score_actual__continuous"]
    parity["archived_continuous"] = compare_scores(actual, archived)
    write_json(args.out / "gate_report.json", parity)
    assert parity["archived_continuous"]["passed"], "archived baseline hardware gate failed; do not relax tolerance"
    hardware_anchor = {}
    for i in range(2):
        rgb = np.load(source["samples"][i]["cache"]["path"], mmap_mode="r")
        images = np.load(manifest["rows"][i]["image_archive"]["path"])
        for c in CONDITIONS:
            run(construct(rgb, images, c))
            hardware_anchor[f"q3_{i:03d}__{c}"] = trace.data["head_importance_actual"][:, :NS * GP].sum(0).reshape(NS, GP).float().cpu().numpy()
    np.savez_compressed(args.out / "hardware_anchors.npz", **hardware_anchor)
    if args.smoke_reference:
        expected = np.load(args.smoke_reference)
        gates = {k: compare_scores(v, expected[k]) for k, v in hardware_anchor.items()}
        parity["smoke_all28_conditions"] = gates
        write_json(args.out / "gate_report.json", parity)
        assert all(v["passed"] for v in gates.values()), "intervention hardware gate failed; do not merge"
    metadata = dict(protocol=PROTOCOL, group=GROUP, tag=args.tag, job_id=os.environ.get("SLURM_JOB_ID"),
        manifest=str(args.manifest), manifest_sha256=digest(args.manifest), start=args.start, stop=args.stop,
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda=torch.version.cuda, dtype="bf16", query_chunk=GP,
        parity=parity, model=source["model"], code_sha256=manifest["code_sha256"], timing_start=time.time(),
        smoke_reference=str(args.smoke_reference) if args.smoke_reference else None)
    write_json(args.out / "metadata.json", metadata)
    print("SINGLE_FRAME_COMPUTE_START", metadata["timing_start"], flush=True)
    times = []
    with (args.out / "timings.jsonl").open("w") as timing:
        for index in range(args.start, args.stop):
            sample, row = source["samples"][index], manifest["rows"][index]
            receiver = np.load(sample["cache"]["path"], mmap_mode="r")
            assert digest(row["image_archive"]["path"]) == row["image_archive"]["sha256"]
            images = np.load(row["image_archive"]["path"])
            assert {k: bytehash(images[k]) for k in images.files} == row["image_sha256"]
            arrays = {"sample_index": np.asarray(index)}
            row_start = time.monotonic()
            for condition in CONDITIONS:
                start = time.monotonic()
                rgb = construct(receiver, images, condition)
                assert bytehash(rgb) == row["input_sha256"][condition]
                trace.data = {}; run(rgb)
                for name, value in trace.data.items():
                    assert bool(torch.isfinite(value).all())
                    arrays[name + "__" + condition] = value.float().cpu().numpy()
                for suffix in ("actual", "no_rope_readout"):
                    imp = trace.data["head_importance_" + suffix]
                    arrays["score_" + suffix + "__" + condition] = imp[:, :NS * GP].sum(0).reshape(NS, GP).float().cpu().numpy()
                timing.write(json.dumps(dict(index=index, condition=condition, seconds=time.monotonic() - start)) + "\n"); timing.flush()
            path = args.out / f"{sample['sample_id']}.npz"
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **arrays); temporary.rename(path)
            elapsed = time.monotonic() - row_start; times.append(elapsed)
            print("SINGLE_FRAME_ROW", index, f"seconds={elapsed:.3f}", flush=True)
    trace.remove()
    summary = dict(protocol=PROTOCOL, run_status="completed", n_rows=args.stop - args.start, n_conditions=14,
        start=args.start, stop=args.stop, sample_seconds=times, elapsed_compute_seconds=time.time() - metadata["timing_start"],
        timing_end=time.time(), peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30, parity=parity)
    write_json(args.out / "summary.json", summary)
    print("SINGLE_FRAME_COMPUTE_END", json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    for name in ("source-manifest", "hdepic-train", "hdepic-root", "out"):
        q.add_argument("--" + name, type=Path, required=True)
    q.add_argument("--workers", type=int, default=4)
    q = sub.add_parser("gpu")
    for name in ("manifest", "out", "archive-anchor"):
        q.add_argument("--" + name, type=Path, required=True)
    q.add_argument("--smoke-reference", type=Path)
    q.add_argument("--start", type=int, default=0); q.add_argument("--stop", type=int, default=2)
    q.add_argument("--tag", required=True); q.add_argument("--parity", action="store_true")
    args = p.parse_args(); {"prepare": prepare, "gpu": gpu}[args.command](args)


if __name__ == "__main__":
    main()
