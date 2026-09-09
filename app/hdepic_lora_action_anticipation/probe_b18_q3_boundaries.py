#!/usr/bin/env python3
"""B18 Q3: matched context lengths, scene cuts, and moving window starts.

All Python execution belongs in the project container on Slurm. The prepare
command stages real, unpadded RGB; GPU runs only frozen predictor-L0 capture.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROTOCOL = "b18-predictor-prune/egtea-q3-window-scene-boundary-v1"
GROUP = "B18-predictor-prune-keep-pattern-causes"
FPS, GP, MASTER_FRAMES = 8, 256, 256
SEED = 20260906


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def conditions():
    result = []
    for seconds in (4, 8, 16, 32):
        result.append(dict(name=f"continuous_{seconds}s", seconds=seconds, shift=0, cut=None))
        result.append(dict(name=f"cut50_{seconds}s", seconds=seconds, shift=0, cut=0.5))
    for seconds in (16, 32):
        for fraction in (0.25, 0.75):
            result.append(dict(name=f"cut{int(100*fraction)}_{seconds}s", seconds=seconds,
                               shift=0, cut=fraction))
    for shift in (4, 8):
        result.append(dict(name=f"shift_m{shift}_16s", seconds=16, shift=shift, cut=None))
    return result


def construct(receiver, donor, condition):
    n = condition["seconds"] * FPS
    end = MASTER_FRAMES - condition["shift"] * FPS
    start = end - n
    assert 0 <= start < end <= MASTER_FRAMES
    clip = np.array(receiver[start:end], copy=True)
    if condition["cut"] is not None:
        cut = int(n * condition["cut"])
        assert cut % 2 == 0 and 0 < cut < n
        clip[:cut] = donor[start:start + cut]
        assert np.array_equal(clip[cut:], receiver[start + cut:end])
    return clip


def stage_video(item):
    from decord import VideoReader, cpu
    sample, cache = item
    vr = VideoReader(sample["video_path"], ctx=cpu(0), num_threads=1, width=256, height=256)
    indices = np.asarray(sample["frame_indices"], dtype=np.int64)
    assert indices.shape == (MASTER_FRAMES,)
    assert indices.min() >= sample["origin_frame"] and indices.max() < len(vr)
    assert np.all(np.diff(indices) > 0), "no duplicate/padded frames allowed"
    actual_fps = float(vr.get_avg_fps())
    assert abs(actual_fps / sample["source_fps"] - 1) < 0.005
    rgb = vr.get_batch(indices.tolist()).asnumpy()
    assert rgb.shape == (MASTER_FRAMES, 256, 256, 3) and rgb.dtype == np.uint8
    path = Path(cache) / f"{sample['sample_id']}.npy"
    np.save(path, rgb)
    del vr
    return dict(sample_id=sample["sample_id"], path=str(path), sha256=digest(path),
                source_fps=actual_fps, bytes=path.stat().st_size)


def prepare(args):
    assert not (args.out / "manifest.json").exists(), "immutable manifest already exists"
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "rgb"
    cache.mkdir(exist_ok=True)
    by_video = defaultdict(list)
    with args.train_csv.open() as f:
        for row_id, row in enumerate(csv.DictReader(f)):
            assert row["split"] == "train"
            if float(row["context_sec"]) != 16:
                continue
            original = [int(v) for v in row["frame_indices"].split(",")]
            end = original[-1]
            fps = float(row["vfps"])
            indices = np.rint(end - np.arange(MASTER_FRAMES - 1, -1, -1) * fps / FPS).astype(np.int64)
            if indices[0] < int(row["origin_frame"]):
                continue
            vid = row["video_id"]
            key = hashlib.sha256(f"{SEED}|{vid}|{row_id}".encode()).hexdigest()
            by_video[vid].append((key, row_id, row, indices.tolist()))
    videos = sorted(by_video, key=lambda v: hashlib.sha256(f"{SEED}|{v}".encode()).hexdigest())
    assert len(videos) >= 32, "insufficient video coverage"
    video_paths = defaultdict(list)
    for path in args.video_root.rglob("*.MP4"):
        video_paths[path.stem].append(path)
    samples = []
    for i, vid in enumerate(videos):
        _, row_id, row, indices = min(by_video[vid], key=lambda x: x[0])
        participant = vid.split("-")[0]
        candidates = video_paths[vid]
        assert len(candidates) == 1, (vid, candidates)
        path = candidates[0]
        samples.append(dict(sample_id=f"q3_{i:03d}", video_id=vid, participant=participant,
                            source_row=row_id, source_fps=float(row["vfps"]),
                            origin_frame=int(row["origin_frame"]), video_path=str(path),
                            frame_indices=indices, calibration_half=i % 2))
    # Each disjoint calibration half owns its donors; no video crosses halves.
    for half in (0, 1):
        ix = [i for i, s in enumerate(samples) if s["calibration_half"] == half]
        shift = min(range(1, len(ix)), key=lambda k: sum(
            samples[ix[j]]["participant"] == samples[ix[(j+k) % len(ix)]]["participant"]
            for j in range(len(ix))))
        for j, i in enumerate(ix):
            donor = samples[ix[(j + shift) % len(ix)]]
            assert donor["video_id"] != samples[i]["video_id"]
            samples[i]["donor_sample_id"] = donor["sample_id"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, item in enumerate(pool.map(stage_video, [(s, cache) for s in samples])):
            assert samples[k]["sample_id"] == item["sample_id"]
            samples[k]["cache"] = item
            print(f"staged {k+1}/{len(samples)} {item['sample_id']}", flush=True)
    for sample in samples:
        receiver = np.load(sample["cache"]["path"], mmap_mode="r")
        donor = np.load(next(s for s in samples if s["sample_id"] == sample["donor_sample_id"])["cache"]["path"], mmap_mode="r")
        clips = {c["name"]: construct(receiver, donor, c) for c in conditions()}
        assert np.array_equal(clips["continuous_4s"], clips["continuous_32s"][-32:])
        assert np.array_equal(clips["shift_m4_16s"][32:], clips["continuous_16s"][:-32])
        assert np.array_equal(clips["shift_m8_16s"][64:], clips["continuous_16s"][:-64])
    provenance = {}
    for key in ("checkpoint", "encoder_lora", "predictor_lora", "parent"):
        path = getattr(args, key)
        assert path.is_file(), path
        provenance[key] = dict(path=str(path), sha256=digest(path))
    code_root = Path(__file__).resolve().parents[2]
    code = [Path(__file__), code_root / "app/hdepic_lora_action_anticipation/train_stream_mtp.py",
            code_root / "app/hdepic_lora_action_anticipation/analyze_encoder_head_attn_corners.py",
            code_root / "app/hdepic_lora_action_anticipation/eval_stream_mtp_kvcache_prune.py"]
    manifest = dict(protocol=PROTOCOL, group=GROUP, seed=SEED, samples=samples, conditions=conditions(),
                    train_csv=str(args.train_csv), train_csv_sha256=digest(args.train_csv),
                    model=provenance, code_sha256={str(p): digest(p) for p in code},
                    preprocessing="256px direct resize; backward native-fps/8 sampling ending at source row last frame; identical suffixes; no padding",
                    n_videos=len(videos), n_participants=len(set(s["participant"] for s in samples)),
                    cross_participant_pairs=sum(s["participant"] != next(x for x in samples if x["sample_id"] == s["donor_sample_id"])["participant"] for s in samples))
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("protocol", "n_videos", "n_participants", "cross_participant_pairs")}), flush=True)


class CaptureComplete(Exception):
    pass


def gpu(args):
    import torch
    from app.hdepic_lora_action_anticipation import train_stream_mtp as T
    from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import HeadAttnCapture20, build_finetuned_20
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 semantics require a compatible GPU"
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads(args.manifest.read_text())
    assert manifest["protocol"] == PROTOCOL and manifest["conditions"] == conditions()
    for path, expected in manifest["code_sha256"].items():
        assert digest(path) == expected, f"code changed after preparation: {path}"
    stop = args.stop or len(manifest["samples"])
    assert 0 <= args.start < stop <= len(manifest["samples"])
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    model_paths = {k: v["path"] for k, v in manifest["model"].items()}
    base, _ = build_finetuned_20(device, max_frames=MASTER_FRAMES, fps=FPS, img_size=256,
                                checkpoint=model_paths["checkpoint"], enc_lora=model_paths["encoder_lora"],
                                pred_lora=model_paths["predictor_lora"], parent_ckpt=model_paths["parent"])
    assert base.num_steps == 1 and base.tubelet_size == 2
    base.requires_grad_(False)
    model = T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).eval()
    module = base.predictor.predictor_blocks[0].attn
    assert not module.is_causal and module.proj_drop_prob == 0
    cap = HeadAttnCapture20(module, chunk_size=256)
    ant = torch.full((1,), 2.0, device=device)
    mean, std = T.IMAGENET_MEAN.to(device), T.IMAGENET_STD.to(device)
    lookup = {s["sample_id"]: s for s in manifest["samples"]}

    def abort(*unused):
        raise CaptureComplete()

    def score(rgb, early=True):
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        tensor = tensor.float().div_(255).sub_(mean).div_(std)
        hook = module.register_forward_hook(abort) if early else None
        cap.importance = None
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(tensor, ant)
            assert not early, "capture hook failed to stop"
        except CaptureComplete:
            assert early
        finally:
            if hook is not None:
                hook.remove()
        assert cap.importance is not None
        imp = cap.importance[0]
        n = rgb.shape[0] // 2 * GP
        assert imp.shape == (12, n + GP) and bool(torch.isfinite(imp).all())
        raw = imp[:, :n].sum(0).reshape(-1, GP).float().cpu().numpy()
        head_time = imp[:, :n].reshape(12, -1, GP).sum(-1).float().cpu().numpy()
        totals = imp.sum(-1).float().cpu().numpy()
        target = imp[:, n:].sum(-1).float().cpu().numpy()
        assert abs(totals.sum() / (12 * (n + GP)) - 1) < 0.01
        return raw, head_time, totals, target

    first = manifest["samples"][0]
    rgb = np.load(first["cache"]["path"], mmap_mode="r")
    parity = {}
    anchor = {}
    for seconds in (4, 16):
        b = score(np.array(rgb[-seconds * FPS:]), early=True)
        anchor[f"score_{seconds}s"] = b[0]
        if args.parity:
            a = score(np.array(rgb[-seconds * FPS:]), early=False)
            parity[f"full_vs_early_{seconds}s"] = all(np.array_equal(x, y) for x, y in zip(a, b))
            assert parity[f"full_vs_early_{seconds}s"]
    np.savez(args.out / "hardware_anchor.npz", **anchor)
    if args.reference:
        reference = np.load(args.reference)
        for key, value in anchor.items():
            expected = reference[key]
            relative_l2 = float(np.linalg.norm(value - expected) / np.linalg.norm(expected))
            k = value.size // 4
            a_idx = set(np.argsort(-value.reshape(-1), kind="stable")[:k].tolist())
            b_idx = set(np.argsort(-expected.reshape(-1), kind="stable")[:k].tolist())
            jaccard = len(a_idx & b_idx) / len(a_idx | b_idx)
            parity[key + "_hardware"] = dict(relative_l2=relative_l2, top25_jaccard=jaccard)
            assert relative_l2 <= 0.005 and jaccard >= 0.98, "cross-GPU numeric parity gate failed"
    metadata = dict(protocol=PROTOCOL, manifest=str(args.manifest), manifest_sha256=digest(args.manifest),
                    start=args.start, stop=stop, tag=args.tag, job_id=os.environ.get("SLURM_JOB_ID"),
                    gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda=torch.version.cuda,
                    dtype="bf16", query_chunk=256, parity=parity, model=manifest["model"],
                    code_sha256=manifest["code_sha256"], score="predictor L0 all-query all-head received attention; original BF16 chunk reductions; FP32 accumulator",
                    timing_start=time.time())
    write_json(args.out / "metadata.json", metadata)
    print("Q3_COMPUTE_START " + str(metadata["timing_start"]), flush=True)
    times = []
    with (args.out / "timings.jsonl").open("w") as timing_file:
        for index in range(args.start, stop):
            sample = manifest["samples"][index]
            donor_sample = lookup[sample["donor_sample_id"]]
            for s in (sample, donor_sample):
                assert digest(s["cache"]["path"]) == s["cache"]["sha256"]
            receiver = np.load(sample["cache"]["path"], mmap_mode="r")
            donor = np.load(donor_sample["cache"]["path"], mmap_mode="r")
            arrays = {}
            sample_start = time.monotonic()
            for cond in conditions():
                start = time.monotonic()
                raw, head_time, total, target = score(construct(receiver, donor, cond))
                for prefix, value in (("score", raw), ("head_time", head_time), ("total", total), ("target", target)):
                    arrays[prefix + "__" + cond["name"]] = value
                elapsed = time.monotonic() - start
                timing_file.write(json.dumps(dict(index=index, condition=cond["name"], seconds=elapsed)) + "\n")
                timing_file.flush()
            arrays["sample_index"] = np.asarray(index, dtype=np.int64)
            path = args.out / f"{sample['sample_id']}.npz"
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.rename(path)
            elapsed = time.monotonic() - sample_start
            times.append(elapsed)
            print(f"Q3_ROW {index} {sample['sample_id']} seconds={elapsed:.3f}", flush=True)
    cap.remove()
    summary = dict(protocol=PROTOCOL, run_status="completed", n_rows=stop-args.start,
                   n_conditions=len(conditions()), start=args.start, stop=stop,
                   sample_seconds=times, elapsed_compute_seconds=time.time()-metadata["timing_start"],
                   peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
                   timing_end=time.time(), parity=parity)
    write_json(args.out / "summary.json", summary)
    print("Q3_COMPUTE_END " + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("train-csv", "video-root", "checkpoint", "encoder-lora", "predictor-lora", "parent", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("gpu")
    for name in ("manifest", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=0)
    p.add_argument("--tag", required=True)
    p.add_argument("--parity", action="store_true")
    p.add_argument("--reference", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "gpu": gpu}[args.command](args)


if __name__ == "__main__":
    main()
