#!/usr/bin/env python3
"""B18 Q3 exact diagonal readout; run only in a Slurm project container."""
from __future__ import annotations
from app.hdepic_lora_action_anticipation.share_paths import DATA_ROOT as SHARE_DATA_ROOT, VJEPA_ROOT as SHARE_VJEPA_ROOT
import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path
import numpy as np
from app.hdepic_lora_action_anticipation.probe_b18_q3_boundaries import digest, write_json, CaptureComplete
from app.hdepic_lora_action_anticipation.probe_b18_q3_oldest_mechanism import bytehash

PROTOCOL = "b18-predictor-prune/egtea-q3-token-self-return-v1"
GROUP = "B18-predictor-prune-keep-pattern-causes"
GP, NS, NH = 256, 64, 12
SUFFIXES = ("actual", "no_rope_readout")
ROW_FIELDS = ("self", "same_slot", "same_slot_other", "other_context", "context", "target", "all")
ROOT = Path(str(SHARE_DATA_ROOT))
OLD = ROOT / "outputs/attn_corner_sink/q3_oldest_mechanism"


def compare_scores(actual, expected):
    rel = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    ids = [set(np.argsort(-z.ravel(), kind="stable")[:4096]) for z in (actual, expected)]
    jac = len(ids[0] & ids[1]) / len(ids[0] | ids[1])
    return dict(relative_l2=rel, topk_jaccard=jac, passed=rel <= .005 and jac >= .98)


def prepare(args):
    source = json.loads(args.source_manifest.read_text())
    prior_path = OLD / "prepared/manifest.json"
    prior = json.loads(prior_path.read_text())
    assert source["n_videos"] == 86 and source["n_participants"] == 32
    assert prior["source_manifest_sha256"] == digest(args.source_manifest)
    assert not args.out.exists()
    (args.out / "reference").mkdir(parents=True)
    dependencies = dict(prior["code_sha256"])
    for p, sha in dependencies.items():
        assert digest(p) == sha, f"legacy source changed: {p}"
    dependencies[str(Path(__file__).resolve())] = digest(__file__)
    for model in source["model"].values():
        assert digest(model["path"]) == model["sha256"]
    rows = []
    for i, sample in enumerate(source["samples"]):
        assert digest(sample["cache"]["path"]) == sample["cache"]["sha256"]
        rgb = np.load(sample["cache"]["path"], mmap_mode="r")[-128:]
        assert rgb.shape == (128, 256, 256, 3) and rgb.dtype == np.uint8
        run = "b18-q3-mechanism-full-a-17092450" if i < 43 else "b18-q3-mechanism-full-b-17092454"
        archive = OLD / run / "capture" / (sample["sample_id"] + ".npz")
        with np.load(archive) as a:
            arrays = {k + "_" + s: a[k + "_" + s + "__continuous"] for s in SUFFIXES for k in ("score", "flow")}
        ref = args.out / "reference" / (sample["sample_id"] + ".npz")
        np.savez_compressed(ref, **arrays)
        rows.append(dict(sample_id=sample["sample_id"], index=i, input_sha256=bytehash(rgb),
                         archive=str(archive), archive_sha256=digest(archive),
                         reference=str(ref), reference_sha256=digest(ref)))
    write_json(args.out / "manifest.json", dict(protocol=PROTOCOL, group=GROUP, source=source, rows=rows,
        source_manifest=str(args.source_manifest), source_manifest_sha256=digest(args.source_manifest),
        prior_manifest=str(prior_path), prior_manifest_sha256=digest(prior_path), code_sha256=dependencies,
        conditions=["continuous"], score="L0 all queries/heads; BF16 autocast, FP32 softmax/chunk256 legacy and matched FP32 decomposition",
        preprocessing="exact original last128 cached uint8 RGB; ImageNet normalization; 16s/8FPS/64 slots; target position72",
        row_fields=ROW_FIELDS, probability_atol=2e-6, column_closure_atol=.01, flow_closure_atol=.03))
    print("SELF_PREPARED", len(rows), flush=True)


class SelfReturnCapture:
    def __init__(self, base):
        self.m = base.predictor.predictor_blocks[0].attn
        self.original = self.m.forward
        self.data = {}

    def install(self):
        self.m.forward = self.forward

    def remove(self):
        self.m.forward = self.original

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        import torch
        from src.models.utils.modules import rotate_queries_or_keys
        m = self.m
        out = self.original(x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        B, N, _ = x.shape
        assert B == 1 and N == 65 * GP and m.num_heads == NH
        assert not m.is_causal and attn_mask is None and mask is not None
        qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q0, k0 = qkv[0], qkv[1]
        mp = mask.unsqueeze(1).repeat(1, NH, 1)
        pos = m.separate_positions(mp, H_patches, W_patches)
        offset, qs, ks = 0, [], []
        for dim, p in zip((m.d_dim, m.h_dim, m.w_dim), pos):
            qs.append(rotate_queries_or_keys(q0[..., offset:offset + dim], pos=p))
            ks.append(rotate_queries_or_keys(k0[..., offset:offset + dim], pos=p))
            offset += dim
        if offset < m.head_dim:
            qs.append(q0[..., offset:]); ks.append(k0[..., offset:])
        q, k = torch.cat(qs, -1), torch.cat(ks, -1)
        self.data = {"position_ids": mp[0, 0]}
        assert torch.equal(mp[0, 0, :NS * GP], torch.arange(NS * GP, device=x.device))
        assert torch.equal(mp[0, 0, NS * GP:], torch.arange(72 * GP, 73 * GP, device=x.device))
        ix = torch.arange(GP, device=x.device)
        for suffix, qq, kk in (("actual", q, k), ("no_rope_readout", q0, k0)):
            legacy = torch.zeros(1, NH, N, device=x.device, dtype=torch.float32)
            columns, excluded = torch.zeros_like(legacy[0]), torch.zeros_like(legacy[0])
            legacy_flow, fp32_flow, diagonals = [], [], []
            rows = {f: [] for f in ROW_FIELDS}
            for ci in range(0, N, GP):
                probs = ((qq[:, :, ci:ci + GP] @ kk.transpose(-2, -1)) * m.scale).softmax(dim=-1)
                assert probs.dtype == torch.float32  # Actual legacy CUDA autocast softmax
                received = probs.sum(2).float()
                legacy += received
                legacy_flow.append(received[0].reshape(NH, 65, GP).sum(-1))
                pf = probs[0].float()
                diag = pf[:, ix, ci + ix].clone()
                diagonals.append(diag)
                columns += pf.sum(1)
                fp32_flow.append(pf.reshape(NH, GP, 65, GP).sum(-1).sum(1))
                if ci < NS * GP:
                    total = pf.sum(-1)
                    context = pf[:, :, :NS * GP].sum(-1)
                    same = pf[:, :, ci:ci + GP].sum(-1)
                    target = pf[:, :, NS * GP:].sum(-1)
                    other = pf[:, :, :ci].sum(-1) + pf[:, :, ci + GP:NS * GP].sum(-1)
                # Directly zero SAME FP32 probability diagonal, then sum.
                # Never subtract a diagonal from the legacy chunked columns.
                pf[:, ix, ci + ix] = 0
                excluded += pf.sum(1)
                if ci < NS * GP:
                    same_other = pf[:, :, ci:ci + GP].sum(-1)
                    values = dict(self=diag, same_slot=same, same_slot_other=same_other,
                                  other_context=other, context=context, target=target, all=total)
                    for name, value in values.items():
                        rows[name].append(value)
                    assert bool((diag <= same + 2e-6).all() and (same <= context + 2e-6).all() and (context <= total + 2e-6).all())
                    assert torch.allclose(diag + same_other + other + target, total, atol=2e-6, rtol=0)
                del pf, probs
            self.data["head_legacy_" + suffix] = legacy[0]
            self.data["head_fp32_" + suffix] = columns
            self.data["head_self_excluded_fp32_" + suffix] = excluded
            self.data["flow_legacy_" + suffix] = torch.stack(legacy_flow, 1)
            self.data["flow_fp32_" + suffix] = torch.stack(fp32_flow, 1)
            self.data["diagonal_all_queries_" + suffix] = torch.stack(diagonals, 1)
            for name, values in rows.items():
                self.data["row_" + name + "_" + suffix] = torch.stack(values, 1)
            for route, imp in (("legacy", legacy[0]), ("fp32", columns), ("self_excluded_fp32", excluded)):
                self.data["score_" + route + "_" + suffix] = imp[:, :NS * GP].sum(0).reshape(NS, GP)
        return out


def gpu(args):
    import torch
    from app.hdepic_lora_action_anticipation import train_stream_mtp as T
    from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import HeadAttnCapture20, build_finetuned_20
    torch.set_num_threads(1); torch.manual_seed(20260907)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    manifest = json.loads(args.manifest.read_text()); source = manifest["source"]
    assert manifest["protocol"] == PROTOCOL and 0 <= args.start < args.stop <= 86
    for p, sha in manifest["code_sha256"].items(): assert digest(p) == sha, p
    args.out.mkdir(parents=True, exist_ok=False)
    paths = {k: v["path"] for k, v in source["model"].items()}
    loadlog = io.StringIO()
    with contextlib.redirect_stdout(loadlog):
        base, _ = build_finetuned_20(torch.device("cuda"), max_frames=256, fps=8, img_size=256,
            checkpoint=paths["checkpoint"], enc_lora=paths["encoder_lora"], pred_lora=paths["predictor_lora"], parent_ckpt=paths["parent"])
    print(loadlog.getvalue(), flush=True)
    assert "[finetuned] parent load: 0 missing, 0 unexpected" in loadlog.getvalue()
    base.requires_grad_(False)
    assert base.num_steps == 1 and base.tubelet_size == 2
    model = T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).eval()
    trace = SelfReturnCapture(base)
    ant = torch.full((1,), 2., device="cuda")
    mean, std = T.IMAGENET_MEAN.cuda(), T.IMAGENET_STD.cuda()
    def rgb_at(i):
        rgb = np.array(np.load(source["samples"][i]["cache"]["path"], mmap_mode="r")[-128:])
        assert bytehash(rgb) == manifest["rows"][i]["input_sha256"]
        return rgb
    def run(rgb, early=True):
        x = torch.from_numpy(rgb).permute(3, 0, 1, 2).unsqueeze(0).cuda().float()
        x = x.div_(255).sub_(mean).div_(std)
        def abort(*unused): raise CaptureComplete()
        hook = trace.m.register_forward_hook(abort) if early else None
        result = None
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(x, ant)
            assert not early
        except CaptureComplete:
            assert early
        finally:
            if hook is not None: hook.remove()
        return result
    def equal_tree(a, b):
        if isinstance(a, torch.Tensor): return torch.equal(a, b)
        if isinstance(a, dict): return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
        return a == b
    gates, anchors = {}, {}
    for i in range(2):
        rgb = rgb_at(i)
        ref = HeadAttnCapture20(trace.m, chunk_size=GP)
        run(rgb); expected = ref.importance[0].clone()
        original_output = run(rgb, early=False) if args.parity else None
        if args.parity: assert torch.equal(expected, ref.importance[0])
        ref.remove(); trace.install()
        # Wrap instrumentation with independent legacy capture: same model execution.
        paired = HeadAttnCapture20(trace.m, chunk_size=GP)
        run(rgb)
        assert torch.equal(trace.data["head_legacy_actual"], paired.importance[0])
        assert torch.equal(expected, trace.data["head_legacy_actual"])
        paired.remove()
        early_data = {k: v.clone() for k, v in trace.data.items()}
        if args.parity:
            instrumented_output = run(rgb, early=False)
            assert equal_tree(original_output, instrumented_output)
            assert all(torch.equal(v, trace.data[k]) for k, v in early_data.items())
        anchors[f"q3_{i:03d}"] = trace.data["score_legacy_actual"].cpu().numpy()
        with np.load(manifest["rows"][i]["reference"]) as archived:
            archive_gate = compare_scores(anchors[f"q3_{i:03d}"], archived["score_actual"])
            assert archive_gate["passed"]
            flow_exact = np.array_equal(trace.data["flow_legacy_actual"].cpu().numpy(), archived["flow_actual"])
            assert np.allclose(trace.data["flow_legacy_actual"].cpu().numpy(), archived["flow_actual"], rtol=.005, atol=.03)
        gates[f"q3_{i:03d}"] = dict(same_execution_reference_exact=True, original_instrumented_exact=True,
            full_early_and_model_output_exact=bool(args.parity), archive=archive_gate, archived_slot_flow_exact=flow_exact)
        trace.remove()
    np.savez_compressed(args.out / "hardware_anchors.npz", **anchors)
    if args.smoke_reference:
        expected = np.load(args.smoke_reference)
        gates["hardware"] = {k: compare_scores(v, expected[k]) for k, v in anchors.items()}
        assert all(g["passed"] for g in gates["hardware"].values())
    write_json(args.out / "gate_report.json", gates)
    trace.install()
    if args.benchmark_seconds:
        begin = time.time(); count = 0
        while time.time() - begin < args.benchmark_seconds or count < 4:
            run(rgb_at(count % 2)); torch.cuda.synchronize(); count += 1
        write_json(args.out / "benchmark.json", dict(start=begin, end=time.time(), repeats=count,
                   seconds=time.time() - begin, seconds_per_capture=(time.time() - begin) / count))
    meta = dict(protocol=PROTOCOL, group=GROUP, job_id=os.environ.get("SLURM_JOB_ID"), tag=args.tag,
        start=args.start, stop=args.stop, manifest=str(args.manifest), manifest_sha256=digest(args.manifest),
        code_sha256=manifest["code_sha256"], model=source["model"], gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__, cuda=torch.version.cuda, timing_start=time.time(), gates=gates,
        dtype="BF16 autocast, FP32 softmax; legacy FP32 query reductions and accumulator; new matched FP32 sums",
        is_causal=False, attn_mask=None, mask="position IDs, context0:16384 and target18432:18688",
        axes="row arrays [head12,contextslot64,spatial256]; diagonal [12,65,256]; flow [12,query65,key65]")
    write_json(args.out / "metadata.json", meta)
    print("SELF_COMPUTE_START", meta["timing_start"], flush=True)
    times, archive_gates, artifacts = [], {}, []
    with (args.out / "timings.jsonl").open("w") as timing:
        for i in range(args.start, args.stop):
            start = time.monotonic(); run(rgb_at(i))
            arrays = {k: v.float().cpu().numpy() for k, v in trace.data.items()}
            assert all(np.isfinite(v).all() for v in arrays.values())
            arrays["sample_index"] = np.asarray(i)
            with np.load(manifest["rows"][i]["reference"]) as archived:
                gate = compare_scores(arrays["score_legacy_actual"], archived["score_actual"])
                assert gate["passed"]
                gate["flow_exact"] = bool(np.array_equal(arrays["flow_legacy_actual"], archived["flow_actual"]))
                assert np.allclose(arrays["flow_legacy_actual"], archived["flow_actual"], rtol=.005, atol=.03)
            archive_gates[str(i)] = gate
            dest = args.out / (source["samples"][i]["sample_id"] + ".npz")
            temp = dest.with_suffix(".partial.npz")
            np.savez_compressed(temp, **arrays); temp.rename(dest)
            elapsed = time.monotonic() - start; times.append(elapsed)
            artifacts.append(dict(path=str(dest), sha256=digest(dest), index=i))
            timing.write(json.dumps(dict(index=i, seconds=elapsed)) + "\n"); timing.flush()
            print("SELF_ROW", i, f"seconds={elapsed:.3f}", flush=True)
    trace.remove()
    summary = dict(protocol=PROTOCOL, run_status="completed", start=args.start, stop=args.stop,
        n_rows=args.stop - args.start, sample_seconds=times, timing_end=time.time(),
        elapsed_compute_seconds=time.time() - meta["timing_start"], peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
        archive_gates=archive_gates, artifacts=artifacts, gates=gates)
    write_json(args.out / "summary.json", summary)
    print("SELF_COMPUTE_END", json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--source-manifest", type=Path, required=True); q.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("gpu")
    for name in ("manifest", "out"): q.add_argument("--" + name, type=Path, required=True)
    q.add_argument("--start", type=int, default=0); q.add_argument("--stop", type=int, default=2)
    q.add_argument("--tag", required=True); q.add_argument("--parity", action="store_true")
    q.add_argument("--smoke-reference", type=Path); q.add_argument("--benchmark-seconds", type=float, default=0)
    args = p.parse_args(); {"prepare": prepare, "gpu": gpu}[args.command](args)


if __name__ == "__main__": main()
