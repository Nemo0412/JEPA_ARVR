#!/usr/bin/env python3
"""Frozen B18 Q3 oldest-slot interventions and compact L0 signal tracing.

Run only in the project container on Slurm. No upstream model is edited.
Actual attention score uses the historical BF16 chunk reductions unchanged.
The no-RoPE result is a readout of the same q/k, not a model intervention.
"""
from __future__ import annotations
from app.hdepic_lora_action_anticipation.share_paths import VJEPA_ROOT as SHARE_VJEPA_ROOT
import argparse
import contextlib
import hashlib
import io
import json
import os
import time
from pathlib import Path
import numpy as np

from app.hdepic_lora_action_anticipation.probe_b18_q3_boundaries import digest, write_json, CaptureComplete

PROTOCOL = "b18-predictor-prune/egtea-q3-oldest-slot-mechanism-v1"
GROUP = "B18-predictor-prune-keep-pattern-causes"
SEEDS = (20260907, 20260908, 20260909)
GP, NS = 256, 64
CONDITIONS = ["continuous", "start_noise_20260907", "start_noise_20260908",
              "start_noise_20260909", "start_gray", "start_donor",
              "middle_noise_20260907", "middle_gray", "middle_donor"]


def bytehash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def replacements(sample, donor):
    tiles = {"gray": np.full((4, 256, 256, 3), 128, dtype=np.uint8),
             "donor": np.array(donor[-128:-124], copy=True)}
    for seed in SEEDS:
        derived = int.from_bytes(hashlib.sha256(f"{seed}|{sample['sample_id']}".encode()).digest()[:8], "little")
        tiles[f"noise_{seed}"] = np.random.default_rng(derived).integers(0, 256, size=(4, 256, 256, 3), dtype=np.uint8)
    return tiles


def construct(receiver, tiles, condition):
    rgb = np.array(receiver[-128:], copy=True)
    if condition != "continuous":
        where, kind = condition.split("_", 1)
        start = 0 if where == "start" else 64
        rgb[start:start + 4] = tiles[kind]
        assert np.array_equal(rgb[:start], receiver[-128:][:start])
        assert np.array_equal(rgb[start + 4:], receiver[-128:][start + 4:])
        assert np.array_equal(rgb[start:start + 4], tiles[kind])
    assert rgb.shape == (128, 256, 256, 3) and rgb.dtype == np.uint8
    return rgb


def prepare(args):
    source = json.loads(args.source_manifest.read_text())
    assert source["n_videos"] == 86 and source["n_participants"] == 32
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    lookup = {s["sample_id"]: s for s in source["samples"]}
    checked = set()
    rows = []
    for sample in source["samples"]:
        donor_sample = lookup[sample["donor_sample_id"]]
        for s in (sample, donor_sample):
            if s["sample_id"] not in checked:
                assert digest(s["cache"]["path"]) == s["cache"]["sha256"]
                checked.add(s["sample_id"])
        receiver = np.load(sample["cache"]["path"], mmap_mode="r")
        donor = np.load(donor_sample["cache"]["path"], mmap_mode="r")
        tiles = replacements(sample, donor)
        again = replacements(sample, donor)
        assert all(np.array_equal(v, again[k]) for k, v in tiles.items())
        clips = {c: construct(receiver, tiles, c) for c in CONDITIONS}
        for kind in ("noise_20260907", "gray", "donor"):
            assert np.array_equal(clips["start_" + kind][:4], clips["middle_" + kind][64:68])
        rows.append(dict(sample_id=sample["sample_id"], replacement_sha256={k: bytehash(v) for k, v in tiles.items()},
                         input_sha256={k: bytehash(v) for k, v in clips.items()}))
    for value in source["model"].values():
        assert digest(value["path"]) == value["sha256"]
    dependencies = dict(source["code_sha256"])
    for path, expected in dependencies.items():
        assert digest(path) == expected, f"archived dependency changed: {path}"
    dependencies[str(Path(__file__).resolve())] = digest(__file__)
    root = Path(__file__).resolve().parents[2]
    for rel in ("src/models/utils/modules.py", "src/models/vision_transformer.py", "src/models/predictor.py"):
        dependencies[str(SHARE_VJEPA_ROOT / rel)] = digest(SHARE_VJEPA_ROOT / rel)
    write_json(args.out / "manifest.json", dict(protocol=PROTOCOL, group=GROUP, conditions=CONDITIONS,
               source_manifest=str(args.source_manifest), source_manifest_sha256=digest(args.source_manifest),
               source=source, replacement_seeds=SEEDS, rows=rows, code_sha256=dependencies,
               intervention="first4 RGB frames (slots0/1), middle4 RGB frames (slots32/33); shared exact tiles",
               score="predictor L0 context+target queries; all heads; context keys; K4096; BF16 query-chunk256; FP32 accumulator"))
    print("PREPARED", len(rows), len(CONDITIONS), flush=True)


def tensor_stats(x):
    """Per temporal slot: norm mean/q10/median/q90, mean-vector norm/alignment."""
    import torch
    assert x.ndim == 3 and x.shape[0] == 1 and x.shape[1] % GP == 0
    z = x[0].float().reshape(-1, GP, x.shape[-1])
    norm = z.norm(dim=-1)
    mu = z.mean(1)
    global_mu = z.mean((0, 1))
    cos = (mu * global_mu).sum(-1) / (mu.norm(dim=-1) * global_mu.norm()).clamp_min(1e-12)
    return torch.stack([norm.mean(-1), *torch.quantile(norm, torch.tensor([.1, .5, .9], device=z.device), dim=-1),
                        mu.norm(dim=-1), cos], dim=-1)


class SignalTrace:
    def __init__(self, base):
        self.base, self.m = base, base.predictor.predictor_blocks[0].attn
        self.original = self.m.forward
        self.data, self.enabled = {}, False
        self.handles = []
        stages = {"patch": base.encoder.patch_embed,
                  "encoder_block00": base.encoder.blocks[0],
                  "encoder_block11": base.encoder.blocks[11],
                  "encoder_block23": base.encoder.blocks[23],
                  "encoder_final_norm": base.encoder.norm,
                  "predictor_embed": base.predictor.predictor_embed,
                  "predictor_l0_norm1": base.predictor.predictor_blocks[0].norm1}
        for name, module in stages.items():
            def hook(m, a, output, name=name):
                if self.enabled:
                    self.data["stage_" + name] = tensor_stats(output)
            self.handles.append(module.register_forward_hook(hook))

    def install(self):
        self.m.forward = self.forward

    def remove(self):
        self.m.forward = self.original

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        import torch
        from src.models.utils.modules import rotate_queries_or_keys
        m = self.m
        out = self.original(x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        B, N, C = x.shape
        assert B == 1 and N == 65 * GP and attn_mask is None
        qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q0, k0, v = qkv[0], qkv[1], qkv[2]
        mp = mask.unsqueeze(1).repeat(1, m.num_heads, 1) if mask is not None else torch.arange(N, device=x.device)
        pos = m.separate_positions(mp, H_patches, W_patches)
        s, qs, ks = 0, [], []
        for dim, p in zip((m.d_dim, m.h_dim, m.w_dim), pos):
            qs.append(rotate_queries_or_keys(q0[..., s:s + dim], pos=p))
            ks.append(rotate_queries_or_keys(k0[..., s:s + dim], pos=p))
            s += dim
        if s < m.head_dim:
            qs.append(q0[..., s:]); ks.append(k0[..., s:])
        q, k = torch.cat(qs, -1), torch.cat(ks, -1)
        self.data["positions"] = mp[0, 0].reshape(65, GP)[:, 0] if mp.ndim == 3 else mp.reshape(65, GP)[:, 0]
        for name, a in (("q_pre", q0), ("k_pre", k0), ("v", v), ("q_post", q), ("k_post", k)):
            z = a[0].float().reshape(12, 65, GP, -1)
            norm = z.norm(dim=-1)
            self.data[name + "_norm"] = torch.stack([norm.mean(-1), *torch.quantile(norm, torch.tensor([.1, .5, .9], device=z.device), dim=-1)], -1)
        for suffix, qq, kk in (("actual", q, k), ("no_rope_readout", q0, k0)):
            imp = torch.zeros(B, 12, N, device=x.device, dtype=torch.float32)
            flow, means, stds = [], [], []
            logsum = torch.zeros(12, N, device=x.device, dtype=torch.float32)
            logsq = torch.zeros_like(logsum)
            for ci in range(0, N, GP):
                logits = (qq[:, :, ci:ci + GP] @ kk.transpose(-2, -1)) * m.scale
                probs = logits.softmax(dim=-1)
                received = probs.sum(dim=2).float()
                imp += received
                flow.append(received[0].reshape(12, 65, GP).sum(-1))
                lf = logits[0].float()
                logsum += lf.sum(1)
                logsq += lf.square().sum(1)
                temporal = lf.reshape(12, GP, 65, GP)
                moments = temporal.mean((1, 3))
                means.append(moments)
                stds.append((temporal.square().mean((1, 3)) - moments.square()).clamp_min(0).sqrt())
                del lf, temporal, logits, probs
            flow = torch.stack(flow, dim=1)
            assert torch.allclose(flow.sum(1), imp[0].reshape(12, 65, GP).sum(-1), rtol=1e-5, atol=.03)
            assert abs(float(imp.sum()) / (12 * N) - 1) < .01
            self.data["head_importance_" + suffix] = imp[0]
            self.data["flow_" + suffix] = flow
            self.data["logit_mean_" + suffix] = torch.stack(means, 1)
            self.data["logit_std_" + suffix] = torch.stack(stds, 1)
            mu = logsum / N
            self.data["key_logit_mean_" + suffix] = mu.reshape(12, 65, GP).mean(-1)
            self.data["key_logit_rms_dispersion_" + suffix] = (logsq / N - mu.square()).clamp_min(0).reshape(12, 65, GP).mean(-1).sqrt()
            qmean = qq[0].float().mean(1)
            kmean = kk[0].float().reshape(12, 65, GP, -1).mean(2)
            self.data["qmean_kmean_cos_" + suffix] = (qmean[:, None] * kmean).sum(-1) / (qmean.norm(dim=-1)[:, None] * kmean.norm(dim=-1)).clamp_min(1e-12)
        return out


def gpu(args):
    import torch
    from app.hdepic_lora_action_anticipation import train_stream_mtp as T
    from app.hdepic_lora_action_anticipation.analyze_encoder_head_attn_corners import HeadAttnCapture20, build_finetuned_20
    torch.set_num_threads(1)
    torch.manual_seed(SEEDS[0])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["protocol"] == PROTOCOL and manifest["conditions"] == CONDITIONS
    for path, expected in manifest["code_sha256"].items():
        assert digest(path) == expected, f"frozen capture code changed: {path}"
    source = manifest["source"]
    stop = args.stop or len(source["samples"])
    assert 0 <= args.start < stop <= 86
    args.out.mkdir(parents=True, exist_ok=False)
    paths = {k: v["path"] for k, v in source["model"].items()}
    loadlog = io.StringIO()
    with contextlib.redirect_stdout(loadlog):
        base, _ = build_finetuned_20(torch.device("cuda"), max_frames=256, fps=8, img_size=256,
                                    checkpoint=paths["checkpoint"], enc_lora=paths["encoder_lora"],
                                    pred_lora=paths["predictor_lora"], parent_ckpt=paths["parent"])
    print(loadlog.getvalue(), flush=True)
    assert "[finetuned] parent load: 0 missing, 0 unexpected" in loadlog.getvalue()
    assert base.num_steps == 1 and base.tubelet_size == 2
    base.requires_grad_(False)
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
            if hook is not None:
                hook.remove()
    anchor_rgb = np.array(np.load(source["samples"][0]["cache"]["path"], mmap_mode="r")[-128:])
    ref = HeadAttnCapture20(trace.m, chunk_size=GP)
    run(anchor_rgb)
    reference = ref.importance.clone()
    parity = {}
    if args.parity:
        run(anchor_rgb, early=False)
        assert torch.equal(reference, ref.importance)
        parity["reference_full_early_exact"] = True
    ref.remove()
    trace.install()
    trace.enabled = True
    run(anchor_rgb)
    assert torch.equal(reference[0], trace.data["head_importance_actual"])
    parity["instrumented_reference_same_execution_exact"] = True
    if args.parity:
        traced = {k: v.clone() for k, v in trace.data.items()}
        run(anchor_rgb, early=False)
        assert all(torch.equal(v, trace.data[k]) for k, v in traced.items())
        parity["instrumented_full_early_exact"] = True
    actual = reference[0, :, :NS * GP].sum(0).reshape(NS, GP).cpu().numpy()
    archived = np.load(args.reference)["score__continuous_16s"]
    rel = float(np.linalg.norm(actual - archived) / np.linalg.norm(archived))
    ix = [set(np.argsort(-z.ravel(), kind="stable")[:4096].tolist()) for z in (actual, archived)]
    jac = len(ix[0] & ix[1]) / len(ix[0] | ix[1])
    assert rel <= .005 and jac >= .98
    parity["archived_baseline"] = dict(relative_l2=rel, topk_jaccard=jac)
    np.savez_compressed(args.out / "anchor.npz", score=actual)
    metadata = dict(protocol=PROTOCOL, group=GROUP, tag=args.tag, job_id=os.environ.get("SLURM_JOB_ID"),
                    manifest=str(args.manifest), manifest_sha256=digest(args.manifest), start=args.start, stop=stop,
                    gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda=torch.version.cuda,
                    dtype="bf16", query_chunk=GP, parity=parity, model=source["model"], code_sha256=manifest["code_sha256"],
                    trace_axes="flow/logit temporal arrays [head12,query65,key65]; norms [head12,slot65,mean/q10/median/q90]; stage [slot,mean_norm/q10/median/q90/mean_vector_norm/global_alignment]",
                    rope=dict(head_dim=trace.m.head_dim, temporal_dim=trace.m.d_dim, height_dim=trace.m.h_dim, width_dim=trace.m.w_dim),
                    timing_start=time.time())
    write_json(args.out / "metadata.json", metadata)
    print("MECHANISM_COMPUTE_START", metadata["timing_start"], flush=True)
    lookup = {s["sample_id"]: s for s in source["samples"]}
    times = []
    with (args.out / "timings.jsonl").open("w") as timing:
        for index in range(args.start, stop):
            sample = source["samples"][index]
            receiver = np.load(sample["cache"]["path"], mmap_mode="r")
            donor = np.load(lookup[sample["donor_sample_id"]]["cache"]["path"], mmap_mode="r")
            tiles = replacements(sample, donor)
            assert {k: bytehash(v) for k, v in tiles.items()} == manifest["rows"][index]["replacement_sha256"]
            arrays = {"sample_index": np.asarray(index)}
            row_start = time.monotonic()
            for condition in CONDITIONS:
                start = time.monotonic()
                rgb = construct(receiver, tiles, condition)
                assert bytehash(rgb) == manifest["rows"][index]["input_sha256"][condition]
                trace.data = {}
                run(rgb)
                for name, value in trace.data.items():
                    assert bool(torch.isfinite(value).all()), (index, condition, name)
                    arrays[name + "__" + condition] = value.float().cpu().numpy()
                for suffix in ("actual", "no_rope_readout"):
                    imp = trace.data["head_importance_" + suffix]
                    arrays["score_" + suffix + "__" + condition] = imp[:, :NS * GP].sum(0).reshape(NS, GP).float().cpu().numpy()
                timing.write(json.dumps(dict(index=index, condition=condition, seconds=time.monotonic() - start)) + "\n")
                timing.flush()
            path = args.out / f"{sample['sample_id']}.npz"
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.rename(path)
            elapsed = time.monotonic() - row_start
            times.append(elapsed)
            print("MECHANISM_ROW", index, f"seconds={elapsed:.3f}", flush=True)
    trace.remove()
    summary = dict(protocol=PROTOCOL, run_status="completed", n_rows=stop - args.start, n_conditions=len(CONDITIONS),
                   start=args.start, stop=stop, sample_seconds=times, elapsed_compute_seconds=time.time() - metadata["timing_start"],
                   timing_end=time.time(), peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30, parity=parity)
    write_json(args.out / "summary.json", summary)
    print("MECHANISM_COMPUTE_END", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("gpu")
    for name in ("manifest", "out", "reference"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=2)
    p.add_argument("--tag", required=True)
    p.add_argument("--parity", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "gpu": gpu}[args.command](args)


if __name__ == "__main__":
    main()
