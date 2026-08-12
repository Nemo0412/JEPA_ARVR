#!/usr/bin/env python3
"""System-motivation profiling for streaming JEPA (before inventing optimizations).

Measures *observable* reuse / stability stats on the real half-split stream index
(+ optional disk tick-cache layout), not OpenCL microbench.

Metrics
-------
1) Temporal encoder-cache opportunity
   - Consecutive ticks in a video: overlap of model frame indices / tubelet slots
   - "Hit rate" = |frames ∩ prev| / |frames_now| if previous tick's encoder tokens
     were cached by absolute frame (or tubelet) id.

2) Window dynamics (grow vs slide)
   - Grow phase (ctx rising): new frames only at end?
   - Slide phase (ctx==10s): how many enter / leave per tick?

3) Prune-set stability (policy simulation on indices; no weights needed)
   - keep=4096, gp=256; scorers:
       * recency (slot linear)
       * recent_hard_reserve (e.g. last 2s) + recency on remainder
       * random (baseline noise floor)
   - Jaccard(keep_{t}, keep_{t-1}) under each policy given the tick's N tokens.

4) Disk tick-cache (optional path)
   - Number of files, uniqueness vs implied consecutive-tick key changes
   - Expected hit rate of *exact* (video_id, frame_indices) cache under sequential epoch

Output: JSON + printed tables under scripts/opencl_kernels/ or --out.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(
    "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/"
    "HD_EPIC_val_stream_mtp.csv"
)
DEFAULT_TICK_CACHE = Path(
    "/scratch/ll5914/datasets/HD-EPIC/caches/stream_mtp_concat_ca_ticks_256"
)


def parse_frame_indices(s) -> np.ndarray:
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=np.int64)
    s = str(s).strip()
    if s.startswith("["):
        return np.asarray(ast.literal_eval(s), dtype=np.int64)
    return np.asarray([int(x) for x in s.split(",") if x.strip() != ""], dtype=np.int64)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def frames_to_tubelets(frames: np.ndarray, tubelet_frames: int) -> np.ndarray:
    """Map model frames to tubelet ids (absolute, video-level) when stride≈tubelet."""
    # Model frame indices are already sparsely sampled (~model fps).
    # Treat successive model frames as packing into groups of tubelet_size if contiguous...
    # Safer identity: each model-frame is a "slice"; groups of (grid spatial) handled separately.
    # For cache granularity we use unique frame ids and also "rank buckets".
    return frames // max(1, tubelet_frames)


def prune_keep_indices(
    n_tokens: int,
    gp: int,
    keep: int,
    scores: np.ndarray,
    recent_keep_tokens: int = 0,
) -> np.ndarray:
    """Match train_stream_mtp_concat_ca._prune_video_suffix logic (single sample)."""
    K = min(keep, max(gp, (keep // gp) * gp))
    K = min(K, (n_tokens // gp) * gp) if n_tokens >= gp else n_tokens
    if K >= n_tokens:
        return np.arange(n_tokens)
    recent_n = min(recent_keep_tokens, K)
    recent_n = (recent_n // gp) * gp if gp > 0 else recent_n
    recent_n = min(recent_n, K)
    if recent_n >= K:
        return np.arange(n_tokens - K, n_tokens)
    if recent_n > 0:
        older_n = n_tokens - recent_n
        need = K - recent_n
        older_scores = scores[:older_n]
        older_idx = np.argpartition(-older_scores, need - 1)[:need]
        older_idx = np.sort(older_idx)
        recent_idx = np.arange(older_n, n_tokens)
        return np.concatenate([older_idx, recent_idx])
    idx = np.argpartition(-scores, K - 1)[:K]
    return np.sort(idx)


def recency_scores(n_tokens: int, gp: int) -> np.ndarray:
    n_slots = max(1, (n_tokens + gp - 1) // gp)
    slot = np.arange(n_tokens) // gp
    return ((slot.astype(np.float64) + 1.0) / float(n_slots)).astype(np.float64)


def simulate_postfuse_like_scores(n_tokens: int, gp: int, rng: np.random.Generator, mix: float = 0.5) -> np.ndarray:
    """Cheap surrogate: recency + spatially shared noise per tubelet slot (AR-ish)."""
    rec = recency_scores(n_tokens, gp)
    n_slots = max(1, (n_tokens + gp - 1) // gp)
    slot_noise = rng.standard_normal(n_slots)
    noise = slot_noise[np.arange(n_tokens) // gp]
    # also tiny per-token noise
    noise = noise + 0.1 * rng.standard_normal(n_tokens)
    # mixture
    s = mix * rec + (1.0 - mix) * (noise - noise.min() + 1e-6)
    return s.astype(np.float64)


def profile_stream_csv(
    csv_path: Path,
    max_videos: int = 50,
    gp: int = 256,
    keep: int = 4096,
    tokens_per_model_frame: int | None = None,
    model_fps: float = 4.0,
    recent_keep_sec: float = 2.0,
    tubelet_sec: float = 0.5,
    seed: int = 0,
) -> dict:
    df = pd.read_csv(csv_path)
    # tokens_per_model_frame: patch grid. For 256px, patch16 → 16x16=256 spatial, 1 tubelet often
    # packs tubelet_size frames. Stream index uses n_model_frames; encoder N ≈ n_model_frames * (H*W / something)
    # In vjepa, num_frames=T, spatial gp = grid^2. Actually tokens N = (T/tubelet) * grid * grid.
    # n_model_frames in CSV is number of sampled frames fed to model (e.g. 32 for 4s @4fps).
    # With tubelet_size=2, T_tubelets = n_model_frames / 2, spatial=256 → N = n_frames/2 * 256.
    # Confirm common recipe: 10s@4fps → 40 frames, tubelet2 → 20, *256 = 5120.
    if tokens_per_model_frame is None:
        # effective tokens per model frame assuming tubelet_size=2, spatial 256:
        # N = (n_model_frames / 2) * 256 = n_model_frames * 128
        tokens_per_model_frame = 128

    rng = np.random.default_rng(seed)

    hit_frame = []
    hit_tubelet = []
    new_frac = []
    drop_frac = []
    ctx_list = []
    n_frames_list = []
    phase = []  # grow vs slide

    # prune jaccard consecutive (mapped into absolute video token axis via start offset)
    jacc_recency = []
    jacc_recency_hard = []
    jacc_postfuse_like = []
    jacc_random = []
    hard_reserve_frac = []  # recent_n / K

    videos = df["video_id"].unique()
    if max_videos > 0:
        videos = videos[:max_videos]

    per_video_summary = []
    n_pairs = 0

    for vid in videos:
        g = df[df["video_id"] == vid].sort_values("tick_frame")
        if len(g) < 2:
            continue
        prev_frames = None
        prev_keep = {"rec": None, "hard": None, "pf": None, "rnd": None}
        prev_start = None
        v_hits_f, v_hits_t = [], []

        for _, row in g.iterrows():
            frames = parse_frame_indices(row["frame_indices"])
            ctx = float(row["context_sec"])
            n_mf = int(row.get("n_model_frames", len(frames)))
            start_f = int(row["start_frame"])
            tick_f = int(row["tick_frame"])

            # token N estimate
            N = int(n_mf * tokens_per_model_frame)
            # align to gp
            N = (N // gp) * gp
            if N < gp:
                continue

            ctx_list.append(ctx)
            n_frames_list.append(len(frames))
            if prev_frames is not None:
                a, b = set(frames.tolist()), set(prev_frames.tolist())
                hit = len(a & b) / max(1, len(a))
                new = len(a - b) / max(1, len(a))
                drop = len(b - a) / max(1, len(b))
                hit_frame.append(hit)
                new_frac.append(new)
                drop_frac.append(drop)
                v_hits_f.append(hit)

                ta = set((frames // 15).tolist())
                tb = set((prev_frames // 15).tolist())
                th = len(ta & tb) / max(1, len(ta))
                hit_tubelet.append(th)
                v_hits_t.append(th)
                n_pairs += 1

            # Prune policies on local sequence [0..N)
            rec = recency_scores(N, gp)
            recent_slots = max(1, int(round(recent_keep_sec / tubelet_sec)))
            recent_n = min(N, recent_slots * gp)
            recent_n = (recent_n // gp) * gp
            K = min(keep, (N // gp) * gp)
            hard_reserve_frac.append(min(recent_n, K) / max(1, K) if N > K else 1.0)

            keep_rec = set(prune_keep_indices(N, gp, keep, rec, recent_keep_tokens=0).tolist())
            keep_hard = set(
                prune_keep_indices(N, gp, keep, rec, recent_keep_tokens=recent_n).tolist()
            )
            pf = simulate_postfuse_like_scores(N, gp, rng, mix=0.45)
            keep_pf = set(prune_keep_indices(N, gp, keep, pf, recent_keep_tokens=recent_n).tolist())
            rnd = rng.random(N)
            keep_rnd = set(prune_keep_indices(N, gp, keep, rnd, recent_keep_tokens=0).tolist())

            # Map keep sets into absolute keys so consecutive ticks with slide can compare.
            # Abs token key ≈ (start_frame, local_idx) but local token grid depends on packing.
            # Approximate absolute key: local_slot + start_frame's slot offset.
            slot_offset = start_f // 15  # coarse absolute time bucket

            def absify(local_idx_set: set[int]) -> set[tuple]:
                out = set()
                for li in local_idx_set:
                    slot = li // gp
                    out.add((slot_offset + slot, li % gp))  # (abs_slot, spatial)
                return out

            if prev_frames is not None and prev_keep["rec"] is not None:
                jacc_recency.append(jaccard(absify(keep_rec), prev_keep["rec"]))
                jacc_recency_hard.append(jaccard(absify(keep_hard), prev_keep["hard"]))
                jacc_postfuse_like.append(jaccard(absify(keep_pf), prev_keep["pf"]))
                jacc_random.append(jaccard(absify(keep_rnd), prev_keep["rnd"]))

            prev_keep = {
                "rec": absify(keep_rec),
                "hard": absify(keep_hard),
                "pf": absify(keep_pf),
                "rnd": absify(keep_rnd),
            }
            prev_frames = frames
            prev_start = start_f

        if v_hits_f:
            per_video_summary.append(
                {
                    "video_id": str(vid),
                    "n_ticks": int(len(g)),
                    "mean_frame_hit": float(np.mean(v_hits_f)),
                    "mean_tubelet_hit": float(np.mean(v_hits_t)),
                }
            )

    def mean_std(xs):
        if not xs:
            return {"mean": None, "std": None, "n": 0}
        a = np.asarray(xs, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size), "p50": float(np.median(a))}

    # Phase-conditional frame hit
    by_phase = defaultdict(list)
    # rebuild phase hits carefully
    # simpler: re-scan not stored phase with hit; store paired was incomplete.
    # Quick second pass for phase stats:
    phase_hits = {"grow": [], "slide": []}
    for vid in videos:
        g = df[df["video_id"] == vid].sort_values("tick_frame")
        prev = None
        prev_start = None
        prev_ctx = None
        for _, row in g.iterrows():
            frames = parse_frame_indices(row["frame_indices"])
            ctx = float(row["context_sec"])
            start_f = int(row["start_frame"])
            if prev is not None:
                hit = len(set(frames.tolist()) & set(prev.tolist())) / max(1, len(frames))
                is_slide = start_f != prev_start and abs(ctx - prev_ctx) < 1e-6
                is_grow = ctx > prev_ctx + 1e-6
                if is_grow:
                    phase_hits["grow"].append(hit)
                elif is_slide or abs(ctx - 10.0) < 1e-6:
                    phase_hits["slide"].append(hit)
            prev = frames
            prev_start = start_f
            prev_ctx = ctx

    # Encoder flop opportunity: if only encode new frames, compute saved fraction
    # per tick: save ≈ hit_rate of frames (approx for patches proportional to frames)
    flop_save = {
        "if_temporal_encoder_cache_mean_save": mean_std(hit_frame)["mean"],
        "note": "Upper bound if encoder activations for overlapping frames are reusable bit-exact.",
    }

    return {
        "csv": str(csv_path),
        "videos_profiled": len(per_video_summary),
        "pairs": n_pairs,
        "config": {
            "gp": gp,
            "keep": keep,
            "tokens_per_model_frame_est": tokens_per_model_frame,
            "recent_keep_sec": recent_keep_sec,
            "tubelet_sec": tubelet_sec,
        },
        "temporal_cache_opportunity": {
            "frame_level_hit_rate": mean_std(hit_frame),
            "tubelet_bucket_hit_rate": mean_std(hit_tubelet),
            "new_frame_fraction": mean_std(new_frac),
            "dropped_frame_fraction_prev": mean_std(drop_frac),
            "by_phase_frame_hit": {k: mean_std(v) for k, v in phase_hits.items()},
            "encoder_flop_save_upper_bound": flop_save,
        },
        "prune_stability_jaccard_abs_keys": {
            "recency_only": mean_std(jacc_recency),
            "recency_plus_hard_recent": mean_std(jacc_recency_hard),
            "postfuse_like_plus_hard": mean_std(jacc_postfuse_like),
            "random": mean_std(jacc_random),
            "hard_reserve_fraction_of_K": mean_std(hard_reserve_frac),
        },
        "context_distribution": {
            "unique_context_sec": sorted({float(x) for x in ctx_list}),
            "mean_n_model_frames_proxy": mean_std(n_frames_list),
        },
        "per_video_head": per_video_summary[:10],
    }


def profile_tick_disk_cache(cache_dir: Path, csv_path: Path, max_rows: int = 5000) -> dict:
    """Exact-key tick cache: consecutive ticks almost always change frame set → miss."""
    if not cache_dir.is_dir():
        return {"exists": False, "path": str(cache_dir)}

    files = list(cache_dir.glob("*.pt"))
    n_files = len(files)
    # size sample
    sample = files[:200]
    sizes = [p.stat().st_size for p in sample] if sample else []

    df = pd.read_csv(csv_path)
    if max_rows > 0:
        df = df.head(max_rows)

    def cache_key(video_id: str, frame_idx: np.ndarray) -> str:
        payload = f"{video_id}|{','.join(map(str, frame_idx.tolist()))}"
        return hashlib.md5(payload.encode()).hexdigest()

    keys = []
    for _, row in df.iterrows():
        frames = parse_frame_indices(row["frame_indices"])
        keys.append(cache_key(str(row["video_id"]), frames))

    # Existence hits for this CSV subset
    present = 0
    for k in keys:
        if (cache_dir / f"{k}.pt").is_file():
            present += 1

    # Consecutive exact-key equality (should be ~0)
    same_as_prev = 0
    pairs = 0
    for vid, g in df.groupby("video_id"):
        g = g.sort_values("tick_frame")
        prev = None
        for _, row in g.iterrows():
            frames = parse_frame_indices(row["frame_indices"])
            k = cache_key(str(row["video_id"]), frames)
            if prev is not None:
                pairs += 1
                if k == prev:
                    same_as_prev += 1
            prev = k

    return {
        "exists": True,
        "path": str(cache_dir),
        "n_cache_files": n_files,
        "avg_file_bytes_sample200": float(np.mean(sizes)) if sizes else None,
        "csv_rows_checked": len(keys),
        "exact_key_hit_rate_csv_subset": present / max(1, len(keys)),
        "consecutive_ticks_identical_key_rate": same_as_prev / max(1, pairs),
        "interpretation": (
            "Exact frame-list tick cache helps *epoch replay* (re-hit same tick), "
            "NOT temporal incremental encode across consecutive ticks "
            "(keys change every tick → consecutive identity ~0)."
        ),
    }


def print_report(rep: dict, tick: dict) -> None:
    tc = rep["temporal_cache_opportunity"]
    pr = rep["prune_stability_jaccard_abs_keys"]
    print("\n========== STREAM SYSTEM MOTIVATION PROFILE ==========")
    print(f"csv: {rep['csv']}")
    print(f"videos={rep['videos_profiled']} consecutive_tick_pairs={rep['pairs']}")
    print("\n--- 1) Temporal reuse (encoder cache opportunity) ---")
    print(f"  frame hit rate          mean={tc['frame_level_hit_rate']['mean']:.3f}  "
          f"p50={tc['frame_level_hit_rate']['p50']:.3f}  (n={tc['frame_level_hit_rate']['n']})")
    print(f"  tubelet-bucket hit rate mean={tc['tubelet_bucket_hit_rate']['mean']:.3f}")
    print(f"  new frames / tick       mean={tc['new_frame_fraction']['mean']:.3f}")
    print(f"  dropped from prev       mean={tc['dropped_frame_fraction_prev']['mean']:.3f}")
    for ph, st in tc["by_phase_frame_hit"].items():
        if st["mean"] is not None:
            print(f"  [{ph:5s}] frame hit mean={st['mean']:.3f} n={st['n']}")
    print(f"  ⇒ encoder FLOP save upper bound ≈ {tc['encoder_flop_save_upper_bound']['if_temporal_encoder_cache_mean_save']:.1%} "
          f"if cached activations reused")

    print("\n--- 2) Prune-set stability across consecutive ticks (Jaccard) ---")
    for name, st in pr.items():
        if st["mean"] is None:
            continue
        print(f"  {name:28s} mean={st['mean']:.3f} p50={st['p50']:.3f}")

    print("\n--- 3) Disk tick cache (exact key) ---")
    if not tick.get("exists"):
        print("  (no cache dir)")
    else:
        print(f"  files≈{tick['n_cache_files']}  path={tick['path']}")
        print(f"  exact hit on CSV subset: {tick['exact_key_hit_rate_csv_subset']:.3f}")
        print(f"  consecutive ticks same key: {tick['consecutive_ticks_identical_key_rate']:.4f}")
        print(f"  note: {tick['interpretation']}")

    print("\n--- Motivation read-out ---")
    fh = tc["frame_level_hit_rate"]["mean"] or 0
    jh = pr["recency_plus_hard_recent"]["mean"] or 0
    print(
        f"  * If frame hit ≫ 0 (here {fh:.0%}): temporal encoder/KV cache is motivated.\n"
        f"  * Disk tick cache consecutive-id rate→0: only helps replaying same ticks, not stream incrementality.\n"
        f"  * Prune Jaccard (hard-recency) {jh:.2f}: high⇒keep set stable⇒memory reuse / delta update; "
        f"low⇒every tick reshuffles⇒cache of pruned tokens less useful.\n"
        f"  * random Jaccard is the noise floor — scorer must beat it for 'stable memory' story."
    )
    print("====================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    ap.add_argument("--tick-cache", type=str, default=str(DEFAULT_TICK_CACHE))
    ap.add_argument("--max-videos", type=int, default=80)
    ap.add_argument("--keep", type=int, default=4096)
    ap.add_argument("--gp", type=int, default=256)
    ap.add_argument("--recent-keep-sec", type=float, default=2.0)
    ap.add_argument("--tubelet-sec", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    rep = profile_stream_csv(
        Path(args.csv),
        max_videos=args.max_videos,
        gp=args.gp,
        keep=args.keep,
        recent_keep_sec=args.recent_keep_sec,
        tubelet_sec=args.tubelet_sec,
    )
    tick = profile_tick_disk_cache(Path(args.tick_cache), Path(args.csv), max_rows=8000)
    print_report(rep, tick)

    out = {
        "stream_profile": rep,
        "tick_disk_cache": tick,
    }
    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent / "profile_stream_motivation.json"
    # if script under opencl_kernels, parent is opencl; put next to it in scripts/
    if out_path.name == "profile_stream_motivation.json" and "opencl" in str(out_path):
        out_path = Path("/home/ll5914/Jepa_yifan/JEPA_ARVR/scripts/profile_stream_motivation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
