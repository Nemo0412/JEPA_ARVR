#!/usr/bin/env python3
"""B17 frozen paired evaluation of fixed-budget token selectors.

The encoder is run once per batch.  A registered protocol varies either the
within-slot spatial selector under one temporal profile or the temporal quota
allocator under one spatial selector.  Every arm keeps the same exact final
token count before independently running the frozen predictor/classifier.  The
primary contract is K=3840; historical attention K=4096 is an external
reference, not a budget-matched arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.egtea_gaze import parse_gtea_gaze
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (
    FpsSubsampledStreamMTPDataset,
    enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.gaze_spatial_pruning import (
    QuotaSpatialTokenSelector,
    gather_tokens,
    valid_gaze_types,
)
from app.hdepic_lora_action_anticipation.temporal_budget_pruning import (
    LOSS_AWARE_PRIMARY_BUDGET,
    LossAwareTemporalQuotaProvider,
    attention_sum_temporal_quotas,
    permuted_temporal_quotas,
    recent_temporal_quotas,
    uniform_temporal_quotas,
)


PROTOCOL_ID = "egtea-stream-mtp-pruning/lossaware3840-gaze-spatial-native-v2"
RANDOM_PROTOCOL_ID = "egtea-stream-mtp-pruning/lossaware3840-random-spatial-native-v1"
TEMPORAL_PROTOCOL_ID = "egtea-stream-mtp-pruning/temporal-allocator-uniform-spatial-k3840-v1"
PROTOCOL_VARIANTS = ("uniform", "calib", "attention", "gaze", "gaze_shift")
RANDOM_PROTOCOL_VARIANTS = ("random",)
TEMPORAL_PROTOCOL_VARIANTS = (
    "lossaware",
    "temporal_uniform",
    "temporal_random_s17",
    "temporal_random_s29",
    "temporal_random_s43",
    "temporal_attention_sum",
    "temporal_recent",
)
PROTOCOL_HORIZONS = (2.0, 4.0, 6.0)
EXPECTED_FULL_VAL_ROWS = 22_741
EXPECTED_INPUT_SHA256 = {
    "train_csv": "6d604dded2f8ba875caace4d9266b87616bcfddb40c522f443a316786cdd2844",
    "val_csv": "d02af8297b1f845a644fd617642ca612fb553f8fa3300f9ae3903227a9dd6393",
    "parent_checkpoint": "7b12fdd545c4330198a3a149e02c40566735d86c427e7631acdaa1ea57c02949",
    "base_checkpoint": "5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012",
    "encoder_lora": "dd60d54737db584bd0897922784c0e2909a7c98e90ed05bbaa55b3ccb4c3a4eb",
    "predictor_lora": "1e70d73a311dd290df4e6137e4b0adfcc6dc8fbdd63b0d905463e59a1705895c",
    "calibrated_spatial_mask": "224c0a1cd1fea45389459064b84a3483dd3fd892b852703d1a65d370fea79563",
    "calibration_scores": "33b9479797b7a7e13078529126644d870b307fc75ad4c08dbf11a230e61ffb8d",
}


def validate_protocol_args(args, variants: list[str], horizons: list[float]) -> None:
    """Fail closed when a run no longer means its registered B17 protocol."""
    exact = {
        "budget": LOSS_AWARE_PRIMARY_BUDGET,
        "position_mode": "true_full",
        "max_frames": 80,
        "fps": 8,
        "src_fps": 8,
        "img_size": 256,
        "anticipation_sec": 2.0,
        "primary_horizon_sec": 2.0,
        "max_val_batches": 0,
        "only_context_sec": 0.0,
    }
    mismatches = [f"{key}={getattr(args, key)!r} (required {want!r})" for key, want in exact.items()
                  if getattr(args, key) != want]
    if tuple(horizons) != PROTOCOL_HORIZONS:
        mismatches.append(f"horizons={horizons!r} (required {list(PROTOCOL_HORIZONS)!r})")
    protocol_id = getattr(args, "protocol_id", PROTOCOL_ID)
    expected_variants = {
        PROTOCOL_ID: PROTOCOL_VARIANTS,
        RANDOM_PROTOCOL_ID: RANDOM_PROTOCOL_VARIANTS,
        TEMPORAL_PROTOCOL_ID: TEMPORAL_PROTOCOL_VARIANTS,
    }.get(protocol_id)
    if expected_variants is None:
        mismatches.append(f"unsupported protocol_id={protocol_id!r}")
    elif tuple(variants) != expected_variants:
        mismatches.append(f"variants={variants!r} (required {list(expected_variants)!r})")
    if protocol_id == RANDOM_PROTOCOL_ID and int(getattr(args, "random_seed", -1)) != 17:
        mismatches.append(f"random_seed={getattr(args, 'random_seed', None)!r} (required 17)")
    if protocol_id == TEMPORAL_PROTOCOL_ID:
        population = getattr(args, "evaluation_population", "unspecified")
        expected_subset = {"smoke": 2048, "full": 0}.get(population)
        if expected_subset is None:
            mismatches.append(
                f"evaluation_population={population!r} (required 'smoke' or 'full')"
            )
        elif int(getattr(args, "val_subset_n", -1)) != expected_subset:
            mismatches.append(
                f"val_subset_n={getattr(args, 'val_subset_n', None)!r} "
                f"(required {expected_subset} for {population})"
            )
    if mismatches:
        raise SystemExit("registered B17 protocol mismatch: " + "; ".join(mismatches))


def _validate_stream_timeline(frame_idx: np.ndarray, *, tick_frame: int, vfps: float) -> None:
    if not math.isclose(float(vfps), 24.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"B17 gaze alignment requires native EGTEA vfps=24, got {vfps}")
    if frame_idx.ndim != 1 or frame_idx.size == 0:
        raise ValueError("frame_indices must be a non-empty vector")
    if bool((frame_idx < 0).any()) or bool((frame_idx[1:] <= frame_idx[:-1]).any()):
        raise ValueError("frame_indices must be non-negative and strictly increasing")
    if int(frame_idx[-1]) >= int(tick_frame):
        raise ValueError(
            f"causality violation: last observed frame {int(frame_idx[-1])} >= tick {int(tick_frame)}"
        )


class IndexedStreamMTPDataset(FpsSubsampledStreamMTPDataset):
    """Streaming-MTP rows plus stable causal row/frame identities."""

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        row = self.rows[idx]
        frame_idx = np.asarray(T._parse_int_list(row["frame_indices"]), dtype=np.int64)
        if self.stride != 1:
            frame_idx = frame_idx[::-1][:: self.stride][::-1].copy()
        tick_frame = int(row["tick_frame"])
        _validate_stream_timeline(frame_idx, tick_frame=tick_frame, vfps=float(row["vfps"]))
        item.update(
            {
                "frame_indices": torch.from_numpy(frame_idx.copy()),
                "video_id": str(row["video_id"]),
                "tick_frame": tick_frame,
            }
        )
        return item


class GazeStreamMTPDataset(IndexedStreamMTPDataset):
    """Indexed stream rows plus gaze aligned to exact decoded frame indices."""

    def __init__(self, *args, gaze_root: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaze_root = Path(gaze_root)
        self._gaze_cache: dict[str, np.ndarray] = {}

    def _gaze(self, video_id: str) -> np.ndarray:
        if video_id not in self._gaze_cache:
            path = self.gaze_root / f"{video_id}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)
            self._gaze_cache[video_id] = parse_gtea_gaze(path)
        return self._gaze_cache[video_id]

    def preload_gaze(self) -> None:
        """Parse each selected session once before forked loader workers start."""
        video_ids = sorted({str(row["video_id"]) for row in self.rows})
        for video_id in video_ids:
            self._gaze(video_id)
        print(
            f"[gaze] preloaded {len(video_ids)} selected sessions for worker-shared read-only access",
            flush=True,
        )

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        video_id = item["video_id"]
        frame_idx = item["frame_indices"].numpy()
        gaze = self._gaze(video_id)
        in_range = (frame_idx >= 0) & (frame_idx < len(gaze))
        safe = np.clip(frame_idx, 0, max(0, len(gaze) - 1))
        types = gaze[safe, 2] if len(gaze) else np.zeros_like(frame_idx)
        xy = gaze[safe, :2].astype(np.float32) if len(gaze) else np.zeros((len(frame_idx), 2), np.float32)
        valid = in_range & valid_gaze_types(torch.from_numpy(np.asarray(types))).numpy()
        valid &= np.isfinite(xy).all(axis=1)
        xy[~np.isfinite(xy)] = 0.0
        item.update(
            {
                "gaze_xy": torch.from_numpy(np.clip(xy, 0.0, 1.0)),
                "gaze_valid": torch.from_numpy(valid.astype(np.bool_)),
            }
        )
        return item


def collate_indexed_stream(batch):
    out = T.collate_stream(batch)
    out.update(
        {
            "frame_indices": torch.stack([x["frame_indices"] for x in batch]),
            "video_id": [x["video_id"] for x in batch],
            "tick_frame": torch.tensor([x["tick_frame"] for x in batch], dtype=torch.long),
        }
    )
    return out


def collate_gaze_stream(batch):
    out = collate_indexed_stream(batch)
    out.update(
        {
            "gaze_xy": torch.stack([x["gaze_xy"] for x in batch]),
            "gaze_valid": torch.stack([x["gaze_valid"] for x in batch]),
        }
    )
    return out


def predict_from_selected_tokens(
    core,
    selected_full: torch.Tensor,
    true_idx: torch.Tensor,
    anticipation_times: torch.Tensor,
    *,
    num_full_tokens: int,
    position_mode: str,
) -> torch.Tensor:
    """Run the native predictor while preserving selected context positions."""
    batch, n_tokens, full_dim = selected_full.shape
    gp = int(core.grid_size**2)
    embed_dim = int(core.encoder.embed_dim)
    selected = selected_full[:, :, -embed_dim:] if full_dim > embed_dim else selected_full
    accumulated = selected.clone()
    anticipation_steps = (
        anticipation_times * core.frames_per_second / core.tubelet_size
    ).to(torch.int64)

    if position_mode == "true_full":
        context_positions = true_idx
        newest_slot = num_full_tokens // gp - 1
        skip_positions = (newest_slot + 1 + anticipation_steps) * gp
    elif position_mode == "rebase":
        context_positions = torch.arange(n_tokens, device=selected.device).unsqueeze(0).expand(batch, -1)
        skip_positions = n_tokens + gp * anticipation_steps
    else:
        raise ValueError(f"unknown position_mode={position_mode!r}")

    n_pred = int(gp * (core.num_output_frames // core.tubelet_size))
    target_positions = torch.arange(n_pred, device=selected.device).unsqueeze(0).expand(batch, -1)
    target_positions = target_positions + skip_positions.unsqueeze(1)
    predictor_input = selected_full
    for _ in range(core.num_steps):
        pred_out = core.predictor(
            predictor_input,
            masks_x=context_positions,
            masks_y=target_positions,
        )
        pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        pred = pred_full[:, :, -embed_dim:] if pred_full.size(-1) != embed_dim else pred_full
        accumulated = torch.cat([accumulated, pred], dim=1)
        pred_input = pred_full if pred_full.size(-1) == predictor_input.size(-1) else pred
        predictor_input = torch.cat([predictor_input[:, n_pred:, :], pred_input], dim=1)
    return accumulated


def _mcnemar_exact_p(a_correct_b_wrong: int, a_wrong_b_correct: int) -> float:
    """Two-sided exact binomial McNemar p-value for paired binary outcomes."""
    n = int(a_correct_b_wrong + a_wrong_b_correct)
    if n == 0:
        return 1.0
    tail = min(int(a_correct_b_wrong), int(a_wrong_b_correct))
    logs = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        for k in range(tail + 1)
    ]
    max_log = max(logs)
    log_tail = max_log + math.log(sum(math.exp(value - max_log) for value in logs)) - n * math.log(2.0)
    return float(min(1.0, 2.0 * math.exp(log_tail)))


def _paired_summary(
    a: list[int],
    b: list[int],
    clusters: list[str],
    *,
    mask: list[bool] | None = None,
    seed: int = 17,
) -> dict:
    """Paired delta with video/session-cluster bootstrap and McNemar test."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    cc = np.asarray(clusters, dtype=str)
    if aa.shape != bb.shape or aa.shape != cc.shape or aa.size == 0:
        raise ValueError("paired correctness arrays must be non-empty and shape-matched")
    if mask is not None:
        mm = np.asarray(mask, dtype=np.bool_)
        if mm.shape != aa.shape:
            raise ValueError("paired subset mask must match correctness arrays")
        aa, bb, cc = aa[mm], bb[mm], cc[mm]
    if aa.size == 0:
        return {"n": 0, "clusters": 0, "status": "no eligible rows"}
    diff = aa - bb
    unique_clusters = np.unique(cc)
    cluster_rows = [np.flatnonzero(cc == cluster) for cluster in unique_clusters]
    rng = np.random.default_rng(seed)
    boot = np.empty(4000, dtype=np.float64)
    for iteration in range(4000):
        sampled = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
        indices = np.concatenate([cluster_rows[index] for index in sampled])
        boot[iteration] = diff[indices].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5]) * 100.0
    a_only = int(((aa == 1) & (bb == 0)).sum())
    b_only = int(((aa == 0) & (bb == 1)).sum())
    return {
        "n": int(diff.size),
        "clusters": int(len(unique_clusters)),
        "delta_pp": float(diff.mean() * 100.0),
        "cluster_bootstrap_ci95_pp": [float(lo), float(hi)],
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "row_level_mcnemar_exact_two_sided_p": _mcnemar_exact_p(a_only, b_only),
    }


def _row_key(row: dict, seed: int) -> str:
    payload = f"{seed}|{row['video_id']}|{row['tick_frame']}"
    return hashlib.md5(payload.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialized_action_map(action_map: dict[tuple[int, int], int]) -> dict[str, int]:
    return {f"{verb},{noun}": int(index) for (verb, noun), index in action_map.items()}


def _validate_checkpoint_contract(
    checkpoint: dict,
    verb_map: dict,
    noun_map: dict,
    action_map: dict,
    horizons: list[float],
) -> None:
    expected = {
        "verb_map": verb_map,
        "noun_map": noun_map,
        "action_map": _serialized_action_map(action_map),
        "horizons": horizons,
    }
    mismatches = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatches:
        raise RuntimeError(f"parent checkpoint protocol metadata mismatch: {mismatches}")


def _build_calibration_ranking(
    calibration_mask: np.ndarray,
    raw_scores: torch.Tensor,
    expected_quotas: torch.Tensor,
    *,
    gp: int,
) -> torch.Tensor:
    """Build selected-first continuous ranking for the fixed Q-calib arm."""
    full_capacity = int(expected_quotas.numel()) * int(gp)
    if calibration_mask.shape != (full_capacity,):
        raise RuntimeError(
            f"calibrated spatial mask must have shape ({full_capacity},), got {calibration_mask.shape}"
        )
    if not bool(np.isin(calibration_mask, (0.0, 1.0)).all()):
        raise RuntimeError("calibrated spatial mask must be exactly binary")
    mask_per_slot = calibration_mask.reshape(-1, gp).sum(axis=1).astype(np.int64)
    if not np.array_equal(mask_per_slot, expected_quotas.cpu().numpy()):
        raise RuntimeError(
            "calibrated 10s mask per-slot counts do not match the declared loss-aware quotas"
        )
    if raw_scores.ndim != 1 or raw_scores.numel() != full_capacity:
        raise RuntimeError(
            f"continuous calibration score shape must be ({full_capacity},), "
            f"got {tuple(raw_scores.shape)}"
        )
    raw_scores = raw_scores.float()
    if not bool(torch.isfinite(raw_scores).all()):
        raise RuntimeError("continuous calibration tie-break scores contain non-finite values")
    score_min, score_max = raw_scores.min(), raw_scores.max()
    normalized_scores = (raw_scores - score_min) / (score_max - score_min).clamp_min(1e-12)
    # Every exact 10s cascade survivor outranks every non-survivor. Continuous
    # final-layer scores break ties in both groups, including short-context
    # quota expansion beyond the original survivor count.
    return normalized_scores + 2.0 * torch.from_numpy(calibration_mask.astype(np.float32))


def main(
    *,
    allowed_protocol_ids: tuple[str, ...] = (
        PROTOCOL_ID,
        RANDOM_PROTOCOL_ID,
        TEMPORAL_PROTOCOL_ID,
    ),
    default_protocol_id: str = PROTOCOL_ID,
    default_variants: tuple[str, ...] = PROTOCOL_VARIANTS,
) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", type=Path, required=True, help="train vocabulary only")
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--gaze-root", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--init-from-ckpt", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, default=None)
    ap.add_argument("--predictor-lora", type=Path, default=None)
    if default_protocol_id not in allowed_protocol_ids:
        raise ValueError("default protocol must be present in allowed protocol IDs")
    ap.add_argument("--variants", default=",".join(default_variants))
    ap.add_argument(
        "--protocol-id",
        choices=list(allowed_protocol_ids),
        default=default_protocol_id,
    )
    ap.add_argument("--random-seed", type=int, default=17)
    ap.add_argument("--calibrated-spatial-mask", type=Path, default=None)
    ap.add_argument("--calibration-scores", type=Path, default=None)
    ap.add_argument("--position-mode", choices=["true_full"], default="true_full")
    ap.add_argument("--budget", type=int, default=LOSS_AWARE_PRIMARY_BUDGET)
    ap.add_argument("--horizons-sec", default="2,4,6")
    ap.add_argument("--primary-horizon-sec", type=float, default=2.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--src-fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-val-batches", type=int, default=0)
    ap.add_argument("--val-subset-n", type=int, default=0)
    ap.add_argument("--val-subset-seed", type=int, default=0)
    ap.add_argument(
        "--evaluation-population",
        choices=["unspecified", "smoke", "full"],
        default="unspecified",
    )
    ap.add_argument("--only-context-sec", type=float, default=0.0)
    ap.add_argument("--audit-samples", type=int, default=8)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--code-root", type=Path, required=True)
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--code-snapshot-sha256", required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    validate_protocol_args(args, variants, horizons)
    paired_path = args.out_json.with_suffix(".paired.npz")
    existing_outputs = [path for path in (args.out_json, paired_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"refusing to overwrite existing results: {existing_outputs}")
    input_paths = {
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "parent_checkpoint": args.init_from_ckpt,
        "base_checkpoint": args.checkpoint,
        "encoder_lora": args.encoder_lora,
        "predictor_lora": args.predictor_lora,
    }
    spatial_protocol = args.protocol_id != TEMPORAL_PROTOCOL_ID
    if spatial_protocol:
        input_paths.update(
            {
                "calibrated_spatial_mask": args.calibrated_spatial_mask,
                "calibration_scores": args.calibration_scores,
            }
        )
    for name, path in input_paths.items():
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"missing protocol input {name}: {path}")
    input_hashes = {name: _sha256_file(Path(path)) for name, path in input_paths.items()}
    hash_mismatches = {
        name: {"actual": digest, "required": EXPECTED_INPUT_SHA256[name]}
        for name, digest in input_hashes.items()
        if digest != EXPECTED_INPUT_SHA256[name]
    }
    if hash_mismatches:
        raise RuntimeError(f"registered v2 input hash mismatch: {hash_mismatches}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("B17 frozen evaluation requires CUDA")
    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    if spatial_protocol:
        if args.gaze_root is None or not args.gaze_root.is_dir():
            raise FileNotFoundError(f"missing gaze root for spatial protocol: {args.gaze_root}")
        dataset = GazeStreamMTPDataset(
            args.val_csv,
            args.video_root,
            args.img_size,
            src_fps=args.src_fps,
            fps=args.fps,
            gaze_root=args.gaze_root,
        )
    else:
        dataset = IndexedStreamMTPDataset(
            args.val_csv,
            args.video_root,
            args.img_size,
            src_fps=args.src_fps,
            fps=args.fps,
        )
    if args.only_context_sec > 0:
        want = float(args.only_context_sec)
        dataset.rows = [r for r in dataset.rows if abs(float(r["context_sec"]) - want) < 1e-6]
    if args.val_subset_n > 0 and args.val_subset_n < len(dataset.rows):
        order = sorted(range(len(dataset.rows)), key=lambda i: _row_key(dataset.rows[i], args.val_subset_seed))
        keep = set(order[: args.val_subset_n])
        dataset.rows = [r for i, r in enumerate(dataset.rows) if i in keep]
    if not dataset.rows:
        raise SystemExit("no validation rows remain after filtering")
    if args.protocol_id == TEMPORAL_PROTOCOL_ID:
        expected_rows = 2048 if args.evaluation_population == "smoke" else EXPECTED_FULL_VAL_ROWS
        if len(dataset.rows) != expected_rows:
            raise RuntimeError(
                f"{args.evaluation_population} population has {len(dataset.rows)} rows, "
                f"required {expected_rows}"
            )
    if spatial_protocol:
        dataset.preload_gaze()

    sampler = T.ContextBucketBatchSampler(dataset, args.batch_size, shuffle=False, seed=0)
    loader_kwargs = dict(
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_gaze_stream if spatial_protocol else collate_indexed_stream,
        pin_memory=False,
        persistent_workers=False,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    # Start forked decode workers before any CUDA model allocation.  Besides
    # avoiding a costly post-CUDA fork, this overlaps the first prefetch with
    # checkpoint/model initialization.
    loader_iter = iter(loader)

    base = T.build_model(device, args.max_frames, args.fps, args.img_size, str(args.checkpoint))
    T.load_lora_sidecars(
        base,
        str(args.encoder_lora) if args.encoder_lora else None,
        str(args.predictor_lora) if args.predictor_lora else None,
    )
    gp = int(base.grid_size**2)
    if gp != 256:
        raise RuntimeError(f"B17 v1 quota contract requires gp=256, got {gp}")
    full_token_capacity = (args.max_frames // int(base.tubelet_size)) * gp
    enlarge_predictor_budget(base, full_token_capacity, gp)
    load_wrapper = T.PrunedAnticipativeModel(base, None, prune_threshold=10**9).to(device)

    classifier = T.AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=int(base.encoder.embed_dim),
        num_heads=16,
        depth=4,
        use_activation_checkpointing=True,
    ).to(device)
    mtp_clf = T.CommunicatingMLPMTPClassifier(
        classifier, horizons_sec=horizons, comm_layers=2, comm_heads=4
    ).to(device)
    ckpt = torch.load(args.init_from_ckpt, map_location="cpu", weights_only=False)
    _validate_checkpoint_contract(ckpt, verb_map, noun_map, action_map, horizons)
    load_wrapper.load_state_dict(ckpt["model"], strict=True)
    mtp_clf.load_state_dict(ckpt["mtp_classifier"], strict=True)
    print(f"[load] strict parent best={ckpt.get('best')} metadata=matched", flush=True)
    del ckpt
    base = load_wrapper.base
    base.eval()
    mtp_clf.eval()
    for module in (base, mtp_clf):
        for parameter in module.parameters():
            parameter.requires_grad = False

    # Patches the final attention forward only to expose received-attention
    # scores; encoder token outputs remain the normal full-context outputs.
    attention_observer = T.TokenPruner(base.encoder, keep_count=args.budget, gp=gp)
    quota_provider = LossAwareTemporalQuotaProvider(total_budget=args.budget, slot_capacity=gp)
    spatial_selector = QuotaSpatialTokenSelector(
        grid_size=int(base.grid_size), tubelet_size=int(base.tubelet_size)
    )
    calibration_scores_full = None
    tie_break_layer = None
    if spatial_protocol:
        calibration_mask_np = np.load(args.calibrated_spatial_mask)
        expected_quotas = quota_provider.quotas(full_token_capacity // gp).numpy()
        calibration_obj = torch.load(
            args.calibration_scores, map_location="cpu", weights_only=False
        )
        raw_tables = calibration_obj.get("per_layer_mean_token_scores", {})
        if not raw_tables:
            raise RuntimeError("calibration score artifact has no per_layer_mean_token_scores")
        tie_break_layer = max(int(layer) for layer in raw_tables)
        raw_scores = raw_tables[tie_break_layer]
        calibration_ranking = _build_calibration_ranking(
            calibration_mask_np,
            raw_scores,
            torch.from_numpy(expected_quotas),
            gp=gp,
        )
        calibration_scores_full = calibration_ranking.to(
            device=device, dtype=torch.float32
        )

    correct: dict[str, dict[str, list[int]]] = {
        variant: {f"{h:g}s": [] for h in horizons} for variant in variants
    }
    selector_stats = {variant: Counter() for variant in variants}
    row_clusters: dict[str, list[str]] = {f"{h:g}s": [] for h in horizons}
    row_keys: dict[str, list[str]] = {f"{h:g}s": [] for h in horizons}
    effective_gaze_rows: dict[str, list[bool]] = {f"{h:g}s": [] for h in horizons}
    context_quota_vectors: dict[str, list[int]] = {}
    temporal_quota_accumulators: dict[str, dict[str, dict]] = (
        {variant: {} for variant in variants}
        if args.protocol_id == TEMPORAL_PROTOCOL_ID
        else {}
    )
    audits: list[dict] = []
    n_batches = 0
    start_time = time.time()
    try:
        for batch_idx, batch in enumerate(loader_iter):
            if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                break
            clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
            clips = clips.sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
            gaze_xy = batch["gaze_xy"].to(device) if spatial_protocol else None
            gaze_valid = batch["gaze_valid"].to(device) if spatial_protocol else None
            mtp_verbs = batch["mtp_verbs"].to(device)
            mtp_nouns = batch["mtp_nouns"].to(device)
            mtp_mask = batch["mtp_mask"].to(device)
            ant = torch.full((clips.size(0),), float(args.anticipation_sec), device=device)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                encoded_full = base.encoder(clips)
                attention_scores = attention_observer._importance
                if attention_scores is None or attention_scores.shape[:2] != encoded_full.shape[:2]:
                    raise RuntimeError("attention observer did not emit scores aligned to encoder tokens")
                n_slots = encoded_full.shape[1] // gp
                quotas = quota_provider.quotas(n_slots)
                stable_row_keys = [
                    f"{batch['video_id'][position]}|{int(batch['tick_frame'][position])}"
                    for position in range(clips.size(0))
                ]
                calibrated_scores = (
                    calibration_scores_full[-encoded_full.shape[1] :]
                    if spatial_protocol else None
                )
                if spatial_protocol:
                    slot_gaze_valid = gaze_valid.view(
                        clips.size(0), n_slots, int(base.tubelet_size)
                    ).any(dim=2)
                    partial_quota = ((quotas > 0) & (quotas < gp)).to(device)
                    row_has_effective_gaze = (
                        slot_gaze_valid & partial_quota.unsqueeze(0)
                    ).any(dim=1)
                context_key = f"{n_slots}_slots"
                context_quota_vectors[context_key] = quotas.tolist()
                variant_outputs = {}
                variant_indices = {}
                variant_quotas = {}
                if args.protocol_id == TEMPORAL_PROTOCOL_ID:
                    variant_quotas = {
                        "lossaware": quotas,
                        "temporal_uniform": uniform_temporal_quotas(
                            n_slots, total=args.budget, capacity=gp
                        ),
                        "temporal_random_s17": permuted_temporal_quotas(
                            quotas, stable_row_keys, seed=17
                        ),
                        "temporal_random_s29": permuted_temporal_quotas(
                            quotas, stable_row_keys, seed=29
                        ),
                        "temporal_random_s43": permuted_temporal_quotas(
                            quotas, stable_row_keys, seed=43
                        ),
                        "temporal_attention_sum": attention_sum_temporal_quotas(
                            attention_scores, spatial_tokens=gp,
                            total=args.budget, capacity=gp,
                        ),
                        "temporal_recent": recent_temporal_quotas(
                            n_slots, total=args.budget, capacity=gp
                        ),
                    }
                else:
                    variant_quotas = {variant: quotas for variant in variants}
                for variant in variants:
                    current_quotas = variant_quotas[variant]
                    quota_rows = (
                        current_quotas
                        if current_quotas.ndim == 2
                        else current_quotas.unsqueeze(0).expand(clips.size(0), -1)
                    )
                    if not bool((quota_rows.sum(dim=1) == args.budget).all()):
                        raise RuntimeError(f"{variant} failed exact K={args.budget} quota contract")
                    if args.protocol_id == TEMPORAL_PROTOCOL_ID:
                        accumulator = temporal_quota_accumulators[variant].setdefault(
                            context_key,
                            {
                                "rows": 0,
                                "sum_per_slot": torch.zeros(n_slots, dtype=torch.float64),
                                "zero_slots": 0,
                                "full_slots": 0,
                                "newest_15_tokens": 0,
                            },
                        )
                        quota_rows_cpu = quota_rows.to(dtype=torch.long, device="cpu")
                        accumulator["rows"] += int(quota_rows_cpu.shape[0])
                        accumulator["sum_per_slot"] += quota_rows_cpu.sum(dim=0)
                        accumulator["zero_slots"] += int((quota_rows_cpu == 0).sum())
                        accumulator["full_slots"] += int((quota_rows_cpu == gp).sum())
                        accumulator["newest_15_tokens"] += int(
                            quota_rows_cpu[:, -15:].sum()
                        )
                    keep_idx, stats = spatial_selector.select(
                        mode="uniform" if args.protocol_id == TEMPORAL_PROTOCOL_ID else variant,
                        quotas=current_quotas,
                        attention_scores=attention_scores,
                        calibrated_scores=calibrated_scores if variant == "calib" else None,
                        gaze_xy=gaze_xy if variant.startswith("gaze") else None,
                        gaze_valid=gaze_valid if variant.startswith("gaze") else None,
                        random_keys=stable_row_keys if variant == "random" else None,
                        random_seed=args.random_seed,
                    )
                    for key, value in stats.items():
                        selector_stats[variant][key] += int(value)
                    selected = gather_tokens(encoded_full, keep_idx)
                    tokens = predict_from_selected_tokens(
                        base,
                        selected,
                        keep_idx,
                        ant,
                        num_full_tokens=encoded_full.shape[1],
                        position_mode=args.position_mode,
                    )
                    variant_outputs[variant] = mtp_clf(tokens)
                    variant_indices[variant] = keep_idx

            for horizon_idx, horizon in enumerate(horizons):
                valid_rows = mtp_mask[:, horizon_idx] > 0.5
                if not bool(valid_rows.any()):
                    continue
                _, _, action_labels, keep = T.map_labels(
                    mtp_verbs[valid_rows, horizon_idx],
                    mtp_nouns[valid_rows, horizon_idx],
                    verb_map,
                    noun_map,
                    action_map,
                    device,
                )
                if not keep:
                    continue
                valid_positions = valid_rows.nonzero(as_tuple=False).flatten()[keep]
                key = f"{horizon:g}s"
                selected_positions = valid_positions.cpu().tolist()
                row_clusters[key].extend(batch["video_id"][position] for position in selected_positions)
                row_keys[key].extend(
                    f"{batch['video_id'][position]}|{int(batch['tick_frame'][position])}"
                    for position in selected_positions
                )
                if spatial_protocol:
                    effective_gaze_rows[key].extend(
                        bool(row_has_effective_gaze[position])
                        for position in selected_positions
                    )
                for variant in variants:
                    logits = variant_outputs[variant][float(horizon)]["action"][valid_positions].float()
                    hit = logits.topk(min(5, logits.shape[1]), dim=1).indices.eq(
                        action_labels.unsqueeze(1)
                    ).any(dim=1)
                    correct[variant][f"{horizon:g}s"].extend(hit.to(torch.int8).cpu().tolist())

            if len(audits) < args.audit_samples:
                for row_idx in range(min(clips.size(0), args.audit_samples - len(audits))):
                    audit = {
                        "video_id": batch["video_id"][row_idx],
                        "tick_frame": int(batch["tick_frame"][row_idx]),
                        "n_slots": int(n_slots),
                        "budget": int(quotas.sum()),
                        "first_observed_frame": int(batch["frame_indices"][row_idx, 0]),
                        "last_observed_frame": int(batch["frame_indices"][row_idx, -1]),
                        "variants": {},
                    }
                    if spatial_protocol:
                        audit["quota_vector"] = quotas.tolist()
                        audit["valid_gaze_frames"] = int(gaze_valid[row_idx].sum())
                        audit["effective_gaze_row"] = bool(row_has_effective_gaze[row_idx])
                    else:
                        audit["lossaware_reference_quota_vector"] = quotas.tolist()
                    for variant in variants:
                        idx = variant_indices[variant][row_idx]
                        per_slot = torch.bincount(idx // gp, minlength=n_slots).cpu().tolist()
                        current_quotas = variant_quotas[variant]
                        quota_row = current_quotas[row_idx] if current_quotas.ndim == 2 else current_quotas
                        audit["variants"][variant] = {
                            "keep_count": int(idx.numel()),
                            "quota_vector": quota_row.tolist(),
                            "per_slot_counts": per_slot,
                            "first_index": int(idx[0]),
                            "last_index": int(idx[-1]),
                        }
                    audits.append(audit)
            n_batches += 1
            if n_batches % 100 == 0:
                primary = f"{args.primary_horizon_sec:g}s"
                status = " ".join(
                    f"{v}={100*np.mean(correct[v][primary]):.2f}" for v in variants if correct[v][primary]
                )
                print(f"[eval] batches={n_batches} rows={len(correct[variants[0]][primary])} {status}", flush=True)
    finally:
        attention_observer.remove()

    results = {}
    for variant in variants:
        metrics = {key: float(np.mean(values)) for key, values in correct[variant].items()}
        counts = {key: len(values) for key, values in correct[variant].items()}
        weighted = sum(weight * metrics[f"{h:g}s"] for weight, h in zip((1.0, 0.7, 0.5), horizons))
        weighted /= sum((1.0, 0.7, 0.5)[: len(horizons)])
        results[variant] = {
            "action_top5": metrics,
            "n": counts,
            "weighted_mtp": float(weighted),
            "selector_stats": dict(selector_stats[variant]),
        }

    paired = {}
    if "gaze" in variants:
        for control in ("gaze_shift", "attention", "calib", "uniform"):
            if control not in variants:
                continue
            paired[f"gaze_minus_{control}"] = {
                f"{h:g}s": {
                    "all_rows": _paired_summary(
                        correct["gaze"][f"{h:g}s"],
                        correct[control][f"{h:g}s"],
                        row_clusters[f"{h:g}s"],
                    ),
                    "effective_gaze_rows": _paired_summary(
                        correct["gaze"][f"{h:g}s"],
                        correct[control][f"{h:g}s"],
                        row_clusters[f"{h:g}s"],
                        mask=effective_gaze_rows[f"{h:g}s"],
                    ),
                }
                for h in horizons
            }
    if args.protocol_id == TEMPORAL_PROTOCOL_ID:
        for variant in variants:
            if variant == "lossaware":
                continue
            paired[f"{variant}_minus_lossaware"] = {
                f"{h:g}s": _paired_summary(
                    correct[variant][f"{h:g}s"],
                    correct["lossaware"][f"{h:g}s"],
                    row_clusters[f"{h:g}s"],
                )
                for h in horizons
            }

    temporal_quota_summaries = {}
    for variant, contexts in temporal_quota_accumulators.items():
        temporal_quota_summaries[variant] = {}
        for context_key, values in contexts.items():
            rows = int(values["rows"])
            temporal_quota_summaries[variant][context_key] = {
                "rows": rows,
                "mean_per_slot": (values["sum_per_slot"] / rows).tolist(),
                "mean_zero_slots_per_row": float(values["zero_slots"] / rows),
                "mean_full_slots_per_row": float(values["full_slots"] / rows),
                "mean_newest_15_tokens": float(values["newest_15_tokens"] / rows),
            }

    temporal_random_summary = {}
    if args.protocol_id == TEMPORAL_PROTOCOL_ID:
        random_variants = [
            "temporal_random_s17",
            "temporal_random_s29",
            "temporal_random_s43",
        ]
        temporal_random_summary = {
            "seeds": [17, 29, 43],
            "action_top5": {
                f"{h:g}s": {
                    "mean_fraction": float(np.mean([
                        results[variant]["action_top5"][f"{h:g}s"]
                        for variant in random_variants
                    ])),
                    "min_fraction": float(np.min([
                        results[variant]["action_top5"][f"{h:g}s"]
                        for variant in random_variants
                    ])),
                    "max_fraction": float(np.max([
                        results[variant]["action_top5"][f"{h:g}s"]
                        for variant in random_variants
                    ])),
                    "range_pp": float(100.0 * (
                        np.max([
                            results[variant]["action_top5"][f"{h:g}s"]
                            for variant in random_variants
                        ])
                        - np.min([
                            results[variant]["action_top5"][f"{h:g}s"]
                            for variant in random_variants
                        ])
                    )),
                }
                for h in horizons
            },
            "weighted_mtp": {
                "mean_fraction": float(np.mean([
                    results[variant]["weighted_mtp"] for variant in random_variants
                ])),
                "min_fraction": float(np.min([
                    results[variant]["weighted_mtp"] for variant in random_variants
                ])),
                "max_fraction": float(np.max([
                    results[variant]["weighted_mtp"] for variant in random_variants
                ])),
                "range_pp": float(100.0 * (
                    np.max([
                        results[variant]["weighted_mtp"] for variant in random_variants
                    ])
                    - np.min([
                        results[variant]["weighted_mtp"] for variant in random_variants
                    ])
                )),
            },
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paired_path,
        **{
            f"{variant}__{h:g}s": np.asarray(correct[variant][f"{h:g}s"], dtype=np.int8)
            for variant in variants
            for h in horizons
        },
        **{
            f"cluster__{h:g}s": np.asarray(row_clusters[f"{h:g}s"], dtype=str)
            for h in horizons
        },
        **{
            f"row_key__{h:g}s": np.asarray(row_keys[f"{h:g}s"], dtype=str)
            for h in horizons
        },
        **(
            {
                f"effective_gaze__{h:g}s": np.asarray(
                    effective_gaze_rows[f"{h:g}s"], dtype=np.bool_
                )
                for h in horizons
            }
            if spatial_protocol
            else {}
        ),
    )
    provenance = {
        "slurm_job_id": args.job_id,
        "code_root": str(args.code_root.resolve()),
        "git_commit": args.code_commit,
        "code_snapshot_sha256": args.code_snapshot_sha256,
        "inputs_sha256": input_hashes,
    }
    if spatial_protocol:
        provenance["calibration_tie_break_layer"] = tie_break_layer
    if "random" in variants:
        provenance["random_seed"] = int(args.random_seed)
    if args.protocol_id == TEMPORAL_PROTOCOL_ID:
        provenance["temporal_random_permutation_seeds"] = [17, 29, 43]

    report = {
        "branch": "B17",
        "group": (
            "B17-temporal-budget-allocation-uniform-spatial"
            if args.protocol_id == TEMPORAL_PROTOCOL_ID
            else "B17-lossaware-temporal-quota-gaze-spatial-pruning"
        ),
        "evaluation_protocol": args.protocol_id,
        "run_kind": "val",
        "metric_scope": "native",
        "eval_path": (
            "frozen EGTEA split1 temporal-half stream-MTP paired temporal-allocator eval"
            if args.protocol_id == TEMPORAL_PROTOCOL_ID
            else "frozen EGTEA split1 temporal-half stream-MTP paired spatial-selector eval"
        ),
        "budget_contract": {
            "primary_actual_final_tokens": int(args.budget),
            "source": (
                "fixed temporal-allocation comparison; lossaware arm uses the B13 final survival histogram"
                if args.protocol_id == TEMPORAL_PROTOCOL_ID
                else "B13 loss-aware final survival histogram"
            ),
            "historical_attention_reference_tokens": 4096,
            "warning": "K=3840 primary arms are not budget-identical to historical attention K=4096",
        },
        "position_mode": args.position_mode,
        "evaluation_population": args.evaluation_population,
        "val_subset_n": int(args.val_subset_n),
        "val_subset_seed": int(args.val_subset_seed),
        "variants": variants,
        "varied_axis": (
            "temporal_quota_only"
            if args.protocol_id == TEMPORAL_PROTOCOL_ID
            else "within_slot_spatial_selector_only"
        ),
        "fixed_spatial_selector": (
            "deterministic_uniform_farthest_point"
            if args.protocol_id == TEMPORAL_PROTOCOL_ID else None
        ),
        "dataset_rows_evaluated": int(len(dataset.rows)),
        "primary_metric_denominator": len(
            correct[variants[0]][f"{args.primary_horizon_sec:g}s"]
        ),
        "batches": n_batches,
        "seconds": time.time() - start_time,
        "results": results,
        "paired": paired,
        "audit": audits,
        "paired_correctness_path": str(paired_path),
        "provenance": provenance,
    }
    if args.protocol_id == TEMPORAL_PROTOCOL_ID:
        report["lossaware_reference_quota_vectors"] = context_quota_vectors
        report["temporal_quota_summaries"] = temporal_quota_summaries
        report["temporal_random_summary"] = temporal_random_summary
    else:
        report["context_quota_vectors"] = context_quota_vectors
        report.pop("fixed_spatial_selector")
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
