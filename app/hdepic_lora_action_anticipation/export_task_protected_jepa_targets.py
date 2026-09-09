#!/usr/bin/env python3
"""Export resumable train-only counterfactual/JEPA target shards for B17."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.hdepic_lora_action_anticipation import train_stream_mtp as T
from app.hdepic_lora_action_anticipation.attn_midlayer_pruning import _MidLayerAttnCapture
from app.hdepic_lora_action_anticipation.eval_stream_mtp_fixed_budget_prune import (
    IndexedStreamMTPDataset,
    _validate_checkpoint_contract,
    collate_indexed_stream,
)
from app.hdepic_lora_action_anticipation.eval_stream_mtp_kvcache_prune import (
    enlarge_predictor_budget,
)
from app.hdepic_lora_action_anticipation.task_protected_jepa_pruning import (
    EXPECTED_TRAIN_CSV_SHA256,
    ORACLE_PROTOCOL_ID,
    OfflineOracleTargetAdapter,
    SpatiotemporalGroupLayout,
    aggregate_token_scores_to_groups,
    sha256_file,
)


EXPECTED_CHECKPOINT_SHA256 = "5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012"
EXPECTED_PARENT_SHA256 = "7b12fdd545c4330198a3a149e02c40566735d86c427e7631acdaa1ea57c02949"
EXPECTED_ENCODER_LORA_SHA256 = "dd60d54737db584bd0897922784c0e2909a7c98e90ed05bbaa55b3ccb4c3a4eb"
EXPECTED_PREDICTOR_LORA_SHA256 = "1e70d73a311dd290df4e6137e4b0adfcc6dc8fbdd63b0d905463e59a1705895c"
HORIZONS = (2.0, 4.0, 6.0)
HORIZON_WEIGHTS = (1.0, 0.7, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--manifest-split", choices=("calibration", "heldout"), required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-lora", type=Path, required=True)
    parser.add_argument("--predictor-lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-start", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--group-side", type=int, choices=(4, 8), default=8)
    parser.add_argument("--deletion-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "unknown"))
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--code-snapshot-sha256", required=True)
    return parser.parse_args()


def _require_hash(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _stack_rows(rows: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    return torch.stack([row[key].detach().float().cpu() for row in rows])


def main() -> None:
    args = parse_args()
    if args.protocol_id != ORACLE_PROTOCOL_ID:
        raise SystemExit(f"protocol mismatch: expected {ORACLE_PROTOCOL_ID}, got {args.protocol_id}")
    if args.shard_start < 0 or args.shard_count <= 0:
        raise ValueError("shard-start must be non-negative and shard-count positive")
    if args.deletion_batch_size <= 0:
        raise ValueError("deletion-batch-size must be positive")
    sidecar_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.output} or {sidecar_path}")

    hashes = {
        "train_csv": _require_hash(args.train_csv, EXPECTED_TRAIN_CSV_SHA256, "train CSV"),
        "manifest_csv": _require_hash(
            args.manifest_csv, args.expected_manifest_sha256, "oracle manifest CSV"
        ),
        "checkpoint": _require_hash(args.checkpoint, EXPECTED_CHECKPOINT_SHA256, "V-JEPA checkpoint"),
        "parent_checkpoint": _require_hash(
            args.parent_checkpoint, EXPECTED_PARENT_SHA256, "stream-MTP parent"
        ),
        "encoder_lora": _require_hash(
            args.encoder_lora, EXPECTED_ENCODER_LORA_SHA256, "encoder LoRA"
        ),
        "predictor_lora": _require_hash(
            args.predictor_lora, EXPECTED_PREDICTOR_LORA_SHA256, "predictor LoRA"
        ),
    }

    dataset = IndexedStreamMTPDataset(
        args.manifest_csv,
        args.video_root,
        256,
        src_fps=8,
        fps=8,
    )
    manifest_rows = len(dataset.rows)
    shard_stop = args.shard_start + args.shard_count
    if shard_stop > manifest_rows:
        raise ValueError(
            f"shard [{args.shard_start},{shard_stop}) exceeds manifest rows {manifest_rows}"
        )
    dataset.rows = dataset.rows[args.shard_start:shard_stop]
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_indexed_stream,
        "pin_memory": False,
        "persistent_workers": False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    loader_iter = iter(loader)

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("target export requires CUDA")
    verb_map, noun_map, action_map = T.load_action_maps(args.train_csv)
    base = T.build_model(device, max_frames=80, fps=8, img_size=256, checkpoint=str(args.checkpoint))
    T.load_lora_sidecars(base, str(args.encoder_lora), str(args.predictor_lora))
    layout = SpatiotemporalGroupLayout(group_side=args.group_side)
    enlarge_predictor_budget(base, layout.num_tokens, layout.tokens_per_slot)
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
    mtp_classifier = T.CommunicatingMLPMTPClassifier(
        classifier,
        horizons_sec=HORIZONS,
        comm_layers=2,
        comm_heads=4,
    ).to(device)
    checkpoint = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    _validate_checkpoint_contract(checkpoint, verb_map, noun_map, action_map, list(HORIZONS))
    load_wrapper.load_state_dict(checkpoint["model"], strict=True)
    mtp_classifier.load_state_dict(checkpoint["mtp_classifier"], strict=True)
    parent_best = checkpoint.get("best")
    del checkpoint
    base = load_wrapper.base
    base.eval()
    mtp_classifier.eval()
    for module in (base, mtp_classifier):
        for parameter in module.parameters():
            parameter.requires_grad = False

    attention_capture = _MidLayerAttnCapture(base.encoder.blocks[16].attn)
    adapter = OfflineOracleTargetAdapter(
        base,
        mtp_classifier,
        verb_map,
        noun_map,
        action_map,
        horizons=HORIZONS,
        horizon_weights=HORIZON_WEIGHTS,
        anticipation_sec=2.0,
        prune_layer=16,
    )
    all_group_tokens = layout.token_indices(device)
    noncore_mask = ~layout.recent_group_mask(device)
    deletion_group_ids = torch.nonzero(noncore_mask, as_tuple=False).flatten()
    deletion_groups = all_group_tokens[deletion_group_ids]

    row_results: list[dict[str, torch.Tensor]] = []
    row_keys: list[str] = []
    video_ids: list[str] = []
    tick_frames: list[int] = []
    parity_max_abs: list[float] = []
    started = time.time()
    try:
        for local_index, batch in enumerate(loader_iter):
            clips = batch["clip"].to(device, non_blocking=False).float().div_(255.0)
            clips = clips.sub_(T.IMAGENET_MEAN.to(device)).div_(T.IMAGENET_STD.to(device))
            raw_verbs = batch["mtp_verbs"].to(device)
            raw_nouns = batch["mtp_nouns"].to(device)
            target_mask = batch["mtp_mask"].to(device)
            if clips.shape[2:] != (80, 256, 256):
                raise RuntimeError(f"oracle requires [80,256,256] video input, got {clips.shape[2:]}")

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                if local_index == 0:
                    split_final, _ = adapter.continuation.forward_full(clips)
                    native_final = base.encoder(clips)
                    parity = float((split_final.float() - native_final.float()).abs().max().item())
                    if parity > 1e-5:
                        raise RuntimeError(f"native/L16 no-delete parity max_abs={parity}")
                    parity_max_abs.append(parity)
                    del split_final, native_final
                targets = adapter.generate_row_targets(
                    clips,
                    raw_verbs,
                    raw_nouns,
                    target_mask,
                    deletion_groups,
                    deletion_batch_size=args.deletion_batch_size,
                )
            if attention_capture.importance is None:
                raise RuntimeError("L16 attention capture did not emit token importance")
            attention_group = aggregate_token_scores_to_groups(
                attention_capture.importance[0], all_group_tokens
            )[deletion_group_ids]
            targets["attention_group_score"] = attention_group
            if int(targets["predictor_lora_modules_disabled"].item()) <= 0:
                raise RuntimeError("no predictor-LoRA modules were disabled for JEPA target")
            for key, value in targets.items():
                if torch.is_floating_point(value) and not torch.isfinite(value).all():
                    raise RuntimeError(f"non-finite target {key}")
            row_results.append(targets)
            video_id = batch["video_id"][0]
            tick_frame = int(batch["tick_frame"][0].item())
            video_ids.append(video_id)
            tick_frames.append(tick_frame)
            row_keys.append(f"{video_id}\t{tick_frame}")
            torch.cuda.synchronize()
            print(
                f"[target] local={local_index + 1}/{args.shard_count} "
                f"manifest={args.shard_start + local_index} row={row_keys[-1]} "
                f"elapsed_sec={time.time() - started:.1f}",
                flush=True,
            )
    finally:
        attention_capture.remove()

    payload = {
        "target_schema_version": 2,
        "protocol_id": ORACLE_PROTOCOL_ID,
        "manifest_split": args.manifest_split,
        "manifest_rows": manifest_rows,
        "shard_start": args.shard_start,
        "shard_count": args.shard_count,
        "shard_stop": shard_stop,
        "row_keys": row_keys,
        "video_ids": video_ids,
        "tick_frames": tick_frames,
        "group_layout": {
            "num_slots": layout.num_slots,
            "grid_size": layout.grid_size,
            "group_side": layout.group_side,
            "recent_slots": layout.recent_slots,
            "num_groups": layout.num_groups,
            "num_recent_groups": layout.num_recent_groups,
            "tokens_per_group": layout.tokens_per_group,
            "deletion_group_ids": deletion_group_ids.cpu(),
        },
        "prune_layer": 16,
        "predictor_position_mode": "rebase",
        "horizons": HORIZONS,
        "horizon_weights": HORIZON_WEIGHTS,
        "anticipation_sec": 2.0,
        "parent_best": parent_best,
        "hashes": hashes,
        "code_commit": args.code_commit,
        "code_snapshot_sha256": args.code_snapshot_sha256,
        "job_id": str(args.job_id),
        "native_l16_parity_max_abs": parity_max_abs,
        "baseline_task_loss": _stack_rows(row_results, "baseline_task_loss").flatten(),
        "baseline_action_loss": _stack_rows(row_results, "baseline_action_loss").flatten(),
        "baseline_action_margin": _stack_rows(row_results, "baseline_action_margin").flatten(),
        "deletion_task_loss": _stack_rows(row_results, "deletion_task_loss"),
        "task_utility": _stack_rows(row_results, "task_utility"),
        "deletion_action_loss": _stack_rows(row_results, "deletion_action_loss"),
        "action_loss_utility": _stack_rows(row_results, "action_loss_utility"),
        "deletion_action_margin": _stack_rows(row_results, "deletion_action_margin"),
        "action_margin_drop": _stack_rows(row_results, "action_margin_drop"),
        "jepa_cosine_residual": _stack_rows(row_results, "jepa_cosine_residual"),
        "jepa_normalized_mse": _stack_rows(row_results, "jepa_normalized_mse"),
        "attention_group_score": _stack_rows(row_results, "attention_group_score"),
        "predictor_lora_modules_disabled": int(
            row_results[0]["predictor_lora_modules_disabled"].item()
        ),
        "elapsed_sec": time.time() - started,
    }
    print(
        "[summary] "
        f"rows={len(row_keys)} groups_per_row={deletion_group_ids.numel()} "
        f"native_l16_parity_max_abs={max(parity_max_abs):.8g} "
        f"predictor_lora_modules_disabled={payload['predictor_lora_modules_disabled']} "
        f"task_utility_range=[{payload['task_utility'].min().item():.6g},"
        f"{payload['task_utility'].max().item():.6g}] "
        f"action_loss_utility_range=[{payload['action_loss_utility'].min().item():.6g},"
        f"{payload['action_loss_utility'].max().item():.6g}] "
        f"jepa_cosine_range=[{payload['jepa_cosine_residual'].min().item():.6g},"
        f"{payload['jepa_cosine_residual'].max().item():.6g}]",
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{args.job_id}")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    artifact_hash = sha256_file(args.output)
    sidecar = {
        "complete": True,
        "protocol_id": ORACLE_PROTOCOL_ID,
        "target_schema_version": 2,
        "manifest_split": args.manifest_split,
        "shard_start": args.shard_start,
        "shard_count": args.shard_count,
        "row_keys_sha256": hashlib.sha256(
            "".join(f"{key}\n" for key in row_keys).encode("utf-8")
        ).hexdigest(),
        "artifact": str(args.output),
        "artifact_sha256": artifact_hash,
        "code_snapshot_sha256": args.code_snapshot_sha256,
        "job_id": str(args.job_id),
    }
    sidecar_temporary = sidecar_path.with_suffix(sidecar_path.suffix + f".tmp.{args.job_id}")
    with sidecar_temporary.open("x", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(sidecar_temporary, sidecar_path)
    print(json.dumps(sidecar, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
