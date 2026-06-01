"""Diagnostics for the HD-EPIC binary-map input adapter branch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml


def _project_root() -> Path:
    return Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()


ROOT = _project_root()
VJEPA2 = ROOT / "vjepa2"
for path in (str(ROOT), str(VJEPA2)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.hdepic_lora_action_anticipation.binary_input_adapter import BinaryMapInputAdapter  # noqa: E402
from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate, patch_metadata_dataloader  # noqa: E402
from evals.action_anticipation_frozen.dataloader import filter_annotations, init_data  # noqa: E402
from evals.action_anticipation_frozen.models import init_module  # noqa: E402


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_loader(cfg: dict, split: str, max_workers: int):
    exp = cfg["experiment"]
    data = exp["data"]
    lora = exp["lora"]
    gaze_cfg = dict(lora.get("gaze", {}))
    model_kwargs = cfg["model_kwargs"]
    gaze_cfg.setdefault("crop_size", data.get("resolution", 384))
    gaze_cfg.setdefault("frames_per_clip", data.get("frames_per_clip", 32))
    gaze_cfg.setdefault("patch_size", model_kwargs.get("pretrain_kwargs", {}).get("encoder", {}).get("patch_size", 16))
    gaze_cfg.setdefault("tubelet_size", model_kwargs.get("pretrain_kwargs", {}).get("encoder", {}).get("tubelet_size", 2))
    disable_train_aug_env = os.environ.get("BINARY_INPUT_ADAPTER_DISABLE_TRAIN_AUG")
    gaze_cfg["disable_train_aug"] = (
        disable_train_aug_env.lower() in {"1", "true", "yes", "on"}
        if disable_train_aug_env is not None
        else True
    )
    patch_metadata_dataloader(emit_binary_map=True, binary_map_cfg=gaze_cfg)

    annotations = filter_annotations(
        data["dataset"],
        data["base_path"],
        data["dataset_train"],
        data["dataset_val"],
        file_format=data.get("file_format", 1),
    )
    training = split == "train"
    _, loader, _ = init_data(
        dataset=data["dataset"],
        training=training,
        base_path=data["base_path"],
        annotations_path=annotations["train" if training else "val"],
        batch_size=exp["optimization"].get("batch_size", 2),
        frames_per_clip=data["frames_per_clip"],
        fps=data["frames_per_second"],
        anticipation_time_sec=data.get("train_anticipation_time_sec" if training else "anticipation_time_sec"),
        anticipation_point=data.get("train_anticipation_point" if training else "val_anticipation_point", [0.0, 0.0]),
        random_resize_scale=data.get("random_resize_scale", (0.08, 1.0)),
        reprob=data.get("reprob", 0.0),
        auto_augment=data.get("auto_augment", False),
        motion_shift=data.get("motion_shift", False),
        crop_size=data.get("resolution", 384),
        world_size=1,
        rank=0,
        num_workers=min(int(data.get("num_workers", 0)), int(max_workers)),
        pin_mem=False,
        persistent_workers=False,
    )
    return loader, gaze_cfg


def _load_adapter(cfg: dict, checkpoint: Path | None, device: torch.device):
    adapter_cfg = dict(cfg["experiment"]["lora"].get("gaze", {}).get("input_adapter", {}))
    adapter = BinaryMapInputAdapter(
        hidden_dim=int(adapter_cfg.get("hidden_dim", 8)),
        scale=float(adapter_cfg.get("scale", 1.0)),
        temporal_kernel=int(adapter_cfg.get("temporal_kernel", 1)),
        binary_center=float(adapter_cfg.get("binary_center", 0.0)),
        residual_clamp=float(adapter_cfg.get("residual_clamp", 1.0)),
    ).to(device)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        adapter.load_state_dict(state.get("input_adapter", state), strict=True)
    adapter.eval()
    return adapter


def _load_model(cfg: dict, device: torch.device):
    data = cfg["experiment"]["data"]
    model_cfg = cfg["model_kwargs"]
    model = init_module(
        module_name=model_cfg["module_name"],
        frames_per_clip=data["frames_per_clip"],
        frames_per_second=data["frames_per_second"],
        resolution=data.get("resolution", 384),
        checkpoint=model_cfg["checkpoint"],
        model_kwargs=model_cfg["pretrain_kwargs"],
        wrapper_kwargs=model_cfg["wrapper_kwargs"],
        device=device,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _accumulate(stats: dict, key: str, value: torch.Tensor):
    value = value.detach().float().cpu()
    stats.setdefault(key, []).append(
        {
            "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)),
            "min": float(value.min()),
            "max": float(value.max()),
        }
    )


def _summarize(items: list[dict]) -> dict:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: float(torch.tensor([item[key] for item in items]).mean()) for key in keys}


def _coord_summary(gate: GazeTokenGate, metadata: list[dict], max_examples: int):
    counts = Counter()
    videos = Counter()
    missing_videos = Counter()
    zero_query_videos = Counter()
    xy_min = []
    xy_max = []
    xy_mean = []
    examples = []
    for meta in metadata:
        video_id = str(meta.get("video_id"))
        videos[video_id] += 1
        counts["samples"] += 1
        record = gate._load_record(video_id)  # noqa: SLF001 - diagnostic reuse of loader internals
        if record is None:
            counts["record_missing"] += 1
            missing_videos[video_id] += 1
            if len(examples) < max_examples:
                examples.append({"video_id": video_id, "status": "record_missing"})
            continue
        counts["record_found"] += 1
        xy = gate._query_crop_xy(  # noqa: SLF001
            record,
            meta.get("frame_indices"),
            meta.get("vfps", 30.0),
            int(meta.get("height", gate.crop_size)),
            int(meta.get("width", gate.crop_size)),
        )
        if xy is None or len(xy) == 0:
            counts["query_empty"] += 1
            zero_query_videos[video_id] += 1
            if len(examples) < max_examples:
                examples.append({"video_id": video_id, "status": "query_empty"})
            continue
        xy = np.asarray(xy, dtype=np.float64)
        finite = np.isfinite(xy).all(axis=1)
        counts["query_ok"] += 1
        counts["finite_frames"] += int(finite.sum())
        counts["nonfinite_frames"] += int((~finite).sum())
        if finite.any():
            xy_f = xy[finite]
            xy_min.append(xy_f.min(axis=0))
            xy_max.append(xy_f.max(axis=0))
            xy_mean.append(xy_f.mean(axis=0))
            out_of_bounds = (
                (xy_f[:, 0] < 0)
                | (xy_f[:, 0] > gate.crop_size - 1)
                | (xy_f[:, 1] < 0)
                | (xy_f[:, 1] > gate.crop_size - 1)
            )
            counts["out_of_bounds_frames"] += int(out_of_bounds.sum())
            if len(examples) < max_examples:
                examples.append(
                    {
                        "video_id": video_id,
                        "status": "query_ok",
                        "frames": int(len(xy)),
                        "xy_first": [float(x) for x in xy_f[0].tolist()],
                        "xy_mean": [float(x) for x in xy_f.mean(axis=0).tolist()],
                        "xy_min": [float(x) for x in xy_f.min(axis=0).tolist()],
                        "xy_max": [float(x) for x in xy_f.max(axis=0).tolist()],
                    }
                )
    out = {
        "counts": dict(counts),
        "unique_videos": len(videos),
        "top_videos": videos.most_common(20),
        "top_missing_record_videos": missing_videos.most_common(20),
        "top_empty_query_videos": zero_query_videos.most_common(20),
        "examples": examples,
    }
    if xy_min:
        out["xy_global_min"] = [float(x) for x in np.stack(xy_min).min(axis=0).tolist()]
        out["xy_global_max"] = [float(x) for x in np.stack(xy_max).max(axis=0).tolist()]
        out["xy_mean_of_sample_means"] = [float(x) for x in np.stack(xy_mean).mean(axis=0).tolist()]
    return out


def run(args):
    cfg = _load_cfg(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader, gaze_cfg = _make_loader(cfg, args.split, args.max_workers)
    adapter = _load_adapter(cfg, args.adapter_checkpoint, device) if args.adapter_checkpoint else None
    model = _load_model(cfg, device) if args.with_model else None
    gate = GazeTokenGate({**gaze_cfg, "mode": "token_gate"})

    stats: dict[str, list] = {}
    map_nonzero = []
    all_zero_samples = 0
    samples = 0
    token_cos = []
    token_rmse = []
    coord_metadata = []
    for idx, batch in enumerate(loader):
        if args.max_batches > 0 and idx >= args.max_batches:
            break
        clips = batch[0].to(device)
        anticipation = batch[4].to(device)
        binary_map = batch[5].to(device)
        metadata = batch[3]
        coord_metadata.extend(metadata)
        samples += int(clips.shape[0])
        flat_map = binary_map.flatten(1)
        nz = flat_map.float().mean(dim=1)
        map_nonzero.extend(float(x) for x in nz.detach().cpu())
        all_zero_samples += int((flat_map.sum(dim=1) == 0).sum().item())
        _accumulate(stats, "rgb", clips)
        _accumulate(stats, "binary_map", binary_map)

        if adapter is not None:
            with torch.no_grad():
                adapted = adapter(clips, binary_map)
                residual = adapted - clips
            _accumulate(stats, "adapted_rgb", adapted)
            _accumulate(stats, "adapter_residual", residual)
            if model is not None:
                with torch.no_grad():
                    base_tokens = model(clips, anticipation)
                    adapted_tokens = model(adapted, anticipation)
                    diff = adapted_tokens.float() - base_tokens.float()
                    token_rmse.append(float(diff.pow(2).mean().sqrt().detach().cpu()))
                    cos = torch.nn.functional.cosine_similarity(
                        base_tokens.flatten(1).float(), adapted_tokens.flatten(1).float(), dim=1
                    )
                    token_cos.extend(float(x) for x in cos.detach().cpu())

    out = {
        "config": str(args.config),
        "split": args.split,
        "batches": min(args.max_batches, idx + 1 if "idx" in locals() else 0),
        "samples": samples,
        "gaze": {
            "binary_radius_px": gaze_cfg.get("binary_radius_px"),
            "binary_map_type": gaze_cfg.get("binary_map_type", gaze_cfg.get("map_type", "binary")),
            "disable_train_aug": gaze_cfg.get("disable_train_aug", False),
        },
        "map_nonzero_fraction_mean": float(torch.tensor(map_nonzero).mean()) if map_nonzero else None,
        "map_nonzero_fraction_min": float(torch.tensor(map_nonzero).min()) if map_nonzero else None,
        "map_nonzero_fraction_max": float(torch.tensor(map_nonzero).max()) if map_nonzero else None,
        "map_all_zero_samples": all_zero_samples,
        "tensor_stats": {key: _summarize(value) for key, value in stats.items()},
        "coordinate_diagnostics": _coord_summary(gate, coord_metadata, args.max_coord_examples),
    }
    if token_cos:
        out["encoder_predictor_token_cosine_mean"] = float(torch.tensor(token_cos).mean())
    if token_rmse:
        out["encoder_predictor_token_rmse_mean"] = float(torch.tensor(token_rmse).mean())

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-coord-examples", type=int, default=12)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
