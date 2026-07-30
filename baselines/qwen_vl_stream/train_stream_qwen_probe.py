#!/usr/bin/env python3
"""Qwen2.5-VL-3B LoRA probe on HD-EPIC P01 Streaming MTP (+2/+4/+6s).

Same annotation protocol as V-JEPA video-only stream MTP / RU-LSTM:
  half-split CSVs with mtp_verbs/nouns/mask for horizons 2/4/6s.

Design (decoder-VLM baseline, not compute-matched to ViT-L):
  - Freeze vision tower; LoRA on LLM q/k/v/o
  - Subsample context frames → --probe-num-frames (default 8)
  - Resize frames to --frame-size (default 256, closer to JEPA resolution)
  - Per-horizon verb/noun/action heads on last-token hidden state
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen_stream")

TASK_PROMPT = (
    "Based on this egocentric video context, predict the person's upcoming actions "
    "at +2s, +4s, and +6s (verb and noun)."
)
def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


def resample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    if frames.shape[0] <= n:
        return frames
    idx = np.linspace(0, frames.shape[0] - 1, n).round().astype(np.int64)
    return frames[idx]


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank
        d_in, d_out = linear.in_features, linear.out_features
        dev, dtype = linear.weight.device, linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(rank, d_in, device=dev, dtype=dtype) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=dev, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora_to_llm(model: nn.Module, rank: int, alpha: float) -> int:
    """Inject LoRA into LLM q/k/v/o; skip vision tower (name contains 'visual')."""
    n_injected = 0
    for mod_name, module in model.named_modules():
        if "visual" in mod_name:
            continue
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            orig = getattr(module, proj_name, None)
            if orig is None or not isinstance(orig, nn.Linear):
                continue
            for p in orig.parameters():
                p.requires_grad = False
            setattr(module, proj_name, LoRALinear(orig, rank=rank, alpha=alpha))
            n_injected += 1
    if n_injected == 0:
        raise RuntimeError("No LLM q/k/v/o projections found for LoRA.")
    for name, p in model.named_parameters():
        p.requires_grad = ("lora_A" in name) or ("lora_B" in name)
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA on %d projections; trainable LoRA %.2fM / total %.1fM",
        n_injected,
        n_lora / 1e6,
        n_total / 1e6,
    )
    return n_injected


class MultiHorizonHeads(nn.Module):
    def __init__(self, hidden: int, horizons: list[float], n_verb: int, n_noun: int, n_action: int):
        super().__init__()
        self.horizons = [float(h) for h in horizons]
        self.verb = nn.ModuleDict({f"{h:g}": nn.Linear(hidden, n_verb) for h in self.horizons})
        self.noun = nn.ModuleDict({f"{h:g}": nn.Linear(hidden, n_noun) for h in self.horizons})
        self.action = nn.ModuleDict({f"{h:g}": nn.Linear(hidden, n_action) for h in self.horizons})

    def forward(self, feat: torch.Tensor) -> dict:
        out = {}
        for h in self.horizons:
            key = f"{h:g}"
            out[h] = {
                "verb": self.verb[key](feat),
                "noun": self.noun[key](feat),
                "action": self.action[key](feat),
            }
        return out


class QwenStreamProbe(nn.Module):
    def __init__(self, backbone, hidden: int, horizons, n_verb, n_noun, n_action):
        super().__init__()
        self.backbone = backbone
        self.heads = MultiHorizonHeads(hidden, horizons, n_verb, n_noun, n_action)

    def forward(self, **inputs):
        out = self.backbone(**inputs, output_hidden_states=True, return_dict=True)
        feat = out.hidden_states[-1][:, -1, :].float()
        return self.heads(feat)


class StreamMTPDataset(Dataset):
    def __init__(self, csv_path: Path, video_root: Path, img_size: int, probe_frames: int):
        self.video_root = Path(video_root)
        self.img_size = int(img_size)
        self.probe_frames = int(probe_frames)
        with Path(csv_path).open() as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        video_id = str(r["video_id"])
        frame_idx = np.asarray(_parse_int_list(r["frame_indices"]), dtype=np.int64)
        pid = video_id.split("_")[0]
        path = self.video_root / pid / f"{video_id}.MP4"
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1, width=self.img_size, height=self.img_size)
        try:
            frame_idx = np.clip(frame_idx, 0, len(vr) - 1)
            frames = vr.get_batch(frame_idx.tolist()).asnumpy()
        finally:
            del vr
        frames = resample_frames(frames, self.probe_frames)
        return {
            "frames": frames,  # T,H,W,C uint8
            "context_sec": float(r["context_sec"]),
            "n_model_frames": int(r["n_model_frames"]),
            "mtp_verbs": torch.tensor(_parse_int_list(r["mtp_verbs"]), dtype=torch.long),
            "mtp_nouns": torch.tensor(_parse_int_list(r["mtp_nouns"]), dtype=torch.long),
            "mtp_mask": torch.tensor(_parse_float_list(r["mtp_mask"]), dtype=torch.float32),
        }


class ContextBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: StreamMTPDataset, batch_size: int, shuffle: bool, seed: int = 0):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.start_batch = 0
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, r in enumerate(dataset.rows):
            buckets[int(r["n_model_frames"])].append(i)
        self.buckets = dict(buckets)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch: int):
        self.start_batch = max(0, int(start_batch))

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        batches = []
        for idxs in self.buckets.values():
            order = list(idxs)
            if self.shuffle:
                rng.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                batches.append(order[i : i + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches[self.start_batch :]

    def __len__(self):
        return sum(math.ceil(len(v) / self.batch_size) for v in self.buckets.values())


def build_qwen_inputs(processor, frames_list: list[np.ndarray], frame_size: int):
    texts, videos = [], []
    for frames in frames_list:
        pil = [Image.fromarray(frames[t]) for t in range(frames.shape[0])]
        if frame_size > 0:
            pil = [im.resize((frame_size, frame_size)) for im in pil]
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil},
                    {"type": "text", "text": TASK_PROMPT},
                ],
            }
        ]
        texts.append(processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False))
        videos.append(pil)
    return processor(text=texts, videos=videos, return_tensors="pt", padding=True)


def load_action_maps(train_csv: Path):
    verbs, nouns, actions = {}, {}, {}
    with Path(train_csv).open() as f:
        for row in csv.DictReader(f):
            vs = _parse_int_list(row["mtp_verbs"])
            ns = _parse_int_list(row["mtp_nouns"])
            ms = _parse_float_list(row["mtp_mask"])
            for v, n, m in zip(vs, ns, ms):
                if m < 0.5 or v < 0 or n < 0:
                    continue
                if v not in verbs:
                    verbs[v] = len(verbs)
                if n not in nouns:
                    nouns[n] = len(nouns)
                if (v, n) not in actions:
                    actions[(v, n)] = len(actions)
    return verbs, nouns, actions


def map_labels(vs, ns, verb_map, noun_map, action_map, device):
    v_out, n_out, a_out, keep = [], [], [], []
    for i, (v, n) in enumerate(zip(vs.tolist(), ns.tolist())):
        if v not in verb_map or n not in noun_map or (v, n) not in action_map:
            continue
        keep.append(i)
        v_out.append(verb_map[v])
        n_out.append(noun_map[n])
        a_out.append(action_map[(v, n)])
    if not keep:
        return None, None, None, []
    return (
        torch.tensor(v_out, device=device, dtype=torch.long),
        torch.tensor(n_out, device=device, dtype=torch.long),
        torch.tensor(a_out, device=device, dtype=torch.long),
        keep,
    )


def topk_acc(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    if labels.numel() == 0:
        return 0.0
    pred = logits.topk(k, dim=-1).indices
    return float((pred == labels.unsqueeze(-1)).any(dim=-1).float().mean().item())


def save_ckpt(path: Path, **payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--loss-weights", type=str, default="1.0,0.7,0.5")
    ap.add_argument("--primary-horizon-sec", type=float, default=2.0)
    ap.add_argument("--probe-num-frames", type=int, default=8)
    ap.add_argument("--frame-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--val-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    weights = [float(x) for x in args.loss_weights.split(",")]
    assert len(horizons) == len(weights)
    primary_h = float(args.primary_horizon_sec)
    primary_idx = horizons.index(primary_h)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = args.out_dir / "TRAINING_DONE"
    if done_flag.is_file() and not args.val_only:
        logger.info("TRAINING_DONE present; exiting")
        return

    stop_flag = {"stop": False}

    def _on_signal(signum, _frame):
        logger.warning("Caught signal %s — stop after current step", signum)
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    verb_map, noun_map, action_map = load_action_maps(args.train_csv)
    logger.info("vocab v=%d n=%d a=%d", len(verb_map), len(noun_map), len(action_map))

    from transformers import AutoConfig, AutoProcessor

    def _resolve_vl_cls(model_id: str):
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=args.local_files_only)
        mt = str(getattr(cfg, "model_type", "") or "").lower()
        if mt in ("qwen2_vl",):
            from transformers import Qwen2VLForConditionalGeneration as Cls
        elif mt in ("qwen2_5_vl",):
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        elif mt in ("qwen3_vl",):
            from transformers import Qwen3VLForConditionalGeneration as Cls
        else:
            # fallback: try 2.5 then 2 then 3
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as Cls
            except ImportError:
                from transformers import Qwen2VLForConditionalGeneration as Cls
            logger.warning("Unknown model_type=%s; using %s", mt, Cls.__name__)
            return Cls, cfg
        return Cls, cfg

    logger.info("Loading %s ...", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    VLCls, _cfg = _resolve_vl_cls(args.model_id)
    logger.info("Using backbone class %s (model_type=%s)", VLCls.__name__, getattr(_cfg, "model_type", "?"))
    # Load then .to(device) — same pattern as ARVR_Video/qwen/train_hdepic_qwen_probe.py
    backbone = VLCls.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    apply_lora_to_llm(backbone, args.lora_rank, args.lora_alpha)

    # Frozen vision tower: drop activations (~VRAM). Gradient checkpointing for LLM.
    if hasattr(backbone, "visual"):
        _orig_visual_fwd = backbone.visual.forward

        @torch.no_grad()
        def _visual_no_grad(*a, **kw):
            return _orig_visual_fwd(*a, **kw)

        backbone.visual.forward = _visual_no_grad
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    hidden = int(getattr(backbone.config, "hidden_size", None) or backbone.config.text_config.hidden_size)
    model = QwenStreamProbe(
        backbone, hidden, horizons, len(verb_map), len(noun_map), len(action_map)
    )
    for p in model.heads.parameters():
        p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    param_info = {
        "model_id": args.model_id,
        "total_m": n_total / 1e6,
        "trainable_m": n_train / 1e6,
        "lora_rank": args.lora_rank,
        "probe_num_frames": args.probe_num_frames,
        "frame_size": args.frame_size,
        "horizons": horizons,
        "n_verb": len(verb_map),
        "n_noun": len(noun_map),
        "n_action": len(action_map),
        "jepa_vanilla_ref": "/scratch/ll5914/experiments/p01_stream_mtp_2_4_6",
    }
    (args.out_dir / "param_count.json").write_text(json.dumps(param_info, indent=2))
    logger.info("param_count %s", param_info)

    train_ds = StreamMTPDataset(args.train_csv, args.video_root, args.frame_size, args.probe_num_frames)
    val_ds = StreamMTPDataset(args.val_csv, args.video_root, args.frame_size, args.probe_num_frames)
    train_sampler = ContextBucketBatchSampler(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_sampler = ContextBucketBatchSampler(val_ds, args.batch_size, shuffle=False, seed=args.seed)

    def collate(batch):
        return {
            "frames": [b["frames"] for b in batch],
            "mtp_verbs": torch.stack([b["mtp_verbs"] for b in batch]),
            "mtp_nouns": torch.stack([b["mtp_nouns"] for b in batch]),
            "mtp_mask": torch.stack([b["mtp_mask"] for b in batch]),
            "context_sec": torch.tensor([b["context_sec"] for b in batch]),
        }

    loader_kw = dict(num_workers=args.num_workers, collate_fn=collate, pin_memory=False)
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kw)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kw)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    latest = args.out_dir / "latest.pt"
    best = -1.0
    history = []
    start_epoch = 0
    start_step = 0
    resume_phase = "train"
    if latest.is_file():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        if ck.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(ck["optimizer"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("optimizer restore failed: %s", exc)
        best = float(ck.get("best", -1.0))
        history = list(ck.get("history") or [])
        start_epoch = int(ck.get("epoch", 0))
        start_step = int(ck.get("step", 0))
        resume_phase = str(ck.get("phase", "train"))
        logger.info(
            "Resumed epoch=%d step=%d phase=%s best=%.4f",
            start_epoch,
            start_step,
            resume_phase,
            best,
        )

    def run_epoch(loader, sampler, train: bool, epoch: int, start_step: int = 0):
        model.train(train)
        sampler.set_epoch(epoch)
        sampler.set_start_batch(start_step)
        totals, counts = defaultdict(float), defaultdict(int)
        loss_meter = 0.0
        n_steps = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        last_it = start_step - 1
        stopped = False
        for local_it, batch in enumerate(loader):
            it = start_step + local_it
            if stop_flag["stop"]:
                stopped = True
                break
            last_it = it
            inputs = build_qwen_inputs(processor, batch["frames"], args.frame_size)
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            mtp_verbs = batch["mtp_verbs"].to(device)
            mtp_nouns = batch["mtp_nouns"].to(device)
            mtp_mask = batch["mtp_mask"].to(device)

            # bf16 weights; heads cast to float32 inside QwenStreamProbe — no GradScaler
            outputs = model(**inputs)
            head_loss = torch.zeros((), device=device)
            for hi, h in enumerate(horizons):
                valid = mtp_mask[:, hi] > 0.5
                if not bool(valid.any()):
                    continue
                v_lab, n_lab, a_lab, keep = map_labels(
                    mtp_verbs[valid, hi], mtp_nouns[valid, hi], verb_map, noun_map, action_map, device
                )
                if not keep:
                    continue
                valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                o = outputs[h]
                step_loss = (
                    F.cross_entropy(o["verb"][valid_pos], v_lab)
                    + F.cross_entropy(o["noun"][valid_pos], n_lab)
                    + F.cross_entropy(o["action"][valid_pos], a_lab)
                )
                head_loss = head_loss + weights[hi] * step_loss
                with torch.no_grad():
                    for name, logits, lab in (
                        ("verb", o["verb"][valid_pos].float(), v_lab),
                        ("noun", o["noun"][valid_pos].float(), n_lab),
                        ("action", o["action"][valid_pos].float(), a_lab),
                    ):
                        for k in (1, 5):
                            key = f"{name}_top{k}@{h:g}s"
                            totals[key] += topk_acc(logits, lab, k=k) * len(keep)
                            counts[key] += len(keep)

            # primary
            h0 = horizons[primary_idx]
            valid = mtp_mask[:, primary_idx] > 0.5
            if bool(valid.any()):
                v_lab, n_lab, a_lab, keep = map_labels(
                    mtp_verbs[valid, primary_idx],
                    mtp_nouns[valid, primary_idx],
                    verb_map,
                    noun_map,
                    action_map,
                    device,
                )
                if keep:
                    valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                    o = outputs[h0]
                    totals["primary_action_top5"] += topk_acc(o["action"][valid_pos].float(), a_lab, 5) * len(keep)
                    counts["primary_action_top5"] += len(keep)
                    totals["primary_action_top1"] += topk_acc(o["action"][valid_pos].float(), a_lab, 1) * len(keep)
                    counts["primary_action_top1"] += len(keep)

            if train:
                if not torch.isfinite(head_loss.detach()):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss = head_loss / max(1, args.grad_accum)
                loss.backward()
                if (n_steps + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            loss_meter += float(head_loss.detach().item()) if torch.isfinite(head_loss.detach()) else 0.0
            n_steps += 1
            if it % args.log_every == 0:
                p5 = totals["primary_action_top5"] / max(1, counts["primary_action_top5"])
                logger.info(
                    "%s itr=%d/%d loss=%.4f primary@%gs top5≈%.1f ctx=%.0fs",
                    "train" if train else "val",
                    it,
                    len(sampler),
                    loss_meter / max(1, n_steps),
                    primary_h,
                    100.0 * p5,
                    float(batch["context_sec"][0]),
                )
            if train and args.save_every > 0 and it > 0 and it % args.save_every == 0:
                save_ckpt(
                    latest,
                    epoch=epoch,
                    step=it + 1,
                    phase="train",
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    best=best,
                    history=history,
                )
                logger.info("Periodic checkpoint at train step=%d", it)

        metrics = {k: totals[k] / max(1, counts[k]) for k in totals}
        metrics["loss"] = loss_meter / max(1, n_steps)
        metrics["seconds"] = time.time() - t0
        metrics["last_step"] = int(last_it + 1)
        metrics["stopped_early"] = stopped
        return metrics

    if args.val_only:
        va = run_epoch(val_loader, val_sampler, train=False, epoch=0, start_step=0)
        (args.out_dir / "val_only_metrics.json").write_text(json.dumps(va, indent=2))
        logger.info("VAL_ONLY %s", {k: round(100 * v, 2) if isinstance(v, float) and v <= 1.5 else v for k, v in va.items() if "top" in k or k == "loss"})
        return

    for epoch in range(start_epoch, args.epochs):
        skip_train = epoch == start_epoch and resume_phase == "val"
        train_start = start_step if (epoch == start_epoch and resume_phase == "train") else 0
        val_start = start_step if (epoch == start_epoch and resume_phase == "val") else 0

        if not skip_train:
            tr = run_epoch(train_loader, train_sampler, train=True, epoch=epoch, start_step=train_start)
            train_sampler.set_start_batch(0)
            if tr.get("stopped_early"):
                save_ckpt(
                    latest,
                    epoch=epoch,
                    step=int(tr["last_step"]),
                    phase="train",
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    best=best,
                    history=history,
                )
                logger.warning("Stopped early during train epoch=%d", epoch)
                return
            save_ckpt(
                latest,
                epoch=epoch,
                step=0,
                phase="val",
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                best=best,
                history=history,
            )
        else:
            tr = {"loss": float("nan"), "stopped_early": False, "last_step": 0}
            logger.info("Skipping train for epoch %d (resume_phase=val step=%d)", epoch, val_start)

        va = run_epoch(val_loader, val_sampler, train=False, epoch=epoch, start_step=val_start)
        val_sampler.set_start_batch(0)
        if va.get("stopped_early"):
            save_ckpt(
                latest,
                epoch=epoch,
                step=int(va["last_step"]),
                phase="val",
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                best=best,
                history=history,
            )
            logger.warning("Stopped early during val epoch=%d", epoch)
            return

        primary = float(va.get("primary_action_top5", 0.0))
        logger.info(
            "epoch %d train_loss=%.4f val_primary_top5=%.2f%% top1=%.2f%% %s",
            epoch,
            tr["loss"],
            100.0 * primary,
            100.0 * float(va.get("primary_action_top1", 0.0)),
            {k: round(100.0 * va[k], 2) for k in sorted(va) if k.startswith("action_top")},
        )
        history.append({"epoch": epoch, "train": {k: v for k, v in tr.items()}, "val": {k: v for k, v in va.items()}})
        save_ckpt(
            latest,
            epoch=epoch + 1,
            step=0,
            phase="train",
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            best=best,
            history=history,
        )
        if primary > best:
            best = primary
            save_ckpt(
                args.out_dir / "best.pt",
                epoch=epoch + 1,
                step=0,
                phase="train",
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                best=best,
                history=history,
            )
            logger.info("New best primary_action_top5=%.4f", best)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
        start_step = 0
        resume_phase = "train"

    done_flag.write_text(f"finished epochs={args.epochs} best={best}\n")
    logger.info("Done. best primary_action_top5=%.4f", best)


if __name__ == "__main__":
    main()
