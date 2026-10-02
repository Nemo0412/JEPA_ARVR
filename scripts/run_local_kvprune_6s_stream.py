#!/usr/bin/env python3
"""Whole-video 6s KV-self stream. One P01 recording from model-frame 0.

Protocol (distinct from the clip trainer in run_local_kvprune_joint_2s.py):
  uniform 8 fps from t=0, not a per-annotation linspace window.
  fill 128 → score Probe.blocks[0] → drop 34 / keep 94 → encode the next 34
  (self-attn only) → packed 128 → probe CE when an annotation of the active
  split falls within half a step of (observation end + horizon).
  Probe always sees 128 frames. Cache is not reset until the video ends.

Prune history is empty on the first prune, so that prune matches last-prediction.
Ranking (protect_hist=3, protect_k=4 frames) first sees three prior Top-K sets
on the 4th prune. Abs (score_hist=3) takes max |score| over this pass and up
to two previous passes; the window is full from the 3rd prune.

Train loss uses train-split labels only. Val metrics use val-split labels only.
The cache still walks the whole video, including frames that belong to the
other split, because every P01 video appears in both splits.

One GPU. Videos have different lengths; a multi-rank DDP step loop would hang.
Truncated BPTT: each supervised step is one forward/backward, then the stream
state is detached.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader, cpu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", str(PROJECT_ROOT / "vjepa2")))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

from app.hdepic_lora_action_anticipation.encoder_lora import (  # noqa: E402
    inject_encoder_lora,
    set_encoder_lora_trainable,
)
from app.hdepic_lora_action_anticipation.stream_kvcache_attn_prune import (  # noqa: E402
    CACHE_FRAMES,
    NEW_FRAMES,
    max_abs_slot_scores,
    probe_blk0_slot_scores,
    token_frame_ids_from_slots,
    topk_slot_ids_from_scores,
)
from app.hdepic_lora_action_anticipation.train_stream_mtp import (  # noqa: E402
    build_model,
    map_labels,
)
from scripts.run_local_kvprune_joint_2s import (  # noqa: E402
    VIDEO_FPS,
    JointBundle,
    build_class_maps,
    cleanup_ddp,
    logger,
    normalize_clip,
    setup_ddp,
)

VIDEO_FPS = float(VIDEO_FPS)


def model_to_native(model_frame: int, video_fps: float, model_fps: float) -> int:
    return int(round(model_frame * float(video_fps) / float(model_fps)))


def n_model_frames(n_native: int, video_fps: float, model_fps: float) -> int:
    if n_native <= 1:
        return 1
    m = int((n_native - 1) * float(model_fps) / float(video_fps))
    while m > 0 and model_to_native(m, video_fps, model_fps) >= n_native:
        m -= 1
    while model_to_native(m + 1, video_fps, model_fps) < n_native:
        m += 1
    return m + 1


def n_stream_steps(n_model: int) -> int:
    if n_model < CACHE_FRAMES + NEW_FRAMES:
        return 0
    return (n_model - CACHE_FRAMES) // NEW_FRAMES


def obs_end_native(step: int, video_fps: float, model_fps: float) -> int:
    """Native frame of the newest model frame after 0-based prune step."""
    newest = CACHE_FRAMES + (int(step) + 1) * NEW_FRAMES - 1
    return model_to_native(newest, video_fps, model_fps)


def half_step_native(video_fps: float, model_fps: float) -> int:
    return int(round(0.5 * NEW_FRAMES * float(video_fps) / float(model_fps)))


def closest_step(
    desired_obs_end: int,
    n_steps: int,
    video_fps: float,
    model_fps: float,
    half: int,
) -> int | None:
    if n_steps <= 0:
        return None
    lo, hi = 0, n_steps - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if obs_end_native(mid, video_fps, model_fps) < desired_obs_end:
            lo = mid + 1
        else:
            hi = mid
    cands = [lo]
    if lo > 0:
        cands.append(lo - 1)
    best = min(
        cands,
        key=lambda s: abs(obs_end_native(s, video_fps, model_fps) - desired_obs_end),
    )
    if abs(obs_end_native(best, video_fps, model_fps) - desired_obs_end) > half:
        return None
    return best


@dataclass
class VideoJob:
    video_id: str
    path: Path
    n_native: int
    n_steps: int
    train_at: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    val_at: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    def labels(self, split: str) -> dict[int, list[tuple[int, int]]]:
        return self.train_at if split == "train" else self.val_at


def video_path(root: Path, video_id: str) -> Path | None:
    pid = video_id.split("_")[0]
    for ext in (".MP4", ".mp4"):
        path = root / pid / f"{video_id}{ext}"
        if path.is_file():
            return path
    return None


def _read_rows(csv_path: Path, participant: str) -> list[dict]:
    rows = []
    with Path(csv_path).open() as f:
        for r in csv.DictReader(f):
            if participant and str(r.get("participant_id", participant)) != participant:
                continue
            rows.append(r)
    return rows


def build_video_jobs(
    train_csv: Path,
    val_csv: Path,
    video_root: Path,
    *,
    participant: str,
    horizon_sec: float,
    model_fps: float,
    video_fps: float,
    img_size: int,
    max_videos: int = 0,
) -> list[VideoJob]:
    grouped: dict[str, dict[str, list[tuple[int, int, int]]]] = {}
    for split, path in (("train", train_csv), ("val", val_csv)):
        for r in _read_rows(path, participant):
            vid = str(r["video_id"])
            grouped.setdefault(vid, {"train": [], "val": []})
            grouped[vid][split].append(
                (int(r["start_frame"]), int(r["verb_class"]), int(r["noun_class"]))
            )
    half = half_step_native(video_fps, model_fps)
    horizon_native = int(round(float(horizon_sec) * float(video_fps)))
    jobs: list[VideoJob] = []
    for vid in sorted(grouped):
        path = video_path(video_root, vid)
        if path is None:
            logger.warning("missing video %s", vid)
            continue
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1, width=img_size, height=img_size)
        try:
            n_native = len(vr)
        finally:
            del vr
        n_model = n_model_frames(n_native, video_fps, model_fps)
        steps = n_stream_steps(n_model)
        job = VideoJob(video_id=vid, path=path, n_native=n_native, n_steps=steps)
        if steps > 0:
            for split, bucket in (("train", job.train_at), ("val", job.val_at)):
                for start, verb, noun in grouped[vid][split]:
                    desired = start - horizon_native
                    s = closest_step(desired, steps, video_fps, model_fps, half)
                    if s is None:
                        continue
                    bucket.setdefault(s, []).append((verb, noun))
        jobs.append(job)
        if max_videos > 0 and len(jobs) >= max_videos:
            break
    return jobs


def detach_state(state):
    state.tokens = state.tokens.detach()
    state.cache_k = [t.detach() for t in state.cache_k]
    state.cache_v = [t.detach() for t in state.cache_v]
    state.last_q = state.last_q.detach()
    state.last_k = state.last_k.detach()
    state.slot_scores = state.slot_scores.detach()
    return state


def decode_span(vr: VideoReader, m0: int, m1: int, n_native: int, video_fps: float, model_fps: float):
    idx = [
        min(max(model_to_native(m, video_fps, model_fps), 0), n_native - 1)
        for m in range(m0, m1)
    ]
    frames = vr.get_batch(idx).asnumpy()
    clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).contiguous()
    return clip.unsqueeze(0)


def _prune_inputs(raw, state, scores: torch.Tensor):
    """Match encode_joint: last / rank / abs. History is mutated by the caller."""
    mode = "attn_protect" if raw.prune_mode == "rank" else "attn"
    protected_ids = None
    slot_scores = scores
    cur_topk = None
    if raw.prune_mode == "rank":
        tubelet = int(raw.stream.tubelet_size)
        k_slots = max(0, int(raw.protect_k_frames) // tubelet)
        cur_topk = topk_slot_ids_from_scores(scores, state.slot_ids, k_slots)
        hist = raw._topk_hist
        if hist:
            protected_ids = torch.cat(hist[-int(raw.protect_hist) :], dim=1)
    elif raw.prune_mode == "abs":
        slot_scores = max_abs_slot_scores(
            state.slot_ids, scores, raw._score_hist, int(raw.score_hist)
        )
    return mode, slot_scores, protected_ids, cur_topk


def _remember_prune(raw, scored_ids, scores, cur_topk):
    if raw.prune_mode == "rank" and cur_topk is not None:
        raw._topk_hist.append(cur_topk.detach())
        keep = max(1, int(raw.protect_hist))
        if len(raw._topk_hist) > keep:
            raw._topk_hist = raw._topk_hist[-keep:]
    elif raw.prune_mode == "abs":
        raw._score_hist.append((scored_ids.detach(), scores.detach()))
        if len(raw._score_hist) > int(raw.score_hist):
            raw._score_hist = raw._score_hist[-int(raw.score_hist) :]


def _reset_hist(raw):
    raw._topk_hist = []
    raw._score_hist = []


def _forward_pack(raw, state):
    tok = state.tokens
    embed_dim = int(raw.backbone.embed_dim)
    if tok.size(-1) != embed_dim:
        tok = tok[:, :, -embed_dim:]
    frame_ids = token_frame_ids_from_slots(state.slot_ids, raw.stream.gp)
    if raw._rope is not None:
        raw._rope.set_frame_ids(frame_ids)
    try:
        return raw.clf(tok)
    finally:
        if raw._rope is not None:
            raw._rope.set_frame_ids(None)


def _loss_from_out(out, pairs, crit, maps, device):
    verbs = torch.tensor([v for v, _n in pairs], dtype=torch.long)
    nouns = torch.tensor([n for _v, n in pairs], dtype=torch.long)
    v_lab, n_lab, a_lab, keep = map_labels(verbs, nouns, maps[0], maps[1], maps[2], device)
    if not keep:
        return out["action"].sum() * 0.0, 0
    n = v_lab.shape[0]
    loss = crit(out["action"][:1].expand(n, -1), a_lab)
    if "verb" in out:
        loss = loss + crit(out["verb"][:1].expand(n, -1), v_lab) + crit(out["noun"][:1].expand(n, -1), n_lab)
    return loss, n


@torch.no_grad()
def _score_pack(raw, state, chunk: int):
    tok = state.tokens
    embed_dim = int(raw.backbone.embed_dim)
    if tok.size(-1) != embed_dim:
        tok = tok[:, :, -embed_dim:]
    return probe_blk0_slot_scores(raw.clf.pooler, tok.detach(), raw.stream.gp, chunk=chunk)


def run_video(
    raw,
    job: VideoJob,
    *,
    split: str,
    device,
    model_fps: float,
    video_fps: float,
    img_size: int,
    chunk: int,
    train: bool,
    crit,
    maps,
    opt,
    scaler,
):
    """Walk one video. Yields (stream_step, loss_float, n_labels) for supervised steps.

    In eval, also yields action-logit hits via the loss slot left unused: the
    caller passes train=False and reads the attached counters on the function
    attribute ``last_hits``.
    """
    labels = job.labels(split)
    _reset_hist(raw)
    vr = VideoReader(
        str(job.path), ctx=cpu(0), num_threads=1, width=img_size, height=img_size
    )
    hits5 = 0
    hits1 = 0
    n_eval = 0
    try:
        fill = decode_span(vr, 0, CACHE_FRAMES, job.n_native, video_fps, model_fps)
        with torch.no_grad():
            state = raw.stream.fill(normalize_clip(fill, device), refresh_scores=False)
        state = detach_state(state)
        del fill
        for step in range(job.n_steps):
            m0 = CACHE_FRAMES + step * NEW_FRAMES
            m1 = m0 + NEW_FRAMES
            pairs = labels.get(step) or []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = _score_pack(raw, state, chunk)
                mode, slot_scores, protected_ids, cur_topk = _prune_inputs(raw, state, scores)
                scored_ids = state.slot_ids
                new = decode_span(vr, m0, m1, job.n_native, video_fps, model_fps)
                clips_new = normalize_clip(new, device)
                del new
                supervised = bool(train and pairs)
                if supervised:
                    state = raw.stream.step(
                        state,
                        clips_new,
                        mode=mode,
                        slot_scores=slot_scores,
                        protected_ids=protected_ids,
                        refresh_scores=False,
                        self_only=True,
                    )
                    out = _forward_pack(raw, state)
                    loss, n_lab = _loss_from_out(out, pairs, crit, maps, device)
                else:
                    with torch.no_grad():
                        state = raw.stream.step(
                            state,
                            clips_new,
                            mode=mode,
                            slot_scores=slot_scores,
                            protected_ids=protected_ids,
                            refresh_scores=False,
                            self_only=True,
                        )
                        out = _forward_pack(raw, state) if pairs else None
            _remember_prune(raw, scored_ids, scores, cur_topk)
            step_yield = None
            if supervised:
                if not torch.isfinite(loss.detach()):
                    loss = out["action"].sum() * 0.0
                    n_lab = 0
                if n_lab > 0:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in raw.parameters() if p.requires_grad], 1.0
                    )
                    scaler.step(opt)
                    scaler.update()
                    step_yield = (step, float(loss.detach()), n_lab)
                opt.zero_grad(set_to_none=True)
                del loss, out
            elif pairs and not train:
                v = torch.tensor([p[0] for p in pairs], dtype=torch.long)
                n = torch.tensor([p[1] for p in pairs], dtype=torch.long)
                _vl, _nl, a_lab, keep = map_labels(v, n, maps[0], maps[1], maps[2], device)
                if keep:
                    logits = out["action"][:1].expand(a_lab.shape[0], -1)
                    top1 = logits.argmax(dim=-1)
                    top5 = logits.topk(min(5, logits.size(-1)), dim=-1).indices
                    hits1 += int((top1 == a_lab).sum().item())
                    hits5 += int((top5 == a_lab.unsqueeze(-1)).any(dim=-1).sum().item())
                    n_eval += int(a_lab.shape[0])
                del out
            state = detach_state(state)
            if step_yield is not None:
                yield step_yield
    finally:
        del vr
    run_video.last_hits = (hits5, hits1, n_eval)


def _trainable_sd(module: nn.Module) -> dict:
    return {n: p.detach().cpu() for n, p in module.named_parameters() if p.requires_grad}


def _truncate_steps(path: Path, keep_step: int):
    if not path.is_file():
        return
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    fields = list(rows[0].keys())
    kept = [r for r in rows if int(float(r["global_step"])) <= keep_step]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)


def main():
    ap = argparse.ArgumentParser(description="Whole-video KV-self stream finetune")
    ap.add_argument("--train-csv", type=Path, default=Path(
        "/mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_train_vjepa.csv"
    ))
    ap.add_argument("--val-csv", type=Path, default=Path(
        "/mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv"
    ))
    ap.add_argument("--video-root", type=Path, default=Path("/mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_videos"))
    ap.add_argument("--checkpoint", type=Path, default=Path("/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--participant", type=str, default="P01")
    ap.add_argument("--horizon", type=float, default=6.0)
    ap.add_argument("--rope", type=int, default=1, choices=[0, 1])
    ap.add_argument("--only-block0", type=int, default=0, choices=[0, 1])
    ap.add_argument("--rope-time", type=str, default="abs", choices=["abs", "relpred"])
    ap.add_argument("--prune-mode", type=str, default="last", choices=["last", "rank", "abs"])
    ap.add_argument("--protect-hist", type=int, default=3)
    ap.add_argument("--protect-k", type=int, default=4)
    ap.add_argument("--score-hist", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--resume", type=int, default=1, choices=[0, 1])
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-lr-mult", type=float, default=0.5)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--probe-depth", type=int, default=4)
    ap.add_argument("--probe-heads", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--dry-schedule", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    if args.prune_mode == "rank" and args.protect_hist <= 0:
        args.protect_hist = args.score_hist
    if args.rope_time == "relpred":
        raise SystemExit("whole-video stream uses --rope-time abs")

    jobs = build_video_jobs(
        args.train_csv, args.val_csv, args.video_root,
        participant=args.participant,
        horizon_sec=args.horizon,
        model_fps=float(args.fps),
        video_fps=VIDEO_FPS,
        img_size=args.img_size,
        max_videos=args.max_videos,
    )
    n_train = sum(len(v) for j in jobs for v in j.train_at.values())
    n_val = sum(len(v) for j in jobs for v in j.val_at.values())
    if args.dry_schedule or "RANK" not in os.environ:
        print(
            f"videos={len(jobs)} steps={sum(j.n_steps for j in jobs)} "
            f"train_labels={n_train} val_labels={n_val} "
            f"prune={args.prune_mode} protect_hist={args.protect_hist} "
            f"protect_k={args.protect_k} score_hist={args.score_hist}"
        )
        for j in jobs:
            print(
                f"  {j.video_id} native={j.n_native} steps={j.n_steps} "
                f"train={sum(len(v) for v in j.train_at.values())} "
                f"val={sum(len(v) for v in j.val_at.values())}"
            )
        if args.dry_schedule or "RANK" not in os.environ:
            return

    rank, world, local = setup_ddp()
    if world != 1:
        raise SystemExit(
            "whole-video stream is 1 GPU (videos differ in length; "
            "multi-rank DDP would hang). Launch one script per GPU."
        )
    device = torch.device(f"cuda:{local}")
    is_main = rank == 0
    out_dir = args.out_dir
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "stream-video participant=%s horizon=%.1fs rope=%s only_block0=%s time=%s "
            "prune=%s protect_hist=%d protect_k=%d score_hist=%d videos=%d "
            "train_labels=%d val_labels=%d steps=%d out=%s",
            args.participant, args.horizon, bool(args.rope), bool(args.only_block0),
            args.rope_time, args.prune_mode, args.protect_hist, args.protect_k,
            args.score_hist, len(jobs), n_train, n_val, sum(j.n_steps for j in jobs), out_dir,
        )

    verb_map, noun_map, action_map = build_class_maps(args.train_csv)
    maps = (verb_map, noun_map, action_map)
    backbone = build_model(
        device, CACHE_FRAMES, args.fps, args.img_size, str(args.checkpoint), no_predictor=True
    )
    for p in backbone.parameters():
        p.requires_grad = False
    inject_encoder_lora(
        backbone,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
        last_n_blocks=12,
        target_suffixes=("attn.qkv", "attn.proj"),
    )
    set_encoder_lora_trainable(backbone, trainable=True)
    from evals.action_anticipation_frozen.models import AttentiveClassifier

    clf = AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=int(backbone.embed_dim),
        num_heads=args.probe_heads,
        depth=args.probe_depth,
        use_activation_checkpointing=False,
    ).to(device)
    for p in clf.parameters():
        p.requires_grad = True
    bundle = JointBundle(
        backbone,
        clf,
        chunk=args.chunk,
        stream_steps=1,
        protect_hist=args.protect_hist,
        protect_k_frames=args.protect_k,
        no_kv=False,
        self_only=True,
        prune_mode=args.prune_mode,
        score_hist=args.score_hist,
    ).to(device)
    bundle.rope_time = args.rope_time
    bundle.horizon_sec = float(args.horizon)
    bundle.model_fps = float(args.fps)
    if args.rope:
        bundle.enable_rope(True, only_block0=bool(args.only_block0), cross_attn_k=False)
    # World size is 1. Call the module directly so a stream step does not have
    # to go through DDP.forward (videos are not equal-length batches).
    raw = bundle
    _reset_hist(raw)

    lora_params = [p for p in raw.backbone.parameters() if p.requires_grad]
    probe_params = [p for p in raw.clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": probe_params, "lr": args.lr},
            {"params": lora_params, "lr": args.lr * args.lora_lr_mult},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    crit = nn.CrossEntropyLoss()

    best_top5 = -1.0
    bad = 0
    history = []
    t_run = time.time()
    global_step = 0
    start_ep = 0
    start_video = 0
    loss_meter = 0.0
    n_step = 0
    loss_steps_path = out_dir / "loss_steps.csv"
    loss_epoch_path = out_dir / "loss_epoch.csv"
    latest_path = out_dir / "latest.pt"

    if args.resume and latest_path.is_file():
        ck = torch.load(latest_path, map_location="cpu")
        raw.clf.load_state_dict(ck["probe"], strict=False)
        raw.backbone.load_state_dict(ck["backbone"], strict=False)
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        best_top5 = float(ck.get("best_top5", -1.0))
        history = list(ck.get("history", []))
        global_step = int(ck.get("global_step", 0))
        start_ep = int(ck.get("epoch", 0))
        start_video = int(ck.get("next_video", 0))
        loss_meter = float(ck.get("loss_meter", 0.0))
        n_step = int(ck.get("n_step", 0))
        if is_main:
            _truncate_steps(loss_steps_path, global_step)
            logger.info(
                "resume epoch=%d next_video=%d global_step=%d best_top5=%.2f",
                start_ep, start_video, global_step, best_top5,
            )

    loss_steps_f = None
    loss_steps_w = None
    if is_main:
        new_file = not loss_steps_path.is_file() or loss_steps_path.stat().st_size == 0
        loss_steps_f = loss_steps_path.open("a", newline="")
        fields = [
            "global_step", "epoch", "video_index", "video_id", "stream_step",
            "loss", "avg_loss", "n_labels", "wall_sec",
        ]
        loss_steps_w = csv.DictWriter(loss_steps_f, fieldnames=fields)
        if new_file:
            loss_steps_w.writeheader()
            loss_steps_f.flush()
        if not loss_epoch_path.is_file():
            with loss_epoch_path.open("w", newline="") as ef:
                csv.DictWriter(
                    ef,
                    fieldnames=["epoch", "train_loss", "val_top5", "val_top1", "val_n", "ep_sec", "wall_sec"],
                ).writeheader()

    def save_ckpt(epoch: int, next_video: int, *, is_best: bool = False):
        if not is_main:
            return
        payload = {
            "epoch": int(epoch),
            "next_video": int(next_video),
            "partial": next_video > 0,
            "loss_meter": float(loss_meter),
            "n_step": int(n_step),
            "world_size": 1,
            "probe": _trainable_sd(raw.clf),
            "backbone": _trainable_sd(raw.backbone),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "best_top5": best_top5,
            "global_step": global_step,
            "history": history,
            "protocol": "stream_video",
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        }
        tmp = latest_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, latest_path)
        if is_best:
            best_path = out_dir / "best.pt"
            tmpb = best_path.with_suffix(".pt.tmp")
            torch.save(payload, tmpb)
            os.replace(tmpb, best_path)

    stop = {"flag": False}

    def _on_signal(signum, _frame):
        if is_main:
            logger.warning("signal %s — stop after this video", signum)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    early = False
    try:
        for ep in range(start_ep, args.epochs):
            if stop["flag"] or early:
                break
            order = list(range(len(jobs)))
            random.Random(ep).shuffle(order)
            skip = start_video if ep == start_ep else 0
            start_video = 0
            if ep != start_ep or skip == 0:
                loss_meter = 0.0
                n_step = 0
            bundle.train()
            t_ep = time.time()
            for vi, jidx in enumerate(order):
                if vi < skip or stop["flag"]:
                    continue
                job = jobs[jidx]
                if is_main:
                    logger.info(
                        "ep%d video %d/%d %s steps=%d train_labels=%d",
                        ep, vi + 1, len(order), job.video_id, job.n_steps,
                        sum(len(v) for v in job.train_at.values()),
                    )
                for stream_step, loss_val, n_lab in run_video(
                    raw, job, split="train", device=device, model_fps=float(args.fps),
                    video_fps=VIDEO_FPS, img_size=args.img_size, chunk=args.chunk,
                    train=True, crit=crit, maps=maps, opt=opt, scaler=scaler,
                ):
                    loss_meter += loss_val
                    n_step += 1
                    global_step += 1
                    if is_main:
                        avg = loss_meter / max(1, n_step)
                        loss_steps_w.writerow({
                            "global_step": global_step,
                            "epoch": ep,
                            "video_index": vi,
                            "video_id": job.video_id,
                            "stream_step": stream_step,
                            "loss": f"{loss_val:.6f}",
                            "avg_loss": f"{avg:.6f}",
                            "n_labels": n_lab,
                            "wall_sec": f"{time.time() - t_run:.1f}",
                        })
                        if global_step % args.log_every == 0:
                            loss_steps_f.flush()
                            logger.info(
                                "[ep%d step%d] loss=%.4f avg=%.4f video=%s stream_step=%d",
                                ep, global_step, loss_val, avg, job.video_id, stream_step,
                            )
                save_ckpt(ep, vi + 1)
                if is_main and loss_steps_f is not None:
                    loss_steps_f.flush()
            if stop["flag"]:
                break
            bundle.eval()
            hits5 = hits1 = n_eval = 0
            for job in jobs:
                for _step, _loss, _n in run_video(
                    raw, job, split="val", device=device, model_fps=float(args.fps),
                    video_fps=VIDEO_FPS, img_size=args.img_size, chunk=args.chunk,
                    train=False, crit=crit, maps=maps, opt=opt, scaler=scaler,
                ):
                    pass
                h5, h1, n = run_video.last_hits
                hits5 += h5
                hits1 += h1
                n_eval += n
            top5 = 100.0 * hits5 / max(1, n_eval)
            top1 = 100.0 * hits1 / max(1, n_eval)
            train_loss = loss_meter / max(1, n_step)
            ep_sec = time.time() - t_ep
            row = {
                "epoch": ep + 1,
                "train_loss": f"{train_loss:.6f}",
                "val_top5": f"{top5:.4f}",
                "val_top1": f"{top1:.4f}",
                "val_n": n_eval,
                "ep_sec": f"{ep_sec:.1f}",
                "wall_sec": f"{time.time() - t_run:.1f}",
            }
            history.append(row)
            is_best = top5 > best_top5
            if is_best:
                best_top5 = top5
                bad = 0
            else:
                bad += 1
            save_ckpt(ep + 1, 0, is_best=is_best)
            if is_main:
                with loss_epoch_path.open("a", newline="") as ef:
                    csv.DictWriter(ef, fieldnames=list(row.keys())).writerow(row)
                logger.info(
                    "epoch %d train_loss=%.4f val_top5=%.2f val_top1=%.2f n=%d",
                    ep + 1, train_loss, top5, top1, n_eval,
                )
            if args.patience > 0 and bad >= args.patience:
                early = True
                break
        if is_main and not stop["flag"] and not early:
            (out_dir / "DONE").write_text("ok\n")
    finally:
        if loss_steps_f is not None:
            loss_steps_f.flush()
            loss_steps_f.close()
        cleanup_ddp()


if __name__ == "__main__":
    main()
