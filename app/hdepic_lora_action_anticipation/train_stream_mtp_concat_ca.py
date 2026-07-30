#!/usr/bin/env python3
"""Streaming MTP with concat+crossattention (video+gaze+pose) backbone.

Same stream half-split protocol as ``train_stream_mtp.py`` (grow 4→10s,
predict +2/+4/+6s), but warm-starts from concat+CA v2 (43.92% @1s):
  5ch BinaryMapInputAdapter (gaze+pose) → encoder → IMU CA → predictor,
  then Communicating-MLP MTP heads.

Prune: after fusion, prune the *video* token suffix (keep≤4096) while
preserving the keep_aux IMU prefix — needed for 10s contexts.

Modes (``--prune-mode``):
  * ``encoder_attn`` (legacy): top-K by encoder last-block attention receive.
  * ``postfuse_recency`` (P0): top-K by *post-fusion* self-attn receive on
    fused video tokens, reweighted by temporal recency (favor recent slots).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import signal
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VJEPA_ROOT))

from app.hdepic_lora_action_anticipation.binary_input_adapter import (  # noqa: E402
    BinaryMapInputAdapter,
)
from app.hdepic_lora_action_anticipation.concat_plus_cross_attn import (  # noqa: E402
    ConcatPlusCrossAttnAdaptedModel,
)
from app.hdepic_lora_action_anticipation.gaze import GazeTokenGate  # noqa: E402
from app.hdepic_lora_action_anticipation.mtp import CommunicatingMLPMTPClassifier  # noqa: E402
from app.hdepic_lora_action_anticipation.pose_map_builder import GazePoseInputMapBuilder  # noqa: E402
from app.hdepic_lora_action_anticipation.tri_modal_fusion import (  # noqa: E402
    ImuTemporalEncoder,
    ImuTrajectoryLoader,
    ProjectedTriModalCrossAttention,
    compute_token_budgets,
    load_tri_modal_fusion_checkpoint,
    trainable_tri_modal_fusion_params,
)
from app.hdepic_lora_action_anticipation import train_stream_mtp as base  # noqa: E402
from evals.action_anticipation_frozen.models import AttentiveClassifier  # noqa: E402
from src.utils.checkpoint_loader import robust_checkpoint_loader  # noqa: E402

logger = logging.getLogger("stream_mtp_concat_ca")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class StreamMTPConcatCADataset(base.StreamMTPDataset):
    """Video decode + gaze/pose map + IMU for each stream tick.

    Builders hold ``threading.Lock`` (unpicklable). Store only ``gaze_cfg`` and
    lazily construct per-process so DataLoader ``num_workers>0`` (spawn) works.

    Optional ``tick_cache_dir``: disk cache of (clip, aux, imu) keyed by
    video_id+frame_indices — first epoch fills cache, later epochs are fast.
    """

    def __init__(
        self,
        csv_path: Path,
        video_root: Path,
        img_size: int,
        gaze_cfg: dict,
        tick_cache_dir: Path | None = None,
    ):
        super().__init__(csv_path, video_root, img_size)
        self.gaze_cfg = dict(gaze_cfg)
        self.tick_cache_dir = Path(tick_cache_dir) if tick_cache_dir else None
        if self.tick_cache_dir is not None:
            self.tick_cache_dir.mkdir(parents=True, exist_ok=True)
        self._map_builder = None
        self._imu_loader = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_map_builder"] = None
        state["_imu_loader"] = None
        return state

    def _ensure_aux_loaders(self):
        if self._map_builder is None or self._imu_loader is None:
            gate = GazeTokenGate({**self.gaze_cfg, "mode": "token_gate", "learnable_gate": False})
            self._map_builder = GazePoseInputMapBuilder(self.gaze_cfg, gate=gate)
            self._imu_loader = ImuTrajectoryLoader(self.gaze_cfg, gate=gate)

    @staticmethod
    def _cache_key(video_id: str, frame_idx: np.ndarray) -> str:
        raw = f"{video_id}|{','.join(map(str, frame_idx.tolist()))}".encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    def _cache_path(self, video_id: str, frame_idx: np.ndarray) -> Path | None:
        if self.tick_cache_dir is None:
            return None
        return self.tick_cache_dir / f"{self._cache_key(video_id, frame_idx)}.pt"

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        video_id = str(r["video_id"])
        frame_idx = np.asarray(base._parse_int_list(r["frame_indices"]), dtype=np.int64)
        cache_path = self._cache_path(video_id, frame_idx)
        if cache_path is not None and cache_path.is_file():
            try:
                blob = torch.load(cache_path, map_location="cpu", weights_only=False)
                sample = {
                    "clip": blob["clip"],
                    "aux_map": blob["aux_map"].float(),
                    "imu": blob["imu"],
                    "imu_len": blob["imu_len"],
                    "context_sec": float(r["context_sec"]),
                    "n_frames": int(blob["clip"].shape[1]),
                    "mtp_verbs": torch.tensor(base._parse_int_list(r["mtp_verbs"]), dtype=torch.long),
                    "mtp_nouns": torch.tensor(base._parse_int_list(r["mtp_nouns"]), dtype=torch.long),
                    "mtp_mask": torch.tensor(base._parse_float_list(r["mtp_mask"]), dtype=torch.float32),
                    "video_id": video_id,
                }
                return sample
            except Exception:  # noqa: BLE001
                pass

        self._ensure_aux_loaders()
        sample = super().__getitem__(idx)
        vfps = float(r.get("vfps") or 30.0)
        meta = {
            "video_id": video_id,
            "frame_indices": frame_idx,
            "vfps": vfps,
            "start_frame": int(r.get("start_frame") or frame_idx[0]),
        }
        t = int(sample["clip"].shape[1])
        h = int(sample["clip"].shape[2])
        w = int(sample["clip"].shape[3])
        aux = self._map_builder.build_cpu([meta], t, h, w)[0]  # 2,T,H,W
        imu = self._imu_loader.load_batch([meta], torch.device("cpu"))
        sample["aux_map"] = aux
        sample["imu"] = imu[0][0] if imu is not None else torch.zeros(t, 128, 6)
        sample["imu_len"] = imu[1][0] if imu is not None else torch.ones(t, dtype=torch.long)
        sample["video_id"] = video_id

        if cache_path is not None:
            try:
                tmp = cache_path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "clip": sample["clip"],
                        "aux_map": sample["aux_map"].half(),  # space: float16 maps
                        "imu": sample["imu"],
                        "imu_len": sample["imu_len"],
                    },
                    tmp,
                )
                os.replace(tmp, cache_path)
            except Exception:  # noqa: BLE001
                pass
            else:
                sample["aux_map"] = sample["aux_map"].float()
        return sample


class AdaptiveContextBucketBatchSampler(Sampler[list[int]]):
    """Like ContextBucketBatchSampler but bs depends on n_model_frames."""

    def __init__(
        self,
        dataset: base.StreamMTPDataset,
        batch_size_by_frames: dict[int, int],
        default_batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.batch_size_by_frames = {int(k): int(v) for k, v in batch_size_by_frames.items()}
        self.default_batch_size = int(default_batch_size)
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

    def _bs(self, n_frames: int) -> int:
        return max(1, int(self.batch_size_by_frames.get(int(n_frames), self.default_batch_size)))

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        batches = []
        for n_frames, idxs in self.buckets.items():
            order = list(idxs)
            bs = self._bs(n_frames)
            if self.shuffle:
                rng.shuffle(order)
            for i in range(0, len(order), bs):
                batches.append(order[i : i + bs])
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches[self.start_batch :]

    def __len__(self):
        return sum(math.ceil(len(v) / self._bs(n)) for n, v in self.buckets.items())


def collate_stream_concat_ca(batch):
    out = base.collate_stream(batch)
    out["aux_map"] = torch.stack([b["aux_map"] for b in batch], dim=0)
    # Pad IMU temporal axis across batch.
    max_t = max(int(b["imu"].shape[0]) for b in batch)
    k = int(batch[0]["imu"].shape[1])
    imu = torch.zeros(len(batch), max_t, k, 6, dtype=torch.float32)
    imu_len = torch.zeros(len(batch), max_t, dtype=torch.long)
    for i, b in enumerate(batch):
        t = int(b["imu"].shape[0])
        imu[i, :t] = b["imu"]
        imu_len[i, :t] = b["imu_len"][:t]
    out["imu"] = imu
    out["imu_len"] = imu_len
    return out


def _load_adapter_ckpt(adapter: BinaryMapInputAdapter, path: str):
    ck = robust_checkpoint_loader(str(path), map_location=torch.device("cpu"))
    state = ck.get("input_adapter", ck)
    if any(str(k).startswith("module.input_adapter.") for k in state):
        state = {
            str(k).removeprefix("module.input_adapter."): v
            for k, v in state.items()
            if str(k).startswith("module.input_adapter.")
        }
    elif any(str(k).startswith("input_adapter.") for k in state):
        state = {
            str(k).removeprefix("input_adapter."): v
            for k, v in state.items()
            if str(k).startswith("input_adapter.")
        }
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    logger.info("Loaded adapter %s missing=%d unexpected=%d", path, len(missing), len(unexpected))


class PrunedConcatCAStreamModel(nn.Module):
    """concat+CA forward with post-fusion video-token prune before predictor AR.

    ``prune_mode``:
      - ``encoder_attn``: legacy — encoder last-attn importance on video suffix.
      - ``postfuse_recency``: P0 — importance from fused video self-attn receive,
        multiplied by a temporal recency prior (later slots preferred).
    """

    def __init__(
        self,
        concat_ca: ConcatPlusCrossAttnAdaptedModel,
        pruner: base.TokenPruner | None,
        keep: int,
        *,
        gp: int,
        prune_mode: str = "encoder_attn",
        recency_strength: float = 1.0,
        prune_chunk_size: int = 256,
    ):
        super().__init__()
        self.concat_ca = concat_ca
        self.pruner = pruner
        self.keep = int(keep)
        self.gp = int(gp)
        self.prune_mode = str(prune_mode)
        self.recency_strength = float(recency_strength)
        self.prune_chunk_size = int(prune_chunk_size)
        self.embed_dim = concat_ca.embed_dim
        if self.prune_mode not in ("encoder_attn", "postfuse_recency"):
            raise ValueError(f"Unknown prune_mode={self.prune_mode!r}")

    @staticmethod
    def _postfuse_attn_importance(video: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        """Attention-receive scores on fused video tokens (no extra params).

        importance[j] = sum_queries softmax(q_i · k_j). Uses L2-normalized
        token features as Q/K (chunked over queries for memory).
        """
        B, N, D = video.shape
        x = F.normalize(video.float(), dim=-1)
        scale = D**-0.5
        imp = torch.zeros(B, N, device=video.device, dtype=torch.float32)
        for ci in range(0, N, chunk_size):
            q = x[:, ci : ci + chunk_size]  # [B, C, D]
            logits = torch.bmm(q, x.transpose(1, 2)) * scale  # [B, C, N]
            imp += logits.softmax(dim=-1).sum(dim=1)
        return imp

    def _recency_weights(self, n_tokens: int, device: torch.device) -> torch.Tensor:
        """Per-token recency in (0, 1], higher for later temporal slots."""
        gp = max(1, self.gp)
        n_slots = max(1, (n_tokens + gp - 1) // gp)
        slot = torch.arange(n_tokens, device=device) // gp
        # Linear ramp: first slot ~1/n_slots, last slot ~1.
        return (slot.float() + 1.0) / float(n_slots)

    def _score_video_tokens(self, video: torch.Tensor) -> torch.Tensor:
        """[B, N] prune scores for fused video suffix."""
        B, N, _D = video.shape
        if self.prune_mode == "postfuse_recency":
            scores = self._postfuse_attn_importance(video, chunk_size=self.prune_chunk_size)
            if self.recency_strength > 0:
                rec = self._recency_weights(N, video.device).unsqueeze(0).expand(B, -1)
                scores = scores * (1.0 + self.recency_strength * rec)
            return scores

        # encoder_attn (legacy)
        if self.pruner is None or self.pruner._importance is None:
            raise RuntimeError("TokenPruner importance missing after encoder")
        imp = self.pruner._importance
        if imp.size(1) != N:
            return video.float().norm(dim=-1)
        return imp

    def _prune_video_suffix(self, x_pred: torch.Tensor, n_aux: int) -> torch.Tensor:
        video = x_pred[:, n_aux:]
        B, N, D = video.shape
        if N <= self.keep:
            return x_pred
        gp = self.gp if self.pruner is None else self.pruner.gp
        K = min(self.keep, (N // gp) * gp)
        if K >= N:
            return x_pred
        scores = self._score_video_tokens(video)
        _, idx = scores.topk(K, dim=1)
        idx = idx.sort(dim=1).values
        video = video.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        return torch.cat([x_pred[:, :n_aux], video], dim=1) if n_aux > 0 else video

    def forward(self, clips, anticipation_times, aux_map=None, imu_batch=None):
        tri = self.concat_ca.tri
        base_m = tri.base_model
        if aux_map is not None:
            clips = self.concat_ca.input_adapter(clips, aux_map)
        x_full = base_m.encoder(clips)
        if not torch.isfinite(x_full).all():
            return None
        if not any(p.requires_grad for p in base_m.encoder.parameters()):
            x_full = x_full.detach()

        embed_dim = base_m.encoder.embed_dim
        use_hier = x_full.size(-1) > embed_dim
        x_last = x_full[:, :, -embed_dim:] if use_hier else x_full
        x_accumulate = x_last.clone()

        x_pred = tri._fuse_for_predictor(x_full, gaze_map=None, imu_batch=imu_batch)
        if x_pred is None or not torch.isfinite(x_pred).all():
            return None

        n_aux = int(getattr(tri, "_n_aux_context_tokens", 0) or 0)
        if x_pred.size(1) - n_aux > self.keep:
            x_pred = self._prune_video_suffix(x_pred, n_aux)

        return tri._forward_single_step(base_m, x_pred, x_accumulate, anticipation_times)


def build_concat_ca_model(
    device,
    max_frames: int,
    fps: int,
    img_size: int,
    checkpoint: str,
    encoder_lora: str | None,
    predictor_lora: str | None,
    adapter_ckpt: str,
    fusion_ckpt: str,
    keep_count: int,
    freeze_adapter: bool = False,
    freeze_fusion: bool = False,
    freeze_encoder_lora: bool = True,
    fusion_num_layers: int = 3,
    reset_gate_bias: float | None = None,
    prune_mode: str = "encoder_attn",
    recency_strength: float = 1.0,
):
    base_model = base.build_model(device, max_frames, fps, img_size, checkpoint)
    for p in base_model.encoder.parameters():
        p.requires_grad = False
    base.load_lora_sidecars(base_model, encoder_lora, predictor_lora)

    embed_dim = int(base_model.embed_dim)
    grid_size = int(base_model.grid_size)
    n_v, _n_g, n_i = compute_token_budgets(
        grid_size * grid_size, gaze_grid_size=10, gaze_token_ratio=0.5, imu_token_ratio=0.1
    )
    adapter = BinaryMapInputAdapter(
        hidden_dim=8, scale=1.0, temporal_kernel=3, binary_center=0.0, residual_clamp=1.0, in_channels=5
    ).to(device)
    _load_adapter_ckpt(adapter, adapter_ckpt)
    for p in adapter.parameters():
        p.requires_grad = not freeze_adapter

    fusion_cfg = {
        "use_gaze_branch": False,
        "use_imu_branch": True,
        "keep_aux_tokens_in_predictor": True,
        "gaze_grid_size": 10,
        "gaze_token_ratio": 0.5,
        "imu_token_ratio": 0.1,
        "imu_encoder_type": "gru",
        "imu_hidden_dim": 128,
        "imu_num_layers": 1,
        "imu_dropout": 0.1,
        "fusion_num_heads": 4,
        "fusion_num_layers": int(fusion_num_layers),
        "use_gated_residual": True,
        "gate_bias_init": -2.0,
        "dropout": 0.0,
    }
    fusion = ProjectedTriModalCrossAttention(
        embed_dim=embed_dim,
        attn_dim=embed_dim,
        num_heads=4,
        num_layers=int(fusion_num_layers),
        dropout=0.0,
        use_gated_residual=True,
        use_gaze_branch=False,
        use_imu_branch=True,
        gate_bias_init=-2.0,
    ).to(device)
    imu_encoder = ImuTemporalEncoder(
        embed_dim=embed_dim,
        input_dim=6,
        hidden_dim=128,
        num_imu_tokens=n_i,
        encoder_type="gru",
        num_layers=1,
        dropout=0.1,
    ).to(device)
    wrapped = ConcatPlusCrossAttnAdaptedModel(
        base_model, input_adapter=adapter, fusion=fusion, imu_encoder=imu_encoder, fusion_cfg=fusion_cfg
    )
    for p in wrapped.base_model.parameters():
        p.requires_grad = False
    # Re-enable predictor LoRA (load_lora_sidecars set it; the blanket freeze above cleared it).
    try:
        from app.hdepic_lora_action_anticipation.predictor_lora import set_predictor_lora_trainable

        set_predictor_lora_trainable(wrapped.base_model, trainable=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not re-enable predictor LoRA: %s", exc)
    # Load fusion weights first, then freeze or leave trainable.
    if Path(fusion_ckpt).is_file():
        load_tri_modal_fusion_checkpoint(wrapped, fusion_ckpt)
        logger.info("Loaded fusion from %s", fusion_ckpt)
    for p in trainable_tri_modal_fusion_params(wrapped):
        p.requires_grad = not freeze_fusion
    if freeze_fusion:
        logger.info("Froze fusion + IMU encoder params (use 43.92%% as fixed backbone)")
    if freeze_encoder_lora:
        try:
            from app.hdepic_lora_action_anticipation.encoder_lora import set_encoder_lora_trainable

            set_encoder_lora_trainable(wrapped.base_model, trainable=False)
        except Exception:  # noqa: BLE001
            pass
    if reset_gate_bias is not None:
        for proj in (wrapped.fusion.video_proj, wrapped.fusion.imu_proj, wrapped.fusion.gaze_proj):
            if proj is None:
                continue
            nn.init.constant_(proj.gate[2].bias, float(reset_gate_bias))
        logger.info("Reset fusion gate bias to %.2f", float(reset_gate_bias))

    gp = int(grid_size**2)
    # encoder_attn needs last-block importance hook; postfuse_recency scores fused tokens.
    pruner = None
    if prune_mode == "encoder_attn":
        pruner = base.TokenPruner(wrapped.base_model.encoder, keep_count=keep_count, gp=gp)
    model = PrunedConcatCAStreamModel(
        wrapped,
        pruner,
        keep=keep_count,
        gp=gp,
        prune_mode=prune_mode,
        recency_strength=recency_strength,
    ).to(device)
    n_pred = sum(p.numel() for p in wrapped.base_model.parameters() if p.requires_grad)
    logger.info(
        "Built concat+CA stream MTP: n_video_spatial=%d n_imu=%d keep=%d adapter_train=%d "
        "fusion_train=%d pred_lora_train=%d freeze_fusion=%s prune_mode=%s recency=%.2f",
        n_v,
        n_i,
        keep_count,
        sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        sum(p.numel() for p in trainable_tri_modal_fusion_params(wrapped) if p.requires_grad),
        n_pred,
        freeze_fusion,
        prune_mode,
        recency_strength,
    )
    return model


def run_epoch(
    model,
    classifier,
    loader,
    device,
    horizons,
    weights,
    primary_idx,
    verb_map,
    noun_map,
    action_map,
    optimizer=None,
    scaler=None,
    train: bool = True,
    anticipation_sec: float = 2.0,
    log_every: int = 20,
    start_step: int = 0,
    save_every: int = 0,
    save_fn=None,
    stop_flag=None,
    metric_state=None,
):
    model.train(mode=train)
    classifier.train(mode=train)
    crit = nn.CrossEntropyLoss()
    totals = defaultdict(float)
    counts = defaultdict(int)
    loss_meter = 0.0
    n_steps = 0
    if metric_state:
        for k, v in (metric_state.get("totals") or {}).items():
            totals[k] = float(v)
        for k, v in (metric_state.get("counts") or {}).items():
            counts[k] = int(v)
        loss_meter = float(metric_state.get("loss_meter", 0.0))
        n_steps = int(metric_state.get("n_steps", 0))
    import time

    t0 = time.time()
    stopped_early = False
    last_it = start_step - 1
    for local_it, batch in enumerate(loader):
        it = start_step + local_it
        if stop_flag is not None and stop_flag["stop"]:
            stopped_early = True
            break
        last_it = it
        clips = batch["clip"].to(device, non_blocking=True).float().div_(255.0)
        clips = clips.sub_(base.IMAGENET_MEAN.to(device)).div_(base.IMAGENET_STD.to(device))
        aux = batch["aux_map"].to(device, non_blocking=True).float()
        imu = batch["imu"].to(device, non_blocking=True).float()
        imu_len = batch["imu_len"].to(device, non_blocking=True)
        mtp_verbs = batch["mtp_verbs"].to(device, non_blocking=True)
        mtp_nouns = batch["mtp_nouns"].to(device, non_blocking=True)
        mtp_mask = batch["mtp_mask"].to(device, non_blocking=True)
        B = clips.size(0)
        ant = torch.full((B,), float(anticipation_sec), device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            tokens = model(clips, ant, aux_map=aux, imu_batch=(imu, imu_len))
            if tokens is None:
                if train:
                    optimizer.zero_grad(set_to_none=True)
                continue
            outputs = classifier(tokens)
            head_loss = clips.new_zeros(())
            for hi, h in enumerate(horizons):
                valid = mtp_mask[:, hi] > 0.5
                if not bool(valid.any()):
                    continue
                v_lab, n_lab, a_lab, keep = base.map_labels(
                    mtp_verbs[valid, hi], mtp_nouns[valid, hi], verb_map, noun_map, action_map, device
                )
                if not keep:
                    continue
                valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                o = outputs[float(h)]
                step_loss = (
                    crit(o["verb"][valid_pos], v_lab)
                    + crit(o["noun"][valid_pos], n_lab)
                    + crit(o["action"][valid_pos], a_lab)
                )
                head_loss = head_loss + float(weights[hi]) * step_loss
                with torch.no_grad():
                    key = f"action_top5@{h:g}s"
                    totals[key] += base.topk_acc(o["action"][valid_pos].float(), a_lab, k=5) * len(keep)
                    counts[key] += len(keep)

            h0 = horizons[primary_idx]
            valid = mtp_mask[:, primary_idx] > 0.5
            if bool(valid.any()):
                v_lab, n_lab, a_lab, keep = base.map_labels(
                    mtp_verbs[valid, primary_idx],
                    mtp_nouns[valid, primary_idx],
                    verb_map,
                    noun_map,
                    action_map,
                    device,
                )
                if keep:
                    valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                    o = outputs[float(h0)]
                    totals["primary_action_top5"] += base.topk_acc(o["action"][valid_pos].float(), a_lab, k=5) * len(
                        keep
                    )
                    counts["primary_action_top5"] += len(keep)

        if train:
            if not torch.isfinite(head_loss.detach()):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(head_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()

        loss_meter += float(head_loss.detach().item()) if torch.isfinite(head_loss.detach()) else 0.0
        n_steps += 1
        if it % log_every == 0:
            logger.info(
                "%s itr=%d/%d loss=%.4f @2s≈%.1f @4s≈%.1f @6s≈%.1f ctx=%.0fs",
                "train" if train else "val",
                it,
                len(loader),
                loss_meter / max(1, n_steps),
                100.0 * totals.get("action_top5@2s", 0.0) / max(1, counts.get("action_top5@2s", 0)),
                100.0 * totals.get("action_top5@4s", 0.0) / max(1, counts.get("action_top5@4s", 0)),
                100.0 * totals.get("action_top5@6s", 0.0) / max(1, counts.get("action_top5@6s", 0)),
                float(batch["context_sec"][0]),
            )
        if save_every > 0 and save_fn is not None and ((it + 1) % save_every == 0):
            phase = "train" if train else "val"
            save_fn(
                step=it + 1,
                phase=phase,
                metric_state={
                    "totals": dict(totals),
                    "counts": dict(counts),
                    "loss_meter": loss_meter,
                    "n_steps": n_steps,
                },
            )
            logger.info("Periodic checkpoint at %s step=%d", phase, it + 1)

    metrics = {k: (totals[k] / max(1, counts[k])) for k in totals}
    metrics["loss"] = loss_meter / max(1, n_steps)
    metrics["seconds"] = time.time() - t0
    metrics["last_step"] = int(last_it + 1)
    metrics["stopped_early"] = bool(stopped_early)
    metrics["_metric_state"] = {
        "totals": dict(totals),
        "counts": dict(counts),
        "loss_meter": loss_meter,
        "n_steps": n_steps,
    }
    return metrics


def default_gaze_cfg(
    gaze_root: str,
    extract_root: str,
    sync_root: str,
    slam_root: str,
    mapping_json: str,
    img_size: int,
) -> dict:
    return {
        "mode": "concat_plus_cross_attn",
        "gaze_root": gaze_root,
        "extract_root": extract_root,
        "sync_root": sync_root,
        "binary_radius_px": 64,
        "crop_size": img_size,
        "pose_map": {"patch_height": 128, "patch_width": 9, "layout": "topleft", "normalize": "none"},
        "pose": {
            "enabled": True,
            "slam_root": slam_root,
            "mapping_json": mapping_json,
            "feature_set": "pose_6d",
            "interframe_k_max": 128,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--encoder-lora", type=Path, required=True)
    ap.add_argument("--predictor-lora", type=Path, required=True)
    ap.add_argument("--adapter-ckpt", type=Path, required=True)
    ap.add_argument("--fusion-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gaze-root", type=str, required=True)
    ap.add_argument("--gaze-extract-root", type=str, required=True)
    ap.add_argument("--gaze-sync-root", type=str, required=True)
    ap.add_argument("--pose-slam-root", type=str, required=True)
    ap.add_argument("--pose-mapping-json", type=str, required=True)
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--loss-weights", type=str, default="1.0,0.7,0.5")
    ap.add_argument("--primary-horizon-sec", type=float, default=2.0)
    ap.add_argument("--anticipation-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--keep-count", type=int, default=4096)
    ap.add_argument(
        "--prune-mode",
        type=str,
        default="encoder_attn",
        choices=("encoder_attn", "postfuse_recency"),
        help="Video-token prune scorer after fusion. postfuse_recency = P0 A/B.",
    )
    ap.add_argument(
        "--recency-strength",
        type=float,
        default=1.0,
        help="For prune-mode=postfuse_recency: scores *= (1 + strength * recency).",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--batch-size-by-frames",
        type=str,
        default="",
        help="Optional adaptive bs map, e.g. '32:4,48:2,64:2,80:1'. Empty = fixed --batch-size.",
    )
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--fusion-lr-mult", type=float, default=2.0)
    ap.add_argument("--adapter-lr-mult", type=float, default=0.25)
    ap.add_argument(
        "--head-lr-mult",
        type=float,
        default=1.0,
        help="LR multiplier for MTP heads (use <1 when warm-starting from video MTP).",
    )
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--freeze-adapter", action="store_true")
    ap.add_argument(
        "--freeze-fusion",
        action="store_true",
        help="Freeze fusion+IMU (keep 43.92%% features); train heads + predictor LoRA only.",
    )
    ap.add_argument(
        "--tick-cache-dir",
        type=Path,
        default=None,
        help="Disk cache for (clip, aux_map, imu) per stream tick. Speeds epoch 2+.",
    )
    ap.add_argument(
        "--mtp-warm-ckpt",
        type=Path,
        default=None,
        help="Optional ckpt with mtp_classifier (e.g. video-only stream best) to warm MTP heads.",
    )
    ap.add_argument(
        "--init-from-ckpt",
        type=Path,
        default=None,
        help="Load model+mtp_classifier weights (no optimizer). Used to continue from a prior CA stream run.",
    )
    ap.add_argument(
        "--carry-best",
        action="store_true",
        help="With --init-from-ckpt, seed early-stop best from that ckpt's best metric.",
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after N vals without primary@2s improvement (0=disabled).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [float(x) for x in args.horizons_sec.split(",")]
    weights = [float(x) for x in args.loss_weights.split(",")]
    primary_h = float(args.primary_horizon_sec)
    primary_idx = horizons.index(primary_h) if primary_h in horizons else 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = args.out_dir / "TRAINING_DONE"
    if done_flag.is_file() and not args.val_only:
        logger.info("TRAINING_DONE present; exiting")
        return

    gaze_cfg = default_gaze_cfg(
        args.gaze_root,
        args.gaze_extract_root,
        args.gaze_sync_root,
        args.pose_slam_root,
        args.pose_mapping_json,
        args.img_size,
    )

    verb_map, noun_map, action_map = base.load_action_maps(args.train_csv)
    logger.info("vocab verbs=%d nouns=%d actions=%d", len(verb_map), len(noun_map), len(action_map))

    tick_cache = Path(args.tick_cache_dir) if args.tick_cache_dir else None
    if tick_cache is not None:
        tick_cache.mkdir(parents=True, exist_ok=True)
        logger.info("Stream-tick cache: %s", tick_cache)

    train_ds = StreamMTPConcatCADataset(
        args.train_csv, args.video_root, args.img_size, gaze_cfg, tick_cache_dir=tick_cache
    )
    val_ds = StreamMTPConcatCADataset(
        args.val_csv, args.video_root, args.img_size, gaze_cfg, tick_cache_dir=tick_cache
    )

    bs_by_frames: dict[int, int] = {}
    if str(args.batch_size_by_frames).strip():
        for part in str(args.batch_size_by_frames).split(","):
            part = part.strip()
            if not part:
                continue
            n_f, bs = part.split(":")
            bs_by_frames[int(n_f)] = int(bs)
    if bs_by_frames:
        logger.info("Adaptive batch sizes by n_model_frames: %s (default=%d)", bs_by_frames, args.batch_size)
        train_sampler = AdaptiveContextBucketBatchSampler(
            train_ds, bs_by_frames, default_batch_size=args.batch_size, shuffle=True, seed=args.seed
        )
        val_sampler = AdaptiveContextBucketBatchSampler(
            val_ds, bs_by_frames, default_batch_size=args.batch_size, shuffle=False, seed=args.seed
        )
    else:
        train_sampler = base.ContextBucketBatchSampler(train_ds, args.batch_size, shuffle=True, seed=args.seed)
        val_sampler = base.ContextBucketBatchSampler(val_ds, args.batch_size, shuffle=False, seed=args.seed)

    loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=collate_stream_concat_ca,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kwargs)

    model = build_concat_ca_model(
        device,
        args.max_frames,
        args.fps,
        args.img_size,
        str(args.checkpoint),
        str(args.encoder_lora),
        str(args.predictor_lora),
        str(args.adapter_ckpt),
        str(args.fusion_ckpt),
        args.keep_count,
        freeze_adapter=bool(args.freeze_adapter),
        freeze_fusion=bool(args.freeze_fusion),
        reset_gate_bias=None,  # keep learned gate from 43.92%
        prune_mode=str(args.prune_mode),
        recency_strength=float(args.recency_strength),
    )
    base_enc = model.concat_ca.base_model
    classifier = AttentiveClassifier(
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        embed_dim=int(base_enc.encoder.embed_dim),
        num_heads=16,
        depth=4,
        use_activation_checkpointing=True,
    ).to(device)
    for name, p in classifier.named_parameters():
        p.requires_grad = name.startswith(("verb_classifier.", "noun_classifier.", "action_classifier."))
        if name.startswith("pooler."):
            p.requires_grad = True
    mtp_clf = CommunicatingMLPMTPClassifier(classifier, horizons_sec=horizons, comm_layers=2, comm_heads=4).to(device)

    # Optional warm-start MTP heads from video-only stream best (same vocab/CSV).
    if args.mtp_warm_ckpt is not None and Path(args.mtp_warm_ckpt).is_file():
        warm = torch.load(Path(args.mtp_warm_ckpt), map_location="cpu", weights_only=False)
        state = warm.get("mtp_classifier", warm)
        missing, unexpected = mtp_clf.load_state_dict(state, strict=False)
        logger.info(
            "Warmed MTP heads from %s (src_best=%s) missing=%d unexpected=%d",
            args.mtp_warm_ckpt,
            warm.get("best"),
            len(missing),
            len(unexpected),
        )
        del warm

    # Continue from a prior concat+CA stream ckpt (overrides mtp warm / sidecar LoRA).
    init_best = -1.0
    if args.init_from_ckpt is not None and Path(args.init_from_ckpt).is_file():
        init_ck = torch.load(Path(args.init_from_ckpt), map_location="cpu", weights_only=False)
        m_miss, m_unexp = model.load_state_dict(init_ck["model"], strict=False)
        h_miss, h_unexp = mtp_clf.load_state_dict(init_ck["mtp_classifier"], strict=False)
        init_best = float(init_ck.get("best", -1.0))
        logger.info(
            "Init from %s best=%.4f model(missing=%d unexpected=%d) mtp(missing=%d unexpected=%d)",
            args.init_from_ckpt,
            init_best,
            len(m_miss),
            len(m_unexp),
            len(h_miss),
            len(h_unexp),
        )
        del init_ck

    # Param groups: heads @ lr*head_lr_mult, fusion @ fusion_lr_mult, adapter @ adapter_lr_mult, predictor LoRA @ lr
    head_params = [p for p in mtp_clf.parameters() if p.requires_grad]
    fusion_params = [p for p in trainable_tri_modal_fusion_params(model.concat_ca) if p.requires_grad]
    adapter_params = [p for p in model.concat_ca.input_adapter.parameters() if p.requires_grad]
    other_params = []
    named = {id(p) for p in head_params + fusion_params + adapter_params}
    for p in model.parameters():
        if p.requires_grad and id(p) not in named:
            other_params.append(p)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr * float(args.head_lr_mult)})
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr})
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": args.lr * args.fusion_lr_mult})
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": args.lr * args.adapter_lr_mult})
    n_train = sum(p.numel() for g in param_groups for p in g["params"])
    logger.info(
        "Trainable params: total=%d heads=%d fusion=%d adapter=%d other(pred-LoRA)=%d "
        "freeze_fusion=%s freeze_adapter=%s head_lr_mult=%.3f fusion_lr_mult=%.3f",
        n_train,
        sum(p.numel() for p in head_params),
        sum(p.numel() for p in fusion_params),
        sum(p.numel() for p in adapter_params),
        sum(p.numel() for p in other_params),
        bool(args.freeze_fusion),
        bool(args.freeze_adapter),
        float(args.head_lr_mult),
        float(args.fusion_lr_mult),
    )
    if not param_groups:
        raise RuntimeError("No trainable parameters — check freeze flags / LoRA load")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best = -1.0
    if bool(args.carry_best) and init_best >= 0:
        best = float(init_best)
        logger.info("Carried best=%.4f from --init-from-ckpt", best)
    bad_vals = 0
    history = []
    start_epoch = 0
    start_step = 0
    resume_phase = "train"
    resume_metric_state = None
    latest = args.out_dir / "latest.pt"
    if latest.is_file():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        mtp_clf.load_state_dict(ck["mtp_classifier"], strict=False)
        if ck.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(ck["optimizer"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Optimizer restore failed: %s", exc)
        if ck.get("scaler") is not None:
            try:
                scaler.load_state_dict(ck["scaler"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Scaler restore failed: %s", exc)
        best = float(ck.get("best", -1.0))
        history = list(ck.get("history") or [])
        # Reconstruct early-stop streak across util-kill resubmits.
        bad_vals = 0
        if history and best >= 0:
            for h in reversed(history):
                p = float((h.get("val") or {}).get("primary_action_top5", -1.0))
                if p + 1e-12 < best:
                    bad_vals += 1
                else:
                    break
        if "step" in ck:
            start_epoch = int(ck.get("epoch", 0))
            start_step = int(ck.get("step", 0))
            resume_phase = str(ck.get("phase", "train"))
            resume_metric_state = ck.get("metric_state")
        else:
            start_epoch = int(ck.get("epoch", 0)) + 1
        logger.info(
            "Resumed epoch=%d step=%d phase=%s best=%.4f", start_epoch, start_step, resume_phase, best
        )

    stop_flag = {"stop": False}
    _ckpt_ctx = {"epoch": start_epoch, "step": start_step, "phase": resume_phase}

    def _periodic_save(step: int, phase: str, metric_state=None):
        _ckpt_ctx["step"] = int(step)
        _ckpt_ctx["phase"] = str(phase)
        base.save_checkpoint(
            latest,
            epoch=_ckpt_ctx["epoch"],
            step=step,
            model=model,
            mtp_clf=mtp_clf,
            optimizer=optimizer,
            scaler=scaler,
            best=best,
            horizons=horizons,
            verb_map=verb_map,
            noun_map=noun_map,
            action_map=action_map,
            history=history,
            phase=phase,
            metric_state=metric_state,
        )

    def _on_signal(signum, _frame):
        logger.warning("Caught signal %s — saving latest", signum)
        stop_flag["stop"] = True
        try:
            _periodic_save(step=max(0, int(_ckpt_ctx["step"])), phase=str(_ckpt_ctx["phase"]))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Emergency save failed: %s", exc)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)

    if args.val_only:
        metrics = run_epoch(
            model,
            mtp_clf,
            val_loader,
            device,
            horizons,
            weights,
            primary_idx,
            verb_map,
            noun_map,
            action_map,
            train=False,
            anticipation_sec=args.anticipation_sec,
            stop_flag=stop_flag,
        )
        metrics_pub = {k: v for k, v in metrics.items() if not str(k).startswith("_")}
        logger.info("VAL_ONLY: %s", json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in metrics_pub.items()}))
        (args.out_dir / "val_only_metrics.json").write_text(json.dumps(metrics_pub, indent=2), encoding="utf-8")
        return

    for epoch in range(start_epoch, args.epochs):
        _ckpt_ctx["epoch"] = epoch
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        epoch_start_step = start_step if (epoch == start_epoch and resume_phase == "train") else 0
        val_start_step = start_step if (epoch == start_epoch and resume_phase == "val") else 0
        skip_train = bool(epoch == start_epoch and resume_phase == "val")
        if skip_train:
            logger.info("Skipping train epoch=%d (resume val step=%d)", epoch, val_start_step)
            tr = {"loss": float("nan"), "stopped_early": False, "last_step": 0}
        else:
            _ckpt_ctx["phase"] = "train"
            train_sampler.set_start_batch(epoch_start_step)
            tr = run_epoch(
                model,
                mtp_clf,
                train_loader,
                device,
                horizons,
                weights,
                primary_idx,
                verb_map,
                noun_map,
                action_map,
                optimizer=optimizer,
                scaler=scaler,
                train=True,
                anticipation_sec=args.anticipation_sec,
                start_step=epoch_start_step,
                save_every=int(args.save_every),
                save_fn=_periodic_save,
                stop_flag=stop_flag,
                metric_state=(resume_metric_state if epoch_start_step > 0 else None),
            )
            train_sampler.set_start_batch(0)
            if tr.get("stopped_early"):
                _periodic_save(step=int(tr["last_step"]), phase="train", metric_state=tr.get("_metric_state"))
                return

        if not skip_train:
            _periodic_save(step=0, phase="val", metric_state=None)
        _ckpt_ctx["phase"] = "val"
        val_sampler.set_start_batch(val_start_step)
        va = run_epoch(
            model,
            mtp_clf,
            val_loader,
            device,
            horizons,
            weights,
            primary_idx,
            verb_map,
            noun_map,
            action_map,
            train=False,
            anticipation_sec=args.anticipation_sec,
            start_step=val_start_step,
            save_every=int(args.save_every),
            save_fn=_periodic_save,
            stop_flag=stop_flag,
            metric_state=(resume_metric_state if (skip_train and val_start_step > 0) else None),
        )
        val_sampler.set_start_batch(0)
        if va.get("stopped_early"):
            _periodic_save(step=int(va["last_step"]), phase="val", metric_state=va.get("_metric_state"))
            return

        primary = float(va.get("primary_action_top5", 0.0))
        logger.info(
            "epoch %d train_loss=%.4f val @2s=%.2f%% @4s=%.2f%% @6s=%.2f%%",
            epoch,
            tr["loss"],
            100.0 * va.get("action_top5@2s", primary),
            100.0 * va.get("action_top5@4s", 0.0),
            100.0 * va.get("action_top5@6s", 0.0),
        )
        tr_pub = {k: v for k, v in tr.items() if not str(k).startswith("_")}
        va_pub = {k: v for k, v in va.items() if not str(k).startswith("_")}
        history.append({"epoch": epoch, "train": tr_pub, "val": va_pub})
        base.save_checkpoint(
            latest,
            epoch=epoch + 1,
            step=0,
            model=model,
            mtp_clf=mtp_clf,
            optimizer=optimizer,
            scaler=scaler,
            best=best,
            horizons=horizons,
            verb_map=verb_map,
            noun_map=noun_map,
            action_map=action_map,
            history=history,
            phase="train",
            metric_state=None,
        )
        if primary > best:
            best = primary
            bad_vals = 0
            base.save_checkpoint(
                args.out_dir / "best.pt",
                epoch=epoch + 1,
                step=0,
                model=model,
                mtp_clf=mtp_clf,
                optimizer=optimizer,
                scaler=scaler,
                best=best,
                horizons=horizons,
                verb_map=verb_map,
                noun_map=noun_map,
                action_map=action_map,
                history=history,
                phase="train",
                metric_state=None,
            )
            logger.info("New best primary@2s=%.4f", best)
        else:
            bad_vals += 1
            logger.info(
                "No improvement primary@2s=%.4f best=%.4f bad_vals=%d/%d",
                primary,
                best,
                bad_vals,
                int(args.early_stop_patience),
            )
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        start_step = 0
        resume_phase = "train"
        resume_metric_state = None
        if int(args.early_stop_patience) > 0 and bad_vals >= int(args.early_stop_patience):
            logger.info(
                "Early stop: patience=%d reached (best primary@2s=%.4f)",
                int(args.early_stop_patience),
                best,
            )
            break

    done_flag.write_text(f"finished epochs={args.epochs} best={best} bad_vals={bad_vals}\n", encoding="utf-8")
    logger.info("Done. best primary@2s=%.4f", best)


if __name__ == "__main__":
    main()
