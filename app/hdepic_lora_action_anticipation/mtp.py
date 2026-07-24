"""Multi-Time Prediction (MTP) for action anticipation.

Optional multi-horizon action heads gated behind ``experiment.lora.mtp.enabled``.
When disabled/absent the single-horizon path is unchanged.

Two head backends (``mtp.head_type``):

* ``cascaded`` (legacy): predictor multi-horizon tokens + short→long cascade.
* ``communicating_mlp`` (default for new 2/4/6 recipes): encode once, then three
  (or N) private MLPs that **mutually communicate** via residual self-attention
  over horizons — no RNN / AR rollout.

``mtp.backbone_mode``:

* ``multi_predict`` — wrap encoder with ``MultiHorizonAnticipativeWrapper``.
* ``shared`` — one standard anticipative forward; heads do multi-horizon.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Sequence

import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation.data_prefetch import DataLoaderPrefetcher

logger = logging.getLogger(__name__)


def parse_mtp_cfg(lora_cfg: dict | None) -> dict | None:
    raw = dict((lora_cfg or {}).get("mtp", {}) or {})
    if not bool(raw.get("enabled", False)):
        return None
    horizons = raw.get("horizons_sec", [2.0, 4.0, 6.0])
    horizons = [float(h) for h in horizons]
    if len(horizons) < 2:
        raise ValueError(f"mtp.horizons_sec needs >=2 horizons, got {horizons}")
    if not all(h > 0 for h in horizons):
        raise ValueError(f"mtp.horizons_sec must be positive, got {horizons}")
    weights = raw.get("loss_weights")
    if weights is None:
        # Shortest horizon dominates (compat with existing val-action-top5).
        weights = [1.0] + [max(0.2, 0.85 ** i) for i in range(1, len(horizons))]
    weights = [float(w) for w in weights]
    if len(weights) != len(horizons):
        raise ValueError(f"mtp.loss_weights length {len(weights)} != horizons {len(horizons)}")
    head_type = str(raw.get("head_type", "communicating_mlp")).lower()
    if head_type not in {"cascaded", "communicating_mlp"}:
        raise ValueError(f"mtp.head_type must be cascaded|communicating_mlp, got {head_type!r}")
    backbone_mode = str(raw.get("backbone_mode") or "").lower()
    if not backbone_mode:
        backbone_mode = "shared" if head_type == "communicating_mlp" else "multi_predict"
    if backbone_mode not in {"shared", "multi_predict"}:
        raise ValueError(f"mtp.backbone_mode must be shared|multi_predict, got {backbone_mode!r}")
    if head_type == "cascaded" and backbone_mode != "multi_predict":
        raise ValueError("mtp.head_type=cascaded requires backbone_mode=multi_predict")
    label_mode = str(raw.get("label_mode", "lookup_all" if head_type == "communicating_mlp" else "primary_from_sample")).lower()
    if label_mode not in {"lookup_all", "primary_from_sample"}:
        raise ValueError(f"mtp.label_mode must be lookup_all|primary_from_sample, got {label_mode!r}")
    return {
        "enabled": True,
        "horizons_sec": horizons,
        "loss_weights": weights,
        "primary_horizon_sec": float(raw.get("primary_horizon_sec", horizons[0])),
        "condition_mode": str(raw.get("condition_mode", "feature_token")).lower(),
        "teacher_forcing": bool(raw.get("teacher_forcing", True)),
        "head_type": head_type,
        "backbone_mode": backbone_mode,
        "label_mode": label_mode,
        "mlp_hidden": int(raw["mlp_hidden"]) if raw.get("mlp_hidden") is not None else None,
        "comm_layers": int(raw.get("comm_layers", 2)),
        "comm_heads": int(raw.get("comm_heads", 4)),
    }


def build_video_action_intervals(annotations: dict) -> dict[str, list[tuple[int, int, int, int]]]:
    """Map video_id -> sorted (start_frame, stop_frame, verb, noun)."""
    out: dict[str, list[tuple[int, int, int, int]]] = {}
    for video_id, df in annotations.items():
        intervals: list[tuple[int, int, int, int]] = []
        for row in df.itertuples(index=False):
            intervals.append(
                (
                    int(getattr(row, "start_frame")),
                    int(getattr(row, "stop_frame")),
                    int(getattr(row, "verb_class")),
                    int(getattr(row, "noun_class")),
                )
            )
        intervals.sort(key=lambda x: (x[0], x[1]))
        out[str(video_id)] = intervals
    return out


def lookup_action_at_frame(
    intervals: Sequence[tuple[int, int, int, int]],
    frame: int,
) -> tuple[int, int, bool]:
    """Return (verb, noun, valid) at ``frame``. Prefer covering segment, else next start."""
    if not intervals:
        return -1, -1, False
    for sf, ef, verb, noun in intervals:
        if sf <= frame <= ef:
            return verb, noun, True
    for sf, ef, verb, noun in intervals:
        if sf >= frame:
            return verb, noun, True
    return -1, -1, False


def _unwrap_core(model: nn.Module) -> nn.Module:
    m = model
    if hasattr(m, "module") and hasattr(m.module, "encoder"):
        m = m.module
    if hasattr(m, "base_model") and hasattr(m.base_model, "encoder"):
        m = m.base_model
    return m


class MultiHorizonAnticipativeWrapper(nn.Module):
    """Encode once, predict future tokens at each MTP horizon."""

    def __init__(self, base_model: nn.Module, horizons_sec: Sequence[float]):
        super().__init__()
        self.base_model = base_model
        self.horizons_sec = [float(h) for h in horizons_sec]
        core = _unwrap_core(base_model)
        # Alias encoder/predictor without re-registering them as children
        # (they already live under base_model).
        self.__dict__["encoder"] = getattr(core, "encoder", None)
        self.__dict__["predictor"] = getattr(core, "predictor", None)
        self.embed_dim = getattr(base_model, "embed_dim", getattr(core, "embed_dim", None))

    def forward(self, x, anticipation_times=None):
        del anticipation_times  # clip built for primary (~1s); MTP uses fixed horizons
        core = _unwrap_core(self.base_model)
        return self._forward_multi(core, x)

    @staticmethod
    def _max_direct_skip(core: nn.Module, context_tokens: int) -> int:
        """Largest absolute target start index still inside predictor pos-embed."""
        grid2 = int(core.grid_size**2)
        pred_frames = int(getattr(core.predictor, "num_frames", 64))
        tubelet = int(getattr(core, "tubelet_size", 2))
        max_pos = (pred_frames // tubelet) * grid2
        n_pred = int(grid2 * (core.num_output_frames // tubelet))
        # leave room for the predicted chunk itself
        return max(context_tokens, max_pos - n_pred)

    def _predict_direct(
        self,
        core: nn.Module,
        x_full: torch.Tensor,
        x_base: torch.Tensor,
        horizon_sec: float,
    ) -> torch.Tensor:
        B, N, _ = x_full.size()
        embed_dim = core.encoder.embed_dim
        grid2 = int(core.grid_size**2)
        n_pred = int(grid2 * (core.num_output_frames // core.tubelet_size))
        ctxt_positions = torch.arange(N, device=x_full.device).unsqueeze(0).repeat(B, 1)
        steps = int(float(horizon_sec) * core.frames_per_second / core.tubelet_size)
        skip = N + grid2 * steps
        tgt = torch.arange(n_pred, device=x_full.device).unsqueeze(0).repeat(B, 1) + skip
        pred_out = core.predictor(x_full, masks_x=ctxt_positions, masks_y=tgt)
        x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        x_pred = x_pred_full[:, :, -embed_dim:] if x_pred_full.size(-1) != embed_dim else x_pred_full
        return torch.cat([x_base, x_pred], dim=1)

    def _predict_ar_rollout(
        self,
        core: nn.Module,
        x_full: torch.Tensor,
        x_base: torch.Tensor,
        horizon_sec: float,
    ) -> torch.Tensor:
        """Coarse sliding-window AR when absolute tgt positions exceed pos-embed.

        Matches ``AutoregressiveAnticipativeWrapper._forward_sliding_window``:
        each step advances up to ``cap_slots - window_slots`` tubelet slabs
        (~4s at fps=8, tubelet=2, 32-frame context), keeping tgt indices local
        (always ``[N, N+n_pred)``).
        """
        B, N, _ = x_full.size()
        device = x_full.device
        embed_dim = core.encoder.embed_dim
        grid2 = int(core.grid_size**2)
        window_slots = max(1, N // grid2)
        pred_frames = int(getattr(core.predictor, "num_frames", 64))
        cap_slots = pred_frames // int(core.tubelet_size)
        max_adv = max(1, cap_slots - window_slots)

        total_slabs = max(1, int(round(float(horizon_sec) * core.frames_per_second / core.tubelet_size)))
        k_steps = max(1, (total_slabs + max_adv - 1) // max_adv)

        ctx = x_full
        ctx_pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        advanced = 0
        target_tokens = None
        for step_idx in range(1, k_steps + 1):
            target_slabs = int(round(total_slabs * step_idx / k_steps))
            adv = max(0, min(max_adv, target_slabs - advanced))
            if adv <= 0:
                continue
            n_pred = grid2 * adv
            tgt_pos = torch.arange(n_pred, device=device).unsqueeze(0).expand(B, -1) + N
            pred_out = core.predictor(ctx, masks_x=ctx_pos, masks_y=tgt_pos)
            pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
            pred_cls = pred_full[:, :, -embed_dim:] if pred_full.size(-1) != embed_dim else pred_full
            pred_in = pred_full if pred_full.size(-1) == ctx.size(-1) else pred_cls
            # Slide window: keep last N tokens after appending prediction.
            full = torch.cat([ctx, pred_in], dim=1)
            ctx = full[:, -N:, :]
            advanced += adv
            if advanced >= total_slabs:
                # Last predicted chunk at this horizon (trim to one output tubelet if needed).
                n_out = int(grid2 * (core.num_output_frames // core.tubelet_size))
                target_tokens = pred_cls[:, :n_out, :] if pred_cls.size(1) >= n_out else pred_cls

        if target_tokens is None:
            # Fallback: classifier sees last window slice as "future".
            target_tokens = ctx[:, : int(grid2 * (core.num_output_frames // core.tubelet_size)), :]
            if target_tokens.size(-1) != embed_dim:
                target_tokens = target_tokens[:, :, -embed_dim:]
        return torch.cat([x_base, target_tokens], dim=1)

    def _forward_multi(self, core: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
        if not hasattr(core, "encoder") or not hasattr(core, "predictor"):
            outs = []
            B = x.size(0)
            device = x.device
            for h in self.horizons_sec:
                times = torch.full((B,), float(h), device=device, dtype=torch.float32)
                outs.append(self.base_model(x, times))
            return outs

        x_full = core.encoder(x)
        B, N, D_full = x_full.size()
        embed_dim = core.encoder.embed_dim
        x_ctx = x_full[:, :, -embed_dim:] if D_full > embed_dim else x_full
        x_base = (
            torch.zeros(B, 0, embed_dim, device=x.device, dtype=x_ctx.dtype)
            if getattr(core, "no_encoder", False)
            else x_ctx
        )

        grid2 = int(core.grid_size**2)
        max_skip = self._max_direct_skip(core, N)
        outs: list[torch.Tensor] = []
        for h in self.horizons_sec:
            steps = int(float(h) * core.frames_per_second / core.tubelet_size)
            skip = N + grid2 * steps
            if skip <= max_skip:
                outs.append(self._predict_direct(core, x_full, x_base, h))
            else:
                if not getattr(MultiHorizonAnticipativeWrapper, "_logged_ar", False):
                    logger.info(
                        "MTP: horizon>=%.1fs exceeds predictor pos-embed (skip=%d > max_skip=%d); "
                        "using sliding-window AR rollout for long horizons",
                        h,
                        skip,
                        max_skip,
                    )
                    MultiHorizonAnticipativeWrapper._logged_ar = True
                outs.append(self._predict_ar_rollout(core, x_full, x_base, h))
        return outs


class CascadedMTPClassifier(nn.Module):
    """Shared AttentiveClassifier with short→long feature conditioning (legacy)."""

    def __init__(
        self,
        base_classifier: nn.Module,
        horizons_sec: Sequence[float],
        condition_mode: str = "feature_token",
    ):
        super().__init__()
        self.base = base_classifier
        self.horizons_sec = [float(h) for h in horizons_sec]
        self.condition_mode = condition_mode
        embed_dim = int(base_classifier.pooler.embed_dim) if hasattr(base_classifier.pooler, "embed_dim") else None
        if embed_dim is None:
            # Fall back to classifier linear in_features.
            embed_dim = int(base_classifier.action_classifier.in_features)
        n_cond = max(0, len(self.horizons_sec) - 1)
        self.cond_proj = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_cond)])
        # Zero-init gates → initially identical to independent multi-horizon heads.
        self.cond_gate = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(n_cond)])

    def _classify_tokens(self, tokens: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        pooled = self.base.pooler(tokens)
        if getattr(self.base, "action_only", False) or getattr(self.base, "num_verb_classes", 1) == 0:
            feat = pooled[:, 0, :]
            out = dict(action=self.base.action_classifier(feat))
            return out, feat
        feat_v, feat_n, feat_a = pooled[:, 0, :], pooled[:, 1, :], pooled[:, 2, :]
        out = dict(
            verb=self.base.verb_classifier(feat_v),
            noun=self.base.noun_classifier(feat_n),
            action=self.base.action_classifier(feat_a),
        )
        return out, feat_a

    def forward(self, tokens_by_horizon: Sequence[torch.Tensor]) -> dict[float, dict[str, torch.Tensor]]:
        if len(tokens_by_horizon) != len(self.horizons_sec):
            raise ValueError(
                f"Expected {len(self.horizons_sec)} token tensors, got {len(tokens_by_horizon)}"
            )
        outputs: dict[float, dict[str, torch.Tensor]] = {}
        prev_feats: list[torch.Tensor] = []
        for i, (h, tokens) in enumerate(zip(self.horizons_sec, tokens_by_horizon)):
            x = tokens
            if i > 0 and prev_feats:
                cond = torch.stack(prev_feats, dim=1).mean(dim=1, keepdim=True)  # [B,1,D]
                cond = self.cond_proj[i - 1](cond)
                gate = torch.sigmoid(self.cond_gate[i - 1])
                x = torch.cat([x, gate * cond], dim=1)
            out, feat = self._classify_tokens(x)
            outputs[float(h)] = out
            prev_feats.append(feat)
        return outputs


class _HorizonPrivateMLP(nn.Module):
    """Per-horizon residual MLP; zero-init last layer → identity at start."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _HorizonCommBlock(nn.Module):
    """Bidirectional communication across horizon features (self-attn + MLP)."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            # Prefer exact division; fall back to 1 head.
            num_heads = 1
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, H, D]
        h = self.norm1(z)
        a, _ = self.attn(h, h, h, need_weights=False)
        z = z + a
        z = z + self.mlp(self.norm2(z))
        return z


class CommunicatingMLPMTPClassifier(nn.Module):
    """N communicating MLPs on a shared pooled feature (no RNN / AR).

    Pool once → private residual MLP per horizon → mutual self-attn across
    horizons → shared verb/noun/action linear heads.
    Zero-init residuals → at init all horizons match the warm-started single head.
    """

    def __init__(
        self,
        base_classifier: nn.Module,
        horizons_sec: Sequence[float],
        mlp_hidden: int | None = None,
        comm_layers: int = 2,
        comm_heads: int = 4,
    ):
        super().__init__()
        self.base = base_classifier
        self.horizons_sec = [float(h) for h in horizons_sec]
        embed_dim = int(base_classifier.pooler.embed_dim) if hasattr(base_classifier.pooler, "embed_dim") else None
        if embed_dim is None:
            embed_dim = int(base_classifier.action_classifier.in_features)
        hidden = int(mlp_hidden) if mlp_hidden is not None else embed_dim
        self.horizon_mlps = nn.ModuleList(
            [_HorizonPrivateMLP(embed_dim, hidden) for _ in self.horizons_sec]
        )
        self.comm = nn.ModuleList(
            [_HorizonCommBlock(embed_dim, num_heads=comm_heads) for _ in range(max(0, int(comm_layers)))]
        )
        self.embed_dim = embed_dim

    def _pool_slots(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.base.pooler(tokens)
        if getattr(self.base, "action_only", False) or getattr(self.base, "num_verb_classes", 1) == 0:
            feat = pooled[:, 0, :]
            return feat, feat, feat
        return pooled[:, 0, :], pooled[:, 1, :], pooled[:, 2, :]

    def _communicate(self, feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        z = torch.stack([mlp(f) for mlp, f in zip(self.horizon_mlps, feats)], dim=1)
        for block in self.comm:
            z = block(z)
        return [z[:, i] for i in range(z.size(1))]

    def forward(self, tokens: torch.Tensor) -> dict[float, dict[str, torch.Tensor]]:
        if isinstance(tokens, (list, tuple)):
            raise TypeError(
                "CommunicatingMLPMTPClassifier expects a single token tensor "
                "(backbone_mode=shared); got a sequence — use CascadedMTPClassifier "
                "or set backbone_mode=shared"
            )
        feat_v, feat_n, feat_a = self._pool_slots(tokens)
        # Communicate on the action slot; verb/noun slots get the same residual
        # shift so the three classifiers stay aligned with the warm-started head.
        shared = self._communicate([feat_a for _ in self.horizons_sec])
        outputs: dict[float, dict[str, torch.Tensor]] = {}
        action_only = getattr(self.base, "action_only", False) or getattr(self.base, "num_verb_classes", 1) == 0
        for h, z_a in zip(self.horizons_sec, shared):
            # Delta from communicated action feature applied to v/n as well.
            delta = z_a - feat_a
            zv = feat_v + delta
            zn = feat_n + delta
            if action_only:
                outputs[float(h)] = dict(action=self.base.action_classifier(z_a))
            else:
                outputs[float(h)] = dict(
                    verb=self.base.verb_classifier(zv),
                    noun=self.base.noun_classifier(zn),
                    action=self.base.action_classifier(z_a),
                )
        return outputs


def _run_mtp_heads(model, classifiers, clips, anticipation, mtp_cfg: dict):
    """Forward backbone + MTP classifiers for either shared or multi_predict mode."""
    if mtp_cfg.get("backbone_mode", "multi_predict") == "shared":
        tokens = model(clips, anticipation)
    else:
        tokens = model(clips, None)
    return [c(tokens) for c in classifiers]


def _map_labels(verbs, nouns, verb_classes, noun_classes, action_classes, device):
    verb_labels, noun_labels, action_labels = [], [], []
    keep = []
    for i, (v, n) in enumerate(zip(verbs.tolist(), nouns.tolist())):
        try:
            verb_labels.append(verb_classes[int(v)])
            noun_labels.append(noun_classes[int(n)])
            action_labels.append(action_classes[(int(v), int(n))])
            keep.append(i)
        except (KeyError, TypeError, ValueError):
            continue
    if not keep:
        empty = torch.zeros(0, device=device, dtype=torch.long)
        return empty, empty, empty, keep
    verb_labels = torch.tensor(verb_labels, device=device, dtype=torch.long)
    noun_labels = torch.tensor(noun_labels, device=device, dtype=torch.long)
    action_labels = torch.tensor(action_labels, device=device, dtype=torch.long)
    return verb_labels, noun_labels, action_labels, keep


def _unpack_mtp_batch(udata, horizons: Sequence[float], device):
    clips = udata[0].to(device, non_blocking=True)
    # Default layout with MTP extras at the end:
    # video, verb, noun, anticipation_time, mtp_verbs, mtp_nouns, mtp_mask
    if len(udata) >= 7:
        anticipation = udata[3].to(device, non_blocking=True)
        mtp_verbs = udata[4].to(device, non_blocking=True)
        mtp_nouns = udata[5].to(device, non_blocking=True)
        mtp_mask = udata[6].to(device, non_blocking=True)
    else:
        # Fallback: only primary labels available — secondary horizons masked out.
        anticipation = udata[-1].to(device, non_blocking=True)
        B = clips.size(0)
        H = len(horizons)
        mtp_verbs = torch.full((B, H), -1, device=device, dtype=torch.long)
        mtp_nouns = torch.full((B, H), -1, device=device, dtype=torch.long)
        mtp_mask = torch.zeros((B, H), device=device, dtype=torch.float32)
        mtp_verbs[:, 0] = udata[1].to(device).long()
        mtp_nouns[:, 0] = udata[2].to(device).long()
        mtp_mask[:, 0] = 1.0
    return clips, anticipation, mtp_verbs, mtp_nouns, mtp_mask


def train_one_epoch_mtp(
    base_eval,
    mtp_cfg: dict,
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
):
    from app.hdepic_lora_action_anticipation.predictor_lora import (
        AverageMeter,
        _clip_optimizer_grads,
        _grad_clip_max_norm,
        keep_nonfinite_grads_enabled,
        predictor_lora_grads_finite,
        zero_predictor_lora_grads,
    )
    from app.hdepic_lora_action_anticipation.val_metrics import summarize_metric_lists

    horizons = list(mtp_cfg["horizons_sec"])
    weights = list(mtp_cfg["loss_weights"])
    primary_h = float(mtp_cfg["primary_horizon_sec"])
    if primary_h not in horizons:
        primary_h = horizons[0]
    primary_idx = horizons.index(primary_h)

    model.train(mode=True)
    for c in classifiers:
        c.train(mode=True)

    if action_is_verb_noun:
        verb_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers
        ]
        noun_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers
        ]
    action_metric_loggers = [
        base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers
    ]
    data_elapsed_time_meter = AverageMeter()

    try:
        max_train_iters = int(os.environ.get("EVAL_MAX_TRAIN_ITERS", os.environ.get("MAX_TRAIN_ITERS", "0")) or "0")
    except ValueError:
        max_train_iters = 0
    if max_train_iters > 0 and max_train_iters < ipe:
        logger.info("Limiting MTP train loop to %d/%d iters", max_train_iters, ipe)
        ipe = max_train_iters
    grad_clip = _grad_clip_max_norm()

    prefetcher = DataLoaderPrefetcher(data_loader, name="mtp-prefetch")
    verb_metrics = noun_metrics = action_metrics = None
    try:
        for itr in range(ipe):
            itr_start = time.time()
            udata, fetch_ms = prefetcher.get()
            data_elapsed_time_meter.update(float(fetch_ms))
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                clips, anticipation, mtp_verbs, mtp_nouns, mtp_mask = _unpack_mtp_batch(udata, horizons, device)
                outputs_by_head = _run_mtp_heads(model, classifiers, clips, anticipation, mtp_cfg)

            loss_terms = []
            for outputs in outputs_by_head:
                head_loss = clips.new_zeros(())
                for hi, h in enumerate(horizons):
                    valid = mtp_mask[:, hi] > 0.5
                    if not bool(valid.any()):
                        continue
                    o = outputs[float(h)]
                    if not action_is_verb_noun:
                        raise NotImplementedError("MTP currently requires verb/noun action space")
                    v_lab, n_lab, a_lab, keep = _map_labels(
                        mtp_verbs[valid, hi],
                        mtp_nouns[valid, hi],
                        verb_classes,
                        noun_classes,
                        action_classes,
                        device,
                    )
                    if not keep:
                        continue
                    valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                    step_loss = (
                        criterion(o["verb"][valid_pos], v_lab)
                        + criterion(o["noun"][valid_pos], n_lab)
                        + criterion(o["action"][valid_pos], a_lab)
                    )
                    head_loss = head_loss + weights[hi] * step_loss
                loss_terms.append(head_loss)
            total_loss = sum(loss_terms) / max(1, len(loss_terms))

            if not torch.isfinite(total_loss.detach()):
                logger.warning("Skipping MTP step itr=%d (non-finite loss)", itr)
                optimizer[0].zero_grad()
                if use_bfloat16:
                    scaler[0].update()
                continue

            if use_bfloat16:
                scaler[0].scale(total_loss).backward()
            else:
                total_loss.backward()

            pred_ok = predictor_lora_grads_finite(model)
            if not pred_ok:
                if keep_nonfinite_grads_enabled():
                    logger.warning("Keeping non-finite predictor grads at MTP itr=%d", itr)
                else:
                    logger.warning("Discarding non-finite predictor grads at MTP itr=%d", itr)
                    zero_predictor_lora_grads(model)
            clip_ok = _clip_optimizer_grads(optimizer, scaler, use_bfloat16, grad_clip, itr)
            if not clip_ok:
                if use_bfloat16:
                    scaler[0].update()
                continue
            if use_bfloat16:
                scaler[0].step(optimizer[0])
                scaler[0].update()
            else:
                optimizer[0].step()
            optimizer[0].zero_grad()

            with torch.no_grad():
                primary_outs = [o[float(primary_h)] for o in outputs_by_head]
                valid = mtp_mask[:, primary_idx] > 0.5
                if bool(valid.any()) and action_is_verb_noun:
                    v_lab, n_lab, a_lab, keep = _map_labels(
                        mtp_verbs[valid, primary_idx],
                        mtp_nouns[valid, primary_idx],
                        verb_classes,
                        noun_classes,
                        action_classes,
                        device,
                    )
                    if keep:
                        valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                        action_metrics = [
                            m(o["action"][valid_pos], a_lab) for o, m in zip(primary_outs, action_metric_loggers)
                        ]
                        verb_metrics = [
                            m(o["verb"][valid_pos], v_lab) for o, m in zip(primary_outs, verb_metric_loggers)
                        ]
                        noun_metrics = [
                            m(o["noun"][valid_pos], n_lab) for o, m in zip(primary_outs, noun_metric_loggers)
                        ]

            if itr % 10 == 0 or itr == ipe - 1:
                step_ms = (time.time() - itr_start) * 1000.0
                if action_metrics is not None and action_is_verb_noun:
                    logger.info(
                        "[MTP %5d] loss=%.4f primary=%.1fs acc(a/v/n)=%.1f/%.1f/%.1f "
                        "[mem=%.2e] [fetch=%.1fms] [step=%.0fms]",
                        itr,
                        float(total_loss.detach().float()),
                        primary_h,
                        max(a["accuracy"] for a in action_metrics),
                        max(v["accuracy"] for v in verb_metrics),
                        max(n["accuracy"] for n in noun_metrics),
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                        data_elapsed_time_meter.avg,
                        step_ms,
                    )
    finally:
        prefetcher.close()

    return summarize_metric_lists(
        action_metrics or [{"accuracy": 0.0, "recall": 0.0, "top1_accuracy": 0.0, "top5_accuracy": 0.0}],
        verb_metrics if action_is_verb_noun else None,
        noun_metrics if action_is_verb_noun else None,
    )


@torch.no_grad()
def validate_mtp(
    base_eval,
    mtp_cfg: dict,
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    data_loader,
    use_bfloat16,
    valid_nouns,
    valid_verbs,
    valid_actions,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
    **kwargs,
):
    del valid_nouns, valid_verbs, valid_actions, criterion, kwargs
    from app.hdepic_lora_action_anticipation.val_metrics import summarize_metric_lists

    horizons = list(mtp_cfg["horizons_sec"])
    primary_h = float(mtp_cfg["primary_horizon_sec"])
    if primary_h not in horizons:
        primary_h = horizons[0]
    primary_idx = horizons.index(primary_h)

    logger.info("Running MTP val (primary=%.1fs, horizons=%s)...", primary_h, horizons)
    for c in classifiers:
        c.train(mode=False)
    model.train(mode=False)

    if action_is_verb_noun:
        verb_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers
        ]
        noun_metric_loggers = [
            base_eval.ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers
        ]
    action_metric_loggers = [
        base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers
    ]
    # Per-horizon action top5 trackers (head 0 only) for logging.
    horizon_loggers = {
        float(h): base_eval.ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for h in horizons
    }

    _loader = iter(data_loader)
    action_metrics = verb_metrics = noun_metrics = None
    horizon_last: dict[float, dict] = {}
    for itr in range(ipe):
        try:
            udata = next(_loader)
        except Exception:
            _loader = iter(data_loader)
            udata = next(_loader)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            clips, anticipation, mtp_verbs, mtp_nouns, mtp_mask = _unpack_mtp_batch(udata, horizons, device)
            outputs_by_head = _run_mtp_heads(model, classifiers, clips, anticipation, mtp_cfg)

        primary_outs = [o[float(primary_h)] for o in outputs_by_head]
        valid = mtp_mask[:, primary_idx] > 0.5
        if bool(valid.any()) and action_is_verb_noun:
            v_lab, n_lab, a_lab, keep = _map_labels(
                mtp_verbs[valid, primary_idx],
                mtp_nouns[valid, primary_idx],
                verb_classes,
                noun_classes,
                action_classes,
                device,
            )
            if keep:
                valid_pos = valid.nonzero(as_tuple=False).view(-1)[keep]
                action_metrics = [
                    m(o["action"][valid_pos], a_lab) for o, m in zip(primary_outs, action_metric_loggers)
                ]
                verb_metrics = [
                    m(o["verb"][valid_pos], v_lab) for o, m in zip(primary_outs, verb_metric_loggers)
                ]
                noun_metrics = [
                    m(o["noun"][valid_pos], n_lab) for o, m in zip(primary_outs, noun_metric_loggers)
                ]

        # Extra horizon diagnostics on head-0.
        o0 = outputs_by_head[0]
        for hi, h in enumerate(horizons):
            vmask = mtp_mask[:, hi] > 0.5
            if not bool(vmask.any()):
                continue
            _, _, a_lab, keep = _map_labels(
                mtp_verbs[vmask, hi],
                mtp_nouns[vmask, hi],
                verb_classes,
                noun_classes,
                action_classes,
                device,
            )
            if not keep:
                continue
            valid_pos = vmask.nonzero(as_tuple=False).view(-1)[keep]
            horizon_last[float(h)] = horizon_loggers[float(h)](o0[float(h)]["action"][valid_pos], a_lab)

    for h, stats in horizon_last.items():
        logger.info(
            "MTP val horizon=%.1fs action_top5=%.2f action_top3=%.2f",
            h,
            float(stats.get("top5_accuracy", stats.get("accuracy", 0.0))),
            float(stats.get("accuracy", 0.0)),
        )

    ret = summarize_metric_lists(
        action_metrics or [{"accuracy": 0.0, "recall": 0.0, "top1_accuracy": 0.0, "top5_accuracy": 0.0}],
        verb_metrics if action_is_verb_noun else None,
        noun_metrics if action_is_verb_noun else None,
    )
    ret["mtp"] = {
        f"mtp_action_top5@{h:g}s": float(s.get("top5_accuracy", s.get("accuracy", 0.0)))
        for h, s in horizon_last.items()
    }
    return ret


def enable_mtp(base_eval, args_eval: dict, mtp_cfg: dict, lora_cfg: dict):
    """Patch init_module / init_classifier / train / validate / dataloader for MTP."""
    horizons = list(mtp_cfg["horizons_sec"])
    head_type = mtp_cfg.get("head_type", "communicating_mlp")
    backbone_mode = mtp_cfg.get("backbone_mode", "shared")
    label_mode = mtp_cfg.get("label_mode", "lookup_all")
    logger.info(
        "Enabling MTP: head=%s backbone=%s label=%s horizons=%s loss_weights=%s primary=%.2fs",
        head_type,
        backbone_mode,
        label_mode,
        horizons,
        mtp_cfg["loss_weights"],
        mtp_cfg["primary_horizon_sec"],
    )

    # --- model wrapper (optional multi-horizon predictor path) ---
    inner_init_module = base_eval.init_module

    def init_module_mtp(*args, **kwargs):
        model = inner_init_module(*args, **kwargs)
        if backbone_mode == "multi_predict":
            wrapped = MultiHorizonAnticipativeWrapper(model, horizons_sec=horizons)
            if getattr(model, "embed_dim", None) is not None:
                wrapped.embed_dim = model.embed_dim
            base_eval._predictor_lora_model = wrapped
            if getattr(base_eval, "_encoder_lora_model", None) is not None:
                base_eval._encoder_lora_model = wrapped
            logger.info("Wrapped model with MultiHorizonAnticipativeWrapper horizons=%s", horizons)
            return wrapped
        # shared: keep the standard anticipative model (encode once).
        base_eval._predictor_lora_model = model
        logger.info("MTP shared backbone: no MultiHorizon wrapper (one anticipative forward)")
        return model

    base_eval.init_module = init_module_mtp

    # --- classifier ---
    inner_init_classifier = base_eval.init_classifier

    def init_classifier_mtp(*args, **kwargs):
        classifiers = inner_init_classifier(*args, **kwargs)
        wrapped = []
        for c in classifiers:
            if head_type == "communicating_mlp":
                mtp_c = CommunicatingMLPMTPClassifier(
                    c,
                    horizons_sec=horizons,
                    mlp_hidden=mtp_cfg.get("mlp_hidden"),
                    comm_layers=int(mtp_cfg.get("comm_layers", 2)),
                    comm_heads=int(mtp_cfg.get("comm_heads", 4)),
                ).to(next(c.parameters()).device)
                extra_modules = list(mtp_c.horizon_mlps) + list(mtp_c.comm)
                for mod in extra_modules:
                    for p in mod.parameters():
                        p.requires_grad = True
                extra_trainable = sum(
                    p.numel() for mod in extra_modules for p in mod.parameters() if p.requires_grad
                )
                head_name = "CommunicatingMLPMTPClassifier"
            else:
                mtp_c = CascadedMTPClassifier(
                    c,
                    horizons_sec=horizons,
                    condition_mode=mtp_cfg.get("condition_mode", "feature_token"),
                ).to(next(c.parameters()).device)
                for p in mtp_c.cond_proj.parameters():
                    p.requires_grad = True
                for p in mtp_c.cond_gate:
                    p.requires_grad = True
                extra_trainable = sum(p.numel() for p in mtp_c.cond_proj.parameters() if p.requires_grad) + sum(
                    p.numel() for p in mtp_c.cond_gate if p.requires_grad
                )
                head_name = "CascadedMTPClassifier"
            if bool(lora_cfg.get("freeze_pooler", False)):
                for name, param in mtp_c.base.named_parameters():
                    if name.startswith(("verb_classifier.", "noun_classifier.", "action_classifier.")):
                        param.requires_grad = bool(lora_cfg.get("train_heads", True))
                    elif name.startswith("pooler."):
                        param.requires_grad = False
            wrapped.append(mtp_c)
            logger.info(
                "Wrapped classifier as %s; trainable_extra=%d trainable_total=%d",
                head_name,
                extra_trainable,
                sum(p.numel() for p in mtp_c.parameters() if p.requires_grad),
            )
        return wrapped

    base_eval.init_classifier = init_classifier_mtp

    # --- train / val ---
    def train_one_epoch(**kwargs):
        return train_one_epoch_mtp(base_eval, mtp_cfg, **kwargs)

    def validate(**kwargs):
        return validate_mtp(base_eval, mtp_cfg, **kwargs)

    base_eval.train_one_epoch = train_one_epoch
    base_eval.validate = validate

    # Refresh clip-balanced dataloader hooks (closure captures the pre-MTP make_*).
    import evals.action_anticipation_frozen.dataloader as dl
    import evals.action_anticipation_frozen.epickitchens as ek

    prev_make = ek.make_webvid

    def _make_with_mtp(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["mtp_horizons_sec"] = horizons
        kwargs["mtp_label_mode"] = label_mode
        annotations_path = kwargs.get("annotations_path")
        if annotations_path is None and len(args) > 1:
            annotations_path = args[1]
        if isinstance(annotations_path, tuple) and len(annotations_path) == 2:
            kwargs["mtp_intervals"] = build_video_action_intervals(annotations_path[1])
        return prev_make(*args, **kwargs)

    ek.make_webvid = _make_with_mtp
    dl.ek100_make_webvid = _make_with_mtp
    logger.info("Patched clip-balanced dataloader for MTP multi-horizon labels (label_mode=%s)", label_mode)
    return mtp_cfg
