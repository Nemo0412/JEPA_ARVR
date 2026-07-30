#!/usr/bin/env python3
"""RU-LSTM baseline on HD-EPIC P01 under the Streaming Video protocol.

Protocol (same as jepa_yifan ``make_hdepic_stream_half_split.py``):
  - Temporal half-split per video (1st half train / 2nd half val)
  - Growing context 4→6→8→10s from half origin, then slide 10s; tick every 2s
  - Predict actions at +2 / +4 / +6 s

Uses official Rolling-Unrolling LSTM (RGB branch) from fpv-iplab/rulstm, with a
causal streaming adaptation: Rolling LSTM over observed context only; Unrolling
LSTM rolls forward ``horizon/alpha`` steps by repeating the last observed feature.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

RULSTM_ROOT = Path(__file__).resolve().parents[1] / "rulstm" / "RULSTM"
sys.path.insert(0, str(RULSTM_ROOT))
from models import OpenLSTM  # noqa: E402


IMAGENET_MEAN = None  # features already extracted


def _make_head(hidden: int, num_class: int, dropout: float, mlp_head: bool) -> nn.Module:
    if mlp_head:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_class),
        )
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_class))


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.drop(F.gelu(self.fc1(h))))
        return x + self.drop(h)


def _init_lstm_(lstm: nn.Module) -> None:
    """Orthogonal recurrent weights + forget-gate bias = 1 (easier training)."""
    for name, p in lstm.named_parameters():
        if "weight_hh" in name:
            for i in range(0, p.size(0), p.size(1)):
                nn.init.orthogonal_(p[i : i + p.size(1)])
        elif "weight_ih" in name:
            nn.init.xavier_uniform_(p)
        elif "bias" in name:
            nn.init.zeros_(p)
            # forget gate is the 2nd block of 4 in PyTorch LSTM biases
            n = p.size(0)
            p.data[n // 4 : n // 2].fill_(1.0)


class StreamingRULSTM(nn.Module):
    """Causal multi-horizon RU-LSTM.

    Capacity knobs (to match V-JEPA ViT-L ~304M):
      - ``hidden`` / ``depth`` on Rolling + Unrolling LSTMs
      - optional ``input_proj``  feat_in → hidden
      - optional ``mlp_head``   hidden → hidden → num_class
      - optional ``trunk_layers`` residual MLP blocks after unrolling (trainable capacity
        without stacking deep LSTMs — deep LSTMs were hard to optimize on this data)

    Default (hidden=1024, depth=1) ≈ paper RGB branch (~18M).
    v1 (failed): hidden=2048, depth=4 ≈290M — underfit.
    v2 (optimized): hidden=4096, depth=1, LayerNorm, better init ≈328M.
    """

    def __init__(
        self,
        num_verb: int,
        num_noun: int,
        num_action: int,
        feat_in: int = 1024,
        hidden: int = 1024,
        depth: int = 1,
        dropout: float = 0.8,
        alpha: float = 0.25,
        horizons_sec: tuple[float, ...] = (2.0, 4.0, 6.0),
        input_proj: bool = False,
        mlp_head: bool = False,
        trunk_layers: int = 0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.horizons_sec = tuple(float(h) for h in horizons_sec)
        self.feat_in = feat_in
        self.hidden = hidden
        self.depth = int(depth)
        self.dropout = nn.Dropout(dropout)
        self.feat_norm = nn.LayerNorm(feat_in) if use_layernorm else nn.Identity()
        self.input_proj = (
            nn.Sequential(nn.Linear(feat_in, hidden), nn.GELU(), nn.Dropout(dropout))
            if input_proj
            else nn.Identity()
        )
        lstm_in = hidden if input_proj else feat_in
        inter_drop = dropout if self.depth > 1 else 0.0
        self.rolling_lstm = OpenLSTM(lstm_in, hidden, num_layers=self.depth, dropout=inter_drop)
        self.unrolling_lstm = nn.LSTM(
            lstm_in, hidden, num_layers=self.depth, dropout=inter_drop
        )
        _init_lstm_(self.rolling_lstm.lstm)
        _init_lstm_(self.unrolling_lstm)
        self.trunk = nn.Sequential(
            *[ResidualMLPBlock(hidden, dropout) for _ in range(int(trunk_layers))]
        )
        self.out_norm = nn.LayerNorm(hidden) if use_layernorm else nn.Identity()
        self.verb_clf = _make_head(hidden, num_verb, dropout, mlp_head)
        self.noun_clf = _make_head(hidden, num_noun, dropout, mlp_head)
        self.action_clf = _make_head(hidden, num_action, dropout, mlp_head)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": int(total),
            "trainable": int(trainable),
            "total_m": total / 1e6,
            "trainable_m": trainable / 1e6,
        }

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        """
        feats: [B, T, D] zero-padded context features ending at the tick
        lengths: [B] number of valid timesteps
        returns dict horizon -> {verb,noun,action} logits [B, C]
        """
        B, T, D = feats.shape
        x = self.input_proj(self.feat_norm(feats))  # [B, T, lstm_in]
        x_t = x.permute(1, 0, 2)
        h_seq, c_seq = self.rolling_lstm(self.dropout(x_t))  # [T, L, B, H]

        idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(B, device=feats.device)
        hid = h_seq.permute(2, 0, 1, 3)[batch_idx, idx].permute(1, 0, 2).contiguous()
        cel = c_seq.permute(2, 0, 1, 3)[batch_idx, idx].permute(1, 0, 2).contiguous()
        last_feat = x[batch_idx, idx]

        out = {}
        for h in self.horizons_sec:
            n_steps = max(1, int(round(h / self.alpha)))
            ins = last_feat.unsqueeze(0).expand(n_steps, B, last_feat.shape[-1]).contiguous()
            h_t, _ = self.unrolling_lstm(self.dropout(ins), (hid, cel))
            z = self.out_norm(self.trunk(h_t[-1]))
            out[h] = {
                "verb": self.verb_clf(z),
                "noun": self.noun_clf(z),
                "action": self.action_clf(z),
            }
        return out


def build_vocab(csvs: list[Path]):
    verbs, nouns, actions = set(), set(), set()
    for p in csvs:
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            vs = [int(x) for x in str(r.mtp_verbs).split(",")]
            ns = [int(x) for x in str(r.mtp_nouns).split(",")]
            ms = [float(x) for x in str(r.mtp_mask).split(",")]
            for v, n, m in zip(vs, ns, ms):
                if m > 0:
                    verbs.add(v)
                    nouns.add(n)
                    actions.add((v, n))
    verb_list = sorted(verbs)
    noun_list = sorted(nouns)
    action_list = sorted(actions)
    verb_map = {v: i for i, v in enumerate(verb_list)}
    noun_map = {n: i for i, n in enumerate(noun_list)}
    action_map = {a: i for i, a in enumerate(action_list)}
    return verb_map, noun_map, action_map


class StreamFeatureDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        feat_dir: Path,
        verb_map: dict,
        noun_map: dict,
        action_map: dict,
        alpha: float = 0.25,
        max_context_sec: float = 10.0,
        horizons_sec: tuple[float, ...] = (2.0, 4.0, 6.0),
    ):
        self.df = pd.read_csv(csv_path)
        self.feat_dir = Path(feat_dir)
        self.verb_map = verb_map
        self.noun_map = noun_map
        self.action_map = action_map
        self.alpha = float(alpha)
        self.max_len = int(round(max_context_sec / alpha))
        self.horizons_sec = tuple(float(h) for h in horizons_sec)
        self._cache: dict[str, tuple[np.ndarray, dict]] = {}

    def _load(self, video_id: str):
        if video_id not in self._cache:
            feat = np.load(self.feat_dir / f"{video_id}.npy")
            meta = json.loads((self.feat_dir / f"{video_id}.json").read_text())
            self._cache[video_id] = (feat, meta)
        return self._cache[video_id]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        r = self.df.iloc[index]
        video_id = str(r.video_id)
        feat, meta = self._load(video_id)
        frame_indices = np.asarray(meta["frame_indices"], dtype=np.int64)
        start_frame = int(r.start_frame)
        tick_frame = int(r.tick_frame)

        # Feature rows whose native frame is inside [start_frame, tick_frame].
        mask = (frame_indices >= start_frame) & (frame_indices <= tick_frame)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            # fallback: nearest feature at/before tick
            idxs = np.array([int(np.searchsorted(frame_indices, tick_frame, side="right") - 1)])
            idxs = np.clip(idxs, 0, len(frame_indices) - 1)

        seq = feat[idxs]  # [L, D]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len :]
        length = len(seq)
        padded = np.zeros((self.max_len, seq.shape[1]), dtype=np.float32)
        padded[:length] = seq

        verbs = [int(x) for x in str(r.mtp_verbs).split(",")]
        nouns = [int(x) for x in str(r.mtp_nouns).split(",")]
        masks = [float(x) for x in str(r.mtp_mask).split(",")]
        assert len(verbs) == len(self.horizons_sec)

        v_lab = np.full(len(self.horizons_sec), -1, np.int64)
        n_lab = np.full(len(self.horizons_sec), -1, np.int64)
        a_lab = np.full(len(self.horizons_sec), -1, np.int64)
        m_lab = np.zeros(len(self.horizons_sec), np.float32)
        for i, (v, n, m) in enumerate(zip(verbs, nouns, masks)):
            if m <= 0:
                continue
            if v not in self.verb_map or n not in self.noun_map or (v, n) not in self.action_map:
                continue
            v_lab[i] = self.verb_map[v]
            n_lab[i] = self.noun_map[n]
            a_lab[i] = self.action_map[(v, n)]
            m_lab[i] = 1.0

        return {
            "feats": torch.from_numpy(padded),
            "length": torch.tensor(length, dtype=torch.int64),
            "mtp_verbs": torch.from_numpy(v_lab),
            "mtp_nouns": torch.from_numpy(n_lab),
            "mtp_actions": torch.from_numpy(a_lab),
            "mtp_mask": torch.from_numpy(m_lab),
        }


def topk_acc(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    if labels.numel() == 0:
        return 0.0
    return float(logits.topk(k, dim=-1).indices.eq(labels.unsqueeze(-1)).any(-1).float().mean().item())


def run_epoch(
    model,
    loader,
    device,
    horizons,
    weights,
    optimizer=None,
    train=True,
    label_smoothing: float = 0.0,
    feat_noise: float = 0.0,
    grad_clip: float = 1.0,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
):
    model.train(mode=train)
    loss_sum = 0.0
    n_steps = 0
    totals = defaultdict(float)
    counts = defaultdict(int)

    for batch in tqdm(loader, desc="train" if train else "val", leave=False):
        feats = batch["feats"].to(device, non_blocking=True)
        lengths = batch["length"].to(device, non_blocking=True)
        mtp_v = batch["mtp_verbs"].to(device, non_blocking=True)
        mtp_n = batch["mtp_nouns"].to(device, non_blocking=True)
        mtp_a = batch["mtp_actions"].to(device, non_blocking=True)
        mtp_m = batch["mtp_mask"].to(device, non_blocking=True)

        if train and feat_noise > 0:
            feats = feats + feat_noise * torch.randn_like(feats)

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp and device.type == "cuda"
        ):
            out = model(feats, lengths)
            loss = feats.new_zeros(())
            for hi, h in enumerate(horizons):
                valid = mtp_m[:, hi] > 0.5
                if not bool(valid.any()):
                    continue
                o = out[h]
                v = mtp_v[valid, hi].reshape(-1).long()
                n = mtp_n[valid, hi].reshape(-1).long()
                a = mtp_a[valid, hi].reshape(-1).long()
                keep = (v >= 0) & (n >= 0) & (a >= 0)
                if not bool(keep.any()):
                    continue
                v, n, a = v[keep], n[keep], a[keep]
                valid_idx = valid.nonzero(as_tuple=False).view(-1)[keep]
                logits_v = o["verb"][valid_idx]
                logits_n = o["noun"][valid_idx]
                logits_a = o["action"][valid_idx]
                step = (
                    F.cross_entropy(logits_v, v, label_smoothing=label_smoothing)
                    + F.cross_entropy(logits_n, n, label_smoothing=label_smoothing)
                    + F.cross_entropy(logits_a, a, label_smoothing=label_smoothing)
                )
                loss = loss + float(weights[hi]) * step
                with torch.no_grad():
                    for name, logits, lab in (
                        ("verb", logits_v.float(), v),
                        ("noun", logits_n.float(), n),
                        ("action", logits_a.float(), a),
                    ):
                        for k in (1, 5):
                            key = f"{name}_top{k}@{h:g}s"
                            totals[key] += topk_acc(logits, lab, k=k) * lab.numel()
                            counts[key] += lab.numel()

        if train:
            if not torch.isfinite(loss.detach()):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        loss_sum += float(loss.detach().float().item()) if torch.isfinite(loss.detach()) else 0.0
        n_steps += 1

    metrics = {k: 100.0 * totals[k] / max(1, counts[k]) for k in totals}
    metrics["loss"] = loss_sum / max(1, n_steps)
    return metrics


def format_metrics(metrics: dict, horizons) -> str:
    lines = [f"loss={metrics.get('loss', 0):.4f}"]
    for h in horizons:
        parts = []
        for name in ("verb", "noun", "action"):
            t1 = metrics.get(f"{name}_top1@{h:g}s", 0.0)
            t5 = metrics.get(f"{name}_top5@{h:g}s", 0.0)
            parts.append(f"{name}@top1={t1:.2f}/top5={t5:.2f}")
        lines.append(f"  +{h:g}s: " + " | ".join(parts))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-csv",
        type=Path,
        default=Path(
            "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/HD_EPIC_train_stream_mtp.csv"
        ),
    )
    ap.add_argument(
        "--val-csv",
        type=Path,
        default=Path(
            "/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/HD_EPIC_val_stream_mtp.csv"
        ),
    )
    ap.add_argument(
        "--feat-dir",
        type=Path,
        default=Path("/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/scratch/ll5914/experiments/rulstm_hdepic_p01_stream"),
    )
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--loss-weights", type=str, default="1.0,0.7,0.5")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--max-context-sec", type=float, default=10.0)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=1, help="LSTM num_layers (Rolling+Unrolling)")
    ap.add_argument("--input-proj", action="store_true", help="Project feat_in→hidden before LSTM")
    ap.add_argument("--mlp-head", action="store_true", help="Use 2-layer MLP classifiers")
    ap.add_argument("--trunk-layers", type=int, default=0, help="Residual MLP blocks after unroll")
    ap.add_argument("--no-layernorm", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--lr-milestones", type=str, default="20,30")
    ap.add_argument("--lr-schedule", type=str, default="multistep", choices=["multistep", "cosine"])
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--feat-noise", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=0, help="0=disabled")
    ap.add_argument("--init-from", type=Path, default=None, help="Warm-start overlapping weights")
    ap.add_argument("--val-only", action="store_true")
    args = ap.parse_args()

    horizons = tuple(float(x) for x in args.horizons_sec.split(",") if x.strip())
    weights = [float(x) for x in args.loss_weights.split(",") if x.strip()]
    assert len(horizons) == len(weights)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    verb_map, noun_map, action_map = build_vocab([args.train_csv, args.val_csv])
    vocab = {
        "num_verb": len(verb_map),
        "num_noun": len(noun_map),
        "num_action": len(action_map),
        "horizons_sec": list(horizons),
        "hidden": args.hidden,
        "depth": args.depth,
        "input_proj": bool(args.input_proj),
        "mlp_head": bool(args.mlp_head),
        "trunk_layers": int(args.trunk_layers),
        "use_layernorm": not bool(args.no_layernorm),
    }
    print(
        f"vocab: verbs={len(verb_map)} nouns={len(noun_map)} actions={len(action_map)}",
        flush=True,
    )

    train_ds = StreamFeatureDataset(
        args.train_csv, args.feat_dir, verb_map, noun_map, action_map,
        alpha=args.alpha, max_context_sec=args.max_context_sec, horizons_sec=horizons,
    )
    val_ds = StreamFeatureDataset(
        args.val_csv, args.feat_dir, verb_map, noun_map, action_map,
        alpha=args.alpha, max_context_sec=args.max_context_sec, horizons_sec=horizons,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    feat_in = int(train_ds[0]["feats"].shape[-1])
    model = StreamingRULSTM(
        num_verb=len(verb_map), num_noun=len(noun_map), num_action=len(action_map),
        feat_in=feat_in, hidden=args.hidden, depth=args.depth, dropout=args.dropout,
        alpha=args.alpha, horizons_sec=horizons, input_proj=bool(args.input_proj),
        mlp_head=bool(args.mlp_head), trunk_layers=int(args.trunk_layers),
        use_layernorm=not bool(args.no_layernorm),
    ).to(device)

    if args.init_from is not None and args.init_from.is_file():
        donor = torch.load(args.init_from, map_location="cpu", weights_only=False)["model"]
        own = model.state_dict()
        loaded = 0
        for k, v in donor.items():
            if k not in own:
                continue
            if own[k].shape == v.shape:
                own[k] = v
                loaded += 1
            elif v.dim() == own[k].dim():
                dst = own[k].clone()
                slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, v.shape))
                dst[slices] = v[slices]
                own[k] = dst
                loaded += 1
        model.load_state_dict(own, strict=False)
        print(f"warm-start from {args.init_from}: touched {loaded} tensors", flush=True)

    pcount = model.count_parameters()
    print(
        f"model: hidden={args.hidden} depth={args.depth} input_proj={args.input_proj} "
        f"mlp_head={args.mlp_head} trunk={args.trunk_layers} → {pcount['total_m']:.1f}M params "
        f"(ViT-L target ≈304M)",
        flush=True,
    )
    vocab["param_count"] = pcount
    (args.out_dir / "param_count.json").write_text(json.dumps(pcount, indent=2), encoding="utf-8")
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    ckpt_best = args.out_dir / "rulstm_stream_best.pt"
    ckpt_last = args.out_dir / "rulstm_stream_last.pt"
    epoch_kwargs = dict(
        label_smoothing=float(args.label_smoothing),
        feat_noise=float(args.feat_noise),
        grad_clip=float(args.grad_clip),
        use_amp=bool(args.amp),
    )

    if args.val_only:
        path = ckpt_best if ckpt_best.is_file() else ckpt_last
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        metrics = run_epoch(model, val_loader, device, horizons, weights, train=False, **epoch_kwargs)
        print("VAL\n" + format_metrics(metrics, horizons), flush=True)
        (args.out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        )

    warmup = max(0, int(args.warmup_epochs))
    if args.lr_schedule == "cosine":
        def lr_lambda(epoch0: int):
            ep = epoch0 + 1
            if warmup > 0 and ep <= warmup:
                return ep / float(warmup)
            progress = (ep - warmup) / max(1, args.epochs - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        milestones = [int(x) for x in args.lr_milestones.split(",") if x.strip()]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=True) if args.amp and device.type == "cuda" else None

    best = -1.0
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, device, horizons, weights, optimizer, train=True,
            scaler=scaler, **epoch_kwargs,
        )
        va = run_epoch(model, val_loader, device, horizons, weights, train=False, **epoch_kwargs)
        scheduler.step()
        primary = va.get("action_top5@2s", 0.0)
        print(f"\n=== epoch {epoch}/{args.epochs} lr={optimizer.param_groups[0]['lr']:.2e} ===", flush=True)
        print("TRAIN\n" + format_metrics(tr, horizons), flush=True)
        print("VAL\n" + format_metrics(va, horizons), flush=True)
        history.append({"epoch": epoch, "train": tr, "val": va})
        torch.save({"model": model.state_dict(), "epoch": epoch, "vocab": vocab, "val": va}, ckpt_last)
        if primary >= best:
            best = primary
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "vocab": vocab, "val": va}, ckpt_best)
            print(f"  saved best (action_top5@2s={best:.2f})", flush=True)
        else:
            bad += 1
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.early_stop_patience > 0 and bad >= args.early_stop_patience:
            print(f"early stop at epoch {epoch} (patience={args.early_stop_patience})", flush=True)
            break

    print(f"\nDone. best action_top5@2s={best:.2f}", flush=True)
    state = torch.load(ckpt_best, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    metrics = run_epoch(model, val_loader, device, horizons, weights, train=False, **epoch_kwargs)
    print("BEST VAL\n" + format_metrics(metrics, horizons), flush=True)
    (args.out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
