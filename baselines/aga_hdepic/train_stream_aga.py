#!/usr/bin/env python3
"""AGA baseline on HD-EPIC P01 under the Streaming Video protocol.

Protocol (same as RU-LSTM / V-JEPA stream MTP):
  - Temporal half-split per video (1st half train / 2nd half val)
  - Growing context 4→6→8→10s from half origin, then slide 10s; tick every 2s
  - Predict actions at +2 / +4 / +6 s

Uses the official Action-Guided Attention model from CorcovadoMing/AGA on the
same TSN-RGB features as ``rulstm_hdepic`` (1024-d @ 4 fps). Observed context is
subsampled to AGA's 1s step; the recurrent loop then unrolls the last visual
token for ``horizon / time_step`` steps so each horizon is a true future query.
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
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

def _find_aga_root() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1] / "AGA",
        Path("/scratch/ll5914/code/AGA"),
        Path("/home/ll5914/Jepa_baseline/AGA"),
    ]
    for p in candidates:
        if (p / "model.py").is_file():
            return p.resolve()
    raise FileNotFoundError("AGA vendor tree not found (expected ../AGA/model.py)")


AGA_ROOT = _find_aga_root()
sys.path.insert(0, str(AGA_ROOT))
from model import AGA  # noqa: E402


def build_vocab(csvs: list[Path]):
    verbs, nouns, actions = set(), set(), set()
    for p in csvs:
        df = pd.read_csv(p, usecols=["mtp_verbs", "mtp_nouns", "mtp_mask"])
        for vs, ns, ms in zip(df.mtp_verbs, df.mtp_nouns, df.mtp_mask):
            vlist = [int(x) for x in str(vs).split(",")]
            nlist = [int(x) for x in str(ns).split(",")]
            mlist = [float(x) for x in str(ms).split(",")]
            for v, n, m in zip(vlist, nlist, mlist):
                if m > 0:
                    verbs.add(v)
                    nouns.add(n)
                    actions.add((v, n))
    verb_list = sorted(verbs)
    noun_list = sorted(nouns)
    action_list = sorted(actions)
    return (
        {v: i for i, v in enumerate(verb_list)},
        {n: i for i, n in enumerate(noun_list)},
        {a: i for i, a in enumerate(action_list)},
    )


class StreamFeatureDataset(Dataset):
    """TSN features in [start_frame, tick_frame], subsampled to ``time_step``."""

    def __init__(
        self,
        csv_path: Path,
        feat_dir: Path,
        verb_map: dict,
        noun_map: dict,
        action_map: dict,
        feat_alpha: float = 0.25,
        time_step: float = 1.0,
        max_context_sec: float = 10.0,
        horizons_sec: tuple[float, ...] = (2.0, 4.0, 6.0),
    ):
        self.df = pd.read_csv(csv_path)
        self.feat_dir = Path(feat_dir)
        self.verb_map = verb_map
        self.noun_map = noun_map
        self.action_map = action_map
        self.feat_alpha = float(feat_alpha)
        self.time_step = float(time_step)
        self.stride = max(1, int(round(self.time_step / self.feat_alpha)))
        self.max_len = max(1, int(round(max_context_sec / self.time_step)))
        self.horizons_sec = tuple(float(h) for h in horizons_sec)
        self.n_keep = [
            max(1, min(self.max_len, int(round(float(c) / self.time_step))))
            for c in self.df["context_sec"].tolist()
        ]
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

        mask = (frame_indices >= start_frame) & (frame_indices <= tick_frame)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            idxs = np.array([int(np.searchsorted(frame_indices, tick_frame, side="right") - 1)])
            idxs = np.clip(idxs, 0, len(frame_indices) - 1)

        seq = feat[idxs]
        if self.stride > 1 and len(seq) > 1:
            take = np.arange(len(seq) - 1, -1, -self.stride)[::-1]
            seq = seq[take]
        n_keep = int(self.n_keep[index])
        if len(seq) > n_keep:
            seq = seq[-n_keep:]
        if len(seq) == 0:
            seq = np.zeros((1, feat.shape[-1]), dtype=np.float32)

        verbs = [int(x) for x in str(r.mtp_verbs).split(",")]
        nouns = [int(x) for x in str(r.mtp_nouns).split(",")]
        masks = [float(x) for x in str(r.mtp_mask).split(",")]

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
            "feats": torch.from_numpy(np.ascontiguousarray(seq, dtype=np.float32)),
            "length": torch.tensor(seq.shape[0], dtype=torch.int64),
            "mtp_verbs": torch.from_numpy(v_lab),
            "mtp_nouns": torch.from_numpy(n_lab),
            "mtp_actions": torch.from_numpy(a_lab),
            "mtp_mask": torch.from_numpy(m_lab),
        }


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool, seed: int = 0):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, L in enumerate(lengths):
            buckets[int(L)].append(i)
        self.buckets = dict(buckets)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

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
        yield from batches

    def __len__(self):
        return sum(int(np.ceil(len(v) / self.batch_size)) for v in self.buckets.values())


def collate_stream(batch: list[dict]) -> dict:
    max_t = max(int(b["length"]) for b in batch)
    dim = int(batch[0]["feats"].shape[-1])
    feats = torch.zeros(len(batch), max_t, dim, dtype=torch.float32)
    lengths = torch.zeros(len(batch), dtype=torch.int64)
    for i, b in enumerate(batch):
        t = int(b["length"])
        lengths[i] = t
        if t <= 0:
            continue
        # Right-align so the tick is the last step; repeat the first feature on the left.
        if t < max_t:
            feats[i, : max_t - t] = b["feats"][:1]
        feats[i, max_t - t :] = b["feats"]
        lengths[i] = max_t
    return {
        "feats": feats,
        "length": lengths,
        "mtp_verbs": torch.stack([b["mtp_verbs"] for b in batch]),
        "mtp_nouns": torch.stack([b["mtp_nouns"] for b in batch]),
        "mtp_actions": torch.stack([b["mtp_actions"] for b in batch]),
        "mtp_mask": torch.stack([b["mtp_mask"] for b in batch]),
    }


class StreamingAGA(nn.Module):
    """Causal AGA over observed context, then unroll last visual token to +2/+4/+6s.

    Verb / noun heads sit on the action-guided hidden state ``hx``. Action logits
    come from AGA's native anticipation classifier so they remain the recurrent
    query (the paper's action-guided memory).
    """

    def __init__(
        self,
        num_verb: int,
        num_noun: int,
        num_action: int,
        in_dim: int = 1024,
        hidden_dim: int = 2048,
        order: int = 40,
        dropout: float = 0.6,
        attention_dropout: float = 0.6,
        recurrent_query: str = "ma",
        ma_ratio: float = 0.8,
        time_step: float = 1.0,
        horizons_sec: tuple[float, ...] = (2.0, 4.0, 6.0),
    ):
        super().__init__()
        self.time_step = float(time_step)
        self.horizons_sec = tuple(float(h) for h in horizons_sec)
        self.horizon_steps = {
            h: max(1, int(round(h / self.time_step))) for h in self.horizons_sec
        }
        self.max_unroll = max(self.horizon_steps.values())
        self.aga = AGA(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=num_action,
            order=order,
            dropout=dropout,
            attention_dropout=attention_dropout,
            recurrent_query=recurrent_query,
            ma_ratio=ma_ratio,
            recurrent_h=False,
            gate_x=True,
            return_embedding=False,
        )
        self.verb_clf = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_verb),
        )
        self.noun_clf = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_noun),
        )

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": int(total),
            "trainable": int(trainable),
            "total_m": total / 1e6,
            "trainable_m": trainable / 1e6,
            "aga_m": sum(p.numel() for p in self.aga.parameters()) / 1e6,
        }

    def _mix_query(self, q: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
        aga = self.aga
        if aga.query_prob:
            query_t = query_t.softmax(-1)
        if aga.recurrent_query == "lstm":
            hx, cx = aga.rnn(query_t.squeeze(1), self._lstm_state)
            self._lstm_state = (hx, cx)
            return hx.unsqueeze(1)
        if aga.recurrent_query == "ma":
            if aga.query_prob:
                return q.softmax(-1) * (1.0 - aga.ma_ratio) + query_t * aga.ma_ratio
            return q * (1.0 - aga.ma_ratio) + query_t * aga.ma_ratio
        return query_t

    def _heads(self, action_logits: torch.Tensor, hx: torch.Tensor) -> dict:
        z = hx.squeeze(1)
        return {
            "verb": self.verb_clf(z),
            "noun": self.noun_clf(z),
            "action": action_logits.squeeze(1),
        }

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        """
        feats: [B, T, D] (zero-padded on the right; tick = last valid step)
        lengths: [B]
        """
        B, T, _ = feats.shape
        aga = self.aga
        aga.memory.reset(B)
        aga.evidence_ratio = []
        if aga.recurrent_query == "lstm":
            z = torch.zeros(B, aga.out_dim, device=feats.device, dtype=feats.dtype)
            self._lstm_state = (z, torch.zeros_like(z))

        enc = aga.encoder(feats)
        q = torch.zeros(B, 1, aga.out_dim, device=feats.device, dtype=enc.dtype)
        query_t = q
        last_x = enc[:, :1]
        active_prev = torch.zeros(B, 1, 1, device=feats.device, dtype=enc.dtype)

        for t in range(T):
            x_t = enc[:, t : t + 1]
            active = (t < lengths).view(B, 1, 1).to(dtype=enc.dtype)
            x = torch.where(active.bool(), x_t, last_x)
            if t > 0:
                q = torch.where(active_prev.bool(), self._mix_query(q, query_t), q)
            query_t, hx = aga.grab_state(x, q, gt=None, update_memory=True)
            last_x = torch.where(active.bool(), x, last_x)
            active_prev = active

        out = {}
        snap_at = {step: h for h, step in self.horizon_steps.items()}
        for step in range(1, self.max_unroll + 1):
            q = self._mix_query(q, query_t)
            query_t, hx = aga.grab_state(last_x, q, gt=None, update_memory=True)
            if step in snap_at:
                out[snap_at[step]] = self._heads(query_t, hx)
        return out


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
        default=Path("/scratch/ll5914/experiments/aga_hdepic_p01_stream"),
    )
    ap.add_argument("--horizons-sec", type=str, default="2,4,6")
    ap.add_argument("--loss-weights", type=str, default="1.0,0.7,0.5")
    ap.add_argument("--feat-alpha", type=float, default=0.25, help="TSN feature stride in seconds")
    ap.add_argument("--time-step", type=float, default=1.0, help="AGA observation / unroll step (paper=1s)")
    ap.add_argument("--max-context-sec", type=float, default=10.0)
    ap.add_argument("--hidden-dim", type=int, default=2048)
    ap.add_argument("--order", type=int, default=40, help="AGA memory length (context + unroll)")
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--attention-dropout", type=float, default=0.6)
    ap.add_argument("--recurrent-query", type=str, default="ma")
    ap.add_argument("--ma-ratio", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    horizons = tuple(float(x) for x in args.horizons_sec.split(",") if x.strip())
    weights = [float(x) for x in args.loss_weights.split(",") if x.strip()]
    assert len(horizons) == len(weights)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = args.out_dir / "TRAINING_DONE"
    if done_flag.is_file() and not args.val_only:
        print(f"TRAINING_DONE present ({done_flag}); exiting.", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    verb_map, noun_map, action_map = build_vocab([args.train_csv, args.val_csv])
    vocab = {
        "num_verb": len(verb_map),
        "num_noun": len(noun_map),
        "num_action": len(action_map),
        "horizons_sec": list(horizons),
        "hidden_dim": args.hidden_dim,
        "order": args.order,
        "time_step": args.time_step,
        "feat_alpha": args.feat_alpha,
    }
    print(
        f"vocab: verbs={len(verb_map)} nouns={len(noun_map)} actions={len(action_map)}",
        flush=True,
    )

    ds_kwargs = dict(
        feat_dir=args.feat_dir,
        verb_map=verb_map,
        noun_map=noun_map,
        action_map=action_map,
        feat_alpha=args.feat_alpha,
        time_step=args.time_step,
        max_context_sec=args.max_context_sec,
        horizons_sec=horizons,
    )
    train_ds = StreamFeatureDataset(args.train_csv, **ds_kwargs)
    val_ds = StreamFeatureDataset(args.val_csv, **ds_kwargs)
    train_sampler = LengthBucketBatchSampler(train_ds.n_keep, args.batch_size, shuffle=True, seed=args.seed)
    val_sampler = LengthBucketBatchSampler(val_ds.n_keep, args.batch_size, shuffle=False, seed=args.seed)
    loader_kw = dict(
        num_workers=args.num_workers,
        collate_fn=collate_stream,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kw)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kw)

    feat_in = int(train_ds[0]["feats"].shape[-1])
    model = StreamingAGA(
        num_verb=len(verb_map),
        num_noun=len(noun_map),
        num_action=len(action_map),
        in_dim=feat_in,
        hidden_dim=args.hidden_dim,
        order=args.order,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        recurrent_query=args.recurrent_query,
        ma_ratio=args.ma_ratio,
        time_step=args.time_step,
        horizons_sec=horizons,
    ).to(device)

    pcount = model.count_parameters()
    print(
        f"StreamingAGA hidden={args.hidden_dim} order={args.order} "
        f"time_step={args.time_step}s unroll={model.horizon_steps} → {pcount['total_m']:.1f}M params",
        flush=True,
    )
    vocab["param_count"] = pcount
    (args.out_dir / "param_count.json").write_text(json.dumps(pcount, indent=2), encoding="utf-8")
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    ckpt_best = args.out_dir / "aga_stream_best.pt"
    ckpt_last = args.out_dir / "aga_stream_last.pt"
    epoch_kwargs = dict(
        label_smoothing=float(args.label_smoothing),
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True) if args.amp and device.type == "cuda" else None

    best = -1.0
    bad = 0
    history = []
    start_epoch = 1
    if ckpt_last.is_file():
        state = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best = float(state.get("best", -1.0))
        hist_path = args.out_dir / "history.json"
        if hist_path.is_file():
            history = json.loads(hist_path.read_text())
        print(f"resumed from {ckpt_last} epoch={start_epoch} best={best:.2f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
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
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "vocab": vocab,
            "val": va,
            "best": best,
        }
        torch.save(payload, ckpt_last)
        if primary >= best:
            best = primary
            payload["best"] = best
            bad = 0
            torch.save(payload, ckpt_best)
            print(f"  saved best (action_top5@2s={best:.2f})", flush=True)
        else:
            bad += 1
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.early_stop_patience > 0 and bad >= args.early_stop_patience:
            print(f"early stop at epoch {epoch} (patience={args.early_stop_patience})", flush=True)
            break

    print(f"\nDone. best action_top5@2s={best:.2f}", flush=True)
    if ckpt_best.is_file():
        state = torch.load(ckpt_best, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    metrics = run_epoch(model, val_loader, device, horizons, weights, train=False, **epoch_kwargs)
    print("BEST VAL\n" + format_metrics(metrics, horizons), flush=True)
    (args.out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    done_flag.write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
