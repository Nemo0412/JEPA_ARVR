#!/usr/bin/env python3
"""Streaming LSTM baseline on HD-EPIC P01.

Protocol: last 4s of observed context at each stream tick, predict the action
+2s ahead. Uses the same TSN-RGB features @4fps and stream CSVs as RU-LSTM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rulstm_hdepic"))
from train_stream_rulstm import (  # noqa: E402
    StreamFeatureDataset,
    build_vocab,
    format_metrics,
    run_epoch,
)


class Plus2sDataset(Dataset):
    """Keep only the +2s horizon from the MTP stream CSV."""

    def __init__(self, base: StreamFeatureDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        for key in ("mtp_verbs", "mtp_nouns", "mtp_actions", "mtp_mask"):
            item[key] = item[key][:1]
        return item


class StreamingLSTM(nn.Module):
    def __init__(
        self,
        num_verb: int,
        num_noun: int,
        num_action: int,
        feat_in: int = 1024,
        hidden: int = 1024,
        depth: int = 1,
        dropout: float = 0.5,
        horizon_sec: float = 2.0,
    ):
        super().__init__()
        self.horizon_sec = float(horizon_sec)
        self.feat_norm = nn.LayerNorm(feat_in)
        self.lstm = nn.LSTM(
            feat_in,
            hidden,
            num_layers=int(depth),
            batch_first=True,
            dropout=float(dropout) if int(depth) > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.verb_clf = nn.Linear(hidden, num_verb)
        self.noun_clf = nn.Linear(hidden, num_noun)
        self.action_clf = nn.Linear(hidden, num_action)

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
        x = self.feat_norm(feats)
        packed = pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = self.drop(h_n[-1])
        return {
            self.horizon_sec: {
                "verb": self.verb_clf(h),
                "noun": self.noun_clf(h),
                "action": self.action_clf(h),
            }
        }


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
        default=Path("/scratch/ll5914/experiments/lstm_hdepic_p01_stream_4s2s"),
    )
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--max-context-sec", type=float, default=4.0)
    ap.add_argument("--horizon-sec", type=float, default=2.0)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = (float(args.horizon_sec),)
    weights = [1.0]
    csv_horizons = (2.0, 4.0, 6.0)

    verb_map, noun_map, action_map = build_vocab([args.train_csv, args.val_csv])
    vocab = {
        "num_verb": len(verb_map),
        "num_noun": len(noun_map),
        "num_action": len(action_map),
        "horizons_sec": list(horizons),
        "max_context_sec": args.max_context_sec,
    }
    print(
        f"device={device} vocab v/n/a={len(verb_map)}/{len(noun_map)}/{len(action_map)}",
        flush=True,
    )

    train_ds = Plus2sDataset(
        StreamFeatureDataset(
            args.train_csv, args.feat_dir, verb_map, noun_map, action_map,
            alpha=args.alpha, max_context_sec=args.max_context_sec, horizons_sec=csv_horizons,
        )
    )
    val_ds = Plus2sDataset(
        StreamFeatureDataset(
            args.val_csv, args.feat_dir, verb_map, noun_map, action_map,
            alpha=args.alpha, max_context_sec=args.max_context_sec, horizons_sec=csv_horizons,
        )
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    feat_in = int(train_ds[0]["feats"].shape[-1])
    model = StreamingLSTM(
        num_verb=len(verb_map), num_noun=len(noun_map), num_action=len(action_map),
        feat_in=feat_in, hidden=args.hidden, depth=args.depth, dropout=args.dropout,
        horizon_sec=args.horizon_sec,
    ).to(device)
    pcount = model.count_parameters()
    print(f"LSTM hidden={args.hidden} depth={args.depth} → {pcount['total_m']:.1f}M", flush=True)
    (args.out_dir / "param_count.json").write_text(json.dumps(pcount, indent=2), encoding="utf-8")
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    epoch_kwargs = dict(grad_clip=float(args.grad_clip), use_amp=bool(args.amp))

    ckpt_best = args.out_dir / "lstm_stream_best.pt"
    ckpt_last = args.out_dir / "lstm_stream_last.pt"
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, device, horizons, weights, optimizer, train=True, **epoch_kwargs,
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
            torch.save({"model": model.state_dict(), "epoch": epoch, "vocab": vocab, "val": va}, ckpt_best)
            print(f"  saved best (action_top5@2s={best:.2f})", flush=True)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nDone. best action_top5@2s={best:.2f}", flush=True)
    state = torch.load(ckpt_best, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    metrics = run_epoch(model, val_loader, device, horizons, weights, train=False, **epoch_kwargs)
    print("BEST VAL\n" + format_metrics(metrics, horizons), flush=True)
    (args.out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
