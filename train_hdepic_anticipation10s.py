"""
HD-EPIC long-horizon action anticipation training (default: 10s ahead).

Freezes V-JEPA2 ViT-L encoder + predictor (AnticipativeWrapper); only the HD-EPIC probe is trained.
Future latent tokens are generated autoregressively by the predictor conditioned on anticipation_time;
the probe performs verb / noun / action classification.

Usage:
  cd /home/ll5914/ARVR_Video/vjepa2
  python ../train_hdepic_anticipation10s.py                 # resumes from last.pt
  python ../train_hdepic_anticipation10s.py --from-scratch

Environment variables:
  HDEPIC_ANTICIPATION_SEC=10.0   anticipation horizon in seconds
  HDEPIC_AR_STEPS=1              predictor AR rollout steps
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/home/ll5914/ARVR_Video")
sys.path.insert(0, "/home/ll5914/ARVR_Video/vjepa2")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import hdepic_anticipation_ar as cfg
from hdepic_anticipation_ar import (
    HDEpicAnticipationDataset,
    HDEpicProbe,
    anticipate_features,
    build_anticipative_model,
    build_transforms,
    load_label_maps,
    split_data,
)

# ── Training hyper-parameters ────────────────────────────────────────
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS = 4


@torch.no_grad()
def evaluate(model, probe, loader, device, anticipation_sec):
    probe.eval()
    v_c = defaultdict(int); v_t = defaultdict(int)
    n_c = defaultdict(int); n_t = defaultdict(int)
    a_c = defaultdict(int); a_t = defaultdict(int)
    v_top3 = n_top3 = a_top3 = 0
    total = 0

    for clips, v_ids, n_ids, a_ids in loader:
        clips = clips.to(device)
        feats = anticipate_features(model, clips, anticipation_sec, device)
        v_logits, n_logits, a_logits = probe(feats)
        for i in range(len(v_ids)):
            vi, ni, ai = int(v_ids[i]), int(n_ids[i]), int(a_ids[i])
            v_t[vi] += 1; n_t[ni] += 1
            if vi in v_logits[i].topk(3).indices.tolist(): v_top3 += 1
            if ni in n_logits[i].topk(3).indices.tolist(): n_top3 += 1
            if vi in v_logits[i].topk(5).indices.tolist(): v_c[vi] += 1
            if ni in n_logits[i].topk(5).indices.tolist(): n_c[ni] += 1
            if ai != -1:
                a_t[ai] += 1
                if ai in a_logits[i].topk(5).indices.tolist(): a_c[ai] += 1
                if ai in a_logits[i].topk(3).indices.tolist(): a_top3 += 1
            total += 1

    def cmr(c, t):
        r = [c.get(k, 0) / v for k, v in t.items()]
        return float(np.mean(r) * 100) if r else 0.0

    n_act = max(sum(a_t.values()), 1)
    return {
        "verb_top3": 100 * v_top3 / max(total, 1),
        "noun_top3": 100 * n_top3 / max(total, 1),
        "action_top3": 100 * a_top3 / n_act,
        "verb_r5": cmr(v_c, v_t),
        "noun_r5": cmr(n_c, n_t),
        "action_r5": cmr(a_c, a_t),
    }


def run(from_scratch=False):
    anticipation_sec = cfg.get_anticipation_sec()
    ar_steps = cfg.get_ar_steps()

    print("=" * 70)
    print("V-JEPA 2 — HD-EPIC Long-Horizon Anticipation Training (AR predictor)")
    print(f"Anticipation = {anticipation_sec}s | AR steps = {ar_steps}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    print("\n[1] Loading annotations...")
    with open(cfg.HD_EPIC_NARR, "rb") as f:
        narr_df = pickle.load(f)
    p01_df = narr_df[narr_df["video_id"].str.startswith("P01")].copy()
    vdf, ndf, verb_map, noun_map, action_map, verb_names, noun_names = load_label_maps(p01_df, pd)
    print(f"  verbs={len(vdf)}, nouns={len(ndf)}, actions={len(action_map)}")

    train_df, val_df, test_df = split_data(p01_df)
    print(f"  Train : {len(train_df):4d} rows | {train_df['video_id'].nunique():2d} videos")
    print(f"  Val   : {len(val_df):4d} rows | {val_df['video_id'].nunique():2d} videos")
    print(f"  Test  : {len(test_df):4d} rows | {test_df['video_id'].nunique():2d} videos", flush=True)

    train_ds = HDEpicAnticipationDataset(train_df, build_transforms(True),  verb_map, noun_map, action_map, anticipation_sec)
    val_ds   = HDEpicAnticipationDataset(val_df,   build_transforms(False), verb_map, noun_map, action_map, anticipation_sec)
    test_ds  = HDEpicAnticipationDataset(test_df,  build_transforms(False), verb_map, noun_map, action_map, anticipation_sec)
    print(f"  Valid samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)} "
          f"(after {anticipation_sec}s filter)", flush=True)
    if len(train_ds) == 0:
        print("  [ERROR] No training samples — reduce HDEPIC_ANTICIPATION_SEC.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print("\n[2] Loading model (encoder + predictor frozen)...")
    model = build_anticipative_model(device, num_steps=ar_steps)
    probe = HDEpicProbe(
        embed_dim=model.embed_dim,
        num_verbs=len(vdf),
        num_nouns=len(ndf),
        num_actions=len(action_map),
    ).to(device)
    print(f"  Probe parameters: {sum(p.numel() for p in probe.parameters()) / 1e6:.1f}M")

    optimizer = optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    def pack_ckpt(completed_epochs, metrics):
        return {
            "epoch": completed_epochs,
            "probe": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "verb_names": verb_names,
            "noun_names": noun_names,
            "action_map": action_map,
            "metrics": metrics,
            "anticipation_sec": anticipation_sec,
            "ar_steps": ar_steps,
        }

    start_epoch = 0
    best_verb_r5 = 0.0
    if not from_scratch and os.path.isfile(cfg.PROBE_LAST):
        print(f"\n  [resume] Loading {cfg.PROBE_LAST} ...", flush=True)
        ck = torch.load(cfg.PROBE_LAST, map_location=device, weights_only=False)
        probe.load_state_dict(ck["probe"])
        start_epoch = int(ck["epoch"])
        if ck.get("optimizer"):
            optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        if os.path.isfile(cfg.PROBE_BEST):
            b = torch.load(cfg.PROBE_BEST, map_location="cpu", weights_only=False)
            if b.get("metrics") and "verb_r5" in b["metrics"]:
                best_verb_r5 = float(b["metrics"]["verb_r5"])
        if start_epoch >= NUM_EPOCHS:
            print(f"  All {NUM_EPOCHS} epochs done.")
            return
        print(f"  Resuming from epoch {start_epoch + 1}/{NUM_EPOCHS}, best verb R@5={best_verb_r5:.1f}%")
    else:
        print("\n  Training probe from scratch.", flush=True)

    print(f"\n[3] Starting training ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR})...", flush=True)
    print(f"    latest -> {cfg.PROBE_LAST}")
    print(f"    best(verb R@5) -> {cfg.PROBE_BEST}")
    print("=" * 70, flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        probe.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, (clips, v_ids, n_ids, a_ids) in enumerate(train_loader):
            clips = clips.to(device)
            v_ids = v_ids.to(device)
            n_ids = n_ids.to(device)
            a_ids = a_ids.to(device)

            feats = anticipate_features(model, clips, anticipation_sec, device)
            v_logits, n_logits, a_logits = probe(feats)
            loss = criterion(v_logits, v_ids) + criterion(n_logits, n_ids)
            valid_a = a_ids >= 0
            if valid_a.any():
                loss = loss + criterion(a_logits[valid_a], a_ids[valid_a])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | step {batch_idx+1}/{len(train_loader)} "
                      f"| loss={loss.item():.3f} | lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} done | avg_loss={avg_loss:.3f} | {time.time()-t0:.0f}s", flush=True)

        metrics = evaluate(model, probe, val_loader, device, anticipation_sec)
        print(f"  Validation (anticipation {anticipation_sec}s):", flush=True)
        print(f"    Verb  Top-3={metrics['verb_top3']:.1f}%  R@5={metrics['verb_r5']:.1f}%", flush=True)
        print(f"    Noun  Top-3={metrics['noun_top3']:.1f}%  R@5={metrics['noun_r5']:.1f}%", flush=True)
        print(f"    Action Top-3={metrics['action_top3']:.1f}%  R@5={metrics['action_r5']:.1f}%", flush=True)

        torch.save(pack_ckpt(epoch + 1, metrics), cfg.PROBE_LAST)
        print(f"  ✓ latest -> {cfg.PROBE_LAST}", flush=True)
        if metrics["verb_r5"] > best_verb_r5:
            best_verb_r5 = metrics["verb_r5"]
            torch.save(pack_ckpt(epoch + 1, metrics), cfg.PROBE_BEST)
            print(f"  ✓ best (verb R@5={best_verb_r5:.1f}%) -> {cfg.PROBE_BEST}", flush=True)
        print("", flush=True)

    print("=" * 70, flush=True)
    print(f"Training complete! Best Val Verb R@5 = {best_verb_r5:.1f}%", flush=True)

    # ── Final evaluation on the held-out test set (best checkpoint) ──
    print("\n[4] Final test-set evaluation (loading best checkpoint)...", flush=True)
    best_ck = torch.load(cfg.PROBE_BEST, map_location=device, weights_only=False)
    probe.load_state_dict(best_ck["probe"])
    test_metrics = evaluate(model, probe, test_loader, device, anticipation_sec)
    print("  Test set results (best val checkpoint):", flush=True)
    print(f"    Verb   Top-3={test_metrics['verb_top3']:.1f}%  R@5={test_metrics['verb_r5']:.1f}%", flush=True)
    print(f"    Noun   Top-3={test_metrics['noun_top3']:.1f}%  R@5={test_metrics['noun_r5']:.1f}%", flush=True)
    print(f"    Action Top-3={test_metrics['action_top3']:.1f}%  R@5={test_metrics['action_r5']:.1f}%", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from-scratch", action="store_true", help="Ignore last.pt and train from scratch")
    args = p.parse_args()
    run(from_scratch=args.from_scratch)
