"""
HD-EPIC long-horizon action anticipation inference/evaluation (default: 10s ahead).

Loads the trained probe (hdepic-vitl-ar10s-probe-*.pt) and evaluates on the val set:
Verb/Noun/Action Top-3 and class-mean Recall@5.

Usage:
  cd /home/ll5914/ARVR_Video/vjepa2
  python ../run_hdepic_anticipation10s.py

Environment variables:
  HDEPIC_ANTICIPATION_SEC=10.0   anticipation horizon (must match training)
  HDEPIC_AR_STEPS=1              AR rollout steps
  HDEPIC_EVAL_MAX=0              evaluate first N samples (0 = all)
  HDEPIC_EVAL_BATCH=4
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/ll5914/ARVR_Video")
sys.path.insert(0, "/home/ll5914/ARVR_Video/vjepa2")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import hdepic_anticipation_ar as cfg
from hdepic_anticipation_ar import (
    HDEpicAnticipationDataset,
    HDEpicProbe,
    anticipate_features,
    build_anticipative_model,
    build_transforms,
    load_label_maps,
    split_train_val,
)


def run():
    anticipation_sec = cfg.get_anticipation_sec()
    ar_steps = cfg.get_ar_steps()
    max_eval = int(os.environ.get("HDEPIC_EVAL_MAX", "0"))
    bs = int(os.environ.get("HDEPIC_EVAL_BATCH", "4"))

    print("=" * 70)
    print("V-JEPA 2 — HD-EPIC Long-Horizon Anticipation Inference (AR predictor)")
    print(f"Anticipation = {anticipation_sec}s | AR steps = {ar_steps}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    ckpt_path = Path(cfg.PROBE_LAST) if Path(cfg.PROBE_LAST).is_file() else Path(cfg.PROBE_BEST)
    if not ckpt_path.is_file():
        print(f"Probe checkpoint not found: {cfg.PROBE_LAST} or {cfg.PROBE_BEST}")
        print("Please run train_hdepic_anticipation10s.py first.")
        return
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"Checkpoint: {ckpt_path} (epoch {ck.get('epoch')}, trained with anticipation {ck.get('anticipation_sec')}s)")
    action_map = ck.get("action_map")
    if action_map is None:
        raise KeyError("checkpoint missing action_map")
    if ck.get("anticipation_sec") is not None and abs(float(ck["anticipation_sec"]) - anticipation_sec) > 1e-6:
        print(f"  [WARNING] Eval anticipation {anticipation_sec}s differs from training {ck['anticipation_sec']}s.")

    print("\n[1] Loading annotations...")
    with open(cfg.HD_EPIC_NARR, "rb") as f:
        narr_df = pickle.load(f)
    p01_df = narr_df[narr_df["video_id"].str.startswith("P01")].copy()
    vdf, ndf, verb_map, noun_map, _action_map_local, _, _ = load_label_maps(p01_df, pd)
    _, val_df = split_train_val(p01_df)

    val_ds = HDEpicAnticipationDataset(val_df, build_transforms(False), verb_map, noun_map, action_map, anticipation_sec)
    if max_eval > 0:
        val_ds.samples = val_ds.samples[:max_eval]
    print(f"  Val samples: {len(val_ds)} (anticipation {anticipation_sec}s)", flush=True)
    if len(val_ds) == 0:
        print("  No valid val samples, exiting.")
        return
    loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    print("\n[2] Loading model (encoder + predictor frozen)...")
    model = build_anticipative_model(device, num_steps=ar_steps)
    probe = HDEpicProbe(
        embed_dim=model.embed_dim,
        num_verbs=len(vdf),
        num_nouns=len(ndf),
        num_actions=len(action_map),
    ).to(device)
    probe.load_state_dict(ck["probe"], strict=True)
    probe.eval()

    v_c = defaultdict(int); v_t = defaultdict(int)
    n_c = defaultdict(int); n_t = defaultdict(int)
    a_c = defaultdict(int); a_t = defaultdict(int)
    v_top3 = n_top3 = a_top3 = 0
    total = 0

    print("\n[3] Running per-batch inference...", flush=True)
    with torch.no_grad():
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
            if total % 40 == 0:
                print(f"  Processed {total} samples...", flush=True)

    def cmr(c, t):
        r = [c.get(k, 0) / v for k, v in t.items()]
        return float(np.mean(r) * 100) if r else 0.0

    n_act = max(sum(a_t.values()), 1)
    print(f"\n{'=' * 70}")
    print(f"HD-EPIC val set  |  {total} samples  |  anticipation {anticipation_sec}s  |  AR steps={ar_steps}")
    print(f"{'=' * 70}")
    print(f"  Verb   Top-3={100*v_top3/max(total,1):.1f}%   class-mean R@5={cmr(v_c, v_t):.1f}%")
    print(f"  Noun   Top-3={100*n_top3/max(total,1):.1f}%   class-mean R@5={cmr(n_c, n_t):.1f}%")
    print(f"  Action Top-3={100*a_top3/n_act:.1f}%   class-mean R@5={cmr(a_c, a_t):.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
