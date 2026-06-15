"""
HD-EPIC long-horizon anticipation inference (10s ahead) + Camera Pose fusion.

Loads hdepic-vitl-ar10s-pose-probe-*.pt and evaluates on the val set:
Verb/Noun/Action Top-3 and class-mean Recall@5.

Usage:
  cd /home/ll5914/ARVR_Video/vjepa2
  python ../run_hdepic_anticipation10s_pose.py

Environment variables:
  HDEPIC_ANTICIPATION_SEC=10.0
  HDEPIC_AR_STEPS=10
  HDEPIC_EVAL_MAX=0     evaluate first N samples (0 = all)
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
    HDEpicProbe,
    anticipate_features,
    build_anticipative_model,
    build_transforms,
    load_label_maps,
    split_data,
)
from hdepic_pose_encoder import PoseEncoder, PoseLoader
from train_hdepic_anticipation10s_pose import (
    PROBE_BEST_POSE,
    PROBE_LAST_POSE,
    HDEpicProbeWithPose,
    HDEpicPoseDataset,
    collate_fn,
)


def run():
    anticipation_sec = cfg.get_anticipation_sec()
    ar_steps  = cfg.get_ar_steps()
    max_eval  = int(os.environ.get("HDEPIC_EVAL_MAX", "0"))
    bs        = int(os.environ.get("HDEPIC_EVAL_BATCH", "4"))

    print("=" * 70)
    print("V-JEPA 2 — HD-EPIC Long-Horizon Anticipation Inference + Camera Pose Fusion")
    print(f"Anticipation={anticipation_sec}s | AR steps={ar_steps}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    ckpt_path = Path(PROBE_LAST_POSE) if Path(PROBE_LAST_POSE).is_file() else Path(PROBE_BEST_POSE)
    if not ckpt_path.is_file():
        print(f"Pose probe checkpoint not found: {PROBE_LAST_POSE}")
        print("Please run train_hdepic_anticipation10s_pose.py first.")
        return
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"Checkpoint: {ckpt_path}  (epoch {ck.get('epoch')}, trained with anticipation {ck.get('anticipation_sec')}s)")
    action_map = ck.get("action_map")
    if action_map is None:
        raise KeyError("checkpoint missing action_map")
    pose_mean = ck.get("pose_mean", np.zeros(14, dtype=np.float32))
    pose_std  = ck.get("pose_std",  np.ones(14,  dtype=np.float32))
    pose_layers = int(os.environ.get("HDEPIC_POSE_LAYERS", "2"))
    if ck.get("anticipation_sec") and abs(float(ck["anticipation_sec"]) - anticipation_sec) > 0.5:
        print(f"  [WARNING] Eval anticipation {anticipation_sec}s differs from training {ck['anticipation_sec']}s.")

    print("\n[1] Loading annotations...")
    with open(cfg.HD_EPIC_NARR, "rb") as f:
        narr_df = pickle.load(f)
    p01_df = narr_df[narr_df["video_id"].str.startswith("P01")].copy()
    vdf, ndf, verb_map, noun_map, _, _, _ = load_label_maps(p01_df, pd)
    _, _val_df, test_df = split_data(p01_df)

    pose_loader = PoseLoader()
    val_ds = HDEpicPoseDataset(test_df, build_transforms(False), verb_map, noun_map,
                               action_map, anticipation_sec, pose_loader)
    if max_eval > 0:
        val_ds.samples = val_ds.samples[:max_eval]
    print(f"  Test samples: {len(val_ds)}", flush=True)
    if len(val_ds) == 0:
        print("  No valid samples, exiting."); return
    loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

    print("\n[2] Loading models...")
    model = build_anticipative_model(device, num_steps=ar_steps)
    D = model.embed_dim
    pose_enc = PoseEncoder(embed_dim=D, num_frames=cfg.FRAMES_PER_CLIP,
                           num_tf_layers=pose_layers).to(device)
    pose_enc.init_pose_stats(pose_mean, pose_std)
    base_probe = HDEpicProbe(D, len(vdf), len(ndf), len(action_map)).to(device)
    probe = HDEpicProbeWithPose(base_probe, pose_enc).to(device)
    probe.load_state_dict(ck["probe"], strict=True)
    probe.eval()

    v_c = defaultdict(int); v_t = defaultdict(int)
    n_c = defaultdict(int); n_t = defaultdict(int)
    a_c = defaultdict(int); a_t = defaultdict(int)
    v3 = n3 = a3 = total = 0

    print("\n[3] Running inference...", flush=True)
    with torch.no_grad():
        for clips, poses, v_ids, n_ids, a_ids in loader:
            clips = clips.to(device); poses = poses.to(device)
            feats = anticipate_features(model, clips, anticipation_sec, device)
            vl, nl, al = probe(feats, poses)
            for i in range(len(v_ids)):
                vi, ni, ai = int(v_ids[i]), int(n_ids[i]), int(a_ids[i])
                v_t[vi] += 1; n_t[ni] += 1
                if vi in vl[i].topk(3).indices.tolist(): v3 += 1
                if ni in nl[i].topk(3).indices.tolist(): n3 += 1
                if vi in vl[i].topk(5).indices.tolist(): v_c[vi] += 1
                if ni in nl[i].topk(5).indices.tolist(): n_c[ni] += 1
                if ai != -1:
                    a_t[ai] += 1
                    if ai in al[i].topk(5).indices.tolist(): a_c[ai] += 1
                    if ai in al[i].topk(3).indices.tolist(): a3 += 1
                total += 1
            if total % 40 == 0:
                print(f"  Processed {total} samples...", flush=True)

    def cmr(c, t):
        r = [c.get(k, 0) / v for k, v in t.items()]
        return float(np.mean(r) * 100) if r else 0.0

    n_act = max(sum(a_t.values()), 1)
    print(f"\n{'=' * 70}")
    print(f"HD-EPIC val set + Pose | {total} samples | anticipation {anticipation_sec}s | AR steps={ar_steps}")
    print(f"{'=' * 70}")
    print(f"  Verb   Top-3={100*v3/max(total,1):.1f}%   class-mean R@5={cmr(v_c,v_t):.1f}%")
    print(f"  Noun   Top-3={100*n3/max(total,1):.1f}%   class-mean R@5={cmr(n_c,n_t):.1f}%")
    print(f"  Action Top-3={100*a3/n_act:.1f}%   class-mean R@5={cmr(a_c,a_t):.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
