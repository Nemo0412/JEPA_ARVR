"""
HD-EPIC long-horizon anticipation training (10s ahead) + Camera Pose fusion.

Extends train_hdepic_anticipation10s.py:
  - Dataset additionally loads per-frame hand+gaze pose (zeros when no MPS data)
  - PoseEncoder maps [B,T,14] -> [B,T,D] pose tokens
  - Concatenated with ViT tokens before AttentivePooler (probe heads unchanged)

Usage:
  cd /home/ll5914/ARVR_Video/vjepa2
  python ../train_hdepic_anticipation10s_pose.py

Environment variables (same as non-pose version, plus):
  HDEPIC_ANTICIPATION_SEC=10.0
  HDEPIC_AR_STEPS=10
  HDEPIC_POSE_LAYERS=2    PoseEncoder Transformer layers (default 2)
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
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset

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

BATCH_SIZE    = 4
NUM_EPOCHS    = 10
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS   = 4

PROBE_BEST_POSE = os.path.join(cfg.SAVE_DIR, "hdepic-vitl-ar10s-pose-probe-best.pt")
PROBE_LAST_POSE = os.path.join(cfg.SAVE_DIR, "hdepic-vitl-ar10s-pose-probe-last.pt")


# ── Pose-aware probe ──────────────────────────────────────────────────
class HDEpicProbeWithPose(nn.Module):
    """
    Concatenates PoseEncoder output tokens with ViT tokens, then uses the same pooler/heads as HDEpicProbe.
    Benefit: AttentivePooler handles variable-length sequences natively.
    """
    def __init__(self, base_probe: HDEpicProbe, pose_encoder: PoseEncoder):
        super().__init__()
        self.base = base_probe
        self.pose_enc = pose_encoder

    def forward(self, visual_tokens: torch.Tensor,
                pose_seq: torch.Tensor | None = None) -> tuple:
        """
        visual_tokens: [B, N, D]
        pose_seq: [B, T, POSE_DIM] or None
        """
        if pose_seq is not None:
            pt = self.pose_enc(pose_seq)                        # [B, T, D]
            x = torch.cat([visual_tokens, pt], dim=1)           # [B, N+T, D]
        else:
            x = visual_tokens
        x = self.base.pooler(x)                                 # [B, 3, D]
        return (self.base.verb_head(x[:, 0, :]),
                self.base.noun_head(x[:, 1, :]),
                self.base.action_head(x[:, 2, :]))


# ── Dataset with pose ─────────────────────────────────────────────────
class HDEpicPoseDataset(Dataset):
    """HDEpicAnticipationDataset extended with per-sample pose tensor."""

    def __init__(self, annotations, transform, verb_map, noun_map, action_map,
                 anticipation_sec, pose_loader: PoseLoader):
        self.transform        = transform
        self.anticipation_sec = float(anticipation_sec)
        self.pose_loader      = pose_loader
        self.samples          = []
        for _, row in annotations.iterrows():
            vcs, ncs = row["verb_classes"], row["noun_classes"]
            if not isinstance(vcs, list) or not isinstance(ncs, list): continue
            if not vcs or not ncs: continue
            v_id = verb_map.get(int(vcs[0]), -1)
            n_id = noun_map.get(int(ncs[0]), -1)
            if v_id == -1 or n_id == -1: continue
            a_id = action_map.get((int(vcs[0]), int(ncs[0])), -1)
            start_sec = float(row["start_timestamp"])
            obs_end   = start_sec - self.anticipation_sec
            if obs_end < 2.0: continue
            vpath = os.path.join(cfg.VIDEO_DIR, f"{row['video_id']}.mp4")
            if not os.path.exists(vpath): continue
            self.samples.append(dict(
                video_path=vpath, video_id=row["video_id"],
                obs_end=obs_end, verb_id=v_id, noun_id=n_id, action_id=a_id,
            ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            vr    = VideoReader(s["video_path"], num_threads=1, ctx=cpu(0))
            vfps  = vr.get_avg_fps()
            step  = max(1, int(vfps / cfg.FPS))
            end_f = int(s["obs_end"] * vfps)
            idxs  = np.arange(end_f - cfg.FRAMES_PER_CLIP * step, end_f, step, dtype=np.int64)
            idxs  = np.clip(idxs, 0, len(vr) - 1)
            frames = vr.get_batch(idxs).asnumpy()
            clip  = self.transform(torch.from_numpy(frames).permute(0, 3, 1, 2))
            # pose: [T, 14]
            pose_np = self.pose_loader.get_pose_for_frames(s["video_id"], idxs, vfps)
            pose    = torch.from_numpy(pose_np)  # [T, 14]
        except Exception:
            clip = torch.zeros(3, cfg.FRAMES_PER_CLIP, cfg.IMG_SIZE, cfg.IMG_SIZE)
            pose = torch.zeros(cfg.FRAMES_PER_CLIP, 14)
        return clip, pose, s["verb_id"], s["noun_id"], s["action_id"]


def collate_fn(batch):
    clips = torch.stack([b[0] for b in batch])
    poses = torch.stack([b[1] for b in batch])  # [B, T, 14]
    v = torch.tensor([b[2] for b in batch], dtype=torch.long)
    n = torch.tensor([b[3] for b in batch], dtype=torch.long)
    a = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return clips, poses, v, n, a


# ── Eval ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, probe, loader, device, anticipation_sec):
    probe.eval()
    v_c = defaultdict(int); v_t = defaultdict(int)
    n_c = defaultdict(int); n_t = defaultdict(int)
    a_c = defaultdict(int); a_t = defaultdict(int)
    v3 = n3 = a3 = total = 0
    for clips, poses, v_ids, n_ids, a_ids in loader:
        clips = clips.to(device)
        poses = poses.to(device)
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

    def cmr(c, t):
        r = [c.get(k, 0) / v for k, v in t.items()]
        return float(np.mean(r) * 100) if r else 0.0

    n_act = max(sum(a_t.values()), 1)
    return dict(verb_top3=100*v3/max(total,1), noun_top3=100*n3/max(total,1),
                action_top3=100*a3/n_act,
                verb_r5=cmr(v_c, v_t), noun_r5=cmr(n_c, n_t), action_r5=cmr(a_c, a_t))


# ── Main ──────────────────────────────────────────────────────────────
def run(from_scratch=False):
    anticipation_sec = cfg.get_anticipation_sec()
    ar_steps         = cfg.get_ar_steps()
    pose_layers      = int(os.environ.get("HDEPIC_POSE_LAYERS", "2"))

    print("=" * 70)
    print("V-JEPA 2 — HD-EPIC Long-Horizon Anticipation Training + Camera Pose Fusion")
    print(f"Anticipation={anticipation_sec}s | AR steps={ar_steps} | PoseEncoder layers={pose_layers}")
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
    print(f"  Train : {len(train_df):4d} rows | {train_df['video_id'].nunique():2d} videos ({cfg.TRAIN_DATE})")
    print(f"  Val   : {len(val_df):4d} rows | {val_df['video_id'].nunique():2d} videos ({cfg.VAL_DATE})")
    print(f"  Test  : {len(test_df):4d} rows | {test_df['video_id'].nunique():2d} videos ({cfg.TEST_DATE})", flush=True)

    pose_loader = PoseLoader()

    train_ds = HDEpicPoseDataset(train_df, build_transforms(True),  verb_map, noun_map, action_map, anticipation_sec, pose_loader)
    val_ds   = HDEpicPoseDataset(val_df,   build_transforms(False), verb_map, noun_map, action_map, anticipation_sec, pose_loader)
    test_ds  = HDEpicPoseDataset(test_df,  build_transforms(False), verb_map, noun_map, action_map, anticipation_sec, pose_loader)
    print(f"  Valid samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}", flush=True)
    if len(train_ds) == 0:
        print("  [ERROR] No training samples — reduce HDEPIC_ANTICIPATION_SEC."); return

    # Compute per-dim mean/std from training set for pose normalization (non-zero frames only)
    print("  Computing pose normalization stats...", flush=True)
    pose_stats_sum  = np.zeros(14, dtype=np.float64)
    pose_stats_sum2 = np.zeros(14, dtype=np.float64)
    pose_n = 0
    for s in train_ds.samples[:min(len(train_ds), 300)]:
        try:
            vr   = VideoReader(s["video_path"], num_threads=1, ctx=cpu(0))
            step = max(1, int(vr.get_avg_fps() / cfg.FPS))
            end_f= int(s["obs_end"] * vr.get_avg_fps())
            idxs = np.arange(end_f - cfg.FRAMES_PER_CLIP * step, end_f, step, dtype=np.int64)
            idxs = np.clip(idxs, 0, len(vr) - 1)
            p = pose_loader.get_pose_for_frames(s["video_id"], idxs, vr.get_avg_fps())
            mask = (np.abs(p).sum(1) > 1e-6)
            if mask.any():
                pose_stats_sum  += p[mask].sum(0)
                pose_stats_sum2 += (p[mask] ** 2).sum(0)
                pose_n          += mask.sum()
        except Exception:
            pass
    if pose_n > 0:
        mean = (pose_stats_sum / pose_n).astype(np.float32)
        std  = np.sqrt(np.maximum(pose_stats_sum2 / pose_n - mean ** 2, 1e-6)).astype(np.float32)
    else:
        mean, std = np.zeros(14, dtype=np.float32), np.ones(14, dtype=np.float32)
    print(f"  Pose normalization: n={pose_n}, mean[:3]={mean[:3].round(3)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn)

    print("\n[2] Loading model (encoder + predictor frozen)...")
    model = build_anticipative_model(device, num_steps=ar_steps)
    D = model.embed_dim

    pose_enc = PoseEncoder(embed_dim=D, num_frames=cfg.FRAMES_PER_CLIP,
                           num_tf_layers=pose_layers).to(device)
    pose_enc.init_pose_stats(mean, std)

    base_probe = HDEpicProbe(D, len(vdf), len(ndf), len(action_map)).to(device)
    probe = HDEpicProbeWithPose(base_probe, pose_enc).to(device)
    n_params = sum(p.numel() for p in probe.parameters()) / 1e6
    print(f"  Probe + PoseEncoder parameters: {n_params:.1f}M")

    optimizer = optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = NUM_EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    def pack_ckpt(ep, met):
        return dict(epoch=ep, probe=probe.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    verb_names=verb_names, noun_names=noun_names,
                    action_map=action_map,
                    pose_mean=mean, pose_std=std,
                    anticipation_sec=anticipation_sec, ar_steps=ar_steps,
                    metrics=met)

    start_epoch, best_verb_r5 = 0, 0.0
    if not from_scratch and os.path.isfile(PROBE_LAST_POSE):
        print(f"\n  [resume] Loading {PROBE_LAST_POSE} ...", flush=True)
        ck = torch.load(PROBE_LAST_POSE, map_location=device, weights_only=False)
        probe.load_state_dict(ck["probe"])
        start_epoch = int(ck["epoch"])
        if ck.get("optimizer"): optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"): scheduler.load_state_dict(ck["scheduler"])
        if os.path.isfile(PROBE_BEST_POSE):
            b = torch.load(PROBE_BEST_POSE, map_location="cpu", weights_only=False)
            best_verb_r5 = float(b.get("metrics", {}).get("verb_r5", 0.0))
        if start_epoch >= NUM_EPOCHS:
            print(f"  All {NUM_EPOCHS} epochs already done."); return
        print(f"  Resuming from epoch {start_epoch+1}/{NUM_EPOCHS}, best={best_verb_r5:.1f}%")
    else:
        print("\n  Training from scratch.", flush=True)

    print(f"\n[3] Starting training ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR})...", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        probe.train()
        epoch_loss = 0.0
        t0 = time.time()
        for bi, (clips, poses, v_ids, n_ids, a_ids) in enumerate(train_loader):
            clips = clips.to(device); poses = poses.to(device)
            v_ids = v_ids.to(device); n_ids = n_ids.to(device); a_ids = a_ids.to(device)
            feats = anticipate_features(model, clips, anticipation_sec, device)
            vl, nl, al = probe(feats, poses)
            loss = criterion(vl, v_ids) + criterion(nl, n_ids)
            valid_a = a_ids >= 0
            if valid_a.any():
                loss = loss + criterion(al[valid_a], a_ids[valid_a])
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item()
            if (bi + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | step {bi+1}/{len(train_loader)} "
                      f"| loss={loss.item():.3f} | lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        print(f"\nEpoch {epoch+1} done | avg_loss={epoch_loss/max(len(train_loader),1):.3f} | {time.time()-t0:.0f}s", flush=True)
        metrics = evaluate(model, probe, val_loader, device, anticipation_sec)
        print(f"  Verb   Top-3={metrics['verb_top3']:.1f}%  R@5={metrics['verb_r5']:.1f}%", flush=True)
        print(f"  Noun   Top-3={metrics['noun_top3']:.1f}%  R@5={metrics['noun_r5']:.1f}%", flush=True)
        print(f"  Action Top-3={metrics['action_top3']:.1f}%  R@5={metrics['action_r5']:.1f}%", flush=True)

        torch.save(pack_ckpt(epoch + 1, metrics), PROBE_LAST_POSE)
        print(f"  ✓ latest → {PROBE_LAST_POSE}", flush=True)
        if metrics["verb_r5"] > best_verb_r5:
            best_verb_r5 = metrics["verb_r5"]
            torch.save(pack_ckpt(epoch + 1, metrics), PROBE_BEST_POSE)
            print(f"  ✓ best (verb R@5={best_verb_r5:.1f}%) → {PROBE_BEST_POSE}", flush=True)
        print("", flush=True)

    print("=" * 70, flush=True)
    print(f"Training complete! Best Val Verb R@5 = {best_verb_r5:.1f}%", flush=True)

    # ── Final evaluation on the held-out test set (best checkpoint) ──
    print("\n[4] Final test-set evaluation (loading best checkpoint)...", flush=True)
    best_ck = torch.load(PROBE_BEST_POSE, map_location=device, weights_only=False)
    probe.load_state_dict(best_ck["probe"])
    test_metrics = evaluate(model, probe, test_loader, device, anticipation_sec)
    print("  Test set results (best val checkpoint):", flush=True)
    print(f"    Verb   Top-3={test_metrics['verb_top3']:.1f}%  R@5={test_metrics['verb_r5']:.1f}%", flush=True)
    print(f"    Noun   Top-3={test_metrics['noun_top3']:.1f}%  R@5={test_metrics['noun_r5']:.1f}%", flush=True)
    print(f"    Action Top-3={test_metrics['action_top3']:.1f}%  R@5={test_metrics['action_r5']:.1f}%", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from-scratch", action="store_true")
    run(from_scratch=p.parse_args().from_scratch)
