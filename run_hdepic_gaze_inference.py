"""
HD-EPIC: use MPS gaze (general_eye_gaze.csv) + video time alignment to build heatmaps,
then multiplicatively gate ViT tokens before the HD-EPIC probe.

Comparison: baseline (no gating) vs gaze-gated.
Requires a trained probe: hdepic-vitl-probe-last.pt (or adjust path).

Usage:
  cd /home/ll5914/ARVR_Video/vjepa2
  python ../run_hdepic_gaze_inference.py

Environment variables (optional):
  HDEPIC_GAZE_EVAL_MAX=N  evaluate first N samples (debug/memory; 0 = all)
  HDEPIC_GAZE_BATCH=B     DataLoader batch size (default 2)

Defaults to the one video with extracted gaze data: P01-20240202-110250.
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

# vjepa2 package (AttentivePooler, load_encoder); script lives in ARVR_Video root
sys.path.insert(0, "/home/ll5914/ARVR_Video/vjepa2")

import train_hdepic_probe as tcfg

from src.models.attentive_pooler import AttentivePooler
from hdepic_gaze_mask import (
    motion_saliency_tubes,
    patch_importance_from_maps,
    gate_encoder_tokens,
    build_gaze_maps_for_indices,
)

# ── Gaze files (first downloaded video) ─────────────────────────────
VIDEO_ID_FOR_GAZE = "P01-20240202-110250"
SYNC_CSV = f"/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos/P01/{VIDEO_ID_FOR_GAZE}_mp4_to_vrs_time_ns.csv"
GAZE_EXTRACT_ROOT = Path("/scratch/ll5914/datasets/HD-EPIC/_gaze_extract/mps_P01-20240202-110250_vrs/eye_gaze")
GAZE_CSV = GAZE_EXTRACT_ROOT / "general_eye_gaze.csv"


class GazeInferenceDataset(Dataset):
    """Same sampling as training; additionally returns decoded frame indices for gaze alignment."""
    def __init__(self, ann_df: pd.DataFrame, transform_fn, verb_map, noun_map, action_map):
        self.transform_fn = transform_fn
        self.samples = []
        for _, row in ann_df.iterrows():
            vcs = row["verb_classes"]
            ncs = row["noun_classes"]
            if not isinstance(vcs, list) or not isinstance(ncs, list):
                continue
            if not vcs or not ncs:
                continue
            v_id = verb_map.get(int(vcs[0]), -1)
            n_id = noun_map.get(int(ncs[0]), -1)
            if v_id == -1 or n_id == -1:
                continue
            a_id = action_map.get((int(vcs[0]), int(ncs[0])), -1)
            start_sec = float(row["start_timestamp"])
            obs_end = start_sec - tcfg.ANTICIPATION_SEC
            if obs_end < 2.0:
                continue
            vpath = os.path.join(tcfg.VIDEO_DIR, f"{row['video_id']}.mp4")
            if not os.path.isfile(vpath):
                continue
            self.samples.append(
                dict(video_path=vpath, obs_end=obs_end, verb_id=v_id, noun_id=n_id, action_id=a_id)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        vr = VideoReader(s["video_path"], num_threads=1, ctx=cpu(0))
        vfps = vr.get_avg_fps()
        frame_step = max(1, int(vfps / tcfg.FPS))
        end_f = int(s["obs_end"] * vfps)
        start_f = end_f - tcfg.FRAMES_PER_CLIP * frame_step
        indices = np.arange(start_f, end_f, frame_step, dtype=np.int64)
        indices = np.clip(indices, 0, len(vr) - 1)
        frames = vr.get_batch(indices).asnumpy()
        H0, W0 = frames.shape[1], frames.shape[2]
        clip = self.transform_fn(torch.from_numpy(frames).permute(0, 3, 1, 2))
        meta = dict(
            frame_indices=np.array(indices.copy(), dtype=np.int64),
            vfps=float(vfps),
            H0=int(H0),
            W0=int(W0),
        )
        return (
            clip,
            int(s["verb_id"]),
            int(s["noun_id"]),
            int(s["action_id"]),
            meta,
        )


def _collate(batch):
    clips = torch.stack([b[0] for b in batch])
    v = torch.tensor([b[1] for b in batch], dtype=torch.long)
    n = torch.tensor([b[2] for b in batch], dtype=torch.long)
    a = torch.tensor([b[3] for b in batch], dtype=torch.long)
    metas = [b[4] for b in batch]
    return clips, v, n, a, metas


def update_metrics(metrics_state, v_logits, n_logits, a_logits, vi, ni, ai):
    for i in range(v_logits.shape[0]):
        vii, nii, aii = int(vi[i]), int(ni[i]), int(ai[i])
        metrics_state["total"] += 1
        metrics_state["verb_t"][vii] += 1
        metrics_state["noun_t"][nii] += 1
        if vii in v_logits[i].topk(3).indices.tolist():
            metrics_state["v3"] += 1
        if nii in n_logits[i].topk(3).indices.tolist():
            metrics_state["n3"] += 1
        if vii in v_logits[i].topk(5).indices.tolist():
            metrics_state["v_c"][vii] += 1
        if nii in n_logits[i].topk(5).indices.tolist():
            metrics_state["n_c"][nii] += 1
        if aii >= 0:
            metrics_state["a_t"][aii] += 1
            if aii in a_logits[i].topk(5).indices.tolist():
                metrics_state["a_c"][aii] += 1
            if aii in a_logits[i].topk(3).indices.tolist():
                metrics_state["a3"] += 1
                metrics_state["a3_denom"] += 1


def class_mean_from_state(c_dic, t_dic):
    r = [c_dic.get(k, 0) / v for k, v in t_dic.items()]
    return float(np.mean(r) * 100) if r else 0.0


def summarize(state):
    t = max(state["total"], 1)
    return dict(
        verb_top3=100 * state["v3"] / t,
        noun_top3=100 * state["n3"] / t,
        verb_r5=class_mean_from_state(state["v_c"], state["verb_t"]),
        noun_r5=class_mean_from_state(state["n_c"], state["noun_t"]),
        action_top3=(100 * state["a3"] / max(state["a3_denom"], 1)) if state["a3_denom"] else 0.0,
        action_r5=class_mean_from_state(state["a_c"], state["a_t"]),
    )


def fresh_state():
    return dict(
        total=0,
        v3=0,
        n3=0,
        a3=0,
        a3_denom=0,
        verb_t=defaultdict(int),
        noun_t=defaultdict(int),
        a_t=defaultdict(int),
        v_c=defaultdict(int),
        n_c=defaultdict(int),
        a_c=defaultdict(int),
    )


class HDEpicProbe(torch.nn.Module):
    """Replicated from train_hdepic_probe to avoid importing the full training module."""
    def __init__(self, embed_dim, num_verbs, num_nouns, num_actions):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=3,
            embed_dim=embed_dim,
            num_heads=16,
            depth=4,
            use_activation_checkpointing=False,
        )
        self.verb_head = torch.nn.Linear(embed_dim, num_verbs)
        self.noun_head = torch.nn.Linear(embed_dim, num_nouns)
        self.action_head = torch.nn.Linear(embed_dim, num_actions)

    def forward(self, x):
        x = self.pooler(x)
        return self.verb_head(x[:, 0, :]), self.noun_head(x[:, 1, :]), self.action_head(x[:, 2, :])


def run():
    if not os.path.isfile(SYNC_CSV) or not GAZE_CSV.is_file():
        print("Missing time-sync or gaze CSV, please check:")
        print(" ", SYNC_CSV)
        print(" ", GAZE_CSV)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    sync_df = pd.read_csv(SYNC_CSV)
    gaze_df = pd.read_csv(GAZE_CSV)

    with open(tcfg.HD_EPIC_NARR, "rb") as f:
        narr = pickle.load(f)
    p01 = narr[narr["video_id"].str.startswith("P01")].copy()

    vdf = pd.read_csv(tcfg.HD_VERB_CSV)
    ndf = pd.read_csv(tcfg.HD_NOUN_CSV)
    verb_map = {int(r["id"]): int(r["id"]) for _, r in vdf.iterrows()}
    noun_map = {int(r["id"]): int(r["id"]) for _, r in ndf.iterrows()}

    ckpt_path = (
        Path(tcfg.PROBE_LAST) if Path(tcfg.PROBE_LAST).is_file()
        else Path(tcfg.PROBE_BEST)
    )
    if not ckpt_path.is_file():
        print("Probe checkpoint not found:", tcfg.PROBE_LAST, "or", tcfg.PROBE_BEST)
        return
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print("Checkpoint:", ckpt_path, "epoch", ck.get("epoch"))
    action_map_ck = ck.get("action_map")
    if action_map_ck is None:
        raise KeyError("checkpoint missing action_map")

    val_df = p01[p01["video_id"].isin(tcfg.VAL_VIDEO_IDS)]
    subset = val_df[val_df["video_id"] == VIDEO_ID_FOR_GAZE]
    print(f"Evaluating video {VIDEO_ID_FOR_GAZE}, samples (filtered): {len(subset)}")

    ds = GazeInferenceDataset(
        subset,
        tcfg.build_transforms(False),
        verb_map,
        noun_map,
        action_map_ck,
    )
    if len(ds) == 0:
        print("No valid samples, exiting.")
        return

    max_eval = int(os.environ.get("HDEPIC_GAZE_EVAL_MAX", "0"))
    if max_eval > 0:
        ds.samples = ds.samples[:max_eval]
        print(f"HDEPIC_GAZE_EVAL_MAX={max_eval}: evaluating first {len(ds.samples)} samples only.")

    bs = int(os.environ.get("HDEPIC_GAZE_BATCH", "2"))

    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=_collate)

    # Load encoder — restore LoRA adapters from checkpoint if they were used during training.
    lora_rank = int(ck.get("lora_rank", 0))
    encoder = tcfg.load_encoder(device, lora_rank=lora_rank)
    if lora_rank > 0 and "encoder_lora" in ck:
        missing, unexpected = encoder.load_state_dict(ck["encoder_lora"], strict=False)
        print(f"  Loaded encoder LoRA weights (rank={lora_rank}, "
              f"missing={len(missing)}, unexpected={len(unexpected)})")
    encoder.eval()

    amap = action_map_ck
    probe = HDEpicProbe(
        embed_dim=encoder.embed_dim,
        num_verbs=len(vdf),
        num_nouns=len(ndf),
        num_actions=len(amap),
    ).to(device)
    probe.load_state_dict(ck["probe"], strict=True)
    probe.eval()

    st_base = fresh_state()
    st_gaze = fresh_state()

    tubelet_size = 2
    patch_sz = 16

    with torch.no_grad():
        for clips, v_ids, n_ids, a_ids, metas in loader:
            clips = clips.to(device)
            b = clips.shape[0]
            feats0 = encoder(clips)

            vb, nb, ab = probe(feats0)
            update_metrics(st_base, vb, nb, ab, v_ids, n_ids, a_ids)

            imp_list = []
            for bi in range(b):
                m = metas[bi]
                gmap = build_gaze_maps_for_indices(
                    m["frame_indices"],
                    m["vfps"],
                    sync_df,
                    gaze_df,
                    m["H0"],
                    m["W0"],
                    out_size=tcfg.IMG_SIZE,
                    sigma_px=40.0,
                )
                g = torch.from_numpy(gmap).to(device).unsqueeze(0).float()  # [1,T,H,W]
                gray = clips[bi : bi + 1].mean(dim=1)
                mot = motion_saliency_tubes(gray, tubelet_size=tubelet_size)
                Tg = g.shape[1]
                Tu = Tg - (Tg % tubelet_size)
                g = g[:, :Tu]
                g_t = g.view(1, Tu // tubelet_size, tubelet_size, g.shape[2], g.shape[3]).mean(dim=2)
                mot_norm = mot / (mot.amax(dim=(1, 2, 3), keepdim=True)[0].clamp(min=1e-6))
                g_t = g_t + 0.15 * mot_norm
                imp_list.append(patch_importance_from_maps(g_t, spatial_size_hw=(tcfg.IMG_SIZE, tcfg.IMG_SIZE), patch_size=patch_sz))

            imp_bn = torch.cat(imp_list, dim=0)
            fg = gate_encoder_tokens(feats0, imp_bn, gamma=0.7)
            vg, ng, ag = probe(fg)
            update_metrics(st_gaze, vg, ng, ag, v_ids, n_ids, a_ids)

    print("\n====== Baseline (no gaze gating) ======")
    for k, v in summarize(st_base).items():
        print(f"  {k}: {v:.2f}%")
    print("\n====== Gaze + motion heatmap -> token gate ======")
    for k, v in summarize(st_gaze).items():
        print(f"  {k}: {v:.2f}%")


if __name__ == "__main__":
    run()
