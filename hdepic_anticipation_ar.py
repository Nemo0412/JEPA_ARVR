"""
HD-EPIC long-horizon action anticipation (default: 10s ahead) — V-JEPA 2 AR predictor.

Unlike train_hdepic_probe.py (discriminative, 1s ahead), this uses the true AR pathway:
  frozen ViT-L encoder -> frozen V-JEPA2 predictor (conditioned on anticipation_time,
  rolls out future latent tokens via RoPE position skipping) -> trainable probe.

Only the probe (AttentivePooler + verb/noun/action heads) is trained; encoder and
predictor are fully frozen.

Environment variables:
  HDEPIC_ANTICIPATION_SEC=10.0   anticipation horizon in seconds (default 10)
  HDEPIC_AR_STEPS=10             number of sliding-window AR steps

Weights: /scratch/ll5914/models/vjepa2/vitl.pt (official pretrain, contains
         target_encoder + predictor).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/ll5914/ARVR_Video/vjepa2")

import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader, cpu
from torch.utils.data import Dataset

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentivePooler
from evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar import (
    init_module as _init_anticipative_module,
)

# ── Paths ─────────────────────────────────────────────────────────────
ENCODER_CKPT = "/scratch/ll5914/models/vjepa2/vitl.pt"
HD_EPIC_NARR = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
HD_VERB_CSV = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_verb_classes.csv"
HD_NOUN_CSV = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_noun_classes.csv"
VIDEO_DIR = "/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos/P01"
SAVE_DIR = "/scratch/ll5914/models/vjepa2"
PROBE_BEST = os.path.join(SAVE_DIR, "hdepic-vitl-ar10s-probe-best.pt")
PROBE_LAST = os.path.join(SAVE_DIR, "hdepic-vitl-ar10s-probe-last.pt")

# ── Hyper-parameters ──────────────────────────────────────────────────
IMG_SIZE = 256
FRAMES_PER_CLIP = 32
FPS = 8
DEFAULT_ANTICIPATION_SEC = 10.0
# More steps -> shorter advance per step -> closer to true frame-by-frame rollout,
# but requires more predictor forward passes.
DEFAULT_AR_STEPS = 10

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Explicit video-ID split: 20 train / 2 val / 5 test (27 videos total, P01 only).
#
# Assignment strategy:
#   - All 12 videos from 20240203 go to train (largest single recording day).
#   - 20240202 and 20240204 are split so val/test stay small and representative.
#   - P01-20240202-110250 (the only video with extracted gaze data) is kept in val.
#   - Test uses the first 5 videos from 20240204 (chronologically earliest in that day).
#
# Train (20): 4 × 20240202  +  12 × 20240203  +  4 × 20240204
# Val   ( 2): 2 × 20240202
# Test  ( 5): 5 × 20240204
TRAIN_VIDEO_IDS: frozenset[str] = frozenset({
    # 20240202 — 4 videos
    "P01-20240202-161948",
    "P01-20240202-171220",
    "P01-20240202-175627",
    "P01-20240202-195538",
    # 20240203 — all 12 videos
    "P01-20240203-093333",
    "P01-20240203-121517",
    "P01-20240203-123350",
    "P01-20240203-130505",
    "P01-20240203-132119",
    "P01-20240203-135502",
    "P01-20240203-150506",
    "P01-20240203-152323",
    "P01-20240203-152956",
    "P01-20240203-161757",
    "P01-20240203-184045",
    "P01-20240203-184214",
    # 20240204 — 4 videos (last four chronologically)
    "P01-20240204-142301",
    "P01-20240204-145458",
    "P01-20240204-152537",
    "P01-20240204-160230",
})

VAL_VIDEO_IDS: frozenset[str] = frozenset({
    "P01-20240202-110250",   # only video with extracted gaze data — keep in val
    "P01-20240202-161354",
})

TEST_VIDEO_IDS: frozenset[str] = frozenset({
    "P01-20240204-095114",
    "P01-20240204-120411",
    "P01-20240204-121042",
    "P01-20240204-124504",
    "P01-20240204-130448",
})


def get_anticipation_sec() -> float:
    return float(os.environ.get("HDEPIC_ANTICIPATION_SEC", DEFAULT_ANTICIPATION_SEC))


def get_ar_steps() -> int:
    return int(os.environ.get("HDEPIC_AR_STEPS", DEFAULT_AR_STEPS))


# ── Video transforms ──────────────────────────────────────────────────
def build_transforms(is_train: bool):
    short_side = int(256.0 / 224 * IMG_SIZE)
    if is_train:
        return video_transforms.Compose([
            video_transforms.Resize(short_side, interpolation="bilinear"),
            video_transforms.RandomCrop(size=(IMG_SIZE, IMG_SIZE)),
            video_transforms.RandomHorizontalFlip(),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return video_transforms.Compose([
        video_transforms.Resize(short_side, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────
class HDEpicAnticipationDataset(Dataset):
    """
    Each sample: the observation window ends anticipation_sec seconds before the
    action onset, and covers FRAMES_PER_CLIP frames sampled at FPS.
    Actions that start too early (obs_end < 2.0s) are filtered out.
    """

    def __init__(self, annotations, transform, verb_map, noun_map, action_map, anticipation_sec):
        self.transform = transform
        self.anticipation_sec = float(anticipation_sec)
        self.samples = []
        for _, row in annotations.iterrows():
            vcs = row["verb_classes"]
            ncs = row["noun_classes"]
            if not isinstance(vcs, list) or not isinstance(ncs, list):
                continue
            if len(vcs) == 0 or len(ncs) == 0:
                continue
            v_id = verb_map.get(int(vcs[0]), -1)
            n_id = noun_map.get(int(ncs[0]), -1)
            if v_id == -1 or n_id == -1:
                continue
            a_id = action_map.get((int(vcs[0]), int(ncs[0])), -1)
            start_sec = float(row["start_timestamp"])
            obs_end = start_sec - self.anticipation_sec
            if obs_end < 2.0:
                continue
            video_path = os.path.join(VIDEO_DIR, f"{row['video_id']}.mp4")
            if not os.path.exists(video_path):
                continue
            self.samples.append({
                "video_path": video_path,
                "obs_end": obs_end,
                "verb_id": v_id,
                "noun_id": n_id,
                "action_id": a_id,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            vr = VideoReader(s["video_path"], num_threads=1, ctx=cpu(0))
            vfps = vr.get_avg_fps()
            frame_step = max(1, int(vfps / FPS))
            end_frame = int(s["obs_end"] * vfps)
            start_frame = end_frame - FRAMES_PER_CLIP * frame_step
            indices = np.arange(start_frame, end_frame, frame_step, dtype=np.int64)
            indices = np.clip(indices, 0, len(vr) - 1)
            frames = vr.get_batch(indices).asnumpy()
            clip = self.transform(torch.from_numpy(frames).permute(0, 3, 1, 2))
        except Exception:
            clip = torch.zeros(3, FRAMES_PER_CLIP, IMG_SIZE, IMG_SIZE)
        return clip, s["verb_id"], s["noun_id"], s["action_id"]


# ── Model ─────────────────────────────────────────────────────────────
class HDEpicProbe(nn.Module):
    """HD-EPIC classification head: AttentivePooler (3 queries) -> verb/noun/action."""

    def __init__(self, embed_dim, num_verbs, num_nouns, num_actions):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=3,
            embed_dim=embed_dim,
            num_heads=16,
            depth=4,
            use_activation_checkpointing=False,
        )
        self.verb_head = nn.Linear(embed_dim, num_verbs)
        self.noun_head = nn.Linear(embed_dim, num_nouns)
        self.action_head = nn.Linear(embed_dim, num_actions)

    def forward(self, x):
        x = self.pooler(x)
        return self.verb_head(x[:, 0, :]), self.noun_head(x[:, 1, :]), self.action_head(x[:, 2, :])


def build_anticipative_model(device, num_steps=None):
    """Build the frozen encoder+predictor AnticipativeWrapper (official modelcustom)."""
    if num_steps is None:
        num_steps = get_ar_steps()

    model_kwargs = {
        "use_v2_1": False,
        "encoder": {
            "model_name": "vit_large",
            "checkpoint_key": "target_encoder",
            "tubelet_size": 2,
            "patch_size": 16,
            "uniform_power": True,
            "use_rope": True,
        },
        "predictor": {
            "model_name": "vit_predictor",
            "checkpoint_key": "predictor",
            "num_frames": 64,
            "depth": 12,
            "num_heads": 12,
            "predictor_embed_dim": 384,
            "num_mask_tokens": 10,
            "uniform_power": True,
            "use_mask_tokens": True,
            "use_sdpa": True,
            "use_silu": False,
            "wide_silu": False,
            "use_rope": True,
        },
    }
    wrapper_kwargs = {
        "no_predictor": False,
        "num_output_frames": 2,
        "num_steps": int(num_steps),
    }
    model = _init_anticipative_module(
        frames_per_clip=FRAMES_PER_CLIP,
        frames_per_second=FPS,
        resolution=IMG_SIZE,
        checkpoint=ENCODER_CKPT,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def anticipate_features(model, clips, anticipation_sec, device, num_steps=None):
    """
    Rebased sliding-window AR rollout to approximate anticipation_sec.

    The predictor's position capacity is limited (num_frames=64 -> 32 temporal slots;
    encoder occupies 16 slots, leaving ~16 slots = ~4s for future prediction).
    A direct jump to 10s would exceed the position budget, so we use a sliding window:
      predict a short future segment -> append predicted tokens to context ->
      drop the oldest tokens (slide window) -> repeat until total advance reaches
      the target horizon. Returns the context window at the target horizon.

    clips: [B, C, T, H, W] -> future latent tokens [B, N, D] (all under no_grad)
    """
    if num_steps is None:
        num_steps = get_ar_steps()

    enc = model.encoder
    prd = model.predictor
    grid = int(model.grid_size)            # 16
    tube = int(model.tubelet_size)         # 2
    fps = int(model.frames_per_second)     # 8
    gp = grid * grid                       # tokens per temporal slot (256)

    x = enc(clips)                         # [B, N, D], D == embed_dim
    B, N, D = x.shape
    W = max(1, N // gp)                    # context window size in temporal slots (16)

    pred_num_frames = getattr(prd, "num_frames", 64)
    cap_slabs = pred_num_frames // tube    # total predictor capacity in slots (32)
    max_adv = max(1, cap_slabs - W)        # max advance per step in slots (16)

    # Total slots to advance (1 slot = tube/fps seconds)
    total_slabs = max(1, int(round(float(anticipation_sec) * fps / tube)))
    # Ensure each step advances at most max_adv slots
    K = max(int(num_steps), 1)
    K = max(K, (total_slabs + max_adv - 1) // max_adv)

    ctx = x                                                  # [B, W*gp, D]
    ctx_pos = torch.arange(W * gp, device=device).unsqueeze(0).repeat(B, 1)

    advanced = 0
    for k in range(1, K + 1):
        target = int(round(total_slabs * k / K))
        adv = min(target - advanced, max_adv)
        if adv <= 0:
            continue
        n_pred = gp * adv
        tgt_pos = torch.arange(n_pred, device=device).unsqueeze(0).repeat(B, 1) + (W * gp)
        pred = prd(ctx, masks_x=ctx_pos, masks_y=tgt_pos)    # [B, n_pred, D]
        if isinstance(pred, tuple):
            pred = pred[0]
        full = torch.cat([ctx, pred], dim=1)                 # [B, W*gp + n_pred, D]
        ctx = full[:, -(W * gp):, :]                         # slide: keep last W slots
        advanced += adv

    return ctx                                               # [B, W*gp, D] at target horizon


# ── Labels / data split ───────────────────────────────────────────────
def load_label_maps(p01_df, pd_module):
    vdf = pd_module.read_csv(HD_VERB_CSV)
    ndf = pd_module.read_csv(HD_NOUN_CSV)
    verb_map = {int(r["id"]): int(r["id"]) for _, r in vdf.iterrows()}
    noun_map = {int(r["id"]): int(r["id"]) for _, r in ndf.iterrows()}
    verb_names = {int(r["id"]): r["key"] for _, r in vdf.iterrows()}
    noun_names = {int(r["id"]): r["key"] for _, r in ndf.iterrows()}

    pairs = set()
    for _, row in p01_df.iterrows():
        vcs, ncs = row["verb_classes"], row["noun_classes"]
        if isinstance(vcs, list) and isinstance(ncs, list) and vcs and ncs:
            pairs.add((int(vcs[0]), int(ncs[0])))
    action_map = {k: i for i, k in enumerate(pairs)}
    return vdf, ndf, verb_map, noun_map, action_map, verb_names, noun_names


def split_data(p01_df):
    """Return (train_df, val_df, test_df) using the explicit video-ID split.

    Train : TRAIN_VIDEO_IDS  (20 videos)
    Val   : VAL_VIDEO_IDS    ( 2 videos) — early stopping / hyperparameter tuning
    Test  : TEST_VIDEO_IDS   ( 5 videos) — held-out final evaluation
    """
    train_df = p01_df[p01_df["video_id"].isin(TRAIN_VIDEO_IDS)]
    val_df   = p01_df[p01_df["video_id"].isin(VAL_VIDEO_IDS)]
    test_df  = p01_df[p01_df["video_id"].isin(TEST_VIDEO_IDS)]
    return train_df, val_df, test_df


def split_train_val(p01_df):
    """Backward-compatible wrapper — returns only (train_df, val_df)."""
    train_df, val_df, _ = split_data(p01_df)
    return train_df, val_df
