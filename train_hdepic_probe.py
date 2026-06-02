"""
HD-EPIC Probe Fine-tuning on V-JEPA 2
======================================
Freeze V-JEPA 2 ViT-L encoder and train a new AttentiveClassifier
on HD-EPIC's own verb/noun vocabulary (106 verbs, 303 nouns).

Data split (HD-EPIC P01, 27 videos total):
  Train: video_id contains '20240203' (12 videos recorded that day)
  Val:   remaining 15 videos (20240202 + 20240204)
  (date-based split; same day never appears in both train and val)

Checkpoints:
  hdepic-vitl-probe-last.pt  — overwritten after every epoch
  hdepic-vitl-probe-best.pt  — saved when val Verb Recall@5 improves

Usage:
  cd /home/ll5914/ARVR_Video/vjepa2
  python ../train_hdepic_probe.py           # resumes from last.pt if present
  python ../train_hdepic_probe.py --from-scratch
"""

import sys, os, pickle, time, argparse
sys.path.insert(0, "/home/ll5914/ARVR_Video/vjepa2")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from decord import VideoReader, cpu

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentivePooler
from src.models.vision_transformer import vit_large_rope

# ── Paths ───────────────────────────────────────────────────────────
ENCODER_CKPT  = "/scratch/ll5914/models/vjepa2/vitl.pt"
HD_EPIC_NARR  = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
HD_VERB_CSV   = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_verb_classes.csv"
HD_NOUN_CSV   = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_noun_classes.csv"
VIDEO_DIR     = "/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos/P01"
SAVE_DIR      = "/scratch/ll5914/models/vjepa2"
PROBE_BEST    = os.path.join(SAVE_DIR, "hdepic-vitl-probe-best.pt")
PROBE_LAST    = os.path.join(SAVE_DIR, "hdepic-vitl-probe-last.pt")

# ── Hyper-parameters ────────────────────────────────────────────────
IMG_SIZE         = 256
FRAMES_PER_CLIP  = 32
FPS              = 8
ANTICIPATION_SEC = 1.0
BATCH_SIZE       = 8
NUM_EPOCHS       = 10
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
WARMUP_EPOCHS    = 2
NUM_WORKERS      = 4

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# 20240203 videos used for training; others for validation
TRAIN_DATE = "20240203"


# ── Dataset ─────────────────────────────────────────────────────────
class HDEpicDataset(Dataset):
    def __init__(self, annotations, video_dir, transform, verb_map, noun_map, action_map, is_train=True):
        self.video_dir  = video_dir
        self.transform  = transform
        self.verb_map   = verb_map    # orig_id → mapped_id
        self.noun_map   = noun_map
        self.action_map = action_map  # (v,n) → mapped_id
        self.is_train   = is_train

        # Use first verb/noun class as primary label (multi-label annotations)
        self.samples = []
        for _, row in annotations.iterrows():
            vcs = row['verb_classes']
            ncs = row['noun_classes']
            if not isinstance(vcs, list) or not isinstance(ncs, list):
                continue
            if len(vcs) == 0 or len(ncs) == 0:
                continue
            v_id = verb_map.get(int(vcs[0]), -1)
            n_id = noun_map.get(int(ncs[0]), -1)
            if v_id == -1 or n_id == -1:
                continue
            a_id = action_map.get((int(vcs[0]), int(ncs[0])), -1)
            start_sec = float(row['start_timestamp'])
            obs_end   = start_sec - ANTICIPATION_SEC
            if obs_end < 2.0:
                continue
            video_path = os.path.join(video_dir, f"{row['video_id']}.mp4")
            if not os.path.exists(video_path):
                continue
            self.samples.append({
                'video_path': video_path,
                'obs_end':    obs_end,
                'verb_id':    v_id,
                'noun_id':    n_id,
                'action_id':  a_id,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            vr   = VideoReader(s['video_path'], num_threads=1, ctx=cpu(0))
            vfps = vr.get_avg_fps()
            frame_step  = max(1, int(vfps / FPS))
            end_frame   = int(s['obs_end'] * vfps)
            start_frame = end_frame - FRAMES_PER_CLIP * frame_step
            indices = np.arange(start_frame, end_frame, frame_step, dtype=np.int64)
            indices = np.clip(indices, 0, len(vr) - 1)
            frames  = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
            clip    = self.transform(torch.from_numpy(frames).permute(0, 3, 1, 2))
        except Exception:
            # Fall back to zero clip on decode error
            clip = torch.zeros(3, FRAMES_PER_CLIP, IMG_SIZE, IMG_SIZE)

        return clip, s['verb_id'], s['noun_id'], s['action_id']


def build_transforms(is_train):
    short_side = int(256.0 / 224 * IMG_SIZE)
    if is_train:
        return video_transforms.Compose([
            video_transforms.Resize(short_side, interpolation="bilinear"),
            video_transforms.RandomCrop(size=(IMG_SIZE, IMG_SIZE)),
            video_transforms.RandomHorizontalFlip(),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return video_transforms.Compose([
            video_transforms.Resize(short_side, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── Model ───────────────────────────────────────────────────────────
class HDEpicProbe(nn.Module):
    """Standalone HD-EPIC classification head (num_queries=3: verb, noun, action)"""
    def __init__(self, embed_dim, num_verbs, num_nouns, num_actions):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=3,
            embed_dim=embed_dim,
            num_heads=16,
            depth=4,
            use_activation_checkpointing=False,
        )
        self.verb_head   = nn.Linear(embed_dim, num_verbs)
        self.noun_head   = nn.Linear(embed_dim, num_nouns)
        self.action_head = nn.Linear(embed_dim, num_actions)

    def forward(self, x):
        x = self.pooler(x)                     # [B, 3, D]
        v = self.verb_head(x[:, 0, :])         # [B, num_verbs]
        n = self.noun_head(x[:, 1, :])         # [B, num_nouns]
        a = self.action_head(x[:, 2, :])       # [B, num_actions]
        return v, n, a


def load_encoder(device):
    print("  Loading ViT-L encoder (frozen)...")
    model = vit_large_rope(
        img_size=(IMG_SIZE, IMG_SIZE),
        num_frames=FRAMES_PER_CLIP,
        tubelet_size=2, patch_size=16,
        uniform_power=True,
    )
    ckpt  = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=True)
    state = ckpt.get("target_encoder", ckpt.get("encoder", ckpt))
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


# ── Evaluation ──────────────────────────────────────────────────────
def evaluate(encoder, probe, loader, device, num_verbs, num_nouns, num_actions):
    probe.eval()
    v_c = defaultdict(int); v_t = defaultdict(int)
    n_c = defaultdict(int); n_t = defaultdict(int)
    a_c = defaultdict(int); a_t = defaultdict(int)
    total = 0

    with torch.no_grad():
        for clips, v_ids, n_ids, a_ids in loader:
            clips = clips.to(device)
            feats = encoder(clips)
            v_logits, n_logits, a_logits = probe(feats)

            for i in range(len(v_ids)):
                vi, ni, ai = int(v_ids[i]), int(n_ids[i]), int(a_ids[i])
                v_t[vi] += 1; n_t[ni] += 1
                if vi in v_logits[i].topk(5).indices.tolist(): v_c[vi] += 1
                if ni in n_logits[i].topk(5).indices.tolist(): n_c[ni] += 1
                if ai != -1:
                    a_t[ai] += 1
                    if ai in a_logits[i].topk(5).indices.tolist(): a_c[ai] += 1
                total += 1

    def cmr(c, t):
        r = [c.get(k, 0) / v for k, v in t.items()]
        return np.mean(r) * 100 if r else 0.0

    # Top-3 accuracy
    v_top3 = noun_top3 = a_top3 = 0
    with torch.no_grad():
        for clips, v_ids, n_ids, a_ids in loader:
            clips = clips.to(device)
            feats = encoder(clips)
            v_logits, n_logits, a_logits = probe(feats)
            for i in range(len(v_ids)):
                vi, ni, ai = int(v_ids[i]), int(n_ids[i]), int(a_ids[i])
                if vi in v_logits[i].topk(3).indices.tolist(): v_top3 += 1
                if ni in n_logits[i].topk(3).indices.tolist(): noun_top3 += 1
                if ai != -1 and ai in a_logits[i].topk(3).indices.tolist(): a_top3 += 1

    n_act = sum(a_t.values())
    return {
        "verb_top3":   100 * v_top3   / max(total, 1),
        "noun_top3":   100 * noun_top3 / max(total, 1),
        "action_top3": 100 * a_top3   / max(n_act, 1),
        "verb_r5":     cmr(v_c, v_t),
        "noun_r5":     cmr(n_c, n_t),
        "action_r5":   cmr(a_c, a_t),
    }


# ── Main training loop ──────────────────────────────────────────────
def run(from_scratch=False):
    print("=" * 65)
    print("V-JEPA 2 — HD-EPIC Probe Fine-tuning")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Load annotations
    print("\n[1] Loading annotations...")
    with open(HD_EPIC_NARR, "rb") as f:
        narr_df = pickle.load(f)
    p01_df = narr_df[narr_df['video_id'].str.startswith('P01')].copy()

    vdf = pd.read_csv(HD_VERB_CSV)
    ndf = pd.read_csv(HD_NOUN_CSV)
    # id column in HD-EPIC verb_classes.csv is the class id directly
    verb_map   = {int(r['id']): int(r['id']) for _, r in vdf.iterrows()}   # id → id (identity)
    noun_map   = {int(r['id']): int(r['id']) for _, r in ndf.iterrows()}
    verb_names = {int(r['id']): r['key'] for _, r in vdf.iterrows()}
    noun_names = {int(r['id']): r['key'] for _, r in ndf.iterrows()}

    # Action classes: unique (verb_class, noun_class) pairs in P01
    pairs = set()
    for _, row in p01_df.iterrows():
        vcs = row['verb_classes']; ncs = row['noun_classes']
        if isinstance(vcs, list) and isinstance(ncs, list) and vcs and ncs:
            pairs.add((int(vcs[0]), int(ncs[0])))
    action_map = {k: i for i, k in enumerate(pairs)}
    print(f"  verbs={len(vdf)}, nouns={len(ndf)}, actions={len(action_map)}")

    # Train / val split
    train_df = p01_df[p01_df['video_id'].str.contains(TRAIN_DATE)]
    val_df   = p01_df[~p01_df['video_id'].str.contains(TRAIN_DATE)]
    train_vids = sorted(train_df["video_id"].unique())
    val_vids   = sorted(val_df["video_id"].unique())
    print(f"  Train annotations: {len(train_df)} rows | {len(train_vids)} videos (date {TRAIN_DATE})")
    print(f"  Val annotations: {len(val_df)} rows | {len(val_vids)} videos (other dates, used as val)", flush=True)

    # Build datasets
    train_ds = HDEpicDataset(train_df, VIDEO_DIR, build_transforms(True),  verb_map, noun_map, action_map, is_train=True)
    val_ds   = HDEpicDataset(val_df,   VIDEO_DIR, build_transforms(False), verb_map, noun_map, action_map, is_train=False)
    print(f"  Valid train samples: {len(train_ds)}, val samples: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Load models
    print("\n[2] Loading models...")
    encoder = load_encoder(device)
    probe   = HDEpicProbe(
        embed_dim=encoder.embed_dim,
        num_verbs=len(vdf),
        num_nouns=len(ndf),
        num_actions=len(action_map),
    ).to(device)
    total_params = sum(p.numel() for p in probe.parameters()) / 1e6
    print(f"  Probe parameters: {total_params:.1f}M")

    # Optimizer + LR schedule
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
            "train_video_ids": sorted(train_df["video_id"].unique().tolist()),
            "val_video_ids": sorted(val_df["video_id"].unique().tolist()),
        }

    # -------- Resume training --------
    start_epoch = 0
    best_verb_r5 = 0.0
    resume_path = None
    if not from_scratch and os.path.isfile(PROBE_LAST):
        resume_path = PROBE_LAST

    if resume_path:
        print(f"\n  [resume] Loading {resume_path} ...", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        probe.load_state_dict(ckpt["probe"])
        start_epoch = int(ckpt["epoch"])
        if start_epoch >= NUM_EPOCHS:
            print(f"  All {NUM_EPOCHS} epochs already done (ckpt epoch={start_epoch}).", flush=True)
            print("=" * 65, flush=True)
            return

        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"    Restored optimizer", flush=True)
        else:
            print(f"    [WARN] No optimizer in ckpt; using fresh optimizer (slight momentum discontinuity)", flush=True)

        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"    Restored scheduler", flush=True)
        else:
            steps_done = start_epoch * len(train_loader)
            print(f"    [WARN] No scheduler in ckpt; fast-forwarding LR schedule by {steps_done} steps", flush=True)
            for _ in range(steps_done):
                scheduler.step()

        if os.path.isfile(PROBE_BEST):
            b = torch.load(PROBE_BEST, map_location="cpu", weights_only=False)
            if b.get("metrics") and "verb_r5" in b["metrics"]:
                best_verb_r5 = float(b["metrics"]["verb_r5"])
                print(f"    Historical best verb R@5 = {best_verb_r5:.1f}% (from best.pt)", flush=True)

        print(f"  [resume] Completed {start_epoch} epochs, resuming from epoch {start_epoch + 1}/{NUM_EPOCHS}", flush=True)
    else:
        if from_scratch:
            print("\n  [train] --from-scratch: training from scratch", flush=True)
        else:
            print("\n  [train] No last.pt found, training from scratch", flush=True)

    print(f"\n[3] Starting training ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR})...")
    print(f"    Saved every epoch: {PROBE_LAST}", flush=True)
    print(f"    Saved on best Verb R@5: {PROBE_BEST}", flush=True)
    print("=" * 65, flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        probe.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (clips, v_ids, n_ids, a_ids) in enumerate(train_loader):
            clips  = clips.to(device)
            v_ids  = v_ids.to(device)
            n_ids  = n_ids.to(device)

            with torch.no_grad():
                feats = encoder(clips)

            a_ids = a_ids.to(device)
            v_logits, n_logits, a_logits = probe(feats)
            loss = criterion(v_logits, v_ids) + criterion(n_logits, n_ids)
            valid_a = (a_ids >= 0)
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

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} done | avg_loss={avg_loss:.3f} | {elapsed:.0f}s", flush=True)

        # Validate after each epoch
        metrics = evaluate(encoder, probe, val_loader, device,
                           len(vdf), len(ndf), len(action_map))
        print(f"  Validation:", flush=True)
        print(f"    Verb  Top-3={metrics['verb_top3']:.1f}%  Recall@5={metrics['verb_r5']:.1f}%", flush=True)
        print(f"    Noun  Top-3={metrics['noun_top3']:.1f}%  Recall@5={metrics['noun_r5']:.1f}%", flush=True)
        print(f"    Action Top-3={metrics['action_top3']:.1f}%  Recall@5={metrics['action_r5']:.1f}%", flush=True)

        torch.save(pack_ckpt(epoch + 1, metrics), PROBE_LAST)
        print(f"  ✓ Saved latest -> {PROBE_LAST}", flush=True)

        if metrics["verb_r5"] > best_verb_r5:
            best_verb_r5 = metrics["verb_r5"]
            torch.save(pack_ckpt(epoch + 1, metrics), PROBE_BEST)
            print(f"  ✓ Saved best (verb R@5={best_verb_r5:.1f}%) -> {PROBE_BEST}", flush=True)

        probe.train()
        print("", flush=True)

    print("=" * 65, flush=True)
    print(f"Training complete! Best Verb Recall@5 = {best_verb_r5:.1f}%", flush=True)
    print(f"Latest: {PROBE_LAST}", flush=True)
    print(f"Best:   {PROBE_BEST}", flush=True)


if __name__ == "__main__":
    _p = argparse.ArgumentParser()
    _p.add_argument("--from-scratch", action="store_true", help="Ignore last.pt and train from scratch")
    _args = _p.parse_args()
    run(from_scratch=_args.from_scratch)
