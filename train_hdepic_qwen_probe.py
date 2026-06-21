"""
HD-EPIC Baseline: Frozen Qwen2.5-VL-3B visual encoder + AttentivePooler probe
=============================================================================
Directly comparable to train_hdepic_probe.py (V-JEPA baseline):
  - Same data split    : 20 train / 2 val / 5 test (explicit video-ID sets)
  - Same anticipation  : observe 4 s of video ending 1 s before each action
  - Same probe arch    : AttentivePooler (depth-4) + 3 linear heads
  - Same metrics       : Verb / Noun / Action — Top-3 accuracy & Recall@5

Key difference: visual backbone is Qwen2.5-VL-3B (frozen) instead of V-JEPA ViT-L.
  V-JEPA ViT-L  → 1024-d features, 32-frame tubelets processed together
  Qwen2.5-VL-3B → 1536-d features, SAMPLE_FRAMES frames encoded independently,
                   each mean-pooled over spatial patch tokens, then stacked as a
                   temporal sequence fed to the AttentivePooler.

Checkpoints saved to SAVE_DIR:
  hdepic-qwen25vl3b-probe-best.pt   (best val Verb Recall@5)
  hdepic-qwen25vl3b-probe-last.pt   (overwritten every epoch)

Usage:
  python train_hdepic_qwen_probe.py               # resume from last.pt if present
  python train_hdepic_qwen_probe.py --from-scratch
"""

import sys, os, pickle, time, argparse
sys.path.insert(0, "/home/ll5914/ARVR_Video/vjepa2")
sys.path.insert(0, "/home/ll5914/ARVR_Video")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image
from decord import VideoReader, cpu

from src.models.attentive_pooler import AttentivePooler
from hdepic_anticipation_ar import TRAIN_VIDEO_IDS, VAL_VIDEO_IDS, TEST_VIDEO_IDS

# ── Paths (reuse same annotation / video directories as V-JEPA probe) ──
HD_EPIC_NARR = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
HD_VERB_CSV  = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_verb_classes.csv"
HD_NOUN_CSV  = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_noun_classes.csv"
VIDEO_DIR    = "/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos/P01"
SAVE_DIR     = "/scratch/ll5914/models/vjepa2"
PROBE_BEST   = os.path.join(SAVE_DIR, "hdepic-qwen25vl3b-probe-best.pt")
PROBE_LAST   = os.path.join(SAVE_DIR, "hdepic-qwen25vl3b-probe-last.pt")

# ── Qwen2.5-VL model (downloaded from HuggingFace Hub on first run) ──
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# ── Anticipation / clip parameters (same as V-JEPA probe) ────────────
FRAMES_PER_CLIP  = 32        # total frames in the clip window (4 s @ 8 fps)
FPS              = 8
ANTICIPATION_SEC = 1.0       # observe up to 1 s before the action start

# ── Qwen-specific parameters ─────────────────────────────────────────
SAMPLE_FRAMES = 8            # frames sampled from the 32-frame window
QWEN_IMG_PX   = 256 * 256   # min_pixels = max_pixels → fixed 256×256 input

# ── Training hyper-parameters ────────────────────────────────────────
BATCH_SIZE    = 4            # keep small; Qwen visual encoder uses ~6 GB alone
NUM_EPOCHS    = 10
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS   = 4            # DataLoader workers (raw frame loading only)


# ── Dataset ─────────────────────────────────────────────────────────
class HDEpicQwenDataset(Dataset):
    """Load video clips and return raw uint8 frames for Qwen processing.

    Returns (frames [T, H, W, C] uint8, verb_id, noun_id, action_id).
    T = SAMPLE_FRAMES frames sampled uniformly from the FRAMES_PER_CLIP window.
    """

    def __init__(self, annotations, video_dir, verb_map, noun_map, action_map):
        self.video_dir  = video_dir
        self.verb_map   = verb_map
        self.noun_map   = noun_map
        self.action_map = action_map
        self.samples    = []

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
            vr         = VideoReader(s['video_path'], num_threads=1, ctx=cpu(0))
            vfps       = vr.get_avg_fps()
            frame_step = max(1, int(vfps / FPS))
            end_frame  = int(s['obs_end'] * vfps)
            start_frame = end_frame - FRAMES_PER_CLIP * frame_step
            all_indices = np.arange(start_frame, end_frame, frame_step, dtype=np.int64)
            all_indices = np.clip(all_indices, 0, len(vr) - 1)

            # Uniformly sample SAMPLE_FRAMES from the clip window
            pick = np.linspace(0, len(all_indices) - 1, SAMPLE_FRAMES, dtype=int)
            indices = all_indices[pick]
            frames  = vr.get_batch(indices).asnumpy()  # [T, H, W, 3] uint8
        except Exception:
            frames = np.zeros((SAMPLE_FRAMES, 256, 256, 3), dtype=np.uint8)

        return frames, s['verb_id'], s['noun_id'], s['action_id']


def collate_fn(batch):
    """Stack numpy frames and convert labels to tensors."""
    frames_list, v_ids, n_ids, a_ids = zip(*batch)
    # frames_list: tuple of [T, H, W, 3] arrays
    frames_np = np.stack(frames_list, axis=0)  # [B, T, H, W, 3]
    return (
        frames_np,
        torch.tensor(v_ids, dtype=torch.long),
        torch.tensor(n_ids, dtype=torch.long),
        torch.tensor(a_ids, dtype=torch.long),
    )


# ── Qwen2.5-VL visual feature extractor ────────────────────────────
class QwenVisualExtractor(nn.Module):
    """Frozen Qwen2.5-VL-3B vision tower.

    forward() takes a batch of video clips as numpy [B, T, H, W, C] uint8
    and returns float32 features [B, T, embed_dim] by encoding each frame
    independently and mean-pooling its spatial patch tokens.
    """

    def __init__(self, model_name: str, processor):
        super().__init__()
        from transformers import Qwen2_5_VLForConditionalGeneration
        print(f"  Loading Qwen2.5-VL from '{model_name}' (this may take a minute)...")
        qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        self.visual    = qwen.visual
        self.processor = processor
        # Freeze all vision tower parameters
        for p in self.visual.parameters():
            p.requires_grad = False
        del qwen  # Free LLM backbone weights; we only keep the vision tower

    @torch.no_grad()
    def encode_frame(self, pil_image: Image.Image, device: torch.device) -> torch.Tensor:
        """Encode a single PIL frame → mean-pooled feature [embed_dim]."""
        img_inputs = self.processor.image_processor(
            images=[pil_image], return_tensors="pt"
        )
        pv  = img_inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        thw = img_inputs["image_grid_thw"].to(device)
        feat = self.visual(pv, grid_thw=thw)   # [N_tokens, embed_dim]
        return feat.mean(0).float()             # [embed_dim]

    def forward(self, frames_np: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        frames_np : [B, T, H, W, C] uint8 numpy
        Returns   : [B, T, embed_dim] float32 on `device`
        """
        B, T = frames_np.shape[:2]
        results = []
        for b in range(B):
            clip_feats = []
            for t in range(T):
                pil = Image.fromarray(frames_np[b, t])
                clip_feats.append(self.encode_frame(pil, device))
            results.append(torch.stack(clip_feats))   # [T, embed_dim]
        return torch.stack(results)                   # [B, T, embed_dim]


# ── Probe (identical structure to train_hdepic_probe.py) ────────────
class HDEpicProbe(nn.Module):
    """AttentivePooler + verb / noun / action classification heads."""

    def __init__(self, embed_dim: int, num_verbs: int, num_nouns: int, num_actions: int):
        super().__init__()
        # num_heads must divide embed_dim; 1536/16 = 96 ✓
        num_heads = 16 if embed_dim % 16 == 0 else 12
        self.pooler      = AttentivePooler(
            num_queries=3,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=4,
            use_activation_checkpointing=False,
        )
        self.verb_head   = nn.Linear(embed_dim, num_verbs)
        self.noun_head   = nn.Linear(embed_dim, num_nouns)
        self.action_head = nn.Linear(embed_dim, num_actions)

    def forward(self, x: torch.Tensor):
        """x: [B, T, embed_dim]  →  verb logits, noun logits, action logits"""
        x = self.pooler(x)                   # [B, 3, embed_dim]
        v = self.verb_head(x[:, 0, :])
        n = self.noun_head(x[:, 1, :])
        a = self.action_head(x[:, 2, :])
        return v, n, a


# ── Evaluation ──────────────────────────────────────────────────────
def evaluate(extractor, probe, loader, device):
    probe.eval()

    v_c = defaultdict(int); v_t = defaultdict(int)
    n_c = defaultdict(int); n_t = defaultdict(int)
    a_c = defaultdict(int); a_t = defaultdict(int)
    v_top3 = n_top3 = a_top3 = total = 0

    for frames_np, v_ids, n_ids, a_ids in loader:
        feats = extractor(frames_np, device)          # [B, T, D]
        with torch.no_grad():
            v_logits, n_logits, a_logits = probe(feats.to(device))

        for i in range(len(v_ids)):
            vi, ni, ai = int(v_ids[i]), int(n_ids[i]), int(a_ids[i])
            v_t[vi] += 1; n_t[ni] += 1
            top5_v = v_logits[i].topk(5).indices.tolist()
            top5_n = n_logits[i].topk(5).indices.tolist()
            if vi in top5_v: v_c[vi] += 1
            if ni in top5_n: n_c[ni] += 1
            if vi in v_logits[i].topk(3).indices.tolist(): v_top3 += 1
            if ni in n_logits[i].topk(3).indices.tolist(): n_top3 += 1
            if ai != -1:
                a_t[ai] += 1
                top5_a = a_logits[i].topk(5).indices.tolist()
                if ai in top5_a: a_c[ai] += 1
                if ai in a_logits[i].topk(3).indices.tolist(): a_top3 += 1
            total += 1

    def cmr(c, t):
        r = [c.get(k, 0) / v for k, v in t.items()]
        return np.mean(r) * 100 if r else 0.0

    n_act = max(sum(a_t.values()), 1)
    return {
        "verb_top3":   100 * v_top3 / max(total, 1),
        "noun_top3":   100 * n_top3 / max(total, 1),
        "action_top3": 100 * a_top3 / n_act,
        "verb_r5":     cmr(v_c, v_t),
        "noun_r5":     cmr(n_c, n_t),
        "action_r5":   cmr(a_c, a_t),
    }


# ── Main training loop ──────────────────────────────────────────────
def run(from_scratch: bool = False):
    print("=" * 65)
    print("Qwen2.5-VL-3B — HD-EPIC Probe Baseline")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ── Annotations ──────────────────────────────────────────────────
    print("\n[1] Loading annotations...")
    with open(HD_EPIC_NARR, "rb") as f:
        narr_df = pickle.load(f)
    p01_df = narr_df[narr_df['video_id'].str.startswith('P01')].copy()

    vdf = pd.read_csv(HD_VERB_CSV)
    ndf = pd.read_csv(HD_NOUN_CSV)
    verb_map   = {int(r['id']): int(r['id']) for _, r in vdf.iterrows()}
    noun_map   = {int(r['id']): int(r['id']) for _, r in ndf.iterrows()}
    verb_names = {int(r['id']): r['key'] for _, r in vdf.iterrows()}
    noun_names = {int(r['id']): r['key'] for _, r in ndf.iterrows()}

    pairs = set()
    for _, row in p01_df.iterrows():
        vcs = row['verb_classes']; ncs = row['noun_classes']
        if isinstance(vcs, list) and isinstance(ncs, list) and vcs and ncs:
            pairs.add((int(vcs[0]), int(ncs[0])))
    action_map = {k: i for i, k in enumerate(pairs)}
    print(f"  verbs={len(vdf)}, nouns={len(ndf)}, actions={len(action_map)}")

    train_df = p01_df[p01_df['video_id'].isin(TRAIN_VIDEO_IDS)]
    val_df   = p01_df[p01_df['video_id'].isin(VAL_VIDEO_IDS)]
    test_df  = p01_df[p01_df['video_id'].isin(TEST_VIDEO_IDS)]
    print(f"  Train : {len(train_df):4d} rows | {train_df['video_id'].nunique():2d} videos")
    print(f"  Val   : {len(val_df):4d} rows | {val_df['video_id'].nunique():2d} videos")
    print(f"  Test  : {len(test_df):4d} rows | {test_df['video_id'].nunique():2d} videos",
          flush=True)

    train_ds = HDEpicQwenDataset(train_df, VIDEO_DIR, verb_map, noun_map, action_map)
    val_ds   = HDEpicQwenDataset(val_df,   VIDEO_DIR, verb_map, noun_map, action_map)
    test_ds  = HDEpicQwenDataset(test_df,  VIDEO_DIR, verb_map, noun_map, action_map)
    print(f"  Valid samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False,
                              collate_fn=collate_fn)

    # ── Load Qwen2.5-VL visual encoder (frozen) ──────────────────────
    print("\n[2] Loading Qwen2.5-VL-3B visual encoder...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        QWEN_MODEL_NAME,
        min_pixels=QWEN_IMG_PX,
        max_pixels=QWEN_IMG_PX,
    )
    extractor = QwenVisualExtractor(QWEN_MODEL_NAME, processor)
    extractor.visual = extractor.visual.to(device)
    extractor.visual.eval()

    # Determine output embed_dim with a probe forward pass
    _dummy = np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)
    _dummy_pil = Image.fromarray(_dummy[0, 0])
    with torch.no_grad():
        _feat = extractor.encode_frame(_dummy_pil, device)
    embed_dim = _feat.shape[0]
    print(f"  Qwen2.5-VL-3B visual embed_dim = {embed_dim}")

    # ── Probe ─────────────────────────────────────────────────────────
    probe = HDEpicProbe(
        embed_dim=embed_dim,
        num_verbs=len(vdf),
        num_nouns=len(ndf),
        num_actions=len(action_map),
    ).to(device)
    probe_params = sum(p.numel() for p in probe.parameters()) / 1e6
    print(f"  Probe parameters: {probe_params:.1f}M")

    # ── Optimizer + LR schedule (probe only; encoder is frozen) ──────
    optimizer    = optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = NUM_EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    def pack_ckpt(epoch, metrics):
        return {
            "epoch":       epoch,
            "probe":       probe.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "verb_names":  verb_names,
            "noun_names":  noun_names,
            "action_map":  action_map,
            "metrics":     metrics,
            "embed_dim":   embed_dim,
            "qwen_model":  QWEN_MODEL_NAME,
            "sample_frames": SAMPLE_FRAMES,
        }

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch  = 0
    best_verb_r5 = 0.0
    if not from_scratch and os.path.isfile(PROBE_LAST):
        print(f"\n  [resume] Loading {PROBE_LAST} ...")
        ck = torch.load(PROBE_LAST, map_location=device, weights_only=False)
        probe.load_state_dict(ck["probe"])
        start_epoch = int(ck["epoch"])
        if start_epoch >= NUM_EPOCHS:
            print(f"  All {NUM_EPOCHS} epochs already done.", flush=True)
            return
        if ck.get("optimizer"):
            optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        if os.path.isfile(PROBE_BEST):
            b = torch.load(PROBE_BEST, map_location="cpu", weights_only=False)
            if b.get("metrics") and "verb_r5" in b["metrics"]:
                best_verb_r5 = float(b["metrics"]["verb_r5"])
        print(f"  [resume] Completed {start_epoch} epochs, resuming from epoch "
              f"{start_epoch + 1}/{NUM_EPOCHS}", flush=True)
    else:
        print("\n  Training from scratch.", flush=True)

    print(f"\n[3] Training ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}, "
          f"lr={LR:.1e}, sample_frames={SAMPLE_FRAMES})...")
    print("=" * 65, flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        probe.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (frames_np, v_ids, n_ids, a_ids) in enumerate(train_loader):
            # Extract Qwen visual features (no grad; encoder is frozen)
            feats = extractor(frames_np, device)    # [B, T, embed_dim]

            v_ids = v_ids.to(device)
            n_ids = n_ids.to(device)
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
            if (step + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | step {step+1}/{len(train_loader)} "
                      f"| loss={loss.item():.3f} | lr={scheduler.get_last_lr()[0]:.2e}",
                      flush=True)

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} done | avg_loss={avg_loss:.3f} | "
              f"{elapsed:.0f}s", flush=True)

        metrics = evaluate(extractor, probe, val_loader, device)
        print(f"  Validation:", flush=True)
        print(f"    Verb   Top-3={metrics['verb_top3']:.1f}%  Recall@5={metrics['verb_r5']:.1f}%", flush=True)
        print(f"    Noun   Top-3={metrics['noun_top3']:.1f}%  Recall@5={metrics['noun_r5']:.1f}%", flush=True)
        print(f"    Action Top-3={metrics['action_top3']:.1f}%  Recall@5={metrics['action_r5']:.1f}%", flush=True)

        torch.save(pack_ckpt(epoch + 1, metrics), PROBE_LAST)
        print(f"  Saved latest -> {PROBE_LAST}", flush=True)

        if metrics["verb_r5"] > best_verb_r5:
            best_verb_r5 = metrics["verb_r5"]
            torch.save(pack_ckpt(epoch + 1, metrics), PROBE_BEST)
            print(f"  Saved best (verb R@5={best_verb_r5:.1f}%) -> {PROBE_BEST}", flush=True)

        probe.train()
        print("", flush=True)

    print("=" * 65, flush=True)
    print(f"Training complete! Best Val Verb Recall@5 = {best_verb_r5:.1f}%", flush=True)

    # ── Final held-out test-set evaluation ───────────────────────────
    print("\n[4] Final test-set evaluation (loading best checkpoint)...", flush=True)
    best_ck = torch.load(PROBE_BEST, map_location=device, weights_only=False)
    probe.load_state_dict(best_ck["probe"])
    test_metrics = evaluate(extractor, probe, test_loader, device)
    print("  Test set results (best val checkpoint):", flush=True)
    print(f"    Verb   Top-3={test_metrics['verb_top3']:.1f}%  Recall@5={test_metrics['verb_r5']:.1f}%", flush=True)
    print(f"    Noun   Top-3={test_metrics['noun_top3']:.1f}%  Recall@5={test_metrics['noun_r5']:.1f}%", flush=True)
    print(f"    Action Top-3={test_metrics['action_top3']:.1f}%  Recall@5={test_metrics['action_r5']:.1f}%", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    _p = argparse.ArgumentParser()
    _p.add_argument("--from-scratch", action="store_true",
                    help="Ignore last.pt and train from scratch")
    _args = _p.parse_args()
    run(from_scratch=_args.from_scratch)
