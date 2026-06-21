"""
HD-EPIC Baseline: Full Qwen2.5-VL-3B (LoRA fine-tuned) + classification heads
===============================================================================
Uses the complete VLM pipeline — vision tower + LLM — taking video frames and a
text prompt as input, then fine-tuning with LoRA.  Classification is performed
by adding verb / noun / action linear heads on top of the LLM's last-token
hidden state; this is directly comparable to train_hdepic_probe.py:

  - Same data split    : 20 train / 2 val / 5 test (explicit video-ID sets)
  - Same anticipation  : observe 4 s ending 1 s before each action start
  - Same metrics       : Verb / Noun / Action — Top-3 accuracy & Recall@5

Multi-GPU:
  Supports single-GPU (python script.py) and multi-GPU DDP via torchrun:
    torchrun --nproc_per_node=2 train_hdepic_qwen_probe.py

Checkpoints saved to SAVE_DIR:
  hdepic-qwen25vl3b-probe-best.pt   (best val Verb Recall@5, rank-0 only)
  hdepic-qwen25vl3b-probe-last.pt   (overwritten every epoch, rank-0 only)
"""

import sys, os, pickle, time, argparse, threading, queue
sys.path.insert(0, "/home/ll5914/ARVR_Video/vjepa2")
sys.path.insert(0, "/home/ll5914/ARVR_Video")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import defaultdict
from PIL import Image
from decord import VideoReader, cpu

from hdepic_anticipation_ar import TRAIN_VIDEO_IDS, VAL_VIDEO_IDS, TEST_VIDEO_IDS

# ── Paths ───────────────────────────────────────────────────────────
HD_EPIC_NARR = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
HD_VERB_CSV  = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_verb_classes.csv"
HD_NOUN_CSV  = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_noun_classes.csv"
VIDEO_DIR    = "/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos/P01"
SAVE_DIR     = "/scratch/ll5914/models/vjepa2"
PROBE_BEST   = os.path.join(SAVE_DIR, "hdepic-qwen25vl3b-probe-best.pt")
PROBE_LAST   = os.path.join(SAVE_DIR, "hdepic-qwen25vl3b-probe-last.pt")

QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# ── Clip / anticipation parameters (same as V-JEPA probe) ───────────
FRAMES_PER_CLIP  = 32
FPS              = 8
ANTICIPATION_SEC = 1.0
SAMPLE_FRAMES    = 8         # frames sent to Qwen per clip

# ── Training hyper-parameters ────────────────────────────────────────
LORA_RANK      = 16
LORA_ALPHA     = 32.0
BATCH_SIZE     = 4           # per-GPU batch size
GRAD_ACCUM     = 2           # effective per-GPU batch = 4×2 = 8; ×num_gpus globally
NUM_EPOCHS     = 10
LR             = 2e-4
WEIGHT_DECAY   = 1e-4
WARMUP_EPOCHS  = 2
NUM_WORKERS    = 4
PREFETCH_QUEUE = 3

TASK_PROMPT = "Based on this video, predict what action the person will perform next."


# ── DDP helpers ──────────────────────────────────────────────────────
def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_ddp() else 0

def world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1

def is_main() -> bool:
    return rank() == 0

def log(*args, **kwargs):
    """Print only from rank 0."""
    if is_main():
        print(*args, **kwargs)


# ── Dataset ─────────────────────────────────────────────────────────
class HDEpicQwenDataset(Dataset):
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
            vr          = VideoReader(s['video_path'], num_threads=1, ctx=cpu(0))
            vfps        = vr.get_avg_fps()
            frame_step  = max(1, int(vfps / FPS))
            end_frame   = int(s['obs_end'] * vfps)
            start_frame = end_frame - FRAMES_PER_CLIP * frame_step
            all_idx     = np.arange(start_frame, end_frame, frame_step, dtype=np.int64)
            all_idx     = np.clip(all_idx, 0, len(vr) - 1)
            pick        = np.linspace(0, len(all_idx) - 1, SAMPLE_FRAMES, dtype=int)
            frames      = vr.get_batch(all_idx[pick]).asnumpy()
        except Exception:
            frames = np.zeros((SAMPLE_FRAMES, 256, 256, 3), dtype=np.uint8)
        return frames, s['verb_id'], s['noun_id'], s['action_id']


def collate_fn(batch):
    frames_list, v_ids, n_ids, a_ids = zip(*batch)
    return (
        np.stack(frames_list, axis=0),
        torch.tensor(v_ids, dtype=torch.long),
        torch.tensor(n_ids, dtype=torch.long),
        torch.tensor(a_ids, dtype=torch.long),
    )


# ── Batch tokenisation ───────────────────────────────────────────────
def prepare_batch_cpu(frames_np: np.ndarray, processor) -> dict:
    """CPU-side: PIL conversion + Qwen processor → CPU tensors for prefetcher."""
    B = frames_np.shape[0]
    texts, video_lists = [], []
    for b in range(B):
        pil_frames = [Image.fromarray(frames_np[b, t]) for t in range(frames_np.shape[1])]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames},
                    {"type": "text", "text": TASK_PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
        video_lists.append(pil_frames)
    return processor(text=texts, videos=video_lists, return_tensors="pt", padding=True)


class BatchPrefetcher:
    """Pre-processes Qwen inputs in a background thread while the GPU runs."""

    def __init__(self, loader, processor, queue_size: int = PREFETCH_QUEUE):
        self._q = queue.Queue(maxsize=queue_size)
        t = threading.Thread(target=self._worker, args=(loader, processor), daemon=True)
        t.start()

    def _worker(self, loader, processor):
        try:
            for frames_np, v_ids, n_ids, a_ids in loader:
                cpu_inputs = prepare_batch_cpu(frames_np, processor)
                self._q.put((cpu_inputs, v_ids, n_ids, a_ids))
        except Exception as exc:
            self._q.put(exc)
        finally:
            self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item


# ── LoRA ─────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.rank   = rank
        self.scale  = alpha / rank
        d_in, d_out = linear.in_features, linear.out_features
        # Same device + dtype as the frozen linear to avoid cross-device errors
        dev, dtype  = linear.weight.device, linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(rank, d_in,  device=dev, dtype=dtype) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=dev, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora_to_llm(model: nn.Module, rank: int, alpha: float) -> None:
    """Inject LoRA into q/k/v/o projections of LLM layers (skip vision tower)."""
    n_injected = 0
    for mod_name, module in model.named_modules():
        if "visual" in mod_name:
            continue
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            orig = getattr(module, proj_name, None)
            if orig is None or not isinstance(orig, nn.Linear):
                continue
            for p in orig.parameters():
                p.requires_grad = False
            setattr(module, proj_name, LoRALinear(orig, rank=rank, alpha=alpha))
            n_injected += 1

    if n_injected == 0:
        raise RuntimeError("No q/k/v/o projections found outside vision tower.")

    for name, p in model.named_parameters():
        p.requires_grad = ("lora_A" in name or "lora_B" in name)

    n_lora  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"  LoRA injected into {n_injected} LLM projections (rank={rank}, alpha={alpha:.1f})")
    log(f"  Trainable LLM LoRA : {n_lora/1e6:.2f}M / total {n_total/1e6:.0f}M "
        f"({100*n_lora/n_total:.2f}%)")


# ── Probe ────────────────────────────────────────────────────────────
class QwenVLProbe(nn.Module):
    """Qwen2.5-VL-3B (LoRA on LLM) + classification heads on last-token hidden state."""

    def __init__(self, qwen_model, lm_hidden_size, num_verbs, num_nouns, num_actions):
        super().__init__()
        self.qwen = qwen_model
        # Heads in float32 on the same device as Qwen (float32 for numerical stability;
        # last_hidden is cast to float32 in forward before being passed here)
        dev = next(qwen_model.parameters()).device
        self.verb_head   = nn.Linear(lm_hidden_size, num_verbs).to(device=dev)
        self.noun_head   = nn.Linear(lm_hidden_size, num_nouns).to(device=dev)
        self.action_head = nn.Linear(lm_hidden_size, num_actions).to(device=dev)

    def forward(self, **model_inputs):
        outputs = self.qwen(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :].float()
        return (
            self.verb_head(last_hidden),
            self.noun_head(last_hidden),
            self.action_head(last_hidden),
        )


def unwrap(model):
    """Return the underlying module, stripping DDP wrapper if present."""
    return model.module if isinstance(model, DDP) else model


# ── Evaluation ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, processor, loader, device):
    model.eval()

    v_c = defaultdict(int); v_t = defaultdict(int)
    n_c = defaultdict(int); n_t = defaultdict(int)
    a_c = defaultdict(int); a_t = defaultdict(int)
    v_top3 = n_top3 = a_top3 = total = 0

    for cpu_inputs, v_ids, n_ids, a_ids in BatchPrefetcher(loader, processor):
        inputs = {k: v.to(device, non_blocking=True) for k, v in cpu_inputs.items()}
        v_logits, n_logits, a_logits = model(**inputs)

        for i in range(len(v_ids)):
            vi, ni, ai = int(v_ids[i]), int(n_ids[i]), int(a_ids[i])
            v_t[vi] += 1; n_t[ni] += 1
            if vi in v_logits[i].topk(5).indices.tolist(): v_c[vi] += 1
            if ni in n_logits[i].topk(5).indices.tolist(): n_c[ni] += 1
            if vi in v_logits[i].topk(3).indices.tolist(): v_top3 += 1
            if ni in n_logits[i].topk(3).indices.tolist(): n_top3 += 1
            if ai != -1:
                a_t[ai] += 1
                if ai in a_logits[i].topk(5).indices.tolist(): a_c[ai] += 1
                if ai in a_logits[i].topk(3).indices.tolist(): a_top3 += 1
            total += 1

    # Aggregate across DDP ranks
    if is_ddp():
        stats = torch.tensor([v_top3, n_top3, a_top3, total,
                               sum(a_t.values())], dtype=torch.long, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        v_top3, n_top3, a_top3, total, n_act = stats.tolist()
    else:
        n_act = sum(a_t.values())

    def cmr(c, t):
        r = [c.get(k, 0) / v for k, v in t.items()]
        return np.mean(r) * 100 if r else 0.0

    return {
        "verb_top3":   100 * v_top3 / max(total, 1),
        "noun_top3":   100 * n_top3 / max(total, 1),
        "action_top3": 100 * a_top3 / max(n_act, 1),
        "verb_r5":     cmr(v_c, v_t),
        "noun_r5":     cmr(n_c, n_t),
        "action_r5":   cmr(a_c, a_t),
    }


# ── Main training loop ──────────────────────────────────────────────
def run(from_scratch: bool = False, lora_rank: int = LORA_RANK):
    # ── DDP initialisation ────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    log("=" * 65)
    log(f"Qwen2.5-VL-3B (LoRA) — HD-EPIC Baseline  "
        f"[{world_size()} GPU{'s' if world_size()>1 else ''}]")
    log("=" * 65)
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(local_rank)})"
                                if device.type == "cuda" else ""))

    # ── Annotations ──────────────────────────────────────────────────
    log("\n[1] Loading annotations...")
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
    log(f"  verbs={len(vdf)}, nouns={len(ndf)}, actions={len(action_map)}")

    train_df = p01_df[p01_df['video_id'].isin(TRAIN_VIDEO_IDS)]
    val_df   = p01_df[p01_df['video_id'].isin(VAL_VIDEO_IDS)]
    test_df  = p01_df[p01_df['video_id'].isin(TEST_VIDEO_IDS)]
    log(f"  Train : {len(train_df):4d} rows | {train_df['video_id'].nunique():2d} videos")
    log(f"  Val   : {len(val_df):4d} rows | {val_df['video_id'].nunique():2d} videos")
    log(f"  Test  : {len(test_df):4d} rows | {test_df['video_id'].nunique():2d} videos",
        flush=True)

    train_ds = HDEpicQwenDataset(train_df, VIDEO_DIR, verb_map, noun_map, action_map)
    val_ds   = HDEpicQwenDataset(val_df,   VIDEO_DIR, verb_map, noun_map, action_map)
    test_ds  = HDEpicQwenDataset(test_df,  VIDEO_DIR, verb_map, noun_map, action_map)
    log(f"  Valid samples — train:{len(train_ds)}, val:{len(val_ds)}, test:{len(test_ds)}",
        flush=True)

    # DistributedSampler splits data across ranks; val/test evaluated on rank 0 only
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp() else None
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS, pin_memory=False,
        drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate_fn,
    )

    # ── Load Qwen2.5-VL-3B ───────────────────────────────────────────
    log("\n[2] Loading Qwen2.5-VL-3B...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    # Load on CPU first, then move to local_rank device (DDP-friendly)
    qwen_raw = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)

    for p in qwen_raw.parameters():
        p.requires_grad = False
    apply_lora_to_llm(qwen_raw, rank=lora_rank, alpha=LORA_ALPHA)

    # Vision tower is frozen → no_grad saves stored activations (~3 GB VRAM)
    _orig_visual_fwd = qwen_raw.visual.forward
    @torch.no_grad()
    def _visual_no_grad(*a, **kw):
        return _orig_visual_fwd(*a, **kw)
    qwen_raw.visual.forward = _visual_no_grad

    # Gradient checkpointing: recompute activations instead of storing them
    qwen_raw.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
    lm_hidden = qwen_raw.config.hidden_size
    log(f"  LLM hidden size: {lm_hidden}")

    # ── Build probe ───────────────────────────────────────────────────
    model = QwenVLProbe(
        qwen_model=qwen_raw,
        lm_hidden_size=lm_hidden,
        num_verbs=len(vdf),
        num_nouns=len(ndf),
        num_actions=len(action_map),
    )
    for p in model.verb_head.parameters():   p.requires_grad = True
    for p in model.noun_head.parameters():   p.requires_grad = True
    for p in model.action_head.parameters(): p.requires_grad = True

    if is_ddp():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"  Total trainable: {sum(p.numel() for p in trainable)/1e6:.2f}M "
        f"(LoRA + 3 heads)", flush=True)

    # ── Optimizer + LR schedule ──────────────────────────────────────
    optimizer     = optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_ep  = len(train_loader) // GRAD_ACCUM
    total_steps   = NUM_EPOCHS * steps_per_ep
    warmup_steps  = WARMUP_EPOCHS * steps_per_ep

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    def lora_state():
        sd = unwrap(model).state_dict()
        return {k: v for k, v in sd.items()
                if "lora_A" in k or "lora_B" in k
                or k.startswith(("verb_head", "noun_head", "action_head"))}

    def pack_ckpt(epoch, metrics):
        return {
            "epoch": epoch, "model_lora": lora_state(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "verb_names": verb_names, "noun_names": noun_names,
            "action_map": action_map, "metrics": metrics,
            "lora_rank": lora_rank, "lm_hidden": lm_hidden,
            "qwen_model": QWEN_MODEL_NAME, "sample_frames": SAMPLE_FRAMES,
        }

    def restore_lora(ck):
        missing, unexpected = unwrap(model).load_state_dict(ck["model_lora"], strict=False)
        log(f"    Restored LoRA+heads (missing={len(missing)}, unexpected={len(unexpected)})",
            flush=True)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0; best_verb_r5 = 0.0
    if not from_scratch and os.path.isfile(PROBE_LAST):
        log(f"\n  [resume] Loading {PROBE_LAST} ...")
        ck = torch.load(PROBE_LAST, map_location=device, weights_only=False)
        restore_lora(ck)
        start_epoch = int(ck["epoch"])
        if start_epoch >= NUM_EPOCHS:
            log(f"  All {NUM_EPOCHS} epochs done.", flush=True)
            if is_ddp(): dist.destroy_process_group()
            return
        if ck.get("optimizer"): optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"): scheduler.load_state_dict(ck["scheduler"])
        if os.path.isfile(PROBE_BEST):
            b = torch.load(PROBE_BEST, map_location="cpu", weights_only=False)
            if b.get("metrics") and "verb_r5" in b["metrics"]:
                best_verb_r5 = float(b["metrics"]["verb_r5"])
        log(f"  [resume] Epoch {start_epoch+1}/{NUM_EPOCHS}", flush=True)
    else:
        log("\n  Training from scratch.", flush=True)

    eff_batch = BATCH_SIZE * GRAD_ACCUM * world_size()
    log(f"\n[3] Training ({NUM_EPOCHS} ep, per-GPU batch={BATCH_SIZE}, "
        f"grad_accum={GRAD_ACCUM}, effective global batch={eff_batch}, "
        f"lora_rank={lora_rank})...")
    log("=" * 65, flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # ensures different shuffle per epoch in DDP
        model.train()
        epoch_loss = 0.0; t0 = time.time()
        optimizer.zero_grad()

        for step, (cpu_inputs, v_ids, n_ids, a_ids) in enumerate(
                BatchPrefetcher(train_loader, processor)):
            inputs = {k: v.to(device, non_blocking=True) for k, v in cpu_inputs.items()}
            v_ids  = v_ids.to(device, non_blocking=True)
            n_ids  = n_ids.to(device, non_blocking=True)
            a_ids  = a_ids.to(device, non_blocking=True)

            v_logits, n_logits, a_logits = model(**inputs)
            loss = criterion(v_logits, v_ids) + criterion(n_logits, n_ids)
            valid_a = (a_ids >= 0)
            if valid_a.any():
                loss = loss + criterion(a_logits[valid_a], a_ids[valid_a])

            (loss / GRAD_ACCUM).backward()
            epoch_loss += loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if is_main() and (step + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | step {step+1}/{len(train_loader)} "
                      f"| loss={loss.item():.3f} | lr={scheduler.get_last_lr()[0]:.2e}",
                      flush=True)

        avg_loss = epoch_loss / len(train_loader)
        log(f"\nEpoch {epoch+1}/{NUM_EPOCHS} done | avg_loss={avg_loss:.3f} | "
            f"{time.time()-t0:.0f}s", flush=True)

        # Evaluate only on rank 0 (val/test loaders are full dataset on all ranks)
        if is_main():
            metrics = evaluate(unwrap(model), processor, val_loader, device)
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

        # Sync all ranks before next epoch
        if is_ddp():
            dist.barrier()

        model.train()
        log("", flush=True)

    log("=" * 65, flush=True)
    log(f"Training complete! Best Val Verb Recall@5 = {best_verb_r5:.1f}%", flush=True)

    # ── Final test-set evaluation (rank 0 only) ───────────────────────
    if is_main():
        print("\n[4] Final test-set evaluation (best checkpoint)...", flush=True)
        best_ck = torch.load(PROBE_BEST, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(best_ck["model_lora"], strict=False)
        test_m = evaluate(unwrap(model), processor, test_loader, device)
        print("  Test set results:", flush=True)
        print(f"    Verb   Top-3={test_m['verb_top3']:.1f}%  Recall@5={test_m['verb_r5']:.1f}%", flush=True)
        print(f"    Noun   Top-3={test_m['noun_top3']:.1f}%  Recall@5={test_m['noun_r5']:.1f}%", flush=True)
        print(f"    Action Top-3={test_m['action_top3']:.1f}%  Recall@5={test_m['action_r5']:.1f}%", flush=True)
    log("=" * 65, flush=True)

    if is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    _p = argparse.ArgumentParser()
    _p.add_argument("--from-scratch", action="store_true")
    _p.add_argument("--lora-rank", type=int, default=LORA_RANK)
    _args = _p.parse_args()
    run(from_scratch=_args.from_scratch, lora_rank=_args.lora_rank)
