"""
HD-EPIC Probe Fine-tuning on V-JEPA 2  (with Attention-based Token Pruning)
=============================================================================
Freeze V-JEPA 2 ViT-L encoder (optionally with LoRA) and train a new
AttentiveClassifier on HD-EPIC's own verb/noun vocabulary.

Attention-based token pruning (optional, PRUNE_KEEP_RATIO > 0):
  After the encoder's last-layer self-attention we compute how much attention
  each token receives from all other tokens across all heads:

      importance[j] = Σ_{h, i}  attn[b, h, i, j]

  The top-K tokens (K = N × PRUNE_KEEP_RATIO) are kept and passed to the
  probe. Importance is computed with chunked Q matmuls so the full O(N²)
  attention matrix never materialises in memory.

Checkpoints:
  hdepic-vitl-probe-pruned-last.pt  — overwritten after every epoch
  hdepic-vitl-probe-pruned-best.pt  — saved when val Verb Recall@5 improves

Usage:
  cd /home/ll5914/ARVR_Video/vjepa2
  python ../vjepa/train_hdepic_probe.py           # resumes from last.pt if present
  python ../vjepa/train_hdepic_probe.py --from-scratch
"""

import sys, os, pickle, time, argparse
sys.path.insert(0, "/home/ll5914/ARVR_Video/vjepa2")
sys.path.insert(0, "/home/ll5914/ARVR_Video")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from decord import VideoReader, cpu

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentivePooler
from src.models.vision_transformer import vit_large_rope
from src.models.utils.modules import rotate_queries_or_keys

# Single source of truth for the train/val/test video-ID split
from hdepic_anticipation_ar import TRAIN_VIDEO_IDS, VAL_VIDEO_IDS, TEST_VIDEO_IDS

# ── Paths ───────────────────────────────────────────────────────────
ENCODER_CKPT  = "/scratch/ll5914/models/vjepa2/vitl.pt"
HD_EPIC_NARR  = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
HD_VERB_CSV   = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_verb_classes.csv"
HD_NOUN_CSV   = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_noun_classes.csv"
VIDEO_DIR     = "/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos/P01"
SAVE_DIR      = "/scratch/ll5914/models/vjepa2"
PROBE_BEST    = os.path.join(SAVE_DIR, "hdepic-vitl-probe-pruned-best.pt")
PROBE_LAST    = os.path.join(SAVE_DIR, "hdepic-vitl-probe-pruned-last.pt")

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

# ── LoRA hyper-parameters ────────────────────────────────────────────
# Set LORA_RANK = 0 to keep the encoder fully frozen (original behaviour).
# Typical choices: rank ∈ {4, 8, 16}, alpha = 2 × rank.
LORA_RANK  = 8     # rank of the low-rank adapters; 0 disables LoRA
LORA_ALPHA = 16.0  # LoRA scaling: effective ΔW is scaled by alpha / rank
LORA_LR    = 5e-5  # separate (smaller) learning rate for encoder LoRA params

# ── Attention-based token pruning ────────────────────────────────────
# After the encoder's last-layer self-attention, each token's importance
# equals the total attention it receives from all other tokens across
# all heads.  The top-K tokens (K = N × PRUNE_KEEP_RATIO) are selected.
#
# Set PRUNE_KEEP_RATIO = 0.0 to disable pruning entirely.
# Typical choices: 0.25 (aggressive), 0.50 (moderate), 0.75 (mild).
#
# For ViT-L with 256×256 input and 32 frames (tubelet_size=2):
#   N = (256/16)² × (32/2) = 256 × 16 = 4 096 tokens per clip.
PRUNE_KEEP_RATIO = 0.5   # keep 50 % of tokens  (4 096 → 2 048)
PRUNE_CHUNK_SIZE = 64    # chunk size for memory-efficient importance computation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Split sizes (defined by the explicit video-ID sets imported above):
#   Train : 20 videos  (TRAIN_VIDEO_IDS)
#   Val   :  2 videos  (VAL_VIDEO_IDS)   — early stopping
#   Test  :  5 videos  (TEST_VIDEO_IDS)  — held-out final evaluation


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


# ── LoRA ────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Wraps a *frozen* nn.Linear with trainable low-rank delta: ΔW = B·A·(alpha/rank).

    Only lora_A and lora_B require gradients; the wrapped linear is kept frozen.
    Initialised so that B·A = 0 (identity-preserving at the start of training).
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear                    # frozen base weight
        self.rank   = rank
        self.scale  = alpha / rank
        d_in  = linear.in_features
        d_out = linear.out_features
        # A is randomly initialised (small), B starts at zero → ΔW = 0 initially
        self.lora_A = nn.Parameter(torch.randn(rank, d_in)  * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    def extra_repr(self) -> str:
        return (f"in={self.linear.in_features}, out={self.linear.out_features}, "
                f"rank={self.rank}, scale={self.scale:.3f}")


def apply_lora_to_encoder(encoder: nn.Module, rank: int, alpha: float) -> nn.Module:
    """Inject LoRA adapters into every transformer block's attention qkv & proj.

    After injection:
      - All *base* encoder parameters have requires_grad=False (frozen).
      - Only the injected lora_A / lora_B parameters have requires_grad=True.

    Returns the modified encoder (in-place).
    """
    for block in encoder.blocks:
        attn = block.attn
        # Freeze the underlying Linear before wrapping so LoRALinear inherits the state
        for p in attn.qkv.parameters():
            p.requires_grad = False
        for p in attn.proj.parameters():
            p.requires_grad = False
        attn.qkv  = LoRALinear(attn.qkv,  rank=rank, alpha=alpha)
        attn.proj = LoRALinear(attn.proj, rank=rank, alpha=alpha)

    # Double-check: only lora_A / lora_B are trainable
    for name, p in encoder.named_parameters():
        p.requires_grad = ("lora_A" in name or "lora_B" in name)

    n_lora = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in encoder.parameters())
    print(f"  LoRA injected into encoder: rank={rank}, alpha={alpha:.1f}")
    print(f"  Trainable encoder params: {n_lora / 1e6:.3f}M  "
          f"/ total encoder params: {n_total / 1e6:.1f}M  "
          f"({100.0 * n_lora / n_total:.2f}%)")
    return encoder


# ── Attention-based Token Pruning ───────────────────────────────────
class TokenPruner:
    """Prune encoder tokens by how much attention each token receives.

    After the encoder's last-layer self-attention we compute:

        importance[b, j] = Σ_{h=0}^{H-1}  Σ_{i=0}^{N-1}  attn[b, h, i, j]

    where attn[b, h, i, j] is the softmax attention weight from query
    token i to key token j in head h.  Tokens that are attended to by
    many other tokens (across all heads) are considered globally important.

    The top-K tokens (K = N × keep_ratio) are returned to the probe in
    their original temporal-spatial order.

    Implementation notes
    --------------------
    * V-JEPA 2 ViT-L uses *RoPEAttention* with ``use_sdpa=True`` by default,
      which calls ``F.scaled_dot_product_attention`` and never materialises
      the explicit attention matrix.  We monkey-patch the last block's
      attention forward so that we additionally compute attention weights
      for importance scoring.
    * To avoid allocating the full [B, H, N, N] matrix (≈ 8.5 GB for
      B=8, H=16, N=4096), we iterate over Q in chunks of size
      ``chunk_size`` and accumulate the per-token column sums:

          for each q_chunk: imp += softmax(q_chunk @ k.T / √d).sum(dim=-2)

      Memory per chunk: B × H × chunk_size × N × sizeof(float)
      (64-chunk → ≈ 128 MB, vs 8.5 GB for the full matrix).
    * The importance computation runs inside ``torch.no_grad()`` so it
      never adds nodes to the autograd graph.
    * Works transparently with LoRA: if LoRA adapters have been injected
      into the last block's ``attn.qkv``, the patched forward uses the
      adapted projection automatically.
    """

    def __init__(self, encoder: nn.Module, keep_ratio: float, chunk_size: int = 64):
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio  = keep_ratio
        self.chunk_size  = chunk_size
        self._importance: torch.Tensor | None = None

        last_attn = encoder.blocks[-1].attn  # RoPEAttention instance
        self._attn_module = last_attn
        self._orig_forward = last_attn.forward

        m       = last_attn   # shorthand captured by closure
        pruner  = self

        def _patched_forward(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
            B, N, C = x.size()
            grid_depth = int(N // (m.grid_size * m.grid_size))

            # ── QKV projection (uses LoRA-adapted qkv when LoRA is active) ──
            qkv = m.qkv(x).unflatten(-1, (3, m.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # each [B, H, N, D_head]

            # ── 3-D RoPE positional encoding (depth / height / width) ──────
            if mask is not None:
                mask_p = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
                d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)
            else:
                if T is None or H_patches is None or W_patches is None:
                    mask_p = torch.arange(int(grid_depth * m.grid_size * m.grid_size), device=x.device)
                else:
                    mask_p = torch.arange(int(T * H_patches * W_patches), device=x.device)
                d_mask, h_mask, w_mask = m.separate_positions(mask_p, H_patches, W_patches)

            s = 0
            qd = rotate_queries_or_keys(q[..., s : s + m.d_dim], pos=d_mask)
            kd = rotate_queries_or_keys(k[..., s : s + m.d_dim], pos=d_mask)
            s += m.d_dim
            qh = rotate_queries_or_keys(q[..., s : s + m.h_dim], pos=h_mask)
            kh = rotate_queries_or_keys(k[..., s : s + m.h_dim], pos=h_mask)
            s += m.h_dim
            qw = rotate_queries_or_keys(q[..., s : s + m.w_dim], pos=w_mask)
            kw = rotate_queries_or_keys(k[..., s : s + m.w_dim], pos=w_mask)
            s += m.w_dim
            if s < m.head_dim:
                q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
                k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
            else:
                q = torch.cat([qd, qh, qw], dim=-1)
                k = torch.cat([kd, kh, kw], dim=-1)

            # ── Importance scoring: chunked attention column-sum ────────────
            # importance[b, j] = Σ_{h, i} attn[b, h, i, j]
            # Chunked over query axis to avoid O(B·H·N²) memory allocation.
            with torch.no_grad():
                imp = torch.zeros(B, N, device=x.device, dtype=x.dtype)
                for ci in range(0, N, pruner.chunk_size):
                    q_c = q[:, :, ci : ci + pruner.chunk_size, :]        # [B, H, cs, D_head]
                    logits = (q_c @ k.transpose(-2, -1)) * m.scale        # [B, H, cs, N]
                    imp += logits.softmax(dim=-1).sum(dim=2).sum(dim=1)   # [B, N]
                pruner._importance = imp   # stored for prune() call

            # ── Main attention output (SDPA for speed & gradient correctness)
            if attn_mask is not None or m.use_sdpa:
                with torch.backends.cuda.sdp_kernel():
                    x = F.scaled_dot_product_attention(
                        q, k, v, dropout_p=m.proj_drop_prob,
                        is_causal=m.is_causal, attn_mask=attn_mask
                    )
            else:
                attn = (q @ k.transpose(-2, -1)) * m.scale
                attn = attn.softmax(dim=-1)
                attn = m.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = m.proj(x)
            x = m.proj_drop(x)
            return x

        last_attn.forward = _patched_forward

        num_tokens = (IMG_SIZE // 16) ** 2 * (FRAMES_PER_CLIP // 2)
        K = max(1, int(num_tokens * keep_ratio))
        print(f"  TokenPruner attached to encoder.blocks[-1].attn")
        print(f"    keep_ratio={keep_ratio:.2f}  →  {num_tokens} tokens → {K} tokens per clip")
        print(f"    chunk_size={chunk_size} (importance computed without full N² matrix)")

    # ------------------------------------------------------------------
    def prune(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-K tokens by importance score.

        Args:
            feats: [B, N, D]  encoder output (all tokens from last layer)

        Returns:
            pruned:     [B, K, D]  top-K tokens kept in original order
            importance: [B, N]    per-token importance scores
        """
        if self._importance is None:
            raise RuntimeError("Call encoder.forward() before pruner.prune().")

        importance = self._importance                                 # [B, N]
        N = feats.shape[1]
        K = max(1, int(N * self.keep_ratio))

        # topk, then sort to restore original token order
        top_k_idx, _ = importance.topk(K, dim=1)                     # [B, K] (unsorted)
        top_k_idx    = top_k_idx.sort(dim=1).values                  # [B, K] sorted

        idx_exp = top_k_idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])
        pruned  = feats.gather(1, idx_exp)                           # [B, K, D]
        return pruned, importance

    # ------------------------------------------------------------------
    def remove(self):
        """Restore the original RoPEAttention.forward method."""
        if self._orig_forward is not None:
            self._attn_module.forward = self._orig_forward
            self._orig_forward = None


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


def load_encoder(device, lora_rank: int = 0):
    """Load the pretrained ViT-L encoder.

    If lora_rank > 0, LoRA adapters are injected into every attention block.
    Only the LoRA parameters (lora_A, lora_B) will have requires_grad=True;
    all pre-trained weights remain frozen.

    The encoder is returned in train() mode when LoRA is active so that
    gradients can flow through the LoRA path, and in eval() mode otherwise.
    (ViT-L uses LayerNorm and default drop=0, so train/eval mode has no effect
    on the frozen path; it matters only for the LoRA gradient computation.)
    """
    lora_active = lora_rank > 0
    mode_str    = f"LoRA rank={lora_rank}" if lora_active else "fully frozen"
    print(f"  Loading ViT-L encoder ({mode_str})...")
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
    model = model.to(device)
    if lora_active:
        apply_lora_to_encoder(model, rank=lora_rank, alpha=LORA_ALPHA)
        model.train()   # train mode so backward passes work through LoRA params
    else:
        model.eval()
    return model


# ── Evaluation ──────────────────────────────────────────────────────
def evaluate(encoder, probe, loader, device, num_verbs, num_nouns, num_actions,
             pruner: "TokenPruner | None" = None):
    """Evaluate probe on *loader*.

    If *pruner* is provided, encoder tokens are pruned by attention importance
    before being fed to the probe (mirrors the training forward pass).
    """
    was_training = encoder.training
    encoder.eval()
    probe.eval()

    v_c = defaultdict(int); v_t = defaultdict(int)
    n_c = defaultdict(int); n_t = defaultdict(int)
    a_c = defaultdict(int); a_t = defaultdict(int)
    total = 0

    with torch.no_grad():
        for clips, v_ids, n_ids, a_ids in loader:
            clips = clips.to(device)
            feats = encoder(clips)
            if pruner is not None:
                feats, _ = pruner.prune(feats)
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
            if pruner is not None:
                feats, _ = pruner.prune(feats)
            v_logits, n_logits, a_logits = probe(feats)
            for i in range(len(v_ids)):
                vi, ni, ai = int(v_ids[i]), int(n_ids[i]), int(a_ids[i])
                if vi in v_logits[i].topk(3).indices.tolist(): v_top3 += 1
                if ni in n_logits[i].topk(3).indices.tolist(): noun_top3 += 1
                if ai != -1 and ai in a_logits[i].topk(3).indices.tolist(): a_top3 += 1

    n_act = sum(a_t.values())
    metrics = {
        "verb_top3":   100 * v_top3   / max(total, 1),
        "noun_top3":   100 * noun_top3 / max(total, 1),
        "action_top3": 100 * a_top3   / max(n_act, 1),
        "verb_r5":     cmr(v_c, v_t),
        "noun_r5":     cmr(n_c, n_t),
        "action_r5":   cmr(a_c, a_t),
    }

    if was_training:
        encoder.train()
    return metrics


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

    # Train / val / test split via explicit video-ID sets
    train_df = p01_df[p01_df['video_id'].isin(TRAIN_VIDEO_IDS)]
    val_df   = p01_df[p01_df['video_id'].isin(VAL_VIDEO_IDS)]
    test_df  = p01_df[p01_df['video_id'].isin(TEST_VIDEO_IDS)]
    train_vids = sorted(train_df["video_id"].unique())
    val_vids   = sorted(val_df["video_id"].unique())
    test_vids  = sorted(test_df["video_id"].unique())
    print(f"  Train : {len(train_df):4d} rows | {len(train_vids):2d} videos")
    print(f"  Val   : {len(val_df):4d} rows | {len(val_vids):2d} videos")
    print(f"  Test  : {len(test_df):4d} rows | {len(test_vids):2d} videos", flush=True)

    # Build datasets
    train_ds = HDEpicDataset(train_df, VIDEO_DIR, build_transforms(True),  verb_map, noun_map, action_map, is_train=True)
    val_ds   = HDEpicDataset(val_df,   VIDEO_DIR, build_transforms(False), verb_map, noun_map, action_map, is_train=False)
    test_ds  = HDEpicDataset(test_df,  VIDEO_DIR, build_transforms(False), verb_map, noun_map, action_map, is_train=False)
    print(f"  Valid samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Load models
    print("\n[2] Loading models...")
    encoder = load_encoder(device, lora_rank=LORA_RANK)

    # Optionally attach attention-based token pruner to the last encoder block
    pruner: TokenPruner | None = None
    if PRUNE_KEEP_RATIO > 0.0:
        pruner = TokenPruner(encoder, keep_ratio=PRUNE_KEEP_RATIO, chunk_size=PRUNE_CHUNK_SIZE)

    probe   = HDEpicProbe(
        embed_dim=encoder.embed_dim,
        num_verbs=len(vdf),
        num_nouns=len(ndf),
        num_actions=len(action_map),
    ).to(device)
    probe_params = sum(p.numel() for p in probe.parameters()) / 1e6
    print(f"  Probe parameters: {probe_params:.1f}M")

    # Optimizer + LR schedule
    # Build param groups: probe always trains; encoder LoRA params use a separate (smaller) LR.
    param_groups: list[dict] = [{"params": list(probe.parameters()), "lr": LR}]
    if LORA_RANK > 0:
        lora_params = [p for p in encoder.parameters() if p.requires_grad]
        param_groups.append({"params": lora_params, "lr": LORA_LR})
        print(f"  Encoder LoRA trainable params added to optimizer (lr={LORA_LR:.1e})")
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
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
        ckpt = {
            "epoch": completed_epochs,
            "probe": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "verb_names": verb_names,
            "noun_names": noun_names,
            "action_map": action_map,
            "metrics": metrics,
            "train_video_ids": sorted(train_df["video_id"].unique().tolist()),
            "val_video_ids":   sorted(val_df["video_id"].unique().tolist()),
            "test_video_ids":  sorted(test_df["video_id"].unique().tolist()),
            "lora_rank":         LORA_RANK,
            "lora_alpha":        LORA_ALPHA,
            "prune_keep_ratio":  PRUNE_KEEP_RATIO,
            "prune_chunk_size":  PRUNE_CHUNK_SIZE,
        }
        if LORA_RANK > 0:
            # Only persist the small LoRA delta weights, not the frozen base encoder.
            ckpt["encoder_lora"] = {
                k: v for k, v in encoder.state_dict().items()
                if "lora_A" in k or "lora_B" in k
            }
        return ckpt

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

        # Restore LoRA weights if the checkpoint contains them
        if LORA_RANK > 0 and "encoder_lora" in ckpt:
            ckpt_lora_rank = ckpt.get("lora_rank", LORA_RANK)
            if ckpt_lora_rank != LORA_RANK:
                print(f"    [WARN] Checkpoint LoRA rank={ckpt_lora_rank} != current LORA_RANK={LORA_RANK}; "
                      f"skipping LoRA weight restore.", flush=True)
            else:
                missing, unexpected = encoder.load_state_dict(ckpt["encoder_lora"], strict=False)
                print(f"    Restored encoder LoRA weights "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
        elif LORA_RANK > 0:
            print(f"    [INFO] No encoder_lora in checkpoint; LoRA adapters start from zero.", flush=True)

        # Restore optimizer — skip if the param-group structure changed (e.g. LoRA added/removed)
        if ckpt.get("optimizer"):
            ckpt_n_groups = len(ckpt["optimizer"].get("param_groups", []))
            cur_n_groups  = len(optimizer.param_groups)
            if ckpt_n_groups == cur_n_groups:
                optimizer.load_state_dict(ckpt["optimizer"])
                print(f"    Restored optimizer", flush=True)
            else:
                print(f"    [WARN] Optimizer param-group count changed "
                      f"({ckpt_n_groups} → {cur_n_groups}); using fresh optimizer.", flush=True)
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

    lora_active = LORA_RANK > 0
    prune_info  = (f", prune_keep={PRUNE_KEEP_RATIO:.2f}" if pruner is not None else ", no pruning")
    print(f"\n[3] Starting training ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}"
          + (f", lora_lr={LORA_LR}, lora_rank={LORA_RANK}" if lora_active else ", encoder frozen")
          + prune_info + ")...")
    print(f"    Saved every epoch: {PROBE_LAST}", flush=True)
    print(f"    Saved on best Verb R@5: {PROBE_BEST}", flush=True)
    print("=" * 65, flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        probe.train()
        if lora_active:
            encoder.train()   # LoRA params need gradients
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (clips, v_ids, n_ids, a_ids) in enumerate(train_loader):
            clips  = clips.to(device)
            v_ids  = v_ids.to(device)
            n_ids  = n_ids.to(device)

            # When LoRA is active we need a gradient graph through the encoder.
            # Frozen base weights have requires_grad=False so they accumulate no grads.
            if lora_active:
                feats = encoder(clips)
            else:
                with torch.no_grad():
                    feats = encoder(clips)

            # Prune tokens by last-layer attention importance (if enabled).
            # The TokenPruner captured importance scores during encoder.forward().
            if pruner is not None:
                feats, _ = pruner.prune(feats)

            a_ids = a_ids.to(device)
            v_logits, n_logits, a_logits = probe(feats)
            loss = criterion(v_logits, v_ids) + criterion(n_logits, n_ids)
            valid_a = (a_ids >= 0)
            if valid_a.any():
                loss = loss + criterion(a_logits[valid_a], a_ids[valid_a])

            optimizer.zero_grad()
            loss.backward()
            # Clip gradients for probe; also clip LoRA grads if active
            nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            if lora_active:
                lora_trainable = [p for p in encoder.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(lora_trainable, 1.0)
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
                           len(vdf), len(ndf), len(action_map), pruner=pruner)
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

        # evaluate() already restored encoder.train() if needed (LoRA case)
        probe.train()
        print("", flush=True)

    print("=" * 65, flush=True)
    print(f"Training complete! Best Val Verb Recall@5 = {best_verb_r5:.1f}%", flush=True)
    print(f"Latest: {PROBE_LAST}", flush=True)
    print(f"Best:   {PROBE_BEST}", flush=True)

    # ── Final evaluation on the held-out test set (best checkpoint) ──
    print("\n[4] Final test-set evaluation (loading best checkpoint)...", flush=True)
    best_ck = torch.load(PROBE_BEST, map_location=device, weights_only=False)
    probe.load_state_dict(best_ck["probe"])
    if LORA_RANK > 0 and "encoder_lora" in best_ck:
        encoder.load_state_dict(best_ck["encoder_lora"], strict=False)
    test_metrics = evaluate(encoder, probe, test_loader, device,
                            len(vdf), len(ndf), len(action_map), pruner=pruner)
    print("  Test set results (best val checkpoint):", flush=True)
    print(f"    Verb   Top-3={test_metrics['verb_top3']:.1f}%  Recall@5={test_metrics['verb_r5']:.1f}%", flush=True)
    print(f"    Noun   Top-3={test_metrics['noun_top3']:.1f}%  Recall@5={test_metrics['noun_r5']:.1f}%", flush=True)
    print(f"    Action Top-3={test_metrics['action_top3']:.1f}%  Recall@5={test_metrics['action_r5']:.1f}%", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    _p = argparse.ArgumentParser()
    _p.add_argument("--from-scratch", action="store_true", help="Ignore last.pt and train from scratch")
    _args = _p.parse_args()
    run(from_scratch=_args.from_scratch)
