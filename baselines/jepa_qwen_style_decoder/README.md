# V-JEPA encoder + Qwen-style decoder (KV-cache AR)

**Request:** use a **Qwen-like causal decoder**, but keep everything else aligned
with V-JEPA stream (frozen ViT-L, same videos/protocol, MTP heads).

This is **not** Exp A. Exp A is a parallel cross-attn latent decoder without
KV cache. This folder is the Qwen-style arm.

## What is aligned with V-JEPA

| Piece | Setting |
|-------|---------|
| Vision encoder | Frozen ViT-L; optional **encoder LoRA finetune** (`--train-encoder-lora`) |
| Data / protocol | HD-EPIC P01 stream half-split, +2/+4/+6s MTP |
| Heads | Communicating-MLP MTP (same as stream JEPA) |
| Decoder size | d=384, L=12, H=12 (≈ JEPA predictor / Exp A) |

## What is Qwen-style

| Piece | Behavior |
|-------|----------|
| Architecture | **Decoder-only** causal Transformer; vision tokens as **prefix** |
| Train | Teacher-forcing over `[BOS(+Δt) \| vision \| future slots]` (like LM SFT) |
| Val / infer | **Autoregressive** future-token decode with **KV cache** |
| Future length | **16** LM-style slots (not 256 spatial tubelets — AR×256 is impractical; Qwen also emits short spans) |

```text
video → [frozen ViT-L] → vision prefix
              │
              ▼
     Qwen-style decoder-only (causal)
       train: teacher forcing
       val:   AR + KV cache
              │
              ▼
     future tokens ⊕ context → MTP heads
```

## vs Exp A / JEPA predictor / full Qwen

| | JEPA predictor | Exp A | **This** | Full Qwen2-VL-2B |
|--|----------------|-------|----------|------------------|
| Vision | ViT-L | ViT-L | ViT-L | Qwen ViT |
| Future module | mask-token SA | cross-attn decoder | **decoder-only + KV AR** | LLM |
| KV cache | no | no | **yes (val)** | yes (gen) |
| Trainable core | pred LoRA | ~29M decoder | ~22M decoder (± enc LoRA) | LoRA ~9M on 2B |

## Resume / walltime notes

- Val `ar_kv` is slow (~4450 steps); Slurm walltime is **03:50:00** with `USR1@180`.
- Mid-val resume uses lightweight `resume_progress.json` (val does not update weights).
- Full `latest.pt` is still written on train periodic saves / epoch boundaries.

## Run

```bash
# Frozen encoder (fair architecture compare)
sbatch baselines/jepa_qwen_style_decoder/submit_qwen_style_decoder.slurm

# Finetune encoder LoRA (ViT-L backbone still frozen)
sbatch baselines/jepa_qwen_style_decoder/submit_qwen_style_enc_lora_ft.slurm
```

Out (frozen): `/scratch/ll5914/experiments/p01_stream_qwen_style_decoder_2_4_6/`  
Out (enc-LoRA FT): `/scratch/ll5914/experiments/p01_stream_qwen_style_enc_lora_ft_2_4_6/`
