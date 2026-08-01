# Experiment A — Causal decoder on V-JEPA latents

**Question:** holding frozen ViT-L + frames + MTP heads fixed, does the
**anticipative JEPA predictor** beat a **param-matched causal Transformer
decoder** on HD-EPIC P01 Streaming Video (+2/+4/+6s)?

## Shared vs swapped

```text
video → [frozen ViT-L encoder] → context latents
              │
              ▼
     ┌────────┴────────┐
     │                 │
 JEPA predictor    Exp A causal decoder   ← only this block differs
     │                 │
     └────────┬────────┘
              ▼
     future tokens ⊕ context → Communicating-MLP MTP → verb/noun/action
```

| Piece | V-JEPA vanilla stream | Exp A (this folder) |
|-------|----------------------|---------------------|
| Encoder | Frozen ViT-L (+ frozen encoder LoRA) | Same |
| Future module | Anticipative `vit_predictor` (d=384, L=12, H=12) + LoRA | Causal cross-attn decoder (same d/L/H, ≈29M) |
| Heads / protocol | Communicating-MLP MTP; stream half-split +2/+4/+6s | Same |

---

## Architecture: JEPA predictor vs causal (CA) Transformer decoder

These are **not** the same stack with/without KV cache. They differ in
**how tokens are arranged and how attention is patterned**.

### V-JEPA anticipative predictor

- **Role:** predict a slab of **future visual latents** conditioned on context
  and anticipation time.
- **Token layout:** project encoder tokens → concat **learnable mask tokens**
  for target (future) positions → one joint sequence.
- **Attention:** stacked Transformer **blocks with (joint) self-attention** over
  context+targets (RoPE / position ids). Targets are filled via mask tokens,
  not via a separate cross-attn decoder stack.
- **Time condition:** anticipation enters as **skipped tubelet positions**
  (`skip = N + grid² · Δt`), so future queries sit at the right temporal index.
- **Decoding style:** typically **one parallel forward** that emits all
  `N_pred` future tokens (optionally multi-step rollout by appending preds).
  Not left-to-right token generation.

### Exp A causal Transformer decoder (“CA decoder”)

- **Role:** same interface (emit future tokens for MTP), different inductive bias.
- **Token layout:** encoder latents = **memory**; separate **learnable future
  queries** (length = `N_pred`).
- **Attention:** standard `TransformerDecoderLayer`s =
  1. **causal self-attn** among future queries, and
  2. **cross-attn** from queries → encoder memory.
- **Time condition:** scalar Δt → small MLP → added to every query.
- **Decoding style:** training uses **parallel teacher-forcing** with a
  triangular causal mask (all future positions in one forward).

### About KV cache (common confusion)

| | JEPA predictor | Exp A causal decoder |
|--|----------------|----------------------|
| Causal among future tokens? | Not the main design (mask-token joint SA) | Yes (causal self-attn on queries) |
| **KV cache in our code?** | **No** | **No** (we do not implement incremental decode) |
| *Could* use KV cache at inference? | Unusual — targets are predicted as a slab, not step-by-step LM tokens | **Yes in principle** — causal decoder admits token-by-token decode with cached K/V |

**Takeaway:** “Predictor has no KV cache, CA Transformer has KV cache” is
**not** the right distinction for this ablation. KV cache is an **inference
optimization** for autoregressive step-by-step decoding. Our Exp A model is
*architecturally causal* (so KV cache is *compatible*), but both sides here
run **full-sequence attention in training**; JEPA’s predictor is a
**masked/anticipative latent predictor**, not an LM-style decoder with cache.

Cleaner one-liner for reviewers:

> Exp A replaces JEPA’s mask-token anticipative predictor (joint self-attn over
> context+future) with a standard causal decoder (causal self-attn + cross-attn
> to frozen encoder memory), capacity-matched (384 / 12 / 12).

---

## Run

```bash
sbatch baselines/jepa_causal_decoder/submit_causal_decoder_stream.slurm
```

- Out: `/scratch/ll5914/experiments/p01_stream_causal_decoder_2_4_6/`
- Compare to: `/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/` (best action@2s Top-5 ≈ 25.88%)
- From-scratch predictor control: [`../jepa_scratch_predictor/`](../jepa_scratch_predictor/)

## Notes

- Decoder is trained **from scratch** (no JEPA predictor weight init). JEPA
  vanilla keeps pretrained predictor + LoRA — intentional if comparing
  “shipped JEPA method” vs decoder substitute.
- Stricter architecture claim needs the from-scratch predictor control above.
- Want **Qwen-style decoder-only + KV-cache AR** (not this Exp A): see
  [`../jepa_qwen_style_decoder/`](../jepa_qwen_style_decoder/).
