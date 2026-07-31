# Experiment A — Causal decoder on V-JEPA latents

**Question:** holding frozen ViT-L + frames + MTP heads fixed, does the
**anticipative JEPA predictor** beat a **param-matched causal Transformer
decoder** on HD-EPIC P01 Streaming Video (+2/+4/+6s)?

## Design

| Piece | JEPA vanilla stream | This ablation |
|-------|---------------------|---------------|
| Encoder | Frozen ViT-L (+ frozen encoder LoRA) | Same |
| Temporal module | Anticipative `vit_predictor` (d=384, L=12, H=12) + LoRA | Causal cross-attn decoder (d=384, L=12, H=12, ≈29M) |
| Heads | Communicating-MLP MTP | Same |
| Protocol | stream half-split, grow 4→10s | Same |

Decoder: learnable future queries (causal among themselves) cross-attend to
encoder memory; anticipation time is an additive embedding. Output future
tokens are concatenated with context tokens for the MTP pooler — same
token layout as JEPA stream MTP.

## Run

```bash
sbatch baselines/jepa_causal_decoder/submit_causal_decoder_stream.slurm
```

- Out: `/scratch/ll5914/experiments/p01_stream_causal_decoder_2_4_6/`
- Compare to: `/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/` (best action@2s Top-5 ≈ 25.88%)

## Notes

- Decoder is trained **from scratch** (no JEPA predictor weight init). JEPA
  vanilla keeps pretrained predictor + LoRA — that is an intentional
  asymmetry if the goal is “JEPA method as shipped” vs decoder substitute.
- A stricter from-scratch predictor control can be added later if needed.
