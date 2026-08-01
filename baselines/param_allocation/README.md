# Encoder-heavy vs decoder-heavy allocation

**Hypothesis:** On streaming closed-set anticipation, Qwen-style models underperform
partly because most capacity sits in a **language decoder**, not the vision stack.
We test allocation on the **same pretrained ViT-L** with a matched ~25–30M budget.

| Arm | Trainable (~) | Frozen | Future module |
|-----|---------------|--------|---------------|
| `encoder_heavy` | last **2** ViT-L blocks ≈25M | rest of encoder | none (pool encoder tokens) |
| `decoder_heavy` | causal decoder ≈29M | entire encoder | Exp-A-style cross-attn decoder |

Both: same stream protocol, same MTP heads, **no encoder LoRA** (fair pair).
`decoder_heavy` here ≠ Exp A (Exp A also freezes encoder LoRA warm-start).

```bash
sbatch baselines/param_allocation/submit_encoder_heavy.slurm
sbatch baselines/param_allocation/submit_decoder_heavy.slurm
```

Out:
- `/scratch/ll5914/experiments/p01_stream_encoder_heavy_2_4_6/`
- `/scratch/ll5914/experiments/p01_stream_decoder_heavy_2_4_6/`

If encoder_heavy ≫ decoder_heavy → supports “put params in vision” for this task.
If ≈ → allocation alone does not explain Qwen’s gap (look at frames / objective / +Δt).
