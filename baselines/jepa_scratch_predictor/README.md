# From-scratch JEPA predictor (control vs Exp A)

Frozen ViT-L + frozen encoder LoRA + MTP (same as Exp A / vanilla stream).

**Only change:** JEPA `vit_predictor` is **re-initialized** and trained fully
(no predictor checkpoint, no predictor LoRA).

| Compare to | Meaning if scratch ≈ Exp A decoder | Meaning if scratch ≫ decoder |
|------------|-------------------------------------|------------------------------|
| Exp A causal decoder | Pretrain/表征 narrative | Anticipative **architecture** helps even from scratch |
| Vanilla (pretrained pred+LoRA) | Gap = value of predictor pretrain | — |

```bash
sbatch baselines/jepa_scratch_predictor/submit_scratch_predictor.slurm
```

Out: `/scratch/ll5914/experiments/p01_stream_scratch_predictor_2_4_6/`
