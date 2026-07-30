# Qwen2-VL-2B stream probe vs V-JEPA vanilla

Decoder-VLM baseline on the **same** HD-EPIC P01 stream half-split protocol
(+2/+4/+6s) used by V-JEPA video-only MTP.

## Model choice

- **Qwen2-VL-2B-Instruct** — official **smallest** Qwen-VL (~2B; ViT≈675M + LLM≈1.5B)
- Still ~**5×** V-JEPA vanilla (~400M total / ~89M trainable); no official ~400M Qwen-VL exists
- Freeze vision; LoRA (r=16) on LLM q/k/v/o; 8 frames @ 256²; multi-horizon CE heads

## Run

```bash
sbatch /home/ll5914/Jepa_baseline/qwen_vl_stream/submit_qwen_stream.slurm
```

Outputs: `/scratch/ll5914/experiments/p01_stream_qwen2vl2b_2_4_6/`

```bash
python compare_to_jepa_vanilla.py \
  --qwen-dir /scratch/ll5914/experiments/p01_stream_qwen2vl2b_2_4_6
```

JEPA vanilla ref: `/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/`
(best action@2s Top-5 ≈ 25.9%).

See `ANALYSIS.md` for why JEPA often wins this closed-set streaming task.
