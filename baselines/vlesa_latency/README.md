# VLESA as 2nd baseline — GPU latency

Official code: [HanjiangHu/VLESA](https://github.com/HanjiangHu/VLESA)

Only the **local Safety Q-Filter** (fine-tuned Qwen3-VL-2B) runs on GPU.
The reasoning agent calls Llama/OpenAI APIs (not measured here).

## Run latency benchmark

```bash
sbatch /home/ll5914/Jepa_baseline/vlesa_latency/submit_latency.slurm
```

Outputs under `/scratch/ll5914/experiments/vlesa_latency/`:

- `latency_max1024.json` — paper default `max_new_tokens=1024`
- `latency_max256.json` — capped generation for a lower-latency profile

Reports mean / p50 / p95 for preprocess, generate, and end-to-end ms, plus tok/s.
