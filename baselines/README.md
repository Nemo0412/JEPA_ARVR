# Jepa_baseline

Baselines and capacity-matched comparisons for **HD-EPIC P01 Streaming Video**
action anticipation (**+2 / +4 / +6 s**).

This tree lives on branch **`baseline`** of
[Nemo0412/JEPA_ARVR](https://github.com/Nemo0412/JEPA_ARVR) under `baselines/`.
V-JEPA training code remains in the main repo (`app/hdepic_lora_action_anticipation/`).

## Protocol (shared)

- Annotations: `/scratch/.../hdepic_vjepa_annotations/stream_half_split/`
- Half-split per video; context grow 4→10s; tick 2s; predict **+2/+4/+6s**
- Primary metric: **action Top-1 / Top-5 @ +2s**

## Layout

| Path | What |
|------|------|
| [`rulstm/`](rulstm/) | Upstream [RU-LSTM](https://github.com/fpv-iplab/rulstm) (vendor) |
| [`rulstm_hdepic/`](rulstm_hdepic/) | Streaming RU-LSTM on HD-EPIC P01 |
| [`jepa_tiny_18m/`](jepa_tiny_18m/) | From-scratch V-JEPA `vit_tiny` ≈18–20M vs RU-LSTM |
| [`qwen_vl_stream/`](qwen_vl_stream/) | Qwen2-VL-2B LoRA probe (official smallest Qwen-VL) |
| [`vlesa/`](vlesa/) | Upstream [VLESA](https://github.com/HanjiangHu/VLESA) (vendor) |
| [`vlesa_latency/`](vlesa_latency/) | VLESA safety-filter GPU latency bench |

---

## Progress (2026-07-30)

### 1) V-JEPA video-only vanilla (reference)

- Config: **ViT-L frozen** + stream MTP; **~401M total / ~89M trainable**
- Out: `/scratch/ll5914/experiments/p01_stream_mtp_2_4_6/`
- **Best (epoch 2): action@2s Top-5 ≈ 25.88%**

### 2) RU-LSTM streaming (original-size temporal ~18M)

- Out: `/scratch/ll5914/experiments/rulstm_hdepic_p01_stream/`
- Val **action@2s: 6.58 / 19.28** (Top-1 / Top-5)
- @4s: 5.34 / 16.40 · @6s: 4.69 / 15.04

### 3) V-JEPA `vit_tiny` ≈18–20M (from scratch, video-only)

- Params: encoder ~5.6M + predictor 320×10 ~12.5M + heads ~1.9M → **~19.9M**
- Out: `/scratch/ll5914/experiments/p01_stream_mtp_tiny18m_2_4_6/`
- Job: `p01_jepa_18m` (12 epochs, auto-resubmit; still running / late epochs)
- **Reliable best (epoch 6, full val): action@2s 5.91 / 18.77**
- vs RU-LSTM @2s: **does not beat** main action (−0.67 / −0.51 pt)
  - **verb** Top-5 better (66.2 vs 59.5); **noun** worse (34.2 vs 37.3)
- After epoch 6: val drops while train Top-5 climbs → **overfitting**
- Note: epoch 9 val only ~355s (incomplete) — ignore inflated numbers

### 4) Qwen-VL decoder baseline

- **No official Qwen-VL ≈400M.** Official smallest = **~2B**
  (Qwen2-VL-2B ≈ ViT 675M + LLM 1.5B; Qwen2.5-VL starts at 3B)
- Switched from mistaken 3B job → **Qwen2-VL-2B-Instruct** LoRA probe
- Code: `qwen_vl_stream/`; weights: `/scratch/ll5914/hf_cache/Qwen2-VL-2B-Instruct/`
- Out (pending): `/scratch/ll5914/experiments/p01_stream_qwen2vl2b_2_4_6/`
- Job: `p01_qwen2vl2b` (queued behind other H100 jobs)
- Fairness note: still **~5×** JEPA vanilla params; use as “smallest official Qwen-VL”
  on the **same stream protocol**, not param-matched

### 5) Other

- **VLESA** latency harness under `vlesa_latency/` (local safety-filter GPU latency)
- Stronger JEPA (concat-CA / MTP variants) tracked in main `JEPA_ARVR` training jobs,
  not this baselines folder

### Snapshot table (action @ +2s)

| Method | Params (approx) | Top-1 | Top-5 | Status |
|--------|-----------------|------:|------:|--------|
| V-JEPA vanilla (ViT-L frozen) | 401M tot / 89M train | — | **25.88** | done |
| RU-LSTM stream | ~18M temporal | **6.58** | **19.28** | done |
| V-JEPA vit_tiny | ~19.9M | 5.91 | 18.77 | best@ep6; not > RU |
| Qwen2-VL-2B LoRA probe | ~2B | TBD | TBD | submitted |

---

## Cluster quick start

```bash
sbatch baselines/rulstm_hdepic/submit_rulstm_p01_stream.slurm
sbatch baselines/jepa_tiny_18m/submit_stream_tiny_18m.slurm
sbatch baselines/qwen_vl_stream/submit_qwen_stream.slurm
```

Absolute paths in `.slurm` / scripts point at NYU Torch scratch; edit for other machines.

## GitHub notes

- Nested vendor `.git` dirs under `rulstm/` and `vlesa/` are **not** pushed
  (copied as plain trees). Prefer submodules later if needed.
- Checkpoints / experiment dumps are gitignored; results live on scratch.
