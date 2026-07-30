# Aligning RU-LSTM capacity with V-JEPA ViT-L

## Why the original RU-LSTM looked tiny

| Model | What is counted | Params |
|------|------------------|--------|
| V-JEPA **ViT-L/16** encoder | frozen visual backbone used at inference | **≈304M** |
| Original Streaming RU-LSTM | `hidden=1024`, `depth=1`, linear heads on TSN feats | **≈18M** |
| TSN BNInception (feature extractor) | frozen, precomputed @4fps | ≈10M |

So the previous baseline compared an **18M temporal head** to a **304M visual encoder + temporal predictor**.

## Alignment principle (this run)

Match **total temporal-model capacity** of RU-LSTM to ViT-L (~304M), keeping the
same Streaming protocol and the same precomputed TSN-RGB features:

```
hidden=2048
depth=4          # Rolling + Unrolling LSTMs
--input-proj     # Linear(1024→2048)+GELU
--mlp-head       # 2-layer classifiers (verb/noun/action)
```

Expected size: **≈290M** (within ~5% of ViT-L’s 304M).

Together with frozen TSN (~10M), the full RU-LSTM *pipeline* is also ≈300M.

## What we are *not* claiming

- Not matching JEPA’s **trainable** subset (stream MTP freezes ViT-L, trains
  predictor-LoRA + heads — often <<304M trainable).
- Not replacing TSN with a ViT-L pixel encoder (that would be a different baseline).
- Same data protocol: half-split streaming, predict +2/+4/+6s.

## v2 optimized recipe (this run)

v1 (`hidden=2048, depth=4`) failed to train (train action@top5 stuck ~11%).

**v2 fixes:**
1. **Shallow-wide**: `hidden=4096, depth=1` (~328M) — capacity without deep LSTM
2. LayerNorm + orthogonal init + forget-gate bias=1
3. Warm-start top-left weights from the working 18M checkpoint
4. AdamW + warmup/cosine, label smoothing 0.1, feature noise, AMP, early stop

```bash
sbatch /home/ll5914/Jepa_baseline/rulstm_hdepic/submit_rulstm_vitl_aligned_v2.slurm
```

Outputs: `/scratch/ll5914/experiments/rulstm_hdepic_p01_stream_vitl_aligned_v2/`

## FLOPs are NOT aligned (measured @ 10s context tick)

Params ≈ matched, but compute is dominated by the **pixel ViT-L encoder**:

| Component | Params | FLOPs / tick @10s |
|-----------|--------|-------------------|
| RU-LSTM original temporal | ~18M | ~1.5 GFLOPs |
| RU-LSTM v2 temporal (param-aligned) | ~328M | ~24 GFLOPs |
| TSN-BNInception (40 feats @4fps) | ~10M | ~190 GFLOPs |
| **RU-LSTM v2 full pipeline** | ~338M | **~214 GFLOPs** |
| V-JEPA ViT-L encoder (80 frames @8fps, 256²) | ~304M | **~16.5 TFLOPs** |
| + predictor (pruned ≤4096 tok) | — | ~0.5 TFLOPs |
| **V-JEPA enc+pred** | — | **~17.0 TFLOPs** |

Rough gaps: ViT-L encoder ≈ **680×** v2 temporal head; JEPA full ≈ **80×** RU-LSTM v2 full pipeline.

Script: `measure_flops_analytical.py` → `flops_comparison.json`
