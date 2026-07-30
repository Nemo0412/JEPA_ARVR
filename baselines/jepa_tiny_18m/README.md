# V-JEPA ≈18M vs original RU-LSTM (~18M)

Reverse of the ViT-L capacity alignment: shrink V-JEPA to match original
Streaming RU-LSTM (~18M), then compare +2/+4/+6s accuracy on the same
HD-EPIC P01 stream half-split protocol.

## Recipe

| Side | Stack | Params |
|------|--------|--------|
| RU-LSTM original | `hidden=1024, depth=1` on TSN feats | **~18.1M** |
| V-JEPA tiny | `vit_tiny` + predictor `320d×10×8` + small heads | **~18M backbone** |

- No Meta tiny checkpoint → **random init**, `--train-encoder`
- Same data: `/scratch/.../stream_half_split/`, horizons 2/4/6s

## Run

```bash
sbatch /home/ll5914/Jepa_baseline/jepa_tiny_18m/submit_stream_tiny_18m.slurm
```

Outputs: `/scratch/ll5914/experiments/p01_stream_mtp_tiny18m_2_4_6/`

```bash
python /home/ll5914/Jepa_baseline/jepa_tiny_18m/compare_to_rulstm.py \
  --jepa-dir /scratch/ll5914/experiments/p01_stream_mtp_tiny18m_2_4_6
```

RU-LSTM reference metrics:
`/scratch/ll5914/experiments/rulstm_hdepic_p01_stream/val_metrics.json`
