# RU-LSTM on HD-EPIC P01 (Streaming Video)

Baseline: official [RU-LSTM](https://github.com/fpv-iplab/rulstm) RGB branch, adapted to the
same **Streaming Video** protocol as `jepa_yifan`:

- Temporal half-split per video (1st half train / 2nd half val)
- Context grows 4→6→8→10s from half origin, then slides 10s; tick every 2s
- Predict actions at **+2 / +4 / +6 s**
- Index: `/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/`

## Pipeline

1. Extract TSN-RGB (BNInception, EK-pretrained) features @ 4 fps
2. Train causal Streaming RU-LSTM (Rolling over context, Unroll `horizon/α` steps)
3. Report verb / noun / action Top-1 & Top-5 accuracy per horizon

## Run

```bash
sbatch /home/ll5914/Jepa_baseline/rulstm_hdepic/submit_rulstm_p01_stream.slurm
```

Outputs:

- Features: `/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01/`
- Checkpoints + metrics: `/scratch/ll5914/experiments/rulstm_hdepic_p01_stream/`
- Logs: `/scratch/ll5914/logs/rulstm_p01_stream_*.out`
