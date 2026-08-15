# AGA on HD-EPIC P01 (Streaming Video)

Baseline: official [Action-Guided Attention (AGA)](https://github.com/CorcovadoMing/AGA)
adapted to the same **Streaming Video** protocol as RU-LSTM / V-JEPA:

- Temporal half-split per video (1st half train / 2nd half val)
- Context grows 4→6→8→10s from half origin, then slides 10s; tick every 2s
- Predict actions at **+2 / +4 / +6 s**
- Index: `/scratch/ll5914/datasets/HD-EPIC/hdepic_vjepa_annotations/stream_half_split/`

AGA is a recurrent action-anticipation model: each observed feature writes an
**action-keyed memory**, and the next query is a moving average of predicted
action distributions. Original EK100 code anticipates a single future action
from a 30×1s window. Here we:

1. Feed the observed stream window (TSN-RGB, subsampled to **1s** like the paper)
2. Unroll the last visual token for 2 / 4 / 6 steps
3. Read verb / noun / action heads at those horizons

Features are **the same TSN-RGB 1024-d @ 4 fps** as [`rulstm_hdepic/`](../rulstm_hdepic/),
so the comparison isolates AGA vs RU-LSTM as the temporal module.

## Run

```bash
sbatch /home/ll5914/Jepa_baseline/aga_hdepic/submit_aga_p01_stream.slurm
```

Outputs:

- Features: `/scratch/ll5914/datasets/HD-EPIC/rulstm_features/rgb_p01/`
- Checkpoints + metrics: `/scratch/ll5914/experiments/aga_hdepic_p01_stream/`
- Logs: `/scratch/ll5914/logs/aga_p01_stream_*.out`

```bash
python /home/ll5914/Jepa_baseline/aga_hdepic/compare_to_rulstm.py \
  --aga-dir /scratch/ll5914/experiments/aga_hdepic_p01_stream
```
