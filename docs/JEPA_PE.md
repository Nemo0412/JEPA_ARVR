# Jepa_PE — Stream KV + Probe Temporal RoPE

Branch for **probe positional encoding / temporal RoPE** work on HD-EPIC P01,
with **train/eval-matched stream KV** encoding (newest-aligned chunked cache)
and a **probe-blk0 attention prune** streaming finetune (± RoPE).

Status snapshot (2026-09-24): `kvmatch112` / `kvrope112` still in **epoch 0**
(no val yet). `kvprune_rope` (probe-blk0 prune FT) queued. Train curves and
configs under `plots/` and `configs/jepa_pe/`. Large `.pt` ckpts stay on
scratch (gitignored).

---

## Download HD-EPIC

Official site: [hd-epic.github.io](https://hd-epic.github.io/)  
DOI / Bristol dump: [data.bris](https://data.bris.ac.uk/data/dataset/3cqb5b81wk2dc2379fx1mrxh47) (~2.3 TiB full)  
Annotations: [hd-epic/hd-epic-annotations](https://github.com/hd-epic/hd-epic-annotations)  
Downloader: [hd-epic/hd-epic-downloader](https://github.com/hd-epic/hd-epic-downloader)

### Recommended (videos + SLAM/gaze, skip VRS)

VRS alone is ~1.9 TB. For V-JEPA anticipation you mainly need **mp4 videos**,
**narrations/action annotations**, and optionally **SLAM-and-Gaze**.

```bash
# 1) Annotations
git clone https://github.com/hd-epic/hd-epic-annotations.git \
  /path/to/HD-EPIC/hd-epic-annotations

# 2) Downloader
git clone https://github.com/hd-epic/hd-epic-downloader.git \
  /path/to/HD-EPIC/hd-epic-downloader
cd /path/to/HD-EPIC/hd-epic-downloader

# 3) Download (resumable via md5). Parent dir will get HD-EPIC/
python hd-epic-downloader.py /path/to/HD-EPIC \
  --videos --slam-gaze \
  --digital-twin --audio --hands \
  --consent-form --acquisition-guidelines
# Optional: --participant 1   (P01 only, faster for smoke tests)
```

Cluster helper used here (paths are machine-specific; edit before submit):

```bash
sbatch scripts/submit_download_hdepic_full_cpu_ll5914.slurm
# or the portable template:
# sbatch scripts/download_hdepic_data_cpu.slurm
```

### Convert to V-JEPA CSV + video link tree

```bash
python scripts/convert_hdepic_to_vjepa_csv.py \
  --annotations-pkl /path/to/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl \
  --video-root /path/to/HD-EPIC/HD-EPIC \
  --output-dir /path/to/HD-EPIC/hdepic_vjepa_annotations/full_pool \
  --link-root /path/to/HD-EPIC/hdepic_vjepa_videos \
  --link-method symlink \
  --skip-missing-videos \
  --no-video-probe --fps 30 \
  --split-preset legacy --val-ratio 0.2
```

P01 clip-level 80/20 split (what the current runs use):

```bash
python scripts/make_hdepic_clip_split.py \
  --pool /path/to/HD-EPIC/hdepic_vjepa_annotations/full_pool \
  --out  /path/to/HD-EPIC/hdepic_vjepa_annotations/clip_split
# All participants P01–P09 (optional):
# python scripts/make_hdepic_clip_split_all.py --pool ... --out .../clip_split_all --force
```

More notes: [`scripts/README_hdepic_action_anticipation.md`](../scripts/README_hdepic_action_anticipation.md).

---

## Design notes (RoPE + prune)

All PE finetunes train **encoder LoRA + probe/heads jointly** (encoder base
weights stay frozen; LoRA `lr_mult=0.5`). Probe-only FT underperforms.

Both prune/RoPE signals live on **`Probe.blocks[0]` self-attn** (not a trained
pruner) unless noted:

| Piece | Default |
|---|---|
| **Probe temporal RoPE** | `only_block0=True`, `rope_cross_attn_k=False` — rotate Q/K of block 0 only |
| **Stream prune scores** (`kvprune*`) | mean received mass from Probe block-0 self-attn; **detached** for keep/drop; used on the **next** admit |
| **Prune geometry** | cache 128f → drop lowest 34f (17 slots) → keep 94 + encode new 34 → packed 128 |
| **Matched stream KV** (`kvmatch*` / `kvrope*`) | already joint enc-LoRA + probe via `evals.main` |

Matched stream-KV train uses dense slot ids `0..S-1` inside the packed window.
Prune / abs-frame RoPE FT arms use **abs** surviving frame/slot ids for RoPE.

---

## Current runs (configs)

| Job name | Encode window | Cache + chunk | Probe RoPE | Horizon | BS | Epochs |
|---|---|---|---|---|---|---|
| **kvmatch0** | **16f** (last 2s) | 0 + 16 | off | 2s | 2 | 8 |
| **kvmatch112** | **128f** | 112 + 16 | off | 2s | 2 | 8 |
| **kvrope112** | **128f** | 112 + 16 | **on** (`only_block0`) | 2s | 2 | 8 |
| **kvprune_rope** | stream 128→94+34 | prune by probe-blk0 | FT ± RoPE (`only_block0`) | **single** 2s or 6s | — | 8 |

Shared settings (matched train):

- Dataset: HD-EPIC **P01** `clip_split`
- Backbone: ViT-L/16 @ 256, encoder LoRA (rank 8)
- Dataloader: `frames_per_clip=128`, `frames_per_second=8` (~16s clip)
- Protocol: newest-aligned **stream KV** (chunked encode; history under `no_grad`, train last chunk only)

Frozen copies of each matched run’s `config.yaml` + `protocol.json`:

- [`configs/jepa_pe/kvmatch0/`](../configs/jepa_pe/kvmatch0/)
- [`configs/jepa_pe/kvmatch112/`](../configs/jepa_pe/kvmatch112/)
- [`configs/jepa_pe/kvrope112/`](../configs/jepa_pe/kvrope112/)
- Progress snapshot: [`configs/jepa_pe/status.json`](../configs/jepa_pe/status.json)

### Submit (cluster) — 3-partition race + low mem

Always race **A100 / H100 / H200** (same `--job-name`); first to start
`scancel`s siblings. Prefer **`--mem=96G --cpus-per-task=8`** (smaller mem
queues faster; ~84G MaxRSS historically OK for matched train).

```bash
SCR=scripts/submit_clip_stream_kv_matched_ll5914.slurm
EXPORT_COMMON=HORIZON=2,BATCH_SIZE=2,NUM_WORKERS=2,VAL_NUM_WORKERS=1,PREFETCH_FACTOR=1

race() {  # usage: race JOBNAME EXPORTS
  local name="$1"; shift
  for spec in 'a100_tandon|gpu:a100:1' 'h100_tandon|gpu:h100:1' 'h200_tandon|gpu:h200:1'; do
    IFS='|' read -r part gres <<< "$spec"
    sbatch --job-name="$name" --partition="$part" --gres="$gres" \
      --mem=96G --cpus-per-task=8 --time=01:50:00 \
      --export="$1" "$SCR"
  done
}

race kvmatch0   "CACHE_FRAMES=0,PROBE_ROPE=0,$EXPORT_COMMON"
race kvmatch112 "CACHE_FRAMES=112,PROBE_ROPE=0,$EXPORT_COMMON"
race kvrope112  "CACHE_FRAMES=112,PROBE_ROPE=1,$EXPORT_COMMON"
```

Probe-blk0 prune FT (± RoPE), **single horizon** (`HORIZON=2` or `6`):

```bash
SCR=scripts/submit_stream_kv_prune_probe_rope_ll5914.slurm
for H in 2 6; do
  for spec in 'a100_tandon|gpu:a100:1' 'h100_tandon|gpu:h100:1' 'h200_tandon|gpu:h200:1'; do
    IFS='|' read -r part gres <<< "$spec"
    sbatch --job-name="kvprune${H}" --partition="$part" --gres="$gres" \
      --mem=96G --cpus-per-task=8 --time=01:50:00 --export=HORIZON=$H "$SCR"
  done
done
```

Matched train at +6s (separate from running h2s jobs):

```bash
race kvmatch6 "CACHE_FRAMES=112,PROBE_ROPE=0,HORIZON=6,BATCH_SIZE=2,NUM_WORKERS=2,VAL_NUM_WORKERS=1,PREFETCH_FACTOR=1"
race kvrope6  "CACHE_FRAMES=112,PROBE_ROPE=1,HORIZON=6,BATCH_SIZE=2,NUM_WORKERS=2,VAL_NUM_WORKERS=1,PREFETCH_FACTOR=1"
```

Scratch outs:

- matched: `/scratch/ll5914/experiments/clip_stream_kv_{matched,rope}_c*_h{2,6}s/`
- prune FT: `/scratch/ll5914/experiments/stream_kv_probe_blk0_prune_ft_rope_h{2,6}s/`

---

## Results so far (train only)

| Run | Logged itrs (epoch 0) | Last train loss | Last train acc action | Last recall@5 action |
|---|---:|---:|---:|---:|
| kvmatch0 | 0 → ~760 | ~12.0 | ~25.7% | ~23.2% |
| kvmatch112 | 0 → 510 | 11.25 | 18.4% | 12.9% |
| kvrope112 | 0 → 230 | 13.69 | 19.5% | 11.7% |
| kvprune_rope | — | — | — | pending |

Overlap train comparison (itr 0–230): RoPE ≈ **−0.2** loss, **+1.4pp** action acc vs kvmatch112 — early / noisy; **no val yet**.

Plots:

- [`plots/kvrope112_vs_kvmatch112_train_loss_from0.png`](../plots/kvrope112_vs_kvmatch112_train_loss_from0.png)
- [`plots/kvrope112_vs_kvmatch112_train_loss_smooth.png`](../plots/kvrope112_vs_kvmatch112_train_loss_smooth.png)
- [`plots/kvrope112_vs_kvmatch112_train_curves_full.csv`](../plots/kvrope112_vs_kvmatch112_train_curves_full.csv)

Earlier abs-frame RoPE ablation (prune survivors keep raw slot id; hurt @2s):  
[`configs/jepa_pe/probe_temporal_rope_abs_frame_128kv.json`](../configs/jepa_pe/probe_temporal_rope_abs_frame_128kv.json)

---

## Code map

| Piece | Path |
|---|---|
| Stream KV encode + probe-blk0 prune + RoPE helpers | `app/hdepic_lora_action_anticipation/stream_kvcache_attn_prune.py` |
| Wire stream KV / probe RoPE into eval | `app/hdepic_lora_action_anticipation/eval.py` |
| Matched train/eval launcher | `scripts/submit_clip_stream_kv_matched_ll5914.slurm` |
| Probe-blk0 prune FT ± RoPE | `scripts/finetune_stream_kv_prune_probe_rope.py` |
| Prune FT launcher | `scripts/submit_stream_kv_prune_probe_rope_ll5914.slurm` |
| Abs-frame RoPE FT/eval (legacy) | `scripts/finetune_eval_probe_temporal_rope_abs_frame.py` |
