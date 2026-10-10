# kvprune joint — how we run it

Local multi-GPU recipe on **HD-EPIC P01 `clip_split`** action anticipation.
Jointly trains **encoder LoRA** (last 12 ViT blocks) + **full AttentiveClassifier
probe**. No predictor.

## Protocol (every run)

Streaming **KV cache + attention prune** so the probe always sees **128 frames**:

1. Encode history **128 frames** → per-layer encoder **KV cache**.
2. Score slots with **Probe `blocks[0]` self-attn** received mass
   (**detached**; discrete top-k, no grad through indices).
3. Drop lowest **34** frames / keep **94**; encode new **34** against kept K/V.
4. Pack **94 + 34 = 128** tokens → probe CE at anticipation horizon `H` seconds.

Prune **always** uses Probe block-0 scores. Optional **temporal RoPE** on Probe
Q/K can cover block 0 only (`--only-block0 1`) or **all** Probe self-attn
blocks (`--only-block0 0`).

Core code:

- `app/hdepic_lora_action_anticipation/stream_kvcache_attn_prune.py`
- `scripts/run_local_kvprune_joint_2s.py`

## Checkpoint & data (this machine)

| Role | Path |
|---|---|
| **Encoder init (required)** | `/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt` — V-JEPA2 ViT-L/16 @256 |
| Probe / LoRA | **from scratch** each run (do not warm from another arm unless intentional) |
| Train CSV | `/mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_train_vjepa.csv` |
| Val CSV | `/mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv` |
| Videos | `/mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_videos/` |
| Python | `/mnt/hdd/datasets/HD-EPIC/conda_envs/vjepa/bin/python` |

Env before launch:

```bash
cd /home/lls/workspace/Jepa
export VJEPA_ROOT=$PWD/vjepa2
export PYTHONPATH=$PWD:$VJEPA_ROOT
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/mnt/hdd/datasets/HD-EPIC/conda_envs/vjepa/bin/python
```

## What we ran (horizon 2s, P01)

Three arms, same protocol / same `vitl.pt` cold start / 8 epochs / `batch_size=1`:

| Arm | Flags | GPUs | Out dir | Status |
|---|---|---|---|---|
| **norope** | `--rope 0` | 2 (`CUDA 0,1`) | `.../joint_2s_norope/` | **Done** |
| **rope_blk0** | `--rope 1 --only-block0 1` | 2 (`CUDA 2,3`) | `.../joint_2s_rope/` | **Done** |
| **rope_all** | `--rope 1 --only-block0 0` | **4** (`CUDA 0–3`) | `.../joint_2s_rope_all/` | Running after the two above |

Root out: `/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_2s/`  
Each run writes `loss_steps.csv`, `loss_epoch.csv`, `best.pt`, `metrics.json`, `DONE`.  
Curves (repo root): `kvprune_joint_2s_loss_curves.png` (step axis aligned to global batch=2).

### Best val action Top-5 (completed arms)

| Arm | Best Top-5 | @ epoch |
|---|---:|---:|
| **No RoPE** | **31.59%** | 8 |
| **RoPE (blk0 only)** | **31.44%** | 7 |

`only_block0` RoPE did **not** beat no-RoPE. `rope_all` is the next check (RoPE on every Probe self-attn block, 4-GPU).

## How to launch

### 1) Legacy 2s ablation (norope + rope_blk0), 2+2 GPUs

```bash
# no RoPE — GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29701 $PY -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/run_local_kvprune_joint_2s.py \
  --horizon 2 --rope 0 \
  --checkpoint /mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt \
  --out-dir /mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_2s \
  --epochs 8 --batch-size 1 --num-workers 2

# RoPE on Probe.blocks[0] only — GPUs 2,3
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29702 $PY -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/run_local_kvprune_joint_2s.py \
  --horizon 2 --rope 1 --only-block0 1 \
  --checkpoint /mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt \
  --out-dir /mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_2s \
  --epochs 8 --batch-size 1 --num-workers 2
```

### 2) 2s RoPE on **all** Probe blocks (4 GPUs)

Waits for the legacy jobs above, then starts:

```bash
bash scripts/run_local_kvprune_joint_2s_rope_all.sh
```

Or manual:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29711 $PY -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  scripts/run_local_kvprune_joint_2s.py \
  --horizon 2 --rope 1 --only-block0 0 \
  --checkpoint /mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt \
  --out-dir /mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_2s \
  --epochs 8 --batch-size 1 --num-workers 2
```

Note: 4-GPU global batch=4 vs 2-GPU global batch=2 → half as many `global_step`s per epoch.
Compare arms by **epoch** (or align step × `global_batch/2`).

## Trainable params (all arms)

- Encoder: LoRA on **last 12** ViT blocks (`attn.qkv`, `attn.proj`, r=8, α=16)
- Probe: full AttentiveClassifier (pooler + heads)
- Early encoder blocks: forward under `no_grad` (memory)
- No predictor / no encoder QK score refresh (probe scores drive prune)

## Next: rope_all + Top-K history protect prune

On top of **rope_all**, change the drop rule only:

1. Each tick: score probe-blk0 importance on the packed **128**, sort, record Top-K
   absolute slot ids (`K=34` frames → 17 slots).
2. Next tick: drop 34 frames by walking scores low→high, **skipping** ids that were
   Top-K in either of the **previous 2** score passes (`protect_hist=2`).

Running recipe: `--stream-steps 3 --protect-hist 2 --protect-k 34`
(`T=128+3×34=230`; probe always sees 128). Cold start `vitl.pt`. Compare to
`joint_2s_rope_all/` (best **35.59%** @ep7).

```bash
bash scripts/run_local_kvprune_joint_2s_rope_all_protect.sh
# out: .../joint_2s_rope_all_protectk34_h2_s3/
# log: .../logs/kvprune_joint_2s_rope_all_protect.log
```

## CLI cheat sheet

```bash
scripts/run_local_kvprune_joint_2s.py \
  --checkpoint /mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt \
  --horizon {2|6} \
  --rope {0|1} \
  --only-block0 {0|1} \   # 0 = all probe blocks (default); 1 = blocks[0] only
  --stream-steps N \      # T=128+N*34; default 1 (same as rope_all)
  --protect-hist H \      # 0=off; >0 enable Top-K history protect
  --protect-k 34 \        # Top-K size in frames
  --out-dir /mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_{H}s \
  --epochs 8 --batch-size 1
```

Tag → out subdir: `joint_{H}s_norope` / `joint_{H}s_rope` / `joint_{H}s_rope_all` /
`joint_{H}s_rope_all_protectk34_h2_s3`.

## Deferred: 6s

6s w/ and w/o all-block RoPE is **cancelled for now**. Placeholder:
`scripts/run_local_kvprune_joint_6s.sh`.

## Vanilla V-JEPA 2.1 (no KV, 32 frames, 2s) — yh6416

Same as `nokv_f32_h2s` (`--no-kv 1 --frames 32 --rope 0`: full-attention 32-frame
clip → all tokens → probe; encoder LoRA last 12 + probe; no KV cache, prune, probe
RoPE or Gaze+IMU) with the backbone switched to **V-JEPA 2.1 ViT-L @384**
(`vjepa2_1_vitl_dist_vitG_384.pt`, `ema_encoder`, official 2.1 EK100
`pretrain_kwargs`).

```bash
# weights: scripts/download_vjepa2_1_vitl384_checkpoint.sh
sbatch --export=NO_KV=1,FRAMES=32,HORIZON=2,ROPE=0,NGPUS=2,EPOCHS=8,VJEPA21=1 \
  scripts/submit_kvprune_joint_ll5914.slurm      # out: nokv_f32_h2s_v21_384/
# or directly: run_local_kvprune_joint_2s.py --no-kv 1 --frames 32 --rope 0 \
#   --vjepa21 1 --img-size 384 --checkpoint .../vjepa2_1_vitl_dist_vitG_384.pt
```

`--vjepa21 1` requires `--no-kv 1`. The stream KV encoder
(`StreamKVAttnPruneEncoder`) re-implements V-JEPA 2.0 blocks: 2.0 RoPE frequency
layout (`repeat`; 2.1 uses `repeat_interleave`), no 2.1 video modality embedding,
`encoder.norm` instead of 2.1 `norms_block[-1]`. Running 2.1 weights through it
does not crash but gives different features (max-rel 0.62 vs native).

Result (NYU Torch, 2×H200, batch 1/GPU, 8 epochs, lr 2e-5 / LoRA ×0.5, seed 0;
clip_split val n=1331, Top-5 / Top-1 %, verb/noun re-evaluated from best.pt):

| Backbone | Best epoch | Action | Verb | Noun |
|---|---:|---|---|---|
| V-JEPA 2.1 @384 | 5 | **38.54** / 13.52 | 76.86 / 25.70 | 65.59 / 29.45 |
| V-JEPA 2.0 @256 (`nokv_f32_h2s`) | 5 (of 5 run) | 35.54 / 12.02 | 75.21 / 24.87 | 60.86 / 27.95 |

2.1 per-epoch action Top-5: 20.81, 28.17, 34.03, 35.09, 38.54, 36.96, 36.21,
35.61 (peaks at epoch 5, then overfits). 2.0 verb/noun (and 35.76 action on
re-eval vs 35.54 logged) come from re-scoring its best.pt with this branch's code.
Checkpoint: `/scratch/yh6416/share_ll5914/b21_vanilla_vjepa21_384_f32_h2s/`.

### Vanilla V-JEPA 2.1 on `clip_split_matched` (9 participants) + metric script

Same recipe on the P01-sized 9-participant split (`clip_split_matched`, train 5537 / val 1400
rows, 64 videos):

```bash
sbatch --export=NO_KV=1,FRAMES=32,HORIZON=2,ROPE=0,NGPUS=2,EPOCHS=8,VJEPA21=1,SPLIT=matched \
  scripts/submit_kvprune_joint_ll5914.slurm      # out: nokv_f32_h2s_v21_384_matched/
```

`SPLIT=matched` stages the 64 videos in `stage_videos.txt` (as the all-P Gaze+IMU launcher)
and is wired for `NO_KV=1` only.

`scripts/eval_kvprune_nokv_vna.py --ckpt RUN/best.pt --video-root VIDEOS [--val-csv CSV]`
rebuilds a `--no-kv` checkpoint (2.0 or 2.1) from its saved args and reports, per verb / noun /
action: micro Top-1/3/5, class-mean recall@1/@5, and verb/noun derived from the action head
(single GPU, no sampler padding, trainer's `map_labels` filter).

Results (best.pt, val, %; micro Top-5 / Top-3 / Top-1):

| Run | split (val n) | Action | Verb | Noun |
|---|---|---|---|---|
| V-JEPA 2.1 @384, ep5 | matched (1215) | 30.54 / 23.29 / 11.03 | 72.18 / 59.01 / 27.49 | 62.80 / 49.96 / 27.65 |
| V-JEPA 2.1 @384, ep5 | P01 clip_split (1330) | 38.57 / 29.40 / 13.53 | 76.84 / 61.28 / 25.71 | 65.64 / 54.59 / 29.47 |
| V-JEPA 2.0 @256 `nokv_f32_h2s`, ep5 | P01 clip_split (1330) | 35.71 / 28.12 / 12.03 | 75.19 / 61.28 / 24.89 | 60.90 / 50.23 / 27.97 |

Matched per-epoch action Top-5 (2.1): 17.45, 23.95, 27.08, 30.21, 30.54, 30.12, 30.04, 30.12.
Class-mean recall@5 (verb / noun / action): matched 2.1 26.77 / 45.92 / 19.04; P01 2.1
35.89 / 52.09 / 23.43; P01 2.0 34.81 / 52.56 / 22.33. Checkpoint:
`/scratch/yh6416/share_ll5914/b21_vanilla_vjepa21_384_f32_h2s_matched/`.
