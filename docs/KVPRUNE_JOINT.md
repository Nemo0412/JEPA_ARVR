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
