# kvprune joint — stream KV + probe-blk0 prune (+ optional Probe RoPE)

Local multi-GPU recipe used on this machine for **HD-EPIC P01 `clip_split`**
action anticipation. Jointly trains **encoder LoRA** (last 12 ViT blocks) and
the **full AttentiveClassifier probe**.

## Protocol

1. Encode history **128 frames** into a per-layer encoder **KV cache**.
2. Score slots with **Probe `blocks[0]` self-attn** received mass (**detached**;
   discrete top-k, no grad through indices).
3. Drop lowest **34** frames / keep **94**; encode new **34** against kept K/V.
4. Packed **128** tokens → probe CE at anticipation horizon `H` seconds.

Prune always uses **block 0** scores. RoPE (when enabled) can cover **block 0
only** or **all Probe self-attn blocks**.

## Checkpoint to start from

| Role | Path |
|---|---|
| **Encoder init (required)** | `/mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt` — official **V-JEPA2 ViT-L/16 @256** (`target_encoder`) |
| Probe / LoRA | **from scratch** each run (no warm probe ckpt) |
| Optional resume | `<out>/best.pt` written by this script (probe + backbone LoRA state) |

CLI flag: `--checkpoint /mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt` (default).

Data (this box):

```text
train CSV : /mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_train_vjepa.csv
val CSV   : /mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_annotations/clip_split/HD_EPIC_val_vjepa.csv
videos    : /mnt/hdd/datasets/HD-EPIC/hdepic_vjepa_videos/
python    : /mnt/hdd/datasets/HD-EPIC/conda_envs/vjepa/bin/python
```

## How we trained (2s, running / completed)

**Goal:** measure RoPE vs no-RoPE under kvprune, horizon **2s**.

| Setting | Value |
|---|---|
| Script | `scripts/run_local_kvprune_joint_2s.py` |
| Horizon | `--horizon 2` |
| RoPE arm | `--rope 1 --only-block0 1` (legacy: **Probe.blocks[0] only**) |
| Control | `--rope 0` |
| GPUs | 2+2 DDP on 4×A6000 (`CUDA_VISIBLE_DEVICES=0,1` / `2,3`) |
| Epochs / BS | 8 / 1 |
| Trainable | encoder LoRA last-12 (`attn.qkv`, `attn.proj`, r=8, α=16) + full probe |
| Out | `/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_2s/joint_2s_{norope,rope}/` |
| Curves | `loss_steps.csv`, `loss_epoch.csv`; plot `kvprune_joint_2s_loss_curves.png` |

Through epoch 5, **only_block0 RoPE ≈ no-RoPE** on val top-5 (~28.4–28.6%).

## Next: 6s with / without RoPE (all Probe self-attn blocks)

**Change vs 2s:** RoPE applies to **every Probe self-attn block**
(`--only-block0 0`, the script default). Still start from the same
`vitl.pt`. Still **scratch** probe+LoRA (do **not** warm from 2s `best.pt`
unless you intentionally want transfer).

```bash
cd /home/lls/workspace/Jepa
export VJEPA_ROOT=/home/lls/workspace/Jepa/vjepa2
export PYTHONPATH=/home/lls/workspace/Jepa:$VJEPA_ROOT
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/mnt/hdd/datasets/HD-EPIC/conda_envs/vjepa/bin/python
OUT=/mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_6s
LOG=/mnt/hdd/datasets/HD-EPIC/experiments/logs
mkdir -p "$OUT" "$LOG"

# no RoPE — GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29701 $PY -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/run_local_kvprune_joint_2s.py \
  --horizon 6 --rope 0 --only-block0 0 \
  --checkpoint /mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt \
  --out-dir "$OUT" --epochs 8 --batch-size 1 --num-workers 4 \
  >> "$LOG/kvprune_joint_6s_norope.log" 2>&1 &

# RoPE on all Probe self-attn blocks — GPUs 2,3
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29702 $PY -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/run_local_kvprune_joint_2s.py \
  --horizon 6 --rope 1 --only-block0 0 \
  --checkpoint /mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt \
  --out-dir "$OUT" --epochs 8 --batch-size 1 --num-workers 4 \
  >> "$LOG/kvprune_joint_6s_rope.log" 2>&1 &
```

Or wait for the 2s jobs to finish, then launch both arms:

```bash
bash scripts/run_local_kvprune_joint_6s.sh
```

Outputs:

```text
$OUT/joint_6s_norope/     # loss_steps.csv, loss_epoch.csv, best.pt, metrics.json
$OUT/joint_6s_rope_all/   # RoPE on all probe self-attn blocks
```

## CLI cheat sheet

```bash
scripts/run_local_kvprune_joint_2s.py \
  --checkpoint /mnt/hdd/jepa/models/vjepa2-vitl/vitl.pt \
  --horizon {2|6} \
  --rope {0|1} \
  --only-block0 {0|1} \   # 0 = all probe blocks (default); 1 = blocks[0] only
  --out-dir /mnt/hdd/datasets/HD-EPIC/experiments/kvprune_joint_{H}s \
  --epochs 8 --batch-size 1
```
