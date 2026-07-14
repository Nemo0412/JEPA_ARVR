# JEPA_ARVR

V-JEPA2 action anticipation experiments on **EGTEA** and **HD-EPIC (P01)**, with optional gaze / SLAM-pose inputs and encoder/predictor LoRA fine-tuning.

Upstream backbone lives in the `vjepa2` submodule / external V-JEPA2 tree. Project-local training entrypoint:

```text
eval_name: app.hdepic_lora_action_anticipation
```

---

## Best Val Action Top-5 (as of 2026-07-13)

Metric: **val action Top-5 accuracy** from `topk_log_r0.csv`.  
Backbone: **ViT-L/16 @ 256**, horizon ≈ **1s**, temporal sampling `phd_reference` unless noted.  
Cluster runs use **1×H100** unless noted.

### EGTEA (split1)

| Setting | Best Top-5 | Notes |
|---|---:|---|
| Video stage-1 (probe + encoder LoRA) | **69.40%** | Best overall on EGTEA |
| Video stage-2 (predictor LoRA only; enc/probe frozen) | 69.37% | ≈ flat vs stage-1 |
| Gaze stage-1 (binary gaze adapter + encoder LoRA) | **68.64%** | From ep5 recovery run; full fixed run peak 65.75% |
| Gaze stage-2 (predictor LoRA only) | 58.11% | Worse than stage-1 |

### HD-EPIC P01 (`clip_split`)

| Setting | Best Top-5 | Notes |
|---|---:|---|
| Video stage-1 (probe + encoder LoRA) | 38.10% | |
| Video stage-2 (predictor LoRA only) | 37.87% | Short / incomplete log |
| Video joint v1 (predictor + **full probe** @ 2e-5) | 38.85% csv / 38.62% tracker | Partial; declining; superseded by heads-only joint |
| Video **joint v2** (predictor + **heads**; pooler frozen) | **40.44%** @ep3 | Completed (early-stop) |
| Gaze+pose stage-1 (gaze map + SLAM pose matrix + encoder LoRA) | 39.16% | Pose = SLAM `pose_6d` (IMU-fused trajectory; not raw IMU CSV) |
| Gaze+pose stage-2 (predictor LoRA only) | 40.59% | Previous P01 best (completed) |
| Gaze+pose **joint v2** (predictor + **heads**; pooler frozen) | **42.74%** @ep2 | **Best P01 Top-5**; early-stopped @ep5 (patience 3) |
| Gaze+pose joint v1 (predictor + **full probe** @ 2e-5) | 37.42% → 28.02% | **Collapsed**; early-stopped @ep5 (forgetting, not NaN) |

**P01 leader:** Gaze+pose joint v2 (heads) **42.74%**.

#### Best P01 checkpoint (saved on scratch)

```text
/scratch/ll5914/experiments/p01_gazepose_pred_joint_heads_clip/action_anticipation_frozen/p01-gazepose-pred-joint-heads-vitl16-256-10ep/
  best.pt                      # probe / classifier @ ep2 (42.74%)
  predictor_lora_best.pt
  encoder_lora_best.pt         # frozen warm-start from stage-1
  binary_input_adapter_best.pt
  topk_log_r0.csv
```

### Known issues fixed recently

1. **`--debugmode false` bug:** `argparse` `type=bool` makes `bool("false") is True`, forcing single-GPU debug path. Fixed parser in `vjepa2/evals/main.py` and removed the bad flag from submit scripts; parent now `join()`s workers.
2. **Gaze joint collapse:** full probe + predictor LoRA caused action forgetting. Joint keeps pooler frozen and trains **heads only**.
3. **Arch depth14 `KeyError: 'predictor'`:** depth must be set under `model_kwargs.pretrain_kwargs.predictor`, not `model_kwargs.predictor`.

---

## How we train

Default V-JEPA2 ViT-L predictor depth is **12**. Entry:

```text
eval_name: app.hdepic_lora_action_anticipation
```

### What the reported numbers use

| Step | Trains | Frozen / loaded | Scripts |
|---|---|---|---|
| **Stage-1** | Probe + encoder LoRA | Predictor = pretrained | EGTEA / P01 stage-1 |
| **Stage-2 predictor-only** | Predictor LoRA (all blocks) | Encoder LoRA + probe from stage-1 | `*_predictor_*.slurm` |
| **Joint** | Predictor LoRA (all blocks) + **heads** | Encoder LoRA frozen; **pooler frozen** | see below |

Metric / early-stop: **`val-action-top5`**. Heads LR `2e-5`; predictor LoRA `lr_mult=0.5`. Full-probe joint collapsed → heads-only.

### Joint video (40.44% @ep3) / Joint gaze+pose (42.74% @ep2)

Same recipe, two inputs:

1. **Stage-1** — train probe + encoder LoRA (gaze+pose also trains 5ch input adapter). Predictor frozen.
2. **Joint** — load stage-1; **freeze** encoder LoRA (+ adapter if gaze); **freeze pooler**; train **predictor LoRA (all blocks)** + **heads**.

| | Stage-1 → Joint | Batch | Extra |
|---|---|---:|---|
| **Video** | `p01_video_enc_clip` → `submit_p01_video_pred_joint_ll5914.slurm` | 16 | `gaze.mode: none` |
| **Gaze+pose** | `p01_gazepose_clip` → `submit_p01_gazepose_pred_joint_ll5914.slurm` | 6 | load+freeze `binary_input_adapter_best.pt` |

```yaml
# Joint (both)
train_heads: true
freeze_pooler: true
pretrained_probe: <stage1>/best.pt
encoder_lora: { freeze: true, load_checkpoint_path: <stage1>/encoder_lora_best.pt }
predictor_lora: { enabled: true, last_n_blocks: 0 }  # all blocks
```

### Depth ±n when baseline is video joint

Keep the **same joint setup** (heads on, pooler frozen, encoder frozen). Only change predictor depth. After changing depth, **full-FT the entire predictor** (all blocks + embeds), not only the last 2 layers.

| | Depth | Init | Train | Script / out |
|---|---:|---|---|---|
| **Baseline (video joint)** | 12 | Standard ckpt | Heads + predictor **LoRA** (all blocks) | `submit_p01_video_pred_joint_ll5914.slurm` → **40.44%** |
| **−2** | 10 | `strict=False`: keep blocks `0..9`, drop 10–11 | Heads + **full-FT entire** predictor | `submit_p01_video_joint_depth10_ll5914.slurm` → `video_joint_depth10/` |
| **+2** | 14 | Load 12-block ckpt; **copy-init** new blocks from block 11 | Heads + **full-FT entire** predictor | `submit_p01_video_joint_depth14_ll5914.slurm` → `video_joint_depth14/` |

```yaml
# Joint ±n (same as video joint + depth change)
model_kwargs.pretrain_kwargs.predictor.depth: 10  # or 14
train_heads: true
freeze_pooler: true
pretrained_probe: <stage1>/best.pt
encoder_lora: { freeze: true, load_checkpoint_path: <stage1>/encoder_lora_best.pt }
predictor_lora:
  last_n_blocks: -1              # no LoRA
  full_ft_last_n_blocks: <depth> # >= depth ⇒ entire predictor
  copy_init_from_pretrained: 12  # only if depth > 12; else 0
```

Depth field path: `model_kwargs.pretrain_kwargs.predictor.depth` only.  
Code: `predictor_lora.py` (`copy_init_extra_predictor_blocks`, `set_predictor_full_ft_last_n`).  
Expect log: `Enabled full fine-tune on ENTIRE predictor`.

**Not comparable:** `submit_p01_predictor_arch_depth{10,12,14}_*.slurm` (`arch_depth*_fullpred/`) freeze heads/probe and only FT the predictor.

### Naming

| Name | Means |
|---|---|
| **Stage-1** | Probe + encoder LoRA |
| **Stage-2 / predictor-only** | Predictor LoRA; enc/probe frozen |
| **Joint** | Predictor LoRA (all blocks) + heads; pooler frozen |
| **Joint ±n** | Same as joint, but depth `12±n` and full-FT entire predictor |

Gaze+pose: `binary_input_adapter_gaze_pose_matrix` (RGB + gaze map + SLAM `pose_6d` patch).

---

## Quick pointers

- Submit scripts: `scripts/submit_egtea_*.slurm`, `scripts/submit_p01_*.slurm`
- Clip-split builder: `scripts/make_hdepic_clip_split.py`
- Older HPC run index: `docs/RECENT_RUNS.md`
- HD-EPIC CSV adapter notes: `scripts/README_hdepic_action_anticipation.md`

Experiment artifacts (checkpoints, logs) live under `/scratch/.../experiments/` on the cluster and are **not** committed here.
