# JEPA_ARVR

V-JEPA2 action anticipation experiments on **EGTEA** and **HD-EPIC (P01)**, with optional gaze / SLAM-pose inputs and encoder/predictor LoRA fine-tuning.

Upstream backbone lives in the `vjepa2` submodule / external V-JEPA2 tree. Project-local training entrypoint:

```text
eval_name: app.hdepic_lora_action_anticipation
```

---

## Best Val Action Top-5 (as of 2026-07-13)

Metric: **val action Top-5 accuracy** from `topk_log_r0.csv` (preferred).  
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

> Note: joint **40.44% / 42.74%** are not a fair depth comparison against arch-depth runs (~39%). Joint also trains classifier heads + predictor LoRA; arch-depth freezes the probe and only trains the predictor (see recipes below).

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
2. **Gaze joint collapse:** full 50M probe + predictor LoRA caused action forgetting (train action↓, verb/noun↑). Joint v2 keeps pooler frozen and trains **heads only**.
3. **Arch depth14 `KeyError: 'predictor'`:** depth must be set under `model_kwargs.pretrain_kwargs.predictor`, not `model_kwargs.predictor`.

---

## Training recipes

Default backbone checkpoint: V-JEPA2 ViT-L predictor has **12** blocks. Entry point:

```text
eval_name: app.hdepic_lora_action_anticipation
```

### 1) Stage-1 → predictor / joint (standard)

| Stage | Trainable | Frozen | Typical script |
|---|---|---|---|
| **Stage-1** | Probe (+ heads) + encoder LoRA | Predictor (pretrained) | `scripts/submit_p01_*` stage-1 / EGTEA video or gaze |
| **Stage-2 predictor-only** | Predictor LoRA | Encoder LoRA + probe (load stage-1 best) | `submit_p01_*_predictor_*.slurm` / EGTEA `*_predictor_*` |
| **Joint v2 (preferred)** | Predictor LoRA + **classifier heads** | Encoder LoRA; **pooler** (`freeze_pooler=True`) | `submit_p01_video_pred_joint_ll5914.slurm`, `submit_p01_gazepose_pred_joint_ll5914.slurm` |
| **Joint v1 (legacy)** | Predictor LoRA + **full probe** @ small LR | Encoder LoRA | Avoid on P01 gaze — collapses |

**Joint v2 knobs (conceptually):**

```yaml
experiment.lora:
  train_heads: true
  freeze_pooler: true          # do NOT train the full ATTPooler
  load_probe_heads: true
  pretrained_probe: <stage1>/best.pt
  encoder_lora:
    freeze: true
    load_checkpoint_path: <stage1>/encoder_lora_best.pt
  predictor_lora:
    enabled: true
    last_n_blocks: 2           # or more; LoRA on last N of the 12 blocks
```

Warm-start from stage-1 `best.pt` + `encoder_lora_best.pt`. Metric / early-stop: **`val-action-top5`**.

### 2) Changing predictor depth (+2 / −2 / baseline 12)

Used to ablate **architecture depth**, not LoRA coverage. Correct recipe: after changing depth, **fully retrain the entire predictor** (not only the last 2 blocks). Encoder LoRA + probe stay frozen from video stage-1.

| Variant | Depth | Init | Train |
|---|---:|---|---|
| **−2** | 10 | Build 10 blocks; `strict=False` load keeps pretrained blocks **0..9**, drops **10..11** | Full-FT **all** predictor params |
| **12 (control)** | 12 | Standard ckpt; all keys match | Full-FT **all** predictor params |
| **+2** | 14 | Build 14 blocks; load 12 with `strict=False` (new blocks missing); **copy-init** blocks 12–13 from block 11 | Full-FT **all** predictor params |

**Config path for depth (important):**

```yaml
model_kwargs:
  pretrain_kwargs:
    predictor:
      depth: 10   # or 12 or 14  — NOT model_kwargs.predictor
experiment.lora:
  train_heads: false
  encoder_lora: { freeze: true, load_checkpoint_path: <stage1>/encoder_lora_best.pt, ... }
  predictor_lora:
    enabled: true
    last_n_blocks: -1              # skip LoRA injection
    full_ft_last_n_blocks: 10      # >= num blocks ⇒ entire predictor
    copy_init_from_pretrained: 0   # 12 when depth=14; 0 when depth≤12
```

Code: `copy_init_extra_predictor_blocks` + `set_predictor_full_ft_last_n` in  
`app/hdepic_lora_action_anticipation/predictor_lora.py` (wired from `eval.py`).  
When `full_ft_last_n_blocks >= depth`, the log should say  
`Enabled full fine-tune on ENTIRE predictor`.

**Submit scripts (P01 video-only, current full-pred dirs):**

- `scripts/submit_p01_predictor_arch_depth10_ll5914.slurm` → `.../arch_depth10_fullpred/`
- `scripts/submit_p01_predictor_arch_depth12_ll5914.slurm` → `.../arch_depth12_fullpred/`
- `scripts/submit_p01_predictor_arch_depth14_ll5914.slurm` → `.../arch_depth14_fullpred/`

**Fairness:** compare 10 / 12 / 14 only under this full-pred recipe. Do **not** score them against joint v2 Top-5 without also matching heads+LoRA training.

### Setting definitions (short)

| Name | Trainable | Inputs |
|---|---|---|
| **Stage-1** | Probe (+ heads) + encoder LoRA | Video, or video+gaze(+pose) |
| **Stage-2 predictor-only** | Predictor LoRA only; encoder LoRA + probe frozen from stage-1 | Same as stage-1 |
| **Joint v1 (legacy)** | Predictor LoRA + **full** probe @ small LR | Same as stage-1 |
| **Joint v2 (current)** | Predictor LoRA + **classifier heads** @ small LR; pooler frozen | Same as stage-1 |
| **Arch depth full-pred** | Entire predictor (all blocks); encoder/probe frozen | Video-only so far |

Gaze+pose path: `binary_input_adapter_gaze_pose_matrix` (RGB + binary gaze map + inter-frame SLAM pose patch → 5-channel adapter).

---

## Quick pointers

- Submit scripts: `scripts/submit_egtea_*.slurm`, `scripts/submit_p01_*.slurm`
- Clip-split builder: `scripts/make_hdepic_clip_split.py`
- Older HPC run index: `docs/RECENT_RUNS.md`
- HD-EPIC CSV adapter notes: `scripts/README_hdepic_action_anticipation.md`

Experiment artifacts (checkpoints, logs) live under `/scratch/.../experiments/` on the cluster and are **not** committed here.
