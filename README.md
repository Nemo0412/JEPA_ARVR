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
| Video **joint v2** (predictor + **heads**; pooler frozen) | **40.44%** @ep3 | Incomplete (killed ~2–2.5h); resume in flight |
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

### Predictor depth sweep (video-only, `last_n_blocks`; incomplete)

Encoder LoRA + probe frozen from video stage-1; only predictor LoRA trains. Killed ~2h; resume in flight.

| `last_n_blocks` | Best Top-5 so far | Vals logged |
|---:|---:|---|
| 1 | 39.37% | 6 |
| 2 | 39.52% | 6 |
| 4 | **39.83%** | 2 |
| 6 | 39.68% | 2 |
| 8 | 39.37% | 2 |
| 12 | 39.29% | 2 |

Arch depth 12→14 (full-FT last 2 blocks): config path fixed (`pretrain_kwargs.predictor.depth`); resubmitted.

### In flight (2026-07-13)

| Job family | Status |
|---|---|
| Video joint v2 resume | Queued / resume from `latest.pt` |
| Predictor depth sweep resume | Queued / resume |
| Predictor arch depth14 | Queued (bugfix) |
| Tri-modal stage-2 (1×H100) | Running |

### Known issues fixed recently

1. **`--debugmode false` bug:** `argparse` `type=bool` makes `bool("false") is True`, forcing single-GPU debug path. Fixed parser in `vjepa2/evals/main.py` and removed the bad flag from submit scripts; parent now `join()`s workers.
2. **Gaze joint collapse:** full 50M probe + predictor LoRA caused action forgetting (train action↓, verb/noun↑). Joint v2 keeps pooler frozen and trains **heads only**.
3. **Arch depth14 `KeyError: 'predictor'`:** depth must be set under `model_kwargs.pretrain_kwargs.predictor`, not `model_kwargs.predictor`.

---

## Setting definitions (short)

| Name | Trainable | Inputs |
|---|---|---|
| **Stage-1** | Probe (+ heads) + encoder LoRA | Video, or video+gaze(+pose) |
| **Stage-2 predictor-only** | Predictor LoRA only; encoder LoRA + probe frozen from stage-1 | Same as stage-1 |
| **Joint v1 (legacy)** | Predictor LoRA + **full** probe @ small LR | Same as stage-1 |
| **Joint v2 (current)** | Predictor LoRA + **classifier heads** @ small LR; pooler frozen | Same as stage-1 |

Gaze+pose path: `binary_input_adapter_gaze_pose_matrix` (RGB + binary gaze map + inter-frame SLAM pose patch → 5-channel adapter).

---

## Quick pointers

- Submit scripts: `scripts/submit_egtea_*.slurm`, `scripts/submit_p01_*.slurm`
- Clip-split builder: `scripts/make_hdepic_clip_split.py`
- Older HPC run index: `docs/RECENT_RUNS.md`
- HD-EPIC CSV adapter notes: `scripts/README_hdepic_action_anticipation.md`

Experiment artifacts (checkpoints, logs) live under `/scratch/.../experiments/` on the cluster and are **not** committed here.
