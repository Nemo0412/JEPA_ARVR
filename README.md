# JEPA_ARVR

V-JEPA2 action anticipation experiments on **EGTEA** and **HD-EPIC (P01)**, with optional gaze / SLAM-pose inputs and encoder/predictor LoRA fine-tuning.

Upstream backbone lives in the `vjepa2` submodule / external V-JEPA2 tree. Project-local training entrypoint:

```text
eval_name: app.hdepic_lora_action_anticipation
```

---

## Best Val Action Top-5 (as of 2026-07-12)

Metric: **val action Top-5 accuracy** from `topk_log_r0.csv` (preferred).  
Backbone: **ViT-L/16 @ 256**, horizon ≈ **1s**, temporal sampling `phd_reference` unless noted.

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
| Video stage-1 (probe + encoder LoRA) | **38.10%** | |
| Video stage-2 (predictor LoRA only) | 37.87% | Short / incomplete log |
| Video joint (predictor LoRA + probe @ small LR) | **38.85%** (csv) / 38.62% (best-tracker @ep2) | **In progress** (resume running) |
| Gaze+pose stage-1 (gaze map + SLAM pose matrix + encoder LoRA) | **39.16%** | Pose = SLAM `pose_6d` (IMU-fused trajectory; not raw IMU CSV) |
| Gaze+pose stage-2 (predictor LoRA only) | **40.59%** | Best completed P01 Top-5 so far |
| Gaze+pose joint (predictor LoRA + probe @ small LR) | 37.42% | **In progress** / partial; below stage-1 so far |

**Running / pending (not in table above as final):** P01 predictor `last_n_blocks` depth sweep (video-only); tri-modal stage-2; joint resumes.

---

## Setting definitions (short)

| Name | Trainable | Inputs |
|---|---|---|
| **Stage-1** | Probe (+ heads) + encoder LoRA | Video, or video+gaze(+pose) |
| **Stage-2 predictor-only** | Predictor LoRA only; encoder LoRA + probe frozen from stage-1 | Same as stage-1 |
| **Joint** | Predictor LoRA + probe (small LR); encoder LoRA frozen | Same as stage-1 |

Gaze+pose path: `binary_input_adapter_gaze_pose_matrix` (RGB + binary gaze map + inter-frame SLAM pose patch → 5-channel adapter).

---

## Quick pointers

- Submit scripts: `scripts/submit_egtea_*.slurm`, `scripts/submit_p01_*.slurm`
- Clip-split builder: `scripts/make_hdepic_clip_split.py`
- Older HPC run index: `docs/RECENT_RUNS.md`
- HD-EPIC CSV adapter notes: `scripts/README_hdepic_action_anticipation.md`

Experiment artifacts (checkpoints, logs) live under `/scratch/.../experiments/` on the cluster and are **not** committed here.
