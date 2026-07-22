# JEPA_ARVR

V-JEPA2 action anticipation experiments on **EGTEA** and **HD-EPIC (P01)**, with optional gaze / SLAM-pose inputs and encoder/predictor LoRA fine-tuning.

Upstream backbone lives in the `vjepa2` submodule / external V-JEPA2 tree. Project-local training entrypoint:

```text
eval_name: app.hdepic_lora_action_anticipation
```

---

## Best Val Action Top-5 (as of 2026-07-17)

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
| Video **joint v2** (predictor + **heads**; pooler frozen) | **40.44%** @ep3 | Completed (early-stop); depth=12 baseline |
| Video joint **predictor depth −2** (depth=10, LoRA+heads) | **40.36%** @ep2 | Completed (early-stop @ep5); job `13593917` |
| Video joint **predictor depth +2** (depth=14, LoRA+heads, copy-init) | **40.74%** | Incomplete: killed ~2h low-GPU-util (`SIGNAL Terminated`, job `13593918`); best so far @ resumed ep1 / logged as epoch 3 |
| Gaze+pose stage-1 (gaze map + SLAM pose matrix + encoder LoRA) | 39.16% | Pose = SLAM `pose_6d` (IMU-fused trajectory; not raw IMU CSV) |
| Gaze+pose stage-2 (predictor LoRA only) | 40.59% | Previous P01 best (completed) |
| Gaze+pose **joint v2** (predictor + **heads**; pooler frozen) | **42.74%** @ep2 | Previous P01 leader; early-stopped @ep5 (patience 3) |
| Gaze+pose joint v1 (predictor + **full probe** @ 2e-5) | 37.42% → 28.02% | **Collapsed**; early-stopped @ep5 (forgetting, not NaN) |
| Tri-modal soft-FT (enc LoRA+heads+fusion from video joint) | **39.82%** → 39.58% | Declining; util-killed @ep3 (`13594209`) |
| Tri-modal **fusion-only** (freeze backbone; cold from video joint) | **39.80%** peak | Below video joint; early-stopped @ep5 after frame-cache resume chain |
| Tri-modal **joint** (fusion + pred LoRA + heads from video joint) | **40.85%** @ep1 | First pure-RGB tri-modal above video 40.44%; early-stopped @ep5 |
| **Idea 1 hybrid** (concat + late IMU cross-attn) | **43.30%** @ep2 | **New P01 leader** (+0.56 vs concat 42.74%); training continues |

**P01 leader:** Idea 1 hybrid (**concat + late IMU cross-attn**) **43.30%** @ep2.  
**Previous concat baseline:** gaze+pose joint v2 **42.74%**.  
**Video baseline:** joint v2 **40.44%**.

#### Tri-modal / hybrid — status 2026-07-22

Architecture notes:
- Pure tri-modal: `ProjectedTriModalCrossAttention` on RGB encoder output (`gaze.mode=projected_tri_modal_cross_attention`).
- **Idea 1 hybrid:** `gaze.mode=concat_plus_cross_attn` — keep **5ch gaze+pose concat** backbone, add **IMU-only late cross-attn** before predictor.

| Recipe | Trainable | Best Top-5 | Outcome |
|---|---|---:|---|
| Soft-FT from video joint | Fusion + enc LoRA + heads | 39.82% | Video path drifted; worse than 40.44% |
| Fusion-only cold (LR 1e-4, bs16) | Fusion only | 39.28% | Still below video; val dropped ep1→3; util-kill AveUtil≈36% |
| Fusion-only resume (LR **1e-5**, bs**32**) | Fusion only from best | **39.80%** peak | Early-stopped @ep5 below video 40.44%; floor held at 39.82% |
| **Joint** from video 40.44% | Fusion + **pred LoRA** + **heads** | **40.85%** @ep1 | First pure-RGB tri-modal > video; early-stopped @ep5 |
| **Joint A "unbrake"** | Same + fusion LR **2e-4**, gate **-1**, jitter | ≤**40.77%** | Failed: never beat floor 40.85% |
| **Joint B "deep+keepaux"** | Fusion **L=3** + **keep aux tokens** | **40.90%** @ep2 | Still far from concat 42.74%; superseded by Idea 1 |
| **Idea 1 hybrid** (running) | Frozen concat adapter/enc LoRA; train **IMU CA + pred LoRA + heads** | **43.30%** @ep2 | **Beats concat**; ep1 42.47 → ep2 **43.30** → ep3 42.62 (patience 1/5) |

**Does Idea 1 use cross-attention?** Yes — but only as a **late IMU refinement** on top of the proven concat backbone (gaze already enters via 5ch adapter; CA does not re-inject gaze).

Why Idea 1 can warm from 42.74% while pure tri-modal cannot: encoder still sees **adapter-fused RGB** (same input distribution as concat joint). Pure-RGB tri-modal cannot load that checkpoint.

Why pure late cross-attn lost to concat (A/B): pathway depth — concat routes gaze through all encoder+predictor layers; pure CA injects once after encoder.

```text
# NEWEST: Idea 1 hybrid — concat(42.74%) + late IMU cross-attn
scripts/submit_b12_concat_plus_cross_attn_from_gazepose42_1xh100.slurm
# Run dir:
/scratch/ll5914/experiments/concat_plus_cross_attn_from_gazepose42/action_anticipation_frozen/concat-plus-ca-from-gazepose42-vitl16-256-12ep-1xh100/

# NEWEST: joint B (deep fusion + keep aux) warm from tri-modal joint 40.85%
scripts/submit_b12_tri_modal_jointB_deep_keepaux_1xh100.slurm
# Run dir:
/scratch/ll5914/experiments/tri_modal_jointB_deep_keepaux/action_anticipation_frozen/tri-modal-jointB-deep-keepaux-vitl16-256-15ep-1xh100/

# Plan A (failed unbrake) — kept for reference
scripts/submit_b12_tri_modal_jointA_unbrake_1xh100.slurm

# Joint tri-modal (pred LoRA + heads + fusion) from video joint 40.44% → 40.85%
scripts/submit_b12_tri_modal_joint_from_p01video_jointv2_1xh100.slurm

# Legacy fusion-only (failed to beat 40.44%)
scripts/submit_b12_tri_modal_s2_from_p01video_jointv2_1xh100.slurm

# Optional CPU prefill (no GPU → no util-kill) before/while GPU runs
scripts/submit_prefill_p01_clip_frame_cache.slurm
```

#### How we raise GPU utilization (tri-modal / decode-bound jobs)

Cluster cancels jobs with **AveUtil &lt; 60% for &gt;2h**. Tri-modal is decode-bound: `decord.get_batch` ≈10–20s/batch vs GPU step ≈4s → theoretical util ≈20–30% even with async prefetch.

**Honest status: mitigated, not “forever ≥60%”.** Two complementary layers:

1. **Actually raise util (primary):** scratch **decoded-clip cache** + **fixed anticipation 1.0s** so cache keys are deterministic. After prefill (~7k `.npy` for P01), decode → `np.load` (tens of ms) and the GPU can stay busy. This is what can push AveUtil toward/above 60%.
2. **Survive util-kill / walltime (safety net):** `#SBATCH --time=01:50:00` (under the 2h window) + `USR1/TERM` auto-`sbatch` of the same script. Even if a chunk’s AveUtil is still low (cold cache, first epoch, jitter on), training **continues** on the next job with `latest.pt` + warm cache.

| Layer | What we do | Effect |
|---|---|---|
| **Decoded-clip cache** | `TRI_MODAL_FRAME_CACHE` on scratch; uint8 `(T,H,W,C)` keyed by `video_id`+frame indices (`clip_frame_cache.py`) | After prefill/warmup, decode → `np.load` → GPU can stay busy (**target AveUtil ≥60%**) |
| **Cache-aligned train horizon** | `train_anticipation_time_sec=[1.0,1.0]` (same as val / `phd_reference` point) | Deterministic indices → cache keys match prefill |
| **CPU prefill** | `submit_prefill_p01_clip_frame_cache.slurm` | Warms cache without holding a GPU |
| **Prefetch RAM budget** | `num_workers=16`, `prefetch_factor=2`, `TRI_MODAL_PREFETCH_DEPTH=4`, `pin_memory=False` | Avoid ~60GB in-flight float batches that stalled workers (prior util≈19%) |
| **Wall-clock dodge** | `#SBATCH --time=01:50:00` + `USR1/TERM` auto-`sbatch` resume | Survives util-kill / timeout; next chunk inherits warm cache + `latest.pt` |
| **Best-metric floor** | Always restore best tracker + `best_metric_floor` | Weaker vals cannot overwrite a stronger `best.pt` after resume |

**Plan A vs Plan B tradeoff:** Plan A restored train anticipation jitter `[0.25,1.75]` and therefore **disabled** the frame cache (random horizons → unbounded/miss keys) — util stayed decode-bound and relied mainly on the 1:50 auto-resubmit dodge. Plan B **re-enables** the frame cache with fixed 1.0s, so it is the stabler util recipe.

Logs to watch: `decode:` / `batch_wait:` should fall well below `step:` once `TRI_MODAL_FRAME_CACHE stats: hit_rate` is high.
#### Best P01 checkpoint (saved on scratch)

```text
# NEW leader (Idea 1 hybrid) — 43.30% @ep2
/scratch/ll5914/experiments/concat_plus_cross_attn_from_gazepose42/action_anticipation_frozen/concat-plus-ca-from-gazepose42-vitl16-256-12ep-1xh100/
  best.pt
  predictor_lora_best.pt
  encoder_lora_best.pt
  binary_input_adapter_best.pt
  tri_modal_fusion_best.pt          # late IMU cross-attn
  topk_log_r0.csv

# Previous concat leader — 42.74% @ep2
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
4. **Tri-modal val crash:** `ClipBalancedDecodeVideosToClips.mtp_horizons_sec` AttributeError in workers — defensive `getattr` (MTP unused for 1s tri-modal).
5. **Tri-modal util-kill:** video decode ≫ GPU step. Not “always ≥60%”; mitigated by (a) scratch **decoded-clip cache** + fixed 1.0s anticipation to actually raise util, (b) **1:50** auto-resubmit chunks so training survives a low-util segment. Plan A (jitter) turned cache off; Plan B turns it back on — see “How we raise GPU utilization” above.

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

Same joint recipe as video joint (**heads + predictor LoRA on all blocks**, pooler/encoder frozen). Only `predictor.depth` changes.

| | Depth | Init | Train | Script / out | Best Top-5 |
|---|---:|---|---|---|---:|
| **Baseline (video joint)** | 12 | Standard ckpt | Heads + predictor **LoRA** (all blocks) | `submit_p01_video_pred_joint_ll5914.slurm` | **40.44%** @ep3 |
| **−2** | 10 | `strict=False`: keep blocks `0..9`, drop 10–11 | Heads + predictor **LoRA** (all 10) | `submit_p01_video_joint_depth10_ll5914.slurm` → `video_joint_depth10/` | **40.36%** @ep2 (completed) |
| **+2** | 14 | Load 12-block ckpt; **copy-init** new blocks from block 11 | Heads + predictor **LoRA** (all 14) | `submit_p01_video_joint_depth14_ll5914.slurm` → `video_joint_depth14/` | **40.74%** (partial; util-killed) |

```yaml
# Joint ±n (identical to video joint except depth / copy-init)
model_kwargs.pretrain_kwargs.predictor.depth: 10  # or 14
train_heads: true
freeze_pooler: true
pretrained_probe: <stage1>/best.pt
encoder_lora: { freeze: true, load_checkpoint_path: <stage1>/encoder_lora_best.pt }
predictor_lora:
  enabled: true
  last_n_blocks: 0                 # LoRA on every predictor block
  copy_init_from_pretrained: 12    # only if depth > 12; else omit / 0
```

Depth field path: `model_kwargs.pretrain_kwargs.predictor.depth` only.  
Code: `predictor_lora.py` (`copy_init_extra_predictor_blocks`).

**Not comparable:** `submit_p01_predictor_arch_depth{10,12,14}_*.slurm` (`arch_depth*_fullpred/`) freeze heads and full-FT the predictor.

### Encoder ±n when baseline is video joint (encoder depth 24)

**Must re-finetune the encoder after dropping/adding layers** (do not freeze the old depth-24 Stage-1 LoRA onto a truncated trunk).

| Step | Train | Freeze |
|---|---|---|
| **Stage-1 @ new depth** | Probe + **encoder LoRA** | Predictor pretrained |
| **Joint** | Predictor LoRA (all blocks) + **heads** | **New** encoder LoRA + pooler |

| | Encoder depth | Init | Scripts |
|---|---:|---|---|
| **Baseline (video joint)** | 24 | Standard | Stage-1 `p01_video_enc_clip` → joint **40.44%** |
| **−2** | 22 | `strict=False` / truncate last 2 blocks | `submit_p01_video_enc_depth22_ll5914.slurm` → `submit_p01_video_joint_enc_depth22_ll5914.slurm` |
| **+2** | 26 | Requires factory depth override, then `copy_init_from_pretrained: 24` | Not scripted yet |

```yaml
# Stage-1 (−2 example)
encoder_lora: { enabled: true, last_n_blocks: 0, arch_depth: 22 }  # train LoRA
predictor_lora: { enabled: false }

# Joint (−2): identical to video joint except encoder arch_depth
train_heads: true
freeze_pooler: true
pretrained_probe: <enc_depth22_stage1>/best.pt
encoder_lora: { freeze: true, arch_depth: 22, load_checkpoint_path: <enc_depth22_stage1>/encoder_lora_best.pt }
predictor_lora: { enabled: true, last_n_blocks: 0 }
```

Code: `encoder_lora.py` (`apply_encoder_arch_depth`,
`copy_init_extra_encoder_blocks`). For encoder **−n**, keep the pinned upstream
factory at its native depth, load the full checkpoint, and use `arch_depth` to
drop suffix blocks before LoRA injection. Do not set
`model_kwargs.pretrain_kwargs.encoder.depth`: the pinned upstream `vit_large` /
`vit_large_rope` factories hard-code their native depth and reject a duplicate
`depth` keyword. Encoder **+n** will require an explicit upstream/factory change
before `copy_init_from_pretrained` can be used.

### Naming

| Name | Means |
|---|---|
| **Stage-1** | Probe + encoder LoRA |
| **Stage-2 / predictor-only** | Predictor LoRA; enc/probe frozen |
| **Joint** | Predictor LoRA (all blocks) + heads; pooler frozen |
| **Joint ±n** | Same as joint (predictor LoRA + heads), predictor depth `12±n` (+ copy-init if deeper) |
| **Encoder ±n** | Stage-1 probe+encoder LoRA at encoder depth `24±n`; then joint (freeze that encoder LoRA) |

Gaze+pose: `binary_input_adapter_gaze_pose_matrix` (RGB + gaze map + SLAM `pose_6d` patch).

---

## Quick pointers

- Submit scripts: `scripts/submit_egtea_*.slurm`, `scripts/submit_p01_*.slurm`
- Clip-split builder: `scripts/make_hdepic_clip_split.py`
- Older HPC run index: `docs/RECENT_RUNS.md`
- HD-EPIC CSV adapter notes: `scripts/README_hdepic_action_anticipation.md`

Experiment artifacts (checkpoints, logs) live under `/scratch/.../experiments/` on the cluster and are **not** committed here.
