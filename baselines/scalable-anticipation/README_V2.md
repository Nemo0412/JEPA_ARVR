# ScalAnt HD-EPIC baseline, v2 (32s history, MAMBA, annotated-action scoring)

This revision builds on the `Baseline_ScalAnt` HD-EPIC P01 code (`8c3b146`,
config `configs/hdepic/clip_4s_2s.yaml`). It moves the baseline to ScalAnt's
default history and encoder, and it scores every method on the same target.
The old 4s/LSTM config is unchanged and still runs.

New config: `configs/hdepic/clip_32s_2s.yaml`.

## Decisions (agreed 2026-10-15)

1. **Encoder: upstream default.** MAMBA v1 with `D_STATE: 16`, as in upstream
   `configs/ek100/default.yaml`. The old HD-EPIC config used LSTM and did not
   set `D_STATE`; the code default is 64. MAMBA needs `mamba-ssm` and
   `causal-conv1d`. `fla` is optional because the PyTorch fallback for
   RMSNorm/SwiGLU is kept.
2. **History: 32s = 28s long memory + 4s working memory**
   (`TAU_O: 32`, `LONG_MEMORY_LENGTH: 28`). ScalAnt's contribution is
   long-memory compression, so a 4s-only window would remove its mechanism.
   The 4s working window matches the 4s per-step input of the other compared
   methods; the extra 28s is ScalAnt's own memory. Report this row as
   "ScalAnt (28s memory)", which sees more history than the 4s-input methods.
   Upstream's own default split is 31s + 1s; 28+4 is chosen to keep the shared
   4s working window.
3. **Scored target: the annotated action of each clip.** Upstream computes
   mean top-5 recall against `argmax` of the multi-hot future target. When
   actions overlap, that returns the *lowest class id* among them. This is an
   implementation artifact: on EK100 validation, only 82.7% of action-start
   targets are the annotated action. RU-LSTM, InAViT and our method score the
   annotated action, so evaluation is aligned here. **Training targets are
   unchanged from upstream**: dense multi-hot union of covering actions, with
   background 0. The old code's training-time overwrite of the last future
   frame is removed.
4. **Loss: equalized action loss on HD-EPIC.** Upstream applies
   `MultipCrossEntropyEqualizedLoss` only to its `EpicKitchens` dataset class,
   with EK100 frequencies hard-coded. It suppresses negative gradients for rare
   classes (frequency < 1.76e-3), the long-tail EK100 recipe. HD-EPIC is
   similarly long-tail, so v2 enables it via `MODEL.EQUALIZED_LOSS: True`, with
   frequencies from the HD-EPIC **training** CSV. Upstream ships no EGTEA
   config, and its EQL cannot run on EGTEA's 106 classes as written; EGTEA
   keeps its existing setting.
5. **One future query at t0: `FUTURE_STEP_IN_SEC: 2` (= `TAU_A`).** The model
   has `N_QUERIES: 1` future slot. Upstream uses `TAU_A = FUTURE_STEP = 1`, so
   its single query sits at the action start; with `TAU_A = 2` the same design
   needs a step of 2. The old config already did this. A step of 1 would
   create two target slots for one query and fail in the loss.

## Code changes

| File | Change |
|---|---|
| `scalant/datasets/hdepic.py` | Removed the last-future-frame overwrite. Returns `scored_act/verb/noun`, the clip's annotated labels in full class space (background = 0). Computes `action_frequency` from the train CSV. |
| `scalant/utils/ouput_target_structure.py` | Optional `Target.scored_actions/verbs/nouns`. |
| `helper.py` | Passes `scored_*` from the batch into `Target` (train and eval). |
| `scalant/criterion/criterion.py` | `_scored_labels()`: when `scored_*` is present, the mt5r labels for action, direct verb/noun heads and marginalized verb/noun use the annotated ids; otherwise upstream argmax. With `MODEL.EQUALIZED_LOSS` and a dataset `action_frequency`, the action loss is EQL with those frequencies. `EpicKitchens` behavior is unchanged. |
| `scalant/config/default_config.py` | `MODEL.EQUALIZED_LOSS: bool = False` (default off, so upstream behavior is unchanged). |
| `configs/hdepic/clip_32s_2s.yaml` | New v2 config (above). |
| `tests/test_v2_changes.py` | Unit tests for label mapping and loss selection; real-data check of the HD-EPIC items. |

## Remaining differences from the upstream ScalAnt default

| Item | Upstream default | v2 | Note |
|---|---|---|---|
| `TAU_A` | 1s | 2s | Task horizon |
| Long / working memory | 31s + 1s | 28s + 4s | Decision 2 |
| Scored target | argmax of multi-hot (lowest overlapping id) | Annotated action | Decision 3 |
| Class space | Fixed RU-LSTM `actions.csv` (EK100) | Closed over train+val (verb, noun) pairs in the HD-EPIC CSVs | HD-EPIC-specific |
| Equalized-loss frequencies | EK100 training counts | HD-EPIC training counts | Decision 4 |
| Per-frame label source | TeSTra `target_perframe` | Rebuilt from CSVs with the same union semantics | Same semantics |
| Features | RU-LSTM/TeSTra TSN, EK100-finetuned, 4 Hz | HD-EPIC TSN-RGB at 4 Hz (`rulstm_features/rgb_p01`) | Dataset-dependent |
| `fla` kernels | Required | PyTorch fallback if missing | Functionally the same |

Everything else matches upstream: CLAM clustering, 4+4 layers, D 1024, FFN
4096, dropout, `CLS_ALL`, `IGNORE_INDEX 0`, AdamW 5e-4, batch 128, 50 epochs,
cosine with 5 warmup epochs, mixup, bf16, seed 42 and `val/mt5r` checkpoint
selection. Early-action handling is upstream's `convert_to_anticipation`:
clips whose window ends before the first feature are dropped, and windows that
start before the video are padded with the first frame.

## Run

```bash
cd baselines/scalable-anticipation
export SCALANT_CKPT_DIR=<output dir>
python main.py --cfg configs/hdepic/clip_32s_2s.yaml --opts USE_WANDB False
python -m unittest tests.test_v2_changes -v   # checks
```

See `expts/submit_scalant_p01_32s2s.slurm`. The environment must provide
`mamba-ssm`.

## Verification (2026-10-15, NYU Torch HPC)

- **Unit and data tests** (`tests/test_v2_changes.py`): 7/7 pass. Real HD-EPIC
  P01 split: train 5,907 / val 1,477 clips, 0 discarded; vocab
  verb/noun/action = 87/153/1228 including background. Every val item has
  128 past features (32s at 4 Hz), one future slot and 128 past labels
  (`CLS_ALL`). The annotated action is present in the last-slot multi-hot
  target for 1,477/1,477 val clips.
- **Effect of the scoring fix on HD-EPIC.** The last future slot is
  multi-hot (overlapping actions) for 512/1,477 val clips. Upstream `argmax`
  scoring would use a different action than the annotated one for **295/1,477
  val clips (19.97%)**; on train, 1,168/5,907 (19.77%).
- **End-to-end GPU smoke** (H200, 2 epochs, MAMBA, bf16): training and
  validation run, every mt5r metric is reported, and `checkpoint_best.pth` is
  written. Validation after 2 warmup epochs: action mt5r 1.83, as a pipeline
  check only. About 3 to 4 minutes per epoch.
- **Open issue: GPU utilization.** Average GPU utilization in the smoke was
  about 5%, because the run is dominated by data loading and CPU work on a
  small model. 50 epochs take roughly 2.5 to 3h in one job. On clusters that
  enforce a utilization rule for jobs over 2h, split the run into shorter jobs
  or speed up loading (more `NUM_WORKERS` or a feature cache). `main.py` can
  reload weights (`TRAIN.CKPT_PATH`) but not optimizer or scheduler state.
