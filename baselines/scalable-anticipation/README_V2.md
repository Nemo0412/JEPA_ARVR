# ScalAnt baseline v2: one code path for EK100, EGTEA and HD-EPIC

This revision builds on the `Baseline_ScalAnt` HD-EPIC P01 code (`8c3b146`).
It moves the baseline to ScalAnt's default history and encoder, scores every
method on the same target, and runs the **same action-instance pipeline on
all three datasets** through one dataset class (`ClipAnticipation`). The old
4s/LSTM HD-EPIC config (`configs/hdepic/clip_4s_2s.yaml`) still runs.

| Dataset | Config | Annotations | Class space | Features (4 Hz npy) |
|---|---|---|---|---|
| EK100 | `configs/ek100/clip_32s_2s.yaml` | RU-LSTM `annotations/ek100_rulstm/{training,validation}.csv` | official `actions.csv`, 3,806 | RU-LSTM EK100 TSN LMDB, converted |
| EGTEA split k | `configs/egtea/clip_32s_2s_s{1,2,3}.yaml` | RU-LSTM `annotations/egtea_rulstm/{training,validation}{k}.csv` | official `actions.csv`, 106 | RU-LSTM split-k TSN LMDB, converted |
| HD-EPIC P01 | `configs/hdepic/clip_32s_2s.yaml` | clip-split V-JEPA CSVs | closed train+val vocabulary (no official list) | `rulstm_features/rgb_p01` |

## Agreed decisions (2026-10-15)

1. **Encoder: upstream default.** MAMBA v1 with `D_STATE: 16`. The old
   HD-EPIC config used LSTM and did not set `D_STATE`, so it would have
   defaulted to 64. MAMBA requires `mamba-ssm` and `causal-conv1d`; `fla` is
   optional because a PyTorch fallback is kept.
2. **History: 32s = 28s long memory + 4s working memory.** ScalAnt's
   contribution is long-memory compression, so a 4s-only window would remove
   its mechanism. The 4s working window matches the 4s per-step input of the
   other compared methods. Report this row as "ScalAnt (28s memory)": it sees
   more history than the 4s-input methods. Upstream itself splits 31s + 1s;
   28+4 keeps the shared 4s working window.
3. **Scored target: each clip's annotated action.** Upstream computes mean
   top-5 recall against the `argmax` of the multi-hot future target, which
   picks the *lowest class id* among overlapping actions. That is an
   implementation artifact. It differs from the annotated action for 17.3% of
   EK100 validation action starts and for 20% of HD-EPIC validation clips.
   RU-LSTM, InAViT and our method score the annotated action. **Training
   targets are unchanged** (upstream multi-hot union, background 0); the old
   code's training-time overwrite of the last future frame was removed.
4. **Equalized action loss on all three datasets**, each with its own
   training frequencies. Upstream enables `MultipCrossEntropyEqualizedLoss`
   only for its EK100 dataset class, with EK100 frequencies hard-coded, so it
   cannot run on EGTEA. v2 computes the frequencies per dataset. EK100 values
   equal upstream's, which come from the same RU-LSTM `training.csv`.
5. **Early actions.** Training drops clips whose window ends before the
   first feature, as upstream does. Evaluation keeps them, with zero features
   and `observable = 0`, and scores them as **misses** on the full official
   population (EK100 val 9,668 rows). Precedent: AVT/S-GEAR add dropped
   segments back with zero scores in challenge submissions; RU-LSTM counts
   them as zeros in EK100 validation; and the challenge needs every segment.
   In all of these, such segments earn no credit.
6. **One future query at t0** (`FUTURE_STEP_IN_SEC = TAU_A = 2`). The model
   has `N_QUERIES: 1`. Upstream's `TAU_A = FUTURE_STEP = 1` puts that query at
   the action start; with `TAU_A = 2` the step must be 2 (this is not a
   difference from upstream).

## Comparison with the upstream ScalAnt default

| # | Item | Upstream default | v2 | Reason |
|---|---|---|---|---|
| 1 | `TAU_A` | 1s | **2s** | Task horizon |
| 2 | Long / working memory | 31s + 1s | **28s + 4s** (same 32s total) | Shared 4s working window |
| 3 | Scored target | argmax of multi-hot (lowest overlapping id) | **Annotated action** | Same target as other methods |
| 4 | Equalized loss | EK100 only, hard-coded EK100 frequencies | **All three datasets, own training frequencies** | Aligned code |
| 5 | Early actions at evaluation | Dropped from scoring | **Kept, scored as misses** | Same population for every method |
| 6 | Class space | Fixed EK100 `actions.csv` | EK100 3,806 and EGTEA 106 (official); HD-EPIC closed vocabulary | No official HD-EPIC list |
| 7 | Per-frame label source | TeSTra files | Rebuilt from each dataset's CSVs, same union semantics | One pipeline for three datasets |
| 8 | Features | TeSTra/RU-LSTM TSN (EK100-finetuned), 4 Hz | EK100: our RU-LSTM LMDB to 4 Hz; EGTEA: split-specific TSN; HD-EPIC: `rgb_p01` | Dataset-dependent; EK100 features equal the RU-LSTM baseline's |
| 9 | `fla` kernels | Required | PyTorch fallback when missing | Environment; functionally the same |

**Unchanged from upstream:** MAMBA v1 `D_STATE 16`; CLAM clustering; 4+4
layers, D 1024, FFN 4096, dropout 0.1/0.2, shared classifier, one query at
t0; multi-hot targets with `CLS_ALL` and `IGNORE_INDEX 0` (background-only
frames excluded from the loss); AdamW 5e-4, wd 0, batch 128, 50 epochs,
cosine with 5 warmup epochs, mixup, bf16, seed 42; `val/mt5r` checkpoint
selection; training-time early-action drop with first-frame padding.

**Changed from the collaborator's previous version:** 4s to 32s history;
LSTM to MAMBA with explicit `D_STATE 16`; the last-future-frame overwrite
moved from training targets to scoring only; HD-EPIC-only code extended to
three datasets.

## Code changes

| File | Change |
|---|---|
| `scalant/datasets/clip_anticipation.py` | New `ClipAnticipation`: RU-LSTM or HD-EPIC CSVs, official or closed class space, dense multi-hot labels from CSVs, `action_frequency`, `scored_act/verb/noun`, `observable` (early clips kept at evaluation). |
| `scalant/datasets/hdepic.py` | `HDEPICClip` is now a thin `ClipAnticipation` alias (HD-EPIC format, closed vocabulary). |
| `scalant/criterion/criterion.py` | `_scored_labels` and `_mt5r_inputs`: mt5r for action, direct verb/noun and marginalized verb/noun uses annotated labels, and unobservable clips are forced to misses. Optional equalized loss with dataset frequencies (`MODEL.EQUALIZED_LOSS`). `EpicKitchens` behavior unchanged. |
| `scalant/utils/ouput_target_structure.py`, `helper.py` | `Target.scored_*` and `Target.observable`, passed from the batch. |
| `scalant/config/default_config.py` | `DATA.ANNOTATION_FORMAT`, `DATA.ACTIONS_CSV`, `DATA.SCORE_ALL_EVAL`, `MODEL.EQUALIZED_LOSS` (defaults keep upstream and old-config behavior). |
| `tools/rulstm_lmdb_to_npy.py` | RU-LSTM 30 fps-keyed LMDB to per-video 4 Hz npy (key `max(1, floor(t*7.5))`, nearest earlier key if missing) plus coverage JSON. |
| `configs/{ek100,egtea,hdepic}/clip_32s_2s*.yaml`, `annotations/egtea_rulstm/` | v2 configs and RU-LSTM EGTEA CSVs. |
| `tests/test_v2_changes.py` | Unit tests (label mapping, miss forcing, loss selection) and real-data checks for all datasets. |

## Run

```bash
cd baselines/scalable-anticipation
# 1) features for EK100 / EGTEA (once), e.g. EK100:
python tools/rulstm_lmdb_to_npy.py --lmdb <rulstm ek100 rgb lmdb dir> \
  --out <DATA_ROOT>/ek100/scalant_features/rgb_4hz \
  --csv annotations/ek100_rulstm/training.csv --csv annotations/ek100_rulstm/validation.csv
# 2) train + validate
export SCALANT_CKPT_DIR=<output dir>
python main.py --cfg configs/ek100/clip_32s_2s.yaml --opts USE_WANDB False DATA.DATA_ROOT_PATH <DATA_ROOT>
python -m unittest tests.test_v2_changes -v
```

`expts/submit_scalant_p01_32s2s.slurm` gives an HD-EPIC example. The Python
environment must provide `mamba-ssm`.

## Verification

### HD-EPIC (2026-10-15)

- Unit and data tests pass. Train 5,907 / val 1,477 clips, 0 dropped; vocab
  verb/noun/action = 87/153/1228 including background. Each item has 128 past
  features, one future slot and 128 past labels. The annotated action is in
  the last-slot multi-hot target for 1,477/1,477 val clips.
- Effect of the scoring fix: the last slot is multi-hot for 512/1,477 val
  clips. Upstream argmax would score a different action for **295/1,477 val
  clips (19.97%)**; on train, 1,168/5,907 (19.77%).
- GPU smoke (H200, 2 epochs, MAMBA, bf16): training and validation run, all
  mt5r metrics are reported and the checkpoint is written. Epoch 1 includes
  about 5 minutes of one-time MAMBA/triton compilation and autotuning. After
  that an epoch takes about 10 s (train 7 s, val 1 s), so 50 epochs take about
  15 minutes.

### EK100 / EGTEA

See below (filled from the conversion audit, tests and smokes).
