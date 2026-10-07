# ScalAnt / CLAM on EGTEA: official-label 4s streaming protocol

This independent baseline branch contains the EGTEA adapter used in Yifan's
completed ScalAnt experiments. This document defines the transferred protocol
and its executable routes. It was written for this branch; experiment logs and
existing workspace documentation have not been imported. The code-sharing
branch is `baseline-scalant-egtea`, based on JEPA_ARVR `a8dd058`.

## Source and scope

| Component | Pinned source |
|---|---|
| ScalAnt model, CLAM, temporal encoder, decoder and EK100 config | `refer_repo/scalable-anticipation` at `f7d8710435af4c6abc9dee28f1fb890099d7ec43` |
| Official EGTEA action memberships and action/verb/noun map | `refer_repo/rulstm` at `e648c7ee40a1a71381bfaa7563f2e8a6935275ab`, `RULSTM/data/egtea/` |
| External EGTEA adaptation | `app/scalant_baseline/`, transferred from the B19 ScalAnt worktree |

Upstream submodules are unmodified. The model is the released `QueryPredictor`
with Mamba, CLAM and its query decoder. The visual encoder is external: the
adapter reads released 1,024-D RGB TSN features and does not train a visual
encoder. This branch changes filesystem/runtime plumbing for sharing; model,
loss, label lookup, scoring and checkpoint-selection behavior are preserved.

The paper/code's native fixed-history action-instance protocol is a separate
experiment. Its 4s working-memory + 28s-history feasibility helpers remain in
the transferred code, but no formal results for that variant are claimed here.
Older temporal-half diagnostic helpers are also retained for provenance; they
are not the formal training/evaluation routes below.

## Protocol identities and official labels

For each split `s` in `{1,2,3}`, the training/selection protocol is
`egtea-action-anticipation/official-split{s}-stream4s-ticks-h2h6-recall5-v1`.
Selected-checkpoint full-head evaluation uses
`egtea-action-anticipation/official-split{s}-stream4s-selected-full-heads-v1`.
The associated route configuration is
[`configs/evaluation_protocols/scalant_egtea.yaml`](../configs/evaluation_protocols/scalant_egtea.yaml).
These IDs retain the completed experiments' meaning; sharing does not introduce
a new split or metric version.

| Official split | Training action rows | Validation action rows |
|---|---:|---:|
| 1 | 8,299 | 2,022 |
| 2 | 8,299 | 2,022 |
| 3 | 8,300 | 2,021 |

Use `training{s}.csv` and `validation{s}.csv`, with the released UID, video,
30-FPS start/end frame and label fields. Their union is 10,321 distinct actions
in 86 videos. Training and validation may share a video, but label membership
is strictly by official action UID. Heads have 106 Action, 19 Verb and 53 Noun
classes. The V-JEPA annotation vocabulary and temporal-midpoint memberships do
not define any formal labels in this adapter.

## Visual features and clock alignment

The completed adapter uses the **same s1 RGB TSN LMDB for all three official
label splits**, `TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd`. Selecting
`--split 2` or `--split 3` changes labels, not this feature producer. The code's
default and this document preserve that choice. Using another split's encoder
features would be a separately identified experiment, even if vector dimensions
and key coverage match.

The released TSN producer has cross-split action-label training overlap. This
was accepted as a feature-source adaptation, but is distinct from the strict
official membership masks applied to ScalAnt's losses and validation. It is
not evidence that the visual encoder was trained without validation labels.
Split-specific TSN weights are not supplied in this branch; the required asset
is the already extracted RGB LMDB.

Video metadata supplies each native FPS `f` and frame count `N`. Official labels
and LMDB keys use a separate 30-FPS clock. Advance native ticks by
`d = max(1, round(4*f))`, starting at `d`, while the tick frame is less than `N`.
Convert a tick frame `t` to `round(30*t/f)` before looking up a target. Reject a
horizon whose native frame position `t + round(h*f)` reaches the physical end.

Every process call reads 16 float32 feature vectors at 4 Hz in the current 4s
window. Key frames are `max(1, floor(30*(t/f - 4 + j/4)))`, `j=0..15`, formatted
as `<video>_frame_<frame:010d>.jpg`. Missing, wrong-sized or nonfinite vectors
before the final scorable tick are errors; they are not filled with zeroes.

## Video history, targets and sparse supervision

Each video starts with empty history for both training and validation. The
current 16 raw feature vectors are appended to all earlier chunks in that
video. The newest 16 tokens form working memory; all earlier tokens form the
growing history. Memory resets between videos and between traversals. The
implementation re-encodes the full causal prefix at every tick; it does not
cache Mamba hidden states or promise equivalence to an incremental encoder.

Predict with independently trained +2s and +6s models, each with one query.
At the 30-FPS target frame, select the first covering action in sorted official
interval order; if none covers it, select the next action by start frame. That
target contributes only when its UID belongs to the active population. A
target belonging to the other population or no remaining target is masked.
Repeated selections of the same UID remain separate ticks. An action that no
eligible tick selects is absent from the metric denominator.

Training sums up to six cross-entropy terms: future Action/Verb/Noun at the
chosen horizon, and past Action/Verb/Noun over known positions in the current
16-token window. A past position is supervised only when exactly one official
action covers it and that action belongs to training. Unknown or ambiguous
positions are masked; validation labels never supervise these terms. A tick
with an active past loss and no future label still contributes gradients.
Unlabelled prefix/gap ticks still update history and run the model.

**Terminal-tail exception:** omit only the suffix after the last tick having a
valid +2s or +6s target for the active population. Some released TSN features
are missing there. Such a suffix cannot affect a later scored prediction, but
omitting it shortens final memory and runtime. Trimming is based on either
horizon, even when training just one. A video with no scored tick for that
population produces no traversal. The code comments retain this exception.

The external `legacy_autograd_fix.py` clones query and memory storage between
decoder/CLAM layers. The upstream layers add positions in place, which caused
backward version-counter failures. Cloning preserves their cumulative forward
values while avoiding mutation of tensors saved by earlier layers; replacing
the additions with a different positional operation is not this adaptation.

## Training and continuation

| Setting | Transferred formal value |
|---|---|
| Horizons | +2s and +6s trained separately; one output query each |
| Optimizer | AdamW, maximum LR `5e-4`, weight decay `0.0` |
| Schedule | Released `WarmUpCosineAnnealingLR`, 50 epochs, 5-epoch warmup, `eta_min=1e-7`; stepped after training each epoch |
| Precision | CUDA bfloat16 autocast |
| Mixup | Disabled because video prefixes have unequal lengths |
| Effective update | Average gradients over 128 ticks with any active loss; final partial group divides by its actual size |
| Seed | 42; deterministic epoch-dependent video shuffling; ticks remain chronological within each video |
| Selection | First completed epoch attaining the highest full-validation Action mean Recall@5, independently per horizon |

Loss accumulation replaces the original fixed batch of 128; it does not
reproduce batch mixup. An update may cross a video boundary, although feature
history never does. No gradient graph is retained through earlier tick calls.

Continuation is at completed epoch boundaries. `latest.pt` stores model,
optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, epoch cursor, best score/epoch,
protocol and source hashes. Each successor checks `--expected-start-epoch` and
fails on incompatible source, split or horizon. Resume does not promise
bitwise numerical equality across GPU types. A manually stopped run selects
among completed epochs and must report its actual training length.

The transferred selected-checkpoint replay helpers have historical completion
guards: split 1/3 require epoch 49; split 2/3 replay also requires at least
epoch 40. They serve the completed 50/41/50-epoch runs. Replaying a shorter
future run requires explicitly adapting those completion guards while retaining
the selection/protocol checks; it must not be presented as 50 completed epochs.

## Validation, metrics and acceptance counts

Validation starts from video time zero and uses only official validation future
labels, including the unlabelled history needed before those predictions.
There is no temporal-half validation warm start. Counts below are for the
audited native metadata and terminal-tail rule, not the number of action rows.

| Split | Processed train ticks | Processed val ticks | Scored train +2s / +6s | Scored val +2s / +6s |
|---|---:|---:|---:|---:|
| 1 | 23,943 | 16,227 | 19,403 / 19,340 | 4,678 / 4,655 |
| 2 | 24,027 | 13,386 | 19,417 / 19,340 | 4,664 / 4,655 |
| 3 | 23,822 | 13,159 | 19,360 / 19,302 | 4,721 / 4,693 |

For each head, Top-5 accuracy is the fraction of scored ticks whose target is
in `torch.topk(5)`. Mean Recall@5 first computes that hit fraction per class,
then averages equally over classes present in scored validation ticks. Classes
with no scored tick are excluded. Report tick and supported-class denominators.
Select by Action mean Recall@5; Verb/Noun do not select a checkpoint.

At the selected checkpoint report direct Action, Verb and Noun heads. Separate
supplementary Verb/Noun variants sum **raw Action logits** over the action map;
these are neither the direct trained heads nor softmax-probability marginals.
Keep ordered tick/UID/label identities, checkpoint/source hashes and GPU Top-5
hit/tie evidence. CPU and GPU may choose different classes at tied fifth-place
scores; retain the original GPU per-vector Top-5 result and disclose replay
differences rather than silently reselection. The training-time Action score
remains the selection score.

Three-split summaries take an equal-weight mean of splitwise mean Recall@5 and
report sample SD. They do not pool all ticks into one accuracy. Validation is
used for both selection and reporting; no untouched test result is implied.
Periodic tick metrics repeat some action instances and omit others, so native
RU-LSTM/InAViT action-instance metrics and old midpoint-stream metrics are
different evaluation populations. The exact latest collaborator-method scorer
was unavailable when these runs were conducted.

## Assets, environment and executable routes

Initialise only the required baseline submodules:

```bash
git submodule update --init refer_repo/scalable-anticipation refer_repo/rulstm
export CODE_ROOT=/path/to/JEPA_ARVR-scalant
export PROJECT_ROOT=/path/to/scalant-assets-and-outputs
export SCALANT_WORKSPACE_ROOT="$PROJECT_ROOT"
export SCALANT_TSN_FEATURES=/path/to/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd
export SCALANT_VIDEO_METADATA=/path/to/native_video_metadata.json
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

By default official CSVs come from the pinned RU-LSTM submodule.
`SCALANT_ANNOTATIONS` may point at an exact copy; changed labels fail identity
checks. `SCALANT_WORKSPACE_ROOT` defaults to the code root for direct Python
calls. The Slurm launchers use `PROJECT_ROOT` as its default. Paths are read
when modules are imported, so configure them before invoking Python.

Metadata is a JSON object containing `video_metadata`, mapping all 86 video IDs
to native `fps` and `frames`. An existing audited `split_stats.json` is usable
because only that metadata field is read by formal routes. Alternatively,
`prepare_video_metadata` probes full session videos with ffprobe and does not
construct any train/validation split:

```bash
cd "$CODE_ROOT"
python -m app.scalant_baseline.prepare_video_metadata \
  --video-root /path/to/EGTEA/session_videos --output "$SCALANT_VIDEO_METADATA"
```

Run metadata creation and Python on a compute node in the appropriate project
environment. Regenerated metadata must pass the counts/source audit before
training. Required packages include PyTorch with CUDA/bfloat16 support, NumPy,
LMDB, `nestconfig==0.2.0`, `fvcore==0.1.5.post20221221`, `yacs==0.1.8`,
`fla-core==0.4.1`, Triton and a CUDA-compatible `mamba-ssm==2.3.0` build.
The original successful build used Torch 2.11.0+cu130 and CUDA 13.0. Match
Mamba's compiled extension to the target PyTorch/CUDA environment; use the
dependency/probe helpers before a new formal run.

The copied launchers support `SCALANT_RUNTIME=singularity` (default), with
configurable `IMAGE`, `OVERLAY`, `SCALANT_BOOTSTRAP`, `SCALANT_CONDA_ENV` and
`SCALANT_BINDS`. They also support `SCALANT_RUNTIME=native` in an already
prepared environment, optionally sourced through `SCALANT_ENV_SETUP`. The
Singularity overlay is read-only; targeted dependency directories can be
configured with `DEPS`/`RULSTM_DEPS`. Account/partition are supplied at submission,
not fixed in transferred scripts. Export `CODE_ROOT` and `PROJECT_ROOT` before
`sbatch`; Slurm executes a spool copy, not the original script pathname.

Audit before training, then choose a fresh output directory per split/horizon:

```bash
# Compute-node commands in the configured environment:
python -m app.scalant_baseline.official_stream_audit
python -m app.scalant_baseline.official_three_split_audit --split 2
python -m app.scalant_baseline.official_three_split_audit --split 3

# Example scheduler submissions; select the account valid on your cluster:
sbatch --account YOUR_ACCOUNT --gres=gpu:1 --time=01:00:00 \
  scripts/egtea/scalant_official_train_gpu.slurm 2 0 1 1
# Arguments: horizon, expected first epoch, epochs in this job, official split.
# A successor waits for the first job to succeed and requires cursor 1:
sbatch --account YOUR_ACCOUNT --gres=gpu:1 --time=01:15:00 \
  --dependency=afterok:PREVIOUS_JOB \
  scripts/egtea/scalant_official_train_gpu.slurm 2 1 2 1
```

GPU launchers retain utilization sampling. Requested walltime must be based on
measured throughput on the new machine; the examples are not a new runtime
measurement. Segments remain below two hours on the original low-utilization
path, with exact epoch-boundary continuation. Automated continuation helpers
are retained, but manual early stopping needs job-by-job human monitoring.

| Purpose | Python route |
|---|---|
| Training + Action selection, any split | `official_stream_train` |
| Split-1 source coverage | `official_stream_audit` |
| Split-2/3 source coverage | `official_three_split_audit` |
| Split-1 selected full-head dump | `official_stream_full_eval_gpu` |
| Split-1 CPU official sequence audit | `official_stream_full_eval_cpu` |
| Split-1 GPU tied-score replay | `official_stream_full_eval_topk_gpu` |
| Split-1 final CPU evidence verification | `official_stream_full_eval_verify_cpu` |
| Split-2/3 selected full-head GPU replay | `official_s23_selected_eval` |
| Shared-code/asset parity check | `scripts/egtea/verify_scalant_share.py` |

On a configured compute node, verify the transferred index with
`python scripts/egtea/verify_scalant_share.py --report /path/to/validation.json`.
When the original experiment checkout is available, add
`--source-root /path/to/b19-scalant` to compare every tick field, source hash
and representative feature vector, and to check unchanged model/loss/scoring
modules byte for byte. Configure the same audited asset paths for both copies.

Python module names above are prefixed with `app.scalant_baseline.`. Source
audits bind official CSVs, tick population and feature coverage. Training
checkpoints bind CSV/action-map and upstream-config hashes; they do not hash
the complete LMDB. Archive feature provenance/hash separately when moving data.
Do not substitute a dimension-compatible feature bundle and infer equivalence
from the checkpoint's CSV hashes alone.
