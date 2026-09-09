# Reproduce B18 accuracy and patch-drop examples

This guide covers the B18 predictor-score pruning study at **16 seconds of encoder
context → a 4096-token predictor budget (4-second equivalent)**. Code source: VJEPA2-EXP
`f057ba21962521a258ccaf4375cfc5abfa64e7ed`. The shared launchers adapt locations and
environments; the existing share MTP/CA implementations and frozen EGTEA gaze gate
remain the baseline. Invoke the new evaluation modules explicitly.

## What to reproduce

The main comparison uses 128 RGB frames at 8 fps, tubelet size 2, 256×256 images,
64×256 encoder tokens, and K=4096 retained tokens. `none` retains all 16,384 tokens.
The budget need not be a contiguous 4s window: most selectors retain patches across the 16s input.
Online scores come from predictor block 0 on the full context; offline masks use
the mean score map from 512 training clips. Kept tokens are ordered chronologically
and predictor positions are rebased. This is video-only streaming MTP, distinct
from the existing concat-CA 10s experiments.

Metrics below are **native Action Top-5 percentages at +2/+4/+6 seconds**, evaluated
only where the label is available and belongs to the training vocabulary.

| Strategy | EGTEA +2s / +4s / +6s | HD-EPIC +2s / +4s / +6s |
|---|---|---|
| `recent` | 39.41 / 36.63 / 34.52 | 24.19 / 21.42 / 18.95 |
| `attention` | 39.06 / 36.62 / 34.42 | 25.19 / 22.62 / 19.76 |
| `pred_offline_high` | 38.86 / 36.35 / 34.09 | 24.74 / 22.10 / 19.41 |
| `none` | 38.29 / 35.87 / 33.89 | 23.80 / 21.45 / 19.05 |
| `pred_attention_high` | 37.95 / 35.64 / 33.80 | 24.61 / 21.49 / 19.31 |
| `pred_offline_low` | 37.57 / 35.23 / 33.36 | 22.80 / 20.42 / 18.16 |
| `pred_attention_low` | 35.16 / 33.05 / 30.98 | 23.58 / 20.64 / 18.26 |

EGTEA: split1-source temporal-half streaming validation, 22,225 ctx16 inputs;
scored counts are **22,225 / 22,139 / 22,053**. HD-EPIC: P01 temporal-half streaming
validation, 4,288 ctx16 inputs; scored counts are **3,092 / 3,081 / 3,061**. HD-EPIC
is not `p01_fixed`; at +2s, 1,196 inputs are excluded for vocabulary coverage.

Observed ordering: keep-high outperforms keep-low for both score sources and
datasets. Recent is highest on EGTEA; encoder attention is highest on HD-EPIC.
These observations do not establish why a position pattern improves accuracy.
The later Q1 paired4000 and Q3 length/random-drop studies use separate populations
and are not rows in this table.

Evidence: source raw record `logs/raw/B18/2026-09-04-predictor-score-pruning-16s.md`
in the original B18 worktree; full jobs **16925482** (EGTEA) and **16961166**
(HD-EPIC). Canonical result basenames are
`egtea-{strategy}-ctx16s-keep4096-mf128.json` and
`hdepic-{strategy}-ctx16s-keep4096-mf128.json`, under the source
`outputs/attn_corner_sink/{multi_strategy,hdepic_multi_strategy}/` directories.
Their dataset metadata was corrected retrospectively; original exact launch
arguments and historical code revision were not recorded. The current source
commit identifies this reproduction implementation, not those historical runs.

## Environment and assets

Export `PROJECT_ROOT` before `sbatch`: Slurm copies scripts into its spool directory,
so the launcher cannot locate the repository from its own filename there.

```bash
export PROJECT_ROOT=/path/to/JEPA_ARVR
export PYTHON=/path/to/your/environment/bin/python
export VJEPA_ROOT=/path/to/compatible/vjepa2
export DATA_ROOT=/path/to/b18_assets_and_outputs
export PYTHONPATH="$PROJECT_ROOT:$VJEPA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EGTEA_ROOT=/path/to/EGTEA
export CHECKPOINT=/path/to/vitl.pt
export ENC_LL=/path/to/egtea/encoder_lora_best.pt
export PRED_LL=/path/to/egtea/predictor_lora_best.pt
export PARENT=/path/to/egtea/stream_mtp/best.pt
mkdir -p "$DATA_ROOT/logs"
```

`PARENT` must contain both the trained `model` and `mtp_classifier`. Checkpoint roles
are not interchangeable. EGTEA identities from the source protocol:

| Role | SHA-256 |
|---|---|
| base ViT-L | `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012` |
| encoder LoRA | `dd60d54737db584bd0897922784c0e2909a7c98e90ed05bbaa55b3ccb4c3a4eb` |
| predictor LoRA | `1e70d73a311dd290df4e6137e4b0adfcc6dc8fbdd63b0d905463e59a1705895c` |
| stream-MTP parent | `7b12fdd545c4330198a3a149e02c40566735d86c427e7631acdaa1ea57c02949` |

Use the environment already providing PyTorch, NumPy, pandas, Decord and
Matplotlib. The upstream tree must provide
`evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py`
and `src/models/utils/modules.py`; a checkout containing only a generic ViT
implementation is insufficient. B18 V-JEPA 2.1 diagnostics additionally require
`app/vjepa_2_1/`; they are optional for this 2.0 accuracy table.

For Singularity, also export `SHARE_IMAGE`, optionally `SHARE_OVERLAY` (without
`:ro`), `SHARE_ENV_SCRIPT` and `SHARE_CONDA_ENV`. `PYTHON` is then the interpreter
path inside the container. For Yifan's validation environment these are the CUDA
image, read-only EgoLifeExp overlay, `/ext3/env.sh`, and `/ext3/envs/EgoLifeExp`.
Select a separate environment for optional HF/DINOv2/test-time-register analyses;
provide local weights and the test-time-register repository explicitly.

All Python commands below run on compute nodes in the chosen environment. Set
scheduler account/partition and log paths through `sbatch` options appropriate to
your allocation. Shared GPU launchers record utilization under
`$DATA_ROOT/outputs/share_monitor/`. Use measured smoke throughput to choose shards
that fit the requested walltime; the full historical EGTEA run took about 2h15m.

## 1. Generate the 16s index and calibration map

The frozen 10s CSVs in share remain intact. The new index is written under DATA_ROOT.

```bash
export SOURCE="$PROJECT_ROOT/data/egtea/vjepa_annotations/v1/split1"
export VIDEO_ROOT="$EGTEA_ROOT/session_videos"
export OUT="$DATA_ROOT/data/egtea/vjepa_annotations/stream_16s_split/split1"
sbatch --account=<account> --output="$DATA_ROOT/logs/index-%j.out" \
  "$PROJECT_ROOT/scripts/egtea/make_egtea_stream_16s_split.slurm"
```

After successful completion, expect 22,741 validation rows: 22,225 at ctx16 and
86 each at 4/6/8/10/12/14 seconds. Source CSV SHA-256 values:

- train: `f4049b74d394b133642b46d587802835d39b846a0f08ef7108d740c010e403ed`
- val: `9a638fec249da49b3d0c9b785c0800834c94e61635c9c56184870c5244c21460`

Evaluation requires the nested session layout `<VIDEO_ROOT>/<session>/<session>.MP4`.
Use the existing `make_egtea_stream_video_tree.py` if only flat sessions exist.

```bash
export ANN_DIR="$OUT"
export VIDEO_ROOT="$EGTEA_ROOT/stream_session_videos"
export OUT_DIR="$DATA_ROOT/outputs/attn_corner_sink/pred_offline_calib"
export NCLIPS=512 BLOCK=0 CONTEXT_SEC=16
sbatch --account=<account> --output="$DATA_ROOT/logs/calib-%j.out" \
  "$PROJECT_ROOT/scripts/egtea/run_b18_calibrate_predblk0.slurm"
```

On completion, the required map is `calib_predblk0_map_64x256.npy` in OUT_DIR.
Keep it fixed across all accuracy shards and example figures. The original EGTEA
calibration job was 16909806. Default seed is 0; only training clips enter calibration.

## 2. Reproduce the accuracy table

```bash
export PRED_CALIB_PATH="$DATA_ROOT/outputs/attn_corner_sink/pred_offline_calib/calib_predblk0_map_64x256.npy"
export OUT_DIR="$DATA_ROOT/outputs/b18_reproduce/acc-smoke"
export STRATEGIES='none recent attention pred_attention_high pred_attention_low pred_offline_high pred_offline_low'
export BATCH_SIZE=4 MODE=smoke
sbatch --account=<account> --time=00:15:00 --output="$DATA_ROOT/logs/acc-smoke-%j.out" \
  "$PROJECT_ROOT/scripts/egtea/run_b18_multi_strategy_eval.slurm"
```

The smoke processes two batches and is never a full result. After it passes, use
`MODE=full` and disjoint half-open ranges in **ctx16-filtered CSV order**. For example,
run `[0,4000)`, `[4000,8000)`, `[8000,12000)`, `[12000,16000)`, `[16000,20000)`,
`[20000,22225)`, reducing shard size if required by measured runtime:

```bash
export OUT_DIR="$DATA_ROOT/outputs/b18_reproduce/acc"
export MODE=full ROW_START=0 ROW_STOP=4000
sbatch --account=<account> --output="$DATA_ROOT/logs/acc-%j.out" \
  "$PROJECT_ROOT/scripts/egtea/run_b18_multi_strategy_eval.slurm"
# Repeat with each remaining range; keep all other settings fixed.
```

Each strategy writes a dataset-prefixed JSON including row range, scored counts,
input hashes and arguments. On a CPU allocation, merge **one strategy at a time**:

```bash
"$PYTHON" "$PROJECT_ROOT/scripts/egtea/b18_share_results.py" merge \
  --reports "$OUT_DIR"/egtea-recent-*-rows*.json \
  --out "$OUT_DIR/egtea-recent-full.json"
```

Choose a fresh output directory for each protocol/parameter combination. The merge
rejects smoke reports, gaps, duplicate ranges, incomplete coverage, changed inputs,
changed evaluator source and mismatched settings. It weights by each horizon's
scored count. Inspect all three `n` values against the expected denominators above.

For HD-EPIC, generate the 16s index with `run_hdepic_stream_16s_split.slurm`, providing
`HDEPIC_STAGE2` and `HDEPIC_VIDEO_ROOT`; use the P01 temporal-half source, corresponding
HD-EPIC LoRAs and video-only MTP parent. Point `TRAIN_CSV`, `VAL_CSV`, `VIDEO_ROOT`
and `PRED_CALIB_PATH` at that dataset and regenerate its training calibration map.
The eligible population is 4,288, not 22,225. Do not reuse the EGTEA calibration or
checkpoint roles. Original HD-EPIC index/calibration jobs: 16959173/16959240.

## 3. Select a video and generate patch-drop patterns

List candidate windows on a CPU allocation; no model or checkpoint is loaded:

```bash
"$PYTHON" "$PROJECT_ROOT/scripts/egtea/b18_share_results.py" samples \
  --val-csv "$ANN_DIR/EGTEA_val_stream_mtp.csv" --context-sec 16 --limit 20
# For several windows within one chosen video:
"$PYTHON" "$PROJECT_ROOT/scripts/egtea/b18_share_results.py" samples \
  --val-csv "$ANN_DIR/EGTEA_val_stream_mtp.csv" --video-id OP01-R01-PastaSalad \
  --all-windows --limit 8
```

The historical illustration used **OP01-R01-PastaSalad**, its first ctx16 window.
Use the emitted `row_index` to freeze a particular example; it is the zero-based
**original CSV data-row index**, unlike the context-filtered accuracy shard index.
An invalid video or incompatible row now fails rather than silently choosing another.

```bash
export VIDEO_ID=OP01-R01-PastaSalad
export ROW_INDEX=6  # first ctx16 row in the frozen EGTEA CSV; choose another from samples
export CALIB_PATH="$PRED_CALIB_PATH"
export OUT_DIR="$DATA_ROOT/outputs/b18_reproduce/patterns/$VIDEO_ID-row$ROW_INDEX"
export STRATEGIES='attention recent pred_attention_high pred_attention_low pred_offline_high pred_offline_low'
export NSLOTS=8 CONTEXT_SEC=16 KEEP_COUNT=4096 FRAMES=128 FRAME_MODE=csv
sbatch --account=<account> --output="$DATA_ROOT/logs/drop-%j.out" \
  "$PROJECT_ROOT/scripts/egtea/run_b18_prune_drop_overlay.slurm"
```

For each strategy, outputs are:

- `drop_<strategy>_<video>_row<index>_64slots_csv.png`: eight uniformly spaced slots;
  visible pixels are kept and **black patches are dropped**.
- `slot_keep_*.npy`: retained patch counts for all 64 slots.
- `kept_indices_*.npy`: exact retained token indices.
- `sample_*.json`: source row, CSV hash, decoded frame indices, shown slots and arguments.

The default `FRAME_MODE=csv` uses exactly the CSV frame indices used by accuracy
evaluation, including occasional nonuniform frame gaps. The historical figure
script reconstructed a constant-stride window from the final frame; use
`FRAME_MODE=legacy_stride` explicitly to reproduce that older sampling. The mode
is included in filenames and metadata, so the two outputs remain distinct.

Set `NSLOTS=64` to show every slot. The displayed image is one frame from each
2-frame tubelet; the mask applies to the tubelet's spatial patches. Compare all
strategies on exactly the same row, context, weights and calibration. Pick examples
to illustrate the pattern, and use the full accuracy population for quantitative
claims. The figure alone does not establish an accuracy benefit.

## Other shared experiments

B13/B17 algorithms, B18 Q1/Q3 evaluators and their analysis scripts retain their
original module names. B17 manifests are generated by
`build_b17_tpjepa_oracle_manifest_cpu.slurm` under `B17_MANIFEST_DIR` and checked
against the frozen training CSV; runtime inputs are not stored under `logs/raw/`.
Q1/Q3 archive-based launchers still require their frozen preparation/reference
bundles under the configured DATA_ROOT layout. Historical archive manifests can embed absolute source paths and hashes of their
original code dependencies. They need a separately verified relocation or fresh
preparation under the shared checkout before reuse on another machine; copying
the directory alone is insufficient. These archive workflows are separate from
the two standalone reproduction paths above. Changing paths must not change
sample identities, input hashes or numeric parity thresholds. Sharing the code
does not promote provisional/failed experimental routes into approved methods;
retain the route states in `configs/evaluation_protocols/egtea_stream_mtp_pruning.yaml`.
