# EGTEA streaming handoff — read this first

Branch: `yf_egtea_share`

This branch is the code-sharing channel from Yifan's EGTEA workspace to ll.  It
contains the EGTEA temporal-half split, the dataset adapter, reference launch
scripts, and ll's unmodified RU-LSTM implementation.  Raw videos, raw BeGaze
exports, checkpoints, and feature tensors are not committed.

## Code-sharing rules

1. **ll owns the model implementation.** The branch starts from
   `opencl@29063fea9ac1fa56d3cfbe4014e885d7cefdf4e7`.  Do not replace its CA,
   MTP, predictor, pruning, or FFN logic with files from another repository.
2. The only modifications to ll's existing gaze code are the EGTEA reader,
   frozen coverage gate, and their CLI plumbing.  Review them with:

   ```bash
   git diff origin/opencl...yf_egtea_share -- \
     app/hdepic_lora_action_anticipation/{binary_input_adapter.py,gaze.py,train_stream_mtp_concat_ca.py,egtea_gaze.py}
   ```

3. `baselines/rulstm_hdepic/` is copied byte-for-byte from ll's
   `baseline@6ce4290d2c907ccef96e35e227ae797a266934c1`.  Dataset wrappers live in
   `scripts/egtea/`; do not edit the copied trainer to port EGTEA.
4. The six-run Vanilla V-JEPA matrix remains defined by ll's current code.  This
   branch supplies its EGTEA split and path/reader wiring; it does not guess or
   redefine the six method configurations.
5. Never commit large data or experiment artifacts to this branch.  Transfer
   them separately and verify semantic role plus SHA-256 before use.
6. Do not reuse an output directory across different horizons, feature types,
   warm starts, gaze settings, or FFN settings.

## File map

| EGTEA-owned logic | Location |
|---|---|
| This entry point | `docs/EGTEA_AGENT_HANDOFF.md` |
| Fixed-clip source CSV builder | `scripts/egtea/build_egtea_csvs_v2.py` |
| Temporal-half stream builder | `scripts/egtea/make_egtea_stream_half_split.py` |
| Stream video-tree builder | `scripts/egtea/make_egtea_stream_video_tree.py` |
| Nested stream video/session gate | `scripts/egtea/verify_stream_video_tree.sh` |
| Frozen split identity gate | `scripts/egtea/verify_stream_split.sh` |
| BeGaze coverage auditor | `scripts/egtea/audit_egtea_stream_gaze_coverage.py` |
| EGTEA gaze reader | `app/hdepic_lora_action_anticipation/egtea_gaze.py` |
| Gaze/MTP reference launcher | `scripts/egtea/submit_stream_gaze_ca_mtp.slurm` |
| Video-only single-horizon reference | `scripts/egtea/submit_stream_video_single_horizon.slurm` |
| Frozen-triplet -> selected-horizon label adapter | `scripts/egtea/prepare_vanilla_single_horizon_csv.py` |
| RU-LSTM V-JEPA feature route | `scripts/egtea/submit_rulstm_vjepa_features.slurm` |
| RU-LSTM TSN small/Large v2 | `scripts/egtea/submit_rulstm_{small_tsn,large_v2}.slurm` |
| RU-LSTM feature-pair/session gate | `scripts/egtea/verify_rulstm_feature_bundle.sh` |
| Frozen model-facing split | `data/egtea/vjepa_annotations/stream_half_split/split1/` |

## Frozen EGTEA temporal-half split

Model-facing files:

| File | Data rows | MD5 |
|---|---:|---|
| `EGTEA_train_stream_mtp.csv` | 25,256 | `dcdc98c9993ae028dbb4a2877deff09a` |
| `EGTEA_val_stream_mtp.csv` | 22,741 | `cb04d6eddc2e65910899fde8b3c08659` |
| `EGTEA_test_stream_mtp.csv` | 22,741 | `cb04d6eddc2e65910899fde8b3c08659` |

Validation and test are intentionally byte-identical.  Run:

```bash
scripts/egtea/verify_stream_split.sh
```

before every new port.  The source CSVs and official split-1 label files needed
to regenerate the split are committed under `data/egtea/`.

### End-to-end regeneration on ll's machine

Use a clean output root; do not point the fixed-clip builder at the tracked
branch data directory.  The Python environment must provide NumPy, pandas,
PyTorch and Decord, and the repository root must be importable:

```bash
export PROJECT_ROOT=/path/to/JEPA_ARVR
export PYTHON=/scratch/ll5914/conda_envs/SVD/bin/python
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EGTEA_ROOT=/scratch/ll5914/datasets/EGTEA
export REGEN_ROOT=/tmp/egtea_regen

mkdir -p "$REGEN_ROOT/action_annotation"
cp "$PROJECT_ROOT"/data/egtea/action_annotation/{action_idx.txt,verb_idx.txt,noun_idx.txt,train_split1.txt,test_split1.txt} \
  "$REGEN_ROOT/action_annotation/"

"$PYTHON" "$PROJECT_ROOT/scripts/egtea/build_egtea_csvs_v2.py" \
  --data-root "$REGEN_ROOT" --split 1 --val-frac 0.1 \
  --no-symlinks --overwrite

"$PYTHON" "$PROJECT_ROOT/scripts/egtea/make_egtea_stream_half_split.py" \
  --source "$REGEN_ROOT/vjepa_annotations/v1/split1" \
  --video-root "$EGTEA_ROOT/session_videos" \
  --out "$REGEN_ROOT/vjepa_annotations/stream_half_split/split1" \
  --tick-sec 2 --min-context-sec 4 --max-context-sec 10 \
  --context-schedule 4,6,8,10 --horizons-sec 2,4,6 --model-fps 8

for split in train val test; do
  cmp \
    "$REGEN_ROOT/vjepa_annotations/stream_half_split/split1/EGTEA_${split}_stream_mtp.csv" \
    "$PROJECT_ROOT/data/egtea/vjepa_annotations/stream_half_split/split1/EGTEA_${split}_stream_mtp.csv"
done
```

### Construction logic

1. `build_egtea_csvs_v2.py` reads the official split-1 annotations and creates
   session-frame V1 CSVs.  It uses stratified 10% validation sampling per action
   with NumPy `default_rng(seed=42)`.
2. The temporal-half builder pools source train/val/test, then deduplicates by
   `(participant_id,start_frame,stop_frame,verb_class,noun_class)`.  Counts are
   7,468 / 831 / 2,022 source rows and 10,321 pooled actions.
3. Each of 86 full-session videos is split at `n_frames // 2`.  The first half
   produces stream training observations; the second half produces validation,
   which is copied to test.  Original official train/test membership is not the
   streaming membership.
4. Observation ticks advance every 2 seconds.  Context grows
   `4 -> 6 -> 8 -> 10` seconds, then remains a sliding 10-second window.
5. Frames are sampled at 8 fps from 24-fps full-session videos.  Labels are
   stored for `+2/+4/+6` seconds.  Lookup uses the action covering the target,
   otherwise the next action start; unavailable labels get `mtp_mask=0`.
6. Training future labels may cross the temporal midpoint.  Counts are
   86/172/258 for `+2/+4/+6`.

Context-bucket counts are 86/86/86/24,998 for train and
86/86/86/22,483 for val/test at 4/6/8/10 seconds.  The expected training-derived
vocabulary is 19 verbs, 51 nouns, and 106 action pairs.

The split was independently regenerated in Yifan's workspace by Slurm job
`15876527` and matched all three frozen CSVs byte-for-byte.

## Required data layouts

Use flat full sessions for split creation and V-JEPA feature extraction:

```text
<EGTEA_ROOT>/session_videos/<session>.mp4
```

ll's streaming datasets require:

```text
<EGTEA_ROOT>/stream_session_videos/<session>/<session>.MP4
```

Build the second view with `make_egtea_stream_video_tree.py`; symlinks are
allowed.  There must be exactly 86 sessions.  Do not use fixed action clips as
streaming videos.

The gaze reader expects one BeGaze export per session:

```text
<GAZE_ROOT>/<session>.txt
```

Set `GAZE_ROOT` explicitly on ll's machine; do not assume Yifan's scratch path
exists there.

## Gaze coverage contract

The frozen manifest is:

```text
data/egtea/vjepa_annotations/stream_half_split/split1/frozen_gaze_coverage.json
```

Training has zero gaze-timeline OOB rows.  Validation has exactly 171:

- `OP01-R06-GreekSalad`: 54
- `OP03-R01-PastaSalad`: 92
- `OP03-R07-Pizza`: 25

Only the exact `(session,start_frame,last_frame)` rows in the manifest may pass
the EGTEA **timeline-OOB gate** and use the user-approved center fallback
`(0.5,0.5)` for their unavailable tail.  Any new OOB row is fatal.  Do not
delete these validation rows.

This gate is distinct from ll's existing per-sample coordinate sanitizer:
inside the available BeGaze timeline, non-finite or `(0,0)` coordinates are
also mapped to center by `_clean_xy`.  That behavior is inherited from ll's
gaze implementation, not introduced by the EGTEA split adapter.  A missing or
unparseable gaze file must be caught by the full coverage audit below before a
formal launch; do not rely on the model loader's empty-record fallback.

Rebuild the manifest, if needed, with:

```bash
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" scripts/egtea/audit_egtea_stream_gaze_coverage.py \
  --annotation-dir data/egtea/vjepa_annotations/stream_half_split/split1 \
  --gaze-root <GAZE_ROOT> \
  --output /tmp/frozen_gaze_coverage.json
cmp /tmp/frozen_gaze_coverage.json \
  data/egtea/vjepa_annotations/stream_half_split/split1/frozen_gaze_coverage.json
```

## Porting ll's +gaze / Vanilla V-JEPA runs

The integrated gaze route is still ll's `opencl@29063fe`:

```text
train_stream_mtp_concat_ca.py
  -> ConcatPlusCrossAttnAdaptedModel
  -> CommunicatingMLPMTPClassifier
```

For the current communicating-MTP +gaze run, the reference arguments are in
`submit_stream_gaze_ca_mtp.slurm`:

```text
--ca-aux gaze
--horizons-sec 2,4,6
--loss-weights 1.0,0.7,0.5
--primary-horizon-sec 2
--anticipation-sec 2
--max-frames 80 --fps 8 --img-size 256 --keep-count 4096
--cap-total-to-keep --max-aux-tokens 1600
--prune-mode postfuse_recency
--batch-size-by-frames 32:4,48:2,64:2,80:1
--epochs 8 --lr 2e-5 --fusion-lr-mult 2 --adapter-lr-mult 1
```

The 1,600-token cap is ll's corrected `29063fe` behavior.  The matched FFN arm
changes only:

```text
USE_FFN=1  # adds --use-ffn --ffn-mult 4
```

For ll's newer six-run Vanilla matrix, keep his current method routing and
replace only:

- train/val CSV paths with the frozen files above;
- video root with the nested 86-session view;
- gaze loading with `gaze_format=egtea`, `egtea_gaze_dir`, and the coverage
  manifest gate when that run uses gaze;
- checkpoint sidecars with the intended EGTEA counterparts;
- output directories and scheduler/environment paths.

`submit_stream_video_single_horizon.slurm` runs one separately trained `+2`,
`+4`, or `+6` model.  The frozen CSV stores labels in `[+2,+4,+6]` columns,
while ll's one-horizon trainer reads label position zero.  The launcher
therefore first calls `prepare_vanilla_single_horizon_csv.py`, which preserves
all 25,256/22,741 rows and all non-label fields but projects the chosen source
column to the single label position.  Do not bypass this adapter for +4/+6.
It is still not authority for the definition of ll's newer six-run matrix.

Runnable video-only commands are:

```bash
export PROJECT_ROOT=/path/to/JEPA_ARVR
export PYTHON=/scratch/ll5914/conda_envs/SVD/bin/python
export VJEPA_ROOT=/path/to/vjepa2
export EGTEA_ROOT=/scratch/ll5914/datasets/EGTEA
export CHECKPOINT=/path/to/vitl.pt
export ENCODER_LORA=/path/to/video/encoder_lora_best.pt
export PREDICTOR_LORA=/path/to/video/predictor_lora_best.pt

for HORIZON in 2 4 6; do
  export HORIZON
  export OUT_DIR="/scratch/ll5914/experiments/egtea_vanilla_video_h${HORIZON}"
  sbatch "$PROJECT_ROOT/scripts/egtea/submit_stream_video_single_horizon.slurm"
done
```

## Checkpoint roles for the current EGTEA mapping

These are source-workspace identities.  Transfer the files separately, place
them wherever ll uses, then set launcher environment variables.

| Role | Yifan source path | SHA-256 |
|---|---|---|
| ViT-L base | `checkpoints/vitl.pt` | `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012` |
| Gaze encoder LoRA | `outputs/egtea_v1_gaze_fixed_3ep_w8/.../encoder_lora_best.pt` | `6affa9ef3768121ad2b415985e54e6a981d59921ae1204df8df71c1954b97eb0` |
| Stream predictor LoRA | `outputs/egtea_stream_mtp/.../predictor_lora_from_stream_best.pt` | `f414f3ae1e545279408a1a27663d3cb4cdf80ac7651b7f43436b929b1128175b` |
| Video encoder LoRA | `outputs/egtea_split1_video_pred_joint_heads_ll_exact/.../encoder_lora_best.pt` | `dd60d54737db584bd0897922784c0e2909a7c98e90ed05bbaa55b3ccb4c3a4eb` |
| Video predictor LoRA | same directory, `predictor_lora_best.pt` | `1e70d73a311dd290df4e6137e4b0adfcc6dc8fbdd63b0d905463e59a1705895c` |

The current +gaze route starts the 4-channel input adapter, gaze fusion, and MTP
heads fresh.  If ll's six-run contract specifies a different role lineage,
retain his role semantics and use the matching EGTEA sidecar; never warm-start
fresh heads from an incompatible formal `best.pt`.

## RU-LSTM

### V-JEPA feature route

The following directory is exact ll source from `baseline@6ce4290`:

```text
baselines/rulstm_hdepic/
```

`submit_rulstm_vjepa_features.slurm` changes only dataset/checkpoint paths.
Keep extraction at 8-fps RGB, tubelet 2, image size 256, chunk 16, batch-chunks
4.  This creates 1,024-D features at 4 Hz.  The flat 86-session video layout is
required.

Yifan already has 86 `.npy` plus 86 `.json` files under
`data/egtea/rulstm_stream_features/vjepa_vitl_split1`; transfer them separately
or regenerate them with the supplied launcher.

All RU launchers call `verify_rulstm_feature_bundle.sh`: the `.npy` and `.json`
stems must each match the exact 86-session union in the frozen train/val CSVs.
This prevents a count-only pass with missing or stale session features.  Asset
provenance (extractor checkpoint, tensor shape, fps and bundle hashes) must
still be supplied with the transferred bundle; it is not encoded in Git.

### TSN small and Large v2

Both use the exact feature type under Yifan's
`data/egtea/rulstm_stream_features/rgb_ll_tsn_4fps` (86 feature/metadata pairs).
Transfer it separately unless ll has the exact same EGTEA extractor/checkpoint.

The valid Large configuration is the optimized shallow-wide v2:

```text
hidden=4096 depth=1 input_proj=true mlp_head=true trunk_layers=0
dropout=0.5
AdamW lr=1e-3 weight_decay=0.05 cosine warmup=3
label_smoothing=0.1 feat_noise=0.05 grad_clip=1 AMP=true
early_stop_patience=12 batch_size=8 epochs=40
init_from=<EGTEA small RU-LSTM best>
```

The small checkpoint used in Yifan's mapping has SHA-256
`0006dad80f575a936723e3c65990b844f986dc1fe8799901cf4e1059a5142b0a`.
Do not use the old `hidden=2048, depth=4` Large v1; ll's own source labels that
configuration as failed.

## ll-machine path changes

The launchers default to ll's usual locations but intentionally require the
ambiguous assets to be set explicitly.  At minimum set:

```bash
export PROJECT_ROOT=/path/to/JEPA_ARVR
export VJEPA_ROOT=/path/to/vjepa2  # required: this branch's vjepa2/ may be empty
export EGTEA_ROOT=/scratch/ll5914/datasets/EGTEA
export GAZE_ROOT=<directory-with-session-txt-files>       # gaze runs
export ENCODER_LORA=<correct-EGTEA-sidecar>
export PREDICTOR_LORA=<correct-EGTEA-sidecar>             # V-JEPA runs
export INIT_FROM=<EGTEA-small-rulstm-best>                # Large v2
```

Scheduler partition/account, Python environment, and output locations are
infrastructure changes and may be edited.  Do not change the frozen CSVs,
sampling/horizon arguments, label mapping, model route, feature type, or
checkpoint role without recording it as a protocol deviation.

## Preflight checklist

1. `scripts/egtea/verify_stream_split.sh` passes.
2. Exactly 86 full sessions exist in both required views.
3. Training-derived label maps are 19 verbs / 51 nouns / 106 actions.
4. Gaze audit reports train OOB 0 and exactly the frozen 171 validation rows.
5. Every checkpoint and feature source has the intended semantic role and
   recorded SHA-256.
6. One real 80-frame gaze batch passes before formal training.
7. Every output directory records branch commit, full argv, CSV MD5s, and
   checkpoint hashes.
8. No job resumes from a checkpoint produced by another method or horizon.
9. For independent Vanilla, `selected_horizon_csv/manifest.json` records the
   requested horizon and source column 0/1/2 before the trainer starts.

If a preflight fails, fix the data port or path.  Do not compensate by changing
ll's model implementation.
