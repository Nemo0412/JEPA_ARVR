# InAViT, official recipe, on a 4s / +2s window (EK100)

Branch `Baseline_InAViT`, based on `Baseline_ScalAnt` (`cdb3fea`). It
replaces the random-init / dummy-box InAViT run. On EK100 that run collapsed:
its predictions did not depend on the input and it scored the class prior
(action CMR@5 0.36 vs 0.367 for constant "top-5 frequent classes").

Agreed setup (2026-10-22): the InAViT authors' configuration and pretrained
weights, real hand/object boxes, and our horizon and input length.

## Taken unchanged from the released InAViT (`baselines/InAViT`)

| Part | Released source |
|---|---|
| Model | MotionFormer + HoIViT from `configs/INAVIT/EK_INAVIT_MF_ant.yaml`: O=4 objects, U=1 hand, HoIViT layers [1, 6, 10], motion stream, two direct heads (verb 97 / noun 300) |
| Initialization | `CP/ek_motionformer_224_16x4.pyth` (MotionFormer trained for EK100 recognition) through `slowfast.utils.checkpoint.load_checkpoint` with epoch reset: shape-matched tensors load, HoIViT modules without a counterpart stay fresh |
| Optimizer | `construct_optimizer`: AdamW, wd 0.05. Backbone and heads `BASE_LR` 1e-5, HoIViT parameters `HOIVIT_BASE_LR` 1e-4. `steps_with_relative_lrs` (x1 / x0.1 / x0.01 at epochs 0 / 29 / 60 of 100), set every iteration |
| Loss | `EKLoss` label smoothing 0.1, verb + noun |
| Precision / batch | FP16 autocast + GradScaler, no gradient clipping, global batch 20 (`--accum` keeps 20 if memory is short) |
| Augmentation | box-aware RandAugment `rand-m7-n4-mstd0.5-inc1` (bicubic), short-side jitter 256-320, random 224 crop, horizontal flip, mean/std 0.5. Eval: short side 224, centre crop, one view |
| Boxes | author h5 (Zenodo 14004926 `epickitchens.h5`): `video -> 1-based frame -> {obj, hand}`, rows `[x1, y1, x2, y2, track]`; `sort_boxes_sorted` slots; `prepare_boxes` (clamp, xyxy -> cxcywh, drop w or h <= 0.05) |

## Changed for our protocol

1. **Window.** 16 frames uniform over the 4s that end at action start - 2s,
   on the native frame clock, the same window as every compared baseline.
   The released EK loader observes 16x4 frames ending one frame before an
   offset *inside* the action segment, so it cannot be reused for a +2s
   horizon.
2. **Selection and scoring.** Select by validation action class-mean
   Recall@5 (first maximum). Verb/noun use the direct heads. Action =
   log p(verb) + log p(noun) over `actions.csv` (3,806 actions). Every
   validation row is scored; rows with no observation before the cutoff
   (B19 rule, 76 on EK100) are misses. Those rows are also dropped from
   training.

## Two loader fixes (deviations from the released code)

- **Hand boxes are augmented with the frames.** The released loader applies
  RandAugment, crop and flip to object boxes only, so hand boxes drift from
  the hands in training.
- **Pixel/normalized handling is explicit.** The released transforms work in
  pixels while the h5 boxes are normalized (checked by `audit_boxes.py`).
  Boxes are scaled to pixels first, transformed together with the frames, and
  divided by the crop size afterwards.

## Box frame clock (important)

The h5 keys are on the authors' **nominal** clock, not the native decoded
index: 60 fps for EK55 video ids (`P01_01`) and 50 fps for EK100-extension ids
(`P01_101`). The detections were run on re-encoded videos; the released
`EpicKitchensVideoRecord` uses the same rule. A decoded index `i` of a video
with native rate `vfps` therefore maps to key `round(i / vfps * F) + 1`
(`--box-clock nominal`, the default).

Checked over all 633 EK100 train+val videos:
- (max key / native frames) divided by (F / vfps) has median 1.0001.
- The last key falls about 0.06 s after the video ends.
- The 29.97 fps videos reach twice as many keys as frames.

With the native index, boxes would be up to about 1 s late at the end of long
59.94 fps videos, and wrong by 2x on 30 fps videos.

## Exactness

Each epoch uses a seeded permutation and a batch cursor. Each sample's
augmentation is seeded from (seed, epoch, row). Optimizer, GradScaler and RNG
states are checkpointed. A resumed run therefore replays the same batches and
augmentations. A segment checks that the uids it sees are the expected ones.

## Files

- `train_inavit_official.py`: trainer and evaluator (`--eval-only` replays `best.pt`).
- `audit_boxes.py`: h5 schema, coordinate range, frame-key clock, and per-row coverage of the requested frames.
- `test_inavit_official.py`: CPU tests for the window formula, box/frame alignment under eval and train
  geometry, the released box-aware RandAugment geometric ops, deterministic augmentation, and unobservable rows.

Inputs: frames as `<frames>/<video>/frame_{i+1:010d}.jpg` plus
`_meta.json` (`n_frames`). Rows CSV columns:
`uid, video_id, tick_frame, vfps, verb, noun, action, mask` (`mtp_*` /
`action_id` aliases are accepted).

```bash
python baselines/inavit_official/train_inavit_official.py \
  --train-csv EK100_train.csv --val-csv EK100_val.csv --actions-csv actions.csv \
  --frames FRAMES --boxes-h5 epickitchens.h5 --box-coords normalized \
  --checkpoint CP/ek_motionformer_224_16x4.pyth --out OUT [--epochs N --lr-steps 0 a b]
```

## Checks and smoke (EK100, 2026-10-22/23, H100)

- **Boxes.** `epickitchens.h5` MD5 matches Zenodo.
  - The audit (nominal clock) passes: all 633 videos are present, and 88% of
    the requested frames have a box entry. That matches the h5's own density;
    frames without an entry get empty boxes, as in the released loader.
  - 99.6% of rows have at least one object box and at least one hand box.
  - Coordinates are normalized (-0.16..1.16).
- **Initialization.** 205 backbone tensors load from the MotionFormer
  checkpoint (108.6M of 177.8M parameters). 138 HoIViT tensors are fresh.
- **CPU tests.** 7/7 pass.
- **Smoke A** (600 train / 400 val rows, batch 20, 2 x 15 steps): finite
  losses. GradScaler skipped one step at warm-up (normal for FP16). Peak
  memory 73.4 GB.
- **Smoke B** (same with `--accum 2`; stopped at step 8, then `--resume`):
  resumed at cursor 8 and passed the uid-order check through step 30. Peak
  memory 38 GB.
- **Zero-shot reference** (MotionFormer checkpoint alone, no training or
  boxes, same window): action CMR@5 4.96, verb 16.5, noun 23.1 on the full
  validation set.
