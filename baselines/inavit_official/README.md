# InAViT, official recipe, on a 4s / +2s window (EK100)

Branch `Baseline_InAViT`, based on `Baseline_ScalAnt` (`cdb3fea`). It
replaces the random-init / dummy-box InAViT run. On EK100 that run collapsed:
its predictions did not depend on the input and it scored the class prior
(action CMR@5 0.36 vs 0.367 for constant "top-5 frequent classes").

Agreed setup (2026-10-22): the InAViT authors' configuration and pretrained
weights, real hand/object boxes, and our horizon and input length.

## Result (EK100 validation, reported)

**Protocol.** Our baseline standard:
- Window: 16 frames uniform over the 4s ending at action start - 2s.
- Scoring: all 9,668 RU-LSTM validation rows; the 76 rows with no observation count as misses.
- Metric: class-mean Recall@5 over classes present in validation. Action = log p(v) + log p(n) top-5 over 3,806 actions.
- Selection: validation action CMR@5, first maximum.

| Model | Selected epoch | Action CMR@5 | Verb CMR@5 | Noun CMR@5 | Action micro top-5 |
|---|---:|---:|---:|---:|---:|
| InAViT (official recipe, real boxes, 4s / +2s) | 27 | **9.51** | 22.75 | 29.21 | 22.33 |

For reference only (not the reported metric), the same checkpoint under the released local MT5R rule (verb top-5 AND noun top-5) scores 18.95.

**Schedule.**
- The run followed the released schedule (100 epochs, LR x0.1 at 29) and was stopped by decision at epoch 39 because it had plateaued:
  - epochs 13-28: action 8.5-9.5;
  - after the LR drop at 29: epochs 30-38 at about 9.0 (8.86-9.16).
- The next drop at epoch 60 was not reached; this is the one deviation from the released 100-epoch schedule.
- Per-epoch history: `history.json` of the run.

**LR exploration** (not adopted).
- Scaling both learning rates x3 or x10 from the epoch-8 checkpoint did not speed up action CMR@5 on matched epochs:
  - x3: 7.76 / 8.27 / 7.49 vs official 8.01 / 8.46 / 7.83;
  - x10: about 6.3.
- The released learning rates were kept.

## Why this is far below the paper's 25.89 (EK100 val, "τa = 1s")

Full standalone report, including which metric we report and why: [`released_window/VERIFICATION.md`](released_window/VERIFICATION.md).

In the released `ek_MF/frame_loader.py`, `temporal_sampling` clamps the sampled offsets to `[0, num_frames - 1]` relative to the action's start frame (`num_frames` = action length).
- The anticipation shift moves the window before the start, but every pre-action offset is clamped back to the action's first frame. So the released loader never samples a frame before the action begins:
  - view 0 is about 16 copies of the action's first frame or two;
  - the other views are frames inside the action.
- Training uses the same function.
- The released `test_net` averages 10 temporal x 3 spatial views and scores action with the local MT5R rule above.

**Check** (`released_window/`). Setup:
- Frames re-extracted for these windows from the MD5-verified MP4s, using the official `EPIC_100_validation.csv` timestamps and the released record's nominal clock.
- Same 8,423-row subset in every cell (3 damaged MP4s, P22_01, P29_05 and P30_08, excluded); one view each; released loading (BGR, test-mode crop).
- `mf` = the pretrained EK100 MotionFormer zero-shot (a recognition model). `inavit` = the trained model above.

| Window | Model | Action CMR@5 | Verb | Noun | Action, released MT5R rule |
|---|---|---:|---:|---:|---:|
| W: ours, 4s ending 2s before start | mf | 7.23 | 19.12 | 27.34 | 15.35 |
| W | inavit | 9.75 | 22.55 | 29.10 | 19.24 |
| A1: honest 1s, 128 nominal frames ending 1s before start | mf | 8.43 | 21.43 | 29.29 | 17.67 |
| A1 | inavit | 9.71 | 22.71 | 30.73 | 20.34 |
| R0: released window, view 0 | mf | 6.96 | 25.90 | 27.26 | 14.25 |
| R0 | inavit | 4.69 | 17.36 | 21.39 | 12.69 |
| R5: released window, view 5 (inside the action) | mf | **15.91** | 37.41 | 37.76 | **27.12** |
| R5 | inavit | 8.95 | 25.97 | 29.43 | 19.59 |

What this shows:
- A recognition model with no anticipation training, fed the released in-action frames and scored with the released rule, reaches 27.1, about the paper's number.
- With honest 1s anticipation, the same model drops to 17.7 on that rule (8.4 action CMR@5).
- The trained model scores the same at honest 1s and at our 2s (9.71 vs 9.75), so the horizon is not why it trails the paper.

Caveats:
- Single views were used, not the 10x3 ensemble.
- The authors' own annotation pickle cannot be checked. The README points to the standard EK100 annotations, which are not shifted.

## Runtime note

Run with `MALLOC_MMAP_THRESHOLD_=1048576`. glibc's dynamic mmap threshold fragments the DataLoader workers' heaps on the evaluation path (full-size float clip buffers): worker memory grows about 0.8 GB per validation batch and killed two runs at the first epoch end, even with 128 GB of host memory. With the fixed threshold, worker memory stays flat. This is an environment variable only; nothing is computed differently.

## Principle

This is a faithful reproduction of the released InAViT code. Only the input
length and the horizon change. Released behaviour is kept even where it looks
questionable (decision 2026-10-23).

## Taken unchanged from the released InAViT (`baselines/InAViT`)

| Part | Released source |
|---|---|
| Model | MotionFormer + HoIViT from `configs/INAVIT/EK_INAVIT_MF_ant.yaml`: O=4 objects, U=1 hand, HoIViT layers [1, 6, 10], motion stream, two direct heads (verb 97 / noun 300) |
| Initialization | `CP/ek_motionformer_224_16x4.pyth` (MotionFormer trained for EK100 recognition) through `slowfast.utils.checkpoint.load_checkpoint` with epoch reset: shape-matched tensors load, HoIViT modules without a counterpart stay fresh |
| Optimizer | `construct_optimizer`: AdamW, wd 0.05. Backbone and heads `BASE_LR` 1e-5, HoIViT parameters `HOIVIT_BASE_LR` 1e-4. `steps_with_relative_lrs` (x1 / x0.1 / x0.01 at epochs 0 / 29 / 60 of 100), set every iteration |
| Loss | `EKLoss` label smoothing 0.1, verb + noun |
| Precision / batch | FP16 autocast + GradScaler, no gradient clipping, global batch 20 (the released 4 GPUs x 5; the model has no BatchNorm). Train loader `drop_last=True` |
| Frame decoding | `utils.retry_load_images` (cv2: **BGR** channel order, as the released model and its MotionFormer init were trained) |
| Train augmentation | `Epickitchens_dataset._aug_frame` called as released: box-aware RandAugment `rand-m7-n4-mstd0.5-inc1`, short-side jitter 256-320, random 224 crop, flip, mean/std 0.5 |
| Eval view | released test-mode path (`tensor_normalize` + `utils.spatial_sampling`, short side 224, centre crop) |
| Boxes | author h5 (Zenodo 14004926 `epickitchens.h5`): `video -> 1-based frame -> {obj, hand}`, rows `[x1, y1, x2, y2, track]`. Read as `EKBoxes.get_boxes` (`sort_boxes_sorted`), then `EKBoxes.prepare_boxes` |

### Released box behaviour, kept as is (consequence measured)

- The released transforms apply geometric changes to **object boxes only**.
  Hand boxes keep their original-frame coordinates through crop and flip.
- The h5 boxes are normalized, but the released transforms treat them as
  pixels: they scale them by the resize ratio and subtract a pixel crop
  offset. They then clip, and `prepare_boxes` drops anything with width or
  height <= 0.05.
- Measured on 400 random train / 400 val rows (`box_survival.py`):
  - **Object boxes after the released loader: 0% non-empty** in training and
    in evaluation, versus 13% / 12% of object slots non-empty in the raw h5.
  - **Hand boxes: unchanged**, 73% / 75% of slots non-empty.
- So the faithful model receives hand boxes (unaugmented) and no object boxes.

## Changes (intended or protocol-level)

1. **Window and horizon (intended).** 16 frames uniform over the 4s that end
   at action start - 2s, on the native frame clock. This replaces the released
   EK window, which observes 16x4 frames ending one frame before an offset
   inside the action segment. Boxes are read for the same instants on the
   authors' clock (next section).
2. **Shared comparison protocol.** These are the same for every baseline in
   our table:
   - Selection by validation action class-mean Recall@5 (first maximum). The
     released code trains 100 epochs with no selection; its in-training val
     uses random crops.
   - Single deterministic view. The released test averages 10 temporal x 3
     spatial views; with a fixed window only one temporal position exists.
   - Action score = log p(verb) + log p(noun) over `actions.csv` (3,806). This
     matches how the EK100 server ranks actions from the verb/noun scores in
     the released challenge JSON. The released local `MT5R` counts an action
     as hit when the verb is in the verb top-5 and the noun is in the noun
     top-5 (up to 25 pairs). That number is reported too
     (`action_released_mt5r_rule`), but it is not used for selection.
   - Rows with no observation before the cutoff: dropped from training (230)
     and scored as misses in validation (76). The released loader would clamp
     the window to the video start.
3. **Engineering only.**
   - A failing sample raises. The released `__getitem__` silently substitutes
     the next index.
   - Per-sample seeded augmentation and exact resume. Same distributions,
     different random streams.
   - Frames come from the MD5-verified original MP4s, decoded at the native
     rate and stored as 456x256 JPEG (q90). The released code reads the EPIC
     `rgb_frames` JPEGs of the authors' re-encoded 60/50 fps videos. The
     sampled instants are the same; pixels differ only by decoding and JPEG
     re-encoding.

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
- `released_window/`: the check above. `extract_released_windows.py` writes the window plans and frames from the original MP4s; `eval_released_windows.py` evaluates a window/model pair; `summarize_released_windows.py` builds the table. Paths are those of our cluster; adjust the constants at the top of each file.
- `test_inavit_official.py`: CPU tests for the window formula, the box clock, BGR decoding as released,
  the released eval box path, deterministic augmentation, and unobservable rows.

Inputs: frames as `<frames>/<video>/frame_{i+1:010d}.jpg` plus
`_meta.json` (`n_frames`). Rows CSV columns:
`uid, video_id, tick_frame, vfps, verb, noun, action, mask` (`mtp_*` /
`action_id` aliases are accepted).

```bash
python baselines/inavit_official/train_inavit_official.py \
  --train-csv EK100_train.csv --val-csv EK100_val.csv --actions-csv actions.csv \
  --frames FRAMES --boxes-h5 epickitchens.h5 \
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
- **CPU tests.** 6/6 pass (released-loader version).
- Smokes A-C below ran on an earlier revision of the trainer (RGB decoding,
  with hand-box augmentation and pixel conversion). Model, optimizer and speed
  are unchanged; smoke D re-checks the released-loader revision.
- **Smoke A** (600 train / 400 val rows, batch 20, 2 x 15 steps): finite
  losses. GradScaler skipped one step at warm-up (normal for FP16). Peak
  memory 73.4 GB.
- **Smoke B** (same with `--accum 2`; stopped at step 8, then `--resume`):
  resumed at cursor 8 and passed the uid-order check through step 30. Peak
  memory 38 GB.
- **Throughput smoke C** (1,000 steps x batch 20, 14 workers): steady 0.58 s
  per step (about 35 clips/s). Mean GPU utilisation 83% over the whole job.
  Train loss 7.97 -> 6.82. GradScaler skipped 2 of 1,000 steps for overflow.
  - Projection for a full EK100 epoch: 3,350 steps, about 32 min of training
    plus validation, roughly 36 min.
  - So 100 epochs come to about 60 H100-hours.
- **Zero-shot reference** (MotionFormer checkpoint alone, no training or
  boxes, same window, full validation set):
  - In the released BGR order: action CMR@5 **6.96**, verb 19.2, noun 27.4.
  - Fed RGB, the same checkpoint scores 4.96 / 16.5 / 23.1, which confirms
    the checkpoint expects BGR.
