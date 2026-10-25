# InAViT on EK100: what the published 25.89 measures, and what we report

Date: 2026-10-25. Scope: EK100 validation, RGB.

The InAViT paper reports action mean top-5 recall 25.89 on EK100 validation, described as anticipation 1s
before the action. Our faithful reproduction of the released recipe reaches 9.51 under our protocol (2s ahead,
4s of observation). This report explains the gap, shows the check we ran, and states which number we report
and why.

## 1. Code finding: the released loader never looks before the action

The upstream release (LAHAproject/InAViT `a699dc8`, `slowfast/datasets/ek_MF/frame_loader.py`) builds the clip
as follows:

```python
start_idx, end_idx = get_start_end_idx(video_record.num_frames,            # num_frames = action length
                                       num_samples * sampling_rate * fps / target_fps,
                                       temporal_sample_index, cfg.TEST.NUM_ENSEMBLE_VIEWS)
if cfg.TRAIN.DATASET == "epickitchens" and cfg.EPICKITCHENS.ANTICIPATION:
    if cfg.TRAIN.ENABLE or cfg.EPICKITCHENS.TEST_LIST == "EPIC_100_validation.pkl":
        end_idx = start_idx - fps / target_fps                               # shift before the offset
        start_idx = end_idx - num_samples * sampling_rate * fps / target_fps
        if start_idx < 0:
            start_idx = 0
start_idx, end_idx = start_idx + 1, end_idx + 1
frame_idx = temporal_sampling(video_record.num_frames, start_idx, end_idx, num_samples,
                              start_frame=video_record.start_frame)

def temporal_sampling(num_frames, start_idx, end_idx, num_samples, start_frame=0):
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, num_frames - 1).long()                    # clamp INTO the action
    return start_frame + index
```

Here `start_frame` is `round(start_timestamp * fps)` from the standard `EPIC_100_*` annotations, with
`fps` = 60 for EK55 videos and 50 for EK100 videos.

- **Offsets are measured from the action's start frame and clamped to `[0, action length - 1]`.** The
  anticipation shift moves the window before the offset, but every pre-action position is clamped back to
  the action's first frame. The returned frames therefore always lie inside the action segment
  `[start, stop)`.
  - Temporal view 0 is about 16 copies of the action's first one or two frames.
  - Views 1-9 are frames from inside the action.
  - Training (random offset in `[0, delta]`) goes through the same function.
- **Evaluation** (`tools/test_net.py`) averages 10 temporal x 3 spatial views. It scores action with the
  local `slowfast/utils/metrics.MT5R`, which counts an action as hit when the verb is in the verb top-5
  **and** the noun is in the noun top-5. That allows up to 25 verb x noun pairs instead of 5 ranked actions.
- **No 1s gap appears anywhere in the code.** The README points to the standard EK100 annotations, which
  are not shifted.

The collaborator's copy in this repository (`baselines/InAViT`) already replaced this branch with an
explicit `OBS_SEC` / `ANTICIPATION_SEC` window that ends before the action. Our trainer
(`train_inavit_official.py`) does not use either loader window: it builds the window itself (section 4).

## 2. Check: same models, different windows

**Windows.** All are 16 frames on the authors' nominal clock (60/50 fps). Frame m maps to native decoded
index `round((m-1)/F * vfps)`; the h5 box key is m.

| Name | Definition |
|---|---|
| **W** (ours) | Uniform over the 4s ending 2s before the action start. This is the window of every baseline in our comparison. |
| **A1** | Honest 1s anticipation: uniform over the released clip span (16 x 4 x F/30 = 128 nominal frames, about 2.1s), ending 1s before the action start. No clamping into the action. 28 rows have no such window and count as misses. |
| **R0** | Released window, temporal view 0, computed exactly as in section 1 (clamp included). |
| **R5** | Released window, temporal view 5 (middle of the action). |

**Models.**
- `mf`: the pretrained EK100 MotionFormer (`ek_motionformer_224_16x4.pyth`, InAViT's initialization), used
  zero-shot. It is an action **recognition** model and has had no anticipation training.
- `inavit`: our trained InAViT, released recipe with real boxes, trained on W, selected epoch 27.

**Common settings.**
- Loading: released `retry_load_images` (BGR) and released test-mode crop (short side 224, centre crop),
  one view.
- Scoring: class-mean Recall@5 (section 3), plus the released MT5R rule for reference.
- Frames are re-extracted for these windows from the MD5-verified original MP4s with the same pixel recipe
  as training (456x256 bilinear, JPEG q90). Window positions use the official `EPIC_100_validation.csv`
  timestamps.
- Every cell uses the same 8,423 of 9,668 validation rows. Three videos (P22_01, P29_05, P30_08) have damaged
  MP4 packets; they are excluded from all cells, not scored as misses.

| Window | Model | Action CMR@5 | Verb | Noun | Action, released MT5R rule |
|---|---|---:|---:|---:|---:|
| W (ours) | mf | 7.23 | 19.12 | 27.34 | 15.35 |
| W (ours) | inavit | 9.75 | 22.55 | 29.10 | 19.24 |
| A1 (honest 1s) | mf | 8.43 | 21.43 | 29.29 | 17.67 |
| A1 (honest 1s) | inavit | 9.71 | 22.71 | 30.73 | 20.34 |
| R0 (released, view 0) | mf | 6.96 | 25.90 | 27.26 | 14.25 |
| R0 (released, view 0) | inavit | 4.69 | 17.36 | 21.39 | 12.69 |
| R5 (released, view 5, inside the action) | mf | **15.91** | 37.41 | 37.76 | **27.12** |
| R5 (released, view 5, inside the action) | inavit | 8.95 | 25.97 | 29.43 | 19.59 |

**Reading.**
1. **The published number matches recognition, not anticipation.** A recognition model with no
   anticipation training, given the released in-action frames and scored with the released rule, reaches
   27.12, which matches the paper's 25.89. The released evaluation averages 30 views, most of them inside
   the action as R5 is.
2. **Honest anticipation drops the same model far below the paper.** At an honest 1s, the same model scores
   17.67 on the released rule and 8.43 on action CMR@5.
3. **The horizon does not explain our gap.** Our trained InAViT scores the same at an honest 1s (9.71) and at
   our 2s (9.75).
4. **R0 is near-static.** About 16 copies of the action's first frame carry little information, so both
   models score low there. Our InAViT, trained on pre-action windows, is also out of distribution on
   in-action frames (R5 8.95 vs mf 15.91).

**Caveats.**
- We used single views R0 and R5, not the full 10 x 3 ensemble.
- We cannot inspect the authors' own annotation pickle. A pickle pre-shifted by 1s would change the picture;
  the released README uses the standard, unshifted annotations.

## 3. What we report, and why

**Reported (EK100 validation):** InAViT action / verb / noun CMR@5 = **9.51 / 22.75 / 29.21** (action micro
top-5 22.33). Window W, all 9,668 rows, selected epoch 27. The released MT5R rule gives 18.95 for the same
checkpoint; it is listed only as a reference.

Definition:
- **Metric: class-mean Recall@5 (CMR@5).** For each class present in validation, the fraction of its
  instances whose label is in the model's top 5; then the mean over those classes. Reported for action,
  verb and noun.
- **Action ranking.** Action scores are log p(verb) + log p(noun) over the 3,806 actions in `actions.csv`;
  the top 5 actions are taken. Verb and noun use the model's direct heads.
- **Population.** All 9,668 RU-LSTM / official validation rows. The 76 rows whose action starts less than
  2s after the video start have no observation before the cutoff; they count as misses (and are excluded
  from training).
- **Window.** 16 frames uniform over the 4s ending 2s before the action start; one deterministic view
  (released test-mode crop).
- **Selection.** The epoch with the highest validation action CMR@5 (first maximum). EK100 test labels are
  not public, so validation is used for both selection and reporting, as for every baseline in the table.

Reasons:
1. **One protocol for every method.** RU-LSTM (8.33), ScalAnt (11.11) and our method are all scored this way:
   same window, horizon, population, metric and selection. A number from another protocol (the paper's
   25.89) cannot share a column with them.
2. **It measures anticipation.** Section 2 shows the released window measures recognition of the ongoing
   action. W observes only frames that end 2s before the action, which is the task we compare on.
3. **CMR@5 is the EK100 anticipation metric.** The official challenge uses class-mean top-5 recall, which
   weighs the long tail of the 3,806 actions instead of the frequent ones. Ranking actions from the
   verb/noun scores matches how the EK100 evaluation derives action scores from submitted verb and noun scores
   (the released challenge JSON contains only verb and noun).
4. **The released local rule is lenient, so it is reference only.** "Verb in top-5 and noun in top-5" can
   accept up to 25 verb x noun combinations. It is about 2x looser (18.95 vs 9.51 for the same
   checkpoint) and is not the challenge metric.
5. **Misses instead of exclusions keep the population fixed.** Every method is scored on the same 9,668
   rows. A method cannot gain by dropping rows it has no observation for.
6. **One view, as for the other baselines.** The released 10 x 3 view ensemble is test-time augmentation
   that no other method in the table uses. With a fixed pre-action window there is only one temporal
   position.

## 4. The reproduction behind the reported number (summary)

See `../README.md` for full details.
- **Taken unchanged from the release:** the EK100 InAViT config and model (MotionFormer + HoIViT), the
  MotionFormer initialization, the released checkpoint loader, the optimizer (backbone 1e-5, HoIViT 1e-4,
  AdamW), the LR steps, the label-smoothed loss, FP16, and the released frame decoding and augmentation code
  (BGR; released box handling kept as is). Boxes come from the author h5 on the authors' 60/50 fps frame
  clock.
- **Changed:** only the window and horizon (W), plus the shared protocol of section 3.
- **Schedule:** the released 100-epoch schedule was stopped at epoch 39 on a plateau. Epochs 30-38, after
  the first LR drop, held at about 9.0. The selected epoch is 27.

## 5. Reproduce

```bash
# 1. window plans + frames for the validation videos (CPU; needs the original MP4s)
python released_window/extract_released_windows.py OUT_DIR P01_11 P01_12 ...
# 2. one window x model cell (GPU); variant in {R0, R5, A1}, model in {mf, inavit}
python released_window/eval_released_windows.py R5 mf OUT_DIR/results_R5_mf.json
# 3. table, including W from the saved training predictions
python released_window/summarize_released_windows.py
```

Paths at the top of each script are those of our cluster; adjust them before running. Results of this run:
`outputs/inavit_official_ek100/released_window/` (`results_*.json`, `plans/`, `summary.md`).
