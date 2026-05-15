# Code Comparison

Date: 2026-05-14

Reference repo inspected locally: `D:\Projects\JEPA_ARVR`

## Summary

The reference implementation is not a LoRA finetune path. It is a standalone HD-EPIC probe training baseline:

- Freeze a V-JEPA2 ViT-L encoder.
- Build a new `HDEpicProbe` from scratch.
- Train an `AttentivePooler` plus verb/noun/action heads directly on HD-EPIC P01.
- Use custom `Dataset` / `DataLoader` instead of V-JEPA's `action_anticipation_frozen` dataloader.

Our current path remains the LoRA-style experiment:

- Use official ViT-g/384 encoder and predictor checkpoint.
- Load official EK100 ViT-g/384 probe pooler weights.
- Insert LoRA adapters into `AttentiveClassifier.pooler`.
- Train LoRA adapters plus new HD-EPIC heads.
- Keep the upstream `vjepa2/` repo untouched and invoke through `eval_name: app.hdepic_lora_action_anticipation`.

## Architecture Difference

| Component | Reference implementation | Current VJEPA2-EXP implementation |
| --- | --- | --- |
| Encoder | `vit_large_rope` | official ViT-g/384 eval template |
| Encoder checkpoint | `/scratch/ll5914/models/vjepa2/vitl.pt` | `/scratch/yh6416/VJEPA2-EXP/checkpoints/vitg-384.pt` |
| Input resolution | 256 | 384 |
| Frames / FPS | 32 frames, 8 FPS | 32 frames, 8 FPS |
| Probe | New `HDEpicProbe` from scratch | EK100 `AttentiveClassifier` pooler + LoRA |
| Pooler init | Random | official EK100 `ek100-vitg-384.pt` pooler |
| Trainable params | Full probe | LoRA adapters + heads |
| Upstream V-JEPA eval | Bypassed | Reused |

The encoder difference means raw results should not be compared as equivalent. If strict comparison is needed later, both runs should share the same backbone, input resolution, split, and metrics.

## Data And Split

Reference implementation:

- Reads `HD_EPIC_Narrations.pkl` directly.
- Uses raw P01 mp4 paths.
- Uses a date split:
  - train: videos whose `video_id` contains `20240203`
  - val: remaining P01 dates

Current implementation:

- Converts HD-EPIC annotations to EK100-style CSVs.
- Uses an EK100-style symlink tree.
- Current P01 split is video-level random ratio split.

Decision: keep our random/video-level split for now. The reference split is useful context, but changing split before the next clean P01 LoRA run would make our own runs harder to compare.

## Metrics To Align

Reference implementation reports:

- Verb Top-3
- Noun Top-3
- Action Top-3
- Verb Recall@5, class mean
- Noun Recall@5, class mean
- Action Recall@5, class mean

Current V-JEPA eval reports:

- action accuracy
- verb accuracy
- noun accuracy
- action recall
- verb recall
- noun recall

Important detail: upstream V-JEPA `ClassMeanRecall(k=5)` returns an `accuracy` field that is actually top-5 accuracy, while `recall` is class-mean Recall@5.

Decision: align metrics, not the encoder. In the project-local LoRA sidecar, `experiment.lora.align_reference_metrics=true` monkey-patches the metric class so:

- logged `acc` = Top-3 accuracy
- logged `recall` = class-mean Recall@5

This keeps the existing V-JEPA CSV/log field names but changes their semantics for the LoRA side path. Treat post-patch LoRA logs as PhD-reference-aligned metrics.

## Why Not Switch The Current LoRA Path To ViT-L

The current LoRA design depends on matching the encoder/pooler dimensions of the official EK100 probe checkpoint. We used ViT-g/384 because the official V-JEPA2 release includes a matching pair:

- `vitg-384.pt`
- `ek100-vitg-384.pt`

There is no currently integrated matching ViT-L/384 LoRA setup in this project. Switching the LoRA path to ViT-L would require locating a compatible encoder/probe pair or changing the probe initialization design. For now, keep ViT-g/384 and make the metric reporting comparable.

## Useful Ideas To Borrow

- Add Top-3 metrics alongside current Top-1 accuracy.
- Keep Recall@5/class-mean metrics, which are already equivalent to the reference script's recall target.
- Consider preserving split metadata in checkpoints/logs.
- For quick manual sanity checks, a direct single-video inference script can be useful, but it should stay separate from the main benchmark path.
