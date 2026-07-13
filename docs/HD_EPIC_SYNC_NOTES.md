# HD-EPIC synchronized implementation notes

This page is the minimal operational context for the synchronized HD-EPIC
code and launchers. It intentionally replaces the source repository's internal
experiment logs rather than copying them.

## Reference-aligned defaults

Runs entering through `app.hdepic_lora_action_anticipation` default to:

- `class_space: phd_reference`: full HD-EPIC verb/noun taxonomies and the
  all-P01 primary `(verb, noun)` action map.
- `temporal_sampling: phd_reference`: observation ends at
  `action_start - anticipation_sec`; train and validation use the same horizon.
- `align_reference_metrics: true`: logged accuracy is Top-3 accuracy and logged
  recall is class-mean Recall@5.
- Encoder-LoRA on all encoder blocks, targeting `attn.qkv` and `attn.proj`,
  with rank 8, alpha 16, LoRA LR multiplier 0.5, and weight decay `1e-4`.

The local LoRA default includes dropout `0.05`; use `0.0` for strict parity
with the reference implementation. Predictor-LoRA is a project extension and
has no reference counterpart.

## Data and split contract

Use `scripts/convert_hdepic_to_vjepa_csv.py --split-preset p01_fixed` (or the
matching refresh launcher) for comparable runs. The fixed P01 split contains
20 train, 2 validation, and 5 test videos. Raw CSV row counts are expected to
be 5965 / 375 / 1044 when all annotations are present.

Training selects checkpoints on validation only. The final test pass writes
`test_log_r*.csv`; test metrics must not be used for checkpoint selection.

`legacy` means the pre-alignment behavior: train-observed class maps, legacy
label conversion/split behavior, and configured action-end temporal anchors.
Legacy and `p01_fixed` results are not directly comparable. In particular,
do not trust a run tag alone: verify the generated CSV metadata and loader
sample counts.

## Paths and scheduler settings

Synchronized files are sanitized. Replace `/path/to/VJEPA2-EXP`, `<email>`,
and `your_slurm_account`, or override the corresponding environment variables,
before submitting jobs.

Generated configs are retained because many launchers consume them directly;
internal run logs, dashboards, and historical result documents are not copied.
