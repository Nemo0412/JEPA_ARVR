# B14 early-exit reproduction interface

This document is the public, path-neutral interface for reproducing the B14
predictor/encoder early-exit diagnostics synchronized from VJEPA2-EXP commit
`4f66ea6`. Internal scheduler logs and experiment-registry files are deliberately
not copied into this repository.

## Scope

B14 evaluates a trained one-step action-anticipation checkpoint in three stages:

1. expose intermediate predictor depths and compare them with the complete
   predictor;
2. expose intermediate encoder depths and compare them with the complete
   encoder while keeping the full predictor/classifier;
3. optionally train a small per-token projector from an encoder prefix into the
   final encoder representation space.

The diagnostic supports a checkpoint-specific split. Do not replace the split
used to train a checkpoint with `p01_fixed` merely because both are HD-EPIC.

## Required checkout and artifacts

Initialize the pinned V-JEPA2 submodule, then provide:

- a training YAML accepted by `app.hdepic_lora_action_anticipation.eval`;
- the exact train/validation CSV directory associated with the checkpoint;
- a checkpoint directory containing `best.pt`, `encoder_lora_best.pt`, and
  `predictor_lora_best.pt`;
- `binary_input_adapter_best.pt` as well when the YAML enables the binary input
  adapter;
- the pretrained checkpoint referenced by the training YAML.

The analyzer validates that train and validation CSVs share one directory and,
when `EXPECTED_SPLIT_DIR` is supplied, aborts before model loading if the paths
do not match. `SPLIT_LABEL` is written into `summary.json`; it is descriptive
metadata, not a substitute for the path check.

## Predictor exits

The default predictor depths are `3,6,9,12`. The final depth is checked against
the unmodified model before any result is accepted: maximum token and action
logit differences must be at most `1e-4`, and Top-1 predictions must be equal.

```bash
mkdir -p logs
export PROJECT_ROOT="$PWD"
export CONFIG=/path/to/parent_checkpoint_training_config.yaml
export CKPT_DIR=/path/to/parent_checkpoint_directory
export EXPECTED_SPLIT_DIR=/path/to/checkpoint_bound_split
export SPLIT_LABEL=checkpoint_bound_split
export MAX_SAMPLES=128          # 0 means the complete validation loader
export DEPTHS=3,6,9,12
sbatch scripts/run_predictor_early_exit_entropy_smoke.slurm
```

For a full validation pass, set `MAX_SAMPLES=0` and give the run an appropriate
wall-time. The launcher accepts `BATCH_SIZE`, `NUM_WORKERS`, `OUT_DIR`, `SIF`,
`OVERLAY`, and `ENV_NAME` overrides.

## Encoder exits

The encoder diagnostic runs every selected prefix through the encoder's shared
final normalization, the complete predictor, and the existing classifier. The
default depths are `6,12,18,24`; depth 24 is checked against the original full
forward using the same equivalence gate.

```bash
mkdir -p logs
export PROJECT_ROOT="$PWD"
export CONFIG=/path/to/parent_checkpoint_training_config.yaml
export CKPT_DIR=/path/to/parent_checkpoint_directory
export EXPECTED_SPLIT_DIR=/path/to/checkpoint_bound_split
export SPLIT_LABEL=checkpoint_bound_split
export MAX_SAMPLES=8
export DEPTHS=6,12,18,24
sbatch scripts/run_encoder_early_exit_entropy_smoke.slurm
```

This is a compatibility gate, not an accuracy estimate. Expand the sample count
only when an early prefix has meaningful agreement with the full encoder.

## Output contract

Both diagnostics write the following files under `OUT_DIR`:

| File | Contract |
|---|---|
| `summary.json` | split provenance, sample count, exit depths, equivalence check, and aggregate metrics |
| `per_sample_depth_entropy.csv` | metadata, label/prediction, normalized action entropy, Top-1/5 hits, final-depth agreement, and Top-5 Jaccard |
| `action_logits_fp16.pt` | action labels and FP16 logits keyed by depth for offline policy analysis |

Normalized entropy is `-sum(p log p) / log(C)` over the action head. Agreement
with the final depth measures behavioral stability, not correctness.

Human-readable disagreement cases can be extracted without a GPU:

```bash
python scripts/extract_b14_predictor_failure_cases.py \
  --per-sample-csv /path/to/per_sample_depth_entropy.csv \
  --train-csv /path/to/HD_EPIC_train_vjepa.csv \
  --val-csv /path/to/HD_EPIC_val_vjepa.csv \
  --verb-classes-csv /path/to/HD_EPIC_verb_classes.csv \
  --noun-classes-csv /path/to/HD_EPIC_noun_classes.csv \
  --output /path/to/paired_cases.json
```

## Encoder-prefix projector

Direct encoder prefixes can be distributionally incompatible with a predictor
trained on the final encoder state. The projector experiment freezes the
original encoder, predictor, LoRA weights, and classifier; it trains only an
identity-initialized `LayerNorm + Linear(embed_dim, embed_dim)` map. Its loss is
the weighted sum of representation `MSE + cosine` and action/verb/noun cross
entropy.

Copy and edit the example rather than committing machine-specific paths:

```bash
cp configs/examples/b14_encoder_d18_projector.yaml /path/to/run.yaml
# Edit source_config, checkpoint_dir, expected_split_dir, output_dir, and split_label.
mkdir -p logs
RUN_CONFIG=/path/to/run.yaml sbatch scripts/run_encoder_exit_projector_train.slurm
```

The trainer writes `metrics.jsonl`, `summary.json`, `latest.pt`, and `best.pt`.
`resume_checkpoint` may point to a previous `latest.pt`. The provided launcher
also samples GPU and host utilization; resource-policy thresholds remain a
cluster-local decision.

## Synchronized reference result

The source experiment used a checkpoint-bound random clip split with 5907 raw
train rows, 1477 raw validation rows, and 1336 effective validation items after
the checkpoint's train-only class filtering. These values describe that
checkpoint only.

On all 1336 effective validation samples, predictor d3/d6/d9/d12 action Top-5
was `42.515/42.216/42.590/42.440%`; Top-1 agreement with d12 was
`95.135/95.584/96.257/100%`. Static predictor truncation was therefore worth
latency benchmarking, while normalized entropy alone was not a reliable
dynamic-routing rule.

The encoder zero-training gate used only eight samples. d6/d12/d18 had zero
Top-1 agreement with d24, so direct coarse encoder truncation was rejected; the
projector is a separate learned-alignment hypothesis. At the synchronized
source commit, the projector smoke had been submitted but no completed result
was recorded. Do not infer a projector result from this document.

## Reproducibility checklist

- Record the source config and checkpoint revision or checksum.
- Record the physical split directory, raw CSV row counts, and effective loader
  sample count.
- Require the full-depth equivalence gate to pass.
- Report `exit_module`, depth, sample count, split label, Top-1/5, normalized
  entropy, final-depth Top-1 agreement, and Top-5 Jaccard together.
- Treat a subset smoke as workflow validation, not a final metric.
- Benchmark true prefix-only execution before claiming latency savings; the
  analyzer intentionally computes several exits in one offline pass.
