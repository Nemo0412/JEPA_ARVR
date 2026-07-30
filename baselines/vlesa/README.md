# VLESA: Vision-Language Embodied Safety Agent
This is the official code for "VLESA: Vision-Language Embodied Safety Agent for Human Activity Monitoring" [PDF](https://arxiv.org/pdf/2606.03954). 

## Environment Setup

Tested with Python 3.10, CUDA 12.4.

```bash
pip install torch openai flash-attn transformers qwen-vl-utils datasets tqdm matplotlib
```

For RL post-training of the Qwen3-VL safety filter, follow the [EasyR1](https://github.com/hiyouga/easyr1) instructions.

**API Keys**: Set the following environment variables as needed:
- `LLAMA_API_KEY` — for Llama API (Scout/Maverick models)
- `OPENAI_API_KEY` — for OpenAI API variants

## Data Preparation

### EgoSafety Dataset

The pre-built EgoSafety dataset is available on Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("hanjianghu/EgoSafety")
```

It contains 9,596 train / 1,202 validation / 1,183 test samples with egocentric frames (pre-action, point-of-no-return, post-action) and safe/unsafe labels with reasoning.

To rebuild from scratch, follow [EASG](https://github.com/fpv-iplab/EASG) to download Ego4D with **Forecasting + Hands & Objects (FHO)** labels and extract per-frame images.

Then generate unsafe scene graph variants from safe Ego4D action graphs using a VLM:

```bash
python data/construct_egosafety.py \
  --graph_file /path/to/EASG_unict_master_final.json \
  --summary_file /path/to/full_summaries_completed_task.json \
  --output-dir ./data
```

This produces parts of `unsafe_scene_graphs_vlm_raw_ALL.json` (a pre-generated version is included in `data/`).

### Package into HuggingFace Format

```bash
python data/data_preprocess_single_current.py
```

Edit the `safety_json_path` and `frames_dir` variables at the top of `main()` to point to your `unsafe_scene_graphs_vlm_raw_ALL.json` and the directory of extracted Ego4D frames. The script builds a `DatasetDict` with train/validation/test splits and saves it to disk as `egosafety_single_current_dataset/`.

Then customize the dataset by following EasyR1 instructions [here](https://github.com/hiyouga/easyr1#custom-dataset) for VLM RL post-training.

### Safety Filter Checkpoint

The VLESA safety filter checkpoint (Qwen3-VL-2B, GRPO fine-tuned) is available on Hugging Face:

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "hanjianghu/VLESA-Qwen3-VL-2B-Safety-Filter",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("hanjianghu/VLESA-Qwen3-VL-2B-Safety-Filter")
```

To reproduce RL post-training with EasyR1, merge the actor checkpoint:

```bash
python3 scripts/model_merger.py \
  --local_dir checkpoints/vlesa/qwen3_vl_2b_grpo/global_step_148/actor
```

The merged HuggingFace-format checkpoint is passed via `--safety-model`.

### ASIMOV-2.0-Video Benchmark

For external evaluation, download from [https://asimov-benchmark.github.io/v2/](https://asimov-benchmark.github.io/v2/) and extract frames into a directory where each subdirectory contains frames for one video:

```
/path/to/asimov_video/extracted_data/
├── video_0001/
│   ├── frame_0000.jpg
│   └── ...
└── video_0002/
    └── ...
```



## Running VLESA

### On ASIMOV Benchmark (VLESA — fine-tuned safety filter)

```bash
python vla_asimov_llamaapi.py \
  --frame-dirs-mode \
  --base-frames-dir /path/to/asimov_video/extracted_data \
  --max-abs-error 3.0 \
  --num-predictions 3 \
  --safety-model /path/to/merged_checkpoint
```

### On ASIMOV Benchmark (Baseline — prompt-based safety)

```bash
python vla_asimov_baseline.py \
  --frame-dirs-mode \
  --base-frames-dir /path/to/asimov_video/extracted_data \
  --max-abs-error 3.0 \
  --num-predictions 3
```

### On Real Egocentric Video
For OpenAI backend, use vla_offline_video.py, for Llama backend, use vla_offline_video_llamaapi.py.
```bash
python vla_offline_video.py \
  --frame-dirs-mode \
  --base-frames-dir /path/to/extracted_frames
```

## Evaluation

### Compute Three Metrics from Raw Results of Previous Running

```bash
python evaluation_from_raw_3metrics.py \
  --raw-results-dir /path/to/output_dir/raw_results \
  --max-abs-error 3.0
```


### Evaluate Safety Filter on EgoSafety Validation Set

```bash
# Fine-tuned VLESA safety filter
python eval_safety_filter_on_valset_ours.py \
  --dataset-dir /path/to/egosafety_single_current_dataset \
  --output-dir ./safety_filter_val_eval_ours

# Prompt-based baseline safety filter
python eval_safety_filter_on_valset.py \
  --dataset-dir /path/to/egosafety_single_current_dataset \
  --output-dir ./safety_filter_val_eval_baseline
```

## Visualization

```bash
# Single Pareto front
python plot_pareto.py -i output_dir/evaluation_1_intervention_accuracy.json

# Compare VLESA vs. baseline
python plot_pareto.py \
  -i vlesa_output/evaluation_1_intervention_accuracy.json \
  -i baseline_output/evaluation_1_intervention_accuracy.json \
  -l "VLESA" -l "Baseline" \
  -o pareto_comparison.png \
  --annotate

# Generate paper comparison figure (VLESA vs. frontier VLMs)
python vlesa_plot.py --output fig_intervention.pdf
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{hu2026vlesa,
  title={VLESA: Vision-Language Embodied Safety Agent for Human Activity Monitoring},
  author={Hu, Hanjiang and Pan, Yiyuan and Li, Jiaxing and Luo, Xusheng and Robey, Alexander and Li, Na and Wang, Yebin and Liu, Changliu},
  journal={arXiv preprint arXiv:2606.03954},
  year={2026}
}
```
