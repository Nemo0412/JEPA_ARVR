#!/usr/bin/env python3
"""
Evaluate the fine-tuned Qwen3-VL safety filter on the EgoSafety validation set.

Outputs per_sample_predictions.jsonl, summary.json, and report.txt to --output-dir.
Re-running with the same --output-dir resumes from where it left off.

Usage:
  python eval_safety_filter_on_valset_ours.py \
      --dataset-dir /path/to/egosafety_single_current_dataset \
      --safety-model /path/to/merged_checkpoint \
      --output-dir  ./safety_filter_val_eval_ours
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Optional

from PIL import Image
from tqdm import tqdm

# Make sure we can import from the same directory as vla_asimov_baseline.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Reuse the production safety filter so we measure exactly what we deploy
from vla_asimov_llamaapi import SafetyQFilter, SafetyFilterConfig  # noqa: E402

try:
    from datasets import load_from_disk
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "datasets",
                    "--break-system-packages", "-q"])
    from datasets import load_from_disk


# ============================================================================
# Helpers
# ============================================================================

def pick_representative_image(sample: Dict[str, Any]) -> Optional[Image.Image]:
    """
    Match the dataset script's main_image rule: prefer pnr, fall back to pre,
    then post. Each *_images_* column is a list (possibly empty).
    """
    for key in ("images_pnr", "images_pre", "images_post"):
        imgs = sample.get(key) or []
        if len(imgs) > 0 and imgs[0] is not None:
            return imgs[0]
    return None


def normalize_label(label: str) -> str:
    """Normalize labels for comparison; returns 'safe', 'unsafe', or 'unknown'."""
    if label is None:
        return "unknown"
    s = str(label).strip().lower()
    if s == "safe":
        return "safe"
    if s == "unsafe":
        return "unsafe"
    return "unknown"


def load_completed_indices(jsonl_path: str) -> set:
    """Return the set of sample indices already present in the JSONL file."""
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                idx = rec.get("sample_idx")
                if idx is not None:
                    done.add(int(idx))
            except json.JSONDecodeError:
                pass
    return done


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(records):
    """
    Compute classification metrics treating 'Unsafe' as the positive class.

    A record is the dict written per-sample to the JSONL file.
    """
    n_total = 0
    n_correct = 0
    n_unknown_pred = 0     # filter returned "Unknown"
    n_unknown_label = 0    # GT label wasn't safe/unsafe (shouldn't happen)
    n_failed = 0           # the filter call itself failed (no prediction)

    # Confusion matrix counts (positive class = Unsafe)
    tp = fp = tn = fn = 0

    # Per-class counts
    n_gt_safe = 0
    n_gt_unsafe = 0

    for r in records:
        if not r.get("ok", True):
            n_failed += 1
            continue

        gt = r["gt_label_normalized"]
        pred = r["pred_label_normalized"]

        if gt == "unknown":
            n_unknown_label += 1
            continue

        n_total += 1
        if gt == "safe":
            n_gt_safe += 1
        else:
            n_gt_unsafe += 1

        if pred == "unknown":
            n_unknown_pred += 1
            # An unknown prediction is treated as "no decision" — counts toward
            # accuracy denominator but never matches either class.
            continue

        if pred == gt:
            n_correct += 1

        if gt == "unsafe" and pred == "unsafe":
            tp += 1
        elif gt == "safe" and pred == "unsafe":
            fp += 1
        elif gt == "safe" and pred == "safe":
            tn += 1
        elif gt == "unsafe" and pred == "safe":
            fn += 1

    accuracy = n_correct / n_total if n_total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    safe_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    unsafe_recall = recall  # by definition

    return {
        "n_total_evaluated": n_total,
        "n_correct": n_correct,
        "n_failed_filter_call": n_failed,
        "n_unknown_predictions": n_unknown_pred,
        "n_unknown_gt_label": n_unknown_label,
        "n_gt_safe": n_gt_safe,
        "n_gt_unsafe": n_gt_unsafe,
        "accuracy": accuracy,
        "confusion_matrix": {
            "tp_unsafe_correct": tp,
            "fp_safe_called_unsafe": fp,
            "tn_safe_correct": tn,
            "fn_unsafe_called_safe": fn,
        },
        "unsafe_as_positive": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_class_recall": {
            "safe_recall_specificity": safe_recall,
            "unsafe_recall_sensitivity": unsafe_recall,
        },
    }


# ============================================================================
# Main eval loop
# ============================================================================

def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "per_sample_predictions.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")
    report_path = os.path.join(args.output_dir, "report.txt")

    print(f"Loading dataset from: {args.dataset_dir}")
    ds_dict = load_from_disk(args.dataset_dir)
    if args.split not in ds_dict:
        print(f"ERROR: split '{args.split}' not in dataset. "
              f"Available: {list(ds_dict.keys())}")
        sys.exit(1)
    ds = ds_dict[args.split]
    print(f"Loaded split '{args.split}' with {len(ds)} samples")

    # Determine sample range
    start_idx = args.start_idx
    end_idx = len(ds) if args.max_samples is None else min(len(ds), start_idx + args.max_samples)
    print(f"Will process samples [{start_idx}, {end_idx})")

    # Resume support
    completed = set()
    if not args.no_resume and os.path.exists(jsonl_path):
        completed = load_completed_indices(jsonl_path)
        print(f"Resume: {len(completed)} samples already in {jsonl_path}, will skip them")
    elif args.no_resume and os.path.exists(jsonl_path):
        # Wipe the file when not resuming
        os.remove(jsonl_path)
        print(f"--no-resume: cleared existing {jsonl_path}")

    # Initialize safety filter
    safety_config = SafetyFilterConfig(
        model_path=args.safety_model,
        safety_threshold=args.safety_threshold,
        constrained_decoding_alpha=args.safety_alpha
    )
    safety_filter = SafetyQFilter(safety_config)

    # Open JSONL in append mode so resume works
    out_f = open(jsonl_path, "a", buffering=1)  # line-buffered

    pbar = tqdm(range(start_idx, end_idx), desc=f"Eval {args.split}")
    for i in pbar:
        if i in completed:
            continue

        sample = ds[i]
        gt_label = sample.get("answer", "")
        gt_norm = normalize_label(gt_label)
        task_summary = sample.get("task_summary", "") or ""
        action_sentence = sample.get("action_sentence", "") or ""
        image = pick_representative_image(sample)

        record = {
            "sample_idx": i,
            "gt_label": gt_label,
            "gt_label_normalized": gt_norm,
            "task_summary": task_summary,
            "action_sentence": action_sentence,
            "has_image": image is not None,
        }

        if image is None:
            record["ok"] = False
            record["error"] = "no_image_in_sample"
            record["pred_label"] = None
            record["pred_label_normalized"] = "unknown"
            record["safety_score"] = None
            out_f.write(json.dumps(record) + "\n")
            continue

        if not action_sentence or not task_summary:
            record["ok"] = False
            record["error"] = "missing_action_or_task_summary"
            record["pred_label"] = None
            record["pred_label_normalized"] = "unknown"
            record["safety_score"] = None
            out_f.write(json.dumps(record) + "\n")
            continue

        # Run the safety filter
        try:
            result = safety_filter.evaluate_safety(
                image_input=image,
                task_summary=task_summary,
                action_sentence=action_sentence,
                loaded_image=True,
            )
            pred_label = result.get("classification", "Unknown")
            record.update({
                "ok": True,
                "pred_label": pred_label,
                "pred_label_normalized": normalize_label(pred_label),
                "safety_score": result.get("safety_score"),
                "safety_reasoning": result.get("reasoning", ""),
                "raw_response": result.get("raw_response", ""),
            })
        except Exception as e:
            record.update({
                "ok": False,
                "error": f"filter_exception: {type(e).__name__}: {e}",
                "pred_label": None,
                "pred_label_normalized": "unknown",
                "safety_score": None,
            })

        out_f.write(json.dumps(record) + "\n")

        # Live accuracy update on the progress bar
        if record.get("ok") and record["gt_label_normalized"] in ("safe", "unsafe"):
            correct_flag = record["pred_label_normalized"] == record["gt_label_normalized"]
            pbar.set_postfix(last_correct=int(correct_flag))

        if args.sleep > 0:
            time.sleep(args.sleep)

    out_f.close()

    # ----- Aggregate -----
    print("\nLoading all per-sample records for metric computation...")
    all_records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    metrics = compute_metrics(all_records)

    summary = {
        "config": {
            "dataset_dir": args.dataset_dir,
            "split": args.split,
            "safety_vlm_model": args.safety_vlm_model,
            "safety_threshold": args.safety_threshold,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "start_idx": args.start_idx,
            "max_samples": args.max_samples,
        },
        "metrics": metrics,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    # Human-readable report
    cm = metrics["confusion_matrix"]
    pos = metrics["unsafe_as_positive"]
    pcr = metrics["per_class_recall"]
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("SAFETY FILTER EVALUATION ON VALIDATION SPLIT\n")
        f.write("=" * 60 + "\n\n")
        f.write("CONFIG\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset:           {args.dataset_dir}\n")
        f.write(f"Split:             {args.split}\n")
        f.write(f"Safety VLM (API):  {args.safety_vlm_model}\n")
        f.write(f"Safety threshold:  {args.safety_threshold}\n")
        f.write(f"Temperature:       {args.temperature}\n\n")

        f.write("OVERALL\n")
        f.write("-" * 40 + "\n")
        f.write(f"Evaluated:               {metrics['n_total_evaluated']}\n")
        f.write(f"Correct:                 {metrics['n_correct']}\n")
        f.write(f"Accuracy:                {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Filter-call failures:    {metrics['n_failed_filter_call']}\n")
        f.write(f"Unknown predictions:     {metrics['n_unknown_predictions']}\n")
        f.write(f"Unknown GT labels:       {metrics['n_unknown_gt_label']}\n\n")

        f.write("CLASS BALANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"GT Safe:    {metrics['n_gt_safe']}\n")
        f.write(f"GT Unsafe:  {metrics['n_gt_unsafe']}\n\n")

        f.write("CONFUSION MATRIX (positive class = Unsafe)\n")
        f.write("-" * 40 + "\n")
        f.write(f"                 Pred Unsafe   Pred Safe\n")
        f.write(f"  GT Unsafe      {cm['tp_unsafe_correct']:>11}   {cm['fn_unsafe_called_safe']:>9}\n")
        f.write(f"  GT Safe        {cm['fp_safe_called_unsafe']:>11}   {cm['tn_safe_correct']:>9}\n\n")

        f.write("METRICS (Unsafe = positive)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision:                  {pos['precision']*100:.2f}%\n")
        f.write(f"Recall (sensitivity):       {pos['recall']*100:.2f}%\n")
        f.write(f"F1:                         {pos['f1']*100:.2f}%\n")
        f.write(f"Safe-recall (specificity):  {pcr['safe_recall_specificity']*100:.2f}%\n")
    print(f"Saved report to:  {report_path}")

    # Console echo of the headline numbers
    print("\n" + "=" * 60)
    print(f"Accuracy:          {metrics['accuracy']*100:.2f}%  "
          f"({metrics['n_correct']}/{metrics['n_total_evaluated']})")
    print(f"Unsafe precision:  {pos['precision']*100:.2f}%")
    print(f"Unsafe recall:     {pos['recall']*100:.2f}%")
    print(f"Unsafe F1:         {pos['f1']*100:.2f}%")
    print(f"Safe recall:       {pcr['safe_recall_specificity']*100:.2f}%")
    print("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate prompt-based SafetyQFilter on a HF dataset split"
    )
    # Data
    parser.add_argument("--dataset-dir", required=True,
                        help="Path passed to load_from_disk() (the directory "
                             "produced by Dataset.save_to_disk in your script)")
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--output-dir", default="safety_filter_val_eval")

    # Filter (forwarded to SafetyFilterConfig)
    parser.add_argument("--api-key",
                        default=os.environ.get("LLAMA_API_KEY", ""))
    parser.add_argument("--safety-vlm-model",
                        default="Llama-4-Scout-17B-16E-Instruct-FP8")
    # parser.add_argument("--safety-threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--safety-model", 
                        default="/path/to/safety_filter_checkpoint")
    parser.add_argument("--safety-threshold", type=float, default=0.5)
    parser.add_argument("--safety-alpha", type=float, default=2.0,
                        help="Weight for safety score in constrained decoding")
    

    # Sampling / pacing
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="If set, evaluate only this many samples (after start-idx)")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to sleep between API calls (rate-limit cushion)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore (and overwrite) existing per_sample_predictions.jsonl")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
    
    # python eval_safety_filter_on_valset_ours.py --dataset-dir /path/to/egosafety_single_current_dataset --output-dir ./safety_filter_val_eval_ours