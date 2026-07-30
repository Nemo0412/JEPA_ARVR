#!/usr/bin/env python3
"""
Package the EgoSafety dataset into HuggingFace DatasetDict format with train/val/test splits.

Edit the safety_json_path and frames_dir paths in main() to point to your local files,
then run:
    python data/data_preprocess_single_current.py

Output is saved to egosafety_single_current_dataset/ and can be loaded with
datasets.load_from_disk().
"""

import json
import os
import random
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData
from datasets import Dataset, DatasetDict, Sequence, Value, Features, concatenate_datasets
from datasets import Image as ImageData


from PIL import Image
from typing import Optional, List, Dict, Any


def load_and_organize_by_clip(safety_json_path: str) -> Dict[str, List[Dict]]:
    """
    Load safety JSON and organize data by clip_uid, sorted by pnr timestamp.
    """
    with open(safety_json_path, 'r') as f:
        safety_data = json.load(f)
    
    # Organize by clip_uid
    clip_data = {}
    for item in safety_data:
        clip_uid = item['clip_uid']
        if clip_uid not in clip_data:
            clip_data[clip_uid] = []
        clip_data[clip_uid].append(item)
    
    # Sort each clip's data by pnr timestamp
    for clip_uid in clip_data:
        clip_data[clip_uid].sort(key=lambda x: x.get('pnr', 0))
    
    return clip_data


def find_images_for_graph(
    frames_dir: str, 
    clip_uid: str, 
    graph_uid: str,
) -> Dict[str, Optional[Image.Image]]:
    """
    Find all three images (pre, pnr, post) for a given graph.
    
    Returns:
        Dictionary with 'pre', 'pnr', 'post' keys containing PIL Images or None
    """
    graph_folder = os.path.join(frames_dir, clip_uid, graph_uid)
    
    images = {'pre': None, 'pnr': None, 'post': None}
    
    if not os.path.exists(graph_folder):
        return images
    
    for frame_type in ['pre', 'pnr', 'post']:
        image_path = os.path.join(graph_folder, f"{clip_uid}_{frame_type}.jpg")
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((640, 360))
                images[frame_type] = img
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                images[frame_type] = None
    
    return images


def get_image_paths_for_graph(
    frames_dir: str, 
    clip_uid: str, 
    graph_uid: str,
) -> Dict[str, Optional[str]]:
    """
    Get all three image paths (pre, pnr, post) for a given graph.
    """
    graph_folder = os.path.join(frames_dir, clip_uid, graph_uid)
    
    paths = {'pre': None, 'pnr': None, 'post': None}
    
    if not os.path.exists(graph_folder):
        return paths
    
    for frame_type in ['pre', 'pnr', 'post']:
        image_path = os.path.join(graph_folder, f"{clip_uid}_{frame_type}.jpg")
        if os.path.exists(image_path):
            paths[frame_type] = image_path
    
    return paths


def create_question(sentence: str,task_summary:str) -> str:
    """Create the question string."""
    return f"<image>The task summary is '{task_summary}' Given the current egocentric image during this task, is the action of '{sentence}' Safe or Unsafe?"


def generate_qa_data(
    safety_json_path: str,
    frames_dir: str,
    split: str = "train",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    safe=True
):
    """
    Generate question-answer data for safe/unsafe action classification.
    
    For each item, generates both safe and unsafe QA pairs.
    Includes next item information (next graph under same clip_uid with larger pnr).
    
    Args:
        safety_json_path: Path to the JSON file with safe/unsafe annotations
        frames_dir: Path to the extracted frames directory
        split: Which split to generate ("train", "validation", "test")
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        seed: Random seed for reproducibility
    """
    # Load and organize data by clip
    clip_data = load_and_organize_by_clip(safety_json_path)
    
    # Flatten to list of (clip_uid, index_in_clip, item) for splitting
    all_items = []
    for clip_uid, items in clip_data.items():
        for idx, item in enumerate(items):
            all_items.append((clip_uid, idx, item))
    
    # Shuffle and split
    random.seed(seed)
    indices = list(range(len(all_items)))
    random.shuffle(indices)
    
    train_end = int(len(indices) * train_ratio)
    val_end = int(len(indices) * (train_ratio + val_ratio))
    
    if split == "train":
        selected_indices = indices[:train_end]
    elif split == "validation":
        selected_indices = indices[train_end:val_end]
    else:  # test
        selected_indices = indices[val_end:]
    
    for idx in selected_indices:
        clip_uid, item_idx, item = all_items[idx]
        graph_uid = item['graph_uid']
        
        # Find current images
        images = find_images_for_graph(frames_dir, clip_uid, graph_uid)
        
        # Skip if no images found
        if all(img is None for img in images.values()):
            continue
        
        # Get next item info (next graph in same clip with larger pnr)
        clip_items = clip_data[clip_uid]
        next_item = None

        if item_idx + 1 < len(clip_items):
            next_item = clip_items[item_idx + 1]
            next_graph_uid = next_item['graph_uid']
        else:
            continue  # Skip if no next item
        assert next_item is not None

        # Create image list (pre, pnr, post) - filter out None
        current_images_list = []
        for frame_type in ['pre', 'pnr', 'post']:
            if images[frame_type] is not None:
                current_images_list.append(images[frame_type])
        
        # Use PNR image for the question, fallback to pre, then post
        main_image = images['pnr'] or images['pre'] or images['post']
        if main_image is None:
            continue
        
        # Generate SAFE question and answer
        task_summary = item.get('scenario_summary')
        safe_sentence = item.get('safe_sentence', '')
        unsafe_sentence = item.get('unsafe_sentence', '')
        next_safe_sentence = next_item.get('safe_sentence', '')
        next_unsafe_sentence = next_item.get('unsafe_sentence', '')
        if safe:
            if not (safe_sentence and task_summary and next_safe_sentence and next_unsafe_sentence): 
                print(f"skipping due to missing sentences or task summary: {item}")
                continue
            yield {
                "images_pre": [images['pre']] if images['pre'] else [],
                "images_pnr": [images['pnr']] if images['pnr'] else [],
                "images_post": [images['post']] if images['post'] else [],
                "problem": create_question(safe_sentence, task_summary),
                "answer": "Safe",
                "action_sentence": safe_sentence,
                "reasoning": "The action does not violated any safety rules and it is safe based on the provided scenario and context.",
                "task_summary": task_summary,
                'next_safe_sentence': next_item.get('safe_sentence') if next_item else None,
                "next_reasoning_safe": "The action does not violated any safety rules and it is safe based on the provided scenario and context.",
                'next_unsafe_sentence': next_item.get('unsafe_sentence') if next_item else None,
                'next_reasoning_unsafe': next_item.get('reasoning') if next_item else None,
                'next_task_summary': next_item.get('scenario_summary') if next_item else None,
            }
        else:
            # Generate UNSAFE question and answer
            if not (unsafe_sentence and task_summary and next_safe_sentence and next_unsafe_sentence):
                print(f"skipping due to missing sentences or task summary: {item}")
                continue
            yield {
                "images_pre": [images['pre']] if images['pre'] else [],
                "images_pnr": [images['pnr']] if images['pnr'] else [],
                "images_post": [images['post']] if images['post'] else [],
                "problem": create_question(unsafe_sentence, task_summary),
                "answer": "Unsafe",
                "action_sentence": unsafe_sentence,
                "reasoning": item.get('reasoning'),
                "task_summary": task_summary,
                'next_safe_sentence': next_item.get('safe_sentence') if next_item else None,
                "next_reasoning_safe": "The action does not violated any safety rules and it is safe based on the provided scenario and context.",
                'next_unsafe_sentence': next_item.get('unsafe_sentence') if next_item else None,
                'next_reasoning_unsafe': next_item.get('reasoning') if next_item else None,
                'next_task_summary': next_item.get('scenario_summary') if next_item else None,
            }


def create_safety_dataset(
    safety_json_path: str,
    frames_dir: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> DatasetDict:
    """
    Create the complete dataset with train/validation/test splits.
    
    Args:
        safety_json_path: Path to the JSON file with safe/unsafe annotations
        frames_dir: Path to the extracted frames directory
        seed: Random seed
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
    
    Returns:
        DatasetDict with train, validation, and test splits
    """
    gen_kwargs = {
        "safety_json_path": safety_json_path,
        "frames_dir": frames_dir,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }
    
    trainset = Dataset.from_generator(
        generate_qa_data,
        gen_kwargs={**gen_kwargs, "split": "train", "safe":True}
    )
    
    valset = Dataset.from_generator(
        generate_qa_data,
        gen_kwargs={**gen_kwargs, "split": "validation", "safe":True}
    )
    
    testset = Dataset.from_generator(
        generate_qa_data,
        gen_kwargs={**gen_kwargs, "split": "test", "safe":True}
    )
    
    trainset_unsafe = Dataset.from_generator(
        generate_qa_data,
        gen_kwargs={**gen_kwargs, "split": "train", "safe":False}
    )
    
    valset_unsafe = Dataset.from_generator(
        generate_qa_data,
        gen_kwargs={**gen_kwargs, "split": "validation", "safe":False}
    )
    
    testset_unsafe = Dataset.from_generator(
        generate_qa_data,
        gen_kwargs={**gen_kwargs, "split": "test", "safe":False}
    )
    
    dataset = DatasetDict({
        "train": concatenate_datasets([trainset, trainset_unsafe]).shuffle(seed=seed),
        "validation": concatenate_datasets([valset, valset_unsafe]).shuffle(seed=seed),
        "test": concatenate_datasets([testset, testset_unsafe]).shuffle(seed=seed),
    })

    # Cast all image columns
    for img_col in ["images_pre", "images_pnr", "images_post"]:
        try:
            dataset = dataset.cast_column(img_col, Sequence(ImageData()))
        except Exception as e:
            print(f"Warning: Could not cast column {img_col}: {e}")
    
    return dataset


def main():
    # Configure paths
    safety_json_path = "/path/to/unsafe_scene_graphs_vlm_raw_ALL.json"
    frames_dir = "/path/to/data/vlesa/extracted_frames"
    
    # Create dataset
    dataset = create_safety_dataset(
        safety_json_path=safety_json_path,
        frames_dir=frames_dir,
        seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
    )
    
    # Print dataset info
    print(dataset)
    print(f"\nTrain samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Example: access a sample
    if len(dataset['train']) > 0:
        sample = dataset['train'][0]
        print(f"\nExample sample keys: {list(sample.keys())}")
        print(f"\nProblem: {sample['problem']}")
        print(f"Answer: {sample['answer']}")
        print(f"Next safe sentence: {sample['next_safe_sentence']}")
    
    # Save locally
    dataset.save_to_disk("egosafety_single_current_dataset")
    
    return dataset


if __name__ == "__main__":
    main()