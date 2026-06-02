"""
V-JEPA 2 Action Anticipation on EK100 — P01_11
Metrics: Verb Top-3, Noun Top-3, Action Recall@5 (class-mean)
"""

import sys
sys.path.insert(0, "/home/ll5914/ARVR_Video/vjepa2")

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from decord import VideoReader, cpu

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from evals.action_anticipation_frozen.models import AttentiveClassifier
from src.models.vision_transformer import vit_large_rope

ENCODER_CKPT = "/scratch/ll5914/models/vjepa2/vitl.pt"
PROBE_CKPT   = "/scratch/ll5914/models/vjepa2/ek100-vitl-256.pt"
VIDEO_PATH   = "/scratch/ll5914/datasets/EPIC-KITCHENS/EPIC-KITCHENS/P01/videos/P01_11.MP4"
VAL_CSV      = "/scratch/ll5914/datasets/EPIC-KITCHENS/annotations/EPIC_100_validation.csv"
TRAIN_CSV    = "/scratch/ll5914/datasets/EPIC-KITCHENS/annotations/EPIC_100_train.csv"

IMAGENET_MEAN    = (0.485, 0.456, 0.406)
IMAGENET_STD     = (0.229, 0.224, 0.225)
IMG_SIZE         = 256
FRAMES_PER_CLIP  = 32
FPS              = 8
ANTICIPATION_SEC = 1.0


def build_transform():
    short_side = int(256.0 / 224 * IMG_SIZE)
    return video_transforms.Compose([
        video_transforms.Resize(short_side, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_class_maps():
    """Same logic as official epickitchens.py to ensure action_id matches checkpoint."""
    tdf = pd.read_csv(TRAIN_CSV)
    tactions = set(zip(tdf["verb_class"].values, tdf["noun_class"].values))
    tverbs   = set(v for v, _ in tactions)
    tnouns   = set(n for _, n in tactions)

    # mapped id → display name
    verb_map = {i: tdf[tdf["verb_class"]==k]["verb"].iloc[0] for i, k in enumerate(tverbs)}
    noun_map = {i: tdf[tdf["noun_class"]==k]["noun"].iloc[0] for i, k in enumerate(tnouns)}
    action_map = {f"{v}_{n}": f"{verb_map[i]}:{noun_map[j]}"
                  for (v, n) in tactions
                  for i in [list(tverbs).index(v)]
                  for j in [list(tnouns).index(n)]}

    # orig class_id → mapped id
    verb_orig2mapped   = {k: i for i, k in enumerate(tverbs)}
    noun_orig2mapped   = {k: i for i, k in enumerate(tnouns)}
    action_orig2mapped = {k: i for i, k in enumerate(tactions)}  # (v,n) → action_id

    return verb_map, noun_map, verb_orig2mapped, noun_orig2mapped, action_orig2mapped, action_map


def load_encoder(device):
    print("  Loading ViT-L encoder...")
    model = vit_large_rope(
        img_size=(IMG_SIZE, IMG_SIZE), num_frames=FRAMES_PER_CLIP,
        tubelet_size=2, patch_size=16, uniform_power=True,
    )
    ckpt  = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=True)
    state = ckpt.get("target_encoder", ckpt.get("encoder", ckpt))
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    msg   = model.load_state_dict(state, strict=False)
    print(f"  missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    return model.to(device).eval()


def load_probe(verb_map, noun_map, action_map, embed_dim, device):
    print(f"  Loading EK100 probe (verbs={len(verb_map)}, nouns={len(noun_map)}, actions={len(action_map)})...")
    classifier = AttentiveClassifier(
        verb_classes=verb_map, noun_classes=noun_map, action_classes=action_map,
        embed_dim=embed_dim, num_heads=16, depth=4, use_activation_checkpointing=False,
    )
    ckpt  = torch.load(PROBE_CKPT, map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in ckpt["classifiers"][0].items()}
    msg   = classifier.load_state_dict(state, strict=False)
    print(f"  missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    return classifier.to(device).eval()


def sample_clip(vr, end_sec, vfps):
    frame_step  = max(1, int(vfps / FPS))
    end_frame   = int(end_sec * vfps)
    start_frame = end_frame - FRAMES_PER_CLIP * frame_step
    indices = np.arange(start_frame, end_frame, frame_step, dtype=np.int64)
    indices = np.clip(indices, 0, len(vr) - 1)
    return vr.get_batch(indices).asnumpy()


def topk_hit(gt_id, logits, k):
    return int(gt_id) in logits.topk(k).indices[0].tolist()


def class_mean_recall(per_class_correct, per_class_total):
    recalls = [per_class_correct.get(c, 0) / t for c, t in per_class_total.items()]
    return np.mean(recalls) * 100 if recalls else 0.0


def run():
    print("=" * 65)
    print("V-JEPA 2 — EK100 Action Anticipation on P01_11")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    print("\n[1] Loading annotations...")
    verb_map, noun_map, verb_o2m, noun_o2m, action_o2m, action_map = build_class_maps()
    df_val    = pd.read_csv(VAL_CSV)
    df_p01_11 = df_val[df_val['video_id'] == 'P01_11'].sort_values('start_frame').reset_index(drop=True)
    print(f"  P01_11: {len(df_p01_11)} annotated actions")

    print("\n[2] Loading models...")
    encoder    = load_encoder(device)
    classifier = load_probe(verb_map, noun_map, action_map, encoder.embed_dim, device)

    print("\n[3] Loading video...")
    vr       = VideoReader(VIDEO_PATH, num_threads=1, ctx=cpu(0))
    vfps     = vr.get_avg_fps()
    duration = len(vr) / vfps
    print(f"  {len(vr)} frames, {vfps:.1f} FPS, duration {duration:.1f}s ({duration/60:.1f}min)")

    transform = build_transform()

    verb_correct   = {1: 0, 3: 0, 5: 0}
    noun_correct   = {1: 0, 3: 0, 5: 0}
    action_correct = {1: 0, 3: 0, 5: 0}
    verb_cls_c5 = defaultdict(int); verb_cls_t = defaultdict(int)
    noun_cls_c5 = defaultdict(int); noun_cls_t = defaultdict(int)
    act_cls_c5  = defaultdict(int); act_cls_t  = defaultdict(int)
    total = 0

    print("\n[4] Running inference per action...")
    for i, (_, row) in enumerate(df_p01_11.iterrows()):
        start_sec   = int(row['start_frame']) / vfps
        obs_end_sec = start_sec - ANTICIPATION_SEC
        if obs_end_sec < 2.0:
            continue

        orig_v = int(row['verb_class'])
        orig_n = int(row['noun_class'])
        v_id = verb_o2m.get(orig_v, -1)
        n_id = noun_o2m.get(orig_n, -1)
        a_id = action_o2m.get((orig_v, orig_n), -1)
        if v_id == -1 or n_id == -1:
            continue

        frames = sample_clip(vr, obs_end_sec, vfps)
        x = transform(torch.from_numpy(frames).permute(0, 3, 1, 2)).unsqueeze(0).to(device)

        with torch.no_grad():
            feats  = encoder(x)
            output = classifier(feats)

        for k in [1, 3, 5]:
            if topk_hit(v_id, output["verb"],   k): verb_correct[k]   += 1
            if topk_hit(n_id, output["noun"],   k): noun_correct[k]   += 1
            if a_id != -1 and topk_hit(a_id, output["action"], k): action_correct[k] += 1

        verb_cls_t[v_id] += 1
        noun_cls_t[n_id] += 1
        if topk_hit(v_id, output["verb"], 5):   verb_cls_c5[v_id] += 1
        if topk_hit(n_id, output["noun"], 5):   noun_cls_c5[n_id] += 1
        if a_id != -1:
            act_cls_t[a_id] += 1
            if topk_hit(a_id, output["action"], 5): act_cls_c5[a_id] += 1

        total += 1
        if (i + 1) % 20 == 0:
            print(f"  Processed {total} actions...")

    v_cmr5 = class_mean_recall(verb_cls_c5, verb_cls_t)
    n_cmr5 = class_mean_recall(noun_cls_c5, noun_cls_t)
    a_cmr5 = class_mean_recall(act_cls_c5,  act_cls_t)
    n_act  = sum(act_cls_t.values())

    print(f"\n{'=' * 65}")
    print(f"EK100 P01_11  |  {total} actions  |  anticipation {ANTICIPATION_SEC}s")
    print(f"{'=' * 65}")
    print(f"{'Metric':<30} {'Verb':>8} {'Noun':>8} {'Action':>8}")
    print(f"{'-' * 58}")
    for k in [1, 3, 5]:
        vp = 100 * verb_correct[k]   / total
        np_ = 100 * noun_correct[k]  / total
        ap = 100 * action_correct[k] / n_act if n_act else 0
        print(f"  Top-{k} Accuracy        {vp:>7.1f}% {np_:>7.1f}% {ap:>7.1f}%")
    print(f"  Class-Mean Recall@5  {v_cmr5:>7.1f}% {n_cmr5:>7.1f}% {a_cmr5:>7.1f}%")
    print(f"{'=' * 65}")
    print("\nNote: single video P01_11 only, not full val set results.")


if __name__ == "__main__":
    run()
