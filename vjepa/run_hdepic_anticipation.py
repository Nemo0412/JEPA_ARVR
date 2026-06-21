"""
V-JEPA 2 Action Anticipation on HD-EPIC — P01 first scene
Metrics: Verb Top-3, Noun Top-3, Action Recall@5 (class-mean)
Scene: P01-20240203-093333 (~3 min, morning coffee preparation)
"""

import sys, pickle
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

ENCODER_CKPT    = "/scratch/ll5914/models/vjepa2/vitl.pt"
PROBE_CKPT      = "/scratch/ll5914/models/vjepa2/ek100-vitl-256.pt"
EK100_TRAIN_CSV = "/scratch/ll5914/datasets/EPIC-KITCHENS/annotations/EPIC_100_train.csv"
HD_EPIC_NARR    = "/scratch/ll5914/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
VIDEO_PATH      = "/scratch/ll5914/datasets/HD-EPIC/HD-EPIC/Videos/P01/P01-20240203-093333.mp4"
VIDEO_ID        = "P01-20240203-093333"

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


def build_ek100_maps():
    """Same logic as official epickitchens.py to ensure action_id matches checkpoint."""
    tdf = pd.read_csv(EK100_TRAIN_CSV)
    tactions = set(zip(tdf["verb_class"].values, tdf["noun_class"].values))
    tverbs   = set(v for v, _ in tactions)
    tnouns   = set(n for _, n in tactions)

    verb_map = {i: tdf[tdf["verb_class"]==k]["verb"].iloc[0] for i, k in enumerate(tverbs)}
    noun_map = {i: tdf[tdf["noun_class"]==k]["noun"].iloc[0] for i, k in enumerate(tnouns)}
    verb_name2id = {v: k for k, v in verb_map.items()}
    noun_name2id = {v: k for k, v in noun_map.items()}

    action_orig2mapped = {k: i for i, k in enumerate(tactions)}
    action_map = {f"{v}_{n}": f"act{i}" for (v, n), i in action_orig2mapped.items()}

    return verb_map, noun_map, verb_name2id, noun_name2id, action_orig2mapped, action_map


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
    print(f"  missing={len(msg.missing_keys)}")
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


def topk_hit_any(gt_words, logits, id2name, k):
    topk_names = {id2name.get(i, "") for i in logits.topk(k).indices[0].tolist()}
    return any(w in topk_names for w in gt_words)


def class_mean_recall(per_class_correct, per_class_total):
    recalls = [per_class_correct.get(c, 0) / t for c, t in per_class_total.items()]
    return np.mean(recalls) * 100 if recalls else 0.0


def run():
    print("=" * 70)
    print("V-JEPA 2 — HD-EPIC Action Anticipation")
    print(f"Scene: {VIDEO_ID}  (morning coffee preparation, ~3 min)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    print("\n[1] Loading annotations...")
    verb_map, noun_map, verb_n2id, noun_n2id, action_o2m, action_map = build_ek100_maps()
    with open(HD_EPIC_NARR, "rb") as f:
        narr_df = pickle.load(f)
    scene_df = narr_df[narr_df['video_id'] == VIDEO_ID].sort_values('start_timestamp').reset_index(drop=True)
    print(f"  {VIDEO_ID} — {len(scene_df)} annotated actions")

    print("\n[2] Loading models...")
    encoder    = load_encoder(device)
    classifier = load_probe(verb_map, noun_map, action_map, encoder.embed_dim, device)

    print("\n[3] Loading video...")
    vr       = VideoReader(VIDEO_PATH, num_threads=1, ctx=cpu(0))
    vfps     = vr.get_avg_fps()
    duration = len(vr) / vfps
    print(f"  {len(vr)} frames, {vfps:.1f} FPS, duration {duration:.1f}s ({duration/60:.1f}min)")

    transform = build_transform()

    # EK100 verb_orig_id → mapped id (for action lookup)
    tdf = pd.read_csv(EK100_TRAIN_CSV)
    tactions  = set(zip(tdf["verb_class"].values, tdf["noun_class"].values))
    tverbs    = set(v for v, _ in tactions)
    tnouns    = set(n for _, n in tactions)
    verb_o2m  = {k: i for i, k in enumerate(tverbs)}
    noun_o2m  = {k: i for i, k in enumerate(tnouns)}

    verb_correct   = {1: 0, 3: 0, 5: 0}
    noun_correct   = {1: 0, 3: 0, 5: 0}
    # Action Recall@5: match first verb/noun to EK100 action_id
    act_cls_c5 = defaultdict(int); act_cls_t = defaultdict(int)
    verb_cls_c5 = defaultdict(int); verb_cls_t = defaultdict(int)
    noun_cls_c5 = defaultdict(int); noun_cls_t = defaultdict(int)
    total = 0

    print("\n[4] Running inference per action...")
    for _, row in scene_df.iterrows():
        start_sec = float(row['start_timestamp'])
        obs_end   = start_sec - ANTICIPATION_SEC
        if obs_end < 2.0:
            continue

        gt_verbs = row['verbs'] if isinstance(row['verbs'], list) else [str(row['verbs'])]
        gt_nouns = row['nouns'] if isinstance(row['nouns'], list) else [str(row['nouns'])]

        frames = sample_clip(vr, obs_end, vfps)
        x = transform(torch.from_numpy(frames).permute(0, 3, 1, 2)).unsqueeze(0).to(device)

        with torch.no_grad():
            feats  = encoder(x)
            output = classifier(feats)

        for k in [1, 3, 5]:
            if topk_hit_any(gt_verbs, output["verb"], verb_map, k): verb_correct[k] += 1
            if topk_hit_any(gt_nouns, output["noun"], noun_map, k): noun_correct[k] += 1

        # Class-mean Recall@5 — using GT words that can be found in EK100 vocabulary
        v_id = verb_n2id.get(gt_verbs[0], None)
        n_id = noun_n2id.get(gt_nouns[0], None)
        if v_id is not None:
            verb_cls_t[v_id] += 1
            if topk_hit_any(gt_verbs, output["verb"], verb_map, 5): verb_cls_c5[v_id] += 1
        if n_id is not None:
            noun_cls_t[n_id] += 1
            if topk_hit_any(gt_nouns, output["noun"], noun_map, 5): noun_cls_c5[n_id] += 1

        # Action Recall@5: look up corresponding action_id in EK100 train set
        # HD-EPIC verb/noun names may partially overlap with EK100
        for gv in gt_verbs:
            for gn in gt_nouns:
                # Look up original verb/noun class id by name
                ek_v_rows = tdf[tdf["verb"]==gv]["verb_class"].values
                ek_n_rows = tdf[tdf["noun"]==gn]["noun_class"].values
                if len(ek_v_rows) > 0 and len(ek_n_rows) > 0:
                    orig_v = ek_v_rows[0]; orig_n = ek_n_rows[0]
                    a_id = action_o2m.get((orig_v, orig_n), None)
                    if a_id is not None:
                        act_cls_t[a_id] += 1
                        if int(a_id) in output["action"].topk(5).indices[0].tolist():
                            act_cls_c5[a_id] += 1
                        break
            else:
                continue
            break

        total += 1
        if total % 10 == 0:
            print(f"  Processed {total} actions...")

    v_cmr5 = class_mean_recall(verb_cls_c5, verb_cls_t)
    n_cmr5 = class_mean_recall(noun_cls_c5, noun_cls_t)
    a_cmr5 = class_mean_recall(act_cls_c5,  act_cls_t)
    n_act  = sum(act_cls_t.values())

    print(f"\n{'=' * 70}")
    print(f"HD-EPIC {VIDEO_ID}  |  {total} actions  |  anticipation {ANTICIPATION_SEC}s")
    print(f"{'=' * 70}")
    print(f"{'Metric':<35} {'Verb':>8} {'Noun':>8} {'Action':>8}")
    print(f"{'-' * 63}")
    for k in [1, 3, 5]:
        vp  = 100 * verb_correct[k]  / total
        np_ = 100 * noun_correct[k]  / total
        print(f"  Top-{k} (vocab hit rate)             {vp:>7.1f}% {np_:>7.1f}%")
    print(f"  Class-Mean Recall@5 (EK100 vocab) {v_cmr5:>7.1f}% {n_cmr5:>7.1f}% {a_cmr5:>7.1f}%")
    print(f"{'=' * 70}")
    print()
    print(f"Note: HD-EPIC vocabulary differs from EK100; {n_act} actions could be matched to EK100 action ids.")


if __name__ == "__main__":
    run()
