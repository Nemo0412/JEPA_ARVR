"""Official InAViT recipe (pretrained MotionFormer, real hand/object boxes) on a 4s / +2s window.

Everything that defines the model and optimization comes from the released InAViT code and its
EK100 config `configs/INAVIT/EK_INAVIT_MF_ant.yaml`:
  - MotionFormer + HoIViT (O=4 objects, U=1 hand, HoIViT layers [1, 6, 10], motion stream),
    two direct heads (verb 97 / noun 300), as the released EK100 model.
  - Initialization from `ek_motionformer_224_16x4.pyth` with the released loader
    (`slowfast.utils.checkpoint.load_checkpoint`, epoch reset, shape-matched tensors; HoIViT
    modules that have no counterpart stay freshly initialized).
  - `construct_optimizer` (AdamW, wd 0.05, backbone/heads BASE_LR 1e-5, HoIViT params 1e-4),
    `steps_with_relative_lrs` evaluated per iteration, label-smoothed CE (0.1) on verb + noun,
    FP16 autocast + GradScaler, no gradient clipping, global batch 20.
  - Train augmentation: RandAugment rand-m7-n4-mstd0.5-inc1 (bicubic, box-aware), short-side
    jitter 256-320, random 224 crop, horizontal flip; mean/std 0.5. Eval: short side 224,
    center 224 crop, single view.
  - Boxes: author-released per-frame detections (h5: video -> 1-based frame on the authors'
    nominal 60 / 50 fps clock -> {obj, hand},
    rows [x1, y1, x2, y2, track_id]), slotted by the released `sort_boxes_sorted`, then the
    released `prepare_boxes` steps (clamp [0, 1], xyxy -> cxcywh, drop boxes with w or h <= 0.05).

Our protocol replaces only the temporal window and the evaluation/selection:
  - 16 frames uniform over the 4s ending at tick = action start - horizon (native frame clock),
    the window used by every baseline in our comparison. (The released EK frame loader observes
    16 x 4 frames ending one frame before an offset inside the action segment.)
  - Selection on validation action class-mean Recall@5 (first max); verb/noun direct heads;
    action score log p(v) + log p(n) over the actions list. Every validation row is scored;
    rows without observation (mask 0) count as misses.

Two implementation deviations from the released loader, both bug fixes:
  - Hand boxes receive the same geometric augmentation as frames and object boxes (the released
    loader transforms only object boxes).
  - Boxes are converted to pixel coordinates before the pixel-space crop/scale/flip and
    normalized by the crop size afterwards (see --box-coords).

Exact resume: per-epoch seeded permutation + batch cursor; each sample's augmentation RNG is
seeded from (seed, epoch, row), so restarts reproduce the same batches and augmentations.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

HERE = Path(__file__).resolve().parent
INAVIT_ROOT = HERE.parent / "InAViT"
sys.path.insert(0, str(INAVIT_ROOT))

from slowfast.config.defaults import get_cfg  # noqa: E402
from slowfast.datasets import transform as sf_transform  # noqa: E402
from slowfast.models import losses as sf_losses  # noqa: E402
from slowfast.models import optimizer as sf_optim  # noqa: E402
from slowfast.models.video_model_builder import Motionformer  # noqa: E402
from slowfast.utils import checkpoint as sf_checkpoint  # noqa: E402
from slowfast.utils.box_ops import box_xyxy_to_cxcywh, zero_empty_boxes  # noqa: E402
from slowfast.utils.LinkBoxes.sort_boxes import sort_boxes_sorted  # noqa: E402

HEADS = ("verb", "noun", "action")
DEFAULT_CFG = INAVIT_ROOT / "configs/INAVIT/EK_INAVIT_MF_ant.yaml"


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def window_indices(tick, vfps, n_frames, obs_sec, num_frames):
    """Uniform `num_frames` over the `obs_sec` seconds ending at `tick` (0-based decoded index)."""
    start = max(0, int(round(tick - obs_sec * vfps)))
    end = max(start + 1, min(tick, n_frames - 1))
    return np.clip(np.linspace(start, end, num_frames), 0, n_frames - 1).astype(np.int64)


def nominal_fps(video):
    """Frame clock of the author detections: videos re-encoded to 60 fps (EK55 ids, e.g. P01_01)
    or 50 fps (EK100 extension ids, e.g. P01_101); the rule of the released EpicKitchensVideoRecord."""
    return 50 if len(video.split("_")[1]) == 3 else 60


def box_keys(video, decoded_idx, vfps, clock):
    """1-based h5 frame keys for 0-based native decoded indices."""
    if clock == "native":
        return decoded_idx + 1
    return np.rint(decoded_idx / vfps * nominal_fps(video)).astype(np.int64) + 1


class BoxReader:
    """Author h5: h5[video][str(frame_1based)] -> group with 'obj' / 'hand' arrays [n, 5]."""

    def __init__(self, path, n_obj, n_hand):
        self.path, self.n_obj, self.n_hand, self._h5 = str(path), n_obj, n_hand, None

    def __call__(self, video, frame_numbers):
        if self._h5 is None:  # opened lazily, once per worker
            self._h5 = h5py.File(self.path, "r")
        group = self._h5[video]
        per_frame = {"obj": [], "hand": []}
        for number in frame_numbers:
            entry = group.get(str(int(number)))
            for key in per_frame:
                value = None if entry is None else entry.get(key)
                per_frame[key].append(np.empty([0, 5]) if value is None else np.asarray(value))
        obj = sort_boxes_sorted(per_frame["obj"], O=self.n_obj, saved_indices=[0, 1])  # [O, T, 4]
        hand = sort_boxes_sorted(per_frame["hand"], O=self.n_hand, saved_indices=[0, 1])  # [U, T, 4]
        return obj.astype(np.float32), hand.astype(np.float32)


class WindowDataset(Dataset):
    """One row per annotated action; frames from <frames>/<video>/frame_{i+1:010d}.jpg."""

    def __init__(self, df, frames_root, boxes, cfg, train, seed, box_coords, obs_sec, num_frames,
                 box_clock="nominal"):
        self.df = df.reset_index(drop=True)
        self.frames_root, self.boxes, self.cfg, self.train = Path(frames_root), boxes, cfg, train
        self.seed, self.box_coords, self.obs_sec, self.num_frames = seed, box_coords, obs_sec, num_frames
        self.box_clock = box_clock
        self.epoch, self._meta = 0, {}

    def __len__(self):
        return len(self.df)

    def _n_frames(self, video):
        if video not in self._meta:
            self._meta[video] = json.loads((self.frames_root / video / "_meta.json").read_text())["n_frames"]
        return self._meta[video]

    def _seed(self, index):
        # Augmentation depends only on (seed, epoch, row): exact under restarts and any worker count.
        digest = hashlib.sha256(f"{self.seed}:{self.epoch}:{index}".encode()).digest()
        value = int.from_bytes(digest[:8], "little")
        random.seed(value)
        np.random.seed(value % (2 ** 32))
        torch.manual_seed(value % (2 ** 63))

    def __getitem__(self, index):
        r = self.df.iloc[index]
        item = {"uid": str(r.uid), "verb": int(r.verb), "noun": int(r.noun), "action": int(r.action),
                "mask": float(r["mask"])}
        crop = self.cfg.DATA.TRAIN_CROP_SIZE if self.train else self.cfg.DATA.TEST_CROP_SIZE
        if item["mask"] <= 0:  # unobservable: scored as a miss, no frames read
            item["frames"] = torch.zeros(3, self.num_frames, crop, crop)
            item["obj"] = torch.zeros(self.num_frames, self.cfg.HOIVIT.O, 4)
            item["hand"] = torch.zeros(self.num_frames, self.cfg.HOIVIT.U, 4)
            return item
        self._seed(index)
        idx = window_indices(int(r.tick_frame), float(r.vfps), self._n_frames(r.video_id),
                             self.obs_sec, self.num_frames)
        frames = np.stack([np.asarray(Image.open(self.frames_root / r.video_id / f"frame_{i + 1:010d}.jpg")
                                      .convert("RGB")) for i in idx])  # T,H,W,C uint8
        height, width = frames.shape[1:3]
        obj, hand = self.boxes(r.video_id, box_keys(r.video_id, idx, float(r.vfps), self.box_clock))
        n_obj = obj.shape[0]
        boxes = np.concatenate([obj, hand], axis=0).astype(np.float64)  # [O+U, T, 4]
        empty = (boxes == 0).all(-1)
        if self.box_coords == "normalized":
            boxes = boxes * np.array([width, height, width, height])
        item["frames"], boxes = self._transform(torch.from_numpy(frames), boxes, crop)
        boxes = boxes / crop
        boxes[empty] = 0
        boxes = torch.from_numpy(boxes).float()
        item["obj"] = self._prepare(boxes[:n_obj])
        item["hand"] = self._prepare(boxes[n_obj:])
        return item

    def _transform(self, frames, boxes, crop):
        shape = boxes.shape
        if self.train:
            aug = sf_transform.create_random_augment(
                input_size=(frames.size(1), frames.size(2)), auto_augment=self.cfg.AUG.AA_TYPE,
                interpolation=self.cfg.AUG.INTERPOLATION, with_boxes=True)
            images = [Image.fromarray(f.numpy()) for f in frames]
            images, boxes = aug(images, boxes=boxes)  # same op for every frame and box
            frames = torch.stack([torch.from_numpy(np.array(im)) for im in images])
        x = frames.float() / 255.0
        x = (x - torch.tensor(self.cfg.DATA.MEAN)) / torch.tensor(self.cfg.DATA.STD)
        x = x.permute(0, 3, 1, 2)  # T,C,H,W for the slowfast transforms
        flat = boxes.reshape(-1, 4)
        if self.train:
            lo, hi = self.cfg.DATA.TRAIN_JITTER_SCALES
            x, flat = sf_transform.random_short_side_scale_jitter(x, lo, hi, boxes=flat)
            x, flat = sf_transform.random_crop(x, crop, boxes=flat)
            x, flat = sf_transform.horizontal_flip(0.5, x, boxes=flat)
        else:
            x, flat = sf_transform.random_short_side_scale_jitter(x, crop, crop, boxes=flat)
            x, flat = sf_transform.uniform_crop(x, crop, 1, boxes=flat)
        return x.permute(1, 0, 2, 3).contiguous(), np.asarray(flat).reshape(shape)

    @staticmethod
    def _prepare(boxes):  # released EKBoxes.prepare_boxes on [O, T, 4] xyxy in [0, 1]
        boxes = boxes.clamp(0, 1).permute(1, 0, 2)  # T, O, 4
        return zero_empty_boxes(box_xyxy_to_cxcywh(boxes), mode="cxcywh", eps=0.05)


def collate(items):
    out = {"uid": [it["uid"] for it in items]}
    for key in ("frames", "obj", "hand"):
        out[key] = torch.stack([it[key] for it in items])
    for key in ("verb", "noun", "action"):
        out[key] = torch.tensor([it[key] for it in items], dtype=torch.long)
    out["mask"] = torch.tensor([it["mask"] for it in items])
    return out


def load_rows(path):
    """Columns: uid, video_id, tick_frame, vfps, verb, noun, action, mask (or mtp_* aliases)."""
    df = pd.read_csv(path)
    df = df.rename(columns={"mtp_verbs": "verb", "mtp_nouns": "noun", "mtp_mask": "mask", "action_id": "action"})
    missing = {"uid", "video_id", "tick_frame", "vfps", "verb", "noun", "action", "mask"} - set(df.columns)
    assert not missing, f"{path}: missing columns {missing}"
    assert df.uid.is_unique
    return df


# ----------------------------------------------------------------------------------------------
# Model / optimization (released components)
# ----------------------------------------------------------------------------------------------
def build_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(str(args.cfg))
    cfg.NUM_GPUS = 1
    cfg.TRAIN.DATASET = "epickitchens"  # two direct heads, as the released EK100 model
    cfg.DATA.NUM_FRAMES = args.num_frames
    if args.epochs is not None:
        cfg.SOLVER.MAX_EPOCH = args.epochs
    if args.lr_steps is not None:
        cfg.SOLVER.STEPS = args.lr_steps
    assert len(cfg.SOLVER.STEPS) == len(cfg.SOLVER.LRS)
    return cfg


def build_model(cfg, checkpoint, device):
    model = Motionformer(cfg)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    sf_checkpoint.load_checkpoint(str(checkpoint), model, data_parallel=False, optimizer=None, scaler=None,
                                  epoch_reset=True,
                                  clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
                                  replace_name_pattern=cfg.TRAIN.CHECKPOINT_REPLACE_NAME_PATTERN,
                                  should_split_qkv=cfg.SPLIT_QKV_CHECKPOINT)
    after = model.state_dict()
    loaded = sorted(k for k in after if not torch.equal(after[k], before[k]))
    fresh = sorted(k for k in after if k not in loaded)
    return model.to(device), {"loaded_tensors": len(loaded), "fresh_tensors": len(fresh), "fresh": fresh,
                              "loaded_elements": int(sum(after[k].numel() for k in loaded)),
                              "total_elements": int(sum(v.numel() for v in after.values()))}


def forward(model, batch, device):
    frames = batch["frames"].to(device, non_blocking=True)
    meta = {"orvit_bboxes": {"obj": batch["obj"].to(device, non_blocking=True),
                             "hand": batch["hand"].to(device, non_blocking=True)}}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _, extra = model([frames], meta)
    return extra


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def evaluate(model, ds, actions, workers, batch_size, device, out_path):
    model.eval()
    a_verb = torch.tensor(actions.verb.to_numpy(), device=device)
    a_noun = torch.tensor(actions.noun.to_numpy(), device=device)
    top5 = {k: np.full((len(ds), 5), -1, dtype=np.int64) for k in HEADS}
    offset = 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate,
                        pin_memory=True)
    with torch.no_grad():
        for batch in loader:
            out = forward(model, batch, device)
            lv, ln = out["verb"].float().log_softmax(-1), out["noun"].float().log_softmax(-1)
            scores = {"verb": lv, "noun": ln, "action": lv[:, a_verb] + ln[:, a_noun]}
            n = len(batch["uid"])
            assert batch["uid"] == ds.df.uid.iloc[offset:offset + n].astype(str).tolist()
            for key in HEADS:
                if not torch.isfinite(scores[key]).all():
                    raise RuntimeError(f"Nonfinite {key} scores at validation")
                top5[key][offset:offset + n] = scores[key].topk(5, dim=-1).indices.cpu().numpy()
            offset += n
    assert offset == len(ds)
    observable = ds.df["mask"].to_numpy() > 0.5
    targets = {"verb": ds.df.verb.to_numpy(), "noun": ds.df.noun.to_numpy(), "action": ds.df.action.to_numpy()}
    result = {"n_val": len(ds), "unobservable_as_miss": int((~observable).sum())}
    for key in HEADS:
        hit = (top5[key] == targets[key][:, None]).any(1) & observable
        present, inverse = np.unique(targets[key], return_inverse=True)
        recall = np.bincount(inverse, weights=hit) / np.bincount(inverse)
        result[key] = {"cmr5": float(recall.mean() * 100), "top5_micro": float(hit.mean() * 100),
                       "top1_micro": float(((top5[key][:, 0] == targets[key]) & observable).mean() * 100),
                       "supported_classes": int(len(present)),
                       "unique_top1": int(len(np.unique(top5[key][observable, 0]))) if observable.any() else 0}
    np.savez_compressed(out_path, uid=ds.df.uid.to_numpy(dtype=str), observable=observable,
                        **{f"{k}_top5": top5[k] for k in HEADS}, **{f"{k}_target": targets[k] for k in HEADS})
    return result


# ----------------------------------------------------------------------------------------------
# Training loop with exact resume
# ----------------------------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(state, path):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--actions-csv", type=Path, required=True, help="id,verb,noun[,...] action list")
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--boxes-h5", type=Path, required=True)
    ap.add_argument("--box-coords", choices=("normalized", "pixel"), required=True,
                    help="coordinate system of the h5 boxes relative to the stored frames")
    ap.add_argument("--box-clock", choices=("nominal", "native"), default="nominal",
                    help="h5 frame keys: nominal 60/50 fps clock of the author detections (EK100), or native index")
    ap.add_argument("--checkpoint", type=Path, required=True, help="ek_motionformer_224_16x4.pyth")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--obs-sec", type=float, default=4.0)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=20, help="global batch (released config: 20)")
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--epochs", type=int, default=None, help="override SOLVER.MAX_EPOCH (released: 100)")
    ap.add_argument("--lr-steps", type=int, nargs="+", default=None, help="override SOLVER.STEPS")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps per allocation (0 = all)")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop and checkpoint after this wall time")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="replay the selected checkpoint on validation")
    ap.add_argument("--train-limit", type=int, default=0, help="smoke: first N train rows")
    ap.add_argument("--val-limit", type=int, default=0, help="smoke: first N validation rows")
    ap.add_argument("--smoke-epoch-steps", type=int, default=0, help="smoke: end each epoch after N steps")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    assert args.batch_size % args.accum == 0
    micro = args.batch_size // args.accum
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = build_cfg(args)
    train_df, val_df = load_rows(args.train_csv), load_rows(args.val_csv)
    train_df = train_df[train_df["mask"] > 0.5].reset_index(drop=True)  # no observation -> not trainable
    if args.train_limit:
        train_df = train_df.head(args.train_limit)
    if args.val_limit:
        val_df = val_df.head(args.val_limit)
    actions = pd.read_csv(args.actions_csv)
    assert (actions.id.to_numpy() == np.arange(len(actions))).all()
    boxes = BoxReader(args.boxes_h5, cfg.HOIVIT.O, cfg.HOIVIT.U)
    common = dict(frames_root=args.frames, boxes=boxes, cfg=cfg, seed=args.seed, box_coords=args.box_coords,
                  obs_sec=args.obs_sec, num_frames=args.num_frames, box_clock=args.box_clock)
    train_ds, val_ds = WindowDataset(train_df, train=True, **common), WindowDataset(val_df, train=False, **common)

    contract = {
        "recipe": "inavit-official-ek-config/pretrained-motionformer/real-boxes/window-obs{}s".format(args.obs_sec),
        "cfg": str(args.cfg), "cfg_sha256": sha256(args.cfg), "checkpoint_sha256": sha256(args.checkpoint),
        "script_sha256": sha256(__file__), "train_csv_sha256": sha256(args.train_csv),
        "val_csv_sha256": sha256(args.val_csv), "actions_sha256": sha256(args.actions_csv),
        "boxes_h5": str(args.boxes_h5), "box_coords": args.box_coords, "box_clock": args.box_clock, "frames": str(args.frames.resolve()),
        "train_rows": len(train_df), "val_rows": len(val_df), "batch": args.batch_size, "accum": args.accum,
        "max_epoch": cfg.SOLVER.MAX_EPOCH, "lr_steps": list(cfg.SOLVER.STEPS), "lrs": list(cfg.SOLVER.LRS),
        "base_lr": cfg.SOLVER.BASE_LR, "hoivit_lr": cfg.SOLVER.HOIVIT_BASE_LR, "seed": args.seed,
        "obs_sec": args.obs_sec, "num_frames": args.num_frames, "selection": "val_action_cmr5_first_max",
        "smoke": bool(args.train_limit or args.val_limit or args.smoke_epoch_steps),
    }
    model, init_report = build_model(cfg, args.checkpoint, device)
    optimizer = sf_optim.construct_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_fn = sf_losses.get_loss_func(cfg)(reduction="mean")

    last, best_path = args.out / "last.pt", args.out / "best.pt"
    epoch, cursor, step, best, history = 0, 0, 0, -1.0, []
    run_loss, run_n, nonfinite_steps = 0.0, 0, 0
    if args.eval_only:
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        assert state["contract"] == contract, "contract changed since the checkpoint"
        model.load_state_dict(state["model"], strict=True)
        metrics = evaluate(model, val_ds, actions, args.workers, args.eval_batch_size, device,
                           args.out / "replay_predictions.npz")
        metrics["epoch"] = state["epoch"]
        (args.out / "replay_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
        return
    if args.resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        assert state["contract"] == contract, "contract changed since the checkpoint"
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"])
        epoch, cursor, step, best, history = (state[k] for k in ("epoch", "cursor", "step", "best", "history"))
        run_loss, run_n, nonfinite_steps = state["run_loss"], state["run_n"], state["nonfinite_steps"]
        print(json.dumps({"resumed": {"epoch": epoch, "cursor": cursor, "step": step}}), flush=True)
    elif last.exists():
        raise RuntimeError("output already has a checkpoint; pass --resume")
    else:
        (args.out / "contract.json").write_text(json.dumps(contract, indent=2))
        (args.out / "init_report.json").write_text(json.dumps(init_report, indent=2))
    print(json.dumps({"params": sum(p.numel() for p in model.parameters()),
                      "init": {k: v for k, v in init_report.items() if k != "fresh"},
                      "train_rows": len(train_df), "val_rows": len(val_df)}), flush=True)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    if args.smoke_epoch_steps:
        steps_per_epoch = min(steps_per_epoch, args.smoke_epoch_steps)
    started, segment_steps = time.monotonic(), 0
    start_pos = {"epoch": epoch, "cursor": cursor, "step": step}

    def save():
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "rng": rng_state(), "epoch": epoch, "cursor": cursor, "step": step, "best": best,
                     "history": history, "run_loss": run_loss, "run_n": run_n,
                     "nonfinite_steps": nonfinite_steps, "contract": contract}, last)

    def finish(done):
        record = {"job": os.getenv("SLURM_JOB_ID"), "elapsed_s": time.monotonic() - started, "start": start_pos,
                  "segment_steps": segment_steps, "epoch": epoch, "cursor": cursor, "step": step,
                  "complete": done, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                  "checkpoint_sha256": sha256(last)}
        with (args.out / "segments.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    while epoch < cfg.SOLVER.MAX_EPOCH:
        model.train()
        train_ds.epoch = epoch
        order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(10000 + args.seed + epoch))
        order = order.tolist()[: steps_per_epoch * args.batch_size]
        loader = DataLoader(Subset(train_ds, order[cursor * args.batch_size:]), batch_size=micro, shuffle=False,
                            num_workers=args.workers, collate_fn=collate, pin_memory=True, drop_last=False)
        micro_batches = iter(loader)
        while cursor < steps_per_epoch:
            expected = train_df.uid.iloc[order[cursor * args.batch_size:(cursor + 1) * args.batch_size]].tolist()
            lr = sf_optim.get_epoch_lr(epoch + cursor / steps_per_epoch, cfg)
            sf_optim.set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            seen, step_loss = [], 0.0
            for _ in range(math.ceil(len(expected) / micro)):
                batch = next(micro_batches)
                seen += batch["uid"]
                out = forward(model, batch, device)
                labels = {"verb": batch["verb"].to(device), "noun": batch["noun"].to(device)}
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    parts = loss_fn(out, labels)
                    loss = (parts["verb_loss"] + parts["noun_loss"]) * len(batch["uid"]) / len(expected)
                scaler.scale(loss).backward()
                step_loss += float(loss.detach())
            assert seen == expected, "repeated, skipped or reordered training rows"
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not math.isfinite(step_loss) or scaler.get_scale() < scale_before:
                nonfinite_steps += 1  # GradScaler skipped this step (overflow), as in the released trainer
            else:
                run_loss += step_loss
                run_n += 1
            cursor += 1
            step += 1
            segment_steps += 1
            if segment_steps % 20 == 0:
                print(json.dumps({"epoch": epoch, "cursor": cursor, "step": step, "lr": lr,
                                  "mean_loss": run_loss / max(1, run_n), "skipped": nonfinite_steps,
                                  "scale": scaler.get_scale(), "s": round(time.monotonic() - started, 1),
                                  "mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}), flush=True)
            out_of_budget = (args.max_steps and segment_steps >= args.max_steps) or \
                            (args.max_minutes and time.monotonic() - started > args.max_minutes * 60)
            if cursor < steps_per_epoch and (out_of_budget or step % args.save_every == 0):
                save()
                if out_of_budget:
                    finish(False)
                    return
        metrics = evaluate(model, val_ds, actions, args.workers, args.eval_batch_size, device,
                           args.out / f"val_epoch{epoch + 1:03d}.npz")
        history.append({"epoch": epoch + 1, "train_loss": run_loss / max(1, run_n),
                        "skipped_steps": nonfinite_steps, "val": metrics})
        if metrics["action"]["cmr5"] > best:
            best = metrics["action"]["cmr5"]
            atomic_save({"model": model.state_dict(), "epoch": epoch + 1, "val": metrics, "contract": contract},
                        best_path)
        print(json.dumps({"epoch_complete": epoch + 1, "val": metrics, "best_action_cmr5": best}), flush=True)
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        epoch, cursor, run_loss, run_n = epoch + 1, 0, 0.0, 0
        save()
        if (args.max_steps and segment_steps >= args.max_steps) or \
                (args.max_minutes and time.monotonic() - started > args.max_minutes * 60):
            break
    finish(epoch >= cfg.SOLVER.MAX_EPOCH)


if __name__ == "__main__":
    main()
