"""Evaluate a model on the released-InAViT validation windows (R0, R5) or the honest 1s control (A1).

Models:
  mf      pretrained EK100 MotionFormer (InAViT's initialization; recognition model), no boxes
  inavit  our trained InAViT (official recipe, 4s / +2s window) best.pt, with the author boxes
Loading and scoring reuse baselines/inavit_official/train_inavit_official.py: released
retry_load_images (BGR) + released test-mode spatial path (short side 224, centre crop), boxes read
with the h5 key = nominal frame number of each sampled frame, `evaluate` (all 9,668 rows; rows
without a window count as misses; action = log p(v) + log p(n) top-5 over actions.csv; the released
local MT5R rule is reported alongside).
Usage: eval_released_windows.py <variant R0|R5|A1> <model mf|inavit> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path("/scratch/yh6416/VJEPA2-EXP")
sys.path.insert(0, str(ROOT / ".worktrees/jepa-arvr-inavit-official/baselines/inavit_official"))
import train_inavit_official as T  # noqa: E402

RW = ROOT / "outputs/inavit_official_ek100/released_window"
CKPT_MF = ROOT / "third_party/inavit/CP/ek_motionformer_224_16x4.pyth"
BEST = ROOT / "outputs/inavit_official_ek100/train_e100/best.pt"
VAL = ROOT / "outputs/collab_inavit_ek100/prep/csv/EK100_val_stream_mtp.csv"
ACTIONS = ROOT / "third_party/rulstm/RULSTM/data/ek100/actions.csv"
H5 = ROOT / "data/ek100/inavit_official/epickitchens.h5"


class ReleasedWindowDataset(Dataset):
    def __init__(self, df, variant, cfg, boxes):
        self.df, self.variant, self.cfg, self.boxes = df, variant, cfg, boxes
        self.plans = {}

    def __len__(self):
        return len(self.df)

    def _plan(self, video):
        if video not in self.plans:
            self.plans[video] = json.loads((RW / "plans" / f"{video}.json").read_text())
        return self.plans[video]

    def __getitem__(self, i):
        r = self.df.iloc[i]
        cfg = self.cfg
        item = {"uid": str(r.uid), "verb": int(r.verb), "noun": int(r.noun), "action": int(r.action),
                "mask": float(r["mask"])}
        entry = self._plan(r.video_id)[r.uid]
        native, keys = entry.get(self.variant + "_native"), entry.get(self.variant)
        if item["mask"] <= 0 or native is None:
            item["frames"] = torch.zeros(3, 16, 224, 224)
            item["obj"], item["hand"] = torch.zeros(16, cfg.HOIVIT.O, 4), torch.zeros(16, cfg.HOIVIT.U, 4)
            return item
        frames = T.sf_utils.retry_load_images(
            [str(RW / "frames" / r.video_id / f"frame_{j + 1:010d}.jpg") for j in native], cfg)
        if self.boxes is not None:
            obj, hand = self.boxes(r.video_id, np.asarray(keys))
        else:
            obj, hand = np.zeros((cfg.HOIVIT.O, 16, 4), np.float32), np.zeros((cfg.HOIVIT.U, 16, 4), np.float32)
        boxes = {"obj": obj, "hand": hand}
        frames = T.sf_utils.tensor_normalize(frames, cfg.DATA.MEAN, cfg.DATA.STD).permute(3, 0, 1, 2)
        frames, boxes = T.sf_utils.spatial_sampling(frames, boxes=boxes, spatial_idx=1, min_scale=224, max_scale=224,
                                                    crop_size=224, random_horizontal_flip=cfg.DATA.RANDOM_FLIP,
                                                    inverse_uniform_sampling=cfg.DATA.INV_UNIFORM_SAMPLE)
        item["frames"] = frames
        item["obj"] = T.WindowDataset._prepare(torch.from_numpy(np.asarray(boxes["obj"], dtype=np.float32)))
        item["hand"] = T.WindowDataset._prepare(torch.from_numpy(np.asarray(boxes["hand"], dtype=np.float32)))
        return item


def main():
    variant, model_name, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])

    class A:
        cfg, num_frames, epochs, lr_steps = T.DEFAULT_CFG, 16, None, None
    cfg = T.build_cfg(A)
    df = T.load_rows(VAL)
    # Diagnostic subset: rows of videos whose frames were extracted (the damaged MP4s P22_01, P30_08,
    # P29_05 are excluded from every variant and model alike, not scored as misses).
    done = {p.stem for p in (RW / "plans").glob("*.json")}
    excluded = sorted(set(df.video_id) - done)
    df = df[df.video_id.isin(done)].reset_index(drop=True)
    df["mask"] = 1.0  # R* always have frames (inside the action); A1 rows without a window are set below
    device = torch.device("cuda")
    if model_name == "mf":
        cfg.HOIVIT.ENABLE = False
        cfg.HOIVIT.LAYERS = []
        model, report = T.build_model(cfg, CKPT_MF, device)
        boxes = None
    else:
        model, report = T.build_model(cfg, CKPT_MF, device)
        state = torch.load(BEST, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        report = {"best_epoch": state["epoch"], "best_val_action_cmr5_own_window": state["val"]["action"]["cmr5"]}
        boxes = T.BoxReader(H5, cfg.HOIVIT.O, cfg.HOIVIT.U)
    ds = ReleasedWindowDataset(df, variant, cfg, boxes)
    # A1 rows whose window would end before the video start: miss.
    for i, r in enumerate(df.itertuples()):
        if ds._plan(r.video_id)[r.uid].get(variant) is None:
            df.at[i, "mask"] = 0.0
    actions = pd.read_csv(ACTIONS)
    metrics = T.evaluate(model, ds, actions, 8, 32, device, out.with_suffix(".npz"))
    metrics.update({"variant": variant, "model": model_name, "excluded_videos": excluded, "n_rows": len(df), "init": {k: v for k, v in report.items() if k != "fresh"}})
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("variant", "model", "unobservable_as_miss")}), flush=True)
    for k in ("verb", "noun", "action"):
        print(k, round(metrics[k]["cmr5"], 2), "micro-top5", round(metrics[k]["top5_micro"], 2), flush=True)
    print("action released-MT5R rule", round(metrics["action_released_mt5r_rule"]["cmr5"], 2), flush=True)


if __name__ == "__main__":
    main()
