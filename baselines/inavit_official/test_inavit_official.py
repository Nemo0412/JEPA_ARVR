"""CPU tests: window formula, box/frame geometric alignment, deterministic augmentation, scoring."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_inavit_official as T  # noqa: E402

H, W, N = 256, 456, 120


class FakeBoxes:
    """obj slot 0 = red rectangle, hand slot 0 = blue rectangle, in normalized xyxy."""
    OBJ = np.array([100, 60, 220, 180]) / np.array([W, H, W, H])
    HAND = np.array([300, 120, 400, 230]) / np.array([W, H, W, H])

    def __call__(self, video, numbers):
        obj = np.zeros((4, len(numbers), 4), np.float32)
        hand = np.zeros((1, len(numbers), 4), np.float32)
        obj[0, :] = self.OBJ
        hand[0, :] = self.HAND
        return obj, hand


def make_frames(root):
    video = root / "P01_01"
    video.mkdir()
    img = np.zeros((H, W, 3), np.uint8)
    img[60:180, 100:220, 0] = 255
    img[120:230, 300:400, 2] = 255
    for i in range(N):
        Image.fromarray(img).save(video / f"frame_{i + 1:010d}.jpg", quality=100, subsampling=0)
    (video / "_meta.json").write_text(json.dumps({"n_frames": N, "vfps": 30.0}))


def rows(n=3):
    return pd.DataFrame({"uid": [f"u{i}" for i in range(n)], "video_id": "P01_01", "tick_frame": [100] * n,
                         "vfps": 30.0, "verb": 1, "noun": 2, "action": 3, "mask": 1.0})


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def iou(a, b):
    lt, rb = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = np.prod(np.clip(rb - lt, 0, None))
    return inter / (np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_frames(cls.root)

        class Args:
            cfg, num_frames, epochs, lr_steps = T.DEFAULT_CFG, 16, None, None
        cls.cfg = T.build_cfg(Args)

    def dataset(self, train, df=None):
        return T.WindowDataset(rows() if df is None else df, self.root, FakeBoxes(), self.cfg, train, 0,
                               "normalized", 4.0, 16)

    def check_alignment(self, item):
        frame = item["frames"][:, 0].numpy()  # C,H,W normalized to [-1, 1]
        crop = frame.shape[-1]
        for key, on, off in (("obj", 0, 2), ("hand", 2, 0)):
            cx, cy, w, h = item[key][0, 0].numpy() * crop
            box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            mask = (frame[on] > 0.5) & (frame[off] < -0.5)
            if w == 0:  # box fully cropped away: the region must be (nearly) absent as well
                self.assertLess(mask.sum(), 0.05 * crop * crop)
                continue
            self.assertGreater(iou(box, bbox_of(mask)), 0.85, key)

    def test_window_formula(self):
        idx = T.window_indices(100, 30.0, N, 4.0, 16)
        self.assertEqual(idx[0], 0)  # 100 - 120 clipped to 0
        self.assertEqual(idx[-1], 100)
        idx = T.window_indices(100, 30.0, 101, 2.0, 16)
        self.assertEqual((idx[0], idx[-1]), (40, 100))

    def test_box_clock(self):
        idx = np.array([0, 599, 1198])
        self.assertEqual(T.box_keys("P01_01", idx, 59.94, "nominal").tolist(), [1, 601, 1200])
        self.assertEqual(T.box_keys("P09_08", np.array([0, 300]), 29.97, "nominal").tolist(), [1, 602])
        self.assertEqual(T.box_keys("P01_101", np.array([500]), 50.0, "nominal").tolist(), [501])
        self.assertEqual(T.box_keys("P01_01", idx, 59.94, "native").tolist(), [1, 600, 1199])

    def test_eval_alignment(self):
        item = self.dataset(False)[0]
        self.assertEqual(tuple(item["frames"].shape), (3, 16, 224, 224))
        self.check_alignment(item)

    def test_train_geometry_alignment(self):
        # Scale jitter / crop / flip only (RandAugment's colour ops make pixel masks ambiguous).
        orig = T.sf_transform.create_random_augment
        T.sf_transform.create_random_augment = lambda **kw: (lambda imgs, boxes: (imgs, boxes))
        try:
            ds = self.dataset(True)
            for epoch in range(12):
                ds.epoch = epoch
                self.check_alignment(ds[epoch % 3])
        finally:
            T.sf_transform.create_random_augment = orig

    def test_train_randaugment_runs_and_is_deterministic(self):
        ds = self.dataset(True)
        for epoch in range(6):
            ds.epoch = epoch
            a, b = ds[1], ds[1]
            self.assertTrue(torch.equal(a["frames"], b["frames"]))
            self.assertTrue(torch.equal(a["obj"], b["obj"]) and torch.equal(a["hand"], b["hand"]))
            self.assertTrue(torch.isfinite(a["frames"]).all())
            self.assertTrue(((a["obj"] >= 0) & (a["obj"] <= 1)).all())
        ds.epoch = 0
        x0 = ds[1]["frames"]
        ds.epoch = 1
        self.assertFalse(torch.equal(x0, ds[1]["frames"]))

    def test_released_geometric_box_ops(self):
        # The released box-aware RandAugment ops must move boxes with the pixels.
        from slowfast.datasets.boxes_autoaugment.autoaugment import AugmentOp, _HPARAMS_DEFAULT
        img = np.zeros((H, W, 3), np.uint8)
        img[60:180, 100:220, 0] = 255
        box = np.array([[[100, 60, 220, 180]] * 2], float)  # [1 object, 2 frames, 4] pixel xyxy
        hp = dict(_HPARAMS_DEFAULT, translate_const=int(min(H, W) * 0.45))
        for name in ("Rotate", "ShearX", "ShearY", "TranslateXRel", "TranslateYRel"):
            for seed in range(4):
                np.random.seed(seed); __import__("random").seed(seed)
                op = AugmentOp(name, prob=1.0, magnitude=7, hparams=hp)
                imgs, boxes = op([Image.fromarray(img)] * 2, boxes=box.copy())
                out = np.asarray(imgs[0])
                mask = (out[..., 0] > 128) & (out[..., 2] < 64)
                # The trainer clips later (crop + clamp); clip to the image here.
                b = np.clip(boxes[0, 0], 0, [W, H, W, H])
                self.assertGreater(iou(b, bbox_of(mask)), 0.8, (name, seed))

    def test_unobservable_row(self):
        df = rows()
        df.loc[0, "mask"] = 0.0
        item = self.dataset(False, df)[0]
        self.assertEqual(float(item["frames"].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
