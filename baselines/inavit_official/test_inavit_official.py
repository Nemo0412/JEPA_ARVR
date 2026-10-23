"""CPU tests: window formula, box clock, released-loader wiring (BGR, box handling), determinism."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_inavit_official as T  # noqa: E402

H, W, N = 256, 456, 120


class FakeBoxes:
    """obj slot 0 and hand slot 0 in normalized xyxy, as the author h5 stores them."""
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
    img[60:180, 100:220, 0] = 255  # red in RGB
    for i in range(N):
        Image.fromarray(img).save(video / f"frame_{i + 1:010d}.jpg", quality=100, subsampling=0)
    (video / "_meta.json").write_text(json.dumps({"n_frames": N, "vfps": 30.0}))


def rows(n=3):
    return pd.DataFrame({"uid": [f"u{i}" for i in range(n)], "video_id": "P01_01", "tick_frame": [100] * n,
                         "vfps": 30.0, "verb": 1, "noun": 2, "action": 3, "mask": 1.0})


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
        return T.WindowDataset(rows() if df is None else df, self.root, FakeBoxes(), self.cfg, train, 0, 4.0, 16)

    def test_window_formula(self):
        idx = T.window_indices(100, 30.0, N, 4.0, 16)
        self.assertEqual((idx[0], idx[-1]), (0, 100))  # 100 - 120 clipped to 0
        idx = T.window_indices(100, 30.0, 101, 2.0, 16)
        self.assertEqual((idx[0], idx[-1]), (40, 100))

    def test_box_clock(self):
        idx = np.array([0, 599, 1198])
        self.assertEqual(T.box_keys("P01_01", idx, 59.94, "nominal").tolist(), [1, 601, 1200])
        self.assertEqual(T.box_keys("P09_08", np.array([0, 300]), 29.97, "nominal").tolist(), [1, 602])
        self.assertEqual(T.box_keys("P01_101", np.array([500]), 50.0, "nominal").tolist(), [501])
        self.assertEqual(T.box_keys("P01_01", idx, 59.94, "native").tolist(), [1, 600, 1199])

    def test_bgr_channel_order_as_released(self):
        # Released retry_load_images uses cv2.imdecode: a red RGB pixel arrives in channel 2.
        x = self.dataset(False)[0]["frames"]  # C,T,H,W normalized with mean/std 0.5
        self.assertEqual(tuple(x.shape), (3, 16, 224, 224))
        means = x[:, 0].mean(dim=(1, 2))
        self.assertGreater(float(means[2]), float(means[0]))

    def test_eval_boxes_follow_released_path(self):
        # Released test path: object boxes go through the pixel-space scale/crop as stored (normalized),
        # hand boxes are untouched; then prepare_boxes (clamp, cxcywh, drop w/h <= 0.05).
        item = self.dataset(False)[0]
        hand = item["hand"][0, 0].numpy()
        x0, y0, x1, y1 = FakeBoxes.HAND
        np.testing.assert_allclose(hand, [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], rtol=1e-5)
        self.assertEqual(tuple(item["obj"].shape), (16, 4, 4))

    def test_train_is_deterministic_per_epoch_and_row(self):
        ds = self.dataset(True)
        for epoch in range(4):
            ds.epoch = epoch
            a, b = ds[1], ds[1]
            self.assertTrue(torch.equal(a["frames"], b["frames"]))
            self.assertTrue(torch.equal(a["obj"], b["obj"]) and torch.equal(a["hand"], b["hand"]))
            self.assertTrue(torch.isfinite(a["frames"]).all())
            self.assertEqual(tuple(a["frames"].shape), (3, 16, 224, 224))
        ds.epoch = 0
        x0 = ds[1]["frames"]
        ds.epoch = 1
        self.assertFalse(torch.equal(x0, ds[1]["frames"]))

    def test_unobservable_row(self):
        df = rows()
        df.loc[0, "mask"] = 0.0
        item = self.dataset(False, df)[0]
        self.assertEqual(float(item["frames"].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
