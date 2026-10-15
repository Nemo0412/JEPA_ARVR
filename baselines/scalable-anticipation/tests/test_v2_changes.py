"""Checks for the v2 ScalAnt HD-EPIC changes (scored labels, equalized loss, 32s config).

Run from baselines/scalable-anticipation:  python -m unittest tests.test_v2_changes -v
The dataset test needs the HD-EPIC P01 features/CSVs referenced by the config.
"""
import os
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scalant.config import load_config
from scalant.criterion.criterion import Criterion_LSTR, _mt5r_inputs, _scored_labels
from scalant.criterion.loss import MultipCrossEntropyEqualizedLoss, MultipCrossEntropyLoss

CFG = "configs/hdepic/clip_32s_2s.yaml"


class ScoredLabelTests(unittest.TestCase):
    def test_upstream_argmax_without_scored_labels(self):
        target = torch.tensor([[0., 1., 1.], [0., 0., 1.]])  # notice space
        self.assertEqual(_scored_labels(target, None, 0).tolist(), [1, 2])

    def test_scored_labels_shift_out_background(self):
        target = torch.zeros(2, 3)
        scored = torch.tensor([3, 1])  # full space, background = 0
        self.assertEqual(_scored_labels(target, scored, 0).tolist(), [2, 0])


class UnobservableMissTests(unittest.TestCase):
    def test_unobservable_rows_are_misses(self):
        logits = torch.zeros(3, 10)
        logits[:, 2] = 5.0  # every row would hit class 2
        scored = torch.tensor([3, 3, 3])  # full-space 3 -> notice-space 2
        out = _mt5r_inputs(logits, torch.zeros(3, 10), scored, torch.tensor([1, 0, 1]), 0)
        top5 = out["logits"].topk(5, dim=1).indices
        hits = (top5 == out["labels"][:, None]).any(dim=1).tolist()
        self.assertEqual(hits, [True, False, True])


class LossSelectionTests(unittest.TestCase):
    def _cfg(self, flag):
        return SimpleNamespace(MODEL=SimpleNamespace(IGNORE_INDEX=0, EQUALIZED_LOSS=flag))

    def test_equalized_loss_uses_dataset_frequency(self):
        dataset = SimpleNamespace(action_frequency=np.array([10., 0., 1.]))
        criterion = Criterion_LSTR(self._cfg(True), dataset)
        self.assertIsInstance(criterion.action_cls, MultipCrossEntropyEqualizedLoss)
        logits = torch.randn(4, 4)
        target = torch.zeros(4, 4)
        target[:, 1] = 1.
        self.assertTrue(torch.isfinite(criterion.action_cls(logits, target)))

    def test_flag_off_keeps_plain_loss(self):
        dataset = SimpleNamespace(action_frequency=np.array([10., 0., 1.]))
        criterion = Criterion_LSTR(self._cfg(False), dataset)
        self.assertIsInstance(criterion.action_cls, MultipCrossEntropyLoss)


@unittest.skipUnless(os.path.isdir("/scratch/ll5914/datasets/HD-EPIC"), "HD-EPIC data not available")
class HDEPICDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scalant.datasets.hdepic import HDEPICClip
        cls.cfg = load_config(CFG)
        cls.train = HDEPICClip(cls.cfg, "train")
        cls.val = HDEPICClip(cls.cfg, "val")

    def test_config_values(self):
        c = self.cfg
        self.assertEqual((c.DATA.TAU_O, c.DATA.LONG_MEMORY_LENGTH, c.DATA.TAU_A, c.DATA.FUTURE_STEP_IN_SEC),
                         (32., 28., 2., 2.))
        self.assertEqual(c.MODEL.N_QUERIES, 1)
        self.assertEqual((c.MODEL.ENCODER_CLASS, c.MODEL.MAMBA_VERSION, c.MODEL.D_STATE, c.MODEL.EQUALIZED_LOSS),
                         ("MAMBA", 1, 16, True))

    def test_frequency_shape(self):
        freq = self.train.action_frequency
        self.assertEqual(len(freq), self.train.num_classes["action"] - 1)
        self.assertGreater(freq.sum(), 0)

    def test_items_and_scored_labels(self):
        own_in_last = 0
        n = len(self.val)
        for i in range(n):
            item = self.val[i]
            self.assertEqual(tuple(item["past_feats"].shape), (128, 1024))
            self.assertEqual(item["future_act"].shape[0], self.cfg.MODEL.N_QUERIES)  # one slot at t0
            self.assertEqual(item["past_act"].shape[0], 128)  # CLS_ALL
            row = self.val.df.iloc[i]
            self.assertEqual(int(item["scored_act"]), int(row["action_class"]) + 1)
            own_in_last += int(item["future_act"][-1, int(item["scored_act"])] == 1)
            # Training targets stay multi-hot (no overwrite of the last slot).
            self.assertGreaterEqual(float(item["future_act"][-1].sum()), 1.0)
        print(f"\nval clips={n}, own action present in last future slot: {own_in_last}/{n}")
        self.assertGreater(own_in_last / n, 0.95)


if __name__ == "__main__":
    unittest.main()


def _rulstm_rows(path):
    import csv
    with open(path) as stream:
        return [r for r in csv.reader(stream) if r]


class RULSTMDatasetTests(unittest.TestCase):
    """EK100 / EGTEA through ClipAnticipation (needs converted 4 Hz features)."""

    CASES = [("configs/ek100/clip_32s_2s.yaml", 3806), ("configs/egtea/clip_32s_2s_s1.yaml", 106),
             ("configs/egtea/clip_32s_2s_s2.yaml", 106), ("configs/egtea/clip_32s_2s_s3.yaml", 106)]

    def test_populations_labels_and_items(self):
        from pathlib import Path
        from scalant.datasets.clip_anticipation import ClipAnticipation
        for path, n_actions in self.CASES:
            cfg = load_config(path)
            feat_dir = Path(cfg.DATA.DATA_ROOT_PATH) / cfg.DATA.FEAT_DIR
            if not (feat_dir / "_conversion_summary.json").is_file():
                print(f"\nskipping {path}: features not converted yet")
                continue
            with self.subTest(path=path):
                train, val = ClipAnticipation(cfg, "train"), ClipAnticipation(cfg, "val")
                self.assertEqual(train.num_classes["action"], n_actions + 1)
                self.assertEqual(len(train.action_frequency), n_actions)
                rows = _rulstm_rows(cfg.DATA.VAL_CSV)
                # Full official validation population is scored.
                self.assertEqual(len(val), len(rows))
                self.assertEqual(int((val.df["observable"] == 0).sum()), len(val.discarded_df))
                # Class mapping reproduces the RU-LSTM action column.
                self.assertTrue((val.df["action_class"] == val.df["action_id"]).all())
                self.assertEqual(len(train), len(_rulstm_rows(cfg.DATA.TRAIN_CSV)) - len(train.discarded_df))
                own = 0
                idx = list(range(0, len(val), max(1, len(val) // 400)))
                for i in idx:
                    item = val[i]
                    self.assertEqual(tuple(item["past_feats"].shape), (128, 1024))
                    if int(item["observable"]):
                        own += int(item["future_act"][-1, int(item["scored_act"])] == 1)
                        self.assertTrue(torch.isfinite(item["past_feats"]).all())
                    else:
                        self.assertEqual(float(item["past_feats"].abs().sum()), 0.0)
                n_obs = sum(int(val[i]["observable"]) for i in idx)
                print(f"\n{path}: train={len(train)} (dropped {len(train.discarded_df)}), "
                      f"val={len(val)} (unobservable {len(val.discarded_df)}), "
                      f"own action in last slot {own}/{n_obs} sampled")
                self.assertGreater(own / max(n_obs, 1), 0.95)


class WindowLabelEquivalenceTests(unittest.TestCase):
    """Per-window label construction equals whole-video construction then slicing."""

    def test_window_labels_match_whole_video(self):
        from pathlib import Path
        from scalant.datasets.clip_anticipation import ClipAnticipation
        cfg = load_config("configs/egtea/clip_32s_2s_s1.yaml")
        if not (Path(cfg.DATA.DATA_ROOT_PATH) / cfg.DATA.FEAT_DIR / "_conversion_summary.json").is_file():
            self.skipTest("EGTEA features not converted")
        ds = ClipAnticipation(cfg, "val")
        for i in range(0, len(ds), 97):
            row = ds.df.iloc[i]
            video = str(row["video_id"])
            t_len = ds._load_features(video).shape[0]
            ref = torch.zeros((t_len, ds.num_classes["action"]))
            for t0, t1, v, n, a in ds.segments_by_video.get(video, []):
                i0 = max(0, min(int(np.floor(t0 * ds.fps)), t_len))
                i1 = max(i0 + 1, min(int(np.ceil(t1 * ds.fps)), t_len))
                ref[i0:i1, a] = 1.0
            ref[ref.sum(-1) == 0, 0] = 1.0
            lo = max(0, min(int(np.floor(max(float(row["start"]), 0.0) * ds.fps)), t_len - 1))
            hi = max(lo, min(int(np.floor(max(float(row["orig_start"]), 0.0) * ds.fps)), t_len - 1)) + 1
            acts, _, _ = ds._labels_for_range(video, t_len, lo, hi)
            self.assertTrue(torch.equal(acts, ref[lo:hi]), f"mismatch at {i}")
