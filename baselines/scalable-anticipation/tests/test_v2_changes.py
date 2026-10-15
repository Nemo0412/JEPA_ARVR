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
from scalant.criterion.criterion import Criterion_LSTR, _scored_labels
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
