"""HD-EPIC P01 clip dataset for ScalAnt.

Matches the JEPA clip protocol used in /home/ll5914/Jepa:
  - observe ``TAU_O`` seconds of past (default 4s)
  - anticipate the action ``TAU_A`` seconds ahead (default 2s)

Uses the same TSN-RGB features @ 4 fps as ``rulstm_hdepic``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scalant.config import Config
from scalant.datasets.build import DATASET_REGISTRY
from scalant.datasets.epickitchens import convert_to_anticipation
from scalant.utils import logging

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class HDEPICClip(torch.utils.data.Dataset):
    """Clip-based anticipation on HD-EPIC (closed-set over train+val actions)."""

    def __init__(self, cfg: Config, mode: str):
        assert mode in ["train", "test", "val"]
        self.cfg = cfg
        self.mode = mode
        self.feat_dir = Path(cfg.DATA.DATA_ROOT_PATH) / cfg.DATA.FEAT_DIR
        self.tau_a = float(cfg.DATA.TAU_A)
        self.tau_o = float(cfg.DATA.TAU_O)
        self.past_step_in_sec = float(cfg.DATA.PAST_STEP_IN_SEC)
        self.future_step_in_sec = float(cfg.DATA.FUTURE_STEP_IN_SEC)
        self.long_memory_length = float(cfg.DATA.LONG_MEMORY_LENGTH)
        self.fps = 1.0 / self.past_step_in_sec
        self.data_cache: dict = {}

        train_csv = Path(cfg.DATA.TRAIN_CSV)
        val_csv = Path(cfg.DATA.VAL_CSV)
        if mode == "train":
            clip_csv = train_csv
        else:
            clip_csv = val_csv

        full_csv = Path(getattr(cfg.DATA, "FULL_CSV", "") or "")
        vocab_csvs = [train_csv, val_csv]
        label_csvs = [p for p in [train_csv, val_csv, full_csv] if str(p) and p.is_file()]

        self.verb_map, self.noun_map, self.action_map = self._build_vocab(vocab_csvs)
        # Class 0 is background (TeSTra / ScalAnt convention).
        self.num_classes = {
            "verb": len(self.verb_map) + 1,
            "noun": len(self.noun_map) + 1,
            "action": len(self.action_map) + 1,
        }
        self.verb_noun_to_action = {(0, 0): 0}
        self.verb_noun_to_action.update({
            (self.verb_map[v] + 1, self.noun_map[n] + 1): a + 1
            for (v, n), a in self.action_map.items()
        })
        self.action_classes = {f"{v}_{n}": i for (v, n), i in self.action_map.items()}
        self.verb_classes = {str(k): i for k, i in self.verb_map.items()}
        self.noun_classes = {str(k): i for k, i in self.noun_map.items()}
        self.class_mappings = self._get_class_mappings()

        df = pd.read_csv(clip_csv)
        df = df.copy()
        df.loc[:, "start"] = df["start_timestamp"].astype(float)
        df.loc[:, "end"] = df["end_timestamp"].astype(float)
        if "uid" not in df.columns:
            df.loc[:, "uid"] = range(1, len(df) + 1)
        df.loc[:, "verb_class"] = df["verb_class"].astype(int)
        df.loc[:, "noun_class"] = df["noun_class"].astype(int)
        df.loc[:, "action_class"] = df.apply(
            lambda r: self.action_map.get((int(r.verb_class), int(r.noun_class)), -1),
            axis=1,
        )
        n_bad = int((df["action_class"] < 0).sum())
        if n_bad:
            logger.warning("Dropping %d clips whose (verb, noun) is outside the vocab", n_bad)
            df = df[df["action_class"] >= 0].reset_index(drop=True)

        self.df, self.discarded_df = convert_to_anticipation(df, self.tau_a, self.tau_o)
        logger.info("Discarded %d HD-EPIC clips in anticipate conversion", len(self.discarded_df))

        self.segments_by_video = self._load_segments(label_csvs)
        # Training-set action frequencies (index = action id without background),
        # used by the optional equalized loss (MODEL.EQUALIZED_LOSS).
        self.action_frequency = self._action_frequency(train_csv)
        self.feat_len = self._load_feat_lengths()
        logger.info(
            "Created HD-EPIC clip dataset (%s) with %d samples, "
            "vocab verb/noun/action=%d/%d/%d (incl. bg)",
            mode, len(self.df),
            self.num_classes["verb"], self.num_classes["noun"], self.num_classes["action"],
        )

    @staticmethod
    def _build_vocab(csvs: list[Path]):
        verbs, nouns, actions = set(), set(), set()
        for path in csvs:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                v, n = int(row.verb_class), int(row.noun_class)
                verbs.add(v)
                nouns.add(n)
                actions.add((v, n))
        verb_map = {v: i for i, v in enumerate(sorted(verbs))}
        noun_map = {n: i for i, n in enumerate(sorted(nouns))}
        action_map = {a: i for i, a in enumerate(sorted(actions))}
        return verb_map, noun_map, action_map

    def _load_segments(self, csvs: list[Path]) -> dict[str, list[tuple[float, float, int, int, int]]]:
        seen = set()
        by_video: dict[str, list] = defaultdict(list)
        for path in csvs:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                key = (
                    str(row.video_id),
                    float(row.start_timestamp),
                    float(row.end_timestamp),
                    int(row.verb_class),
                    int(row.noun_class),
                )
                if key in seen:
                    continue
                seen.add(key)
                v, n = int(row.verb_class), int(row.noun_class)
                if v not in self.verb_map or n not in self.noun_map:
                    continue
                if (v, n) not in self.action_map:
                    continue
                by_video[str(row.video_id)].append((
                    float(row.start_timestamp),
                    float(row.end_timestamp),
                    self.verb_map[v] + 1,
                    self.noun_map[n] + 1,
                    self.action_map[(v, n)] + 1,
                ))
        return by_video

    def _action_frequency(self, train_csv: Path) -> np.ndarray:
        counts = np.zeros(len(self.action_map), dtype=np.float64)
        for _, row in pd.read_csv(train_csv).iterrows():
            key = (int(row.verb_class), int(row.noun_class))
            if key in self.action_map:
                counts[self.action_map[key]] += 1.0
        return counts

    def _load_feat_lengths(self) -> dict[str, int]:
        lengths = {}
        for path in self.feat_dir.glob("*.json"):
            meta = json.loads(path.read_text())
            lengths[path.stem] = int(meta.get("n_feat_frames", 0))
        return lengths

    def _get_class_mappings(self):
        num_verbs = self.num_classes["verb"]
        num_nouns = self.num_classes["noun"]
        num_actions = self.num_classes["action"]
        verb_in_action = torch.zeros((num_actions, num_verbs), dtype=torch.float)
        noun_in_action = torch.zeros((num_actions, num_nouns), dtype=torch.float)
        for (verb, noun), action in self.verb_noun_to_action.items():
            verb_in_action[action, verb] = 1.0
            noun_in_action[action, noun] = 1.0
        return {
            ("verb", "action"): verb_in_action,
            ("noun", "action"): noun_in_action,
        }

    def _labels_for_video(self, video_id: str, t_len: int):
        acts = torch.zeros((t_len, self.num_classes["action"]), dtype=torch.float32)
        verbs = torch.zeros((t_len, self.num_classes["verb"]), dtype=torch.float32)
        nouns = torch.zeros((t_len, self.num_classes["noun"]), dtype=torch.float32)
        for t0, t1, v, n, a in self.segments_by_video.get(video_id, []):
            i0 = int(np.floor(t0 * self.fps))
            i1 = int(np.ceil(t1 * self.fps))
            i0 = max(0, min(i0, t_len))
            i1 = max(i0 + 1, min(i1, t_len))
            acts[i0:i1, a] = 1.0
            verbs[i0:i1, v] = 1.0
            nouns[i0:i1, n] = 1.0
        for tensor in (acts, verbs, nouns):
            empty = tensor.sum(dim=-1) == 0
            tensor[empty, 0] = 1.0
        return acts, verbs, nouns

    def _load_video_arrays(self, video_id: str):
        if video_id not in self.data_cache:
            feat_path = self.feat_dir / f"{video_id}.npy"
            feats = torch.from_numpy(np.load(feat_path)).float()
            acts, verbs, nouns = self._labels_for_video(video_id, feats.size(0))
            self.data_cache[video_id] = [feats, acts, verbs, nouns]
        return self.data_cache[video_id]

    def __len__(self):
        return len(self.df)

    def _get_video(self, df_row):
        video_id = str(df_row["video_id"])
        start = max(float(df_row["start"]), 0.0)
        end = max(float(df_row["orig_start"]), 0.0)
        start_f = int(np.floor(start * self.fps))
        end_f = int(np.floor(end * self.fps))
        feats, acts, verbs, nouns = self._load_video_arrays(video_id)
        t_len = feats.size(0)
        start_f = max(0, min(start_f, t_len - 1))
        end_f = max(start_f, min(end_f, t_len - 1))
        tensors = [el[start_f:end_f + 1] for el in (feats, acts, verbs, nouns)]

        frames_per_clip = int(round((self.tau_o + self.tau_a) / self.past_step_in_sec))

        def adjust(*items):
            cur_len = items[0].size(0)
            if cur_len < frames_per_clip:
                npad = frames_per_clip - cur_len
                items = [torch.cat([x[:1]] * npad + [x], dim=0) for x in items]
            elif cur_len > frames_per_clip:
                items = [x[-frames_per_clip:] for x in items]
            return items

        video, acts, verbs, nouns = adjust(*tensors)
        # Training targets keep the upstream dense multi-hot labels (union of all
        # covering actions). Only the scored label is the clip's own action;
        # see scored_labels().
        return video, acts, verbs, nouns

    def scored_labels(self, df_row):
        """Annotated action/verb/noun of this clip, in full (background = 0) space."""
        return (int(df_row["action_class"]) + 1,
                self.verb_map[int(df_row["verb_class"])] + 1,
                self.noun_map[int(df_row["noun_class"])] + 1)

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        video, action_label, verb_label, noun_label = self._get_video(df_row)
        scored = self.scored_labels(df_row)

        past_len = int(self.tau_o // self.past_step_in_sec)
        sample_rate = int(self.future_step_in_sec / self.past_step_in_sec)

        def split_fn(arr):
            if sample_rate == 1:
                return arr[:past_len], arr[past_len:]
            return arr[:past_len], arr[past_len:].flip(0)[::sample_rate].flip(0)

        past_feat, future_feat = split_fn(video)
        past_act, future_act = split_fn(action_label)
        past_verb, future_verb = split_fn(verb_label)
        past_noun, future_noun = split_fn(noun_label)

        assert self.cfg.MODEL.CLS_WORK + self.cfg.MODEL.CLS_ALL + self.cfg.MODEL.CLS_LAST == 1
        if self.cfg.MODEL.CLS_WORK:
            loss_idx = int(self.long_memory_length // self.past_step_in_sec)
        elif self.cfg.MODEL.CLS_LAST:
            loss_idx = -1
        else:
            loss_idx = 0

        return {
            "past_feats": past_feat,
            "past_act": past_act[loss_idx:],
            "past_verb": past_verb[loss_idx:],
            "past_noun": past_noun[loss_idx:],
            "future_feats": future_feat,
            "future_act": future_act,
            "future_verb": future_verb,
            "future_noun": future_noun,
            # Evaluation scores the annotated action instance, as RU-LSTM/InAViT do,
            # instead of the argmax of a multi-hot target (lowest overlapping ID).
            "scored_act": torch.tensor(scored[0]),
            "scored_verb": torch.tensor(scored[1]),
            "scored_noun": torch.tensor(scored[2]),
        }
