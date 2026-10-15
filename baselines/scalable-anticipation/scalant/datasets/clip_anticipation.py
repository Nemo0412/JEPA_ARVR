"""Action-instance (clip) anticipation dataset shared by EK100, EGTEA and HD-EPIC.

One sample per annotated action: TAU_O seconds of 4 Hz features ending TAU_A
seconds before the action start, plus the future slot(s). Training targets are
upstream ScalAnt/TeSTra dense multi-hot per-frame labels (union of every
covering action, background = class 0), rebuilt from the annotation CSVs.
Each item also carries the clip's annotated labels (`scored_*`), which the
criterion uses for mean top-5 recall.

Annotation formats (DATA.ANNOTATION_FORMAT):
  - "rulstm": RU-LSTM CSVs without header: id, video, start_frame, end_frame,
    verb, noun, action, frames on a 30 fps clock (EK100, EGTEA).
  - "hdepic": HD-EPIC V-JEPA CSVs with header (video_id, start_timestamp,
    end_timestamp, verb_class, noun_class).
Class space (DATA.ACTIONS_CSV): an official RU-LSTM actions.csv, or "" for a
closed vocabulary over the train+val (verb, noun) pairs.

Early actions: training drops clips whose observation ends before the first
feature (upstream `convert_to_anticipation`). With DATA.SCORE_ALL_EVAL, those
evaluation clips are kept with `observable = 0`; the criterion scores them as
misses so every method is scored on the full official population.
"""
from __future__ import annotations

import csv
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

RULSTM_FPS = 30.0
RULSTM_COLUMNS = ["uid", "video_id", "start_frame", "end_frame", "verb_class", "noun_class", "action_id"]


def _read_annotations(path: Path, fmt: str) -> pd.DataFrame:
    """Return columns uid, video_id, start, end (seconds), verb_class, noun_class."""
    if fmt == "rulstm":
        df = pd.read_csv(path, header=None, names=RULSTM_COLUMNS, skipinitialspace=True)
        df["video_id"] = df["video_id"].astype(str).str.strip()
        df["start"] = df["start_frame"].astype(float) / RULSTM_FPS
        df["end"] = df["end_frame"].astype(float) / RULSTM_FPS
    elif fmt == "hdepic":
        df = pd.read_csv(path)
        df["start"] = df["start_timestamp"].astype(float)
        df["end"] = df["end_timestamp"].astype(float)
        if "uid" not in df.columns:
            df["uid"] = range(1, len(df) + 1)
    else:
        raise ValueError(f"Unknown ANNOTATION_FORMAT {fmt}")
    df["verb_class"] = df["verb_class"].astype(int)
    df["noun_class"] = df["noun_class"].astype(int)
    return df


def _read_official_actions(path: Path):
    """RU-LSTM actions.csv -> {(verb, noun): action_id}; EK100 (header) or EGTEA (id, v_n, name)."""
    with path.open() as stream:
        rows = [r for r in csv.reader(stream, skipinitialspace=True) if r]
    mapping = {}
    if rows[0][0] == "id":  # EK100: id,verb,noun,action
        for r in rows[1:]:
            mapping[(int(r[1]), int(r[2]))] = int(r[0])
    else:  # EGTEA: id, verb_noun, name
        for r in rows:
            verb, noun = (int(x) for x in r[1].split("_"))
            mapping[(verb, noun)] = int(r[0])
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError(f"Non-contiguous action ids in {path}")
    return mapping


@DATASET_REGISTRY.register()
class ClipAnticipation(torch.utils.data.Dataset):
    def __init__(self, cfg: Config, mode: str):
        assert mode in ["train", "test", "val"]
        self.cfg = cfg
        self.mode = mode
        data = cfg.DATA
        self.fmt = getattr(data, "ANNOTATION_FORMAT", "hdepic")
        self.feat_dir = Path(data.DATA_ROOT_PATH) / data.FEAT_DIR
        self.tau_a = float(data.TAU_A)
        self.tau_o = float(data.TAU_O)
        self.past_step_in_sec = float(data.PAST_STEP_IN_SEC)
        self.future_step_in_sec = float(data.FUTURE_STEP_IN_SEC)
        self.long_memory_length = float(data.LONG_MEMORY_LENGTH)
        self.fps = 1.0 / self.past_step_in_sec
        self.data_cache: dict = {}

        train_csv, val_csv = Path(data.TRAIN_CSV), Path(data.VAL_CSV)
        full_csv = Path(getattr(data, "FULL_CSV", "") or "")
        label_csvs = [p for p in [train_csv, val_csv, full_csv] if str(p) and p.is_file()]
        frames = {p: _read_annotations(p, self.fmt) for p in label_csvs}

        actions_csv = str(getattr(data, "ACTIONS_CSV", "") or "")
        if actions_csv:
            pair_to_action = _read_official_actions(Path(actions_csv))
            verbs = sorted({v for v, _ in pair_to_action})
            nouns = sorted({n for _, n in pair_to_action})
            self.verb_map = {v: v for v in range(max(verbs) + 1)}
            self.noun_map = {n: n for n in range(max(nouns) + 1)}
            self.action_map = pair_to_action
        else:
            self.verb_map, self.noun_map, self.action_map = self._closed_vocab(
                [frames[train_csv], frames[val_csv]])

        self.num_classes = {"verb": len(self.verb_map) + 1, "noun": len(self.noun_map) + 1,
                            "action": len(self.action_map) + 1}
        self.verb_noun_to_action = {(0, 0): 0}
        self.verb_noun_to_action.update({(self.verb_map[v] + 1, self.noun_map[n] + 1): a + 1
                                         for (v, n), a in self.action_map.items()})
        self.action_classes = {f"{v}_{n}": a for (v, n), a in self.action_map.items()}
        self.verb_classes = {str(k): i for k, i in self.verb_map.items()}
        self.noun_classes = {str(k): i for k, i in self.noun_map.items()}
        self.class_mappings = self._get_class_mappings()

        df = frames[train_csv if mode == "train" else val_csv].copy()
        df["action_class"] = [self.action_map.get((v, n), -1)
                              for v, n in zip(df["verb_class"], df["noun_class"])]
        n_bad = int((df["action_class"] < 0).sum())
        if n_bad:
            logger.warning("Dropping %d clips whose (verb, noun) is outside the class space", n_bad)
            df = df[df["action_class"] >= 0].reset_index(drop=True)
        n_official = len(df)
        df["_row"] = np.arange(len(df))  # official CSV order
        kept, discarded = convert_to_anticipation(df, self.tau_a, self.tau_o)
        kept = kept.copy()
        kept["observable"] = 1
        if mode != "train" and getattr(data, "SCORE_ALL_EVAL", True) and len(discarded):
            discarded = discarded.copy()
            discarded["observable"] = 0
            kept = pd.concat([kept, discarded], ignore_index=True)
        # Keep the official order so unobservable clips are not batched together.
        self.df = kept.sort_values("_row", kind="stable").reset_index(drop=True)
        self.discarded_df = discarded
        self.n_official = n_official
        logger.info("%s: %d official clips, %d dropped by the anticipation window, %d items (%s)",
                    mode, n_official, len(discarded), len(self.df),
                    "unobservable kept as misses" if len(self.df) == n_official else "dropped")

        self.segments_by_video = self._load_segments(label_csvs, frames)
        self.action_frequency = self._action_frequency(frames[train_csv])
        logger.info("Created ClipAnticipation (%s, %s) with %d samples, verb/noun/action=%d/%d/%d (incl. bg)",
                    self.fmt, mode, len(self.df), self.num_classes["verb"],
                    self.num_classes["noun"], self.num_classes["action"])

    @staticmethod
    def _closed_vocab(dfs):
        pairs = set()
        for df in dfs:
            pairs.update(zip(df["verb_class"].astype(int), df["noun_class"].astype(int)))
        verb_map = {v: i for i, v in enumerate(sorted({v for v, _ in pairs}))}
        noun_map = {n: i for i, n in enumerate(sorted({n for _, n in pairs}))}
        action_map = {a: i for i, a in enumerate(sorted(pairs))}
        return verb_map, noun_map, action_map

    def _load_segments(self, csvs, frames):
        seen = set()
        by_video = defaultdict(list)
        for path in csvs:
            df = frames[path]
            for video, t0, t1, v, n in zip(df["video_id"].astype(str), df["start"], df["end"],
                                           df["verb_class"], df["noun_class"]):
                key = (video, float(t0), float(t1), int(v), int(n))
                if key in seen or (int(v), int(n)) not in self.action_map:
                    continue
                seen.add(key)
                by_video[video].append((float(t0), float(t1), self.verb_map[int(v)] + 1,
                                        self.noun_map[int(n)] + 1,
                                        self.action_map[(int(v), int(n))] + 1))
        return by_video

    def _action_frequency(self, train_df) -> np.ndarray:
        counts = np.zeros(len(self.action_map), dtype=np.float64)
        for v, n in zip(train_df["verb_class"], train_df["noun_class"]):
            if (int(v), int(n)) in self.action_map:
                counts[self.action_map[(int(v), int(n))]] += 1.0
        return counts

    def _get_class_mappings(self):
        verb_in_action = torch.zeros((self.num_classes["action"], self.num_classes["verb"]))
        noun_in_action = torch.zeros((self.num_classes["action"], self.num_classes["noun"]))
        for (verb, noun), action in self.verb_noun_to_action.items():
            verb_in_action[action, verb] = 1.0
            noun_in_action[action, noun] = 1.0
        return {("verb", "action"): verb_in_action, ("noun", "action"): noun_in_action}

    def _labels_for_range(self, video_id: str, t_len: int, lo: int, hi: int):
        """Dense multi-hot labels for feature steps [lo, hi) only. Built per clip so
        large class spaces (EK100: 3,807) never hold whole-video label tensors."""
        n = hi - lo
        acts = torch.zeros((n, self.num_classes["action"]), dtype=torch.float32)
        verbs = torch.zeros((n, self.num_classes["verb"]), dtype=torch.float32)
        nouns = torch.zeros((n, self.num_classes["noun"]), dtype=torch.float32)
        segs = self._segment_arrays(video_id)
        if len(segs):
            i0 = np.clip(np.floor(segs[:, 0] * self.fps).astype(np.int64), 0, t_len)
            i1 = np.maximum(i0 + 1, np.clip(np.ceil(segs[:, 1] * self.fps).astype(np.int64), 0, t_len))
            overlap = np.flatnonzero((i0 < hi) & (i1 > lo))
        else:
            overlap = []
        for k in overlap:
            v, n_, a = (int(x) for x in segs[k, 2:5])
            j0, j1 = max(int(i0[k]), lo) - lo, min(int(i1[k]), hi) - lo
            if j0 < j1:
                acts[j0:j1, a] = 1.0
                verbs[j0:j1, v] = 1.0
                nouns[j0:j1, n_] = 1.0
        for tensor in (acts, verbs, nouns):
            tensor[tensor.sum(dim=-1) == 0, 0] = 1.0
        return acts, verbs, nouns

    def _segment_arrays(self, video_id: str) -> np.ndarray:
        """(t0, t1, verb, noun, action) rows of one video as a cached array."""
        cache = self.__dict__.setdefault("_seg_cache", {})
        if video_id not in cache:
            cache[video_id] = np.asarray(self.segments_by_video.get(video_id, []), dtype=np.float64).reshape(-1, 5)
        return cache[video_id]

    def _load_features(self, video_id: str):
        # Memory-mapped: DataLoader workers share the OS page cache instead of
        # each holding a private copy (EK100 features are about 5 GB).
        if video_id not in self.data_cache:
            self.data_cache[video_id] = np.load(self.feat_dir / f"{video_id}.npy", mmap_mode="r")
        return self.data_cache[video_id]

    def __len__(self):
        return len(self.df)

    def _frames_per_clip(self):
        return int(round((self.tau_o + self.tau_a) / self.past_step_in_sec))

    def _get_video(self, row):
        video_id = str(row["video_id"])
        feats = self._load_features(video_id)
        t_len = feats.shape[0]
        start_f = int(np.floor(max(float(row["start"]), 0.0) * self.fps))
        end_f = int(np.floor(max(float(row["orig_start"]), 0.0) * self.fps))
        start_f = max(0, min(start_f, t_len - 1))
        end_f = max(start_f, min(end_f, t_len - 1))
        clip = torch.from_numpy(np.array(feats[start_f:end_f + 1], dtype=np.float32))
        items = [clip, *self._labels_for_range(video_id, t_len, start_f, end_f + 1)]
        n = self._frames_per_clip()
        cur = items[0].size(0)
        if cur < n:  # pad with the first frame (upstream)
            items = [torch.cat([x[:1]] * (n - cur) + [x], dim=0) for x in items]
        elif cur > n:
            items = [x[-n:] for x in items]
        return items

    def _unobservable_item(self, row):
        """Zero features and background-only targets: excluded from every loss."""
        feats = self._load_features(str(row["video_id"]))
        n = self._frames_per_clip()
        video = torch.zeros((n, feats.shape[1]))
        labels = []
        for key in ("action", "verb", "noun"):
            target = torch.zeros((n, self.num_classes[key]))
            target[:, 0] = 1.0
            labels.append(target)
        return [video, *labels]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        observable = int(row["observable"])
        video, acts, verbs, nouns = self._get_video(row) if observable else self._unobservable_item(row)

        past_len = int(self.tau_o // self.past_step_in_sec)
        sample_rate = int(self.future_step_in_sec / self.past_step_in_sec)

        def split_fn(arr):
            if sample_rate == 1:
                return arr[:past_len], arr[past_len:]
            return arr[:past_len], arr[past_len:].flip(0)[::sample_rate].flip(0)

        past_feat, future_feat = split_fn(video)
        past_act, future_act = split_fn(acts)
        past_verb, future_verb = split_fn(verbs)
        past_noun, future_noun = split_fn(nouns)

        assert self.cfg.MODEL.CLS_WORK + self.cfg.MODEL.CLS_ALL + self.cfg.MODEL.CLS_LAST == 1
        if self.cfg.MODEL.CLS_WORK:
            loss_idx = int(self.long_memory_length // self.past_step_in_sec)
        elif self.cfg.MODEL.CLS_LAST:
            loss_idx = -1
        else:
            loss_idx = 0

        verb, noun = int(row["verb_class"]), int(row["noun_class"])
        return {
            "past_feats": past_feat,
            "past_act": past_act[loss_idx:],
            "past_verb": past_verb[loss_idx:],
            "past_noun": past_noun[loss_idx:],
            "future_feats": future_feat,
            "future_act": future_act,
            "future_verb": future_verb,
            "future_noun": future_noun,
            # Scoring uses the annotated action instance (see criterion._scored_labels).
            "scored_act": torch.tensor(int(row["action_class"]) + 1),
            "scored_verb": torch.tensor(self.verb_map[verb] + 1),
            "scored_noun": torch.tensor(self.noun_map[noun] + 1),
            "observable": torch.tensor(observable),
        }
