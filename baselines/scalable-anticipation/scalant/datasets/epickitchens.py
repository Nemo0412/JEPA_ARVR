"""
Implementation based on AFFT (WACV 2023) and AVT (ICCV 2021)
"""
import os

# import lmdb
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd
from typing import List, Dict, Sequence, Tuple, Union
from datetime import datetime, date
import pickle as pkl
import csv
from pathlib import Path

from scalant.utils import logging
from scalant.config import Config
from scalant.datasets.build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


EGTEA_VERSION = -1  # This class also supports EGTEA Gaze+
EPIC55_VERSION = 0.1
EPIC100_VERSION = 0.2
# This is specific to EPIC kitchens
RULSTM_TSN_FPS = 30.0  # the frame rate the feats were stored by RULSTM


def convert_to_anticipation(
        df: pd.DataFrame,
        tau_a: float = 1,
        tau_o: float = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug(
        'Converting data to anticipation with tau_a=%s and '
        'tau_o=%s.', tau_a, tau_o)
    # Copy over the current start and end times
    df.loc[:, 'orig_start'] = df.start
    df.loc[:, 'orig_end'] = df.end
    # Convert using tau_o and tau_a
    df.loc[:, 'end'] = df.loc[:, 'start'] - tau_a
    df.loc[:, 'start'] = df.loc[:, 'end'] - tau_o

    # first frame seconds
    f1_sec = 1 / RULSTM_TSN_FPS
    old_df = df
    # at least 1 frame
    df = df[df.end >= f1_sec]
    discarded_df = pd.concat([old_df, df]).drop_duplicates(subset=['uid'], keep=False)
    df.reset_index(inplace=True, drop=True)
    return df, discarded_df


class EpicNumpyFeatsReader:
    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        self.fps = 4

    def __call__(self, video_path, frames, data_cache=None):
        video_name = Path(video_path).stem

        if video_name not in data_cache:
            feats = torch.from_numpy(
                np.load(os.path.join(self.feat_dir, f"{video_name}.npy"))).float()
            feat_root_dir = os.path.dirname(self.feat_dir)
            acts = torch.from_numpy(
                np.load(os.path.join(feat_root_dir, "target_perframe", f"{video_name}.npy"))).float()
            verbs = torch.from_numpy(
                np.load(os.path.join(feat_root_dir, "verb_perframe", f"{video_name}.npy"))).float()
            nouns = torch.from_numpy(
                np.load(os.path.join(feat_root_dir, "noun_perframe", f"{video_name}.npy"))).float()
            data_cache[video_name] = [feats, acts, verbs, nouns]
        else:
            feats, acts, verbs, nouns = data_cache[video_name]

        return [el[frames[0]:frames[1]+1] for el in [feats, acts, verbs, nouns]]


@DATASET_REGISTRY.register()
class EpicKitchens(torch.utils.data.Dataset):
    def __init__(self, cfg: Config, mode: str):
        assert mode in ["train", "test", "val"]
        self.cfg = cfg
        suffix = ("validation" if mode == "val"
                  else "test_timestamps" if mode == "test" else "train")
        annotation_path = f"annotations/ek100_ori/EPIC_100_{suffix}.pkl"

        action_labels_fpath = "annotations/ek100_rulstm/actions.csv"
        annotation_dir = "annotations/ek100_ori"
        version = EPIC100_VERSION

        self.feat_dir = os.path.join(cfg.DATA.DATA_ROOT_PATH, cfg.DATA.FEAT_DIR)

        # Save the data in shared dictionary, allowing faster i/o
        self.data_cache = {}  #Manager().dict()  # shared dict


        self.version = version
        self.annotation_dir = Path(annotation_dir)
        self.tau_a = cfg.DATA.TAU_A  # ahead of true action, in seconds
        self.tau_o = cfg.DATA.TAU_O  # length of observation, in seconds
        self.past_step_in_sec = cfg.DATA.PAST_STEP_IN_SEC
        self.future_step_in_sec = cfg.DATA.FUTURE_STEP_IN_SEC
        self.long_memory_length = cfg.DATA.LONG_MEMORY_LENGTH

        # Load annotations
        df = self.load_df(annotation_path)
        self.df = df

        # Load verb and noun classes
        epic_postfix = ''
        if self.version == EPIC100_VERSION:
            epic_postfix = '_100'
        if self.version != EGTEA_VERSION:
            verb_classes = self._load_class_names(
                self.annotation_dir / f'EPIC{epic_postfix}_verb_classes.csv')
            noun_classes = self._load_class_names(
                self.annotation_dir / f'EPIC{epic_postfix}_noun_classes.csv')
        else:
            verb_classes, noun_classes = [], []

        # Create action classes
        load_action_fn = (
            self._load_action_classes if self.version != EGTEA_VERSION
            else self._load_action_classes_egtea)

        action_classes, verb_noun_to_action = load_action_fn(action_labels_fpath)

        # QUICK FIX for background class 0
        self.verb_noun_to_action = {(0, 0): 0}
        self.verb_noun_to_action.update({
            (k1 + 1, k2 + 1): val + 1 for (k1, k2), val in verb_noun_to_action.items()
        })

        self.action_classes = action_classes
        self.verb_classes = verb_classes
        self.noun_classes = noun_classes
        # Include background as in Testra (ECCV22) and MAT (ICCV23)
        self.num_classes = {
            "action": len(self.action_classes) + 1,
            "verb": len(self.verb_classes) + 1,
            "noun": len(self.noun_classes) + 1,
        }

        self.class_mappings = self._get_class_mappings()

        # Add the action classes to the data frame
        if 'action_class' not in df.columns and {'noun_class', 'verb_class'}.issubset(df.columns):
            df.loc[:, 'action_class'] = df.loc[:, ('verb_class', 'noun_class')].apply(
                lambda row: (verb_noun_to_action[(row.at['verb_class'], row.at['noun_class'])]
                             if (row.at['verb_class'], row.at['noun_class']) in verb_noun_to_action else -1), axis=1)
        elif 'action_class' not in df.columns:
            df.loc[:, 'action_class'] = -1
            df.loc[:, 'verb_class'] = -1
            df.loc[:, 'noun_class'] = -1
        num_undefined_actions = len(df[df['action_class'] == -1].index)
        if num_undefined_actions > 0:
            logger.error(f'Did not found valid action label for {num_undefined_actions}/{len(df)} samples!')

        # To be consistent with EPIC, add a uid column if not already present
        if 'uid' not in self.df.columns:
            self.df.loc[:, 'uid'] = range(1, len(self.df) + 1)

        # Convert df to anticipative df
        self.df, self.discarded_df = convert_to_anticipation(df, self.tau_a, self.tau_o)
        logger.info(f'Discarded {len(self.discarded_df)} elements in anticipate conversion')
        logger.info(f'Created EPIC {self.version} dataset with {len(self)} samples')
        
    def load_df(self, annotation_path):
        """Loading the original EPIC Kitchens annotations"""

        def timestr_to_sec(s, fmt='%H:%M:%S.%f'):
            # Convert timestr to seconds
            timeobj = datetime.strptime(s, fmt).time()
            td = datetime.combine(date.min, timeobj) - datetime.min
            return td.total_seconds()

        # Load the DF from annot path
        logger.info(f'Loading original EPIC pkl annotations {annotation_path}')
        with open(annotation_path, 'rb') as fin:
            df = pkl.load(fin)
        # Make a copy of the UID column, since that will be needed to gen output files
        df.reset_index(drop=False, inplace=True)

        # parse timestamps from the video
        df.loc[:, 'start'] = df.start_timestamp.apply(timestr_to_sec)
        df.loc[:, 'end'] = df.stop_timestamp.apply(timestr_to_sec)

        # original annotations have text in weird format - fix that
        if 'noun' in df.columns:
            df.loc[:, 'noun'] = df.loc[:, 'noun'].apply(
                lambda s: ' '.join(s.replace(':', ' ').split(sep=' ')[::-1]))
        if 'verb' in df.columns:
            df.loc[:, 'verb'] = df.loc[:, 'verb'].apply(
                lambda s: ' '.join(s.replace('-', ' ').split(sep=' ')))

        if self.version == EGTEA_VERSION:
            df.loc[:, 'video_path'] = df.apply(lambda x: Path(x.video_id + '.mp4'), axis=1)
        else:
            df.loc[:, 'video_path'] = df.apply(lambda x: (Path(x.participant_id) / Path(x.video_id + '.MP4')), axis=1)

        df.reset_index(inplace=True, drop=True)
        return df

    def _load_class_names(self, annot_path: Path):
        res = {}
        with open(annot_path, 'r') as fin:
            reader = csv.DictReader(fin, delimiter=',')
            for lno, line in enumerate(reader):
                res[line['class_key' if self.version == EPIC55_VERSION else 'key']] = lno
        return res

    @staticmethod
    def _load_action_classes(action_labels_fpath: Path) -> Tuple[Dict[str, int], Dict[Tuple[int, int], int]]:
        """
        Given a CSV file with the actions (as from RULSTM paper), construct the set of actions and mapping from verb/noun to action
        Args:
            action_labels_fpath: path to the file
        Returns:
            class_names: Dict of action class names
            verb_noun_to_action: Mapping from verb/noun to action IDs
        """
        class_names = {}
        verb_noun_to_action = {}
        with open(action_labels_fpath, 'r') as fin:
            reader = csv.DictReader(fin, delimiter=',')
            for lno, line in enumerate(reader):
                class_names[line['action']] = lno
                verb_noun_to_action[(int(line['verb']), int(line['noun']))] = int(line['id'])
        return class_names, verb_noun_to_action

    @staticmethod
    def _load_action_classes_egtea(action_labels_fpath: Path) -> Tuple[Dict[str, int], Dict[Tuple[int, int], int]]:
        """
        Given a CSV file with the actions (as from RULSTM paper), construct the set of actions and mapping from verb/noun to action
        Args:
            action_labels_fpath: path to the file
        Returns:
            class_names: Dict of action class names
            verb_noun_to_action: Mapping from verb/noun to action IDs
        """
        class_names = {}
        verb_noun_to_action = {}
        with open(action_labels_fpath, 'r') as fin:
            reader = csv.DictReader(
                fin, delimiter=',', fieldnames=['id', 'verb_noun', 'action'])
            for lno, line in enumerate(reader):
                class_names[line['action']] = lno
                verb, noun = [int(el) for el in line['verb_noun'].split('_')]
                verb_noun_to_action[(verb, noun)] = int(line['id'])
        return class_names, verb_noun_to_action

    def _get_class_mappings(self) -> Dict[Tuple[str, str], torch.Tensor]:
        num_verbs = self.num_classes["verb"]
        num_nouns = self.num_classes["noun"]
        num_actions = self.num_classes["action"]
        verb_in_action = torch.zeros((num_actions, num_verbs), dtype=torch.float)
        noun_in_action = torch.zeros((num_actions, num_nouns), dtype=torch.float)
        for (verb, noun), action in self.verb_noun_to_action.items():
            verb_in_action[action, verb] = 1.0
            noun_in_action[action, noun] = 1.0
        return {
            ('verb', 'action'): verb_in_action,
            ('noun', 'action'): noun_in_action
        }

    def __len__(self):
        return len(self.df)

    def _get_video(self, df_row):
        if not hasattr(self, 'reader_fn'):
            self.reader_fn = EpicNumpyFeatsReader(feat_dir=self.feat_dir)

        # get video segments of tau_o + tau_a
        video_path = df_row['video_path']
        start = max(df_row['start'], 0)
        end = max(df_row['orig_start'], 0)
        fps = self.reader_fn.fps
        req_fps = 1 / self.past_step_in_sec

        # Select frames to read
        start_f = np.floor(start * fps).astype(int)
        end_f = np.floor(end * fps).astype(int)

        video, acts, verbs, nouns = self.reader_fn(video_path, (start_f, end_f), self.data_cache)

        if fps != req_fps:
            frames_to_keep = range(len(video))[::-max(int(round(fps / req_fps)), 1)][::-1]
            video = video[frames_to_keep]
            acts = acts[frames_to_keep]
            verbs = verbs[frames_to_keep]
            nouns = nouns[frames_to_keep]

        frames_per_clip = int(round((self.tau_o + self.tau_a) / self.past_step_in_sec))

        def adjust_tensor(*tensors):
            cur_len = next(iter(tensors)).size(0)

            # Pad the video with the first frame, or crop out the extra frames
            if cur_len < frames_per_clip:
                npad = frames_per_clip - cur_len
                def padding_fn(T, npad):
                    return torch.cat([T[:1]] * npad + [T], dim=0)
                tensors = [padding_fn(tensor, npad) for tensor in tensors]
            elif cur_len > frames_per_clip:
                tensors = [tensor[-frames_per_clip:] for tensor in tensors]
            return tensors

        video, acts, verbs, nouns = adjust_tensor(video, acts, verbs, nouns)

        return video, acts, verbs, nouns

    def __getitem__(self, idx):
        df_row = self.df.loc[idx, :]
        video, action_label, verb_label, noun_label = self._get_video(df_row)

        # Now to split the whole video segments into
        # past (input for past summarization model),
        # future (input for future prediction model)
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

        # We only calculate loss on the working memory,
        # not on the long-term memory as done in Testra and MAT
        assert self.cfg.MODEL.CLS_WORK + self.cfg.MODEL.CLS_ALL + self.cfg.MODEL.CLS_LAST == 1
        if self.cfg.MODEL.CLS_WORK:
            loss_idx = int(self.long_memory_length // self.past_step_in_sec)
        elif self.cfg.MODEL.CLS_LAST:
            loss_idx = -1
        elif self.cfg.MODEL.CLS_ALL:
            loss_idx = 0
        else:
            raise NotImplementedError

        item = {
            'past_feats': past_feat,
            'past_act': past_act[loss_idx:],
            'past_verb': past_verb[loss_idx:],
            'past_noun': past_noun[loss_idx:],
            'future_feats': future_feat,
            'future_act': future_act,
            'future_verb': future_verb,
            'future_noun': future_noun,
        }

        return item
