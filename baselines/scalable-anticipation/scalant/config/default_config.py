from dataclasses import dataclass, field

import torch
from nestconfig import NestConfig
from typing import Union


@dataclass
class DataConfig:
    DATA_ROOT_PATH = "~/Documents/datasets"  # path where all datasets are saved
    DATASET_CLASS: str = "EpicKitchens"
    FEAT_DIR = "epickitchens100/features/rgb"
    DROP_LAST: bool = False
    TRAIN_CSV: str = ""
    VAL_CSV: str = ""
    FULL_CSV: str = ""

    # Short-term specific
    TAU_A: float = 1.  # ahead of true action, in seconds
    TAU_O: float = 69.  # length of observation, in seconds (working mem + long mem)
    PAST_STEP_IN_SEC: float = 0.25
    PAST_SAMPLE_RATE: int = 1
    FUTURE_STEP_IN_SEC: float = 1.
    FUTURE_SAMPLE_RATE: int = 4  # step_in_sec * feature_fps
    LONG_MEMORY_LENGTH: float = 64.  # secs of long-term memory, if fps is 4 -> 256 sequence length

    # For thumos or online action detection
    RGB_TYPE: str = 'rgb_kinetics_resnet50'
    FLOW_TYPE: str = 'flow_kinetics_bninception'
    STRIDE: float = 1.  # stride in seconds
    DATA_NAME: str = 'THUMOS'
    NUM_CLASSES: int = 22  # including background


@dataclass
class ModelConfig:
    ENCODER_CLASS: str = "MAMBA"
    MAMBA_VERSION: int = 1
    CRITERION_CLASS: str = "Criterion"
    INPUT_DIM: int = 1024
    D_MODEL: int = 512
    N_LAYER: int = 2
    N_DEC_LAYER: int = 2
    IGNORE_INDEX: int = -1  # class that does not contribute to the loss
    D_FFN: int = 2048
    N_HEADS: int = 8

    SHARE_CLASSIFIER: bool = False
    PAST_CLS: bool = True

    # Action, Verb, Noun Classification
    ACTION_CLS: bool = True
    VERB_CLS: bool = False
    NOUN_CLS: bool = False

    # Past classification
    CLS_WORK: bool = True
    CLS_LAST: bool = False
    CLS_ALL: bool = False

    DROPOUT: float = 0.
    DROP_CLS: float = 0.
    DROP_DEC: float = 0.1

    # MAMBA
    D_STATE: int = 64
    D_CONV: int = 4
    RETURN_INTERMEDIATE: bool = False
    INTERMEDIATE_LAYER_IDX: int = -1

    # For querydecoder
    PRENORM: bool = False
    N_QUERIES: int = 1

    # activation
    ACTIVATION: str = 'relu'

    # Uniformly sample from the past instead of last tokens
    SAMPLE_UNIFORM: bool = False
    # Frequency-equalized action loss for non-EpicKitchens datasets that expose
    # `action_frequency` (EpicKitchens always uses it, as upstream).
    EQUALIZED_LOSS: bool = False


@dataclass
class ClusteringConfig:
    ENABLE: bool = False
    ON_FRAME_TOKENS: bool = True
    N_CLUSTERS: int = 10
    USE_EMBEDDING: bool = True
    CAT_WORK_LAST: bool = False
    GATE_STATE: bool = True
    EXPAND_K: float = 0.5
    USE_SCAN: bool = False
    LAYERS: int = 1


@dataclass
class TrainConfig:
    ENABLE: bool = True
    CKPT_PATH: str = None
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 8
    OPTIMIZER: str = "sgd"
    WEIGHT_DECAY: float = 0.
    SCHEDULER: str = "cosine"
    EPOCHS: int = 50
    WARMUP_STEPS: int = 5
    LR: float = 0.001
    MIN_LR: float = 1e-7
    GRADIENT_CLIPPING: Union[float, None] = None
    USE_MIXUP: bool = False
    SAVE_MODEL: bool = True


@dataclass
class ValConfig:
    ENABLE: bool = True
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 8
    EVALUATE_EVERY: int = 1
    MAX_LEN: int = 1


@dataclass
class TestConfig:
    ENABLE: bool = False
    CKPT_PATH: str = None


@dataclass
class Config(NestConfig):
    SEED = 42
    PRIMARY_METRIC = "val/mt5r"
    NOTE = None   # some notes of the experiment
    USE_WANDB = False  # whether to use wandb to visualize logs
    LOG_LEVEL = 'info'  # info or debug
    WANDB_PROJECT = None
    METRIC_DESCENDING: bool = False
    DTYPE: str = "float32"

    MODEL: ModelConfig = field(default_factory=ModelConfig)
    TRAIN: TrainConfig = field(default_factory=TrainConfig)
    VAL: ValConfig = field(default_factory=ValConfig)
    TEST: TestConfig = field(default_factory=TestConfig)
    DATA: DataConfig = field(default_factory=DataConfig)
    CLUSTERING: ClusteringConfig = field(default_factory=ClusteringConfig)
