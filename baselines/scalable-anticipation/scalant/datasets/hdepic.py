"""HD-EPIC P01 clip dataset for ScalAnt (kept for existing configs).

Thin alias of `ClipAnticipation` with the HD-EPIC annotation format and a
closed train+val vocabulary. Configs: configs/hdepic/clip_4s_2s.yaml (old 4s
LSTM), configs/hdepic/clip_32s_2s.yaml (v2).
"""
from scalant.datasets.build import DATASET_REGISTRY
from scalant.datasets.clip_anticipation import ClipAnticipation


@DATASET_REGISTRY.register()
class HDEPICClip(ClipAnticipation):
    def __init__(self, cfg, mode: str):
        cfg.DATA.ANNOTATION_FORMAT = "hdepic"
        super().__init__(cfg, mode)
