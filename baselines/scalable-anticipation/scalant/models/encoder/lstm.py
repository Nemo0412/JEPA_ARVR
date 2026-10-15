import torch
import torch.nn as nn

from scalant.config import Config
from scalant.models.encoder.build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class LSTM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = nn.LSTM(
            cfg.MODEL.D_MODEL, cfg.MODEL.D_MODEL, cfg.MODEL.N_LAYER,
            batch_first=True, dropout=cfg.MODEL.DROPOUT,
        )

    def forward(self, past):
        B, T, _ = past.size()
        past, _ = self.encoder(past)

        return past

