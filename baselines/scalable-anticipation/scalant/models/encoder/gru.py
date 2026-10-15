import torch
import torch.nn as nn

from scalant.config import Config
from scalant.models.encoder.build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class GRU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.num_layers = cfg.MODEL.N_LAYER
        self.hidden_dim = cfg.MODEL.D_MODEL
        self.encoder = nn.GRU(
            cfg.MODEL.D_MODEL, cfg.MODEL.D_MODEL, cfg.MODEL.N_LAYER,
            batch_first=True, dropout=cfg.MODEL.DROPOUT
        )

    def forward(self, past):
        B, _, _ = past.shape
        past, _ = self.encoder(past)

        return past