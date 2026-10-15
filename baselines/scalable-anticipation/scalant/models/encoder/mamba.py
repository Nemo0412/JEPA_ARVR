import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm, rms_norm_fn
from functools import partial

from scalant.config import Config
from scalant.models.encoder.build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class MAMBA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        mamba_cfg = {'d_state': cfg.MODEL.D_STATE, 'd_conv': cfg.MODEL.D_CONV}
        self.return_intermediate = cfg.MODEL.RETURN_INTERMEDIATE
        self.layers = nn.ModuleList(
            [
                create_block(
                    cfg.MODEL.D_MODEL,
                    d_intermediate=0,
                    ssm_cfg=mamba_cfg,
                    rms_norm=True,
                    fused_add_norm=True,
                    layer_idx=i,
                )
                for i in range(cfg.MODEL.N_LAYER)
            ]
        )

        self.norm_f = RMSNorm(cfg.MODEL.D_MODEL)
        self.apply(partial(_init_weights, n_layer=cfg.MODEL.N_LAYER))

    def forward(self, hidden_states, inference_params=None):
        residual = None
        all_res = []
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if self.return_intermediate:
                all_res.append((hidden_states, residual))

        fused_add_norm_fn = rms_norm_fn
        if not self.return_intermediate:
            # Set prenorm=False here since we don't need the residual
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
            )
            return hidden_states

        return [
            fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
            )
            for hidden_states, residual in all_res
        ]
