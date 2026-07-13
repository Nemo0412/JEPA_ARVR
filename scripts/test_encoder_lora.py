#!/usr/bin/env python
"""Smoke tests for project-local encoder LoRA injection."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.hdepic_lora_action_anticipation.encoder_lora import (
    assert_encoder_lora_device_consistency,
    inject_encoder_lora,
    set_encoder_lora_trainable,
    trainable_encoder_lora_params,
)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(4, 12)
        self.attn.proj = nn.Linear(4, 4)
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock(), TinyBlock()])
        self.embed_dim = 4


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()


def _is_encoder_lora(module: nn.Module) -> bool:
    return bool(getattr(module, "_is_encoder_lora", False))


def test_injects_last_n_blocks_and_preserves_device_dtype():
    model = TinyModel().to(dtype=torch.float64)
    replaced = inject_encoder_lora(model, rank=2, alpha=4.0, dropout=0.0, last_n_blocks=2)

    assert replaced == 8
    assert not any(_is_encoder_lora(m) for m in model.encoder.blocks[0].modules())
    assert sum(_is_encoder_lora(m) for m in model.encoder.blocks[1:].modules()) == 8
    assert_encoder_lora_device_consistency(model)

    for module in model.modules():
        if _is_encoder_lora(module):
            assert module.lora_A.weight.device == module.base.weight.device
            assert module.lora_B.weight.device == module.base.weight.device
            assert module.lora_A.weight.dtype == module.base.weight.dtype
            assert module.lora_B.weight.dtype == module.base.weight.dtype


def test_trainable_restore_after_freeze():
    model = TinyModel()
    inject_encoder_lora(model, rank=2, alpha=4.0, dropout=0.0, last_n_blocks=1)
    for param in model.parameters():
        param.requires_grad = False

    restored = set_encoder_lora_trainable(model, trainable=True)
    params = trainable_encoder_lora_params(model)

    assert restored == sum(p.numel() for p in params)
    assert restored > 0
    assert all(p.requires_grad for p in params)


if __name__ == "__main__":
    test_injects_last_n_blocks_and_preserves_device_dtype()
    test_trainable_restore_after_freeze()
    print("encoder_lora smoke tests passed")
