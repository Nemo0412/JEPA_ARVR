"""CPU parity and backward check for ScalAnt's released query decoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from .egtea_cache import MAIN_ROOT, ROOT


def main() -> None:
    sys.path.insert(0, str(ROOT / "refer_repo/scalable-anticipation"))
    from scalant.config import load_config
    from scalant.models.decoder import QueryDecoder

    cfg = load_config(str(ROOT / "refer_repo/scalable-anticipation/configs/ek100/default.yaml"))
    cfg.MODEL.D_MODEL = 128
    cfg.MODEL.D_FFN = 512
    cfg.MODEL.N_HEADS = 4
    cfg.MODEL.N_DEC_LAYER = 4
    cfg.MODEL.DROPOUT = 0.1
    cfg.CLUSTERING.ENABLE = True
    torch.manual_seed(731)
    decoder = QueryDecoder(cfg).train()
    memory = torch.randn(2, 14, 128, requires_grad=True)
    pos = torch.randn(2, 14, 128, requires_grad=True)
    rng = torch.get_rng_state()
    with torch.no_grad():
        original = decoder(memory.clone(), memory_pos=pos)
    torch.set_rng_state(rng)
    from .legacy_autograd_fix import install_clone_per_layer_fix
    install_clone_per_layer_fix()
    with torch.no_grad():
        fixed = decoder(memory.clone(), memory_pos=pos)
    max_abs_diff = float((original - fixed).abs().max())
    if max_abs_diff > 1e-5:
        raise ValueError(f"Original/clone-fix decoder forward mismatch: {max_abs_diff}")
    out = decoder(memory.clone(), memory_pos=pos)
    out.square().mean().backward()
    if memory.grad is None or pos.grad is None:
        raise ValueError("Missing decoder input or positional gradient")
    if not torch.isfinite(memory.grad).all() or not torch.isfinite(pos.grad).all():
        raise ValueError("Nonfinite decoder input or positional gradient")
    report = {
        "kind": "decoder_cpu_engineering_check_not_training",
        "shape": list(out.shape),
        "train_mode_original_forward_max_abs_diff": max_abs_diff,
        "memory_grad_norm": float(memory.grad.norm()),
        "position_grad_norm": float(pos.grad.norm()),
    }
    path = MAIN_ROOT / "outputs/b19_scalant/implementation/decoder_clone_fix_cpu.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
