"""Diagnostic only: exercise pinned ScalAnt modules with one EGTEA instance."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .egtea_cache import EGTEAInstanceCache, ROOT


UPSTREAM = ROOT / "refer_repo/scalable-anticipation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--population", choices=("training", "validation"), default="training")
    parser.add_argument("--encoder", choices=("MAMBA", "GRU"), default="MAMBA")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tiny", action="store_true", help="Reduced model for interface diagnosis only")
    parser.add_argument("--verify-file-hash", action="store_true")
    parser.add_argument("--minimum-seconds", type=float, default=0.0,
                        help="Repeat forward/backward for monitored GPU smoke")
    parser.add_argument("--out-of-place-pos", action="store_true",
                        help="Diagnostic monkeypatch for upstream inplace positional adds")
    parser.add_argument("--clone-layer-inputs", action="store_true",
                        help="Preserve original forward additions on new storage per layer")
    parser.add_argument("--compare-original-forward", action="store_true",
                        help="Compare fixed and original train-mode forward with identical RNG")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = EGTEAInstanceCache(args.split, args.horizon, args.population,
                              verify_file_hash=args.verify_file_hash)
    # The original package is imported from the pinned submodule. No method
    # implementation is copied or edited in this worktree.
    sys.path.insert(0, str(UPSTREAM))
    from scalant.config import load_config
    from scalant.models import QueryPredictor
    if args.out_of_place_pos and args.clone_layer_inputs:
        raise ValueError("Choose only one positional compatibility trial")
    if args.out_of_place_pos:
        from scalant.models.clam import GLADecoderLayer
        from scalant.models.decoder import TransformerDecoderLayer

        def out_of_place_pos(self, tensor, pos, pos2=None):
            result = tensor
            if pos is not None:
                result = result + pos
            if pos2 is not None:
                result = result + pos2
            return result

        GLADecoderLayer.with_pos_embed = out_of_place_pos
        TransformerDecoderLayer.with_pos_embed = out_of_place_pos

    cfg = load_config(str(UPSTREAM / "configs/ek100/default.yaml"))
    cfg.DATA.TAU_O = 4.0
    cfg.DATA.TAU_A = float(args.horizon)
    cfg.DATA.LONG_MEMORY_LENGTH = 3.0
    cfg.MODEL.ENCODER_CLASS = args.encoder
    cfg.MODEL.INPUT_DIM = 1024
    cfg.MODEL.IGNORE_INDEX = -1
    cfg.MODEL.N_QUERIES = 1
    cfg.MODEL.ACTION_CLS = True
    cfg.MODEL.VERB_CLS = True
    cfg.MODEL.NOUN_CLS = True
    if args.tiny:
        cfg.MODEL.N_LAYER = 1
        cfg.MODEL.N_DEC_LAYER = 1
        cfg.MODEL.D_MODEL = 128
        cfg.MODEL.D_FFN = 512
        cfg.MODEL.N_HEADS = 4
        cfg.CLUSTERING.N_CLUSTERS = 2
    model = QueryPredictor(cfg, {"action": 106, "verb": 19, "noun": 53}, data)
    model.to(args.device).train()
    sample = data[0]
    features = sample["past_feats"].unsqueeze(0).to(args.device)
    parity = None
    if args.clone_layer_inputs:
        if args.compare_original_forward:
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if args.device == "cuda" else None
            with torch.no_grad():
                original = model(features)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
        from .legacy_autograd_fix import install_clone_per_layer_fix
        install_clone_per_layer_fix()
        if args.compare_original_forward:
            with torch.no_grad():
                fixed = model(features)
            parity = {
                name: float((getattr(original, name) - getattr(fixed, name)).abs().max())
                for name in ("past_actions", "past_verbs", "past_nouns",
                             "future_actions", "future_verbs", "future_nouns",
                             "clusters")
            }
            if any(delta > 1e-5 for delta in parity.values()):
                raise ValueError(f"Original/fixed train-mode forward mismatch: {parity}")
    start = time.monotonic()
    steps = 0
    while True:
        model.zero_grad(set_to_none=True)
        output = model(features)
        if output.future_actions.shape != (1, 1, 106):
            raise ValueError(f"Action output shape {tuple(output.future_actions.shape)}")
        if output.future_verbs.shape != (1, 1, 19) or output.future_nouns.shape != (1, 1, 53):
            raise ValueError("Verb or noun output shape differs from EGTEA class spaces")
        # This one-term loss checks differentiability. It is not the proposed
        # training objective; past-frame supervision is still a design decision.
        loss = F.cross_entropy(output.future_actions[:, -1],
                               sample["action"].view(1).to(args.device))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite diagnostic loss")
        loss.backward()
        if args.device == "cuda":
            torch.cuda.synchronize()
        steps += 1
        if time.monotonic() - start >= args.minimum_seconds:
            break
    active = [name for name, param in model.named_parameters()
              if param.grad is not None and torch.isfinite(param.grad).all()
              and param.grad.abs().sum().item() > 0]
    if not any(name.startswith("encoder.") for name in active) or not any(name.startswith("clustering.") for name in active):
        raise ValueError("Encoder or CLAM has no nonzero diagnostic gradient")
    report = {"kind": "engineering_probe_not_training", "upstream": str(UPSTREAM),
              "split": args.split, "horizon": args.horizon, "population": args.population,
              "population_size": len(data), "uid": int(sample["uid"]),
              "input_shape": list(features.shape), "encoder": args.encoder,
              "tiny": args.tiny, "device": args.device,
              "out_of_place_pos_diagnostic": args.out_of_place_pos,
              "clone_layer_inputs": args.clone_layer_inputs,
              "original_forward_max_abs_diff": parity,
              "steps": steps, "compute_seconds": time.monotonic() - start,
              "action_shape": list(output.future_actions.shape),
              "verb_shape": list(output.future_verbs.shape),
              "noun_shape": list(output.future_nouns.shape),
              "loss": float(loss.detach()), "nonzero_finite_gradient_tensors": len(active)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
