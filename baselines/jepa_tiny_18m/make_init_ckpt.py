#!/usr/bin/env python3
"""Write a random-init V-JEPA checkpoint compatible with init_module loaders.

Creates keys ``target_encoder`` / ``predictor`` matching the requested architecture.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoder-model", default="vit_tiny")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-frames", type=int, default=80)
    ap.add_argument("--predictor-embed-dim", type=int, default=320)
    ap.add_argument("--predictor-depth", type=int, default=10)
    ap.add_argument("--predictor-heads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    import sys
    import os

    vjepa = os.environ.get("VJEPA_ROOT", "/home/ll5914/ARVR_Video/vjepa2")
    sys.path.insert(0, vjepa)
    import src.models.vision_transformer as vit
    import src.models.predictor as vit_pred

    enc_kwargs = dict(
        model_name=args.encoder_model,
        checkpoint_key="target_encoder",
        tubelet_size=2,
        patch_size=16,
        uniform_power=True,
        use_rope=True,
    )
    encoder = vit.__dict__[args.encoder_model](
        img_size=args.img_size, num_frames=args.num_frames, **enc_kwargs
    )
    predictor = vit_pred.vit_predictor(
        img_size=args.img_size,
        embed_dim=encoder.embed_dim,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        num_frames=64,
        depth=args.predictor_depth,
        num_heads=args.predictor_heads,
        predictor_embed_dim=args.predictor_embed_dim,
        num_mask_tokens=10,
        uniform_power=True,
        use_mask_tokens=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=False,
        use_rope=True,
    )

    enc_n = sum(p.numel() for p in encoder.parameters())
    pred_n = sum(p.numel() for p in predictor.parameters())
    print(f"encoder={args.encoder_model} {enc_n/1e6:.2f}M")
    print(f"predictor dim={args.predictor_embed_dim} L={args.predictor_depth} {pred_n/1e6:.2f}M")
    print(f"enc+pred={(enc_n+pred_n)/1e6:.2f}M")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "meta": {
                "encoder_model": args.encoder_model,
                "predictor_embed_dim": args.predictor_embed_dim,
                "predictor_depth": args.predictor_depth,
                "predictor_heads": args.predictor_heads,
                "encoder_m": enc_n / 1e6,
                "predictor_m": pred_n / 1e6,
                "enc_plus_pred_m": (enc_n + pred_n) / 1e6,
            },
        },
        args.out,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
