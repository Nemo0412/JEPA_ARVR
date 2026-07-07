#!/usr/bin/env python
"""Offline calibration for loss-aware token pruning with JEPA-loss gradients.

Calibration pass (no pruning):
  1. Forward encoder + predictor, backward JEPA loss
  2. Collect grad-based token scores at selected layers (full N tokens)
  3. Average scores across calibration samples -> per-position mean importance
  4. Allocate cascade conditional prune ratios r_l with product_l(1-r_l) = 1 - R_total
  5. Save JSON config + .scores.pt artifact for runtime cache build
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

SHARED = os.environ.get("SHARED_PROJECT_ROOT", "/path/to/VJEPA2-EXP")
CODE_ROOT = os.environ.get("PROJECT_ROOT", SHARED)
SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE_ROOT, "vjepa2"))
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, SCRIPT_ROOT)

from loss_aware_pruning import (  # noqa: E402
    LossAwarePruningConfig,
    LossAwareScoreCollector,
    accumulate_position_scores,
    finalize_calibration_config,
    resolve_prune_layers,
    save_calibration_artifacts,
    simulate_cascade_keep_counts,
    verify_cascade_global_ratio,
)
from models.vit_encoder_pruning import forward_encoder_with_hooks  # noqa: E402


def jepa_loss(z, h, masks_pred, loss_exp: float = 1.0) -> torch.Tensor:
    from src.masks.utils import apply_masks

    h_masked = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
    loss, n = 0.0, 0
    for zi, hi in zip(z, h_masked):
        for zij, hij in zip(zi, hi):
            loss = loss + torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
            n += 1
    return loss / max(n, 1)


@torch.enable_grad()
def calibrate_batch_jepa(
    encoder,
    target_encoder,
    predictor,
    clips,
    masks_enc,
    masks_pred,
    prune_layers: list[int],
    *,
    loss_exp: float = 1.0,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    collector = LossAwareScoreCollector(prune_layers)

    with torch.no_grad():
        h = target_encoder(clips)
        h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]

    z_enc, _, _ = forward_encoder_with_hooks(
        encoder,
        clips,
        masks=masks_enc,
        score_collector=collector,
    )
    z = predictor(z_enc, masks_enc, masks_pred)
    loss = jepa_loss(z, h, masks_pred, loss_exp=loss_exp)
    encoder.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    loss.backward()

    collector.finalize()
    return collector.scores, collector.sensitivities


def _synthetic_jepa_calibration_batch(
    num_tokens: int,
    embed_dim: int,
    prune_layers: list[int],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Gradient-shaped synthetic batch for smoke tests without a real checkpoint."""
    from loss_aware_pruning import compute_token_importance, layer_sensitivity

    z = torch.randn(2, num_tokens, embed_dim, device=device, requires_grad=True)
    z.retain_grad()
    loss = z.square().mean()
    loss.backward()
    scores = {layer: compute_token_importance(z, z.grad) for layer in prune_layers}
    sens = {layer: float(layer_sensitivity(scores[layer]).cpu()) for layer in prune_layers}
    return scores, sens


def main():
    ap = argparse.ArgumentParser(description="Calibrate loss-aware token pruning with JEPA gradients.")
    ap.add_argument("--out-config", required=True, help="Output LossAwarePruningConfig JSON path")
    ap.add_argument("--pruning-mode", choices=["single_layer", "cross_layer"], default="single_layer")
    ap.add_argument("--single-prune-layer", type=int, default=12, help="0-indexed transformer layer")
    ap.add_argument("--prune-layers", default="8,12,16,20", help="comma-separated 0-indexed layers")
    ap.add_argument("--global-prune-ratio", type=float, default=0.5)
    ap.add_argument("--num-tokens", type=int, default=61440, help="full encoder token count N")
    ap.add_argument("--gp", type=int, default=256, help="tokens per temporal slot (grid_size^2)")
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--calibration-samples", type=int, default=32)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic JEPA-like gradients (smoke test)")
    args = ap.parse_args()

    prune_layers = (
        [args.single_prune_layer]
        if args.pruning_mode == "single_layer"
        else [int(x) for x in args.prune_layers.split(",") if x.strip()]
    )

    config = LossAwarePruningConfig(
        enable_loss_aware_pruning=True,
        pruning_mode=args.pruning_mode,
        single_prune_layer=args.single_prune_layer,
        prune_layers=prune_layers,
        global_prune_ratio=args.global_prune_ratio,
        num_tokens_full=args.num_tokens,
        use_offline_calibration=True,
        importance_source="calibrated",
    )

    running_scores: dict[int, torch.Tensor] = {}
    running_sensitivities: dict[int, float] = {}
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.synthetic:
        raise SystemExit(
            "Real-data JEPA calibration requires wiring your dataloader + checkpoint here. "
            "For now use --synthetic to validate the cascade policy math, or call "
            "calibrate_batch_jepa() from your training entrypoint."
        )

    for _ in range(args.calibration_samples):
        batch_scores, batch_sens = _synthetic_jepa_calibration_batch(
            args.num_tokens, args.embed_dim, prune_layers, device
        )
        count = accumulate_position_scores(
            running_scores, running_sensitivities, batch_scores, batch_sens, count
        )

    config = finalize_calibration_config(
        config,
        args.num_tokens,
        running_scores,
        running_sensitivities,
        gp=args.gp,
    )
    verify_cascade_global_ratio(config.per_layer_prune_ratios, config.global_prune_ratio)

    scores_path = os.path.splitext(args.out_config)[0] + ".scores.pt"
    config = save_calibration_artifacts(
        config,
        running_scores,
        running_sensitivities,
        args.out_config,
        scores_path=scores_path,
    )

    cascade = simulate_cascade_keep_counts(
        args.num_tokens,
        config.per_layer_prune_ratios,
        gp=args.gp,
        round_to_frame_tokens=True,
    )
    print(f"[calibrate] wrote config -> {args.out_config}", flush=True)
    print(f"[calibrate] wrote scores -> {scores_path}", flush=True)
    print(f"  prune_layers={resolve_prune_layers(config)}", flush=True)
    print(f"  per_layer_prune_ratios={config.per_layer_prune_ratios}", flush=True)
    print(f"  cascade_keep_counts={cascade}", flush=True)
    print(
        f"  final_keep_ratio={1.0 - args.global_prune_ratio:.6f} "
        f"(product(1-r_l) verified)",
        flush=True,
    )


if __name__ == "__main__":
    main()
