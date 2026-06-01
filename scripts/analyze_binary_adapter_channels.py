"""Diagnose how much the binary-gaze input channel was actually used.

The binary input adapter's first conv layer is a Conv3d(4, hidden_dim, k=1):
    channel 0..2 = RGB
    channel 3    = binary gaze map

We compare per-input-channel weight column norms after training. If channel 3
norm is comparable to RGB channels, the adapter learned to use the gaze map.
If it's much smaller (e.g. an order of magnitude below RGB), the gaze channel
was effectively unused — which explains why zero-channel val matches normal val.

Usage:
    python scripts/analyze_binary_adapter_channels.py \
        /path/to/binary_input_adapter_latest.pt

Works against any checkpoint saved by
``app.hdepic_lora_action_anticipation.eval`` (key ``input_adapter``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def analyze(ckpt_path: Path) -> int:
    if not ckpt_path.is_file():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob.get("input_adapter", blob) if isinstance(blob, dict) else blob
    if not isinstance(sd, dict):
        print(f"ERROR: unexpected checkpoint structure: {type(blob)}", file=sys.stderr)
        return 1

    first_conv_key = None
    for k in ("net.0.weight", "input_adapter.net.0.weight"):
        if k in sd:
            first_conv_key = k
            break
    if first_conv_key is None:
        print(f"ERROR: could not find net.0.weight in checkpoint. Keys: {list(sd)[:8]}", file=sys.stderr)
        return 1

    w = sd[first_conv_key]
    if w.dim() != 5 or w.shape[1] != 4:
        print(f"ERROR: expected Conv3d weight [out, 4, 1, 1, 1], got {tuple(w.shape)}", file=sys.stderr)
        return 1

    hidden_dim = w.shape[0]
    rgb = w[:, :3, :, :, :]
    gaze = w[:, 3:4, :, :, :]

    rgb_l2 = rgb.norm().item()
    gaze_l2 = gaze.norm().item()
    rgb_mean_abs = rgb.abs().mean().item()
    gaze_mean_abs = gaze.abs().mean().item()
    rgb_per_channel_l2 = [w[:, c, :, :, :].norm().item() for c in range(3)]
    gaze_per_channel_l2 = gaze_l2

    print(f"Checkpoint: {ckpt_path}")
    print(f"net.0.weight shape: {tuple(w.shape)}  (hidden_dim={hidden_dim})")
    print()
    print("Per-input-channel L2 norm of net.0.weight columns:")
    print(f"  channel 0 (R):    {rgb_per_channel_l2[0]:.6f}")
    print(f"  channel 1 (G):    {rgb_per_channel_l2[1]:.6f}")
    print(f"  channel 2 (B):    {rgb_per_channel_l2[2]:.6f}")
    print(f"  channel 3 (gaze): {gaze_per_channel_l2:.6f}")
    print()
    rgb_avg = rgb_l2 / (3 ** 0.5)
    ratio_l2 = gaze_l2 / max(rgb_avg, 1e-12)
    ratio_mean = gaze_mean_abs / max(rgb_mean_abs, 1e-12)
    print(f"Aggregate:")
    print(f"  RGB combined L2   = {rgb_l2:.6f}  (per-channel-avg {rgb_avg:.6f})")
    print(f"  gaze L2           = {gaze_l2:.6f}")
    print(f"  gaze_L2 / rgb_avg = {ratio_l2:.4f}")
    print(f"  gaze_mean_abs / rgb_mean_abs = {ratio_mean:.4f}")
    print()
    if ratio_l2 < 0.1:
        verdict = "gaze channel weights are <10% of RGB columns -> effectively unused"
    elif ratio_l2 < 0.5:
        verdict = "gaze channel weights are below half of RGB columns -> weakly used"
    elif ratio_l2 < 1.5:
        verdict = "gaze channel weights are comparable to RGB columns -> used"
    else:
        verdict = "gaze channel weights exceed RGB columns -> dominated by gaze"
    print(f"Verdict: {verdict}")

    last_conv_key = None
    for k in ("net.4.weight", "input_adapter.net.4.weight"):
        if k in sd:
            last_conv_key = k
            break
    if last_conv_key is not None:
        last_w = sd[last_conv_key]
        last_b = sd.get(last_conv_key.replace("weight", "bias"))
        print()
        print(f"net.4.weight (zero-init output proj) L2 = {last_w.norm().item():.6f}")
        if last_b is not None:
            print(f"net.4.bias                    L2 = {last_b.norm().item():.6f}")
        print("(If this is still ~0, the whole adapter is identity and neither RGB nor gaze branch contributes.)")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path, help="Path to binary_input_adapter_latest.pt")
    args = ap.parse_args()
    return analyze(args.checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
