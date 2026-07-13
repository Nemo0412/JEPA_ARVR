"""MPJPE / MPJVE metrics for Ego-Exo4D 3D motion prediction."""

from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Element-wise MSE with joint validity mask [B, T, J]."""
    diff = (pred - target) ** 2
    mask3 = mask.unsqueeze(-1).expand_as(diff)
    denom = mask3.sum().clamp_min(1.0)
    return (diff * mask3).sum() / denom


def compute_mpjpe_mpjve(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    fps: float,
) -> dict[str, float]:
    """
    pred/target: [B, T, J, 3] in meters (or consistent units).
    mask: [B, T, J] bool/float, 1 = valid joint.
    """
    diff = (pred - target).norm(dim=-1)
    valid = mask > 0
    if valid.sum() == 0:
        return {"mpjpe_cm": float("nan"), "mpjve_cm_s": float("nan")}

    mpjpe_m = (diff * valid).sum() / valid.sum().clamp_min(1.0)
    mpjpe_cm = float(mpjpe_m.item() * 100.0)

    if pred.size(1) < 2:
        return {"mpjpe_cm": mpjpe_cm, "mpjve_cm_s": float("nan")}

    vel_pred = (pred[:, 1:] - pred[:, :-1]) * float(fps)
    vel_tgt = (target[:, 1:] - target[:, :-1]) * float(fps)
    vel_diff = (vel_pred - vel_tgt).norm(dim=-1)
    vel_valid = valid[:, 1:] & valid[:, :-1]
    if vel_valid.sum() == 0:
        mpjve_cm_s = float("nan")
    else:
        mpjve_m_s = (vel_diff * vel_valid).sum() / vel_valid.sum().clamp_min(1.0)
        mpjve_cm_s = float(mpjve_m_s.item() * 100.0)

    return {"mpjpe_cm": mpjpe_cm, "mpjve_cm_s": mpjve_cm_s}
