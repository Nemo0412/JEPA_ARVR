"""Checkpoint helpers for mixed-precision training resume."""

from __future__ import annotations

from typing import Any


def serialize_checkpoint_scalers(scaler) -> list[dict[str, Any]] | None:
    """Serialize GradScaler objects for ``latest.pt``.

    fp32 runs historically returned ``[None]`` from ``init_opt``. The upstream
    save path must skip ``None`` placeholders instead of calling ``state_dict()``.
    """
    if scaler is None:
        return None
    states = [s.state_dict() for s in scaler if s is not None]
    return states or None


def restore_checkpoint_scalers(scaler, states) -> None:
    """Restore GradScaler state saved by :func:`serialize_checkpoint_scalers`."""
    if scaler is None or not states:
        return
    live = [s for s in scaler if s is not None]
    if len(live) != len(states):
        raise ValueError(
            f"checkpoint scaler count mismatch: live_scalers={len(live)} saved_states={len(states)}"
        )
    for live_scaler, saved_state in zip(live, states):
        live_scaler.load_state_dict(saved_state)
