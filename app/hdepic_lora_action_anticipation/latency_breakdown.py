"""CUDA + wall-clock latency breakdown for encoder-LoRA fp32 training paths."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.logging import AverageMeter

logger = logging.getLogger(__name__)

_LOG_INTERVAL_ENV = "EVAL_LATENCY_LOG_INTERVAL"
_REPORT_PATH_ENV = "EVAL_LATENCY_REPORT"


def latency_breakdown_enabled() -> bool:
    return os.environ.get("EVAL_LATENCY_BREAKDOWN", "0").lower() in {"1", "true", "yes", "on"}


def _log_interval(default: int = 10) -> int:
    raw = os.environ.get(_LOG_INTERVAL_ENV, str(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _report_path() -> Path | None:
    raw = os.environ.get(_REPORT_PATH_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


@dataclass
class LatencyBreakdown:
    """Track per-section GPU and wall time across training iterations."""

    enabled: bool = field(default_factory=latency_breakdown_enabled)
    meters_cuda: dict[str, AverageMeter] = field(default_factory=dict)
    meters_wall: dict[str, AverageMeter] = field(default_factory=dict)
    iter_wall_ms: AverageMeter = field(default_factory=AverageMeter)

    def _cuda_meter(self, name: str) -> AverageMeter:
        return self.meters_cuda.setdefault(name, AverageMeter())

    def _wall_meter(self, name: str) -> AverageMeter:
        return self.meters_wall.setdefault(name, AverageMeter())

    @contextmanager
    def section(self, name: str, *, sync_before: bool = True):
        if not self.enabled:
            yield
            return
        if sync_before and torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        start_event = end_event = None
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        try:
            yield
        finally:
            cuda_ms = -1.0
            if start_event is not None and end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                cuda_ms = float(start_event.elapsed_time(end_event))
                self._cuda_meter(name).update(cuda_ms)
            wall_ms = (time.perf_counter() - wall_t0) * 1000.0
            self._wall_meter(name).update(wall_ms)

    def record_wall(self, name: str, wall_ms: float) -> None:
        if not self.enabled:
            return
        self._wall_meter(name).update(wall_ms)

    def section_order(self) -> list[str]:
        preferred = [
            "data_load",
            "h2d",
            "binary_map",
            "fwd_input_adapter",
            "fwd_encoder",
            "fwd_predictor",
            "fwd_model",
            "fwd_classifier",
            "loss",
            "bwd_probe",
            "bwd_encoder",
            "bwd_total",
            "grad_clip",
            "optimizer",
        ]
        seen = set(preferred)
        extra = [k for k in self.meters_cuda if k not in seen]
        extra += [k for k in self.meters_wall if k not in seen and k not in extra]
        return preferred + sorted(extra)

    def to_dict(self) -> dict[str, Any]:
        rows: dict[str, dict[str, float]] = OrderedDict()
        for name in self.section_order():
            cuda = self.meters_cuda.get(name)
            wall = self.meters_wall.get(name)
            if cuda is None and wall is None:
                continue
            rows[name] = {
                "cuda_ms_avg": float(cuda.avg) if cuda is not None and cuda.count else 0.0,
                "cuda_ms_last": float(cuda.val) if cuda is not None and cuda.count else 0.0,
                "wall_ms_avg": float(wall.avg) if wall is not None and wall.count else 0.0,
                "wall_ms_last": float(wall.val) if wall is not None and wall.count else 0.0,
                "count": int((cuda or wall).count),
            }
        cuda_total = sum(row["cuda_ms_avg"] for row in rows.values())
        wall_total = sum(row["wall_ms_avg"] for row in rows.values())
        return {
            "sections": rows,
            "iter_wall_ms_avg": float(self.iter_wall_ms.avg),
            "cuda_ms_sum_avg": cuda_total,
            "wall_ms_sum_avg": wall_total,
        }

    def format_table(self, *, use_cuda: bool = True) -> str:
        data = self.to_dict()["sections"]
        if not data:
            return "(no latency sections recorded)"
        key = "cuda_ms_avg" if use_cuda and torch.cuda.is_available() else "wall_ms_avg"
        total = sum(row[key] for row in data.values()) or 1.0
        lines = [f"{'section':24s}  {'ms_avg':>10s}  {'pct':>6s}"]
        for name, row in data.items():
            ms = row[key]
            pct = 100.0 * ms / total
            lines.append(f"{name:24s}  {ms:10.1f}  {pct:5.1f}%")
        lines.append(f"{'TOTAL':24s}  {total:10.1f}  {'100.0':>5s}%")
        if self.iter_wall_ms.count:
            lines.append(f"{'iter_wall':24s}  {self.iter_wall_ms.avg:10.1f}")
        return "\n".join(lines)

    def log(self, itr: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        interval = _log_interval()
        if not force and itr % interval != 0:
            return
        logger.info("[latency itr=%d]\n%s", itr, self.format_table())

    def write_report(self, path: Path | None = None) -> Path | None:
        if not self.enabled:
            return None
        out = path or _report_path()
        if out is None:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        logger.info("Wrote latency breakdown report: %s", out)
        return out


def _unwrap(module: nn.Module) -> nn.Module:
    if hasattr(module, "module") and not hasattr(module, "encoder"):
        module = module.module
    if hasattr(module, "base_model"):
        module = module.base_model
    if hasattr(module, "module") and not hasattr(module, "encoder"):
        module = module.module
    return module


def _wrap_module_forward(module: nn.Module, name: str, breakdown: LatencyBreakdown) -> None:
    if not breakdown.enabled or getattr(module, "_latency_wrapped", False):
        return

    original = module.forward

    def wrapped_forward(*args, **kwargs):
        with breakdown.section(name):
            return original(*args, **kwargs)

    module.forward = wrapped_forward  # type: ignore[method-assign]
    module._latency_wrapped = True


def instrument_model_for_breakdown(model: nn.Module, breakdown: LatencyBreakdown) -> None:
    """Attach per-module forward timers for encoder / predictor / input adapter."""
    if not breakdown.enabled:
        return
    inner = model
    if hasattr(inner, "module") and not hasattr(inner, "input_adapter"):
        inner = inner.module
    if hasattr(inner, "input_adapter"):
        _wrap_module_forward(inner.input_adapter, "fwd_input_adapter", breakdown)
    base = getattr(inner, "base_model", inner)
    base = _unwrap(base)
    if hasattr(base, "encoder"):
        _wrap_module_forward(base.encoder, "fwd_encoder", breakdown)
    if hasattr(base, "predictor") and not getattr(base, "no_predictor", False):
        _wrap_module_forward(base.predictor, "fwd_predictor", breakdown)
    logger.info(
        "Latency breakdown: instrumented model components (adapter=%s encoder=%s predictor=%s)",
        hasattr(inner, "input_adapter"),
        hasattr(base, "encoder"),
        hasattr(base, "predictor") and not getattr(base, "no_predictor", False),
    )
