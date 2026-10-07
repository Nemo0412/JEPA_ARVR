"""Action scoring for split-1 streaming ticks with explicit denominators."""

from __future__ import annotations

from collections import Counter

import torch

from .stream_data import StreamTick


class ActionTickScorer:
    def __init__(self, horizon: int):
        if horizon not in (2, 6):
            raise ValueError("Expected an independently trained +2s or +6s model")
        self.horizon = horizon
        self.by_class_total = Counter()
        self.by_class_top5 = Counter()

    def update(self, tick: StreamTick, logits: torch.Tensor) -> None:
        label = tick.target_actions[0 if self.horizon == 2 else 1]
        if label < 0:
            return
        if logits.shape != (106,):
            raise ValueError(f"Expected 106 action logits, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite action logits")
        self.by_class_total[label] += 1
        if label in logits.topk(5).indices.tolist():
            self.by_class_top5[label] += 1

    def report(self) -> dict:
        scored = sum(self.by_class_total.values())
        correct = sum(self.by_class_top5.values())
        mean_recall = (sum(self.by_class_top5[c] / n for c, n in self.by_class_total.items())
                       / len(self.by_class_total) if self.by_class_total else None)
        return {"metric_scope": "split1_streaming_tick",
                "horizon_seconds": self.horizon,
                "scored_tick_count": scored,
                "classes_with_scored_ticks": len(self.by_class_total),
                "action_top5_micro": correct / scored if scored else None,
                "action_mean_recall_at5_present_classes": mean_recall,
                "by_class_total": dict(sorted(self.by_class_total.items()))}
