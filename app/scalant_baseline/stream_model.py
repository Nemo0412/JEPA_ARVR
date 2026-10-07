"""External full-prefix streaming interface for the unchanged ScalAnt model.

Each call consumes one 4s TSN chunk, appends it to the current video's causal
history, and keeps only the newest 16 tokens as working memory. Re-encoding
the prefix is deliberately explicit so its budget can be measured; no
incremental Mamba parity is claimed by this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .stream_data import StreamTick


class FullPrefixStreamingScalAnt:
    def __init__(self, model):
        self.model = model
        self.video: str | None = None
        self.history: torch.Tensor | None = None

    def reset(self, video: str) -> None:
        self.video = video
        self.history = None

    def process(self, video: str, chunk: torch.Tensor):
        if self.video != video:
            self.reset(video)
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1:] != (16, 1024):
            raise ValueError("One 4s 4Hz 1024-D chunk is required")
        # Input features do not require gradients; retaining the raw past does
        # not retain optimizer-step graphs.
        self.history = chunk if self.history is None else torch.cat((self.history, chunk), dim=1)
        self.model.long_mem_len = self.history.shape[1] - 16
        return self.model(self.history)


@dataclass(frozen=True)
class StreamLoss:
    value: torch.Tensor | None
    valid_future: int
    valid_past: int


def masked_stream_loss(output, tick: StreamTick, horizon: int) -> StreamLoss:
    """Six released action/verb/noun terms, masked to known current labels."""
    if horizon not in (2, 6):
        raise ValueError("Expected independently trained 2s or 6s horizon")
    index = 0 if horizon == 2 else 1
    device = output.future_actions.device
    terms = []
    valid_future = int(tick.target_actions[index] >= 0)
    if valid_future:
        for logits, label in ((output.future_actions[:, -1], tick.target_actions[index]),
                              (output.future_verbs[:, -1], tick.target_verbs[index]),
                              (output.future_nouns[:, -1], tick.target_nouns[index])):
            terms.append(F.cross_entropy(logits, torch.tensor([label], device=device)))

    past_action = torch.tensor(tick.past_actions, device=device)
    valid = past_action >= 0
    valid_past = int(valid.sum())
    if valid_past:
        for logits, labels in ((output.past_actions[:, -16:], tick.past_actions),
                               (output.past_verbs[:, -16:], tick.past_verbs),
                               (output.past_nouns[:, -16:], tick.past_nouns)):
            targets = torch.tensor(labels, device=device)
            terms.append(F.cross_entropy(logits[:, valid, :].reshape(-1, logits.shape[-1]),
                                         targets[valid]))
    # The released objective sums its active terms. Unknown/ambiguous past
    # positions never become artificial background labels.
    return StreamLoss(sum(terms) if terms else None, valid_future, valid_past)
