"""Real-feature +6s optimizer smoke on a tick with known past labels."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

import torch

from .egtea_cache import MAIN_ROOT, ROOT
from .legacy_autograd_fix import install_clone_per_layer_fix
from .stream_data import StreamingTSNReader, build_split1_ticks
from .stream_model import FullPrefixStreamingScalAnt, masked_stream_loss


def main() -> None:
    upstream = ROOT / "refer_repo/scalable-anticipation"
    sys.path.insert(0, str(upstream))
    from scalant.config import load_config
    from scalant.models import QueryPredictor

    ticks_by_video = defaultdict(list)
    for tick in build_split1_ticks("train"):
        ticks_by_video[tick.video].append(tick)
    candidates = [(index, video) for video, ticks in ticks_by_video.items()
                  for index, tick in enumerate(ticks)
                  if tick.target_actions[1] >= 0 and any(x >= 0 for x in tick.past_actions)]
    if not candidates:
        raise ValueError("No +6s train tick with a known past annotation")
    index, video = min(candidates)
    prefix = ticks_by_video[video][:index + 1]
    reader = StreamingTSNReader()
    try:
        features = torch.cat([reader.read(tick) for tick in prefix], dim=0).unsqueeze(0)
    finally:
        reader.close()
    if features.shape[1] != 16 * len(prefix) or not torch.isfinite(features).all():
        raise ValueError("Bad causal prefix")

    install_clone_per_layer_fix()
    cfg = load_config(str(upstream / "configs/ek100/default.yaml"))
    cfg.DATA.TAU_O = 4.0
    cfg.DATA.LONG_MEMORY_LENGTH = 0.0
    cfg.MODEL.INPUT_DIM = 1024
    cfg.MODEL.IGNORE_INDEX = -1
    model = QueryPredictor(cfg, {"action": 106, "verb": 19, "noun": 53}, None).cuda().train()
    stream = FullPrefixStreamingScalAnt(model)
    stream.reset(video)
    stream.history = features[:, :-16].cuda() if len(prefix) > 1 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    output = stream.process(video, features[:, -16:].cuda())
    loss = masked_stream_loss(output, prefix[-1], horizon=6)
    if loss.value is None or loss.valid_future != 1 or loss.valid_past <= 0:
        raise ValueError("+6s target or known-past auxiliary mask not exercised")
    if not torch.isfinite(loss.value):
        raise ValueError("Nonfinite masked six-head loss")
    loss.value.backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum().item() > 0 for p in model.parameters()):
        raise ValueError("No finite nonzero model gradients")
    optimizer.step()
    torch.cuda.synchronize()
    report = {"kind": "real_feature_h6_known_past_optimizer_smoke_not_accuracy",
              "split": 1, "video": video, "tick_frame": prefix[-1].tick_frame,
              "prefix_ticks": len(prefix), "history_tokens": stream.history.shape[1],
              "valid_future": loss.valid_future, "valid_past": loss.valid_past,
              "loss": float(loss.value.detach()),
              "seconds_including_compile": time.monotonic() - started}
    path = MAIN_ROOT / "outputs/b19_scalant/implementation/split1_stream_h6_known_past_smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
