"""Two real split-1 4s streaming ticks through full ScalAnt and masked loss."""

from __future__ import annotations

import json
import sys
import time

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
    install_clone_per_layer_fix()
    ticks = build_split1_ticks("train")
    video = ticks[0].video
    selected = [tick for tick in ticks if tick.video == video][:2]
    if len(selected) != 2:
        raise ValueError("First session lacks two processing ticks")
    cfg = load_config(str(upstream / "configs/ek100/default.yaml"))
    cfg.DATA.TAU_O = 4.0
    cfg.DATA.LONG_MEMORY_LENGTH = 0.0
    cfg.MODEL.INPUT_DIM = 1024
    cfg.MODEL.IGNORE_INDEX = -1
    model = QueryPredictor(cfg, {"action": 106, "verb": 19, "noun": 53}, None).cuda().train()
    stream = FullPrefixStreamingScalAnt(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    reader = StreamingTSNReader()
    steps = []
    try:
        for tick in selected:
            chunk = reader.read(tick).unsqueeze(0).cuda()
            optimizer.zero_grad(set_to_none=True)
            started = time.monotonic()
            output = stream.process(tick.video, chunk)
            loss = masked_stream_loss(output, tick, horizon=2)
            if loss.value is None or not torch.isfinite(loss.value):
                raise ValueError("No finite source-linked masked loss")
            loss.value.backward()
            optimizer.step()
            torch.cuda.synchronize()
            steps.append({"video": tick.video, "tick_frame": tick.tick_frame,
                          "history_tokens": stream.history.shape[1],
                          "target_action": tick.target_actions[0],
                          "valid_future": loss.valid_future,
                          "valid_past": loss.valid_past,
                          "loss": float(loss.value.detach()),
                          "seconds": time.monotonic() - started})
    finally:
        reader.close()
    if [step["history_tokens"] for step in steps] != [16, 32]:
        raise ValueError("Causal memory failed to accumulate two chunks")
    report = {"kind": "real_feature_stream_optimizer_smoke_not_accuracy",
              "split": 1, "horizon_seconds": 2,
              "clone_between_layers_autograd_fix": True,
              "steps": steps}
    path = MAIN_ROOT / "outputs/b19_scalant/implementation/split1_stream_gpu_smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
