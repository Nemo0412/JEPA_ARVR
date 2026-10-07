"""Real-feature gradient-accumulation and epoch-boundary resume parity smoke."""

from __future__ import annotations

import json
import random
import time
from collections import Counter

import numpy as np
import torch

from .egtea_cache import MAIN_ROOT
from .official_stream_data import build_official_split1_ticks
from .official_stream_train import (checkpoint_payload, load_checkpoint,
                                    make_model_and_optimizer, save_checkpoint,
                                    train_epoch)
from .stream_data import StreamingTSNReader


OUTPUT = MAIN_ROOT / "outputs/b19_scalant/implementation/official_split1_resume_smoke.json"


def main():
    ticks = build_official_split1_ticks("train")
    counts = Counter(t.video for t in ticks)
    video = max(counts, key=lambda name: (counts[name], name))
    selected = [tick for tick in ticks if tick.video == video][:130]
    if len(selected) != 130:
        raise ValueError("Longest video lacks 130 official stream ticks")
    by_video = {video: selected}
    reader = StreamingTSNReader()
    report = {"kind": "official_split1_real_feature_resume_smoke_not_accuracy",
              "video": video, "ticks_per_mini_epoch": len(selected),
              "horizons": {}}
    try:
        for horizon in (2, 6):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            np.random.seed(42)
            random.seed(42)
            model, optimizer, scheduler = make_model_and_optimizer(horizon)
            started = time.monotonic()
            first = train_epoch(model, optimizer, reader, by_video, horizon, 0)
            scheduler.step()
            if first["optimizer_steps"] != 2 or first["supervised_ticks"] < 129:
                raise ValueError(f"Unexpected accumulation coverage: {first}")
            path = MAIN_ROOT / f"outputs/b19_scalant/implementation/official_resume_h{horizon}.pt"
            save_checkpoint(path, checkpoint_payload(model, optimizer, scheduler,
                                                     horizon, 0, 0.0, -1))
            checkpoint_weights = {name: value.detach().clone()
                                  for name, value in model.state_dict().items()}
            expected_cpu_rng = torch.get_rng_state().clone()
            expected_cuda_rng = torch.cuda.get_rng_state().clone()
            expected_lr = optimizer.param_groups[0]["lr"]
            uninterrupted_grads = {}

            def remember_grads(current_model, _, count, step):
                if step == 0:
                    uninterrupted_grads.update({
                        name: parameter.grad.detach().cpu().clone()
                        for name, parameter in current_model.named_parameters()
                        if parameter.grad is not None})

            continuous_trace = []
            continuous = train_epoch(model, optimizer, reader, by_video, horizon, 1,
                                     loss_trace=continuous_trace,
                                     before_step=remember_grads)
            reference = {name: value.detach().clone() for name, value in model.state_dict().items()}
            reference_steps = [state["step"].item() for state in optimizer.state.values()
                               if "step" in state]
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            resumed_model, resumed_optimizer, resumed_scheduler = make_model_and_optimizer(horizon)
            restored = load_checkpoint(path, resumed_model, resumed_optimizer,
                                       resumed_scheduler, horizon)
            if restored["completed_epoch"] != 0:
                raise ValueError("Checkpoint cursor changed")
            reload_weight_difference = max(
                float((value - resumed_model.state_dict()[name]).abs().max())
                for name, value in checkpoint_weights.items())
            cpu_rng_equal = torch.equal(expected_cpu_rng, torch.get_rng_state())
            cuda_rng_equal = torch.equal(expected_cuda_rng, torch.cuda.get_rng_state())
            resumed_lr = resumed_optimizer.param_groups[0]["lr"]
            loaded_state = resumed_optimizer.state_dict()["state"]
            saved_state = restored["optimizer"]["state"]
            optimizer_reload_max_difference = max(
                float((saved_value.cpu() - loaded_state[param_id][key].cpu()).abs().max())
                for param_id, state in saved_state.items()
                for key, saved_value in state.items()
                if isinstance(saved_value, torch.Tensor))
            gradient_max_difference = 0.0

            def compare_grads(current_model, _, count, step):
                nonlocal gradient_max_difference
                if step == 0:
                    gradient_max_difference = max(
                        float((uninterrupted_grads[name] - parameter.grad.detach().cpu()).abs().max())
                        for name, parameter in current_model.named_parameters()
                        if parameter.grad is not None)

            resumed_trace = []
            resumed = train_epoch(resumed_model, resumed_optimizer, reader,
                                  by_video, horizon, 1, loss_trace=resumed_trace,
                                  before_step=compare_grads)
            for key in ("epoch", "processed_ticks", "supervised_ticks", "optimizer_steps",
                        "future_labels", "known_past_positions"):
                if continuous[key] != resumed[key]:
                    raise ValueError(f"Continuous/resumed coverage differs at {key}")
            max_difference = max(float((value - resumed_model.state_dict()[name]).abs().max())
                                 for name, value in reference.items())
            resumed_steps = [state["step"].item() for state in resumed_optimizer.state.values()
                             if "step" in state]
            if reference_steps != resumed_steps:
                raise ValueError("Optimizer step states differ after resume")
            report["horizons"][horizon] = {
                "first_mini_epoch": first, "second_mini_epoch": continuous,
                "resumed_second_mini_epoch": resumed,
                "reload_weight_difference": reload_weight_difference,
                "cpu_rng_equal_at_resume": cpu_rng_equal,
                "cuda_rng_equal_at_resume": cuda_rng_equal,
                "lr_at_resume_expected": expected_lr,
                "lr_at_resume_loaded": resumed_lr,
                "optimizer_reload_max_difference": optimizer_reload_max_difference,
                "first_group_gradient_max_difference": gradient_max_difference,
                "mean_loss_difference": abs(continuous["mean_loss_per_supervised_tick"]
                                            - resumed["mean_loss_per_supervised_tick"]),
                "first_loss_difference_index": next((i for i, (a, b) in enumerate(zip(
                    continuous_trace, resumed_trace)) if a != b), None),
                "first10_loss_differences": [abs(a - b) for a, b in zip(
                    continuous_trace[:10], resumed_trace[:10])],
                "max_loss_difference": max(abs(a - b) for a, b in zip(
                    continuous_trace, resumed_trace)),
                "max_weight_difference": max_difference,
                "optimizer_state_steps": sorted(set(reference_steps)),
                "seconds_including_compile_io_and_checkpoint": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "checkpoint": str(path),
            }
            print(json.dumps({"horizon": horizon, **report["horizons"][horizon]}), flush=True)
            del resumed_model, resumed_optimizer, resumed_scheduler
            torch.cuda.empty_cache()
    finally:
        reader.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
