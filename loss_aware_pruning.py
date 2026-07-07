"""Loss-aware token pruning for V-JEPA-style video encoders.

Token importance (calibration only) uses a first-order Taylor approximation of JEPA loss:

    s_{l,i} = mean_D( | grad(z_{l,i}) * z_{l,i} | )

Runtime pruning applies a **saved calibrated policy**:
  * conditional per-layer prune ratios r_l on the *current* remaining tokens
  * cascade:  N_final = N_0 * product_l (1 - r_l)
  * token ranking via calibrated mean importance scores (position-indexed), not feature_norm

Cross-layer ratio allocation (inverse layer sensitivity S_l = mean(score_l)):

    w_l = (1 / (S_l + eps)) / sum_j (1 / (S_j + eps))
    r_l = 1 - (1 - R_total)^(w_l)     where R_total = global_prune_ratio
    =>  product_l (1 - r_l) = 1 - R_total
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

import torch


PruningMode = Literal["single_layer", "cross_layer"]
ImportanceSource = Literal["gradient", "calibrated"]


@dataclass
class LossAwarePruningConfig:
    enable_loss_aware_pruning: bool = False
    pruning_mode: PruningMode = "single_layer"
    single_prune_layer: int = 12
    prune_layers: list[int] = field(default_factory=lambda: [8, 12, 16, 20])
    global_prune_ratio: float = 0.5
    per_layer_prune_ratios: dict[int, float] | None = None
    num_tokens_full: int = 0
    use_offline_calibration: bool = True
    calibration_path: str = ""
    calibration_scores_path: str = ""
    importance_source: ImportanceSource = "calibrated"
    eps: float = 1e-8
    round_to_frame_tokens: bool = True
    protected_token_indices: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "LossAwarePruningConfig":
        cfg = cls()
        for key, value in data.items():
            if not hasattr(cfg, key):
                continue
            if key in ("prune_layers", "protected_token_indices") and value is not None:
                setattr(cfg, key, list(value))
            elif key == "per_layer_prune_ratios" and value is not None:
                setattr(cfg, key, {int(k): float(v) for k, v in value.items()})
            else:
                setattr(cfg, key, value)
        return cfg

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["prune_layers"] = list(self.prune_layers)
        payload["protected_token_indices"] = list(self.protected_token_indices)
        if self.per_layer_prune_ratios is not None:
            payload["per_layer_prune_ratios"] = {
                str(k): float(v) for k, v in self.per_layer_prune_ratios.items()
            }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LossAwarePruningConfig":
        data = json.loads(Path(path).read_text())
        if "per_layer_prune_ratios" in data and data["per_layer_prune_ratios"] is not None:
            data["per_layer_prune_ratios"] = {
                int(k): float(v) for k, v in data["per_layer_prune_ratios"].items()
            }
        return cls.from_dict(data)

    def load_calibrated_scores(self, device: torch.device | str = "cpu") -> dict[int, torch.Tensor]:
        if not self.calibration_scores_path:
            raise ValueError("calibration_scores_path is not set on config")
        obj = torch.load(self.calibration_scores_path, map_location=device, weights_only=False)
        raw = obj["per_layer_mean_token_scores"]
        return {int(k): v.float() for k, v in raw.items()}


def save_calibration_artifacts(
    config: LossAwarePruningConfig,
    per_layer_mean_token_scores: dict[int, torch.Tensor],
    layer_sensitivities: dict[int, float],
    config_path: str | Path,
    scores_path: str | Path | None = None,
) -> LossAwarePruningConfig:
    config_path = Path(config_path)
    if scores_path is None:
        scores_path = config_path.with_suffix(".scores.pt")
    else:
        scores_path = Path(scores_path)

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "per_layer_mean_token_scores": {int(k): v.detach().cpu().float() for k, v in per_layer_mean_token_scores.items()},
            "layer_sensitivities": {int(k): float(v) for k, v in layer_sensitivities.items()},
        },
        scores_path,
    )

    out = LossAwarePruningConfig.from_dict(asdict(config))
    out.calibration_path = str(config_path)
    out.calibration_scores_path = str(scores_path)
    out.use_offline_calibration = True
    out.importance_source = "calibrated"
    out.save(config_path)
    return out


def compute_token_importance(z: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """JEPA gradient-based score: mean_D(|grad * z|), shape [B, N]."""
    if grad is None:
        raise ValueError("compute_token_importance requires grad from JEPA loss backward")
    return torch.mean(torch.abs(grad * z), dim=-1)


def gather_tokens(z: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    idx_exp = keep_idx.unsqueeze(-1).expand(-1, -1, z.shape[-1])
    return z.gather(1, idx_exp)


def _round_keep_count(keep_count: int, num_tokens: int, gp: int | None, round_to_frame: bool) -> int:
    keep_count = max(1, min(int(keep_count), num_tokens))
    if round_to_frame and gp is not None and gp > 1:
        keep_count = max(gp, (keep_count // gp) * gp)
        keep_count = min(keep_count, (num_tokens // gp) * gp)
    return keep_count


def conditional_keep_count(
    num_tokens: int,
    conditional_prune_ratio: float,
    *,
    gp: int | None = None,
    round_to_frame_tokens: bool = True,
) -> int:
    """Keep count after pruning fraction r_l of the *current* num_tokens."""
    keep = int(num_tokens * (1.0 - conditional_prune_ratio))
    return _round_keep_count(keep, num_tokens, gp, round_to_frame_tokens)


def topk_token_prune(
    z: torch.Tensor,
    score: torch.Tensor,
    keep_ratio: float | None = None,
    keep_count: int | None = None,
    conditional_prune_ratio: float | None = None,
    *,
    gp: int | None = None,
    round_to_frame_tokens: bool = True,
    protected_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard top-K pruning. Returns (pruned_z, keep_idx) sorted along dim=1."""
    bsz, num_tokens, _ = z.shape
    if keep_count is None:
        if conditional_prune_ratio is not None:
            keep_count = conditional_keep_count(
                num_tokens,
                conditional_prune_ratio,
                gp=gp,
                round_to_frame_tokens=round_to_frame_tokens,
            )
        elif keep_ratio is not None:
            keep_count = conditional_keep_count(
                num_tokens,
                keep_ratio,
                gp=gp,
                round_to_frame_tokens=round_to_frame_tokens,
            )
        else:
            raise ValueError("topk_token_prune requires keep_count, keep_ratio, or conditional_prune_ratio")

    keep_count = _round_keep_count(keep_count, num_tokens, gp, round_to_frame_tokens)

    if protected_indices is not None and protected_indices.numel() > 0:
        score = score.clone()
        score[:, protected_indices] = score.max(dim=1, keepdim=True).values + 1.0

    if keep_count >= num_tokens:
        keep_idx = torch.arange(num_tokens, device=z.device).unsqueeze(0).expand(bsz, -1)
        return z, keep_idx

    keep_idx = score.topk(keep_count, dim=1).indices.sort(dim=1).values
    return gather_tokens(z, keep_idx), keep_idx


def layer_sensitivity(token_score: torch.Tensor) -> torch.Tensor:
    return token_score.mean()


def cascade_final_keep_ratio(per_layer_prune_ratios: dict[int, float]) -> float:
    """product_l (1 - r_l)."""
    keep = 1.0
    for layer in sorted(per_layer_prune_ratios):
        keep *= 1.0 - float(per_layer_prune_ratios[layer])
    return keep


def verify_cascade_global_ratio(
    per_layer_prune_ratios: dict[int, float],
    global_prune_ratio: float,
    *,
    tol: float = 1e-5,
) -> float:
    """Assert product_l (1-r_l) ~= 1 - global_prune_ratio; return achieved final keep ratio."""
    achieved = cascade_final_keep_ratio(per_layer_prune_ratios)
    expected = 1.0 - global_prune_ratio
    if abs(achieved - expected) > tol:
        raise ValueError(
            f"cascade ratios do not match global_prune_ratio: "
            f"product(1-r_l)={achieved:.6f}, expected {expected:.6f}"
        )
    return achieved


def allocate_cascade_prune_ratios(
    layer_sensitivity_map: dict[int, float | torch.Tensor],
    global_prune_ratio: float,
    *,
    eps: float = 1e-8,
) -> dict[int, float]:
    """Allocate conditional prune ratios r_l with product_l (1-r_l) = 1 - global_prune_ratio.

    Less sensitive layers (lower S_l) receive larger r_l via inverse-sensitivity weights.
    """
    layers = sorted(layer_sensitivity_map.keys())
    if not layers:
        return {}

    if len(layers) == 1:
        return {layers[0]: float(global_prune_ratio)}

    final_keep_ratio = 1.0 - float(global_prune_ratio)
    if not (0.0 < final_keep_ratio < 1.0):
        raise ValueError("global_prune_ratio must be in (0, 1) for cross-layer cascade allocation")

    inv = {}
    for layer in layers:
        sens = layer_sensitivity_map[layer]
        if isinstance(sens, torch.Tensor):
            sens = float(sens.detach().cpu())
        inv[layer] = 1.0 / (float(sens) + eps)

    total_inv = sum(inv.values())
    ratios: dict[int, float] = {}
    for layer in layers:
        weight = inv[layer] / total_inv
        keep_factor = final_keep_ratio ** weight
        ratios[layer] = 1.0 - keep_factor

    verify_cascade_global_ratio(ratios, global_prune_ratio)
    return ratios


def simulate_cascade_keep_counts(
    num_tokens_full: int,
    per_layer_prune_ratios: dict[int, float],
    *,
    gp: int | None = None,
    round_to_frame_tokens: bool = True,
) -> dict[int, int]:
    """Simulate cascade pruning to get keep count after each layer (post-rounding)."""
    n = num_tokens_full
    out: dict[int, int] = {}
    for layer in sorted(per_layer_prune_ratios):
        keep = conditional_keep_count(
            n,
            per_layer_prune_ratios[layer],
            gp=gp,
            round_to_frame_tokens=round_to_frame_tokens,
        )
        out[layer] = keep
        n = keep
    return out


def expected_final_keep_count(
    num_tokens_full: int,
    per_layer_prune_ratios: dict[int, float],
    *,
    gp: int | None = None,
    round_to_frame_tokens: bool = True,
) -> int:
    counts = simulate_cascade_keep_counts(
        num_tokens_full,
        per_layer_prune_ratios,
        gp=gp,
        round_to_frame_tokens=round_to_frame_tokens,
    )
    if not counts:
        return num_tokens_full
    return counts[max(counts)]


def lookup_calibrated_scores(
    layer_idx: int,
    token_pos: torch.Tensor,
    per_layer_mean_token_scores: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Map current token positions to calibrated mean scores, shape [B, N]."""
    table = per_layer_mean_token_scores[int(layer_idx)]
    if table.ndim != 1:
        raise ValueError("per_layer_mean_token_scores must be 1-D tensors indexed by full position")
    return table[token_pos]


def resolve_prune_layers(config: LossAwarePruningConfig) -> list[int]:
    if config.pruning_mode == "single_layer":
        return [config.single_prune_layer]
    return list(config.prune_layers)


def build_prune_ratio_schedule(
    config: LossAwarePruningConfig,
    layer_sensitivity_map: dict[int, float] | None = None,
) -> dict[int, float]:
    if config.per_layer_prune_ratios is not None:
        ratios = {int(k): float(v) for k, v in config.per_layer_prune_ratios.items()}
        verify_cascade_global_ratio(ratios, config.global_prune_ratio)
        return ratios

    prune_layers = resolve_prune_layers(config)
    if config.pruning_mode == "single_layer":
        return {config.single_prune_layer: float(config.global_prune_ratio)}

    if layer_sensitivity_map is None:
        layer_sensitivity_map = {layer: 1.0 for layer in prune_layers}
    return allocate_cascade_prune_ratios(
        layer_sensitivity_map,
        config.global_prune_ratio,
        eps=config.eps,
    )


class LossAwareScoreCollector:
    """Capture intermediate encoder features and JEPA gradients at pruning layers."""

    def __init__(self, prune_layers: list[int]):
        self.prune_layers = set(prune_layers)
        self.features: dict[int, torch.Tensor] = {}
        self.scores: dict[int, torch.Tensor] = {}
        self.sensitivities: dict[int, float] = {}

    def maybe_retain(self, layer_idx: int, z: torch.Tensor) -> torch.Tensor:
        if layer_idx in self.prune_layers:
            if not z.requires_grad:
                z = z.requires_grad_(True)
            z.retain_grad()
            self.features[layer_idx] = z
        return z

    def finalize(self) -> None:
        self.scores.clear()
        self.sensitivities.clear()
        for layer, z in self.features.items():
            score = compute_token_importance(z, z.grad)
            self.scores[layer] = score.detach()
            self.sensitivities[layer] = float(layer_sensitivity(score).detach().cpu())


def accumulate_position_scores(
    running_scores: dict[int, torch.Tensor],
    running_sensitivities: dict[int, float],
    batch_scores: dict[int, torch.Tensor],
    batch_sensitivities: dict[int, float],
    count: int,
) -> int:
    """Online average of per-position scores [N] and layer sensitivities."""
    count += 1
    for layer, score in batch_scores.items():
        pos_mean = score.detach().float().mean(dim=0)
        if layer not in running_scores:
            running_scores[layer] = pos_mean.clone()
            running_sensitivities[layer] = float(batch_sensitivities[layer])
            continue
        running_scores[layer] += (pos_mean - running_scores[layer]) / count
        running_sensitivities[layer] += (float(batch_sensitivities[layer]) - running_sensitivities[layer]) / count
    return count


def finalize_calibration_config(
    config: LossAwarePruningConfig,
    num_tokens_full: int,
    per_layer_mean_token_scores: dict[int, torch.Tensor],
    layer_sensitivities: dict[int, float],
    *,
    gp: int | None = None,
) -> LossAwarePruningConfig:
    out = LossAwarePruningConfig.from_dict(asdict(config))
    out.use_offline_calibration = True
    out.importance_source = "calibrated"
    out.num_tokens_full = int(num_tokens_full)

    ratios = build_prune_ratio_schedule(out, layer_sensitivities)
    out.per_layer_prune_ratios = ratios
    verify_cascade_global_ratio(ratios, out.global_prune_ratio)

    # Attach scores via caller (save_calibration_artifacts); keep in-memory for tests.
    out._calibrated_scores_cache = {int(k): v.detach().float().clone() for k, v in per_layer_mean_token_scores.items()}  # type: ignore[attr-defined]
    return out


ScoreProvider = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]
