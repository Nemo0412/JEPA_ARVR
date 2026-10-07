"""Count all registered model parameters for the actual EGTEA adapters."""

from __future__ import annotations

import json
import os
import importlib.util
from collections import defaultdict
from pathlib import Path

from torch import nn

from app.scalant_baseline.egtea_cache import MAIN_ROOT, ROOT
from app.scalant_baseline.official_stream_train import file_sha256, make_model_and_optimizer


def by_top_level(model) -> dict[str, int]:
    groups = defaultdict(int)
    for name, parameter in model.named_parameters():
        groups[name.split(".", 1)[0]] += parameter.numel()
    return dict(sorted(groups.items()))


def main() -> None:
    scalant, optimizer, scheduler = make_model_and_optimizer(2)
    scalant_parts = by_top_level(scalant)
    scalant_total = sum(parameter.numel() for parameter in scalant.parameters())
    temporal_encoder = sum(parameter.numel() for parameter in scalant.encoder.parameters())
    clam_module = sum(parameter.numel() for parameter in scalant.clustering.parameters())
    assert scalant_total == sum(scalant_parts.values())
    assert temporal_encoder == scalant_parts["encoder"]
    assert clam_module == scalant_parts["clustering"]
    del optimizer, scheduler

    # The fixed-horizon wrapper adds no parameters. Instantiate its unchanged
    # upstream core directly so this audit needs no files from the main repo.
    upstream = ROOT / "refer_repo/rulstm/RULSTM/models.py"
    if file_sha256(upstream) != "31137d0a2e0ad42200e4c3de4365aab3c09681500f519eb4e9ba592f90ea72d0":
        raise ValueError("RU-LSTM model source differs from the counted experiment")
    spec = importlib.util.spec_from_file_location("official_rulstm_models", upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rulstm = module.RULSTM(106, 1024, 1024, dropout=0.8, depth=1)
    rulstm_parts = by_top_level(rulstm)
    rulstm_total = sum(parameter.numel() for parameter in rulstm.parameters())
    assert rulstm_total == sum(rulstm_parts.values())
    if rulstm_total != 16902250:
        raise ValueError("RU-LSTM instantiated total differs from formal run metadata")

    # RU-LSTM's released RGB feature-extraction example uses this exact
    # BNInception constructor and removes last_linear before saving 1024-D
    # features. TSN's three temporal segments share one CNN parameter set.
    tsn_source = MAIN_ROOT / "outputs/b19_scalant/deps_tsn/bninception.py"
    tsn_spec = importlib.util.spec_from_file_location("pinned_bninception", tsn_source)
    tsn_module = importlib.util.module_from_spec(tsn_spec)
    tsn_spec.loader.exec_module(tsn_module)
    tsn = tsn_module.bninception(pretrained=None)
    removed_classifier = sum(parameter.numel() for parameter in tsn.last_linear.parameters())
    feature_width = tsn.last_linear.in_features
    tsn.last_linear = nn.Identity()
    tsn.global_pool = nn.AdaptiveAvgPool2d(1)
    tsn_visual_total = sum(parameter.numel() for parameter in tsn.parameters())
    if feature_width != 1024:
        raise ValueError("Expected RU-LSTM RGB TSN 1024-D feature encoder")

    report = {
        "kind": "all_registered_model_parameters_not_trainable_only",
        "scalant_egtea": {
            "model": "ScalAnt QueryPredictor with CLAM, split-1 +2s EGTEA configuration",
            "with_temporal_encoder": scalant_total,
            "temporal_encoder": temporal_encoder,
            "without_temporal_encoder": scalant_total - temporal_encoder,
            "with_tsn_visual_encoder": scalant_total + tsn_visual_total,
            "without_temporal_encoder_with_tsn_visual_encoder":
                scalant_total - temporal_encoder + tsn_visual_total,
            "clam_module_only": clam_module,
            "by_top_level_module": scalant_parts,
            "checkpoint": str(MAIN_ROOT / "outputs/b19_scalant/train/official_s1_h2/best.pt"),
            "configuration": str(ROOT / "refer_repo/scalable-anticipation/configs/ek100/default.yaml"),
        },
        "rulstm_egtea_rgb": {
            "model": "Pinned upstream RGB RULSTM core; parameter-free fixed-horizon wrapper omitted, 106 classes, 1024 input/hidden, depth 1",
            "total": rulstm_total,
            "with_tsn_visual_encoder": rulstm_total + tsn_visual_total,
            "by_core_module": rulstm_parts,
            "upstream_model_source": str(upstream),
            "upstream_model_source_sha256": file_sha256(upstream),
            "expected_anticipation_parameters": 16902250,
        },
        "tsn_rgb_feature_encoder": {
            "architecture": "pretrainedmodels.bninception; last_linear replaced by Identity",
            "feature_width": feature_width,
            "total": tsn_visual_total,
            "removed_default_classifier_parameters": removed_classifier,
            "extraction_example": str(ROOT / "refer_repo/rulstm/FEATEXT/extract_example_rgb.py"),
            "source_scope": "Architecture count for the shared EGTEA RGB TSN BNInception feature extractor; split-specific TSN weights are not stored locally.",
            "source": str(tsn_source),
            "source_sha256": file_sha256(tsn_source),
        },
        "scope_note": "With-TSN totals count one shared RGB BNInception feature encoder plus each anticipation model. The TSN recognition classifier is removed by the released feature-extraction script and is excluded. ScalAnt also contains a distinct Mamba temporal encoder.",
    }
    output = MAIN_ROOT / "outputs/b19_scalant/implementation/scalant_rulstm_parameter_count.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
