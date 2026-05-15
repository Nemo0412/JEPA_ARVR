import logging
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from evals.action_anticipation_frozen.models import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger(__name__)


class Top3AccuracyRecallAt5:
    """Metric adapter for matching the reference script's reporting convention.

    The upstream V-JEPA action anticipation eval instantiates ClassMeanRecall(k=5)
    and logs the returned "accuracy" and "recall" fields. In upstream code,
    both are based on top-5 predictions. The reference HD-EPIC script reports
    Top-3 accuracy and class-mean Recall@5, so this adapter keeps the same return
    keys while changing only "accuracy" to Top-3.
    """

    def __init__(self, num_classes: int, device: torch.device, k=5):
        self.num_classes = num_classes
        self.top3_tp = torch.zeros(num_classes).to(device)
        self.top3_fn = torch.zeros(num_classes).to(device)
        self.r5_tp = torch.zeros(num_classes).to(device)
        self.r5_fn = torch.zeros(num_classes).to(device)

    def __call__(self, logits, labels, valid_classes=None, eps=1e-8):
        logits = F.sigmoid(logits)

        if valid_classes is not None:
            filtered = torch.zeros(logits.shape).to(logits.device)
            for c in valid_classes:
                filtered[:, c] = logits[:, c]
            logits = filtered

        k3 = min(3, logits.shape[1])
        k5 = min(5, logits.shape[1])
        preds3 = logits.topk(k3, dim=1).indices
        preds5 = logits.topk(k5, dim=1).indices

        for p3, p5, gt in zip(preds3, preds5, labels):
            if gt in p3:
                self.top3_tp[gt] += 1
            else:
                self.top3_fn[gt] += 1
            if gt in p5:
                self.r5_tp[gt] += 1
            else:
                self.r5_fn[gt] += 1

        top3_tp, top3_fn = self.top3_tp.clone(), self.top3_fn.clone()
        r5_tp, r5_fn = self.r5_tp.clone(), self.r5_fn.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(top3_tp)
            dist.all_reduce(top3_fn)
            dist.all_reduce(r5_tp)
            dist.all_reduce(r5_fn)

        top3_total = torch.sum(top3_tp + top3_fn)
        top3_accuracy = 100.0 * torch.sum(top3_tp) / torch.clamp(top3_total, min=1.0)

        r5_seen = torch.sum((r5_tp + r5_fn) > 0)
        r5_recall = 100.0 * torch.sum(r5_tp / (r5_tp + r5_fn + eps)) / torch.clamp(r5_seen, min=1)

        return dict(recall=r5_recall, accuracy=top3_accuracy)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _replace_linears_with_lora(module: nn.Module, rank: int, alpha: float, dropout: float, prefix: str = ""):
    replaced = []
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(child_prefix)
        else:
            replaced.extend(_replace_linears_with_lora(child, rank, alpha, dropout, child_prefix))
    return replaced


def _load_pooler_from_probe(classifier: AttentiveClassifier, checkpoint_path: str):
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning("LoRA pretrained probe not found: %s", checkpoint_path)
        return

    checkpoint = robust_checkpoint_loader(str(path), map_location=torch.device("cpu"))
    state_dicts = checkpoint.get("classifiers", [])
    if not state_dicts:
        logger.warning("No classifier state dicts found in probe checkpoint: %s", checkpoint_path)
        return

    source = state_dicts[0]
    target = classifier.state_dict()
    pooler_state = {}
    for key, value in source.items():
        clean_key = key.removeprefix("module.")
        if clean_key.startswith("pooler.") and clean_key in target and target[clean_key].shape == value.shape:
            pooler_state[clean_key] = value

    missing, unexpected = classifier.load_state_dict(pooler_state, strict=False)
    logger.info(
        "Loaded %d pooler tensors from %s; ignored heads and mismatches. missing=%d unexpected=%d",
        len(pooler_state),
        checkpoint_path,
        len(missing),
        len(unexpected),
    )


def _freeze_for_lora(classifier: AttentiveClassifier, train_heads: bool):
    for param in classifier.parameters():
        param.requires_grad = False
    for module in classifier.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True
    if train_heads:
        for name, param in classifier.named_parameters():
            if name.startswith(("verb_classifier.", "noun_classifier.", "action_classifier.")):
                param.requires_grad = True


def _log_trainable_params(classifier: nn.Module):
    total = sum(p.numel() for p in classifier.parameters())
    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    logger.info("LoRA classifier trainable params: %d / %d (%.2f%%)", trainable, total, pct)


def _make_lora_init_classifier(lora_cfg):
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16.0))
    dropout = float(lora_cfg.get("dropout", 0.05))
    train_heads = bool(lora_cfg.get("train_heads", True))
    pretrained_probe = lora_cfg.get("pretrained_probe", None)

    def init_classifier(
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        device: torch.device,
        num_classifiers: int,
        action_classes: dict,
        verb_classes: dict,
        noun_classes: dict,
    ):
        classifiers = []
        for _ in range(num_classifiers):
            classifier = AttentiveClassifier(
                verb_classes=verb_classes,
                noun_classes=noun_classes,
                action_classes=action_classes,
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=num_blocks,
                use_activation_checkpointing=True,
            )
            _load_pooler_from_probe(classifier, pretrained_probe)
            replaced = _replace_linears_with_lora(classifier.pooler, rank=rank, alpha=alpha, dropout=dropout)
            _freeze_for_lora(classifier, train_heads=train_heads)
            logger.info("Inserted LoRA into %d pooler Linear layers", len(replaced))
            _log_trainable_params(classifier)
            classifiers.append(classifier.to(device))

        print(classifiers[0])
        return classifiers

    return init_classifier


def main(args_eval, resume_preempt=False):
    lora_cfg = args_eval.get("experiment", {}).get("lora", {})
    if not lora_cfg.get("enabled", True):
        raise ValueError("app.hdepic_lora_action_anticipation requires experiment.lora.enabled=true")

    import evals.action_anticipation_frozen.eval as base_eval

    if bool(lora_cfg.get("align_reference_metrics", True)):
        logger.info("Using aligned metrics: accuracy=Top-3, recall=class-mean Recall@5")
        base_eval.ClassMeanRecall = Top3AccuracyRecallAt5

    base_eval.init_classifier = _make_lora_init_classifier(lora_cfg)
    return base_eval.main(args_eval=args_eval, resume_preempt=resume_preempt)
