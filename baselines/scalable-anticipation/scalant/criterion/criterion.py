import torch
from torch import Tensor

from scalant.config import Config
from scalant.utils import accuracy
from scalant.utils.ouput_target_structure import Prediction, Target
from scalant.criterion.build import Criterion_REGISTRY
from scalant.datasets import EpicKitchens, action2verbnoun, verbnoun2action
from scalant.criterion.loss import *


def _scored_labels(last_target: Tensor, scored: Tensor, ignore_index: int) -> Tensor:
    """Labels for mean top-5 recall in the notice (non-ignored) class space.

    Without `scored`, keep upstream behavior: argmax of the multi-hot target
    (the lowest class id when actions overlap). With `scored` (annotated
    full-space ids), score the annotated action instance instead.
    """
    if scored is None:
        return last_target.argmax(dim=-1)
    scored = scored.to(last_target.device).long()
    if ignore_index >= 0:
        scored = scored - (scored > ignore_index).long()
    return scored


def _mt5r_inputs(logits: Tensor, last_target: Tensor, scored: Tensor, observable: Tensor,
                 ignore_index: int) -> dict:
    """Logits/labels for MeanTopKRecallMeter. Unobservable clips (no feature
    before the cutoff) get scores that rank their label last: a counted miss."""
    labels = _scored_labels(last_target, scored, ignore_index)
    if observable is not None:
        keep = observable.to(logits.device).bool()
        if not keep.all():
            miss = torch.zeros_like(logits)
            miss.scatter_(1, labels.view(-1, 1), -1.0)
            logits = torch.where(keep[:, None], logits, miss)
    return {"logits": logits, "labels": labels}


@Criterion_REGISTRY.register()
class Criterion_LSTR:
    def __init__(self, cfg: Config, dataset: EpicKitchens):
        ignore_index = cfg.MODEL.IGNORE_INDEX

        if isinstance(dataset, EpicKitchens):
            self.action_cls = MultipCrossEntropyEqualizedLoss(ignore_index=ignore_index)
        elif getattr(cfg.MODEL, "EQUALIZED_LOSS", False) and getattr(dataset, "action_frequency", None) is not None:
            # Same equalized loss, with this dataset's training frequencies.
            self.action_cls = MultipCrossEntropyEqualizedLoss(
                ignore_index=ignore_index, freq_info=dataset.action_frequency)
        else:
            self.action_cls = MultipCrossEntropyLoss(ignore_index=ignore_index)
        self.verb_noun_cls = MultipCrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.dataset = dataset
        self.cfg = cfg

    def __call__(self, pred: Prediction, target: Target, is_training=True) -> (Tensor, dict):
        notice_index = [i for i in range(target.past_actions.shape[-1]) if i != self.ignore_index]
        past_cls = self.action_cls(pred.past_actions, target.past_actions)
        future_cls = self.action_cls(pred.future_actions, target.future_actions)

        loss = past_cls + future_cls

        # Compute metrics
        (past_top1,), past_counts = accuracy(
            pred.past_actions[..., notice_index], target.past_actions[..., notice_index])

        (future_top1,), future_counts = accuracy(
            pred.future_actions[..., notice_index], target.future_actions[..., notice_index])

        # Mean top 5
        mt5r_dict = _mt5r_inputs(pred.future_actions[:, -1, notice_index],
                                 target.future_actions[:, -1, notice_index],
                                 target.scored_actions, target.observable, self.ignore_index)

        loss_dict = {
            "past_cls_loss": past_cls.item(),
            "future_cls_loss": future_cls.item(),
            "past_top1": [None, past_top1, past_counts],
            "future_top1": [None, future_top1, future_counts],
            "mt5r": ["MeanTopKRecallMeter", mt5r_dict, None],
        }

        if pred.past_verbs is not None:
            past_verb = self.verb_noun_cls(pred.past_verbs, target.past_verbs)
            past_noun = self.verb_noun_cls(pred.past_nouns, target.past_nouns)
            future_verb = self.verb_noun_cls(pred.future_verbs, target.future_verbs)
            future_noun = self.verb_noun_cls(pred.future_nouns, target.future_nouns)

            loss += past_verb + past_noun + future_verb + future_noun

            loss_dict.update({
                "past_verb_loss": past_verb.item(),
                "past_noun_loss": past_noun.item(),
                "future_verb_loss": future_verb.item(),
                "future_noun_loss": future_noun.item(),
            })

            verb_notice_index = [i for i in range(target.future_verbs.shape[-1]) if i != self.ignore_index]
            noun_notice_index = [i for i in range(target.future_nouns.shape[-1]) if i != self.ignore_index]
            # Mean top 5
            verb_mt5r_dict = _mt5r_inputs(pred.future_verbs[:, -1, verb_notice_index],
                                          target.future_verbs[:, -1, verb_notice_index],
                                          target.scored_verbs, target.observable, self.ignore_index)
            noun_mt5r_dict = _mt5r_inputs(pred.future_nouns[:, -1, noun_notice_index],
                                          target.future_nouns[:, -1, noun_notice_index],
                                          target.scored_nouns, target.observable, self.ignore_index)
            loss_dict.update({
                "verb_mt5r_cls": ["MeanTopKRecallMeter", verb_mt5r_dict, None],
                "noun_mt5r_cls": ["MeanTopKRecallMeter", noun_mt5r_dict, None],
            })


        future_verbs, future_nouns = action2verbnoun(pred.future_actions, self.dataset.class_mappings)
        verb_notice_index = [i for i in range(target.future_verbs.shape[-1]) if i != self.ignore_index]
        noun_notice_index = [i for i in range(target.future_nouns.shape[-1]) if i != self.ignore_index]
        # Mean top 5
        verb_mt5r_dict = _mt5r_inputs(future_verbs[:, -1, verb_notice_index],
                                      target.future_verbs[:, -1, verb_notice_index],
                                      target.scored_verbs, target.observable, self.ignore_index)
        noun_mt5r_dict = _mt5r_inputs(future_nouns[:, -1, noun_notice_index],
                                      target.future_nouns[:, -1, noun_notice_index],
                                      target.scored_nouns, target.observable, self.ignore_index)
        loss_dict.update({
            "verb_mt5r": ["MeanTopKRecallMeter", verb_mt5r_dict, None],
            "noun_mt5r": ["MeanTopKRecallMeter", noun_mt5r_dict, None],
        })

        return loss, loss_dict

