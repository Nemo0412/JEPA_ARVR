import torch
from torch import Tensor
import json


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def dump_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


def action2verbnoun(
        res_action: Tensor,
        class_mappings: dict[tuple[str, str], Tensor]
) -> tuple[Tensor, Tensor]:
    res_verb = torch.matmul(res_action, class_mappings[('verb', 'action')].to(res_action.device))
    res_noun = torch.matmul(res_action, class_mappings[('noun', 'action')].to(res_action.device))
    return res_verb, res_noun


def verbnoun2action(
        res_verb: Tensor,
        res_noun: Tensor,
        verb_noun_to_action: dict[tuple[int, int], int]
) -> Tensor:
    verb_ids, noun_ids = zip(*verb_noun_to_action.keys())

    # Convert to tensors
    verb_ids = torch.tensor(verb_ids, device=res_verb.device)
    noun_ids = torch.tensor(noun_ids, device=res_noun.device)

    # Index into the verb and noun probabilities
    verb_action_probs = res_verb[..., verb_ids]
    noun_action_probs = res_noun[..., noun_ids]

    # Calculate action probabilities
    res_action = verb_action_probs * noun_action_probs  # element-wise multiplication

    return res_action

