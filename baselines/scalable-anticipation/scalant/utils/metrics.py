import torch


@torch.inference_mode()
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions
    for the specified values of k
    Args:
        output (*, K) predictions
        target (*, ) targets
    """
    # flatten the initial dimensions, to deal with 3D+ input
    output = output.flatten(0, -2)

    if target.ndim == 3:
        target = target.argmax(dim=-1)

    target = target.flatten()
    # Now compute the accuracy
    maxk = max(topk)
    counts = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)
        res.append(correct_k * (100.0 / counts))
    return res, counts


