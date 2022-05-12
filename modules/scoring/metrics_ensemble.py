import numpy as np
import torch

from modules.types.selection_types import SelectRejectMode


def _get_entropy(probs):
    entropy = - probs * probs.log()
    entropy = entropy.sum(dim=-1)
    return entropy


def _vote_for_lbls(pred_lbls):
    return torch.mode(pred_lbls, dim=0)[0]


def get_labels_multi_models(all_logits):
    pred_lbls = all_logits.max(dim=-1)[1]
    return _vote_for_lbls(pred_lbls)


def get_data_U(all_probs): return _get_entropy(all_probs).mean(dim=0)


def get_total_U(all_probs): return _get_entropy(all_probs.mean(dim=0))


def get_know_U(all_probs): return get_total_U(all_probs) - get_data_U(all_probs)


def get_consistency(all_probs): return all_probs.std(dim=0).mean(dim=-1)


def get_agreement(all_probs):
    pred_lbls = all_probs.max(dim=-1)[1]
    most_frequent_lbls = _vote_for_lbls(pred_lbls)
    return (pred_lbls == most_frequent_lbls).float().mean(dim=0)


def get_mean_confs(all_probs): return all_probs.max(dim=-1)[0].mean(dim=0)


def get_mean_margins(all_probs):
    probs = all_probs.sort(dim=-1)[0]
    margins = probs[:, :, -1] - probs[:, :, -2]
    return margins.mean(dim=0)


def invert(fn): return lambda x: -fn(x)


def get_score_total_U(x):
    return invert(get_total_U)(x)


def get_weighted_scores_multi(probs, weights=np.array([0.33, 0.33, 0.33])):
    conf = get_mean_confs(probs).numpy()
    margin = get_mean_margins(probs).numpy()

    max_entropy = np.log2(probs.shape[-1])
    total_U = 1 + get_score_total_U(probs).numpy() / max_entropy

    scores = weights[0] * conf + weights[1] * margin + weights[2] * total_U
    return scores


def get_all_tops_multi(class_predictions, tops):
    indexed_conf = get_mean_confs(class_predictions).numpy() if SelectRejectMode.CONFIDENCE_MULTI in tops else None
    indexed_margin = get_mean_margins(class_predictions).numpy() if SelectRejectMode.MARGIN_MULTI in tops else None

    max_entropy = np.log2(class_predictions.shape[-1])
    indexed_total_U = 1 + get_score_total_U(class_predictions).numpy() / max_entropy if SelectRejectMode.TOTAL_U in tops else None

    all_tops = list(filter(lambda x: x is not None, [indexed_conf, indexed_margin, indexed_total_U]))

    assert all_tops != []
    return all_tops
