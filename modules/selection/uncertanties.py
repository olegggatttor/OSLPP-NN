import math
from enum import Enum
from functools import reduce

import numpy as np
import torch


from modules.types.selection_types import SelectRejectMode

def get_confidence_scores(pseudo_probs) -> np.ndarray:
    return pseudo_probs


def get_entropy_scores(predictions) -> np.ndarray:
    max_entropy = np.log2(predictions.shape[1])
    entropy = -(predictions * torch.log2(predictions)).mean(dim=1).numpy()
    return 1 - entropy / max_entropy


def get_margin_scores(predictions) -> np.ndarray:
    probs_k, _ = torch.topk(predictions, 2)
    margin = probs_k[:, 0] - probs_k[:, 1]
    return margin.numpy()


def get_weighted_scores(pseudo_probs, predictions, weights=np.array([0.33, 0.33, 0.33])):
    conf = get_confidence_scores(pseudo_probs)
    inv_entropy = get_entropy_scores(predictions)
    margin = get_margin_scores(predictions)

    scores = weights[0] * conf + weights[1] * inv_entropy + weights[2] * margin
    return scores


def get_all_tops(class_probs, class_predictions, tops):
    indexed_conf = get_confidence_scores(class_probs) if SelectRejectMode.CONFIDENCE in tops else None
    indexed_inv_entropy = get_entropy_scores(class_predictions) if SelectRejectMode.ENTROPY in tops else None
    indexed_margin = get_margin_scores(class_predictions) if SelectRejectMode.MARGIN in tops else None

    all_tops = list(filter(lambda x: x is not None, [indexed_conf, indexed_inv_entropy, indexed_margin]))

    assert all_tops != []
    return all_tops


def select_initial_rejected(pseudo_probs, predictions, n_r, mode: SelectRejectMode, weights=None, tops=None):
    is_rejected = np.zeros((len(pseudo_probs),), dtype=np.int)
    if mode == SelectRejectMode.CONFIDENCE:
        scores = get_confidence_scores(pseudo_probs)
    elif mode == SelectRejectMode.ENTROPY:
        scores = get_entropy_scores(predictions)
    elif mode == SelectRejectMode.MARGIN:
        scores = get_margin_scores(predictions)
    elif mode == SelectRejectMode.WEIGHTED:
        if weights is None:
            raise Exception("Weights must be not None with mode SelectRejectMode.WEIGHTED")
        scores = get_weighted_scores(pseudo_probs, predictions, weights)
    elif mode == SelectRejectMode.TOPS:
        if tops is None:
            raise Exception("Tops must be not None with mode SelectRejectMode.TOPS")
        all_tops = get_all_tops(pseudo_probs, predictions, tops)
        rejects_tops = list(map(lambda scores: np.argsort(scores)[:2 * n_r], all_tops))
        is_rejected[reduce(np.intersect1d, rejects_tops)] = 1
        return is_rejected
    else:
        raise Exception("Wrong select/reject mode.")
    is_rejected[np.argsort(scores)[:n_r]] = 1
    return is_rejected


def get_new_rejected(pseudo_probs, predictions, selected, rejected, mode: SelectRejectMode, weights=None, tops=None):
    if mode == SelectRejectMode.CONFIDENCE:
        scores = get_confidence_scores(pseudo_probs)
    elif mode == SelectRejectMode.ENTROPY:
        scores = get_entropy_scores(predictions)
    elif mode == SelectRejectMode.MARGIN:
        scores = get_margin_scores(predictions)
    elif mode == SelectRejectMode.WEIGHTED:
        scores = get_weighted_scores(pseudo_probs, predictions, weights)
    elif mode == SelectRejectMode.TOPS:
        conf = get_confidence_scores(pseudo_probs.numpy()) if SelectRejectMode.CONFIDENCE in tops else None
        inv_entropy = get_entropy_scores(predictions) if SelectRejectMode.ENTROPY in tops else None
        margin = get_margin_scores(predictions) if SelectRejectMode.MARGIN in tops else None

        all_rejects = list(filter(lambda x: x is not None, [conf, inv_entropy, margin]))

        def reject_by_scores(scores, rejected, selected):
            return scores < min(scores[rejected == 1].max(), scores[selected == 1].min())

        mask = np.logical_and.reduce(list(map(lambda x: reject_by_scores(x, rejected, selected), all_rejects)))
        return mask
    else:
        raise Exception("Wrong select/reject mode.")
    scores = torch.tensor(scores)
    return (scores < min(scores[rejected == 1].max().item(), scores[selected == 1].min().item())).int()


def select_closed_set_pseudo_labels_by_mode(class_indices, selected, pseudo_probs, predictions,
                                            t, T, Nc,
                                            mode: SelectRejectMode,
                                            uniform_ratio=15, balanced=False, weights=None, tops=None):
    if mode == SelectRejectMode.CONFIDENCE:
        scores = pseudo_probs[class_indices]
        class_probs = np.sort(scores)
        if balanced and t != T - 1:
            to_draw_border = math.floor(uniform_ratio * t / (T - 1))
            threshold = list(reversed(class_probs))[min(to_draw_border, Nc - 1)]
        else:
            threshold = class_probs[math.floor(Nc * (1 - t / (T - 1)))]
    elif mode == SelectRejectMode.ENTROPY:
        scores = get_entropy_scores(predictions[class_indices])
        entropy = np.sort(scores)
        threshold = entropy[math.floor(Nc * (1 - t / (T - 1)))]
    elif mode == SelectRejectMode.MARGIN:
        scores = get_margin_scores(predictions[class_indices])
        margin = np.sort(scores)
        threshold = margin[math.floor(Nc * (1 - t / (T - 1)))]
    elif mode == SelectRejectMode.WEIGHTED:
        if weights is None:
            raise Exception("Weights must be not None with mode SelectRejectMode.WEIGHTED")
        class_probs = pseudo_probs[class_indices]
        class_predictions = predictions[class_indices]
        scores = get_weighted_scores(class_probs, class_predictions, weights)
        weighted = np.sort(scores)
        threshold = weighted[math.floor(Nc * (1 - t / (T - 1)))]
    elif mode == SelectRejectMode.TOPS:
        def get_threshold(all_scores, Nc, top_length_multiplier=2):
            idx_tr = math.floor(Nc * (1 - t / (T - 1)))
            return all_scores[max(len(all_scores) - (len(all_scores) - idx_tr) * top_length_multiplier, 0)]

        def get_mask(arr, thresh):
            return arr >= thresh

        class_probs = pseudo_probs[class_indices]
        class_predictions = predictions[class_indices]

        all_tops = get_all_tops(class_probs, class_predictions, tops)

        sorted_all_tops = list(map(np.sort, all_tops))
        threshold_all_tops = list(map(lambda x: get_threshold(x, Nc), sorted_all_tops))
        masks_all_tops = list(map(lambda x: get_mask(*x), zip(all_tops, threshold_all_tops)))

        selected_indices = class_indices[np.logical_and.reduce(masks_all_tops)]
        assert (selected[selected_indices] == 0).all()
        selected[selected_indices] = 1
        return selected
    else:
        raise Exception("Wrong select/reject mode.")
    mask = scores >= threshold
    selected_indices = class_indices[mask]
    assert (selected[selected_indices] == 0).all()
    selected[selected_indices] = 1
    return selected
