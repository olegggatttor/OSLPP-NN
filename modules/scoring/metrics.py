import numpy as np
import torch

from modules.types.selection_types import SelectRejectMode


def mean(arr: np.ndarray):
    if arr.shape[0] == 0:
        return np.nan
    return arr.mean()


def get_acc_c(labels: np.ndarray, predictions: np.ndarray, classes: np.ndarray):
    per_class_acc = np.array([(predictions[labels == c] == c).mean() if predictions[labels == c] != [] else 0
                              for c in classes])
    return mean(per_class_acc)


def get_acc_i(labels: np.ndarray, predictions: np.ndarray):
    return mean(labels == predictions)


def get_h_score(labels: np.ndarray, predictions: np.ndarray, common_classes: np.ndarray, unk_class: int):
    knw = get_acc_c(labels, predictions, common_classes)
    unk = mean(predictions[labels == unk_class] == unk_class)
    h_score = 2 * knw * unk / (knw + unk)
    return h_score


def get_confidence_scores(pseudo_probs) -> np.ndarray:
    return pseudo_probs


def get_labels_single_model(probs: torch.Tensor):
    assert probs.shape[0] == 1
    return probs[0].max(dim=1)[1]


def get_entropy_scores(predictions) -> np.ndarray:
    max_entropy = np.log2(predictions.shape[-1])
    entropy = -(predictions * torch.log2(predictions)).sum(dim=1).numpy()
    res = 1 - entropy / max_entropy
    return res


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
