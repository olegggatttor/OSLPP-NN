import numpy as np


def mean(arr: np.ndarray):
    if arr.shape[0] == 0:
        return np.nan
    return arr.mean()


def get_acc_c(labels: np.ndarray, predictions: np.ndarray, classes: np.ndarray):
    per_class_acc = np.array([(predictions[labels == c] == c).mean() for c in classes])
    return mean(per_class_acc)


def get_acc_i(labels: np.ndarray, predictions: np.ndarray):
    return mean(labels == predictions)


def get_h_score(labels: np.ndarray, predictions: np.ndarray, common_classes: np.ndarray, unk_class: int):
    knw = get_acc_c(labels, predictions, common_classes)
    unk = mean(predictions[labels == unk_class] == unk_class)
    h_score = 2 * knw * unk / (knw + unk)
    return h_score
