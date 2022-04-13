import pickle

import numpy as np
import torch

from modules.algorithms.base.OSLPP import get_num_classes


def _load_tensors(dataset, domain):
    with open(f'../features/{dataset}/nocrop_{domain}.plk', 'rb') as f:
        features, labels = pickle.load(f)
    features, labels = features.numpy(), labels.numpy()
    assert len(features) == len(labels)
    return features, labels


def create_datasets(dataset, source, target, num_common, num_src_priv, num_tgt_priv):
    num_src, num_total = get_num_classes(num_common, num_src_priv, num_tgt_priv)

    src_features, src_labels = _load_tensors(dataset, source)
    idxs = src_labels < num_src
    src_features, src_labels = src_features[idxs], src_labels[idxs]
    print(np.unique(src_labels), np.arange(0, num_src))
    assert (np.unique(src_labels) == np.arange(0, num_src)).all()

    tgt_features, tgt_labels = _load_tensors(dataset, target)
    idxs = (tgt_labels < num_common) + ((tgt_labels >= num_src) * (tgt_labels < num_total))
    tgt_features, tgt_labels = tgt_features[idxs], tgt_labels[idxs]
    tgt_labels[tgt_labels >= num_src] = num_src

    assert len(src_features) == len(src_labels)
    assert len(tgt_features) == len(tgt_labels)

    return (src_features, src_labels), (tgt_features, tgt_labels)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
