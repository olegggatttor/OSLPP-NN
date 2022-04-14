import pickle

import numpy as np
from torch.utils.data import Subset

from modules.loaders.images.image_loader import UNDAImageFolder


def _load_tensors_sub_images(dataset, domain, mode):
    assert dataset == 'DomainNet_DCC'
    with open(f'/mnt/tank/scratch/tpolevaya/datasets/domainnet_cleaned/{domain}_{mode}.txt', 'r') as f:
        info = f.readlines()
        valid_files = set(list(map(lambda x: x.split(' ')[0].split('/')[-1], info)))
    df = UNDAImageFolder(f'/mnt/tank/scratch/tpolevaya/datasets/domainnet_cleaned/{domain}',
                         mode,
                         is_valid_file=lambda x: x in valid_files)
    return df, np.array(df.targets)


def _load_tensors_sub(dataset, domain, mode):
    with open(f'./features/{dataset}/nocrop_{domain}_{mode}.plk', 'rb') as f:
        features, labels = pickle.load(f)
    features, labels = features.numpy(), labels.numpy()
    assert len(features) == len(labels)
    return features, labels


def create_datasets_sub(dataset, source, target, common_classes, tgt_private_classes, is_images=False):
    if is_images:
        src_features, src_labels = _load_tensors_sub_images(dataset, source, 'train')
        idxs = np.isin(src_labels, common_classes)
        src_features, src_labels = Subset(src_features, np.arange(0, len(src_labels))[idxs]), src_labels[idxs]
    else:
        src_features, src_labels = _load_tensors_sub(dataset, source, 'train')
        idxs = np.isin(src_labels, common_classes)
        src_features, src_labels = src_features[idxs], src_labels[idxs]
    assert (set(np.unique(src_labels)) == set(common_classes))

    remapper = {}
    next_i = 0

    def get_new(x):
        nonlocal remapper, next_i
        remapper[x] = next_i
        next_i += 1
        return remapper[x]

    src_labels = np.array(list(map(lambda x: remapper[x] if x in remapper else get_new(x), src_labels)))

    if is_images:
        tgt_features, tgt_labels = _load_tensors_sub_images(dataset, target, 'test')
        idxs = np.isin(tgt_labels, common_classes) + np.isin(tgt_labels, tgt_private_classes)
        tgt_features, tgt_labels = Subset(tgt_features, np.arange(0, len(tgt_labels))[idxs]), tgt_labels[idxs]
    else:
        tgt_features, tgt_labels = _load_tensors_sub(dataset, target, 'test')
        idxs = np.isin(tgt_labels, common_classes) + np.isin(tgt_labels, tgt_private_classes)
        tgt_features, tgt_labels = tgt_features[idxs], tgt_labels[idxs]

    priv = len(remapper.values())

    tgt_labels = np.array(list(map(lambda x: remapper[x] if x in remapper else get_new(x), tgt_labels)))

    tgt_labels[tgt_labels >= priv] = priv

    assert len(src_features) == len(src_labels)
    assert len(tgt_features) == len(tgt_labels)

    return (src_features, src_labels), (tgt_features, tgt_labels)
