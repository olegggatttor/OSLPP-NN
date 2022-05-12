import pickle
from typing import Optional

import numpy as np
import torch

from modules.augmentation.transformations import Augmentation
from modules.loaders.images.filelist_loader import read_filelist


def _load_tensors_sub_images(dataset, domain, mode, common_classes, tgt_private_classes, transform=None):
    assert dataset == 'DomainNet_DCC'
    features, labels = read_filelist(f'/mnt/tank/scratch/tpolevaya/datasets/domainnet_cleaned/{domain}_{mode}.txt',
                                     common_classes, tgt_private_classes, transform)
    return features, labels


def _load_tensors_sub(dataset, domain, mode, augmentation_test):
    with open(f'./features/{dataset}/nocrop_{domain}_{mode}.plk', 'rb') as f:
        if augmentation_test and mode != 'train':
            features, labels = torch.load(f)
        else:
            features, labels = pickle.load(f)
    features, labels = features.numpy(), labels.numpy()
    assert len(features) == len(labels)
    return features, labels


def create_datasets_sub(dataset, source, target, common_classes, tgt_private_classes, transform: Optional[Augmentation],
                        is_images=False,
                        augmentation_test=False):
    assert (augmentation_test and transform is not None) or not augmentation_test

    if is_images:
        src_features, src_labels = _load_tensors_sub_images(dataset, source, 'train',
                                                            common_classes,
                                                            tgt_private_classes)
    else:
        src_features, src_labels = _load_tensors_sub(dataset, source, 'train', augmentation_test)
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
        assert not augmentation_test
        tgt_features, tgt_labels = _load_tensors_sub_images(dataset, target, 'test',
                                                            common_classes,
                                                            tgt_private_classes)
    else:
        if augmentation_test:
            assert dataset == 'DomainNet_DCC'
            tgt_features, tgt_labels = _load_tensors_sub(f'DomainNet_DCC_aug/{transform.name}', target, 'test_raw', augmentation_test)
        else:
            tgt_features, tgt_labels = _load_tensors_sub(dataset, target, 'test', augmentation_test)
    idxs = np.isin(tgt_labels, common_classes) + np.isin(tgt_labels, tgt_private_classes)
    tgt_features, tgt_labels = tgt_features[idxs], tgt_labels[idxs]

    priv = len(remapper.values())

    tgt_labels = np.array(list(map(lambda x: remapper[x] if x in remapper else get_new(x), tgt_labels)))

    tgt_labels[tgt_labels >= priv] = priv

    assert len(src_features) == len(src_labels)
    assert len(tgt_features) == len(tgt_labels)

    return (src_features, src_labels), (tgt_features, tgt_labels)
