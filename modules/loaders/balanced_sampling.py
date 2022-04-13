from collections import Counter

from torch.utils.data import DataLoader, WeightedRandomSampler

from modules.loaders.resnet_features.features_dataset import FeaturesDataset


def _create_balanced_sampler(ds: FeaturesDataset):
    freq2 = Counter(ds.labels.tolist())
    class_weight = {x: 1.0 / freq2[x] for x in freq2}
    source_weights = [class_weight[x] for x in ds.labels.tolist()]
    sampler = WeightedRandomSampler(source_weights, len(ds.labels.tolist()))
    return sampler


def create_train_dataloader(ds, batch_size, balanced):
    if balanced:
        return DataLoader(ds, batch_size=batch_size, sampler=_create_balanced_sampler(ds),
                          drop_last=True)
    else:
        DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
