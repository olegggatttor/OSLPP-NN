import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from modules.algorithms.base.OSLPP import get_l2_normalized


def get_euclidian_distances(a, b):
    return torch.square(a - b).sum(dim=-1)


def get_cosine_distances(a, b):
    a = get_l2_normalized(a)
    b = get_l2_normalized(b)
    return 1 - (a * b).sum(dim=-1)


def get_pairwise_euclidian_distances(a, b):
    n, d = a.shape
    m, d2 = b.shape
    assert d == d2
    return torch.square(a.view(n, 1, d) - b.view(1, m, d)).sum(dim=-1)

def get_pairwise_cosine_distances(a, b):
    a = get_l2_normalized(a)
    b = get_l2_normalized(b)
    n, d = a.shape
    m, d2 = b.shape
    assert d == d2
    return 1 - (a.view(n, 1, d) * b.view(1, m, d)).sum(dim=-1)


def get_cosine_similarity(anchor, key):
    # both shapes Nxd
    anchor = get_l2_normalized(anchor)
    key = get_l2_normalized(key)
    return (anchor * key).sum(dim=-1)


def get_bulk_cosine_similarity(anchor, keys):
    # anchor - Nxd, neg - NxKxd
    anchor = get_l2_normalized(anchor).unsqueeze(1)
    keys = get_l2_normalized(keys)
    return (anchor * keys).sum(dim=-1)


def get_centroids(features, labels):
    return torch.stack([features[labels == c].mean(dim=0) for c in labels.unique()], dim=0)


def get_l2_normalized(t: torch.Tensor):
    l2 = t.square().sum(dim=-1, keepdim=True).sqrt()
    return t / l2


class TripletDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.mapping = {c.item(): set(torch.where(labels == c)[0].tolist()) for c in labels.unique()}

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        c = self.labels[i].item()
        pos = np.random.choice(sorted(self.mapping[c].difference({i})), size=1).item()
        c_neg = np.random.choice(sorted(set(self.mapping.keys()).difference({c})), size=1).item()
        neg = np.random.choice(sorted(self.mapping[c_neg]), size=1).item()
        return {'anchor': (self.features[i], self.labels[i]),
                'pos': (self.features[pos], self.labels[pos]),
                'neg': (self.features[neg], self.labels[neg])}


class TripletLoss(nn.Module):
    def __init__(self, distance_fn, margin):
        super().__init__()
        self.distance_fn = distance_fn
        self.margin = margin

    def forward(self, anchor, pos, neg):
        return F.relu(self.distance_fn(anchor, pos) - self.distance_fn(anchor, neg) + self.margin).mean()


class InfoNCEDataset(Dataset):
    def __init__(self, feats, lbls, pos, neg):
        self.features = feats
        self.labels = lbls
        self.mapping = {c.item(): set(torch.where(lbls == c)[0].tolist()) for c in lbls.unique()}
        self.pos = pos
        self.neg = neg

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        c = self.labels[i].item()
        pos = np.random.choice(sorted(self.mapping[c].difference({i})), size=self.pos, replace=False)
        c_neg = np.random.choice(sorted(set(self.mapping.keys()).difference({c})), size=self.neg)
        neg = np.stack([np.random.choice(sorted(self.mapping[c]), size=1).item() for c in c_neg])
        return {'anchor': (self.features[i], self.labels[i]),
                'pos': (self.features[pos], self.labels[pos]),
                'neg': (self.features[neg], self.labels[neg])}


class InfoNCELoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, anchor, pos, neg):
        p_sim = get_bulk_cosine_similarity(anchor, pos) / self.tau
        n_sim = get_bulk_cosine_similarity(anchor, neg) / self.tau
        p_sim = p_sim.exp().sum(dim=1)
        n_sim = n_sim.exp().sum(dim=1)
        return -torch.log(p_sim / (p_sim + n_sim)).mean()
