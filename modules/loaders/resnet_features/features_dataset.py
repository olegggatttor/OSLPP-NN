from torch.utils.data import Dataset


class FeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, i): return self.features[i], self.labels[i]