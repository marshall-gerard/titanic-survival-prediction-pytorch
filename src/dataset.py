import torch
from torch.utils.data import Dataset
import pandas as pd

class TitanicDataset(Dataset):
    def __init__(self, features_path, labels_path=None):
        self.X = pd.read_csv(features_path).values
        self.X = torch.tensor(self.X, dtype=torch.float32)
        if labels_path:
            self.y = pd.read_csv(labels_path).values
            self.y = torch.tensor(self.y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]