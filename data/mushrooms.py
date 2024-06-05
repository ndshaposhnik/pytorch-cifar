import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import sklearn.datasets


def _normalize_dataframe(df):
    return (df - df.mean()) / df.std()


class Mushrooms(Dataset):
    def __init__(self):
        X, y = sklearn.datasets.load_svmlight_file('data/mushrooms.libxvm')

        self.data = torch.from_numpy(X.toarray()).float()
        self.data = _normalize_dataframe(self.data)

        self.labels = torch.from_numpy(y).long()
        self.labels -= torch.ones_like(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

