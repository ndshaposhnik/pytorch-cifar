import torch
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
import sklearn.datasets
from typing import List


def _normalize_dataframe(df):
    return (df - df.mean()) / df.std()


class Mushrooms(Dataset):
    def __init__(self):
        X, y = sklearn.datasets.load_svmlight_file('data/mushrooms.libxvm')

        self.data = torch.from_numpy(X.toarray()).float()
        self.labels = torch.from_numpy(y).long()
        self.labels -= torch.ones_like(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def split_dataset(dataset: Dataset, n: int) -> List[Dataset]:
    lengths = [1 / n] * n
    return random_split(dataset, lengths)
