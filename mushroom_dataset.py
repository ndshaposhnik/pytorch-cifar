import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sklearn.datasets


def _normalize_dataframe(df):
    return (df - df.mean()) / df.std()


class LibSVMDataset(Dataset):
    def __init__(self, svm_file, normalize=True):
        X, y = sklearn.datasets.load_svmlight_file(svm_file)
        y -= np.ones_like(y)
        X = X.toarray()
        if normalize:
            X = _normalize_dataframe(X)
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

