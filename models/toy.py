import torch
from torch import nn

class ToyModel(nn.Module):
    def __init__(self, init_dim, num_classes=10):
        super(ToyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.net1 = nn.Linear(init_dim, 20)
        self.net2 = nn.Linear(20, 20)
        self.net3 = nn.Linear(20, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.net3(self.relu(self.net2(self.relu(self.net1(self.flatten(x))))))


class LogisticRegression(nn.Module):
    def __init__(self, init_dim, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(init_dim, num_classes)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
