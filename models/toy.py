from torch import nn

class ToyModel(nn.Module):
    def __init__(self, init_dim=28*28, num_classes=10):
        super(ToyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.net1 = nn.Linear(init_dim, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, num_classes)
    def forward(self, x):
        return self.net2(self.relu(self.net1(self.flatten(x))))
