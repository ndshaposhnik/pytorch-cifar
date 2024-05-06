from torch import nn

class ToyModel(nn.Module):
    def __init__(self, init_dim=28*28, num_classes=10):
        super(ToyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.net1 = nn.Linear(init_dim, 20)
        self.net2 = nn.Linear(20, 20)
        self.net3 = nn.Linear(20, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.net3(self.relu(self.net2(self.relu(self.net1(self.flatten(x))))))
