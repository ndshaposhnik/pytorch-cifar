import os
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

def base_setup_data(batch_size=64, download=False, use_cuda=False):
    root = './data'
    
    if not os.path.isdir(os.path.join(root, 'MNIST')):
        download = True

    train_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=download,
    )
    
    return train_dataset
