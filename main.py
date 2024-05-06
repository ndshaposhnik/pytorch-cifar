'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import tqdm

from models import *
from utils import progress_bar
from data.mushroom_dataset import LibSVMDataset
from data.mnist import base_setup_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

def train(epoch):
    global model, trainloader, optimizer, criterion
    global loss_history, transmitted_coordinates_history
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    loss_history.append(train_loss)
    transmitted_coordinates_history.append(optimizer.last_coordinates_transmitted)


def main():
    global model, trainloader, optimizer, criterion
    global loss_history, transmitted_coordinates_history

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)


    print('==> Building model..')
    # model = ToyModel()
    model = SimpleDLA()
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    NUMBER_OF_EPOCHS = 300
    dim = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss()

    compressor = 'Mult'
    optimizer = optim.compressedSGD(model.parameters(), dim=dim, compressor=compressor, device=device,
                                    lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=NUMBER_OF_EPOCHS)

    loss_history = []
    transmitted_coordinates_history = []

    for epoch in range(start_epoch, start_epoch+NUMBER_OF_EPOCHS):
        train(epoch)
        scheduler.step()

    history_path = "exps/" + compressor
    for i in range(100): # Find first unused number
        end = ''
        if i > 0:
            end = f'_{i}'
        if not os.path.isdir(history_path + end):
            history_path += end
            break

    os.mkdir(history_path)

    with open(os.path.join(history_path, 'loss.txt'), "w") as f:
        print(*loss_history, sep='\n', file=f)

    with open(os.path.join(history_path, 'coords.txt'), "w") as f:
        print(*transmitted_coordinates_history, sep='\n', file=f)


if __name__ == '__main__':
    main()
