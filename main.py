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
from data.mushrooms import Mushrooms
from data.mnist import base_setup_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 10
    

def parallel_train(epoch):
    global model, trainloader, optimizer, criterion
    global loss_history, transmitted_coordinates_history
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0

    for worker in range(NUM_WORKERS):

        for batch_idx, (inputs, targets) in enumerate(trainloader[worker]):
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

            progress_bar(batch_idx, len(trainloader), f'Worker #{worker+1}, Loss: %.3f' % train_loss)

        optimizer.step()

    loss_history.append(train_loss)
    transmitted_coordinates_history.append(optimizer.last_coordinates_transmitted)


def train(epoch):
    global model, trainloader, optimizer, criterion
    global loss_history, transmitted_coordinates_history
    print('\nEpoch: %d' % epoch)
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f' % train_loss)

    loss_history.append(train_loss)
    transmitted_coordinates_history.append(optimizer.last_coordinates_transmitted)


def main():
    global model, trainloader, optimizer, criterion
    global loss_history, transmitted_coordinates_history

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..')

    trainset = Mushrooms()
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)


    print('==> Building model..')
    model = LogisticRegression(init_dim=112, num_classes=2)
    model = model.to(device)
    model.train()
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    compressor = 'None'
    NUMBER_OF_EPOCHS = 30

    dim = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.compressedSGD(model.parameters(), dim=dim, compressor=compressor, device=device,
                                    lr=0.5, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=NUMBER_OF_EPOCHS)

    loss_history = []
    transmitted_coordinates_history = []

    for epoch in range(start_epoch, start_epoch+NUMBER_OF_EPOCHS):
        train(epoch)
        # parallel_train(epoch)
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
