'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
from tqdm import tqdm

from models import *
from utils import progress_bar
from data.mushrooms import Mushrooms
from data.mnist import base_setup_data


def parallel_train(epoch):
    global trainloader, model, device, criterion, optimizer, scheduler
    global loss_history, coords_history

    print('\nEpoch: %d' % epoch)

    grads = [] # : List[List[Tensor]]
    losses = []
    number_of_parameter_groups = len([p for p in model.parameters()])
 
    for worker in tqdm(range(NUM_WORKERS), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
        optimizer.zero_grad()
        grads.append([])
        losses.append(0)

        for p in model.parameters():
            if p.requires_grad:
                grads[worker].append(torch.clone(p))
            else:
                grads[worker].append(None)

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            losses[-1] += loss.item()

            for i, p in enumerate(model.parameters()):
                if p.requires_grad:
                    assert p.grad is not None, 'Unexpected None in p.grad'
                    grads[worker][i] += p.grad

        assert len(grads[worker]) == number_of_parameter_groups

    train_loss = sum(losses) / len(losses)

    for g in grads:
        g = g # compress gradients
    
    gradient = []
    for i in range(number_of_parameter_groups):
        avg_grad_in_group = sum([grads[worker][i] for worker in range(NUM_WORKERS)]) / NUM_WORKERS
        gradient.append(avg_grad_in_group)
    
    for i, p in enumerate(model.parameters()):
        if p.requires_grad:
            p.grad = gradient[i]
        else:
            assert gradient[i] is None

    optimizer.step()
    scheduler.step()

    loss_history.append(train_loss)
    # coords_history.append(compressor.last_coordinates_transmitted)
    print(f'Epoch: {epoch}, Loss: %.3f' % train_loss)


def main():
    global trainloader, model, device, criterion, optimizer, scheduler, NUM_WORKERS
    global loss_history, coords_history

    NUM_WORKERS = 10
    NUMBER_OF_EPOCHS = 30
        

    trainset = Mushrooms()
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2)


    model = LogisticRegression(init_dim=112, num_classes=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=NUMBER_OF_EPOCHS)


    loss_history = []
    coords_history = []

    for epoch in range(NUMBER_OF_EPOCHS):
        parallel_train(epoch)

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
