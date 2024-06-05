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
from data.mushrooms import Mushrooms, split_dataset
from data.mnist import base_setup_data
from markov_compressors import *


def parallel_train(epoch):
    global trainloaders, model, device, criterion, optimizer, scheduler, NUM_WORKERS, compressors
    global loss_history, coords_history

    print('\nEpoch: %d' % epoch)

    grads = [[]] * NUM_WORKERS # : List[List[Tensor]]
    losses = [0] * NUM_WORKERS
    number_of_parameter_groups = len([p for p in model.parameters()])
 
    for w in tqdm(range(NUM_WORKERS), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
        optimizer.zero_grad()
        losses.append(0)

        for p in model.parameters():
            if p.requires_grad:
                grads[w].append(torch.zeros_like(p))
            else:
                grads[w].append(None)

        for batch_idx, (inputs, targets) in enumerate(trainloaders[w]):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            losses[w] += loss.item()

            for i, p in enumerate(model.parameters()):
                if p.requires_grad:
                    assert p.grad is not None, 'Unexpected None in p.grad'
                    grads[w][i] += p.grad

        assert len(grads[w]) == number_of_parameter_groups

    train_loss = sum(losses) / len(losses)

    for w in range(NUM_WORKERS):
       apply_compressor(compressors[w], grads[w])
    
    gradient = []
    for i in range(number_of_parameter_groups):
        avg_grad_in_group = sum([grads[worker][i] for worker in range(NUM_WORKERS)]) / NUM_WORKERS
        gradient.append(avg_grad_in_group)

    optimizer.zero_grad()
    
    for i, p in enumerate(model.parameters()):
        if p.requires_grad:
            p.grad = gradient[i]
        else:
            assert gradient[i] is None

    optimizer.step()
    scheduler.step()

    loss_history.append(train_loss)
    coords_history.append(compressors[0].k)
    print(f'Epoch: {epoch}, Loss: %.3f' % train_loss)


def main():
    global trainloaders, model, device, criterion, optimizer, scheduler, NUM_WORKERS, compressors
    global loss_history, coords_history

    NUM_WORKERS = 1
    NUMBER_OF_EPOCHS = 50
        
    splitted_trainset = split_dataset(Mushrooms(), NUM_WORKERS)  
    trainloaders = [
        torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
        for dataset in splitted_trainset
    ]


    model = LogisticRegression(init_dim=112, num_classes=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    dim = sum(p.numel() for p in model.parameters() if p.requires_grad)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=NUMBER_OF_EPOCHS)

    compressor = NoneCompressor
    kwargs = {
        'dim': dim,
        'device': device,
    }
    compressors = [compressor(**kwargs) for _ in range(NUM_WORKERS)]


    loss_history = []
    coords_history = []

    for epoch in range(NUMBER_OF_EPOCHS):
        parallel_train(epoch)

    history_path = "exps/" + str(compressors[0])
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
        print(*coords_history, sep='\n', file=f)


if __name__ == '__main__':
    main()
