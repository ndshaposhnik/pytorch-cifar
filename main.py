'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from mushroom_dataset import LibSVMDataset


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')

trainset = LibSVMDataset(svm_file='mushrooms.libxvm')
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2)


N_FEATURES = 112
N_CLASSES = 2

# Model
print('==> Building model..')

model = LogisticRegression(n_inputs=N_FEATURES, n_outputs=N_CLASSES)
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

dim = sum(p.numel() for p in model.parameters() if p.requires_grad)


NUMBER_OF_EPOCHS = 300

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.compressedSGD(model.parameters(), dim=dim, lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
#optimizer = optim.SGD(model.parameters(), lr=args.lr,
#                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.PolinomialLR(optimizer, total_iters=NUMBER_OF_EPOCHS)


loss_history = []
transmitted_coordinates_history = []

# Training
def train(epoch):
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


for epoch in range(start_epoch, start_epoch+NUMBER_OF_EPOCHS):
    train(epoch)
    with open('loss_history.txt', "w") as f:
        print(*loss_history, sep='\n', file=f)
    with open('transmitted_coordinates_history.txt', "w") as f:
        print(*transmitted_coordinates_history, sep='\n', file=f)
    print(*loss_history)
    print(*transmitted_coordinates_history)
    # test(epoch)
    if loss_history[-1] < 0.5:
      break
    scheduler.step()

with open('loss_history.txt', "w") as f:
    print(*loss_history, sep='\n', file=f)

with open('transmitted_coordinates_history.txt', "w") as f:
    print(*transmitted_coordinates_history, sep='\n', file=f)



def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
