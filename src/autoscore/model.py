import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import argparse
from os import listdir, walk
from os.path import join
from random import shuffle

DEF_H = 50
DEF_W = 10

DEF_DIR_FN = '/home/feras/Documents/auto-score/datasets/Artificial/data/'
DEF_TRANSFORM = transforms.Compose([transforms.Resize((DEF_H, DEF_W), interpolation=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])


class DatasetLoader(Dataset):
    def __init__(self, dir_fn=DEF_DIR_FN, transform=DEF_TRANSFORM, train=True):
        super(DatasetLoader, self).__init__()
        self.transform = transform
        self.X_fn = []
        class_names = listdir(dir_fn)
        self.lookup = {name: i for i, name in enumerate(class_names)}
        for name in class_names:
            for prefix, _, files in walk(join(dir_fn, name)):
                for file in files:
                    x = join(prefix, file), self.lookup[name]
                    self.X_fn.append(x)
        shuffle(self.X_fn)
        if not train:
            self.X_fn = self.X_fn[0:1000]

    def __len__(self):
        return len(self.X_fn)

    def __getitem__(self, i):
        fn, label = self.X_fn[i]
        img = Image.open(fn)
        if self.transform:
            img = self.transform(img)
        return img, label


class Net(nn.Module):
    def __init__(self, n_classes, n_conv=2, n_maps=30):
        super(Net, self).__init__()
        self.n_maps = n_maps
        self.conv = []
        self.conv.append(nn.Conv2d(1, n_maps, kernel_size=3))
        for i in range(n_conv):
            if i == 0:
                continue
            self.conv.append(nn.Conv2d(i * n_maps, (i+1) * n_maps, kernel_size=3))
        self.fc_size = self._get_conv_size(DEF_H, DEF_W)
        self.fc1 = nn.Linear(self.fc_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, n_classes)

    def forward(self, x):
        for conv in self.conv:
            x = F.max_pool2d(F.relu(conv(x)), 2)
        x = x.view(-1, self.fc_size)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)1
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def _reduce(x, kernel_size=3):
        return (x - (kernel_size - 1)) // (kernel_size - 1)

    def _get_conv_size(self, x, y):
        for _ in range(len(self.conv)):
            y = self._reduce(y)
            x = self._reduce(x)
        return x * y * len(self.conv) * self.n_maps


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    my_loader = DatasetLoader()
    train_loader = DataLoader(my_loader, batch_size=10, shuffle=True)
    # test_loader = DataLoader(DatasetLoader(train=False))

    model = Net(n_classes=len(my_loader.lookup))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        # test(args, model, test_loader)


if __name__ == '__main__':
    main()
