import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

class ResBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, ksize):
          super(ResBlock, self).__init__()
          self.inp_ch = inp_ch
          self.out_ch = out_ch
          self.ksize = ksize
          self.conv1 = nn.Conv2d(inp_ch, inp_ch, ksize, padding=2)
          self.conv2 = nn.Conv2d(inp_ch, out_ch, (1, 1))

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        # print(x1.size())
        # print(x.size())
        return self.conv2(x + x1)


class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, self.num_classes)


    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = ResBlock(5, 10, (5, 5))
        self.bn2 = nn.BatchNorm2d(10)
        self.classifier = nn.Linear(4410, self.num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.classifier(x)
        return x

def evaluate(mlp, test_loader, batch_size):
    val_acc = 0.0
    mlp.eval()
    for i, data in enumerate(test_loader):
        X, y = data
        out = mlp(X)
        out = F.softmax(out)
        pred = out.argmax(axis=1)
        val_acc += (pred == y).sum() / batch_size
    val_acc /= len(test_loader) 
    return val_acc

def main(args):
    train_transform = transforms.Compose([transforms.RandomRotation(5),
                                          transforms.RandomCrop((25, 25)), 
                                          transforms.ToTensor(), 
                                          ])
    test_transform = transforms.Compose([transforms.Resize((25, 25)), transforms.ToTensor()])
    writer = SummaryWriter('./logs/{}'.format(datetime.now()))
    
    if args.use_conv:
        net = CNN(args.num_classes)
    else: 
        net = MLP(args.num_classes)
    
    data_train = MNIST(root='./data', download=True, train=True, transform=train_transform)
    data_test = MNIST(root='./data', download=True, train=False, transform=test_transform)

    print(len(data_train), len(data_test))

    train_loader = DataLoader(data_train, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(data_test, batch_size=args.val_batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), args.lr)
    for e in range(args.total_epochs):
        for i, data in enumerate(train_loader):
            X, y = data
            out = net(X)
            optimizer.zero_grad()
            loss = criterion(out, y)
            print(loss.item(), e, i)
            writer.add_scalar('Loss/train', loss.item(),  e * len(train_loader) + i)
            loss.backward()

            optimizer.step()
            if i % 100 == 0:
                val_acc = evaluate(net, test_loader, args.val_batch_size)
                print(val_acc.item(), e, i)
                writer.add_scalar('Val/acc', val_acc, e * len(train_loader) + i)
            net.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Playing with MNIST.')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes.')
    parser.add_argument('--train_batch_size', default=32, type=int, help='Train batch size.')
    parser.add_argument('--val_batch_size', default=4, type=int, help='Val batch size.')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('--total_epochs', default=5, type=int, help='Total epochs for training.')
    parser.add_argument('--use_conv', action='store_true', help='Use CNN instead of MLP.')
    args = parser.parse_args()
    main(args)