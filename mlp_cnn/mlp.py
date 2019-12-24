import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, num_classes)


    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

def main():
    transform = transforms.ToTensor()
    writer = SummaryWriter('./logs/')
    mlp = MLP(10)
    
    data_train = MNIST(root='./data', download=True, train=True, transform=transform)
    data_test = MNIST(root='./data', download=True, train=False, transform=transform)

    print(len(data_train), len(data_test))

    train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp.parameters(), 0.001)
    for i, data in enumerate(train_loader):
        X, y = data
        out = mlp(X)
        optimizer.zero_grad()
        loss = criterion(out, y)
        print(loss.item(), i)
        writer.add_scalar('Loss/train', loss.item(), i)
        loss.backward()

        optimizer.step()
        


if __name__ == '__main__':
    main()