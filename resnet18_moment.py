import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from pid import PIDOptimizer
from torch.optim.sgd import SGD
import os
import numpy as np
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.nn.functional as F
# find segmentation fault
import faulthandler

faulthandler.enable()
# Hyper Parameters
input_size = 784
hidden_size = 1000
num_classes = 10
num_epochs = 100
batch_size = 128
learning_rate = 0.001

I = 3
I = float(I)
D = 130
D = float(D)

logger = Logger('resnet18_moment.txt', title='resenet18_moment')
logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR10 Dataset
train_dataset = dsets.CIFAR10(root='./data',
                              train=True,
                              transform=transform_train,
                              download=True)

test_dataset = dsets.CIFAR10(root='./data',
                             train=False,
                             transform=transform_test,
                             download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Resnet  Neural Network Model (1 hidden layer)
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)


net = ResNet18()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
net.train()
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
# Train the Model
for epoch in range(num_epochs):

    train_loss_log = AverageMeter()
    train_acc_log = AverageMeter()
    val_loss_log = AverageMeter()
    val_acc_log = AverageMeter()
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        train_loss_log.update(train_loss.item(), images.size(0))
        train_acc_log.update(prec1.item(), images.size(0))

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, train_loss_log.avg,
                     train_acc_log.avg))

    # Test the Model
    net.eval()
    correct = 0
    loss = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        outputs = net(images)
        test_loss = criterion(outputs, labels)
        val_loss_log.update(test_loss.data.item(), images.size(0))
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        val_acc_log.update(prec1.item(), images.size(0))

    logger.append([learning_rate, train_loss_log.avg, val_loss_log.avg, train_acc_log.avg, val_acc_log.avg])
    print('Accuracy of the network on the 10000 test images: %d %%' % (val_acc_log.avg))
    print('Loss of the network on the 10000 test images: %.8f' % (val_loss_log.avg))

logger.close()
logger.plot()

