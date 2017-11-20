# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg19_bn


class ConvBlock(nn.Module):

    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.bn(x)
        x = F.relu(self.conv2(x))
        x = self.bn(x)

        return x


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.blocks = nn.Sequential(
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
        )
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.blocks(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        return x


class Discriminator(nn.Module):

    def __init__(self, input_ch):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, 48, 11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(48, 128, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 192, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 128, 3, stride=2, padding=1)
        self.fc = nn.Linear(1, 1024)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.relu(self.conv5(x))
        x = self.bn1(x)

        x = F.sigmoid(self.fc(x))
        return x


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.model = vgg19_bn(True).features

    def forward(self, x):
        x = self.model(x)
        return x







