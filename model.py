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


class TVLoss(nn.Module):

    def __init__(self, tv_weight):
        super(TVLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, x):

        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]),2).sum()
        return self.tv_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):

        return t.size()[1]*t.size()[2]*t.size()[3]


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)







