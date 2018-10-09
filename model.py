# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn
import torch.optim as optim


class ConvBlock(nn.Module):

    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(64, affine=True)
        self.instance_norm2 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.instance_norm1(self.conv1(x)))
        y = self.relu(self.instance_norm2(self.conv2(y))) + x
        return y


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class GrayLayer(nn.Module):

    def __init__(self):
        super(GrayLayer, self).__init__()

    def forward(self, x):
        result = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        return result.unsqueeze(1)


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
        self.conv4 = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.blocks(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.tanh(self.conv4(x)) * 0.58 + 0.5

        return x


class Discriminator(nn.Module):

    def __init__(self, input_ch):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_ch, 48, 11, stride=4, padding=5),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(48, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.Conv2d(128, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.Conv2d(192, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True),
        )

        self.fc = nn.Linear(128*7*7, 1024)
        self.out = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 128*7*7)
        x = F.leaky_relu(self.fc(x), negative_slope=0.2)
        x = F.softmax(self.out(x))
        return x


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.model = vgg19_bn(True).features
        self.mean = torch.Tensor([123.68,  116.779,  103.939]).cuda().view(1,3,1,1)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x*255 - self.mean
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

    @staticmethod
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class WESPE:

    def __init__(self, config, cuda=True, training=True):

        self.generator_g = Generator()
        self.generator_f = Generator()

        self.training = training
        self.cuda = cuda

        if self.cuda:
            self.generator_f.cuda()
            self.generator_g.cuda()

        if self.training:
            self.discriminator_c = Discriminator(input_ch=3)
            self.discriminator_t = Discriminator(input_ch=1)

            self.content_criterion = nn.L1Loss()
            self.tv_criterion = TVLoss(config.tv_weight)
            self.color_criterion = nn.CrossEntropyLoss()
            self.texture_criterion = nn.CrossEntropyLoss()

            self.g_optimizer = optim.Adam(lr=config.generator_g_lr, params=self.generator_g.parameters())
            self.f_optimizer = optim.Adam(lr=config.generator_f_lr, params=self.generator_f.parameters())
            self.t_optimizer = optim.Adam(lr=config.discriminator_t_lr, params=self.discriminator_t.parameters())
            self.c_optimizer = optim.Adam(lr=config.discriminator_c_lr, params=self.discriminator_c.parameters())

            self.vgg = VGG()
            self.blur = GaussianBlur()
            self.gray = GrayLayer()

            if self.cuda:
                self.discriminator_c.cuda()
                self.discriminator_t.cuda()
                self.content_criterion = self.content_criterion.cuda()
                self.tv_criterion = self.tv_criterion.cuda()
                self.vgg = self.vgg.cuda()
                self.color_criterion = self.color_criterion.cuda()
                self.texture_criterion = self.texture_criterion.cuda()
                self.blur = self.blur.cuda()
                self.gray = self.gray.cuda()

            if len(config.model_path):
                self.load_model(config.model_path, False)
        else:
            if len(config.model_path):
                self.load_model(config.model_path, True)

    def train_step(self, x, y):

        batch_size = x.size()[0]

        # generate
        y_fake = self.generator_g(x)
        x_fake = self.generator_f(y_fake)

        # ------- train generator ------- #

        self.g_optimizer.zero_grad()
        self.f_optimizer.zero_grad()
        # content loss
        # x_fake.requires_grad = True
        vgg_x_true = self.vgg(x).detach()
        vgg_x_fake = self.vgg(x_fake)
        _, c1, h1, w1 = x_fake.size()
        chw1 = c1 * h1 * w1
        content_loss = 1.0/chw1 * self.content_criterion(vgg_x_fake, vgg_x_true)

        # TV loss
        _, c2, h2, w2 = y_fake.size()
        chw2 = c2 * h2 * w2
        tv_loss = 1.0/chw2 * self.tv_criterion.forward(y_fake)

        pos_labels = torch.LongTensor([1]*batch_size).to(x.device)
        neg_labels = torch.LongTensor([0]*batch_size).to(x.device)

        y_fake_blur = self.blur(y_fake)
        y_real_blur = self.blur(y)

        y_fake_blur_dc_pred = self.discriminator_c(y_fake_blur)
        y_real_blur_dc_pred = self.discriminator_c(y_real_blur)
        gen_dc_loss = self.color_criterion(y_fake_blur_dc_pred, pos_labels)

        y_fake_gray = self.gray(y_fake)
        y_real_gray = self.gray(y)

        y_fake_gray_dt_pred = self.discriminator_t(y_fake_gray)
        y_real_gray_dt_pred = self.discriminator_t(y_real_gray)
        gen_dt_loss = self.texture_criterion(y_fake_gray_dt_pred, pos_labels)

        gen_loss = content_loss + tv_loss + (gen_dc_loss + gen_dt_loss) * 5 * 1e-3

        gen_loss.backward()

        self.g_optimizer.step()
        self.f_optimizer.step()

        # ------- train discriminator -------- #

        self.c_optimizer.zero_grad()
        self.t_optimizer.zero_grad()

        y_fake_blur_dc_pred = self.discriminator_c(y_fake_blur.detach())
        dc_loss = self.color_criterion(y_fake_blur_dc_pred, neg_labels) \
            + self.color_criterion(y_real_blur_dc_pred, pos_labels)

        y_fake_gray_dt_pred = self.discriminator_t(y_fake_gray.detach())
        dt_loss = self.texture_criterion(y_fake_gray_dt_pred, neg_labels) \
            + self.texture_criterion(y_real_gray_dt_pred, pos_labels)

        discri_loss = (dt_loss + dc_loss) * 5 * 1e-3

        discri_loss.backward()

        self.c_optimizer.step()
        self.t_optimizer.step()

        loss_dict = {
            "content": content_loss.item(),
            "tv": tv_loss.item(),
            "gen_dc": gen_dc_loss.item(),
            "gen_dt": gen_dt_loss.item(),
            "texture_loss": dt_loss.item(),
            "color_loss": dc_loss.item()
        }

        return loss_dict

    def save_model(self, model_path):
        torch.save(self.generator_f.state_dict(), model_path+'_generator_f.pth')
        torch.save(self.generator_g.state_dict(), model_path+'_generator_g.pth')
        torch.save(self.discriminator_t.state_dict(), model_path+'_discriminator_t.pth')
        torch.save(self.discriminator_c.state_dict(), model_path+'_discriminator_c.pth')

    def load_model(self, model_path, only_gen):
        self.generator_f.load_state_dict(torch.load(model_path+'_generator_f.pth'))
        self.generator_g.load_state_dict(torch.load(model_path+'_generator_g.pth'))
        if not only_gen:
            self.discriminator_c.load_state_dict(torch.load(model_path + '_discriminator_c.pth'))
            self.discriminator_t.load_state_dict(torch.load(model_path + '_discriminator_t.pth'))

    def inference(self, x):

        return self.generator_g(x)





