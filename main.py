# -*- coding: utf-8 -*-

import argparse

import torchvision
import torch
import torch.optim as optimizer

import model


def train_step(vgg, gennet_g, gennet_f, discrinet_c, discrinet_t, x, y, optimizer):

    y_fake = gennet_g(x)
    x_fake = gennet_f(y_fake)

    x_fake_features = vgg(x)
    x_features = vgg(x_fake)

    content_loss =




def train(opt):

    gennet_g = model.Generator()
    gennet_f = model.Generator()
    vgg = model.VGG()

    discrinet_c = model.Discriminator(3)
    discrinet_t = model.Discriminator(3)

    g_optimizer = optimizer.Adam(params=gennet_g.parameters(), lr=opt.lr)
    f_optimizer = optimizer.Adam(params=gennet_f.parameters(), lr=opt.lr)
    c_optimizer = optimizer.Adam(params=discrinet_c.parameters(), lr=opt.lr)
    t_optimizer = optimizer.Adam(params=discrinet_t.parameters(), lr=opt.lr)

    optimizers = dict()
    optimizers['g'] = g_optimizer
    optimizers['f'] = f_optimizer
    optimizers['c'] = c_optimizer
    optimizers['t'] = t_optimizer





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')






