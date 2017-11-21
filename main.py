# -*- coding: utf-8 -*-

import argparse

import torchvision
import torch
import torch.nn as nn
import torch.optim as optimizer

import model


def train_step(
        vgg,
        gennet_g,
        gennet_f,
        discrinet_c,
        discrinet_t,
        content_criterion,
        texture_critetion,
        color_criterion,
        tv_criterion,
        x,
        y,
        optimizers
):

    y_fake = gennet_g(x)
    x_fake = gennet_f(y_fake)

    x_fake_features = vgg(x)
    x_features = vgg(x_fake).detach()

    content_loss = content_criterion(x_fake_features, x_features)
    tv_loss = tv_criterion(y_fake)

    y_fake_color_predict = discrinet_c(y_fake)
    y_color_predict = discrinet_c(y).detach()

    color_loss = 0.5*(color_criterion(y_fake_color_predict, False) + color_criterion(y_color_predict, True))

    y_fake_texture_predict = discrinet_t(y_fake)
    y_texture_predict = discrinet_t(y).detach()

    texture_loss = 0.5*(texture_critetion(y_fake_texture_predict, False) + texture_critetion(y_texture_predict, True))

    loss = dict()

    loss['tv_loss'] = tv_loss
    loss['color_loss'] = color_loss
    loss['content_loss'] = content_loss
    loss['texture_loss'] = texture_loss

    g_optimizer = optimizers['g']
    f_optimizer = optimizers['f']
    c_optimizer = optimizers['c']
    t_optimizer = optimizers['t']

    g_optimizer.zero_grad()
    f_optimizer.zero_grad()
    c_optimizer.zero_grad()
    t_optimizer.zero_grad()

    total_loss = content_loss + 5*10**(-3)*(color_loss + texture_loss) + 10*tv_loss
    total_loss.backward()
    loss['total_loss'] = total_loss

    g_optimizer.step()
    f_optimizer.step()
    c_optimizer.step()
    t_optimizer.step()

    return loss


def generate_batches(images, batch_size):
    """
    Generate image batches
    :param images:
    :param batch_size:
    :return:
    """
    yield {
        'x': 0,
        'y': 0
    }



def train(opt, images):

    gennet_g = model.Generator()
    gennet_f = model.Generator()
    vgg = model.VGG()

    discrinet_c = model.Discriminator(3)
    discrinet_t = model.Discriminator(1)

    g_optimizer = optimizer.Adam(params=gennet_g.parameters(), lr=opt.lr)
    f_optimizer = optimizer.Adam(params=gennet_f.parameters(), lr=opt.lr)
    c_optimizer = optimizer.Adam(params=discrinet_c.parameters(), lr=opt.lr)
    t_optimizer = optimizer.Adam(params=discrinet_t.parameters(), lr=opt.lr)

    optimizers = dict()
    optimizers['g'] = g_optimizer
    optimizers['f'] = f_optimizer
    optimizers['c'] = c_optimizer
    optimizers['t'] = t_optimizer

    content_criterion = nn.L1Loss()
    texture_criterion = model.GANLoss()
    color_criterion = model.GANLoss()
    tv_criterion = model.TVLoss(1.0)

    num_samples = 0

    for i in xrange(opt.epoches):

        for j in xrange(num_samples/opt.batch_size):

            batch = generate_batches(images, opt.batch_size)
            loss = train_step(
                vgg=vgg,
                gennet_g=gennet_g,
                gennet_f=gennet_f,
                discrinet_c=discrinet_c,
                discrinet_t=discrinet_t,
                content_criterion=content_criterion,
                color_criterion=color_criterion,
                texture_critetion=texture_criterion,
                tv_criterion=tv_criterion,
                x=batch['x'],
                y=batch['y'],
                optimizers=optimizers
            )

            print("\nEpoch: %s\n" % i)
            print("Total Loss: %s \n" % loss['total_loss'])
            print("Color Loss: %s \n" % loss['color_loss'])
            print("Content Loss: %s \n" % loss['content_loss'])
            print("TV Loss: %s \n" % loss['tv_loss'])
            print("Texture Loss: %s \n" % loss['texture_loss'])

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='training learning rate')
    parser.add_argument('--epoches', type=int, default=32, help='training epoches')






