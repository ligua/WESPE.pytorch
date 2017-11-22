# -*- coding: utf-8 -*-

import argparse
import cv2 as cv
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optimizer
from torchvision import transforms
import numpy as np
import model


def gray(image):
    R = image[0]
    G = image[1]
    B = image[2]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor


def batch_gray(images):
    """
    Grayscale for Batch Images
    :param images: [Batch, Channel, W, H]
    :return: [Batch, W, H]
    """
    batch_size = images.size()[0]
    h = images.size()[2]
    w = images.size()[3]

    result = Variable(torch.zeros([batch_size, h, w]))
    for i in xrange(batch_size):
        result[i] = gray(images[i])

    return result


# to Tensor [Batch, Channel, W, H]
transform1 = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform2 = transforms.Compose([
    transforms.ToPILImage(),
])


def batch_gaussian(images):

    batch_size = images.size()[0]

    result = Variable(torch.zeros(images.size()))

    for i in xrange(batch_size):

        img = images[i]
        img = transform2(img.data)
        img = np.asarray(img)
        blurred = cv.GaussianBlur(img, ksize=(0, 0), sigmaX=3)
        img = transform1(blurred)
        result[i] = img

    return result


def save_model(
        name,
        gennet_g,
        gennet_f,
        discrinet_c,
        discrinet_t,
):

    torch.save(gennet_f.state_dict(), '%s_gennet_f.pth' % name)
    torch.save(gennet_g.state_dict(), '%s_gennet_g.pth' % name)
    torch.save(discrinet_c.state_dict(), '%s_discrinet_c.pth' % name)
    torch.save(discrinet_t.state_dict(), '%s_discrinet_t.pth' % name)


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

    # Generator
    y_fake = gennet_g(x)
    x_fake = gennet_f(y_fake)

    # Content
    x_fake_features = vgg(x_fake)
    x_features = vgg(x).detach()

    content_loss = content_criterion(x_fake_features, x_features) / \
                   (x_fake_features.size()[1]*x_fake_features.size()[2]*x_fake_features.size()[3])

    tv_loss = tv_criterion(y_fake) / (y_fake.size()[1]*y_fake.size()[2]*y_fake.size()[3])

    # Colot Discriminator
    y_blur_fake = batch_gaussian(y_fake)
    y_blur = batch_gaussian(y)
    y_fake_color_predict = discrinet_c(y_blur_fake)
    y_color_predict = discrinet_c(y_blur).detach()

    color_loss = 0.5*(color_criterion(y_fake_color_predict, False) + color_criterion(y_color_predict, True))

    # Texture Discriminator
    y_gray_fake = batch_gray(y_fake)
    y_gray = batch_gray(y)
    y_fake_texture_predict = discrinet_t(y_gray_fake)
    y_texture_predict = discrinet_t(y_gray).detach()

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






