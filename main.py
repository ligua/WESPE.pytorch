# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optimizer
from torch.utils.data import DataLoader

import model
from data.data_provider import DPEDDataset


def save_model(
        name,
        gennet_g,
        gennet_f,
        discrinet_c,
        discrinet_t):
    """

    :param name:
    :param gennet_g:
    :param gennet_f:
    :param discrinet_c:
    :param discrinet_t:
    :return:
    """

    torch.save(gennet_f.state_dict(), '%s_gennet_f.pth' % name)
    torch.save(gennet_g.state_dict(), '%s_gennet_g.pth' % name)
    torch.save(discrinet_c.state_dict(), '%s_discrinet_c.pth' % name)
    torch.save(discrinet_t.state_dict(), '%s_discrinet_t.pth' % name)


def train_step(use_cuda,
               vgg,
               generator_g,
               generator_f,
               discriminator_c,
               discriminator_t,
               gray_layer,
               blur_layer,
               input_image,
               target_image,
               optimizers
               ):
    """
    :param use_cuda: 
    :param vgg: 
    :param generator_g: 
    :param generator_f: 
    :param discriminator_c:
    :param discriminator_t: 
    :param gray_layer: 
    :param blur_layer: 
    :param input_image: 
    :param target_image: 
    :param optimizers: 
    :return: 
    """

    loss = dict()

    batch_size = input_image.size()[0]

    content_criterion = nn.L1Loss()
    texture_criterion = nn.CrossEntropyLoss()
    color_criterion = nn.CrossEntropyLoss()
    tv_criterion = model.TVLoss(1.0)

    if use_cuda:
        content_criterion = content_criterion.cuda()
        texture_criterion = texture_criterion.cuda()
        color_criterion = color_criterion.cuda()
        tv_criterion = tv_criterion.cuda()

    g_optimizer = optimizers['g']
    f_optimizer = optimizers['f']
    c_optimizer = optimizers['c']
    t_optimizer = optimizers['t']

    # Generator
    y_fake = generator_g(input_image)
    x_fake = generator_f(y_fake)

    # Content
    x_fake_features = vgg(x_fake)
    x_features = vgg(input_image).detach()
    _, c1, h1, w1 = x_fake_features.size()
    chw1 = c1 * h1 * w1
    content_loss = 1.0/chw1 * content_criterion(x_fake_features, x_features)

    # TV Loss
    _, c2, h2, w2 = y_fake.size()
    chw2 = c2 * h2 * w2
    tv_loss = 1.0/chw2 * tv_criterion.forward(y_fake)

    # Train Discriminators

    # Color Discriminator
    y_blur = blur_layer(target_image).detach()
    y_fake_blur = blur_layer(y_fake)

    random_discrim_label = Variable(torch.LongTensor(np.random.randint(0, 2, batch_size)))
    blur_input = Variable(torch.zeros((batch_size, c2, h2, w2)))
    if use_cuda:
        random_discrim_label = random_discrim_label.cuda()
        blur_input = blur_input.cuda()
    tmp_random_label = random_discrim_label.float()
    for batch_index in xrange(batch_size):
        blur_input[batch_index] = tmp_random_label[batch_index] * y_blur[batch_index] \
                                  + (1-tmp_random_label[batch_index]) * y_fake_blur[batch_index].detach()

    y_color_predict = discriminator_c(blur_input)
    # print(y_color_predict.size(), random_discrim_label.size())
    color_loss = color_criterion(y_color_predict, random_discrim_label)
    loss['discrim_color_loss'] = color_loss.data[0]

    # Texture Discriminator
    y_texture = gray_layer(target_image).detach()
    y_fake_texture = gray_layer(y_fake)
    random_discrim_label = Variable(torch.LongTensor(np.random.randint(0, 2, batch_size)))
    texture_input = Variable(torch.zeros((batch_size, 1, h2, w2)))
    if use_cuda:
        random_discrim_label = random_discrim_label.cuda()
        texture_input = texture_input.cuda()
    tmp_random_label = random_discrim_label.float()
    for batch_index in xrange(batch_size):
        texture_input[batch_index] = tmp_random_label[batch_index] * y_texture[batch_index] \
                                  + (1-tmp_random_label[batch_index]) * y_fake_texture[batch_index].detach()
    y_texture_predict = discriminator_t(texture_input)
    texture_loss = texture_criterion(y_texture_predict, random_discrim_label)
    loss['discrim_texture_loss'] = texture_loss.data[0]

    # optimizing Discriminator
    c_optimizer.zero_grad()
    t_optimizer.zero_grad()
    color_loss.backward()
    texture_loss.backward()
    g_optimizer.step()
    f_optimizer.step()

    # Train Generator
    label = Variable(torch.LongTensor([1]*batch_size))
    if use_cuda:
        label = label.cuda()

    y_blur_fake_prediction = discriminator_c(y_fake_blur)
    color_loss = color_criterion(y_blur_fake_prediction, label)

    y_texture_fake_prediction = discriminator_t(y_fake_texture)
    texture_loss = color_criterion(y_texture_fake_prediction, label)

    total_loss = content_loss + 5*10**(-3)*(color_loss + texture_loss) + 10*tv_loss

    loss['content_loss'] = content_loss.data[0]
    loss['tv_loss'] = tv_loss.data[0]
    loss['fake_color_loss'] = color_loss.data[0]
    loss['fake_texture_loss'] = texture_loss.data[0]
    loss['total_loss'] = total_loss.data[0]

    g_optimizer.zero_grad()
    f_optimizer.zero_grad()

    total_loss.backward()

    c_optimizer.step()
    t_optimizer.step()

    return loss


def train(opt, images):

    gennet_g = model.Generator()
    gennet_f = model.Generator()

    vgg = model.VGG()

    discrinet_c = model.Discriminator(3)
    discrinet_t = model.Discriminator(1)

    gray_layer = model.GrayLayer(opt.use_cuda)
    gaussian = model.GaussianBlur()

    if opt.use_cuda:
        gennet_g = gennet_g.cuda()
        gennet_f = gennet_f.cuda()
        discrinet_c = discrinet_c.cuda()
        discrinet_t = discrinet_t.cuda()
        gray_layer = gray_layer.cuda()
        gaussian = gaussian.cuda()
        vgg = vgg.cuda()

    g_optimizer = optimizer.Adam(params=gennet_g.parameters(), lr=opt.lr)
    f_optimizer = optimizer.Adam(params=gennet_f.parameters(), lr=opt.lr)
    c_optimizer = optimizer.Adam(params=discrinet_c.parameters(), lr=opt.lr)
    t_optimizer = optimizer.Adam(params=discrinet_t.parameters(), lr=opt.lr)

    optimizers = dict()
    optimizers['g'] = g_optimizer
    optimizers['f'] = f_optimizer
    optimizers['c'] = c_optimizer
    optimizers['t'] = t_optimizer

    dataset = DPEDDataset(opt.data_dir)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    num_samples = len(dataset)
    dataiter = iter(dataloader)
    for i in xrange(opt.epoches):

        for j in xrange(num_samples/opt.batch_size):

            batch_images, batch_target = dataiter.next()
            batch_images = Variable(batch_images)
            batch_target = Variable(batch_target)

            if opt.use_cuda:

                batch_target = batch_target.cuda()
                batch_images = batch_images.cuda()

            loss = train_step(
                use_cuda=opt.use_cuda,
                vgg=vgg,
                generator_f=gennet_f,
                generator_g=gennet_g,
                discriminator_c=discrinet_c,
                discriminator_t=discrinet_t,
                gray_layer=gray_layer,
                blur_layer=gaussian,
                input_image=batch_images,
                target_image=batch_target,
                optimizers=optimizers
            )

            print("\nEpoch: %s Batch: %s" % (i, j))
            print("Discriminator:")
            print("Color Loss: %s " % loss['discrim_color_loss'])
            print("Texture Loss: %s " % loss['discrim_texture_loss'])
            print("Generator:")
            print("Total Loss: %s " % loss['total_loss'])
            print("Content Loss: %s " % loss['content_loss'])
            print("Color Loss: %s " % loss['fake_color_loss'])
            print("Texture Loss: %s " % loss['fake_texture_loss'])
            print("TV Loss: %s " % loss['tv_loss'])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='DPED Dataset')
    parser.add_argument('--use_cuda', type=int, default=1, help='use gpu to train')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='training learning rate')
    parser.add_argument('--epoches', type=int, default=32, help='training epoches')

    args = parser.parse_args()

    train(args, '')


if __name__ == '__main__':
    main()





