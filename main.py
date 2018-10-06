# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch.optim as optimizer
from torch.utils.data import DataLoader
from config import config
from model import WESPE
from data_provider import DPEDDataset


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, help='DPED Dataset')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='use gpu to train')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--epoches', type=int, default=32, help='training epoches')
    parser.add_argument('--infer', action='store_true', default=False, help='training or inference')
    parser.add_argument('--phone', type=str, default='iphone', help='Phone type')
    parser.add_argument('-s', '--save_model_path', type=str, default='', help='model path')

    args = parser.parse_args()

    wespe = WESPE(config,
                  cuda=args.use_cuda,
                  training=not args.infer)

    dataset = DPEDDataset(data_dir=args.data_dir, phone=args.phone)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    for epoch in range(args.epoches):
        train_iter = iter(data_loader)
        for i in range(len(data_loader)):

            x, y = next(train_iter)
            if args.use_cuda:
                x = x.cuda()
                y = y.cuda()

            loss = wespe.train_step(x, y)

            print("e:{} i:{} content loss: {}, tv loss:{}, gen_color_loss: {}, gen_texture_loss:{}, "
                  "discri_color_loss: {}, discri_texture_loss: {}".format(epoch, i, loss['content'],
                                                                          loss['tv'], loss['gen_dc'],
                                                                          loss['gen_dt'], loss['color_loss'],
                                                                          loss['texture_loss']))
        wespe.save_model(args.save_model_path)


if __name__ == '__main__':
    main()





