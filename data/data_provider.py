# -*- coding: utf-8 -*-

"""
Data Provider

DPED Dataset see:http://people.ee.ethz.ch/~ihnatova/

"""
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


to_tensor = transforms.Compose([
    transforms.ToTensor()
])


class DPEDDataset(Dataset):

    def __init__(self, data_dir, phone='iphone'):

        self.phone_dir = data_dir + phone + '/'
        self.camera = data_dir+'canon/'

        self.num_images = len([name for name in os.listdir(self.phone_dir)
                               if os.path.isfile(os.path.join(self.phone_dir, name))])

        # Load Data
        self.input_images = []
        self.target_images = []
        for i in xrange(self.num_images):
            input_ = Image.open(self.phone_dir+'%s.jpg' % i)
            target_ = Image.open(self.camera+'%s.jpg' % i)

            self.input_images.append(to_tensor(input_))
            self.target_images.append(to_tensor(target_))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        return self.input_images[idx], self.target_images[idx]


