import os
import numpy as np
import torch
import random
import json
import matplotlib.pyplot as plt
import h5py
import scipy.io
import cv2
from PIL import Image, ImageEnhance
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import spectral

class ModidDataset(object):

    def __init__(self, root_dir, input_shape, train):
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.train = train
        self.channels = 16
        self.lam = np.array([460.48099, 465.215098, 474.1194, 482.578653, 492.59933, 504.367465, 512.469275, 521.987657, 533.930667, 541.203169, 551.996143, 559.719978, 570.1331, 579.729623, 585.387574, 595.018809])
        if self.train:
            data_dir = os.path.join(root_dir, 'train', 'data')
            label_dir = os.path.join(root_dir, 'train', 'label')
        else:
            data_dir = os.path.join(root_dir, 'test', 'data')
            label_dir = os.path.join(root_dir, 'test', 'label')
        self.hsi = list(sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.hdr')]))
        self.label = list(sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.png')]))
        self.data_dir = data_dir
        self.label_dir = label_dir

    def __getitem__(self, idx):
        (h, w) = self.input_shape
        hsi_path = os.path.join(self.data_dir, self.hsi[idx])
        label_path = os.path.join(self.label_dir, self.label[idx])
        image = spectral.open_image(hsi_path)
        cube = np.transpose(image.load(), (2, 0, 1))
        hsi = np.array(cube, dtype=np.float32)
        (c, orig_h, orig_w) = hsi.shape
        if orig_h != h or orig_w != w:
            resized_hsi = np.empty((c, h, w), dtype=np.float32)
            for i in range(c):
                resized_hsi[i] = cv2.resize(hsi[i], (w, h), interpolation=cv2.INTER_LINEAR)
            hsi = resized_hsi
        label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label_image.shape[0] != h or label_image.shape[1] != w:
            label_image = cv2.resize(label_image, (w, h), interpolation=cv2.INTER_NEAREST)
        label_image[label_image != 0] = 1
        label = torch.tensor(label_image, dtype=torch.uint8)
        return (hsi, label)

    def __len__(self):
        return len(self.label)

    def random_flip_LR(self, hsi_one_channel_list, rgb, label):
        if random.random() > 0.5:
            for index in range(len(hsi_one_channel_list)):
                hsi_one_channel_list[index] = hsi_one_channel_list[index].transpose(Image.FLIP_LEFT_RIGHT)
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return (hsi_one_channel_list, rgb, label)

    def random_flip_UD(self, hsi_one_channel_list, rgb, label):
        if random.random() > 0.5:
            for index in range(len(hsi_one_channel_list)):
                hsi_one_channel_list[index] = hsi_one_channel_list[index].transpose(Image.FLIP_TOP_BOTTOM)
            rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        return (hsi_one_channel_list, rgb, label)

    def get_params(self, low):
        (self.h, self.w) = low.size
        self.crop_height = 128
        self.crop_width = 128
        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return (i, j)

    def Random_Crop(self, hsi_one_channel_list, rgb, label):
        (self.i, self.j) = self.get_params(rgb)
        if random.random() > 0.5:
            rgb = rgb.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
            label = label.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
            for index in range(len(hsi_one_channel_list)):
                hsi_one_channel_list[index] = hsi_one_channel_list[index].crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
        return (hsi_one_channel_list, rgb, label)
