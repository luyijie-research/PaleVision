import os
import numpy as np
import torch
import random
import cv2
import h5py
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class MinneAppleHSIDataset(Dataset):

    def __init__(self, root_dir, input_shape, train=True, augment=True):
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.train = train
        self.augment = augment and train
        self.channels = 129
        self.lam = np.linspace(400, 1000, 129)
        split = 'train' if self.train else 'test'
        self.image_dir = os.path.join(root_dir, split, 'hsi')
        self.mask_dir = os.path.join(root_dir, split, 'label')
        self.images = list(sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith('.mat')]))
        print(f'[MinneAppleHSI] Loaded {len(self.images)} samples from {split}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        (h, w) = self.input_shape
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        with h5py.File(img_path, 'r') as f:
            if 'cube' in f:
                hsi = np.array(f['cube'])
            elif 'data' in f:
                hsi = np.array(f['data'])
            elif 'hsi' in f:
                hsi = np.array(f['hsi'])
            else:
                key = list(f.keys())[0]
                hsi = np.array(f[key])
        if hsi.ndim == 3:
            if hsi.shape[0] == self.channels:
                pass
            elif hsi.shape[2] == self.channels:
                hsi = np.transpose(hsi, (2, 0, 1))
            elif hsi.shape[1] == self.channels:
                hsi = np.transpose(hsi, (1, 0, 2))
        hsi = hsi.astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Mask not found: {mask_path}')
        (c, orig_h, orig_w) = hsi.shape
        if orig_h != h or orig_w != w:
            resized_hsi = np.empty((c, h, w), dtype=np.float32)
            for i in range(c):
                resized_hsi[i] = cv2.resize(hsi[i], (w, h), interpolation=cv2.INTER_LINEAR)
            hsi = resized_hsi
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            (hsi, mask) = self._augment(hsi, mask)
        mask[mask > 0] = 1
        hsi = torch.tensor(hsi, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.uint8)
        return (hsi, mask)

    def _augment(self, hsi, mask):
        if random.random() > 0.5:
            hsi = np.flip(hsi, axis=2).copy()
            mask = cv2.flip(mask, 1)
        if random.random() > 0.5:
            hsi = np.flip(hsi, axis=1).copy()
            mask = cv2.flip(mask, 0)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            hsi = np.rot90(hsi, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k)
        mask = np.ascontiguousarray(mask)
        return (hsi, mask)

class MinneAppleDataset(Dataset):

    def __init__(self, root_dir, input_shape, train=True, augment=True):
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.train = train
        self.augment = augment and train
        self.channels = 3
        split = 'train' if self.train else 'test'
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        self.images = list(sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]))
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        (h, w) = self.input_shape
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            (image, mask) = self._augment(image, mask)
        mask[mask > 0] = 1
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        for c in range(3):
            image[c] = (image[c] - self.mean[c]) / self.std[c]
        mask = torch.tensor(mask, dtype=torch.uint8)
        return (image, mask)

    def _augment(self, image, mask):
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        return (image, mask)

class MinneAppleDatasetRaw(Dataset):

    def __init__(self, root_dir, input_shape, train=True, augment=True):
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.train = train
        self.augment = augment and train
        self.channels = 3
        split = 'train' if self.train else 'test'
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        self.images = list(sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        (h, w) = self.input_shape
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            (image, mask) = self._augment(image, mask)
        mask[mask > 0] = 1
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.uint8)
        return (image, mask)

    def _augment(self, image, mask):
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        return (image, mask)
