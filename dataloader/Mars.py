import os
import numpy as np
import torch
import random
import json
import h5py
import cv2
from PIL import Image
from torchvision import transforms
json_data = '{\n    "items": [\n        {"name": "background", "color_hex_code": "#c00000", "class_group": 0},\n        {"name": "Martian Soil", "color_hex_code": "#800000", "class_group": 1},\n        {"name": "Sands", "color_hex_code": "#008000", "class_group": 2},\n        {"name": "Gravel", "color_hex_code": "#808000", "class_group": 3},\n        {"name": "Bedrock", "color_hex_code": "#000080", "class_group": 4},\n        {"name": "Rocks", "color_hex_code": "#800080", "class_group": 5},\n        {"name": "Tracks", "color_hex_code": "#008080", "class_group": 6},\n        {"name": "Shadows", "color_hex_code": "#808080", "class_group": 7},\n        {"name": "unknown", "color_hex_code": "#400000", "class_group": 8}\n    ]\n}'
data = json.loads(json_data)
color_to_class = {}
for item in data['items']:
    hex_color = item['color_hex_code']
    rgb_color = tuple((int(hex_color[i:i + 2], 16) for i in (1, 3, 5)))
    color_to_class[rgb_color] = item['name']
color_to_label = {}
for (idx, item) in enumerate(data['items']):
    hex_color = item['color_hex_code']
    rgb_color = tuple((int(hex_color[i:i + 2], 16) for i in (1, 3, 5)))
    color_to_label[rgb_color] = item['class_group']

class MarsDataset(object):

    def __init__(self, root_dir, input_shape, train):
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.train = train
        img_dir = os.path.join(root_dir, 'img')
        hsi_dir = os.path.join(root_dir, 'hsi')
        label_dir = os.path.join(root_dir, 'label')

        def _list_valid_files(folder):
            return sorted([f for f in os.listdir(folder) if not f.startswith('.') and os.path.isfile(os.path.join(folder, f))])
        img_files = _list_valid_files(img_dir)
        hsi_files = _list_valid_files(hsi_dir)
        label_files = _list_valid_files(label_dir)
        img_map = {os.path.splitext(f)[0]: f for f in img_files}
        hsi_map = {os.path.splitext(f)[0]: f for f in hsi_files}
        label_map = {os.path.splitext(f)[0]: f for f in label_files}
        common_stems = sorted(set(img_map.keys()) & set(hsi_map.keys()) & set(label_map.keys()))
        if len(common_stems) == 0:
            raise RuntimeError(f'No aligned samples found under {root_dir}. Please check file names in img/hsi/label.')
        self.imgs = [img_map[s] for s in common_stems]
        self.hsi = [hsi_map[s] for s in common_stems]
        self.label = [label_map[s] for s in common_stems]

    def __getitem__(self, idx):
        rgb_mean = (0.0, 0.0, 0.0)
        rgb_std = (255.0, 255.0, 255.0)
        (h, w) = self.input_shape
        img_path = os.path.join(self.root_dir, 'img', self.imgs[idx])
        hsi_path = os.path.join(self.root_dir, 'hsi', self.hsi[idx])
        label_path = os.path.join(self.root_dir, 'label', self.label[idx])
        if self.train:
            with h5py.File(hsi_path, 'r') as mat_data:
                cube = np.array(mat_data['cube'])
            intensity = 1
            lam31 = np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700])
            hsi_one_channel_list = []
            resized_data = np.empty((31, h, w), dtype=np.float32)
            for i in range(cube.shape[0]):
                img_nb = i
                img_slice = cube[img_nb, :, :]
                img_adjusted = img_slice * intensity
                img_adjusted = np.clip(img_adjusted, 0, 1)
                img_adjusted = np.transpose(img_adjusted, (1, 0))
                img_uint8 = (img_adjusted * 255).astype(np.uint8)
                band_img = Image.fromarray(img_uint8)
                hsi_one_channel_list.append(band_img)
            rgb = Image.open(img_path).convert('RGB')
            label_image = cv2.imread(label_path)
            label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
            (height, width, _) = label_image.shape
            label = np.zeros((height, width), dtype=np.uint8)
            for (rgb_color, labels) in color_to_label.items():
                mask = np.all(label_image == np.array(rgb_color), axis=-1)
                label[mask] = labels
            label_pil = Image.fromarray(label)
            (hsi_one_channel_list, rgb, label_pil) = self.random_flip_LR(hsi_one_channel_list, rgb, label_pil)
            (hsi_one_channel_list, rgb, label_pil) = self.random_flip_UD(hsi_one_channel_list, rgb, label_pil)
            (hsi_one_channel_list, rgb, label_pil) = self.Random_Crop(hsi_one_channel_list, rgb, label_pil)
            for index in range(len(hsi_one_channel_list)):
                hsi_one_channel_list[index] = hsi_one_channel_list[index].resize((w, h), Image.BILINEAR)
            rgb = rgb.resize((w, h), Image.BILINEAR)
            label_pil = label_pil.resize((w, h), Image.NEAREST)
            for i in range(31):
                resized_data[i, :, :] = np.array(hsi_one_channel_list[i], dtype=np.float32)
            hsi = resized_data
            rgb = np.array(rgb, dtype=np.float32)
            label = np.array(label_pil, dtype=np.uint8)
            label[label >= 9] = 255
            label = torch.tensor(label, dtype=torch.uint8)
            mask = torch.where(label == 7, torch.tensor(1), torch.tensor(0))
            transform_rgb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)])
            rgb = transform_rgb(rgb)
            return (hsi, rgb, label, mask)
        else:
            img_path = os.path.join(self.root_dir, 'img', self.imgs[idx])
            hsi_path = os.path.join(self.root_dir, 'hsi', self.hsi[idx])
            label_path = os.path.join(self.root_dir, 'label', self.label[idx])
            with h5py.File(hsi_path, 'r') as mat_data:
                cube = np.array(mat_data['cube'])
            intensity = 1
            lam31 = np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700])
            hsi_one_channel_list = []
            resized_data = np.empty((31, h, w), dtype=np.float32)
            for i in range(cube.shape[0]):
                img_nb = i
                img_slice = cube[img_nb, :, :]
                img_adjusted = img_slice * intensity
                img_adjusted = np.clip(img_adjusted, 0, 1)
                img_adjusted = np.transpose(img_adjusted, (1, 0))
                img_uint8 = (img_adjusted * 255).astype(np.uint8)
                band_img = Image.fromarray(img_uint8)
                hsi_one_channel_list.append(band_img)
            rgb = Image.open(img_path).convert('RGB')
            label_image = cv2.imread(label_path)
            label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
            (height, width, _) = label_image.shape
            label = np.zeros((height, width), dtype=np.uint8)
            for (rgb_color, labels) in color_to_label.items():
                mask = np.all(label_image == np.array(rgb_color), axis=-1)
                label[mask] = labels
            label_pil = Image.fromarray(label)
            for index in range(len(hsi_one_channel_list)):
                hsi_one_channel_list[index] = hsi_one_channel_list[index].resize((w, h), Image.BILINEAR)
            rgb = rgb.resize((w, h), Image.BILINEAR)
            label_pil = label_pil.resize((w, h), Image.NEAREST)
            for i in range(31):
                resized_data[i, :, :] = np.array(hsi_one_channel_list[i], dtype=np.float32)
            hsi = resized_data
            rgb = np.array(rgb, dtype=np.float32)
            label = np.array(label_pil, dtype=np.uint8)
            label[label >= 9] = 255
            label = torch.tensor(label, dtype=torch.uint8)
            mask = torch.where(label == 7, torch.tensor(1), torch.tensor(0))
            transform_rgb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)])
            rgb = transform_rgb(rgb)
            return (hsi, rgb, label, mask)

    def __len__(self):
        return len(self.imgs)

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
        self.crop_height = min(128, self.h)
        self.crop_width = min(128, self.w)
        i = random.randint(0, max(self.h - self.crop_height, 0))
        j = random.randint(0, max(self.w - self.crop_width, 0))
        return (i, j)

    def Random_Crop(self, hsi_one_channel_list, rgb, label):
        (self.i, self.j) = self.get_params(rgb)
        if random.random() > 0.5:
            rgb = rgb.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
            label = label.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
            for index in range(len(hsi_one_channel_list)):
                hsi_one_channel_list[index] = hsi_one_channel_list[index].crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
        return (hsi_one_channel_list, rgb, label)
