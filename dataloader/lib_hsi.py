import os
import numpy as np
import torch
import random
import cv2
from pathlib import Path
from PIL import Image
from torch.utils.data.dataset import Dataset
import h5py
import scipy.io

class LIBHSIDataset(Dataset):
    RGB_LABEL_COLORS = [(0, 122, 16), (2, 195, 145), (4, 108, 16), (12, 229, 68), (13, 186, 180), (32, 49, 228), (33, 98, 194), (34, 31, 153), (37, 78, 32), (65, 170, 96), (67, 145, 3), (69, 174, 186), (80, 84, 195), (82, 185, 203), (98, 250, 227), (108, 152, 134), (113, 134, 124), (128, 169, 150), (139, 150, 228), (144, 142, 142), (160, 187, 213), (167, 174, 171), (172, 228, 99), (181, 42, 181), (184, 117, 40), (187, 5, 74), (188, 114, 203), (189, 196, 26), (195, 127, 115), (199, 197, 189), (201, 141, 150), (210, 184, 98), (212, 78, 78), (212, 204, 228), (216, 182, 111), (222, 222, 222), (232, 183, 100), (236, 233, 124), (237, 128, 22), (240, 212, 173), (253, 149, 138), (255, 0, 128), (255, 128, 64), (255, 128, 192)]
    PALETTE = RGB_LABEL_COLORS

    def __init__(self, root_dir, input_shape, train=True, num_classes=44, target_bands=31, spectral_range=(300, 700)):
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.train = train
        self.num_classes = num_classes
        self.original_channels = 204
        self.target_bands = target_bands
        self.spectral_range = spectral_range
        self.ignore_index = 255
        self._label_warning_printed = False
        self._label_mapping_warning_printed = False
        self._palette_rgb = np.array(self.RGB_LABEL_COLORS, dtype=np.int16)
        self.max_color_distance = 8.0
        self.original_wavelengths = np.linspace(397, 1003, 204)
        if target_bands is not None:
            self.channels = target_bands
            start_idx = np.argmin(np.abs(self.original_wavelengths - spectral_range[0]))
            end_idx = np.argmin(np.abs(self.original_wavelengths - spectral_range[1]))
            self.spectral_start_idx = start_idx
            self.spectral_end_idx = end_idx
            self.bands_in_range = end_idx - start_idx + 1
            self.wavelengths = np.linspace(spectral_range[0], spectral_range[1], target_bands)
            print(f'[LIBHSIDataset] Downsampling: {self.original_channels} -> {target_bands} bands')
            print(f'  Spectral range: {spectral_range[0]}-{spectral_range[1]}nm')
            print(f'  Original indices: {start_idx} to {end_idx} ({self.bands_in_range} bands)')
        else:
            self.channels = 204
            self.wavelengths = self.original_wavelengths
        if self.train:
            self.hsi_dir = os.path.join(root_dir, 'train/hsi')
            self.label_dir = os.path.join(root_dir, 'train/labels')
        else:
            self.hsi_dir = os.path.join(root_dir, 'test/hsi')
            self.label_dir = os.path.join(root_dir, 'test/labels')
        self.hsi_files = self._collect_hdr_files(self.hsi_dir)
        self.label_files = self._match_files_by_stem(self.hsi_files, self.label_dir, 'label')
        self.rgb_to_class = self._build_rgb_label_mapping()

    def _build_rgb_label_mapping(self):
        rgb_to_class = {}
        for (class_id, rgb) in enumerate(self.RGB_LABEL_COLORS):
            rgb_to_class[rgb] = class_id
        return rgb_to_class

    def _collect_hdr_files(self, folder_path):
        hdr_files = []
        for file_name in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)
            if not os.path.isfile(file_path):
                continue
            if file_name.lower().endswith('.hdr'):
                hdr_files.append(file_name)
        if len(hdr_files) == 0:
            raise RuntimeError(f'No .hdr files found in {folder_path}')
        return hdr_files

    def _match_files_by_stem(self, hsi_files, target_dir, target_name):
        target_map = {}
        for file_name in sorted(os.listdir(target_dir)):
            file_path = os.path.join(target_dir, file_name)
            if not os.path.isfile(file_path):
                continue
            stem = os.path.splitext(file_name)[0]
            target_map[stem] = file_name
        matched_files = []
        missing_stems = []
        for hsi_file in hsi_files:
            stem = os.path.splitext(hsi_file)[0]
            if stem not in target_map:
                missing_stems.append(stem)
            else:
                matched_files.append(target_map[stem])
        if len(missing_stems) > 0:
            preview = ', '.join(missing_stems[:5])
            raise RuntimeError(f'Missing {target_name} files for {len(missing_stems)} HSI samples in {target_dir}. Examples: {preview}')
        return matched_files

    def __len__(self):
        return len(self.hsi_files)

    def __getitem__(self, idx):
        (h, w) = self.input_shape
        hsi_path = os.path.join(self.hsi_dir, self.hsi_files[idx])
        hsi = self._load_hsi(hsi_path)
        print('shit1', hsi.shape)
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        label = self._load_label(label_path)
        if self.train:
            (hsi, label) = self._augment(hsi, label)
        print('shit2', hsi.shape)
        hsi = self._resize_hsi(hsi, (h, w))
        label = self._resize_label(label, (h, w))
        print('shit3', hsi.shape)
        hsi = torch.tensor(hsi, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return (hsi, label)

    def _load_hsi(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            hsi = np.load(path).astype(np.float32)
        elif ext == '.mat':
            try:
                with h5py.File(path, 'r') as f:
                    for key in ['cube', 'hsi', 'data', 'image']:
                        if key in f.keys():
                            hsi = np.array(f[key], dtype=np.float32)
                            break
            except:
                mat = scipy.io.loadmat(path)
                for key in ['cube', 'hsi', 'data', 'image']:
                    if key in mat.keys():
                        hsi = mat[key].astype(np.float32)
                        break
        elif ext == '.hdr':
            import spectral
            hsi_img = spectral.open_image(path)
            hsi = np.array(hsi_img.load(), dtype=np.float32)
            hsi = np.transpose(hsi, (2, 0, 1))
        else:
            raise ValueError(f'Unsupported HSI format: {ext}')
        if len(hsi.shape) == 3 and hsi.shape[-1] == self.original_channels:
            hsi = np.transpose(hsi, (2, 0, 1))
        if self.target_bands is not None:
            hsi = self._downsample_bands(hsi)
        hsi_min = np.percentile(hsi, 1)
        hsi_max = np.percentile(hsi, 99)
        if hsi_max > hsi_min:
            hsi = np.clip(hsi, hsi_min, hsi_max)
            hsi = (hsi - hsi_min) / (hsi_max - hsi_min)
        elif hsi.max() > 0:
            hsi = hsi / hsi.max()
        return hsi

    def _downsample_bands(self, hsi):
        (C, H, W) = hsi.shape
        hsi_in_range = hsi[self.spectral_start_idx:self.spectral_end_idx + 1, :, :]
        bands_in_range = hsi_in_range.shape[0]
        if self.target_bands >= bands_in_range:
            indices = np.linspace(0, bands_in_range - 1, self.target_bands).astype(int)
            downsampled = hsi_in_range[indices, :, :]
        else:
            downsampled = np.zeros((self.target_bands, H, W), dtype=np.float32)
            step = bands_in_range / self.target_bands
            for i in range(self.target_bands):
                idx_start = int(i * step)
                idx_end = int((i + 1) * step)
                idx_end = min(idx_end, bands_in_range)
                if idx_end > idx_start:
                    downsampled[i] = np.mean(hsi_in_range[idx_start:idx_end, :, :], axis=0)
                else:
                    downsampled[i] = hsi_in_range[idx_start, :, :]
        return downsampled

    def _load_label(self, path):
        raw_label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw_label is None:
            raw_label = np.array(Image.open(path))
        if raw_label.ndim == 2:
            label = raw_label.astype(np.int64)
            invalid_mask = (label < 0) | (label >= self.num_classes)
            if np.any(invalid_mask):
                label[invalid_mask] = self.ignore_index
            return label
        if raw_label.ndim == 3 and raw_label.shape[2] == 4:
            raw_label = raw_label[:, :, :3]
        if raw_label.ndim != 3 or raw_label.shape[2] != 3:
            raise ValueError(f'Unsupported label shape {raw_label.shape} for file: {path}')
        label_rgb = cv2.cvtColor(raw_label, cv2.COLOR_BGR2RGB)
        return self._rgb_to_class_label(label_rgb, path)

    def _rgb_to_class_label(self, label_rgb, path):
        (h, w) = label_rgb.shape[:2]
        label = np.full((h, w), fill_value=self.ignore_index, dtype=np.int64)
        for (rgb, class_id) in self.rgb_to_class.items():
            mask = (label_rgb[:, :, 0] == rgb[0]) & (label_rgb[:, :, 1] == rgb[1]) & (label_rgb[:, :, 2] == rgb[2])
            label[mask] = class_id
        unmapped_mask = label == self.ignore_index
        if np.any(unmapped_mask):
            (ys, xs) = np.where(unmapped_mask)
            chunk_size = 50000
            for start in range(0, len(ys), chunk_size):
                end = min(start + chunk_size, len(ys))
                y_chunk = ys[start:end]
                x_chunk = xs[start:end]
                rgb_chunk = label_rgb[y_chunk, x_chunk].astype(np.int16)
                dist = np.linalg.norm(rgb_chunk[:, None, :] - self._palette_rgb[None, :, :], axis=2)
                nearest_idx = np.argmin(dist, axis=1)
                nearest_dist = dist[np.arange(len(nearest_idx)), nearest_idx]
                valid_fallback = nearest_dist <= self.max_color_distance
                if np.any(valid_fallback):
                    label[y_chunk[valid_fallback], x_chunk[valid_fallback]] = nearest_idx[valid_fallback]
            unmapped_mask = label == self.ignore_index
        if np.any(unmapped_mask) and (not self._label_warning_printed):
            unmapped_rgb = label_rgb[unmapped_mask]
            unique_unmapped = np.unique(unmapped_rgb.reshape(-1, 3), axis=0)[:5]
            unmapped_ratio = float(unmapped_mask.mean())
            print(f'[LIBHSI] Found unmapped RGB in {path}, ignore_ratio={unmapped_ratio:.4f}, examples={unique_unmapped.tolist()}')
            self._label_warning_printed = True
        if not self._label_mapping_warning_printed:
            print(f'[LIBHSI] Applied RGB->class mapping. num_colors={len(self.rgb_to_class)}, target ids range=[0, {self.num_classes - 1}]')
            self._label_mapping_warning_printed = True
        return label

    def _resize_hsi(self, hsi, target_size):
        (h, w) = target_size
        (c, orig_h, orig_w) = hsi.shape
        if (orig_h, orig_w) == (h, w):
            return hsi
        resized = np.empty((c, h, w), dtype=np.float32)
        for i in range(c):
            resized[i] = cv2.resize(hsi[i], (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    def _resize_label(self, label, target_size):
        (h, w) = target_size
        if label.shape == (h, w):
            return label
        return cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

    def _augment(self, hsi, label):
        if random.random() > 0.5:
            (c, orig_h, orig_w) = hsi.shape
            scale = random.uniform(0.7, 1.0)
            crop_h = max(64, int(orig_h * scale))
            crop_w = max(64, int(orig_w * scale))
            if crop_h < orig_h and crop_w < orig_w:
                top = random.randint(0, orig_h - crop_h)
                left = random.randint(0, orig_w - crop_w)
                hsi = hsi[:, top:top + crop_h, left:left + crop_w]
                label = label[top:top + crop_h, left:left + crop_w]
        if random.random() > 0.5:
            hsi = np.flip(hsi, axis=2).copy()
            label = np.flip(label, axis=1).copy()
        if random.random() > 0.5:
            hsi = np.flip(hsi, axis=1).copy()
            label = np.flip(label, axis=0).copy()
        k = random.randint(0, 3)
        if k > 0:
            hsi = np.rot90(hsi, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
        return (hsi, label)

    def get_wavelengths(self):
        return self.wavelengths

class LIBHSIColorDataset(LIBHSIDataset):

    def __init__(self, root_dir, input_shape, train=True, num_classes=44):
        super().__init__(root_dir, input_shape, train, num_classes)
        if self.train:
            self.rgb_dir = os.path.join(root_dir, 'train/rgb')
        else:
            self.rgb_dir = os.path.join(root_dir, 'test/rgb')
        if os.path.exists(self.rgb_dir):
            rgb_map = {}
            for file_name in sorted(os.listdir(self.rgb_dir)):
                file_path = os.path.join(self.rgb_dir, file_name)
                if not os.path.isfile(file_path):
                    continue
                stem = os.path.splitext(file_name)[0]
                rgb_map[stem] = file_name
            self.rgb_files = []
            for hsi_file in self.hsi_files:
                stem = os.path.splitext(hsi_file)[0]
                self.rgb_files.append(rgb_map.get(stem))
        else:
            self.rgb_files = None

    def __getitem__(self, idx):
        (h, w) = self.input_shape
        hsi_path = os.path.join(self.hsi_dir, self.hsi_files[idx])
        hsi = self._load_hsi(hsi_path)
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        label = self._load_label(label_path)
        rgb = None
        if self.rgb_files is not None and self.rgb_files[idx] is not None:
            rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
            rgb = cv2.imread(rgb_path)
            if rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (w, h))
                rgb = rgb.astype(np.float32) / 255.0
                rgb = np.transpose(rgb, (2, 0, 1))
        if self.train:
            (hsi, label) = self._augment(hsi, label)
            if rgb is not None:
                pass
        hsi = self._resize_hsi(hsi, (h, w))
        label = self._resize_label(label, (h, w))
        hsi = torch.tensor(hsi, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        if rgb is not None:
            rgb = torch.tensor(rgb, dtype=torch.float32)
            return (hsi, rgb, label)
        return (hsi, label)
if __name__ == '__main__':
    root_path = 'datasets/LIB-HSI'
    input_shape = [256, 256]
    train_dataset = LIBHSIDataset(root_dir=root_path, input_shape=input_shape, train=True)
    print(f'Dataset size: {len(train_dataset)}')
    print(f'Wavelengths: {train_dataset.get_wavelengths()[:5]}...{train_dataset.get_wavelengths()[-5:]}')
    if len(train_dataset) > 0:
        (hsi, label) = train_dataset[0]
        print(f'HSI shape: {hsi.shape}')
        print(f'Label shape: {label.shape}')
        print(f'Label unique values: {torch.unique(label)}')
