import math
import torch
import numpy as np
import cv2
from torchvision import transforms
from torchvision.transforms import functional as F

class RandomResizedCrop(transforms.RandomResizedCrop):

    @staticmethod
    def get_params(img, scale, ratio):
        (width, height) = F._get_image_size(img)
        area = height * width
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        w = min(w, width)
        h = min(h, height)
        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()
        return (i, j, h, w)

def erode_dilate_fg(fg_mask, it=3, r_min=0.7, r_max=0.8):
    assert r_min < r_max, 'r_min should be less or equal to r_max'
    fg_mask = np.float32(fg_mask.cpu())
    kernel_sz = np.arange(1, 21, 2)
    i = 0
    sz_found = False
    while i < len(kernel_sz) and (not sz_found):
        k = kernel_sz[i]
        kernel = np.ones((k, k), np.uint8)
        mask_erode = cv2.erode(fg_mask, kernel, iterations=it)
        ratio = mask_erode.sum() / fg_mask.sum()
        if ratio < r_max and ratio >= r_min:
            sz_found = True
            mask_dilate = cv2.dilate(fg_mask, kernel, iterations=it)
        i = i + 1
    if not sz_found:
        k = 3
        kernel = np.ones((k, k), np.uint8)
        mask_erode = cv2.erode(fg_mask, kernel, iterations=it)
        mask_dilate = cv2.dilate(fg_mask, kernel, iterations=it)
    mask_erode = torch.tensor(mask_erode)
    mask_dilate = torch.tensor(mask_dilate)
    return (mask_erode, mask_dilate)
