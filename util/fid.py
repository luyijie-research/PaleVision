import torch
import torch.nn as nn
import torchvision.models as models
from scipy.linalg import sqrtm
import numpy as np
from torchvision import transforms
from .crop import erode_dilate_fg

_inception_v3 = None

def _get_inception():
    global _inception_v3
    if _inception_v3 is None:
        _inception_v3 = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False).eval()
    return _inception_v3

def get_inception_features(imgs):
    device = imgs.device
    model = _get_inception().to(device)
    imgs_resized = nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imgs_normalized = normalize(imgs_resized)
    with torch.no_grad():
        features = model(imgs_normalized)
    return features

def calculate_fid(features_real, features_fake, device='cuda'):
    features_real = features_real.to(device)
    features_fake = features_fake.to(device)
    mu_real = torch.mean(features_real, dim=0)
    mu_fake = torch.mean(features_fake, dim=0)
    cov_real = torch.cov(features_real.T).to(device)
    cov_fake = torch.cov(features_fake.T).to(device)
    mean_diff = torch.sum((mu_real - mu_fake) ** 2)
    cov_real_np = cov_real.detach().cpu().numpy()
    cov_fake_np = cov_fake.detach().cpu().numpy()
    cov_real_np += np.eye(cov_real_np.shape[0]) * 1e-06
    cov_fake_np += np.eye(cov_fake_np.shape[0]) * 1e-06
    cov_prod = sqrtm(cov_real_np @ cov_fake_np)
    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real
    cov_prod = torch.from_numpy(cov_prod).float().to(device)
    fid_value = mean_diff + torch.trace(cov_real + cov_fake - 2 * cov_prod).item()
    return fid_value

def fid_loss(real_imgs, generated_imgs):
    real_features = get_inception_features(real_imgs)
    fake_features = get_inception_features(generated_imgs)
    fid = calculate_fid(real_features, fake_features)
    return fid

def camouflage_loss(img_batch, mask_batch):
    device = img_batch.device
    fg_batch = []
    bg_batch = []
    img_list = []
    for img in img_batch:
        img_list.append(img)
    for i, mask in enumerate(mask_batch, 0):
        mask = mask.cpu().numpy().astype(np.uint8)
        trimap = erode_dilate_fg(mask)
        mask_erode, mask_dilate = trimap
        im_masked_fg = torch.tensor(mask_erode, device=device).unsqueeze(0) * img_list[i]
        fg_batch.append(im_masked_fg)
        im_masked_bg = (1 - torch.tensor(mask_dilate, device=device).unsqueeze(0)) * img_list[i]
        bg_batch.append(im_masked_bg)
    foreground_batch = torch.stack(fg_batch)
    background_batch = torch.stack(bg_batch)
    fid_value = fid_loss(real_imgs=foreground_batch, generated_imgs=background_batch)
    loss = torch.sigmoid(fid_value).item()
    return loss
