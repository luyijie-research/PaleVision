import cv2
import numpy as np
import skimage.io
import albumentations as A

def resize(image, size=(128, 128)):
    return cv2.resize(image, size)

def add_gaussian_noise(x, sigma):
    x += np.random.randn(*x.shape) * sigma
    x = np.clip(x, 0.0, 1.0)
    return x

def _evaluate_ratio(ratio):
    if ratio <= 0.0:
        return False
    return np.random.uniform() < ratio

def apply_aug(aug, image, mask=None):
    augment = aug(image=image, mask=mask)
    return (augment['image'], augment['mask'])

class Transform:

    def __init__(self, size=None, train=True, BrightContrast_ration=0.0, noise_ratio=0.0, cutout_ratio=0.0, grid_distortion_ratio=0.0, elastic_distortion_ratio=0.0, piece_affine_ratio=0.0, ssr_ratio=0.0, Rotate_ratio=0.0, Flip_ratio=0.0):
        self.size = size
        self.train = train
        self.noise_ratio = noise_ratio
        self.BrightContrast_ration = BrightContrast_ration
        self.cutout_ratio = cutout_ratio
        self.grid_distortion_ratio = grid_distortion_ratio
        self.elastic_distortion_ratio = elastic_distortion_ratio
        self.piece_affine_ratio = piece_affine_ratio
        self.ssr_ratio = ssr_ratio
        self.Rotate_ratio = Rotate_ratio
        self.Flip_ratio = Flip_ratio

    def __call__(self, example):
        if self.train:
            (x, y) = example
        else:
            x = example
        if self.size is not None:
            x = resize(x, size=self.size)
        if _evaluate_ratio(self.BrightContrast_ration):
            x = apply_aug(A.RandomBrightnessContrast(p=1.0), x)
        if _evaluate_ratio(self.noise_ratio):
            r = np.random.uniform()
            if r < 0.5:
                x = apply_aug(A.GaussNoise(var_limit=5.0 / 255.0, p=1.0), x)
            else:
                x = apply_aug(A.MultiplicativeNoise(p=1.0), x)
        if _evaluate_ratio(self.grid_distortion_ratio):
            (x, y) = apply_aug(A.GridDistortion(p=1.0), x, y)
        if _evaluate_ratio(self.elastic_distortion_ratio):
            (x, y) = apply_aug(A.ElasticTransform(sigma=50, alpha=1, p=1.0), x, y)
        if _evaluate_ratio(self.Rotate_ratio):
            (x, y) = apply_aug(A.Rotate(p=1.0), x, y)
        if _evaluate_ratio(self.Flip_ratio):
            (x, y) = apply_aug(A.Flip(p=1.0), x, y)
        if self.train:
            return (x, y)
        else:
            return x
