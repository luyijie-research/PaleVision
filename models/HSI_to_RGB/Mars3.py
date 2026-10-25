import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .invertible_isp_model.model import InvISPNet

wavelengths = np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700])

class EVPL(nn.Module):

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device):
        super(EVPL, self).__init__()
        self.filter_spectral_response = Gaussian_kernel_conv2d(lamda_max_1, lamda_max_2, lamda_max_3, device)
        self.multi_to_rgb = nn.Conv2d(3, 3, kernel_size=(1, 1))
        self.InvISPNet = InvISPNet()

    def forward(self, x):
        x = self.filter_spectral_response(x)
        return x

class Gaussian_kernel_conv2d(nn.Module):

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device):
        super(Gaussian_kernel_conv2d, self).__init__()
        self.lamda_max_1 = nn.Parameter(torch.tensor(lamda_max_1, device=device), requires_grad=True)
        self.lamda_max_2 = nn.Parameter(torch.tensor(lamda_max_2, device=device), requires_grad=True)
        self.lamda_max_3 = nn.Parameter(torch.tensor(lamda_max_3, device=device), requires_grad=True)
        self.wavelengths = torch.tensor(wavelengths, dtype=torch.float32, device=device, requires_grad=False)
        self.std = 10.0

    def forward(self, input):
        if torch.isnan(input).any():
            input = torch.nan_to_num(input, nan=0.0)
        std = self.std
        x = self.wavelengths
        kernel_weight_1 = 1 / (std * (2 * math.pi) ** 0.5) * torch.exp(-(x - self.lamda_max_1) ** 2 / (2 * std ** 2)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        kernel_weight_2 = 1 / (std * (2 * math.pi) ** 0.5) * torch.exp(-(x - self.lamda_max_2) ** 2 / (2 * std ** 2)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        kernel_weight_3 = 1 / (std * (2 * math.pi) ** 0.5) * torch.exp(-(x - self.lamda_max_3) ** 2 / (2 * std ** 2)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        kernel = torch.cat([kernel_weight_1, kernel_weight_2, kernel_weight_3], dim=0)
        output = F.conv2d(input, kernel)
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
        return output
