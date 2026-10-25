import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import logging
from .invertible_isp_model.model import InvISPNet
wavelengths = np.array([460.48099, 465.215098, 474.1194, 482.578653, 492.59933, 504.367465, 512.469275, 521.987657, 533.930667, 541.203169, 551.996143, 559.719978, 570.1331, 579.729623, 585.387574, 595.018809])

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
        self.register_buffer('wavelengths', torch.tensor(wavelengths, dtype=torch.float32))
        self.std = 70.0

    def forward(self, input):
        std = self.std
        kernel_weight_1 = 1 / (std * (2 * math.pi) ** 0.5) * torch.exp(-(self.wavelengths - self.lamda_max_1) ** 2 / (2 * std ** 2)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        kernel_weight_2 = 1 / (std * (2 * math.pi) ** 0.5) * torch.exp(-(self.wavelengths - self.lamda_max_2) ** 2 / (2 * std ** 2)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        kernel_weight_3 = 1 / (std * (2 * math.pi) ** 0.5) * torch.exp(-(self.wavelengths - self.lamda_max_3) ** 2 / (2 * std ** 2)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        kernel = torch.cat([kernel_weight_1, kernel_weight_2, kernel_weight_3], dim=0)
        output = F.conv2d(input, kernel)
        return output
if __name__ == '__main__':
    input = torch.randn((4, 129, 400, 400))
    model = Gaussian_kernel_conv2d(20.0, 30.0, 60.0, 90.0)
    output = model(input)
    print(output.shape)
