import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
from .invertible_isp_model.model import InvISPNet

class EVPL(nn.Module):

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device):
        super(EVPL, self).__init__()
        self.filter_spectral_response = Gaussian_kernel_conv2d(lamda_max_1, lamda_max_2, lamda_max_3, device)
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
        wavelengths = torch.linspace(400, 700, steps=60).to(device, non_blocking=True)
        self.register_buffer('wavelengths', torch.tensor(wavelengths, dtype=torch.float32, requires_grad=False))
        self.std = 15.0

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
