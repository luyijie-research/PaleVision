import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .invertible_isp_model.model import InvISPNet

def get_wavelengths(num_bands=204, spectral_range=(400, 1000)):
    return np.linspace(spectral_range[0], spectral_range[1], num_bands)
wavelengths = get_wavelengths(204, (400, 1000))

class EVPL(nn.Module):

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device, use_isp=False, num_bands=31, spectral_range=(300, 700)):
        super(EVPL, self).__init__()
        self.num_bands = num_bands
        self.spectral_range = spectral_range
        self.filter_spectral_response = Gaussian_kernel_conv2d(lamda_max_1, lamda_max_2, lamda_max_3, device, num_bands=num_bands, spectral_range=spectral_range)
        self.use_isp = use_isp
        if use_isp:
            self.InvISPNet = InvISPNet()

    def forward(self, x):
        x = self.filter_spectral_response(x)
        if self.use_isp:
            x = self.InvISPNet(x)
        else:
            (B, C, H, W) = x.shape
            x_flat = x.view(B, C, -1)
            x_min = x_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
            x_max = x_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
            x = (x - x_min) / (x_max - x_min + 1e-06)
        return x

    def get_lambda_max(self):
        return {'lambda_1': self.filter_spectral_response.lamda_max_1.item(), 'lambda_2': self.filter_spectral_response.lamda_max_2.item(), 'lambda_3': self.filter_spectral_response.lamda_max_3.item(), 'std': self.filter_spectral_response.std}

class Gaussian_kernel_conv2d(nn.Module):

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device, trainable=True, num_bands=204, spectral_range=(400, 1000)):
        super(Gaussian_kernel_conv2d, self).__init__()
        self.num_bands = num_bands
        self.spectral_range = spectral_range
        self.lamda_max_1 = nn.Parameter(torch.tensor(lamda_max_1, device=device, dtype=torch.float32), requires_grad=trainable)
        self.lamda_max_2 = nn.Parameter(torch.tensor(lamda_max_2, device=device, dtype=torch.float32), requires_grad=trainable)
        self.lamda_max_3 = nn.Parameter(torch.tensor(lamda_max_3, device=device, dtype=torch.float32), requires_grad=trainable)
        wl_array = get_wavelengths(num_bands, spectral_range)
        self.register_buffer('wavelengths', torch.tensor(wl_array, dtype=torch.float32))
        self.std = 30.0
        self.device = device
        print(f'[Gaussian_kernel_conv2d] num_bands={num_bands}, range={spectral_range}')

    def forward(self, input):
        kernel_weight_1 = self._gaussian_kernel(self.lamda_max_1)
        kernel_weight_2 = self._gaussian_kernel(self.lamda_max_2)
        kernel_weight_3 = self._gaussian_kernel(self.lamda_max_3)
        kernel = torch.cat([kernel_weight_1, kernel_weight_2, kernel_weight_3], dim=0)
        output = F.conv2d(input, kernel)
        return output

    def _gaussian_kernel(self, lamda_max):
        weight = torch.exp(-(self.wavelengths - lamda_max) ** 2 / (2 * self.std ** 2))
        weight = weight / (weight.sum() + 1e-06)
        return weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def print_lambda(self):
        print(f'lamda_max_1: {self.lamda_max_1.data.item():.2f} nm, lamda_max_2: {self.lamda_max_2.data.item():.2f} nm, lamda_max_3: {self.lamda_max_3.data.item():.2f} nm, std: {self.std:.2f} nm (fixed)')

class EVPL_Tetrachromatic(nn.Module):

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, lamda_max_4, device):
        super(EVPL_Tetrachromatic, self).__init__()
        self.filter_spectral_response = Gaussian_kernel_conv2d_4(lamda_max_1, lamda_max_2, lamda_max_3, lamda_max_4, device)

    def forward(self, x):
        return self.filter_spectral_response(x)

class Gaussian_kernel_conv2d_4(nn.Module):

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, lamda_max_4, device, trainable=True):
        super(Gaussian_kernel_conv2d_4, self).__init__()
        self.lamda_max_1 = nn.Parameter(torch.tensor(lamda_max_1, device=device), requires_grad=trainable)
        self.lamda_max_2 = nn.Parameter(torch.tensor(lamda_max_2, device=device), requires_grad=trainable)
        self.lamda_max_3 = nn.Parameter(torch.tensor(lamda_max_3, device=device), requires_grad=trainable)
        self.lamda_max_4 = nn.Parameter(torch.tensor(lamda_max_4, device=device), requires_grad=trainable)
        self.wavelengths = torch.tensor(wavelengths, dtype=torch.float32, device=device, requires_grad=True)
        self.std = 10.0
        self.device = device

    def forward(self, input):
        kernel_weight_1 = self._gaussian_kernel(self.lamda_max_1)
        kernel_weight_2 = self._gaussian_kernel(self.lamda_max_2)
        kernel_weight_3 = self._gaussian_kernel(self.lamda_max_3)
        kernel_weight_4 = self._gaussian_kernel(self.lamda_max_4)
        kernel = torch.cat([kernel_weight_1, kernel_weight_2, kernel_weight_3, kernel_weight_4], dim=0)
        output = F.conv2d(input, kernel)
        return output

    def _gaussian_kernel(self, lamda_max):
        weight = 1 / (self.std * (2 * math.pi) ** 0.5) * torch.exp(-(self.wavelengths - lamda_max) ** 2 / (2 * self.std ** 2))
        return weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 60)
    print('Test 1: 204 bands (original)')
    model_204 = EVPL(lamda_max_1=450.0, lamda_max_2=530.0, lamda_max_3=560.0, device=device, num_bands=204, spectral_range=(400, 1000))
    model_204.to(device)
    input_204 = torch.randn((4, 204, 256, 256)).to(device)
    output_204 = model_204(input_204)
    print(f'  Input: {input_204.shape} -> Output: {output_204.shape}')
    print('\nTest 2: 129 bands (downsampled)')
    model_129 = EVPL(lamda_max_1=450.0, lamda_max_2=530.0, lamda_max_3=560.0, device=device, num_bands=129, spectral_range=(400, 1000))
    model_129.to(device)
    input_129 = torch.randn((4, 129, 256, 256)).to(device)
    output_129 = model_129(input_129)
    print(f'  Input: {input_129.shape} -> Output: {output_129.shape}')
    print('\nTest 3: 31 bands (visible only)')
    model_31 = EVPL(lamda_max_1=450.0, lamda_max_2=530.0, lamda_max_3=580.0, device=device, num_bands=31, spectral_range=(400, 700))
    model_31.to(device)
    input_31 = torch.randn((4, 31, 256, 256)).to(device)
    output_31 = model_31(input_31)
    print(f'  Input: {input_31.shape} -> Output: {output_31.shape}')
    print('=' * 60)
    print(f'Input shape: {input.shape}')
    print(f'Output shape: {output.shape}')
    print(f'Lambda max values: {model.get_lambda_max()}')
