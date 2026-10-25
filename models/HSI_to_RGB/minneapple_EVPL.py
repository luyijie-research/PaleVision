import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .invertible_isp_model.model import InvISPNet

class EVPL(nn.Module):

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device, num_channels=129, use_isp=True):
        super(EVPL, self).__init__()
        self.filter_spectral_response = Gaussian_kernel_conv2d(lamda_max_1, lamda_max_2, lamda_max_3, device, num_channels)
        self.use_isp = use_isp
        if use_isp:
            self.InvISPNet = InvISPNet()

    def forward(self, x):
        x = self.filter_spectral_response(x)
        if self.use_isp:
            x = torch.clamp(x, 0, 1)
            x = self.InvISPNet(x)
            if torch.isnan(x).any():
                print('Warning: InvISPNet output contains NaN!')
                x = torch.nan_to_num(x, nan=0.0)
        return x

class Gaussian_kernel_conv2d(nn.Module):

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device, num_channels=129):
        super(Gaussian_kernel_conv2d, self).__init__()
        self.num_channels = num_channels
        self.device = device
        self.lamda_max_1 = nn.Parameter(torch.tensor(lamda_max_1, device=device), requires_grad=False)
        self.lamda_max_2 = nn.Parameter(torch.tensor(lamda_max_2, device=device), requires_grad=False)
        self.lamda_max_3 = nn.Parameter(torch.tensor(lamda_max_3, device=device), requires_grad=False)
        self.std = 9.0
        self.register_buffer('x', torch.linspace(400, 1000, steps=num_channels))

    def forward(self, input):
        if torch.isnan(input).any():
            print('Warning: Input contains NaN!')
            input = torch.nan_to_num(input, nan=0.0)
        std = self.std
        x = self.x
        kernel_weight_1 = 1 / (std * (2 * math.pi) ** 0.5) * torch.exp(-(x - self.lamda_max_1) ** 2 / (2 * std ** 2)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        kernel_weight_2 = 1 / (std * (2 * math.pi) ** 0.5) * torch.exp(-(x - self.lamda_max_2) ** 2 / (2 * std ** 2)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        kernel_weight_3 = 1 / (std * (2 * math.pi) ** 0.5) * torch.exp(-(x - self.lamda_max_3) ** 2 / (2 * std ** 2)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        kernel = torch.cat([kernel_weight_1, kernel_weight_2, kernel_weight_3], dim=0)
        output = F.conv2d(input, kernel)
        if torch.isnan(output).any():
            print('Warning: Output contains NaN after conv!')
            output = torch.nan_to_num(output, nan=0.0)
        return output

class EVPLPassthrough(nn.Module):

    def __init__(self, device=None):
        super(EVPLPassthrough, self).__init__()

    def forward(self, x):
        return x

class EVPLWithNormalization(nn.Module):

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device, num_channels=129, use_isp=True):
        super(EVPLWithNormalization, self).__init__()
        self.evpl = EVPL(lamda_max_1, lamda_max_2, lamda_max_3, device, num_channels, use_isp)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        out = self.evpl(x)
        out = (out - self.mean.to(out.device)) / self.std.to(out.device)
        return out

class ColorBlindEVPL(nn.Module):

    def __init__(self, blindness_type='protanopia', device=None):
        super(ColorBlindEVPL, self).__init__()
        self.blindness_type = blindness_type
        matrices = {'protanopia': torch.tensor([[0.567, 0.433, 0.0], [0.558, 0.442, 0.0], [0.0, 0.242, 0.758]]), 'deuteranopia': torch.tensor([[0.625, 0.375, 0.0], [0.7, 0.3, 0.0], [0.0, 0.3, 0.7]]), 'tritanopia': torch.tensor([[0.95, 0.05, 0.0], [0.0, 0.433, 0.567], [0.0, 0.475, 0.525]]), 'monochromacy': torch.tensor([[0.299, 0.587, 0.114], [0.299, 0.587, 0.114], [0.299, 0.587, 0.114]])}
        if blindness_type in matrices:
            self.register_buffer('transform_matrix', matrices[blindness_type].float())
        else:
            self.register_buffer('transform_matrix', torch.eye(3).float())
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        (B, C, H, W) = x.shape
        x_flat = x.view(B, C, -1)
        x_flat = x_flat.permute(0, 2, 1)
        x_transformed = torch.matmul(x_flat, self.transform_matrix.T.to(x.device))
        x_transformed = x_transformed.permute(0, 2, 1)
        out = x_transformed.view(B, C, H, W)
        out = torch.clamp(out, 0, 1)
        out = (out - self.mean.to(out.device)) / self.std.to(out.device)
        return out
