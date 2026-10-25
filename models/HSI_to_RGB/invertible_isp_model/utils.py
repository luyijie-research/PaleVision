import math
import torch

def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]
    if isinstance(stride, int):
        stride = [stride]
    assert len(stride) == len(kernel_size), 'Pass kernel size and stride both as int, or both as equal length iterable'
    return [((k - 1) * s + 1) // 2 for (k, s) in zip(kernel_size, stride)]

def uniform_binning_correction(x, n_bits=8):
    (b, c, h, w) = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)
    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return (x, objective)

def split_feature(tensor, type='split'):
    C = tensor.size(1)
    if type == 'split':
        return (tensor[:, :1, ...], tensor[:, 1:, ...])
    elif type == 'cross':
        return (tensor[:, 0::2, ...], tensor[:, 1::2, ...])
